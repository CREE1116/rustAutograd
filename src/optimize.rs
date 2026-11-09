use crate::graph::{Graph, Op, ReduceKind};
use crate::plan::{eval, Plan};
use crate::tensor::Tensor;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::Arc;

const FOLD_LIMIT: usize = 4096;
const CSE_CONST_LIMIT: usize = 16;
/// Below this many elements a value is not worth recomputing to save memory.
pub const REMAT_MIN: usize = 1 << 14;
const SPLIT_SUM_MIN: usize = 4096;

#[derive(Default, Clone, Debug)]
pub struct OptStats {
    pub nodes_before: usize,
    pub nodes_after: usize,
    pub folded: usize,
    pub simplified: usize,
    pub cse: usize,
    pub rematerialized: usize,
}

impl OptStats {
    pub fn json(&self) -> Value {
        json!({
            "nodes_before": self.nodes_before,
            "nodes_after": self.nodes_after,
            "constants_folded": self.folded,
            "algebraic_simplified": self.simplified,
            "cse_merged": self.cse,
            "rematerialized": self.rematerialized,
        })
    }
}

/// The value of a constant whose elements are all equal.
fn splat(g: &Graph, id: u32) -> Option<f64> {
    let n = &g.nodes[id as usize];
    let Op::Const(t) = &n.op else { return None };
    if t.n() == 0 || t.n() > FOLD_LIMIT {
        return None;
    }
    let v = t.data.to_f64();
    v.iter().all(|&x| x == v[0]).then_some(v[0])
}

/// An existing value equal to the node. Never rewrites `x * 0`: that would hide NaN/Inf in `x`.
fn identity(g: &Graph, op: &Op, ins: &[u32], shape: &[usize]) -> Option<u32> {
    use Op::*;
    let same = |id: u32| g.shape(id) == shape;
    let is = |id: u32, c: f64| splat(g, id) == Some(c);
    match op {
        StopGradient | Custom { .. } => Some(ins[0]),
        Add if is(ins[1], 0.0) && same(ins[0]) => Some(ins[0]),
        Add if is(ins[0], 0.0) && same(ins[1]) => Some(ins[1]),
        Sub if is(ins[1], 0.0) && same(ins[0]) => Some(ins[0]),
        Mul if is(ins[1], 1.0) && same(ins[0]) => Some(ins[0]),
        Mul if is(ins[0], 1.0) && same(ins[1]) => Some(ins[1]),
        Div if is(ins[1], 1.0) && same(ins[0]) => Some(ins[0]),
        Pow if is(ins[1], 1.0) && same(ins[0]) => Some(ins[0]),
        Neg => match &g.nodes[ins[0] as usize].op {
            Neg => Some(g.nodes[ins[0] as usize].inputs[0]),
            _ => None,
        },
        Reshape(_) | BroadcastTo(_) if same(ins[0]) => Some(ins[0]),
        Cast(d) if *d == g.dtype(ins[0]) => Some(ins[0]),
        Transpose(p) if p.iter().enumerate().all(|(i, &v)| i == v) => Some(ins[0]),
        _ => None,
    }
}

/// Peak memory above which recomputing costly functions beats storing them. Below it, storing
/// wins: recomputing tanh cost a small MLP about 30% of its speed, while above about 12 GB on a
/// 24 GB machine, swapping made a large model three times slower.
const COSTLY_REMAT_ABOVE_BYTES: usize = 2 << 30;

/// Elementwise and cheap enough to compute twice rather than store until the backward pass.
fn recomputable(op: &Op, costly_too: bool) -> bool {
    use Op::*;
    let elementwise = op.is_unary_elementwise() || op.is_binary_elementwise() || matches!(op, Where);
    let costly = matches!(op, Exp | Log | Sin | Cos | Tanh | Sqrt);
    elementwise && !matches!(op, Pow) && (costly_too || !costly)
}

/// Replaces stored forward values with recomputation in the backward pass, where that does not
/// retain more. A value is dropped when recomputing it adds at most one new large value to what
/// must stay alive; walking from late to early lets whole chains collapse onto their source.
fn rematerialize(g: &Graph, outputs: &[u32], backward: &[bool], min_n: usize, costly_too: bool, st: &mut OptStats) -> (Graph, Vec<u32>) {
    let keep = g.reachable(outputs);
    let large = |id: u32| g.nodes[id as usize].n() >= min_n;
    let mut retained = vec![false; g.nodes.len()];
    for (i, n) in g.nodes.iter().enumerate() {
        if keep[i] && backward[i] {
            for &x in &n.inputs {
                retained[x as usize] |= !backward[x as usize];
            }
        }
    }
    let mut remat = vec![false; g.nodes.len()];
    for i in (0..g.nodes.len()).rev() {
        let n = &g.nodes[i];
        if !retained[i] || backward[i] || !recomputable(&n.op, costly_too) || !large(i as u32) {
            continue;
        }
        let always_alive = |x: u32| matches!(g.nodes[x as usize].op, Op::Input(_) | Op::Const(_));
        let mut fresh: Vec<u32> = n.inputs.iter().copied().filter(|&x| large(x) && !retained[x as usize] && !always_alive(x)).collect();
        fresh.dedup();
        if fresh.len() <= 1 {
            remat[i] = true;
            retained[i] = false;
            for &x in &n.inputs {
                retained[x as usize] = true;
            }
        }
    }

    // Rebuild, cloning dropped values right where the backward pass first needs them.
    let mut h = Graph::new();
    let mut map = vec![u32::MAX; g.nodes.len()];
    let mut clone: Vec<Option<u32>> = vec![None; g.nodes.len()];
    fn value(i: u32, g: &Graph, h: &mut Graph, map: &[u32], clone: &mut [Option<u32>], remat: &[bool], st: &mut OptStats) -> u32 {
        if !remat[i as usize] {
            return map[i as usize];
        }
        if let Some(c) = clone[i as usize] {
            return c;
        }
        let n = &g.nodes[i as usize];
        let ins: Vec<u32> = n.inputs.iter().map(|&x| value(x, g, h, map, clone, remat, st)).collect();
        h.cur_source = n.source.clone();
        let id = h.add(n.op.clone(), &ins).expect("clone of a checked node");
        clone[i as usize] = Some(id);
        st.rematerialized += 1;
        id
    }
    for (i, n) in g.nodes.iter().enumerate() {
        if let Op::Input(_) = n.op {
            h.cur_source = n.source.clone();
            map[i] = h.input(&n.shape, n.dtype);
            continue;
        }
        if !keep[i] {
            continue;
        }
        let ins: Vec<u32> = if backward[i] {
            n.inputs.iter().map(|&x| value(x, g, &mut h, &map, &mut clone, &remat, st)).collect()
        } else {
            n.inputs.iter().map(|&x| map[x as usize]).collect()
        };
        h.cur_source = n.source.clone();
        map[i] = match &n.op {
            Op::Const(t) => h.constant(Tensor::clone(t)),
            op => h.add(op.clone(), &ins).expect("copy of a checked node"),
        };
    }
    let out = outputs.iter().map(|&o| map[o as usize]).collect();
    (h, out)
}

pub fn optimize(g: &Graph, outputs: &[u32]) -> (Graph, Vec<u32>, OptStats) {
    optimize_for(g, outputs, false)
}

/// `metal`: a full sum of a large tensor would run on a single GPU thread, so it is split into a
/// parallel sum along the last axis and a small sum of the results.
pub fn optimize_for(g: &Graph, outputs: &[u32], metal: bool) -> (Graph, Vec<u32>, OptStats) {
    let (ng, out, mut st, backward) = simplify(g, outputs, metal);
    let (mut h, mut h_out) = rematerialize(&ng, &out, &backward, REMAT_MIN, false, &mut st);
    if Plan::build(&h, &h_out, true).peak_live_bytes > COSTLY_REMAT_ABOVE_BYTES {
        st.rematerialized = 0;
        (h, h_out) = rematerialize(&ng, &out, &backward, REMAT_MIN, true, &mut st);
    }
    st.nodes_after = h.reachable(&h_out).iter().filter(|&&k| k).count();
    (h, h_out, st)
}

/// `remat_min` is the smallest value recomputation applies to; tests lower it to reach every path.
pub fn optimize_with(g: &Graph, outputs: &[u32], remat_min: usize) -> (Graph, Vec<u32>, OptStats) {
    let (ng, out, mut st, backward) = simplify(g, outputs, false);
    let (h, out) = rematerialize(&ng, &out, &backward, remat_min, true, &mut st);
    st.nodes_after = h.reachable(&out).iter().filter(|&&k| k).count();
    (h, out, st)
}

/// Single forward rewrite: SSA order means folds and identities cascade without a fixpoint loop.
fn simplify(g: &Graph, outputs: &[u32], metal: bool) -> (Graph, Vec<u32>, OptStats, Vec<bool>) {
    let keep = g.reachable(outputs);
    let mut st = OptStats { nodes_before: keep.iter().filter(|&&k| k).count(), ..Default::default() };
    let mut ng = Graph::new();
    let mut map = vec![u32::MAX; g.nodes.len()];
    let mut seen: HashMap<String, u32> = HashMap::new();
    let mut backward: Vec<bool> = vec![];

    for (i, node) in g.nodes.iter().enumerate() {
        ng.cur_source = node.source.clone();
        ng.cur_origin = None;
        if let Op::Input(_) = node.op {
            map[i] = ng.input(&node.shape, node.dtype);
            backward.resize(ng.nodes.len(), false);
            continue;
        }
        if !keep[i] {
            continue;
        }
        let ins: Vec<u32> = node.inputs.iter().map(|&x| map[x as usize]).collect();

        if let Some(existing) = identity(&ng, &node.op, &ins, &node.shape) {
            map[i] = existing;
            st.simplified += 1;
            continue;
        }

        let (mut op, mut ins) = (node.op.clone(), ins);
        if matches!(op, Op::Pow) && splat(&ng, ins[1]) == Some(2.0) && ng.shape(ins[0]) == node.shape.as_slice() {
            op = Op::Mul;
            ins = vec![ins[0], ins[0]];
            st.simplified += 1;
        }

        // Binary kernels broadcast by themselves, so a materialized broadcast is a wasted copy.
        if op.is_binary_elementwise() {
            for k in 0..2 {
                let n = &ng.nodes[ins[k] as usize];
                if let Op::BroadcastTo(_) = n.op {
                    let source = n.inputs[0];
                    let other = ins[1 - k];
                    if crate::graph::broadcast_shapes(ng.shape(source), ng.shape(other)).as_deref() == Some(node.shape.as_slice()) {
                        ins[k] = source;
                        st.simplified += 1;
                    }
                }
            }
        }

        // BLAS reads a transposed operand directly; the copy then goes away with DCE.
        if matches!(op, Op::MatMul) {
            let flipped = |id: u32| match &ng.nodes[id as usize].op {
                Op::Transpose(p) if p.len() >= 2 && p.iter().enumerate().all(|(i, &v)| if i + 2 < p.len() { i == v } else { i + v == 2 * p.len() - 3 }) => Some(ng.nodes[id as usize].inputs[0]),
                _ => None,
            };
            let (fa, fb) = (flipped(ins[0]), flipped(ins[1]));
            if fa.is_some() || fb.is_some() {
                op = Op::MatMulT { ta: fa.is_some(), tb: fb.is_some() };
                ins = vec![fa.unwrap_or(ins[0]), fb.unwrap_or(ins[1])];
                st.simplified += 1;
            }
        }

        let foldable = !ins.is_empty() && node.n() <= FOLD_LIMIT && ins.iter().all(|&x| matches!(ng.nodes[x as usize].op, Op::Const(_)));
        if foldable {
            let mut tmp = Graph::new();
            let ids: Vec<u32> = ins
                .iter()
                .map(|&x| match &ng.nodes[x as usize].op {
                    Op::Const(t) => tmp.constant(Tensor::clone(t)),
                    _ => unreachable!(),
                })
                .collect();
            let id = tmp.add(op.clone(), &ids).expect("node already passed shape inference");
            let value = eval(&tmp, &[id], &[]).remove(0);
            op = Op::Const(Arc::new(value));
            ins = vec![];
            st.folded += 1;
        }

        if let Op::Reduce { kind: ReduceKind::Sum, axes, keepdims } = &op {
            let shape = ng.shape(ins[0]).to_vec();
            if metal && shape.len() >= 2 && axes.len() == shape.len() && node.n() == 1 && ng.nodes[ins[0] as usize].n() >= SPLIT_SUM_MIN && shape[shape.len() - 1] > 1 {
                let rows = ng.add(Op::Reduce { kind: ReduceKind::Sum, axes: vec![shape.len() - 1], keepdims: false }, &[ins[0]]).expect("axis exists");
                backward.resize(ng.nodes.len(), false);
                backward[rows as usize] = node.origin.is_some();
                op = Op::Reduce { kind: ReduceKind::Sum, axes: (0..shape.len() - 1).collect(), keepdims: *keepdims };
                ins = vec![rows];
            }
        }

        let is_big_const = matches!(&op, Op::Const(t) if t.n() > CSE_CONST_LIMIT);
        let key = (!is_big_const).then(|| format!("{:?}|{:?}", op, ins));
        if let Some(&existing) = key.as_ref().and_then(|k| seen.get(k)) {
            map[i] = existing;
            st.cse += 1;
            continue;
        }
        let id = match op {
            Op::Const(t) => ng.constant(Tensor::clone(&t)),
            op => ng.add(op, &ins).expect("node already passed shape inference"),
        };
        ng.nodes[id as usize].origin = None;
        backward.resize(ng.nodes.len(), false);
        backward[id as usize] = node.origin.is_some();
        if let Some(k) = key {
            seen.insert(k, id);
        }
        map[i] = id;
    }

    let out: Vec<u32> = outputs.iter().map(|&o| map[o as usize]).collect();
    backward.resize(ng.nodes.len(), false);
    (ng, out, st, backward)
}
