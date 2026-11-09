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
/// A rewrite must promise this cost ratio. Rewriting changes rounding, and small estimated gains
/// are within the model's error, so below this the original program stays.
const SEARCH_MIN_GAIN: f64 = 0.9;

#[derive(Default, Clone, Debug)]
pub struct OptStats {
    pub nodes_before: usize,
    pub nodes_after: usize,
    pub folded: usize,
    pub simplified: usize,
    pub cse: usize,
    pub rematerialized: usize,
    /// Equality saturation, if it ran: whether its result was accepted, and its statistics.
    pub search: Option<(bool, crate::egraph::SearchStats)>,
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
            "search": self.search.as_ref().map(|(accepted, s)| json!({ "accepted": accepted, "iterations": s.iterations, "egraph_nodes": s.egraph_nodes, "cost_before": s.cost_before, "cost_after": s.cost_after })),
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
            h.nodes[map[i] as usize].op = n.op.clone();
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
    optimize_mode(g, outputs, metal, Mode::Standard)
}

/// How hard the compiler looks for a better expression of the same math.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Mode {
    /// Only dead-code removal, to measure what optimization is worth.
    None,
    Standard,
    /// Standard plus equality saturation with rules exact in IEEE 754.
    SearchExact,
    /// Exact rules plus matrix-product reassociation and factoring.
    SearchLinear,
    /// Also rules that change rounding, such as associativity. Compile validation still checks
    /// the result against the original program.
    Search,
    /// Compiles `Standard` and `SearchLinear`, times both on the example inputs and keeps the
    /// faster. Timing varies, so unlike the other modes this can pick differently between runs.
    Auto,
}

impl Mode {
    pub fn parse(s: &str) -> Option<Mode> {
        match s {
            "none" => Some(Mode::None),
            "standard" => Some(Mode::Standard),
            "search-exact" => Some(Mode::SearchExact),
            "search-linear" => Some(Mode::SearchLinear),
            "search" => Some(Mode::Search),
            "auto" => Some(Mode::Auto),
            _ => None,
        }
    }
}

pub fn optimize_mode(g: &Graph, outputs: &[u32], metal: bool, mode: Mode) -> (Graph, Vec<u32>, OptStats) {
    assert!(mode != Mode::Auto, "compile_mode resolves Auto into concrete modes");
    if mode == Mode::None {
        let keep = g.reachable(outputs).iter().filter(|&&k| k).count();
        return (g.clone(), outputs.to_vec(), OptStats { nodes_before: keep, nodes_after: keep, ..Default::default() });
    }
    let (ng, out, mut st, backward) = match mode {
        Mode::Search | Mode::SearchExact | Mode::SearchLinear => {
            let baseline = simplify(g, outputs, metal, true);
            let (first, first_out, _, _) = simplify(g, outputs, false, false);
            if mode == Mode::SearchLinear && !has_matrix_structure(&first, &first_out) {
                let (a, b, mut st, c) = baseline;
                st.nodes_before = g.reachable(outputs).iter().filter(|&&k| k).count();
                return finish(&a, &b, &c, st);
            }
            let model = if metal { crate::egraph::CostModel::METAL } else { crate::egraph::CostModel::CPU };
            let set = match mode {
                Mode::Search => crate::egraph::RuleSet::Rounding,
                Mode::SearchLinear => crate::egraph::RuleSet::Linear,
                _ => crate::egraph::RuleSet::Exact,
            };
            let (found, mut search) = crate::egraph::search(&first, &first_out, model, set, &crate::egraph::DEFAULT_LIMITS);
            // Compare final forms: both sides have folded constants and flagged transposes.
            search.cost_before = model.graph_cost(&baseline.0, &baseline.1);
            let candidate = found.map(|(h, o)| simplify(&h, &o, metal, true));
            let cost = candidate.as_ref().map(|c| model.graph_cost(&c.0, &c.1));
            search.cost_after = cost.unwrap_or(search.cost_before);
            let accepted = cost.is_some_and(|c| c < search.cost_before * SEARCH_MIN_GAIN);
            let (a, b, mut st, c) = if accepted { candidate.expect("accepted implies a candidate") } else { baseline };
            st.nodes_before = g.reachable(outputs).iter().filter(|&&k| k).count();
            st.search = Some((accepted, search));
            (a, b, st, c)
        }
        _ => simplify(g, outputs, metal, true),
    };
    finish(&ng, &out, &backward, st)
}

fn finish(ng: &Graph, out: &[u32], backward: &[bool], mut st: OptStats) -> (Graph, Vec<u32>, OptStats) {
    let (mut h, mut h_out) = rematerialize(ng, out, backward, REMAT_MIN, false, &mut st);
    if Plan::build(&h, &h_out, true).peak_live_bytes > COSTLY_REMAT_ABOVE_BYTES {
        st.rematerialized = 0;
        (h, h_out) = rematerialize(ng, out, backward, REMAT_MIN, true, &mut st);
    }
    st.nodes_after = h.reachable(&h_out).iter().filter(|&&k| k).count();
    (h, h_out, st)
}

/// Whether reassociating or factoring matrix products could apply at all. Without a product feeding
/// a product, or a sum of products, the linear rules have nothing to do and the search is skipped.
fn has_matrix_structure(g: &Graph, outputs: &[u32]) -> bool {
    let keep = g.reachable(outputs);
    let is_product = |id: u32| {
        let mut n = &g.nodes[id as usize];
        if matches!(&n.op, Op::Transpose(p) if p.as_slice() == [1, 0]) {
            n = &g.nodes[n.inputs[0] as usize];
        }
        matches!(n.op, Op::MatMul)
    };
    g.nodes.iter().enumerate().any(|(i, n)| {
        keep[i]
            && match n.op {
                Op::MatMul => n.inputs.iter().any(|&x| is_product(x)),
                Op::Add => n.inputs.iter().all(|&x| is_product(x)),
                _ => false,
            }
    })
}

/// `remat_min` is the smallest value recomputation applies to; tests lower it to reach every path.
pub fn optimize_with(g: &Graph, outputs: &[u32], remat_min: usize) -> (Graph, Vec<u32>, OptStats) {
    let (ng, out, mut st, backward) = simplify(g, outputs, false, true);
    let (h, out) = rematerialize(&ng, &out, &backward, remat_min, true, &mut st);
    st.nodes_after = h.reachable(&out).iter().filter(|&&k| k).count();
    (h, out, st)
}

/// Single forward rewrite: SSA order means folds and identities cascade without a fixpoint loop.
fn simplify(g: &Graph, outputs: &[u32], metal: bool, flip_transposes: bool) -> (Graph, Vec<u32>, OptStats, Vec<bool>) {
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
            ng.nodes[map[i] as usize].op = node.op.clone(); // reverse scans renumber their inputs
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
        if flip_transposes && matches!(op, Op::MatMul) {
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

        let foldable = !op.is_control_flow() && !matches!(op, Op::Output { .. }) && !ins.is_empty() && node.n() <= FOLD_LIMIT && ins.iter().all(|&x| matches!(ng.nodes[x as usize].op, Op::Const(_)));
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
        ng.nodes[id as usize].origin = node.origin; // only presence is read after this point
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
