use crate::tensor::{DType, Tensor};
use serde_json::{json, Value};
use std::sync::Arc;

/// Diagnostics are plain JSON: the machine-readable form is the only form.
pub type Diag = Value;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReduceKind {
    Sum,
    Max,
    Min,
}

#[derive(Clone, Debug)]
pub enum Op {
    Input(usize),
    Const(Arc<Tensor>),
    Neg,
    Exp,
    Log,
    Sqrt,
    Sin,
    Cos,
    Tanh,
    Abs,
    Sign,
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Max,
    Min,
    // shortcut: comparisons return 0/1 floats, add a bool dtype when integer tensors arrive
    Gt,
    Ge,
    Where,
    MatMul,
    /// Produced by the optimizer only: multiply with stored operands read as transposed, no copy.
    MatMulT { ta: bool, tb: bool },
    Reduce { kind: ReduceKind, axes: Vec<usize>, keepdims: bool },
    Reshape(Vec<usize>),
    Transpose(Vec<usize>),
    BroadcastTo(Vec<usize>),
    Slice { starts: Vec<usize>, stops: Vec<usize> },
    Pad { starts: Vec<usize>, shape: Vec<usize> },
    Cast(DType),
    StopGradient,
    Floor,
    /// `x[.., idx, ..]` along `axis`; the index shape replaces that axis.
    Take { axis: usize },
    /// `take_along_axis` on the last axis.
    TakeAlong,
    /// Adjoints of the takes: add values back at the indexed positions.
    TakeGrad { axis: usize, shape: Vec<usize> },
    TakeAlongGrad { shape: Vec<usize> },
    Concat { axis: usize },
    /// Index of the largest element on the last axis.
    Argmax,
    RandUniform { shape: Vec<usize>, dtype: DType },
    RandNormal { shape: Vec<usize>, dtype: DType },
    KeySplit { n: usize },
    /// Stands for the cotangent inside a custom VJP rule; never executed.
    Placeholder,
    /// Identity on input 0, differentiated by the rule instead of through it.
    Custom { placeholder: u32, rules: Vec<(u32, u32)> },
}

impl Op {
    pub fn name(&self) -> &'static str {
        use Op::*;
        match self {
            Input(_) => "input",
            Const(_) => "const",
            Neg => "neg",
            Exp => "exp",
            Log => "log",
            Sqrt => "sqrt",
            Sin => "sin",
            Cos => "cos",
            Tanh => "tanh",
            Abs => "abs",
            Sign => "sign",
            Add => "add",
            Sub => "sub",
            Mul => "mul",
            Div => "div",
            Pow => "pow",
            Max => "maximum",
            Min => "minimum",
            Gt => "gt",
            Ge => "ge",
            Where => "where",
            MatMul => "matmul",
            MatMulT { .. } => "matmul_t",
            Reduce { kind: ReduceKind::Sum, .. } => "sum",
            Reduce { kind: ReduceKind::Max, .. } => "max",
            Reduce { kind: ReduceKind::Min, .. } => "min",
            Reshape(_) => "reshape",
            Transpose(_) => "transpose",
            BroadcastTo(_) => "broadcast_to",
            Slice { .. } => "slice",
            Pad { .. } => "pad",
            Cast(_) => "cast",
            StopGradient => "stop_gradient",
            Floor => "floor",
            Take { .. } => "take",
            TakeAlong => "take_along",
            TakeGrad { .. } => "take_grad",
            TakeAlongGrad { .. } => "take_along_grad",
            Concat { .. } => "concat",
            Argmax => "argmax",
            RandUniform { .. } => "random_uniform",
            RandNormal { .. } => "random_normal",
            KeySplit { .. } => "key_split",
            Placeholder => "placeholder",
            Custom { .. } => "custom_vjp",
        }
    }
    pub fn is_unary_elementwise(&self) -> bool {
        use Op::*;
        matches!(self, Neg | Exp | Log | Sqrt | Sin | Cos | Tanh | Abs | Sign | Floor)
    }
    /// Ops defined on int32. Everything else is float-only.
    pub fn int_ok(&self) -> bool {
        use Op::*;
        matches!(self, Add | Sub | Mul | Gt | Ge | Where | Reshape(_) | Transpose(_) | BroadcastTo(_) | Slice { .. } | Concat { .. } | Take { .. } | TakeAlong | Cast(_) | StopGradient)
    }
    pub fn is_binary_elementwise(&self) -> bool {
        use Op::*;
        matches!(self, Add | Sub | Mul | Div | Pow | Max | Min | Gt | Ge)
    }
    /// Piecewise ops make finite differences unreliable at kinks, so mismatches stay `suspected`.
    pub fn is_nonsmooth(&self) -> bool {
        use Op::*;
        matches!(self, Abs | Sign | Floor | Max | Min | Gt | Ge | Where | Custom { .. } | Reduce { kind: ReduceKind::Max | ReduceKind::Min, .. })
    }
}

#[derive(Clone, Debug)]
pub struct Node {
    pub op: Op,
    pub inputs: Vec<u32>,
    pub shape: Vec<usize>,
    pub dtype: DType,
    pub source: Option<Arc<str>>,
    /// For backward nodes: the forward node whose VJP produced this node.
    pub origin: Option<u32>,
}

impl Node {
    pub fn n(&self) -> usize {
        self.shape.iter().product()
    }
}

#[derive(Clone, Debug)]
pub struct GradRecord {
    pub y: u32,
    pub wrt: u32,
    pub grad: u32,
}

#[derive(Clone, Debug, Default)]
pub struct Graph {
    pub nodes: Vec<Node>,
    pub n_inputs: usize,
    pub grads: Vec<GradRecord>,
    pub notes: Vec<Diag>,
    pub cur_source: Option<Arc<str>>,
    pub cur_origin: Option<u32>,
}

pub fn diag(code: &str, severity: &str, node: Value, observation: Value, cause: &str, certainty: &str) -> Diag {
    json!({
        "status": if severity == "error" { "validation_failed" } else { "warning" },
        "severity": severity,
        "code": code,
        "node": node,
        "observation": observation,
        "diagnosis": { "cause": cause, "certainty": certainty },
    })
}

pub fn broadcast_shapes(a: &[usize], b: &[usize]) -> Option<Vec<usize>> {
    let r = a.len().max(b.len());
    let mut out = vec![0; r];
    for i in 0..r {
        let x = if i < a.len() { a[a.len() - 1 - i] } else { 1 };
        let y = if i < b.len() { b[b.len() - 1 - i] } else { 1 };
        if x != y && x != 1 && y != 1 {
            return None;
        }
        out[r - 1 - i] = x.max(y);
    }
    Some(out)
}

impl Graph {
    pub fn new() -> Graph {
        Graph::default()
    }

    pub fn node_json(&self, id: u32) -> Value {
        let n = &self.nodes[id as usize];
        let mut v = json!({
            "id": id,
            "operation": n.op.name(),
            "source": n.source.as_deref(),
            "role": if n.origin.is_some() { "backward" } else { "forward" },
        });
        if let Some(o) = n.origin {
            let f = &self.nodes[o as usize];
            v["forward_node"] = json!({ "id": o, "operation": f.op.name(), "source": f.source.as_deref() });
        }
        v
    }

    /// Node description for a failure before the node exists.
    fn pending_json(&self, op: &Op) -> Value {
        json!({ "id": null, "operation": op.name(), "source": self.cur_source.as_deref(), "role": "forward" })
    }

    fn static_err(&self, code: &str, op: &Op, obs: Value, cause: &str) -> Diag {
        diag(code, "error", self.pending_json(op), obs, cause, "verified")
    }

    fn push(&mut self, op: Op, inputs: Vec<u32>, shape: Vec<usize>, dtype: DType) -> u32 {
        self.nodes.push(Node { op, inputs, shape, dtype, source: self.cur_source.clone(), origin: self.cur_origin });
        (self.nodes.len() - 1) as u32
    }

    pub fn input(&mut self, shape: &[usize], dtype: DType) -> u32 {
        let idx = self.n_inputs;
        self.n_inputs += 1;
        self.push(Op::Input(idx), vec![], shape.to_vec(), dtype)
    }

    pub fn placeholder(&mut self, shape: &[usize], dtype: DType) -> u32 {
        self.push(Op::Placeholder, vec![], shape.to_vec(), dtype)
    }

    pub fn constant(&mut self, t: Tensor) -> u32 {
        let (s, d) = (t.shape.clone(), t.dtype());
        self.push(Op::Const(Arc::new(t)), vec![], s, d)
    }

    pub fn scalar(&mut self, d: DType, x: f64) -> u32 {
        self.constant(Tensor::full(d, &[], x))
    }

    pub fn shape(&self, id: u32) -> &[usize] {
        &self.nodes[id as usize].shape
    }

    pub fn dtype(&self, id: u32) -> DType {
        self.nodes[id as usize].dtype
    }

    pub fn add(&mut self, op: Op, inputs: &[u32]) -> Result<u32, Diag> {
        use Op::*;
        for &i in inputs {
            if i as usize >= self.nodes.len() {
                return Err(self.static_err("INVALID_NODE_REF", &op, json!({ "input": i }), "unknown_node_id"));
            }
        }
        let arity = match &op {
            Input(_) | Const(_) | Placeholder => 0,
            Where => 3,
            MatMul | MatMulT { .. } | Take { .. } | TakeAlong | TakeGrad { .. } | TakeAlongGrad { .. } => 2,
            Concat { .. } => inputs.len().max(1),
            o if o.is_binary_elementwise() => 2,
            _ => 1,
        };
        if inputs.len() != arity {
            return Err(self.static_err("ARITY_MISMATCH", &op, json!({ "expected": arity, "actual": inputs.len() }), "wrong_input_count"));
        }
        let dtypes: Vec<DType> = inputs.iter().map(|&i| self.dtype(i)).collect();
        let indexes = matches!(op, Take { .. } | TakeAlong | TakeGrad { .. } | TakeAlongGrad { .. });
        if indexes && dtypes[1] != DType::I32 {
            return Err(self.static_err("DTYPE_MISMATCH", &op, json!({ "index_dtype": dtypes[1].name() }), "index_must_be_int32"));
        }
        if matches!(op, RandUniform { .. } | RandNormal { .. } | KeySplit { .. }) && (dtypes[0] != DType::I32 || self.shape(inputs[0]) != [2]) {
            return Err(self.static_err("INVALID_KEY", &op, json!({ "key_shape": self.shape(inputs[0]), "key_dtype": dtypes[0].name() }), "key_must_be_int32_of_shape_2"));
        }
        let random = matches!(op, RandUniform { .. } | RandNormal { .. } | KeySplit { .. });
        if !indexes && !random && !matches!(op, Cast(_)) && dtypes.windows(2).any(|w| w[0] != w[1]) {
            let names: Vec<_> = dtypes.iter().map(|d| d.name()).collect();
            return Err(self.static_err("DTYPE_MISMATCH", &op, json!({ "input_dtypes": names }), "implicit_promotion_not_allowed"));
        }
        if !random && !matches!(op, Argmax) && dtypes[0] == DType::I32 && !op.int_ok() {
            return Err(self.static_err("UNSUPPORTED_DTYPE", &op, json!({ "dtype": "int32" }), "operation_is_float_only_cast_first"));
        }
        let dtype = match &op {
            Cast(d) => *d,
            RandUniform { dtype, .. } | RandNormal { dtype, .. } => *dtype,
            KeySplit { .. } => DType::I32,
            Argmax => DType::I32,
            _ => dtypes[0],
        };
        let shapes: Vec<Vec<usize>> = inputs.iter().map(|&i| self.nodes[i as usize].shape.clone()).collect();

        if matches!(op, Where) {
            // Materializing the broadcast keeps the Where kernel same-shape only.
            let joined = shapes.iter().try_fold(vec![], |acc: Vec<usize>, s| broadcast_shapes(&acc, s));
            let Some(target) = joined else {
                return Err(self.static_err("SHAPE_MISMATCH", &op, json!({ "input_shapes": shapes }), "incompatible_broadcast"));
            };
            let mut ids = vec![];
            for (k, &i) in inputs.iter().enumerate() {
                ids.push(if shapes[k] == target { i } else { self.add(BroadcastTo(target.clone()), &[i])? });
            }
            return Ok(self.push(op, ids, target, dtype));
        }

        let shape = self.infer(&op, &shapes)?;
        Ok(self.push(op, inputs.to_vec(), shape, dtype))
    }

    fn infer(&self, op: &Op, s: &[Vec<usize>]) -> Result<Vec<usize>, Diag> {
        use Op::*;
        let product = |x: &[usize]| x.iter().product::<usize>();
        match op {
            o if o.is_unary_elementwise() => Ok(s[0].clone()),
            Cast(_) | StopGradient => Ok(s[0].clone()),
            o if o.is_binary_elementwise() => broadcast_shapes(&s[0], &s[1]).ok_or_else(|| {
                self.static_err("SHAPE_MISMATCH", op, json!({ "lhs_shape": s[0], "rhs_shape": s[1] }), "incompatible_broadcast")
            }),
            MatMul => {
                let (a, b) = (&s[0], &s[1]);
                if a.is_empty() || b.is_empty() {
                    return Err(self.static_err("UNSUPPORTED_RANK", op, json!({ "lhs_shape": a, "rhs_shape": b, "minimum_rank": 1 }), "matmul_of_scalar"));
                }
                let k_a = a[a.len() - 1];
                let k_b = if b.len() >= 2 { b[b.len() - 2] } else { b[0] };
                if k_a != k_b {
                    return Err(self.static_err("SHAPE_MISMATCH", op, json!({ "lhs_shape": a, "rhs_shape": b, "lhs_inner": k_a, "rhs_inner": k_b }), "matmul_inner_dimension_mismatch"));
                }
                let (ba, bb) = (&a[..a.len().saturating_sub(2)], &b[..b.len().saturating_sub(2)]);
                let mut out = broadcast_shapes(ba, bb).ok_or_else(|| self.static_err("SHAPE_MISMATCH", op, json!({ "lhs_shape": a, "rhs_shape": b }), "matmul_batch_dimensions_incompatible"))?;
                if a.len() >= 2 {
                    out.push(a[a.len() - 2]);
                }
                if b.len() >= 2 {
                    out.push(b[b.len() - 1]);
                }
                Ok(out)
            }
            MatMulT { ta, tb } => {
                let view = |t: &[usize], flip: bool| {
                    let mut v = t.to_vec();
                    if flip {
                        v.swap(t.len() - 2, t.len() - 1);
                    }
                    v
                };
                self.infer(&MatMul, &[view(&s[0], *ta), view(&s[1], *tb)])
            }
            Take { axis } => {
                if *axis >= s[0].len() || s[0][*axis] == 0 {
                    return Err(self.static_err("INVALID_AXIS", op, json!({ "shape": s[0], "axis": axis }), "axis_out_of_range_or_empty"));
                }
                Ok([&s[0][..*axis], &s[1][..], &s[0][*axis + 1..]].concat())
            }
            TakeAlong => {
                let ok = !s[0].is_empty() && s[0].len() == s[1].len() && s[0][..s[0].len() - 1] == s[1][..s[1].len() - 1] && s[0][s[0].len() - 1] > 0;
                if !ok {
                    return Err(self.static_err("SHAPE_MISMATCH", op, json!({ "values_shape": s[0], "index_shape": s[1] }), "take_along_requires_matching_batch_dimensions"));
                }
                Ok(s[1].clone())
            }
            TakeGrad { shape, .. } | TakeAlongGrad { shape } => Ok(shape.clone()),
            Concat { axis } => {
                let rank = s[0].len();
                let ok = *axis < rank && s.iter().all(|t| t.len() == rank && (0..rank).all(|d| d == *axis || t[d] == s[0][d]));
                if !ok {
                    return Err(self.static_err("SHAPE_MISMATCH", op, json!({ "input_shapes": s, "axis": axis }), "concat_shapes_differ_off_axis"));
                }
                let mut out = s[0].clone();
                out[*axis] = s.iter().map(|t| t[*axis]).sum();
                Ok(out)
            }
            Argmax => {
                if s[0].is_empty() || s[0][s[0].len() - 1] == 0 {
                    return Err(self.static_err("INVALID_AXIS", op, json!({ "shape": s[0] }), "argmax_needs_a_nonempty_last_axis"));
                }
                Ok(s[0][..s[0].len() - 1].to_vec())
            }
            RandUniform { shape, .. } | RandNormal { shape, .. } => Ok(shape.clone()),
            KeySplit { n } => Ok(vec![*n, 2]),
            Custom { .. } => Ok(s[0].clone()),
            Reduce { axes, keepdims, .. } => {
                let r = s[0].len();
                let mut seen = vec![false; r];
                for &a in axes {
                    if a >= r || seen[a] {
                        return Err(self.static_err("INVALID_AXIS", op, json!({ "shape": s[0], "axes": axes }), "axis_out_of_range_or_duplicate"));
                    }
                    seen[a] = true;
                }
                if axes.iter().any(|&a| s[0][a] == 0) {
                    return Err(self.static_err("INVALID_AXIS", op, json!({ "shape": s[0], "axes": axes }), "reduction_over_empty_axis"));
                }
                Ok((0..r).filter_map(|i| if seen[i] { keepdims.then_some(1) } else { Some(s[0][i]) }).collect())
            }
            Reshape(t) => {
                if product(t) != product(&s[0]) {
                    return Err(self.static_err("SHAPE_MISMATCH", op, json!({ "input_shape": s[0], "target_shape": t }), "reshape_element_count_mismatch"));
                }
                Ok(t.clone())
            }
            Transpose(p) => {
                let mut sorted = p.clone();
                sorted.sort_unstable();
                if sorted != (0..s[0].len()).collect::<Vec<_>>() {
                    return Err(self.static_err("INVALID_AXIS", op, json!({ "shape": s[0], "perm": p }), "invalid_permutation"));
                }
                Ok(p.iter().map(|&i| s[0][i]).collect())
            }
            BroadcastTo(t) => {
                if broadcast_shapes(&s[0], t).as_deref() != Some(t.as_slice()) {
                    return Err(self.static_err("SHAPE_MISMATCH", op, json!({ "input_shape": s[0], "target_shape": t }), "incompatible_broadcast"));
                }
                Ok(t.clone())
            }
            Slice { starts, stops } => {
                let ok = starts.len() == s[0].len() && stops.len() == s[0].len() && (0..s[0].len()).all(|i| starts[i] <= stops[i] && stops[i] <= s[0][i]);
                if !ok {
                    return Err(self.static_err("INVALID_SLICE", op, json!({ "shape": s[0], "starts": starts, "stops": stops }), "slice_out_of_bounds"));
                }
                Ok((0..s[0].len()).map(|i| stops[i] - starts[i]).collect())
            }
            Pad { starts, shape } => {
                let ok = starts.len() == s[0].len() && shape.len() == s[0].len() && (0..s[0].len()).all(|i| starts[i] + s[0][i] <= shape[i]);
                if !ok {
                    return Err(self.static_err("INVALID_SLICE", op, json!({ "shape": s[0], "starts": starts, "target_shape": shape }), "pad_out_of_bounds"));
                }
                Ok(shape.clone())
            }
            _ => unreachable!("op without inputs reached infer"),
        }
    }

    /// Nodes needed to compute `outputs`, as a mask over node ids.
    pub fn reachable(&self, outputs: &[u32]) -> Vec<bool> {
        let mut keep = vec![false; self.nodes.len()];
        for &o in outputs {
            keep[o as usize] = true;
        }
        for i in (0..self.nodes.len()).rev() {
            if keep[i] {
                for &x in &self.nodes[i].inputs {
                    keep[x as usize] = true;
                }
            }
        }
        keep
    }

    /// Same graph evaluated entirely in f64: finite differences are meaningless at f32 precision.
    pub fn to_f64(&self) -> Graph {
        let mut g = self.clone();
        for n in &mut g.nodes {
            if n.dtype == DType::I32 {
                continue;
            }
            n.dtype = DType::F64;
            match &mut n.op {
                Op::Const(t) => {
                    *t = Arc::new(Tensor::from_f64(DType::F64, &t.shape, &t.data.to_f64()));
                }
                Op::Cast(d) => *d = DType::F64,
                Op::RandUniform { dtype, .. } | Op::RandNormal { dtype, .. } => *dtype = DType::F64,
                _ => {}
            }
        }
        g
    }
}
