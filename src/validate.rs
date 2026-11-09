use crate::graph::{diag, Diag, Graph, Op};
use crate::optimize::OptStats;
use crate::metal::MetalProgram;
use crate::plan::{CpuExec, Exec, Fuse, Instr, Observer, Plan};
use crate::tensor::{Buf, DType, Elem, Tensor};
use serde_json::{json, Value};

#[derive(Clone, Copy, PartialEq, PartialOrd, Debug)]
pub enum Level {
    None,
    Basic,
    Standard,
    Thorough,
}

impl Level {
    pub fn parse(s: &str) -> Option<Level> {
        match s {
            "none" => Some(Level::None),
            "basic" => Some(Level::Basic),
            "standard" => Some(Level::Standard),
            "thorough" => Some(Level::Thorough),
            _ => None,
        }
    }
}

/// JSON cannot carry Inf/NaN as numbers.
pub fn num(x: f64) -> Value {
    if x.is_finite() {
        json!(x)
    } else if x.is_nan() {
        json!("nan")
    } else if x > 0.0 {
        json!("inf")
    } else {
        json!("-inf")
    }
}

#[derive(Clone, Debug)]
struct Stats {
    min: f64,
    max: f64,
    nan: usize,
    inf: usize,
    zero: usize,
    has_fraction: bool,
    first_bad: Option<usize>,
    n: usize,
}

fn stats_of<T: Elem>(v: &[T]) -> Stats {
    let mut s = Stats { min: f64::INFINITY, max: f64::NEG_INFINITY, nan: 0, inf: 0, zero: 0, has_fraction: false, first_bad: None, n: v.len() };
    for (i, &x) in v.iter().enumerate() {
        let f = x.to_json_f64();
        if f.is_nan() {
            s.nan += 1;
        } else if f.is_infinite() {
            s.inf += 1;
        } else {
            s.min = s.min.min(f);
            s.max = s.max.max(f);
            s.zero += (f == 0.0) as usize;
            s.has_fraction |= f.fract() != 0.0;
            continue;
        }
        s.first_bad.get_or_insert(i);
    }
    s
}

fn stats(b: &Buf, n: usize) -> Stats {
    match b {
        Buf::F32(v) => stats_of(&v[..n]),
        Buf::F64(v) => stats_of(&v[..n]),
        Buf::I32(v) => stats_of(&b.prefix(n).to_f64()[..v[..n].len()]),
    }
}

fn any_nonfinite(b: &Buf, n: usize) -> bool {
    match b {
        Buf::F32(v) => v[..n].iter().any(|x| !x.is_finite()),
        Buf::F64(v) => v[..n].iter().any(|x| !x.is_finite()),
        Buf::I32(_) => false,
    }
}

fn value_at(b: &Buf, i: usize) -> f64 {
    match b {
        Buf::F32(v) => v[i].to_json_f64(),
        Buf::F64(v) => v[i],
        Buf::I32(v) => v[i] as f64,
    }
}

/// Root-cause rule given finite inputs and a non-finite output: returns (code, cause, certainty).
fn classify(i: &Instr, args: &[Stats], out: &Stats) -> (&'static str, &'static str, &'static str) {
    use Op::*;
    match &i.op {
        Exp => ("NUMERIC_OVERFLOW", "exponential_overflow", "verified"),
        Log if args[0].min < 0.0 => ("DOMAIN_ERROR", "log_of_negative", "verified"),
        Log => ("DOMAIN_ERROR", "log_of_zero", "verified"),
        Sqrt => ("DOMAIN_ERROR", "sqrt_of_negative", "verified"),
        Div if args[1].zero > 0 => ("DIVIDE_BY_ZERO", "division_by_zero", "verified"),
        Div => ("NUMERIC_OVERFLOW", "division_overflow", "verified"),
        Pow => {
            // Evidence is per-tensor, so it only pins a single element when the exponent is one value.
            let aligned = args[1].n == 1;
            let c = if aligned { "verified" } else { "suspected" };
            if args[0].min < 0.0 && args[1].has_fraction {
                ("DOMAIN_ERROR", "negative_base_fractional_exponent", c)
            } else if args[0].zero > 0 && args[1].min < 0.0 {
                ("DIVIDE_BY_ZERO", "zero_base_negative_exponent", c)
            } else {
                ("NUMERIC_OVERFLOW", "power_overflow", c)
            }
        }
        Cast(_) => ("NUMERIC_OVERFLOW", "cast_overflow", "verified"),
        Add | Sub | Mul | Reduce { .. } | MatMul => {
            if out.nan == 0 {
                ("NUMERIC_OVERFLOW", "arithmetic_overflow", "verified")
            } else {
                // Finite inputs give NaN here only by cancelling intermediate infinities.
                ("NUMERIC_OVERFLOW", "overflow_then_cancellation", "suspected")
            }
        }
        _ => ("NONFINITE_OUTPUT", "unknown", "unknown"),
    }
}

struct NonFinite<'a> {
    g: &'a Graph,
    /// Control-flow bodies being executed, innermost last.
    scopes: Vec<(std::sync::Arc<crate::graph::Body>, usize)>,
    roots: Vec<Diag>,
    propagated: usize,
    /// Findings that are wrong regardless of whether the outputs stay finite.
    hard: Vec<Diag>,
}

impl NonFinite<'_> {
    /// Node description in the current scope, with where in the control flow it ran.
    fn node_json(&self, id: u32) -> Value {
        match self.scopes.last() {
            None => self.g.node_json(id),
            Some((body, iteration)) => {
                let mut v = body.graph.node_json(id);
                v["control_flow"] = json!({ "iteration": iteration, "depth": self.scopes.len() });
                v
            }
        }
    }
}

fn index_check(o: &mut NonFinite, i: &Instr, slots: &[Buf]) {
    let dim = match &i.op {
        Op::Take { axis } => i.arg_shapes[0][*axis],
        Op::TakeAlong => *i.arg_shapes[0].last().expect("checked by infer"),
        _ => return,
    };
    let n: usize = i.arg_shapes[1].iter().product();
    let s = stats(&slots[i.args[1]], n);
    if n > 0 && (s.min < 0.0 || s.max >= dim as f64) && o.hard.len() < MAX_ROOTS {
        let obs = json!({ "index_min": s.min, "index_max": s.max, "axis_size": dim, "behavior_at_runtime": "clamped" });
        let node = o.node_json(i.node);
        o.hard.push(diag("INDEX_OUT_OF_RANGE", "error", node, obs, "index_outside_axis", "verified"));
    }
}

const MAX_ROOTS: usize = 8;

impl Observer for NonFinite<'_> {
    fn after(&mut self, i: &Instr, slots: &[Buf]) {
        index_check(self, i, slots);
        if !any_nonfinite(&slots[i.dst], i.n) {
            return;
        }
        let ns: Vec<usize> = i.arg_shapes.iter().map(|s| s.iter().product()).collect();
        let args: Vec<Stats> = i.args.iter().zip(&ns).map(|(&a, &n)| stats(&slots[a], n)).collect();
        if args.iter().any(|a| a.first_bad.is_some()) {
            self.propagated += 1;
            return;
        }
        if self.roots.len() == MAX_ROOTS {
            return;
        }
        let out = stats(&slots[i.dst], i.n);
        let (code, cause, certainty) = classify(i, &args, &out);
        let bad = out.first_bad.expect("output was just found non-finite");
        let elementwise = i.op.is_unary_elementwise() || i.op.is_binary_elementwise();
        let witness = (elementwise && i.mode != crate::plan::Mode::General).then(|| {
            let vals: Vec<Value> = i.args.iter().zip(&ns).map(|(&a, &n)| num(value_at(&slots[a], if n == i.n { bad } else { 0 }))).collect();
            json!({ "index": bad, "inputs": vals, "output": num(value_at(&slots[i.dst], bad)) })
        });
        let mut obs = json!({
            "dtype": i.dtype.name(),
            "inputs": args.iter().zip(&i.arg_shapes).map(|(a, s)| json!({
                "shape": s, "min": num(a.min), "max": num(a.max), "contains_zero": a.zero > 0,
            })).collect::<Vec<_>>(),
            "output_shape": i.shape,
            "output_contains_inf": out.inf > 0,
            "output_contains_nan": out.nan > 0,
            "nonfinite_count": out.nan + out.inf,
            "witness": witness,
        });
        if args.len() == 1 {
            obs["input_max"] = num(args[0].max);
            obs["input_min"] = num(args[0].min);
            if matches!(i.op, Op::Exp) {
                let limit = if i.dtype == DType::F32 { f32::EXP_MAX } else { f64::EXP_MAX };
                obs["overflow_threshold"] = num(limit);
            }
        }
        let node = self.node_json(i.node);
        self.roots.push(diag(code, "error", node, obs, cause, certainty));
    }
    fn enter_body(&mut self, body: &std::sync::Arc<crate::graph::Body>, iteration: usize) {
        self.scopes.push((body.clone(), iteration));
    }
    fn leave_body(&mut self) {
        self.scopes.pop();
    }
    fn while_limit(&mut self, i: &Instr) {
        if let Op::While { max_iterations, .. } = i.op {
            let node = self.node_json(i.node);
            self.hard.push(diag("WHILE_LIMIT_REACHED", "error", node, json!({ "max_iterations": max_iterations }), "condition_stayed_true_past_the_limit", "verified"));
        }
    }
}

pub struct Compiled {
    pub raw: Graph,
    pub outputs: Vec<u32>,
    pub plan: Plan,
    raw_plan: Plan,
    pub report: Value,
    pub level: Level,
    /// Present when the plan runs on the GPU; diagnostics always stay on the CPU reference.
    pub gpu: Option<MetalProgram>,
    /// Finished CPU frames: first-touching a large frame's pages costs many steps' worth of time.
    frames: std::sync::Mutex<Vec<crate::plan::Frame>>,
    /// The optimization mode, which the equivalence check must reproduce.
    mode: crate::optimize::Mode,
    pub opt_stats: OptStats,
}

#[derive(Clone, Debug)]
pub struct RunSpec {
    /// (input index, output index) pairs fed back after every step.
    pub state: Vec<(usize, usize)>,
    pub data: Vec<usize>,
    /// Scalar outputs recorded per step.
    pub history: Vec<usize>,
}

#[derive(Debug)]
pub struct RunOut {
    pub state: Vec<Tensor>,
    pub outputs: Vec<Tensor>,
    pub history: Vec<Vec<f64>>,
    pub steps: usize,
    pub nonfinite_step: Option<usize>,
    /// Time in the step loop, excluding one-time buffer setup.
    pub seconds: f64,
    /// Set when the device failed; `steps` then counts only the completed steps.
    pub device_error: Option<String>,
}

fn read_scalar(b: &Buf) -> f64 {
    match b {
        Buf::F32(v) => v[0] as f64,
        Buf::F64(v) => v[0],
        Buf::I32(v) => v[0] as f64,
    }
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

/// The hot loop: no Python and no observer, with no allocation in the common path.
fn run_loop<E: Exec>(p: &Plan, e: &mut E, spec: &RunSpec, state: &[Tensor], batches: &[Vec<Tensor>], steps: usize) -> RunOut {
    for (&(i, _), t) in spec.state.iter().zip(state) {
        e.load(i, t);
    }
    let load_batch = |e: &mut E, b: &[Tensor]| {
        for (&i, t) in spec.data.iter().zip(b) {
            e.load(i, t);
        }
    };
    if batches.len() == 1 {
        load_batch(e, &batches[0]);
    }

    // State moves by swapping buffers when the output's slot is private and the same size.
    let out_slots: Vec<usize> = spec.state.iter().map(|&(_, o)| p.outputs[o].slot).collect();
    let mut swaps = vec![];
    let mut copies = vec![];
    for &(i, o) in &spec.state {
        let (is, os) = (p.input_slots[i], p.outputs[o].slot);
        if is == os {
            continue;
        }
        let unique = out_slots.iter().filter(|&&s| s == os).count() == 1;
        if !p.pinned[os] && unique && p.slots[os].1 == p.slots[is].1 {
            swaps.push((is, os));
        } else {
            copies.push((is, os, p.outputs[o].n));
        }
    }
    let hist_slots: Vec<usize> = spec.history.iter().map(|&o| p.outputs[o].slot).collect();
    let mut history = vec![Vec::with_capacity(steps); hist_slots.len()];
    let mut outputs = vec![];
    let mut done = 0;
    let mut nonfinite_step = None;
    let mut device_error = None;

    let started = std::time::Instant::now();
    for k in 0..steps {
        if batches.len() > 1 {
            load_batch(e, &batches[k % batches.len()]);
        }
        e.step();
        device_error = e.failure();
        if device_error.is_some() {
            break;
        }
        done = k + 1;
        let mut bad = false;
        for (h, &s) in history.iter_mut().zip(&hist_slots) {
            let v = e.scalar(s);
            bad |= !v.is_finite();
            h.push(v);
        }
        if bad {
            nonfinite_step = Some(k);
        }
        if k + 1 == steps || bad {
            outputs = p.outputs.iter().map(|o| Tensor { shape: o.shape.clone(), data: e.prefix(o.slot, o.n) }).collect();
        }
        let temp: Vec<(usize, Buf)> = copies.iter().map(|&(is, os, n)| (is, e.prefix(os, n))).collect();
        for &(is, os) in &swaps {
            e.swap(is, os);
        }
        for (is, b) in temp {
            e.write_prefix(is, &b);
        }
        if bad {
            break;
        }
    }
    let seconds = started.elapsed().as_secs_f64();
    let final_state = spec.state.iter().map(|&(i, o)| Tensor { shape: p.outputs[o].shape.clone(), data: e.prefix(p.input_slots[i], p.input_ns[i]) }).collect();
    RunOut { state: final_state, outputs, history, steps: done, nonfinite_step, seconds, device_error }
}

pub fn device_failure(error: &str) -> Diag {
    diag("DEVICE_EXECUTION_FAILED", "error", Value::Null, json!({ "device": "metal", "detail": error }), "gpu_reported_a_failure", "verified")
}

pub const VALIDATION_SEED: u64 = 0x9E37_79B9_7F4A_7C15;
const GRAD_CHECK_SAMPLES: usize = 6;
const GRAD_CHECK_REL_TOL: f64 = 1e-4;
const GRAD_CHECK_ABS_TOL: f64 = 1e-7;
const PERTURBATION: f64 = 1e-3;
const EQUIVALENCE_TOL: f64 = 1e-7;
const DEVICE_TOL: f64 = 1e-2;

fn as_f64(t: &Tensor) -> Tensor {
    if t.dtype() == DType::I32 {
        t.clone()
    } else {
        Tensor::from_f64(DType::F64, &t.shape, &t.data.to_f64())
    }
}

pub fn compile(raw: Graph, outputs: Vec<u32>, examples: &[Tensor], level: Level) -> Result<Compiled, Diag> {
    compile_on(raw, outputs, examples, level, false)
}

pub fn compile_on(raw: Graph, outputs: Vec<u32>, examples: &[Tensor], level: Level, use_metal: bool) -> Result<Compiled, Diag> {
    compile_mode(raw, outputs, examples, level, use_metal, crate::optimize::Mode::Standard)
}

pub fn compile_mode(raw: Graph, outputs: Vec<u32>, examples: &[Tensor], level: Level, use_metal: bool, mode: crate::optimize::Mode) -> Result<Compiled, Diag> {
    use crate::optimize::Mode;
    if mode == Mode::Auto {
        return compile_auto(raw, outputs, examples, level, use_metal);
    }
    let mut c = build(raw.clone(), outputs.clone(), examples, level, use_metal, mode)?;
    // Verify, then adopt: a rewrite that fails the equivalence check is discarded, not reported as
    // a failure of the user's program.
    let rewritten = c.opt_stats.search.as_ref().is_some_and(|s| s.0);
    if rewritten && has_equivalence_failure(&c.report) {
        let rejected = c.report["diagnostics"].as_array().and_then(|d| d.iter().find(|x| x["code"] == "OPTIMIZER_MISMATCH")).cloned();
        c = build(raw, outputs, examples, level, use_metal, Mode::Standard)?;
        if let Some(d) = rejected {
            let note = diag("REWRITE_REJECTED", "warning", d["node"].clone(), d["observation"].clone(), "rewritten_program_failed_the_equivalence_check_standard_plan_used", "verified");
            push_diagnostic(&mut c.report, note);
        }
    }
    Ok(c)
}

fn has_equivalence_failure(report: &Value) -> bool {
    report["diagnostics"].as_array().is_some_and(|d| d.iter().any(|x| x["code"] == "OPTIMIZER_MISMATCH"))
}

fn push_diagnostic(report: &mut Value, d: Diag) {
    if let Some(list) = report["diagnostics"].as_array_mut() {
        list.push(d);
    }
    if report["status"] == "ok" {
        report["status"] = json!("warning");
    }
}

/// Rounds of timing, and how long they may take. A long-running step ends the rounds early.
const AUTO_ROUNDS: usize = 7;
const AUTO_BUDGET_SECONDS: f64 = 0.5;
const AUTO_MIN_SPEEDUP: f64 = 1.05;

/// Median seconds per run of each candidate, interleaved so neither gets the warmed-up machine.
fn timed_pair(a: &Compiled, b: &Compiled, examples: &[Tensor]) -> (f64, f64) {
    let (mut ta, mut tb) = (vec![], vec![]);
    let started = std::time::Instant::now();
    for round in 0..AUTO_ROUNDS {
        for k in [round % 2, 1 - round % 2] {
            let (c, times) = if k == 0 { (a, &mut ta) } else { (b, &mut tb) };
            let t0 = std::time::Instant::now();
            let _ = c.try_call(examples);
            times.push(t0.elapsed().as_secs_f64());
        }
        if started.elapsed().as_secs_f64() > AUTO_BUDGET_SECONDS {
            break;
        }
    }
    let median = |v: &mut Vec<f64>| {
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    };
    (median(&mut ta), median(&mut tb))
}

/// Lets the hardware decide between the plain plan and the rewritten one.
fn compile_auto(raw: Graph, outputs: Vec<u32>, examples: &[Tensor], level: Level, use_metal: bool) -> Result<Compiled, Diag> {
    use crate::optimize::Mode;
    let standard = build(raw.clone(), outputs.clone(), examples, Level::None, use_metal, Mode::Standard)?;
    let searched = compile_mode(raw.clone(), outputs.clone(), examples, Level::Basic, use_metal, Mode::SearchLinear)?;
    let rewritten = searched.opt_stats.search.as_ref().is_some_and(|s| s.0);
    let (mut use_search, mut timing) = (false, Value::Null);
    if rewritten {
        // The first call of each pays for allocation; only the later runs count.
        let _ = (standard.try_call(examples), searched.try_call(examples));
        let (a, b) = timed_pair(&standard, &searched, examples);
        use_search = a / b >= AUTO_MIN_SPEEDUP;
        timing = json!({ "standard_seconds": a, "rewritten_seconds": b });
    }
    drop(standard);
    drop(searched);
    let chosen = if use_search { Mode::SearchLinear } else { Mode::Standard };
    let mut c = compile_mode(raw, outputs, examples, level, use_metal, chosen)?;
    c.report["auto"] = json!({ "chosen": if use_search { "search-linear" } else { "standard" }, "rewrite_proposed": rewritten, "timing": timing });
    Ok(c)
}

fn build(raw: Graph, outputs: Vec<u32>, examples: &[Tensor], level: Level, use_metal: bool, mode: crate::optimize::Mode) -> Result<Compiled, Diag> {
    let (opt, opt_out, opt_stats) = crate::optimize::optimize_mode(&raw, &outputs, use_metal, mode);
    let plan = Plan::build_fused(&opt, &opt_out, if use_metal { Fuse::METAL } else { Fuse::DEFAULT });
    let gpu = if use_metal {
        Some(MetalProgram::new(&plan).map_err(|e| diag("DEVICE_UNAVAILABLE", "error", Value::Null, json!({ "device": "metal", "detail": e }), "metal_device_buffer_limit_or_kernel_compilation_failed", "verified"))?)
    } else {
        None
    };
    let mut all = outputs.clone();
    all.extend(raw.grads.iter().map(|r| r.grad));
    let raw_plan = Plan::build(&raw, &all, false);
    let mut c = Compiled { raw, outputs, plan, raw_plan, report: Value::Null, level, gpu, frames: Default::default(), mode, opt_stats };
    c.check_inputs(examples)?;
    let stats = c.opt_stats.clone();
    c.report = c.validate(examples, &stats);
    Ok(c)
}

impl Compiled {
    pub fn check_inputs(&self, inputs: &[Tensor]) -> Result<(), Diag> {
        let nodes: Vec<(u32, &crate::graph::Node)> = self.raw.nodes.iter().enumerate().filter(|(_, n)| matches!(n.op, Op::Input(_))).map(|(i, n)| (i as u32, n)).collect();
        if nodes.len() != inputs.len() {
            let obs = json!({ "expected_inputs": nodes.len(), "actual_inputs": inputs.len(), "recompile_required": true });
            return Err(diag("INPUT_SIGNATURE_MISMATCH", "error", Value::Null, obs, "input_count_changed", "verified"));
        }
        for (k, ((id, node), t)) in nodes.iter().zip(inputs).enumerate() {
            if node.shape != t.shape || node.dtype != t.dtype() {
                let obs = json!({
                    "input_index": k,
                    "expected_shape": node.shape, "actual_shape": t.shape,
                    "expected_dtype": node.dtype.name(), "actual_dtype": t.dtype().name(),
                    "recompile_required": true,
                });
                return Err(diag("INPUT_SIGNATURE_MISMATCH", "error", self.raw.node_json(*id), obs, "input_signature_changed", "verified"));
            }
        }
        Ok(())
    }

    /// One execution of the optimized plan. Only a device can fail; the CPU path always succeeds.
    pub fn try_call(&self, inputs: &[Tensor]) -> Result<Vec<Tensor>, Diag> {
        if let Some(gpu) = &self.gpu {
            let mut e = gpu.exec(&self.plan).map_err(|error| device_failure(&error))?;
            for (k, t) in inputs.iter().enumerate() {
                e.load(k, t);
            }
            e.step();
            if let Some(error) = e.failure() {
                return Err(device_failure(&error));
            }
            return Ok(self.plan.outputs.iter().map(|o| Tensor { shape: o.shape.clone(), data: e.prefix(o.slot, o.n) }).collect());
        }
        let mut f = self.plan.frame();
        for (k, t) in inputs.iter().enumerate() {
            self.plan.load_input(&mut f, k, t);
        }
        self.plan.exec(&mut f, &mut ());
        Ok((0..self.outputs.len()).map(|k| self.plan.read_output(&f, k)).collect())
    }

    pub fn call(&self, inputs: &[Tensor]) -> Vec<Tensor> {
        self.try_call(inputs).expect("use try_call where the device can fail")
    }

    /// Instrumented run of the unoptimized graph, so node ids and sources match the trace.
    /// Returns the outputs plus the gradient values, in `raw_plan` output order.
    fn observed(&self, inputs: &[Tensor]) -> (Vec<Diag>, Vec<Diag>, usize, Vec<Tensor>) {
        let mut o = NonFinite { g: &self.raw, scopes: vec![], roots: vec![], propagated: 0, hard: vec![] };
        let keep = self.raw.reachable(&self.raw_outputs());
        for (id, n) in self.raw.nodes.iter().enumerate() {
            let t = match &n.op {
                Op::Input(k) => &inputs[*k],
                Op::Const(t) => &**t,
                _ => continue,
            };
            if keep[id] && !t.is_finite() {
                let s = stats(&t.data, t.n());
                let obs = json!({ "dtype": n.dtype.name(), "shape": n.shape, "contains_inf": s.inf > 0, "contains_nan": s.nan > 0, "nonfinite_count": s.nan + s.inf, "first_bad_index": s.first_bad });
                o.roots.push(diag("NONFINITE_INPUT", "error", self.raw.node_json(id as u32), obs, "nonfinite_value_supplied", "verified"));
            }
        }
        let mut f = self.raw_plan.frame();
        for (k, t) in inputs.iter().enumerate() {
            self.raw_plan.load_input(&mut f, k, t);
        }
        self.raw_plan.exec(&mut f, &mut o);
        let vals = (0..self.raw_plan.outputs.len()).map(|k| self.raw_plan.read_output(&f, k)).collect();
        (o.roots, o.hard, o.propagated, vals)
    }

    fn raw_outputs(&self) -> Vec<u32> {
        let mut all = self.outputs.clone();
        all.extend(self.raw.grads.iter().map(|r| r.grad));
        all
    }

    fn gradient_stats(&self, vals: &[Tensor]) -> (Vec<Value>, Vec<Diag>) {
        let mut out = vec![];
        let mut diags = vec![];
        for (k, r) in self.raw.grads.iter().enumerate() {
            let v = vals[self.outputs.len() + k].data.to_f64();
            if v.is_empty() {
                continue;
            }
            let zeros = v.iter().filter(|&&x| x == 0.0).count();
            let sq: f64 = v.iter().map(|x| x * x).sum();
            let max_abs = v.iter().fold(0.0f64, |m, x| if x.abs() > m || x.is_nan() { x.abs() } else { m });
            let zero_ratio = if v.is_empty() { 0.0 } else { zeros as f64 / v.len() as f64 };
            out.push(json!({
                "output_node": r.y, "wrt_node": r.wrt, "grad_node": r.grad,
                "source": self.raw.nodes[r.wrt as usize].source.as_deref(),
                "norm": num(sq.sqrt()),
                "rms": num((sq / v.len().max(1) as f64).sqrt()),
                "max_abs": num(max_abs),
                "zero_ratio": num(zero_ratio),
            }));
            let is_const = matches!(self.raw.nodes[r.grad as usize].op, Op::Const(_));
            if !v.is_empty() && zeros == v.len() && !is_const {
                let obs = json!({ "wrt_node": r.wrt, "zero_ratio": 1.0, "elements": v.len() });
                // One sample cannot separate vanishing gradients from a flat point: never `verified`.
                diags.push(diag("GRAD_ALL_ZERO", "warning", self.raw.node_json(r.grad), obs, "zero_gradient_at_sample", "suspected"));
            }
        }
        (out, diags)
    }

    /// Finite-value check for arbitrary inputs. Severity depends on whether the output is affected:
    /// a NaN masked by `where` is real evidence but does not invalidate the result.
    pub fn check(&self, inputs: &[Tensor]) -> Value {
        let (mut roots, hard, propagated, vals) = self.observed(inputs);
        let bad_outputs: Vec<usize> = (0..self.outputs.len()).filter(|&k| !vals[k].is_finite()).collect();
        if bad_outputs.is_empty() {
            for r in &mut roots {
                r["severity"] = json!("warning");
                r["status"] = json!("warning");
                r["masked"] = json!(true);
            }
        }
        let (grads, grad_diags) = self.gradient_stats(&vals);
        let status = if !bad_outputs.is_empty() || !hard.is_empty() {
            "validation_failed"
        } else if !roots.is_empty() || !grad_diags.is_empty() {
            "warning"
        } else {
            "ok"
        };
        let found = roots.len();
        roots.extend(grad_diags);
        roots.extend(hard);
        json!({
            "status": status,
            "nonfinite": { "root_nodes": found, "propagated_nodes": propagated, "affected_outputs": bad_outputs },
            "gradients": grads,
            "diagnostics": roots,
        })
    }

    /// Compares the optimized, fused, buffer-reusing plan with the plain graph, both in f64.
    /// In f32, rounding alone can differ visibly: Adam divides by near-zero gradients and
    /// amplifies it. In f64 that noise vanishes, so a tight tolerance only catches real bugs.
    fn optimizer_check(&self, examples: &[Tensor]) -> Result<Value, Diag> {
        let shadow = self.raw.to_f64();
        let inputs: Vec<Tensor> = examples.iter().map(as_f64).collect();
        let (opt, ids, _) = crate::optimize::optimize_mode(&shadow, &self.outputs, false, self.mode);
        let run = |plan: &Plan| {
            let mut f = plan.frame();
            for (k, t) in inputs.iter().enumerate() {
                plan.load_input(&mut f, k, t);
            }
            plan.exec(&mut f, &mut ());
            (0..self.outputs.len()).map(|k| plan.read_output(&f, k)).collect::<Vec<_>>()
        };
        let fast = run(&Plan::build_fused(&opt, &ids, Fuse::DEFAULT));
        let reference = run(&Plan::build(&shadow, &self.outputs, false));
        let mut max_rel = 0.0f64;
        for (k, (a, b)) in fast.iter().zip(&reference).enumerate() {
            let (va, vb) = (a.data.to_f64(), b.data.to_f64());
            // Relative to the tensor's scale: elements near zero come from cancellation.
            let scale = vb.iter().filter(|v| v.is_finite()).fold(1e-30f64, |m, v| m.max(v.abs()));
            for (j, (&x, &y)) in va.iter().zip(&vb).enumerate() {
                if x == y || (x.is_nan() && y.is_nan()) {
                    continue;
                }
                let rel = (x - y).abs() / scale;
                max_rel = if rel.is_nan() { f64::INFINITY } else { max_rel.max(rel) };
                if !(rel <= EQUIVALENCE_TOL) {
                    let obs = json!({ "output_index": k, "element": j, "optimized": num(x), "reference": num(y), "relative_error": num(rel), "tolerance": EQUIVALENCE_TOL, "precision": "float64" });
                    let node = self.raw.node_json(self.outputs[k]);
                    return Err(diag("OPTIMIZER_MISMATCH", "error", node, obs, "optimized_plan_disagrees_with_reference", "verified"));
                }
            }
        }
        Ok(json!({ "max_relative_error": num(max_rel), "tolerance": EQUIVALENCE_TOL, "precision": "float64" }))
    }

    /// Compares GPU and CPU results in the program's own dtype. Float rounding differs between
    /// them and normalizations amplify it, so a disagreement is evidence and not a verdict.
    fn device_check(&self, examples: &[Tensor]) -> (Value, Vec<Diag>) {
        let (opt, ids, _) = crate::optimize::optimize_mode(&self.raw, &self.outputs, false, self.mode);
        let cpu_plan = Plan::build_fused(&opt, &ids, Fuse::DEFAULT);
        let mut f = cpu_plan.frame();
        for (k, t) in examples.iter().enumerate() {
            cpu_plan.load_input(&mut f, k, t);
        }
        cpu_plan.exec(&mut f, &mut ());
        let gpu = match self.try_call(examples) {
            Ok(values) => values,
            Err(d) => return (json!({ "skipped": "device_failed" }), vec![d]),
        };
        let mut max_rel = 0.0f64;
        let mut diags = vec![];
        for (k, g) in gpu.iter().enumerate() {
            let (vg, vc) = (g.data.to_f64(), cpu_plan.read_output(&f, k).data.to_f64());
            let scale = vc.iter().filter(|v| v.is_finite()).fold(1e-30f64, |m, v| m.max(v.abs()));
            for (j, (&x, &y)) in vg.iter().zip(&vc).enumerate() {
                if x == y || (x.is_nan() && y.is_nan()) {
                    continue;
                }
                let rel = (x - y).abs() / scale;
                let rel = if rel.is_nan() { f64::INFINITY } else { rel };
                max_rel = max_rel.max(rel);
                if rel > DEVICE_TOL && diags.is_empty() {
                    let obs = json!({ "output_index": k, "element": j, "metal": num(x), "cpu": num(y), "relative_error": num(rel), "tolerance": DEVICE_TOL });
                    diags.push(diag("DEVICE_MISMATCH", "warning", self.raw.node_json(self.outputs[k]), obs, "metal_result_differs_from_cpu", "suspected"));
                }
            }
        }
        (json!({ "max_relative_error": num(max_rel), "tolerance": DEVICE_TOL }), diags)
    }

    /// Central differences in f64 against the analytic gradient of every recorded `grad` call.
    fn grad_check(&self, examples: &[Tensor]) -> (Value, Vec<Diag>) {
        let shadow = self.raw.to_f64();
        let inputs64: Vec<Tensor> = examples.iter().map(as_f64).collect();
        let mut rng = Rng(VALIDATION_SEED);
        let (mut checked, mut skipped, mut max_rel, mut diags) = (0usize, vec![], 0.0f64, vec![]);
        for r in &self.raw.grads {
            let keep = self.raw.reachable(&[r.y]);
            let Op::Input(k) = self.raw.nodes[r.wrt as usize].op else {
                skipped.push(json!({ "wrt_node": r.wrt, "reason": "wrt_is_not_an_input" }));
                continue;
            };
            if matches!(self.raw.nodes[r.grad as usize].op, Op::Const(_)) {
                skipped.push(json!({ "wrt_node": r.wrt, "reason": "disconnected" }));
                continue;
            }
            let on_path = |pred: &dyn Fn(&Op) -> bool| self.raw.nodes.iter().enumerate().any(|(i, n)| keep[i] && pred(&n.op));
            // A stop_gradient may be inert (softmax) or intentional, so disagreement cannot be a verdict.
            let certainty = if on_path(&|o| o.is_nonsmooth() || matches!(o, Op::StopGradient)) { "suspected" } else { "verified" };

            let analytic_plan = Plan::build(&shadow, &[r.grad], true);
            let y_plan = Plan::build(&shadow, &[r.y], true);
            let mut af = analytic_plan.frame();
            for (j, t) in inputs64.iter().enumerate() {
                analytic_plan.load_input(&mut af, j, t);
            }
            analytic_plan.exec(&mut af, &mut ());
            let analytic = analytic_plan.read_output(&af, 0).data.to_f64();

            let mut yf = y_plan.frame();
            for (j, t) in inputs64.iter().enumerate() {
                y_plan.load_input(&mut yf, j, t);
            }
            let mut probe = inputs64[k].clone();
            let base = inputs64[k].data.to_f64();
            for s in 0..GRAD_CHECK_SAMPLES.min(base.len()) {
                let e = if base.len() <= GRAD_CHECK_SAMPLES { s } else { rng.next() as usize % base.len() };
                let h = 1e-6 * base[e].abs().max(1.0);
                let Buf::F64(p) = &mut probe.data else { unreachable!("shadow graph is f64") };
                let mut eval_at = |x: f64| {
                    p[e] = x;
                    let view = Buf::F64(p.clone());
                    yf.slots[y_plan.input_slots[k]].copy_prefix_from(&view, base.len());
                    y_plan.exec(&mut yf, &mut ());
                    read_scalar(&yf.slots[y_plan.outputs[0].slot])
                };
                let (up, down) = (eval_at(base[e] + h), eval_at(base[e] - h));
                p[e] = base[e];
                let numeric = (up - down) / (2.0 * h);
                if !numeric.is_finite() || !analytic[e].is_finite() {
                    continue;
                }
                checked += 1;
                let err = (numeric - analytic[e]).abs();
                let rel = err / numeric.abs().max(analytic[e].abs()).max(1e-12);
                max_rel = max_rel.max(if err <= GRAD_CHECK_ABS_TOL { 0.0 } else { rel });
                if err > GRAD_CHECK_ABS_TOL && rel > GRAD_CHECK_REL_TOL {
                    let obs = json!({ "wrt_node": r.wrt, "element": e, "analytic": num(analytic[e]), "numeric": num(numeric), "relative_error": num(rel), "tolerance": GRAD_CHECK_REL_TOL, "finite_difference_h": h, "seed": VALIDATION_SEED });
                    let severity = if certainty == "verified" { "error" } else { "warning" };
                    diags.push(diag("GRADIENT_MISMATCH", severity, self.raw.node_json(r.grad), obs, "analytic_gradient_disagrees_with_finite_difference", certainty));
                    break;
                }
            }
        }
        (json!({ "elements_checked": checked, "max_relative_error": num(max_rel), "skipped": skipped }), diags)
    }

    fn perturbation_check(&self, examples: &[Tensor]) -> (Value, Vec<Diag>) {
        let mut rng = Rng(VALIDATION_SEED ^ 0xABCD);
        let moved: Vec<Tensor> = examples
            .iter()
            .map(|t| {
                if t.dtype() == DType::I32 {
                    return t.clone(); // tokens and keys are identities, not magnitudes
                }
                let v: Vec<f64> = t.data.to_f64().iter().map(|&x| x * (1.0 + if rng.next() % 2 == 0 { PERTURBATION } else { -PERTURBATION })).collect();
                Tensor::from_f64(t.dtype(), &t.shape, &v)
            })
            .collect();
        let (roots, _, _, _) = self.observed(&moved);
        let diags = roots
            .into_iter()
            .map(|mut d| {
                d["severity"] = json!("warning");
                d["status"] = json!("warning");
                d["trigger"] = json!({ "kind": "input_perturbation", "relative_magnitude": PERTURBATION, "seed": VALIDATION_SEED });
                d
            })
            .collect();
        (json!({ "relative_magnitude": PERTURBATION }), diags)
    }

    fn validate(&self, examples: &[Tensor], opt: &OptStats) -> Value {
        let mut checks = vec![json!({ "name": "static", "status": "pass", "detail": "shape and dtype inference passed during tracing" })];
        let device = if self.gpu.is_some() { "metal" } else { "cpu" };
        let mut diagnostics = self.raw.notes.clone();
        let mut gradients = vec![];
        let check_status = |d: &[Diag]| if d.is_empty() { "pass" } else if d.iter().any(|x| x["severity"] == "error") { "fail" } else { "warn" };

        if self.level >= Level::Basic {
            match self.try_call(examples) {
                Ok(out) => {
                    let shapes: Vec<_> = out.iter().map(|t| t.shape.clone()).collect();
                    checks.push(json!({ "name": "test_run", "status": "pass", "output_shapes": shapes }));
                }
                Err(d) => {
                    checks.push(json!({ "name": "test_run", "status": "fail" }));
                    diagnostics.push(d);
                }
            }
        }
        if self.level >= Level::Standard {
            let found = self.check(examples);
            let ds = found["diagnostics"].as_array().cloned().unwrap_or_default();
            checks.push(json!({ "name": "finite", "status": check_status(&ds), "nonfinite": found["nonfinite"] }));
            gradients = found["gradients"].as_array().cloned().unwrap_or_default();
            diagnostics.extend(ds);
            match self.optimizer_check(examples) {
                Ok(info) => checks.push(json!({ "name": "optimizer_equivalence", "status": "pass", "detail": info })),
                Err(d) => {
                    checks.push(json!({ "name": "optimizer_equivalence", "status": "fail" }));
                    diagnostics.push(d);
                }
            }
            if self.gpu.is_some() {
                let (info, ds) = self.device_check(examples);
                checks.push(json!({ "name": "device_equivalence", "status": check_status(&ds), "detail": info }));
                diagnostics.extend(ds);
            }
        }
        if self.level >= Level::Thorough {
            let (info, ds) = self.grad_check(examples);
            checks.push(json!({ "name": "grad_check", "status": check_status(&ds), "detail": info }));
            diagnostics.extend(ds);
            let (info, ds) = self.perturbation_check(examples);
            checks.push(json!({ "name": "input_perturbation", "status": check_status(&ds), "detail": info }));
            diagnostics.extend(ds);
        }

        let status = if diagnostics.iter().any(|d| d["severity"] == "error") {
            "validation_failed"
        } else if diagnostics.is_empty() {
            "ok"
        } else {
            "warning"
        };
        let (naive, planned) = (self.plan.naive_bytes, self.plan.planned_bytes());
        json!({
            "status": status,
            "device": device,
            "graph": {
                "optimization": opt.json(),
                "steps": self.plan.steps.len(),
                "fused_groups": self.plan.fused().len(),
                "fused_ops": self.plan.fused().iter().map(|f| f.ops.len()).sum::<usize>(),
            },
            "memory": {
                "intermediate_bytes_without_reuse": naive,
                "intermediate_bytes_planned": planned,
                "peak_live_bytes": self.plan.peak_live_bytes,
            },
            "checks": checks,
            "gradients": gradients,
            "diagnostics": diagnostics,
            "repro": { "level": format!("{:?}", self.level).to_lowercase(), "seed": VALIDATION_SEED },
        })
    }

    /// Times every step of the optimized plan and aggregates by operation and shape.
    pub fn profile(&self, inputs: &[Tensor], repeats: usize) -> Value {
        use crate::plan::Step;
        let p = &self.plan;
        let mut seconds = vec![0.0; p.steps.len()];
        let repeats = repeats.max(1);
        let mut breakdown = None;
        if let Some(gpu) = &self.gpu {
            let mut e = match gpu.exec(p) {
                Ok(e) => e,
                Err(error) => return json!({ "error": device_failure(&error) }),
            };
            for (k, t) in inputs.iter().enumerate() {
                e.load(k, t);
            }
            breakdown = Some(e.breakdown(repeats));
            for _ in 0..repeats {
                e.step_timed(&mut seconds);
            }
        } else {
            let mut f = p.frame();
            for (k, t) in inputs.iter().enumerate() {
                p.load_input(&mut f, k, t);
            }
            for _ in 0..repeats {
                p.exec_timed(&mut f, &mut seconds);
            }
        }
        let mut rows: std::collections::BTreeMap<String, (f64, usize, u64)> = Default::default();
        for (s, &t) in p.steps.iter().zip(&seconds) {
            let (key, flops) = match s {
                Step::One(i) => (format!("{} {:?} <- {:?}", i.op.name(), i.shape, i.arg_shapes), i.n as u64),
                Step::Fused(g) => (format!("fused[{}] n={}", g.ops.iter().map(|m| m.op.name()).collect::<Vec<_>>().join(","), g.n), (g.n * g.ops.len()) as u64),
            };
            let e = rows.entry(key).or_default();
            e.0 += t;
            e.1 += 1;
            e.2 += flops;
        }
        let total: f64 = seconds.iter().sum();
        let mut list: Vec<_> = rows.into_iter().collect();
        list.sort_by(|a, b| b.1 .0.total_cmp(&a.1 .0));
        let top: Vec<Value> = list.iter().map(|(k, (t, n, _))| json!({ "operation": k, "count": n, "microseconds_per_run": t / repeats as f64 * 1e6, "share": if total > 0.0 { t / total } else { 0.0 } })).collect();
        json!({ "repeats": repeats, "microseconds_per_run": total / repeats as f64 * 1e6, "operations": top, "breakdown": breakdown })
    }

    pub fn validate_spec(&self, spec: &RunSpec) -> Result<(), Diag> {
        for &(i, o) in &spec.state {
            let input = self.raw.nodes.iter().position(|n| matches!(n.op, Op::Input(k) if k == i)).expect("spec indexes a known input") as u32;
            let out = self.outputs[o];
            if self.raw.shape(input) != self.raw.shape(out) || self.raw.dtype(input) != self.raw.dtype(out) {
                let obs = json!({
                    "input_index": i, "output_index": o,
                    "input_shape": self.raw.shape(input), "output_shape": self.raw.shape(out),
                    "input_dtype": self.raw.dtype(input).name(), "output_dtype": self.raw.dtype(out).name(),
                });
                return Err(diag("STATE_CONTRACT_VIOLATION", "error", self.raw.node_json(out), obs, "returned_state_differs_from_input_state", "verified"));
            }
        }
        Ok(())
    }

    pub fn run(&self, spec: &RunSpec, state: &[Tensor], batches: &[Vec<Tensor>], steps: usize) -> RunOut {
        match &self.gpu {
            Some(gpu) => match gpu.exec(&self.plan) {
                Ok(mut e) => run_loop(&self.plan, &mut e, spec, state, batches, steps),
                Err(error) => RunOut { state: state.to_vec(), outputs: vec![], history: vec![vec![]; spec.history.len()], steps: 0, nonfinite_step: None, seconds: 0.0, device_error: Some(error) },
            },
            None => {
                let frame = self.frames.lock().expect("no panic while pooling").pop().unwrap_or_else(|| self.plan.frame());
                let mut e = CpuExec { plan: &self.plan, frame };
                let out = run_loop(&self.plan, &mut e, spec, state, batches, steps);
                self.frames.lock().expect("no panic while pooling").push(e.frame);
                out
            }
        }
    }

    /// Replays with the fast plan until a state or output first turns non-finite.
    /// Returns (step, batch index, state before that step).
    pub fn replay(&self, spec: &RunSpec, state: &[Tensor], batches: &[Vec<Tensor>], steps: usize) -> Option<(usize, usize, Vec<Tensor>)> {
        let mut cur = state.to_vec();
        for k in 0..steps {
            let b = if batches.is_empty() { 0 } else { k % batches.len() };
            let slice = if batches.is_empty() { &[][..] } else { std::slice::from_ref(&batches[b]) };
            let out = self.run(spec, &cur, slice, 1);
            if out.device_error.is_some() {
                return None;
            }
            if out.state.iter().chain(&out.outputs).any(|t| !t.is_finite()) {
                return Some((k, b, cur));
            }
            cur = out.state;
        }
        None
    }
}
