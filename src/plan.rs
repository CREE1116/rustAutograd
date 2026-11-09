use crate::graph::{Body, Graph, Op, ReduceKind};
use crate::tensor::{Buf, DType, Elem, Storage, Tensor, VFn};
use rayon::prelude::*;
use std::sync::Arc;

/// Elements per interpreter block; small enough that a group's registers stay in L1.
const BLOCK: usize = 256;
/// Elements per parallel task. Fixed, so vector-math tail handling and therefore
/// results cannot depend on the thread count.
const TASK: usize = 16 * BLOCK;

#[derive(Clone, Copy, Debug)]
pub struct Fuse {
    /// Estimated group cost at which going through the thread pool pays for itself.
    /// Below it plain kernels win: measured on M4 Pro, sequential block interpretation
    /// was never faster than whole-array kernels, so a group exists only to be threaded.
    pub parallel_ns: f64,
    /// Compile groups to GPU kernels: float32 only, chains of any cost, limited buffer bindings.
    pub metal: bool,
    /// Order GPU work by dependency level: fewer barriers, but values live longer.
    pub levels: bool,
}

impl Fuse {
    pub const DEFAULT: Fuse = Fuse { parallel_ns: 200_000.0, metal: false, levels: false };
    pub const METAL: Fuse = Fuse { parallel_ns: 0.0, metal: true, levels: true };
}

/// What a device needs to execute a plan; `Compiled::run` is written against this.
pub trait Exec {
    fn load(&mut self, k: usize, t: &Tensor);
    fn step(&mut self);
    fn swap(&mut self, a: usize, b: usize);
    fn prefix(&self, slot: usize, n: usize) -> Buf;
    fn write_prefix(&mut self, slot: usize, b: &Buf);
    fn scalar(&self, slot: usize) -> f64;
    /// Why the last step failed, on devices where execution can fail.
    fn failure(&self) -> Option<String> {
        None
    }
}

pub struct CpuExec<'a> {
    pub plan: &'a Plan,
    pub frame: Frame,
}

impl Exec for CpuExec<'_> {
    fn load(&mut self, k: usize, t: &Tensor) {
        self.plan.load_input(&mut self.frame, k, t);
    }
    #[inline(always)]
    fn step(&mut self) {
        self.plan.exec(&mut self.frame, &mut ());
    }
    fn swap(&mut self, a: usize, b: usize) {
        self.frame.slots.swap(a, b);
    }
    fn prefix(&self, slot: usize, n: usize) -> Buf {
        self.frame.slots[slot].prefix(n)
    }
    fn write_prefix(&mut self, slot: usize, b: &Buf) {
        self.frame.slots[slot].copy_prefix_from(b, b.len());
    }
    #[inline(always)]
    fn scalar(&self, slot: usize) -> f64 {
        match &self.frame.slots[slot] {
            Buf::F32(v) => v[0] as f64,
            Buf::F64(v) => v[0],
            Buf::I32(v) => v[0] as f64,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ArgKind {
    Full,
    Splat,
    /// Shape `[.., L]` repeated along the outer dims of the group's shape.
    Row(usize),
    /// Any other broadcast. Generated GPU kernels compute its index; CPU groups never see it.
    Broadcast,
}

#[derive(Clone, Copy, Debug)]
pub enum Operand {
    Input(usize),
    Reg(usize),
}

#[derive(Clone, Debug)]
pub struct MicroOp {
    pub op: Op,
    pub reg: usize,
    pub args: Vec<Operand>,
}

#[derive(Clone, Debug)]
pub struct Fused {
    pub dtype: DType,
    pub n: usize,
    /// (slot, kind, register for non-Full inputs).
    pub inputs: Vec<(usize, ArgKind, usize)>,
    pub ops: Vec<MicroOp>,
    /// (register, destination slot). Destinations never alias inputs.
    pub outputs: Vec<(usize, usize)>,
    pub regs: usize,
    pub shape: Vec<usize>,
    /// Shape of each input, parallel to `inputs`.
    pub input_shapes: Vec<Vec<usize>>,
}

#[derive(Clone, Debug)]
pub enum Step {
    One(Instr),
    Fused(Fused),
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Mode {
    Same,
    AScalar,
    BScalar,
    General,
}

#[derive(Clone, Debug)]
pub struct Instr {
    pub op: Op,
    pub node: u32,
    pub dtype: DType,
    pub dst: usize,
    pub args: Vec<usize>,
    pub shape: Vec<usize>,
    pub arg_shapes: Vec<Vec<usize>>,
    pub n: usize,
    pub mode: Mode,
    /// Control flow only: (output index, slot) of each output in use. `dst` is unused then.
    pub outs: Vec<(usize, usize)>,
}

#[derive(Clone, Debug)]
pub struct Output {
    pub slot: usize,
    pub shape: Vec<usize>,
    pub n: usize,
}

#[derive(Clone, Debug)]
pub struct Plan {
    pub steps: Vec<Step>,
    pub slots: Vec<(DType, usize)>,
    /// Input and constant slots: never overwritten by an instruction.
    pub pinned: Vec<bool>,
    pub consts: Vec<(usize, Arc<Tensor>)>,
    pub input_slots: Vec<usize>,
    pub input_ns: Vec<usize>,
    pub outputs: Vec<Output>,
    /// Built for diagnostics: control-flow bodies then use their reference plans too.
    pub reference: bool,
    pub naive_bytes: usize,
    /// Largest total of intermediate values alive at once: what no allocator can beat in this order.
    pub peak_live_bytes: usize,
}

pub struct Frame {
    pub slots: Vec<Buf>,
}

/// The executor is monomorphized over this, so the `()` observer leaves no trace in normal runs.
pub trait Observer {
    fn after(&mut self, i: &Instr, slots: &[Buf]);
    /// Instructions until `leave_body` belong to this body, at this iteration.
    fn enter_body(&mut self, _body: &Arc<Body>, _iteration: usize) {}
    fn leave_body(&mut self) {}
    fn while_limit(&mut self, _i: &Instr) {}
}

impl Observer for () {
    #[inline(always)]
    fn after(&mut self, _: &Instr, _: &[Buf]) {}
}

struct Allocator {
    slots: Vec<(DType, usize)>,
    pinned: Vec<bool>,
    refs: Vec<u32>,
    free: Vec<usize>,
    reuse: bool,
    /// Size of the value each slot currently holds, which can be below the slot's capacity.
    held: Vec<usize>,
    live: usize,
    peak: usize,
}

impl Allocator {
    fn new_slot(&mut self, d: DType, n: usize, pinned: bool) -> usize {
        self.slots.push((d, n));
        self.pinned.push(pinned);
        self.refs.push(1);
        self.held.push(0);
        self.slots.len() - 1
    }
    fn alloc(&mut self, d: DType, n: usize) -> usize {
        let s = self.pick(d, n);
        self.held[s] = n * d.size();
        self.live += self.held[s];
        self.peak = self.peak.max(self.live);
        s
    }
    fn pick(&mut self, d: DType, n: usize) -> usize {
        let best = self
            .free
            .iter()
            .enumerate()
            // A small value in a big buffer would pin the big buffer for its whole lifetime.
            .filter(|(_, &s)| self.slots[s].0 == d && self.slots[s].1 >= n && self.slots[s].1 <= n.max(1) * 4)
            .min_by_key(|(_, &s)| self.slots[s].1)
            .map(|(k, _)| k);
        match best {
            Some(k) => {
                let s = self.free.swap_remove(k);
                self.refs[s] = 1;
                s
            }
            None => self.new_slot(d, n, false),
        }
    }
    fn release(&mut self, s: usize) {
        self.refs[s] -= 1;
        if self.refs[s] == 0 && !self.pinned[s] {
            self.live -= self.held[s];
            if self.reuse {
                self.free.push(s);
            }
        }
    }
}

fn is_alias(g: &Graph, id: usize) -> bool {
    let n = &g.nodes[id];
    match &n.op {
        Op::Reshape(_) | Op::StopGradient | Op::Custom { .. } => true,
        Op::Cast(d) => *d == g.nodes[n.inputs[0] as usize].dtype,
        _ => false,
    }
}

/// GPU kernels can bind about 31 buffers; a group stays below that with room for outputs.
const MAX_GROUP_BINDINGS: usize = 28;
const LEVEL_ORDER_MAX_PEAK: usize = 2 << 30;

fn arg_kind(s: &[usize], shape: &[usize]) -> Option<ArgKind> {
    let n: usize = s.iter().product();
    if s == shape {
        Some(ArgKind::Full)
    } else if n == 1 {
        Some(ArgKind::Splat)
    } else if s.last().is_some() && s.last() == shape.last() && n == *s.last().unwrap() {
        Some(ArgKind::Row(n))
    } else {
        None
    }
}

/// Rough per-element cost in ns, used only to decide when threads pay off.
fn op_cost_ns(op: &Op) -> f64 {
    use Op::*;
    match op {
        Exp | Log | Sin | Cos | Tanh => 0.6,
        Pow => 10.0,
        Div | Sqrt => 0.2,
        _ => 0.06,
    }
}

fn fusable(g: &Graph, id: usize, metal: bool) -> bool {
    let n = &g.nodes[id];
    let elementwise = n.op.is_unary_elementwise() || n.op.is_binary_elementwise() || matches!(n.op, Op::Where);
    elementwise && n.dtype != DType::I32 && (metal || n.inputs.iter().all(|&x| arg_kind(g.shape(x), &n.shape).is_some()))
}

enum Item {
    Node(usize),
    Group(Vec<usize>),
}

/// Work the GPU backend leaves on the CPU that is also cheap there: float64, small integer
/// bookkeeping such as splitting keys, and normal sampling. Scheduled first, so reaching it never
/// makes a step wait for the GPU.
pub(crate) fn metal_on_cpu(op: &Op, dtype: DType, n: usize, arg_dtypes: &[DType]) -> bool {
    let wide = dtype == DType::F64 || arg_dtypes.contains(&DType::F64);
    let small_int = dtype == DType::I32 && n <= 64 && arg_dtypes.iter().all(|&d| d == DType::I32);
    wide || small_int || matches!(op, Op::RandNormal { .. } | Op::KeySplit { .. })
}

/// Nodes the GPU plan can run on the CPU before any GPU work starts.
fn hoisted(g: &Graph, keep: &[bool]) -> Vec<bool> {
    let mut up = vec![false; g.nodes.len()];
    for (i, n) in g.nodes.iter().enumerate() {
        if !keep[i] || matches!(n.op, Op::Input(_) | Op::Const(_)) {
            continue;
        }
        let ready = n.inputs.iter().all(|&x| up[x as usize] || matches!(g.nodes[x as usize].op, Op::Input(_) | Op::Const(_)));
        let dtypes: Vec<DType> = n.inputs.iter().map(|&x| g.dtype(x)).collect();
        up[i] = ready && (metal_on_cpu(&n.op, n.dtype, n.n(), &dtypes) || (is_alias(g, i) && n.inputs.iter().all(|&x| up[x as usize])));
    }
    up
}

/// Greedy fusion for the CPU: one open group, because threads make only large groups worthwhile.
/// Instructions independent of the group are emitted first; one that reads a member closes the
/// group, which is what keeps the group free of cycles.
fn schedule(g: &Graph, keep: &[bool], cfg: Fuse) -> Vec<Item> {
    if cfg.metal && cfg.levels {
        return schedule_metal(g, keep);
    }
    let mut items = vec![];
    let mut open: Vec<usize> = vec![];
    let close = |open: &mut Vec<usize>, items: &mut Vec<Item>| {
        let cost: f64 = open.iter().map(|&m| op_cost_ns(&g.nodes[m].op) * g.nodes[m].n() as f64).sum();
        if open.len() >= if cfg.metal { 2 } else { 1 } && cost >= cfg.parallel_ns {
            items.push(Item::Group(std::mem::take(open)));
        } else {
            items.extend(open.drain(..).map(Item::Node));
        }
    };
    let up = if cfg.metal { hoisted(g, keep) } else { vec![false; g.nodes.len()] };
    items.extend((0..g.nodes.len()).filter(|&i| up[i]).map(Item::Node));
    for (i, n) in g.nodes.iter().enumerate() {
        if !keep[i] || up[i] || matches!(n.op, Op::Input(_) | Op::Const(_)) {
            continue;
        }
        if fusable(g, i, cfg.metal) && !(cfg.metal && n.dtype == DType::F64) {
            let same = open.first().is_none_or(|&m| g.nodes[m].shape == n.shape && g.nodes[m].dtype == n.dtype);
            let inputs = open.iter().flat_map(|&m| &g.nodes[m].inputs).chain(&n.inputs).filter(|x| !open.contains(&(**x as usize))).collect::<std::collections::HashSet<_>>().len();
            if !same || (cfg.metal && inputs + open.len() + 1 > MAX_GROUP_BINDINGS) {
                close(&mut open, &mut items);
            }
            open.push(i);
        } else {
            if n.inputs.iter().any(|x| open.contains(&(*x as usize))) {
                close(&mut open, &mut items);
            }
            items.push(Item::Node(i));
        }
    }
    close(&mut open, &mut items);
    items
}

/// Fusion and ordering for the GPU, where the number of dispatches and barriers is the main cost.
///
/// Every node gets a dependency level, and a level holds one group per shape. A node joins the
/// level of its inputs only when all inputs on that level are fusable values of its own shape, so
/// groups on one level never depend on each other: merging cannot create a cycle, they run
/// concurrently with no barrier between them, and chains still collapse into one kernel.
fn schedule_metal(g: &Graph, keep: &[bool]) -> Vec<Item> {
    let up = hoisted(g, keep);
    let mut items: Vec<Item> = (0..g.nodes.len()).filter(|&i| up[i]).map(Item::Node).collect();
    let gpu_fusable = |i: usize| fusable(g, i, true) && g.nodes[i].dtype != DType::F64;
    let mut level = vec![0usize; g.nodes.len()];
    let mut by_level: Vec<Vec<usize>> = vec![];
    for (i, n) in g.nodes.iter().enumerate() {
        if !keep[i] || up[i] || matches!(n.op, Op::Input(_) | Op::Const(_)) {
            continue;
        }
        let deepest = n.inputs.iter().map(|&x| level[x as usize]).max().unwrap_or(0);
        let joins = gpu_fusable(i)
            && deepest > 0
            && n.inputs.iter().all(|&x| {
                let x = x as usize;
                level[x] < deepest || (gpu_fusable(x) && g.nodes[x].shape == n.shape && g.nodes[x].dtype == n.dtype)
            });
        level[i] = if is_alias(g, i) || joins { deepest } else { deepest + 1 };
        if by_level.len() <= level[i] {
            by_level.resize(level[i] + 1, vec![]);
        }
        by_level[level[i]].push(i);
    }
    for nodes in by_level {
        let mut groups: Vec<Vec<usize>> = vec![];
        let mut aliases = vec![];
        for &i in &nodes {
            let n = &g.nodes[i];
            if is_alias(g, i) {
                aliases.push(i); // reads a value of this level, so it follows the groups
            } else if gpu_fusable(i) {
                // The last group of this shape, which earlier members may feed.
                let found = groups.iter().rposition(|m| g.nodes[m[0]].shape == n.shape && g.nodes[m[0]].dtype == n.dtype);
                let fits = found.is_some_and(|k| {
                    let m = &groups[k];
                    let inputs = m.iter().flat_map(|&x| &g.nodes[x].inputs).chain(&n.inputs).filter(|x| !m.contains(&(**x as usize))).collect::<std::collections::HashSet<_>>().len();
                    inputs + m.len() + 1 <= MAX_GROUP_BINDINGS
                });
                match (found, fits) {
                    (Some(k), true) => groups[k].push(i),
                    _ => groups.push(vec![i]),
                }
            } else {
                items.push(Item::Node(i));
            }
        }
        for m in groups {
            if m.len() >= 2 {
                items.push(Item::Group(m));
            } else {
                items.extend(m.into_iter().map(Item::Node));
            }
        }
        items.extend(aliases.into_iter().map(Item::Node));
    }
    items
}

fn compile_group(g: &Graph, members: &[usize], slot_of: &[usize], out_slots: &[(usize, usize)]) -> Fused {
    let first = &g.nodes[members[0]];
    let mut inputs: Vec<(usize, ArgKind, usize)> = vec![];
    let mut ext: Vec<(u32, Operand)> = vec![];
    let mut input_shapes = vec![];
    let mut regs = 0;
    for &m in members {
        for &x in &g.nodes[m].inputs {
            if members.contains(&(x as usize)) || ext.iter().any(|e| e.0 == x) {
                continue;
            }
            let kind = arg_kind(g.shape(x), &first.shape).unwrap_or(ArgKind::Broadcast);
            input_shapes.push(g.shape(x).to_vec());
            let operand = if kind == ArgKind::Full {
                Operand::Input(inputs.len())
            } else {
                regs += 1;
                Operand::Reg(regs - 1)
            };
            let reg = if let Operand::Reg(r) = operand { r } else { usize::MAX };
            inputs.push((slot_of[x as usize], kind, reg));
            ext.push((x, operand));
        }
    }
    let mut reg_of = vec![usize::MAX; g.nodes.len()];
    let mut ops = vec![];
    for &m in members {
        let args = g.nodes[m]
            .inputs
            .iter()
            .map(|&x| match ext.iter().find(|e| e.0 == x) {
                Some(e) => e.1,
                None => Operand::Reg(reg_of[x as usize]),
            })
            .collect();
        reg_of[m] = regs;
        ops.push(MicroOp { op: g.nodes[m].op.clone(), reg: regs, args });
        regs += 1;
    }
    let outputs = out_slots.iter().map(|&(m, s)| (reg_of[m], s)).collect();
    Fused { dtype: first.dtype, n: first.n(), inputs, ops, outputs, regs, shape: first.shape.clone(), input_shapes }
}

impl Plan {
    /// Linear-scan liveness. `reuse = false` gives every value its own buffer, which keeps
    /// inputs intact for diagnostics and serves as the reference for the optimized plan.
    pub fn build(g: &Graph, outputs: &[u32], reuse: bool) -> Plan {
        Plan::build_with(g, outputs, reuse, None)
    }

    /// The production plan: buffer reuse plus threaded elementwise groups, or GPU kernels.
    pub fn build_fused(g: &Graph, outputs: &[u32], cfg: Fuse) -> Plan {
        let plan = Plan::build_with(g, outputs, true, Some(cfg));
        if cfg.metal && cfg.levels && plan.peak_live_bytes > LEVEL_ORDER_MAX_PEAK {
            // Level order keeps values alive longer: 5 GB became 13 GB on a large model.
            return Plan::build_with(g, outputs, true, Some(Fuse { levels: false, ..cfg }));
        }
        plan
    }

    pub fn instrs(&self) -> Vec<&Instr> {
        self.steps.iter().filter_map(|s| if let Step::One(i) = s { Some(i) } else { None }).collect()
    }

    pub fn fused(&self) -> Vec<&Fused> {
        self.steps.iter().filter_map(|s| if let Step::Fused(f) = s { Some(f) } else { None }).collect()
    }

    fn build_with(g: &Graph, outputs: &[u32], reuse: bool, fuse: Option<Fuse>) -> Plan {
        let keep = g.reachable(outputs);
        let total = g.nodes.len();
        let mut uses = vec![0u32; total];
        for (i, n) in g.nodes.iter().enumerate() {
            if keep[i] {
                for &x in &n.inputs {
                    uses[x as usize] += 1;
                }
            }
        }
        for &o in outputs {
            uses[o as usize] += 1; // outputs stay live past the last instruction
        }

        let mut a = Allocator { slots: vec![], pinned: vec![], refs: vec![], free: vec![], reuse, held: vec![], live: 0, peak: 0 };
        let mut value_slot = vec![usize::MAX; total];
        let mut steps = vec![];
        let mut consts = vec![];
        let mut input_slots = vec![0; g.n_inputs];
        let mut input_ns = vec![0; g.n_inputs];
        let mut naive_bytes = 0;

        for (i, node) in g.nodes.iter().enumerate() {
            match &node.op {
                Op::Input(k) => {
                    let s = a.new_slot(node.dtype, node.n(), true);
                    input_slots[*k] = s;
                    input_ns[*k] = node.n();
                    value_slot[i] = s;
                }
                Op::Const(t) if keep[i] => {
                    let s = a.new_slot(node.dtype, node.n(), true);
                    consts.push((s, t.clone()));
                    value_slot[i] = s;
                }
                _ => {}
            }
        }

        let items = match fuse {
            Some(cfg) => schedule(g, &keep, cfg),
            None => (0..total).filter(|&i| keep[i] && !matches!(g.nodes[i].op, Op::Input(_) | Op::Const(_))).map(Item::Node).collect(),
        };

        for item in items {
            let i = match item {
                Item::Node(i) => i,
                Item::Group(members) => {
                    let mut external: Vec<u32> = vec![];
                    for &m in &members {
                        naive_bytes += g.nodes[m].n() * g.nodes[m].dtype.size();
                        for &x in &g.nodes[m].inputs {
                            uses[x as usize] -= 1;
                            if !members.contains(&(x as usize)) && !external.contains(&x) {
                                external.push(x);
                            }
                        }
                    }
                    // Destinations come first so they can never share a buffer with an input.
                    let mut out_slots = vec![];
                    for &m in &members {
                        if uses[m] > 0 {
                            let s = a.alloc(g.nodes[m].dtype, g.nodes[m].n());
                            value_slot[m] = s;
                            out_slots.push((m, s));
                        }
                    }
                    steps.push(Step::Fused(compile_group(g, &members, &value_slot, &out_slots)));
                    for &x in &external {
                        if uses[x as usize] == 0 {
                            a.release(value_slot[x as usize]);
                        }
                    }
                    continue;
                }
            };
            let node = &g.nodes[i];
            if matches!(node.op, Op::Output { .. }) {
                continue; // its slot was allocated with the control-flow step
            }
            if is_alias(g, i) {
                let x = node.inputs[0] as usize;
                let s = value_slot[x];
                a.refs[s] += 1;
                value_slot[i] = s;
                uses[x] -= 1;
                if uses[x] == 0 {
                    a.release(s);
                }
                continue;
            }

            let arg_shapes: Vec<Vec<usize>> = node.inputs.iter().map(|&x| g.shape(x).to_vec()).collect();
            let args: Vec<usize> = node.inputs.iter().map(|&x| value_slot[x as usize]).collect();
            let mode = binary_mode(&node.op, &node.shape, &arg_shapes);
            for &x in &node.inputs {
                uses[x as usize] -= 1;
            }
            let mut dying: Vec<u32> = node.inputs.iter().copied().filter(|&x| uses[x as usize] == 0).collect();
            dying.sort_unstable();
            dying.dedup();

            // A dying same-shape argument of an elementwise op can be overwritten in place.
            let elementwise = node.op.is_unary_elementwise() || node.op.is_binary_elementwise();
            let inplace = |x: u32| elementwise && mode != Mode::General && g.shape(x) == node.shape.as_slice() && g.dtype(x) == node.dtype;
            for &x in dying.iter().filter(|&&x| inplace(x)) {
                a.release(value_slot[x as usize]);
            }
            let mut outs = vec![];
            let dst = if node.op.is_control_flow() {
                for (o, out) in g.nodes.iter().enumerate() {
                    if let (true, Op::Output { index, .. }) = (keep[o] && out.inputs.first() == Some(&(i as u32)), &out.op) {
                        let slot = a.alloc(out.dtype, out.n());
                        value_slot[o] = slot;
                        outs.push((*index, slot));
                    }
                }
                usize::MAX
            } else {
                a.alloc(node.dtype, node.n())
            };
            for &x in dying.iter().filter(|&&x| !inplace(x)) {
                a.release(value_slot[x as usize]);
            }
            value_slot[i] = dst;
            naive_bytes += node.n() * node.dtype.size();
            steps.push(Step::One(Instr { op: node.op.clone(), node: i as u32, dtype: node.dtype, dst, args, shape: node.shape.clone(), arg_shapes, n: node.n(), mode, outs }));
        }

        let outputs = outputs.iter().map(|&o| Output { slot: value_slot[o as usize], shape: g.shape(o).to_vec(), n: g.nodes[o as usize].n() }).collect();
        Plan { reference: !reuse, steps, slots: a.slots, pinned: a.pinned, consts, input_slots, input_ns, outputs, naive_bytes, peak_live_bytes: a.peak }
    }

    pub fn planned_bytes(&self) -> usize {
        self.slots.iter().zip(&self.pinned).filter(|(_, &p)| !p).map(|((d, n), _)| d.size() * n).sum()
    }

    pub fn frame(&self) -> Frame {
        let mut slots: Vec<Buf> = self.slots.iter().map(|&(d, n)| Buf::zeros(d, n)).collect();
        for (s, t) in &self.consts {
            slots[*s].copy_prefix_from(&t.data, t.n());
        }
        Frame { slots }
    }

    pub fn load_input(&self, f: &mut Frame, k: usize, t: &Tensor) {
        f.slots[self.input_slots[k]].copy_prefix_from(&t.data, self.input_ns[k]);
    }

    pub fn read_output(&self, f: &Frame, k: usize) -> Tensor {
        let o = &self.outputs[k];
        Tensor { shape: o.shape.clone(), data: f.slots[o.slot].prefix(o.n) }
    }

    pub fn exec<O: Observer>(&self, f: &mut Frame, obs: &mut O) {
        for s in &self.steps {
            self.exec_step(s, f, obs);
        }
    }

    /// Seconds per step, for `Compiled::profile`. A separate loop, so `exec` carries no timers.
    pub fn exec_timed(&self, f: &mut Frame, seconds: &mut [f64]) {
        for (s, t) in self.steps.iter().zip(seconds) {
            let t0 = std::time::Instant::now();
            self.exec_step(s, f, &mut ());
            *t += t0.elapsed().as_secs_f64();
        }
    }

    #[inline(always)]
    fn exec_step<O: Observer>(&self, s: &Step, f: &mut Frame, obs: &mut O) {
        {
            match s {
                Step::One(i) if i.op.is_control_flow() => run_control(i, &mut f.slots, obs, self.reference),
                Step::One(i) => {
                    exec_instr(i, &mut f.slots);
                    obs.after(i, &f.slots);
                }
                Step::Fused(g) => match g.dtype {
                    DType::F32 => run_fused::<f32>(g, &mut f.slots),
                    DType::F64 => run_fused::<f64>(g, &mut f.slots),
                    DType::I32 => unreachable!("integer ops are never fused"),
                },
            }
        }
    }
}

/// Runs one instruction on CPU buffers. `Metal` uses this for what it has no kernel for.
#[inline(always)]
pub fn exec_instr(i: &Instr, s: &mut [Buf]) {
    match (i.dtype, &i.op) {
        (_, op) if op.is_control_flow() => run_control(i, s, &mut (), false),
        (_, Op::Cast(_)) => cast(i, s),
        (_, Op::ArgTopK { k }) => match s[i.args[0]].dtype() {
            DType::F32 => arg_top_k::<f32>(i, s, *k),
            DType::F64 => arg_top_k::<f64>(i, s, *k),
            DType::I32 => unreachable!("arg_top_k is float-only"),
        },
        (_, Op::KeySplit { .. }) => key_split(i, s),
        (_, Op::Argmax) => match s[i.args[0]].dtype() {
            DType::F32 => argmax::<f32>(i, s),
            DType::F64 => argmax::<f64>(i, s),
            DType::I32 => unreachable!("argmax is float-only"),
        },
        (DType::F32, _) => step::<f32>(i, s),
        (DType::F64, _) => step::<f64>(i, s),
        (DType::I32, _) => step_int(i, s),
    }
}

fn body_plan(body: &Body, reference: bool) -> &Plan {
    if reference {
        &body.reference
    } else {
        &body.fast
    }
}

fn load_inputs(plan: &Plan, f: &mut Frame, s: &[Buf], args: &[usize]) {
    for (k, &arg) in args.iter().enumerate() {
        f.slots[plan.input_slots[k]].copy_prefix_from(&s[arg], plan.input_ns[k]);
    }
}

/// Copies body outputs into the carry inputs. Goes through copies because an output may be
/// another carry's input.
fn feed_carry(plan: &Plan, f: &mut Frame, n_carry: usize) {
    let next: Vec<Buf> = plan.outputs[..n_carry].iter().map(|o| f.slots[o.slot].prefix(o.n)).collect();
    for (k, b) in next.iter().enumerate() {
        f.slots[plan.input_slots[k]].copy_prefix_from(b, b.len());
    }
}

fn run_control<O: Observer>(i: &Instr, s: &mut [Buf], obs: &mut O, reference: bool) {
    match &i.op {
        Op::Scan { body, n_carry, n_x, length, reverse } => {
            let plan = body_plan(body, reference);
            let (c, x) = (*n_carry, *n_x);
            let n_y = plan.outputs.len() - c;
            let mut f = plan.frame();
            // Carries and captures once; the xs change every step.
            for (k, &arg) in i.args.iter().enumerate() {
                if k < c || k >= c + x {
                    f.slots[plan.input_slots[k]].copy_prefix_from(&s[arg], plan.input_ns[k]);
                }
            }
            for step in 0..*length {
                let at = if *reverse { *length - 1 - step } else { step };
                for k in c..c + x {
                    let m = plan.input_ns[k];
                    f.slots[plan.input_slots[k]].copy_range_from(0, &s[i.args[k]], at * m, m);
                }
                for &(index, slot) in &i.outs {
                    if index >= c + n_y {
                        let k = index - c - n_y;
                        let m = plan.input_ns[k];
                        s[slot].copy_range_from(at * m, &f.slots[plan.input_slots[k]], 0, m);
                    }
                }
                obs.enter_body(body, at);
                plan.exec(&mut f, obs);
                obs.leave_body();
                for &(index, slot) in &i.outs {
                    if (c..c + n_y).contains(&index) {
                        let o = &plan.outputs[index];
                        s[slot].copy_range_from(at * o.n, &f.slots[o.slot], 0, o.n);
                    }
                }
                feed_carry(plan, &mut f, c);
            }
            for &(index, slot) in &i.outs {
                if index < c {
                    s[slot].copy_prefix_from(&f.slots[plan.input_slots[index]], plan.input_ns[index]);
                }
            }
        }
        Op::While { cond, body, n_carry, max_iterations } => {
            let (cp, bp) = (body_plan(cond, reference), body_plan(body, reference));
            let (mut cf, mut bf) = (cp.frame(), bp.frame());
            load_inputs(cp, &mut cf, s, &i.args);
            load_inputs(bp, &mut bf, s, &i.args);
            let mut count = 0;
            loop {
                obs.enter_body(cond, count);
                cp.exec(&mut cf, obs);
                obs.leave_body();
                let o = &cp.outputs[0];
                if cf.slots[o.slot].prefix(1).to_f64()[0] == 0.0 {
                    break;
                }
                if count == *max_iterations {
                    obs.while_limit(i);
                    break;
                }
                obs.enter_body(body, count);
                bp.exec(&mut bf, obs);
                obs.leave_body();
                feed_carry(bp, &mut bf, *n_carry);
                for k in 0..*n_carry {
                    cf.slots[cp.input_slots[k]].copy_prefix_from(&bf.slots[bp.input_slots[k]], bp.input_ns[k]);
                }
                count += 1;
            }
            for &(index, slot) in &i.outs {
                if index < *n_carry {
                    s[slot].copy_prefix_from(&bf.slots[bp.input_slots[index]], bp.input_ns[index]);
                } else {
                    i32::slice_mut(&mut s[slot])[0] = count as i32;
                }
            }
        }
        Op::Cond { then_body, else_body } => {
            let taken = s[i.args[0]].prefix(1).to_f64()[0] != 0.0;
            let body = if taken { then_body } else { else_body };
            let plan = body_plan(body, reference);
            let mut f = plan.frame();
            load_inputs(plan, &mut f, s, &i.args[1..]);
            obs.enter_body(body, 0);
            plan.exec(&mut f, obs);
            obs.leave_body();
            for &(index, slot) in &i.outs {
                let o = &plan.outputs[index];
                s[slot].copy_prefix_from(&f.slots[o.slot], o.n);
            }
        }
        other => unreachable!("{} is not control flow", other.name()),
    }
}

fn arg_top_k<T: Elem>(i: &Instr, s: &mut [Buf], k: usize) {
    let v = *i.arg_shapes[0].last().expect("checked by infer");
    let mut db = take(s, i.dst);
    let a = T::slice(&s[i.args[0]]);
    let n_in: usize = i.arg_shapes[0].iter().product();
    // A total order: larger first, ties by index, NaN last.
    let rank = |row: &[T], x: i32, y: i32| -> std::cmp::Ordering {
        let (p, q) = (row[x as usize], row[y as usize]);
        match q.partial_cmp(&p) {
            Some(std::cmp::Ordering::Equal) => x.cmp(&y),
            Some(order) => order,
            None => (p != p).cmp(&(q != q)).then(x.cmp(&y)),
        }
    };
    let select = |(out, row): (&mut [i32], &[T])| {
        let mut idx: Vec<i32> = (0..v as i32).collect();
        if k < v {
            idx.select_nth_unstable_by(k - 1, |&x, &y| rank(row, x, y));
        }
        idx[..k].sort_unstable_by(|&x, &y| rank(row, x, y));
        out.copy_from_slice(&idx[..k]);
    };
    let dst = &mut i32::slice_mut(&mut db)[..i.n];
    if n_in >= PARALLEL_MINMAX {
        dst.par_chunks_mut(k).zip(a[..n_in].par_chunks_exact(v)).for_each(select);
    } else {
        dst.chunks_mut(k).zip(a[..n_in].chunks_exact(v)).for_each(select);
    }
    s[i.dst] = db;
}

fn cumsum<T: Storage + std::ops::Add<Output = T>>(i: &Instr, s: &[Buf], reverse: bool, dst: &mut [T]) {
    let v = *i.shape.last().expect("checked by infer");
    let a = T::slice(&s[i.args[0]]);
    if v == 0 {
        return;
    }
    for (o, r) in dst.chunks_exact_mut(v).zip(a[..i.n].chunks_exact(v)) {
        let mut acc = T::ZERO;
        if reverse {
            for j in (0..v).rev() {
                acc = acc + r[j];
                o[j] = acc;
            }
        } else {
            for j in 0..v {
                acc = acc + r[j];
                o[j] = acc;
            }
        }
    }
}

struct SendPtr<T>(*mut T);
// SAFETY: tasks write disjoint index ranges, see `run_fused`.
unsafe impl<T> Sync for SendPtr<T> {}
unsafe impl<T> Send for SendPtr<T> {}

fn run_fused<T: Elem>(f: &Fused, s: &mut [Buf]) {
    let mut outs: Vec<Buf> = f.outputs.iter().map(|&(_, slot)| take(s, slot)).collect();
    let ptrs: Vec<SendPtr<T>> = outs
        .iter_mut()
        .map(|b| {
            let v = T::slice_mut(b);
            assert!(v.len() >= f.n);
            SendPtr(v.as_mut_ptr())
        })
        .collect();
    let inputs: Vec<&[T]> = f.inputs.iter().map(|&(slot, _, _)| T::slice(&s[slot])).collect();
    let task = |temp: &mut Vec<T>, start: usize| {
        let end = (start + TASK).min(f.n);
        for b in (start..end).step_by(BLOCK) {
            let len = BLOCK.min(end - b);
            run_block(f, &inputs, temp, b, len);
            for (k, &(reg, _)) in f.outputs.iter().enumerate() {
                let src = &temp[reg * BLOCK..reg * BLOCK + len];
                // SAFETY: capacity asserted above; each task owns [start, end) of every output,
                // and destinations are never inputs (planner invariant).
                unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), ptrs[k].0.add(b), len) };
            }
        }
    };
    (0..f.n.div_ceil(TASK)).into_par_iter().for_each_init(|| vec![T::ZERO; f.regs * BLOCK], |t, c| task(t, c * TASK));
    for (b, &(_, slot)) in outs.into_iter().zip(&f.outputs) {
        s[slot] = b;
    }
}

/// Registers only read lower registers, so `split_at_mut` gives safe access without aliasing.
fn run_block<T: Elem>(f: &Fused, inputs: &[&[T]], temp: &mut [T], start: usize, len: usize) {
    for (k, &(_, kind, reg)) in f.inputs.iter().enumerate() {
        match kind {
            ArgKind::Full | ArgKind::Broadcast => {}
            ArgKind::Splat => temp[reg * BLOCK..reg * BLOCK + len].fill(inputs[k][0]),
            ArgKind::Row(l) => {
                let row = &inputs[k][..l];
                for (o, j) in temp[reg * BLOCK..reg * BLOCK + len].iter_mut().zip(start..) {
                    *o = row[j % l];
                }
            }
        }
    }
    for m in &f.ops {
        let (lo, hi) = temp.split_at_mut(m.reg * BLOCK);
        let d = &mut hi[..len];
        let arg = |i: usize| match m.args[i] {
            Operand::Input(k) => &inputs[k][start..start + len],
            Operand::Reg(r) => &lo[r * BLOCK..r * BLOCK + len],
        };
        use Op::*;
        match &m.op {
            Neg => map1(d, arg(0), |x| -x),
            Exp => T::vmath(VFn::Exp, arg(0), d),
            Log => T::vmath(VFn::Log, arg(0), d),
            Sin => T::vmath(VFn::Sin, arg(0), d),
            Cos => T::vmath(VFn::Cos, arg(0), d),
            Tanh => T::vmath(VFn::Tanh, arg(0), d),
            Sqrt => map1(d, arg(0), T::sqrt),
            Abs => map1(d, arg(0), T::abs),
            Sign => map1(d, arg(0), sign),
            Floor => map1(d, arg(0), T::floor),
            Add => map2(d, arg(0), arg(1), |a, b| a + b),
            Sub => map2(d, arg(0), arg(1), |a, b| a - b),
            Mul => map2(d, arg(0), arg(1), |a, b| a * b),
            Div => map2(d, arg(0), arg(1), |a, b| a / b),
            Pow => map2(d, arg(0), arg(1), T::powf),
            Max => map2(d, arg(0), arg(1), nan_max),
            Min => map2(d, arg(0), arg(1), nan_min),
            Gt => map2(d, arg(0), arg(1), |a, b| if a > b { T::ONE } else { T::ZERO }),
            Ge => map2(d, arg(0), arg(1), |a, b| if a >= b { T::ONE } else { T::ZERO }),
            Where => {
                select(d, arg(0), arg(1), arg(2));
            }
            other => unreachable!("{} is not fusable", other.name()),
        }
    }
}

#[inline(always)]
fn select<T: Storage>(d: &mut [T], c: &[T], x: &[T], y: &[T]) {
    for ((o, &k), (&p, &q)) in d.iter_mut().zip(c).zip(x.iter().zip(y)) {
        *o = if k != T::ZERO { p } else { q };
    }
}

#[inline(always)]
fn map2<T: Storage>(d: &mut [T], a: &[T], b: &[T], f: impl Fn(T, T) -> T) {
    for ((x, &p), &q) in d.iter_mut().zip(a).zip(b) {
        *x = f(p, q);
    }
}

#[inline(always)]
fn sign<T: Elem>(x: T) -> T {
    if x > T::ZERO {
        T::ONE
    } else if x < T::ZERO {
        -T::ONE
    } else {
        x
    }
}

/// Runs a graph once with a fresh frame.
pub fn eval(g: &Graph, outputs: &[u32], inputs: &[&Tensor]) -> Vec<Tensor> {
    let plan = Plan::build(g, outputs, true);
    let mut f = plan.frame();
    for (k, t) in inputs.iter().enumerate() {
        plan.load_input(&mut f, k, t);
    }
    plan.exec(&mut f, &mut ());
    (0..outputs.len()).map(|k| plan.read_output(&f, k)).collect()
}

fn binary_mode(op: &Op, shape: &[usize], args: &[Vec<usize>]) -> Mode {
    if !op.is_binary_elementwise() {
        return Mode::Same;
    }
    let (a, b) = (&args[0], &args[1]);
    let one = |s: &[usize]| s.iter().product::<usize>() == 1;
    if a == shape && b == shape {
        Mode::Same
    } else if one(a) && b == shape {
        Mode::AScalar
    } else if a == shape && one(b) {
        Mode::BScalar
    } else {
        Mode::General
    }
}

fn take(s: &mut [Buf], d: usize) -> Buf {
    std::mem::replace(&mut s[d], Buf::F32(Vec::new()))
}

#[inline(always)]
fn map1<T: Storage>(dst: &mut [T], src: &[T], f: impl Fn(T) -> T) {
    for (x, &a) in dst.iter_mut().zip(src) {
        *x = f(a);
    }
}

#[inline(always)]
fn map1_inplace<T: Storage>(dst: &mut [T], f: impl Fn(T) -> T) {
    for x in dst {
        *x = f(*x);
    }
}

#[inline(always)]
fn unary<T: Storage>(i: &Instr, s: &mut [Buf], f: impl Fn(T) -> T) {
    let (a, d, n) = (i.args[0], i.dst, i.n);
    if a == d {
        map1_inplace(&mut T::slice_mut(&mut s[d])[..n], f);
    } else {
        let mut db = take(s, d);
        map1(&mut T::slice_mut(&mut db)[..n], &T::slice(&s[a])[..n], f);
        s[d] = db;
    }
}

/// Vector math routines promise nothing about overlap, so in-place runs go through a stack block.
fn unary_v<T: Elem>(i: &Instr, s: &mut [Buf], f: VFn) {
    let (a, d, n) = (i.args[0], i.dst, i.n);
    if a == d {
        let mut tmp = [T::ZERO; 512];
        for chunk in T::slice_mut(&mut s[d])[..n].chunks_mut(512) {
            let x = &mut tmp[..chunk.len()];
            x.copy_from_slice(chunk);
            T::vmath(f, x, chunk);
        }
    } else {
        let mut db = take(s, d);
        T::vmath(f, &T::slice(&s[a])[..n], &mut T::slice_mut(&mut db)[..n]);
        s[d] = db;
    }
}

#[inline(always)]
fn binary<T: Storage>(i: &Instr, s: &mut [Buf], f: impl Fn(T, T) -> T + Copy) {
    let (a, b, d, n) = (i.args[0], i.args[1], i.dst, i.n);
    let mut db = take(s, d);
    let dst = &mut T::slice_mut(&mut db)[..n];
    match i.mode {
        Mode::Same => match (a == d, b == d) {
            (true, true) => map1_inplace(dst, |x| f(x, x)),
            (true, false) => {
                for (x, &y) in dst.iter_mut().zip(&T::slice(&s[b])[..n]) {
                    *x = f(*x, y);
                }
            }
            (false, true) => {
                for (x, &y) in dst.iter_mut().zip(&T::slice(&s[a])[..n]) {
                    *x = f(y, *x);
                }
            }
            (false, false) => {
                let (av, bv) = (&T::slice(&s[a])[..n], &T::slice(&s[b])[..n]);
                for ((x, &p), &q) in dst.iter_mut().zip(av).zip(bv) {
                    *x = f(p, q);
                }
            }
        },
        Mode::AScalar => {
            let c = T::slice(&s[a])[0];
            if b == d {
                map1_inplace(dst, |y| f(c, y));
            } else {
                map1(dst, &T::slice(&s[b])[..n], |y| f(c, y));
            }
        }
        Mode::BScalar => {
            let c = T::slice(&s[b])[0];
            if a == d {
                map1_inplace(dst, |x| f(x, c));
            } else {
                map1(dst, &T::slice(&s[a])[..n], |x| f(x, c));
            }
        }
        Mode::General => {
            let (av, bv) = (T::slice(&s[a]), T::slice(&s[b]));
            let ds = strides(&i.shape);
            let (sa, sb) = (bstrides(&i.arg_shapes[0], i.shape.len()), bstrides(&i.arg_shapes[1], i.shape.len()));
            let (outer, len) = split_rows(&i.shape);
            let (ra, rb) = (i.shape.last().map_or(0, |_| sa[sa.len() - 1]), i.shape.last().map_or(0, |_| sb[sb.len() - 1]));
            walk(outer, [&ds, &sa, &sb], |[o, p, q]| {
                let row = &mut dst[o..o + len];
                match (ra, rb) {
                    (0, 0) => row.fill(f(av[p], bv[q])),
                    (0, _) => {
                        let c = av[p];
                        map1(row, &bv[q..q + len], |y| f(c, y));
                    }
                    (_, 0) => {
                        let c = bv[q];
                        map1(row, &av[p..p + len], |x| f(x, c));
                    }
                    _ => {
                        for ((x, &m), &r) in row.iter_mut().zip(&av[p..p + len]).zip(&bv[q..q + len]) {
                            *x = f(m, r);
                        }
                    }
                }
            });
        }
    }
    s[d] = db;
}

/// Splits a shape into the dims the odometer walks and the row length the inner loop vectorizes.
fn split_rows(shape: &[usize]) -> (&[usize], usize) {
    match shape.split_last() {
        Some((&len, outer)) => (outer, len),
        None => (shape, 1),
    }
}

pub(crate) fn strides(shape: &[usize]) -> Vec<usize> {
    let mut out = vec![1; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        out[i] = out[i + 1] * shape[i + 1];
    }
    out
}

/// Strides of `shape` right-aligned into `rank` dims, with 0 on broadcast dims.
pub(crate) fn bstrides(shape: &[usize], rank: usize) -> Vec<usize> {
    let st = strides(shape);
    let lead = rank - shape.len();
    (0..rank).map(|i| if i < lead || shape[i - lead] == 1 { 0 } else { st[i - lead] }).collect()
}

/// Odometer over `shape`, tracking N flat offsets.
#[inline(always)]
fn walk<const N: usize>(shape: &[usize], st: [&[usize]; N], mut f: impl FnMut([usize; N])) {
    let total: usize = shape.iter().product();
    let mut idx = vec![0usize; shape.len()];
    let mut off = [0usize; N];
    for _ in 0..total {
        f(off);
        for d in (0..shape.len()).rev() {
            idx[d] += 1;
            for k in 0..N {
                off[k] += st[k][d];
            }
            if idx[d] < shape[d] {
                break;
            }
            for k in 0..N {
                off[k] -= st[k][d] * shape[d];
            }
            idx[d] = 0;
        }
    }
}

#[inline(always)]
fn nan_max<T: Elem>(a: T, b: T) -> T {
    if a > b {
        a
    } else if b >= a {
        b
    } else {
        T::NAN
    }
}

#[inline(always)]
fn nan_min<T: Elem>(a: T, b: T) -> T {
    if a < b {
        a
    } else if b <= a {
        b
    } else {
        T::NAN
    }
}

/// Lane-wise accumulation lets LLVM vectorize a float sum it must otherwise keep in order.
fn fast_sum<T: Elem>(a: &[T]) -> T {
    let mut acc = [T::ZERO; 8];
    let chunks = a.chunks_exact(8);
    let rest = chunks.remainder();
    for c in chunks {
        for k in 0..8 {
            acc[k] = acc[k] + c[k];
        }
    }
    let mut out = T::ZERO;
    for x in acc {
        out = out + x;
    }
    for &x in rest {
        out = out + x;
    }
    out
}

#[inline(always)]
fn step<T: Elem>(i: &Instr, s: &mut [Buf]) {
    use Op::*;
    match &i.op {
        Neg => unary::<T>(i, s, |x| -x),
        Exp => unary_v::<T>(i, s, VFn::Exp),
        Log => unary_v::<T>(i, s, VFn::Log),
        Sqrt => unary::<T>(i, s, T::sqrt),
        Sin => unary_v::<T>(i, s, VFn::Sin),
        Cos => unary_v::<T>(i, s, VFn::Cos),
        Tanh => unary_v::<T>(i, s, VFn::Tanh),
        Abs => unary::<T>(i, s, T::abs),
        Sign => unary::<T>(i, s, sign),
        Add => binary::<T>(i, s, |a, b| a + b),
        Sub => binary::<T>(i, s, |a, b| a - b),
        Mul => binary::<T>(i, s, |a, b| a * b),
        Div => binary::<T>(i, s, |a, b| a / b),
        Pow => binary::<T>(i, s, T::powf),
        Max => binary::<T>(i, s, nan_max),
        Min => binary::<T>(i, s, nan_min),
        Gt => binary::<T>(i, s, |a, b| if a > b { T::ONE } else { T::ZERO }),
        Ge => binary::<T>(i, s, |a, b| if a >= b { T::ONE } else { T::ZERO }),
        Floor => unary::<T>(i, s, T::floor),
        Cumsum { reverse } => {
            let mut db = take(s, i.dst);
            cumsum::<T>(i, s, *reverse, &mut T::slice_mut(&mut db)[..i.n]);
            s[i.dst] = db;
        }
        Where | Transpose(_) | BroadcastTo(_) | Slice { .. } | Pad { .. } | Concat { .. } | Take { .. } | TakeAlong => with_dst::<T>(i, s, structural::<T>),
        _ => with_dst::<T>(i, s, numeric::<T>),
    }
}

/// The planner never aliases dst with an argument for these ops.
fn with_dst<T: Storage>(i: &Instr, s: &mut [Buf], f: fn(&Instr, &[Buf], &mut [T])) {
    let mut db = take(s, i.dst);
    f(i, s, &mut T::slice_mut(&mut db)[..i.n]);
    s[i.dst] = db;
}

fn step_int(i: &Instr, s: &mut [Buf]) {
    use Op::*;
    match &i.op {
        Add => binary::<i32>(i, s, i32::wrapping_add),
        Sub => binary::<i32>(i, s, i32::wrapping_sub),
        Mul => binary::<i32>(i, s, i32::wrapping_mul),
        Gt => binary::<i32>(i, s, |a, b| (a > b) as i32),
        Ge => binary::<i32>(i, s, |a, b| (a >= b) as i32),
        Cumsum { reverse } => {
            let mut db = take(s, i.dst);
            cumsum::<i32>(i, s, *reverse, &mut i32::slice_mut(&mut db)[..i.n]);
            s[i.dst] = db;
        }
        _ => with_dst::<i32>(i, s, structural::<i32>),
    }
}

/// Copying kernels, shared by every dtype.
fn structural<T: Storage>(i: &Instr, s: &[Buf], dst: &mut [T]) {
    use Op::*;
    let a = T::slice(&s[i.args[0]]);
    let ds = strides(&i.shape);
    match &i.op {
        Where => {
            let (x, y) = (T::slice(&s[i.args[1]]), T::slice(&s[i.args[2]]));
            select(dst, &a[..i.n], &x[..i.n], &y[..i.n]);
        }
        Transpose(perm) => {
            let src = strides(&i.arg_shapes[0]);
            let mapped: Vec<usize> = perm.iter().map(|&p| src[p]).collect();
            let (outer, len) = split_rows(&i.shape);
            let last = mapped.last().copied().unwrap_or(1);
            let copy_row = |row: &mut [T], p: usize| {
                if last == 1 {
                    row.copy_from_slice(&a[p..p + len]);
                } else {
                    for (j, x) in row.iter_mut().enumerate() {
                        *x = a[p + j * last];
                    }
                }
            };
            if i.n >= PARALLEL_COPY && len > 0 {
                let mut offsets = Vec::with_capacity(i.n / len);
                walk(outer, [&mapped], |[p]| offsets.push(p));
                dst.par_chunks_mut(len).zip(offsets.par_iter()).for_each(|(row, &p)| copy_row(row, p));
            } else {
                walk(outer, [&ds, &mapped], |[o, p]| copy_row(&mut dst[o..o + len], p));
            }
        }
        BroadcastTo(_) => {
            let src = bstrides(&i.arg_shapes[0], i.shape.len());
            let (outer, len) = split_rows(&i.shape);
            let last = src.last().copied().unwrap_or(0);
            walk(outer, [&ds, &src], |[o, p]| {
                if last == 0 {
                    dst[o..o + len].fill(a[p]);
                } else {
                    dst[o..o + len].copy_from_slice(&a[p..p + len]);
                }
            });
        }
        Slice { starts, .. } => {
            let src = strides(&i.arg_shapes[0]);
            let base: usize = starts.iter().zip(&src).map(|(x, y)| x * y).sum();
            let (outer, len) = split_rows(&i.shape);
            walk(outer, [&ds, &src], |[o, p]| dst[o..o + len].copy_from_slice(&a[base + p..base + p + len]));
        }
        Pad { starts, .. } => {
            dst.fill(T::ZERO);
            let src = strides(&i.arg_shapes[0]);
            let base: usize = starts.iter().zip(&ds).map(|(x, y)| x * y).sum();
            let (outer, len) = split_rows(&i.arg_shapes[0]);
            walk(outer, [&ds, &src], |[o, p]| dst[base + o..base + o + len].copy_from_slice(&a[p..p + len]));
        }
        Concat { axis } => {
            let tail: usize = i.shape[*axis + 1..].iter().product();
            let outer: usize = i.shape[..*axis].iter().product();
            let mut at = 0;
            for o in 0..outer {
                for (k, &slot) in i.args.iter().enumerate() {
                    let len = i.arg_shapes[k][*axis] * tail;
                    dst[at..at + len].copy_from_slice(&T::slice(&s[slot])[o * len..o * len + len]);
                    at += len;
                }
            }
        }
        Take { axis } => {
            let xs = &i.arg_shapes[0];
            let block: usize = xs[*axis + 1..].iter().product();
            let outer: usize = xs[..*axis].iter().product();
            let dim = xs[*axis];
            let n_idx: usize = i.arg_shapes[1].iter().product();
            let idx = &i32::slice(&s[i.args[1]])[..n_idx];
            for o in 0..outer {
                for (j, &ix) in idx.iter().enumerate() {
                    let from = (o * dim + clamp_index(ix, dim)) * block;
                    let to = (o * n_idx + j) * block;
                    dst[to..to + block].copy_from_slice(&a[from..from + block]);
                }
            }
        }
        TakeAlong => {
            let v = *i.arg_shapes[0].last().expect("checked by infer");
            let k = *i.arg_shapes[1].last().expect("checked by infer");
            let idx = i32::slice(&s[i.args[1]]);
            for j in 0..i.n {
                dst[j] = a[(j / k) * v + clamp_index(idx[j], v)];
            }
        }
        other => unreachable!("{} has no structural kernel", other.name()),
    }
}

/// Out-of-range indices clamp, so no input can read out of bounds; validation reports them.
#[inline(always)]
fn clamp_index(ix: i32, dim: usize) -> usize {
    (ix.max(0) as usize).min(dim - 1)
}

const GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;
const GAMMA_SPLIT: u64 = 0xD1B5_4A32_D192_ED03;
const PARALLEL_RANDOM: usize = 1 << 15;
const PARALLEL_COPY: usize = 1 << 19;
/// Sums vectorize and are very cheap per element; max and min are not.
const PARALLEL_SUM: usize = 1 << 20;
const PARALLEL_MINMAX: usize = 1 << 16;
/// Rows per task when reducing leading axes. Fixed, so rounding cannot depend on thread count.
const REDUCE_ROWS: usize = 64;
/// Multiplications at which a batched matmul goes through the thread pool.
const PARALLEL_FLOPS: usize = 2_000_000;

/// SplitMix64: every element depends only on (key, index), so the stream ignores thread count.
#[inline(always)]
fn mix(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn read_key(s: &[Buf], slot: usize) -> u64 {
    let k = i32::slice(&s[slot]);
    ((k[0] as u32 as u64) << 32) | (k[1] as u32 as u64)
}

fn key_split(i: &Instr, s: &mut [Buf]) {
    let key = read_key(s, i.args[0]);
    let mut db = take(s, i.dst);
    for (j, pair) in i32::slice_mut(&mut db)[..i.n].chunks_exact_mut(2).enumerate() {
        let bits = mix(key.wrapping_add((j as u64 + 1).wrapping_mul(GAMMA_SPLIT)));
        pair[0] = (bits >> 32) as u32 as i32;
        pair[1] = bits as u32 as i32;
    }
    s[i.dst] = db;
}

fn fill_random<T: Elem>(dst: &mut [T], f: impl Fn(usize) -> T + Send + Sync) {
    if dst.len() >= PARALLEL_RANDOM {
        dst.par_iter_mut().enumerate().for_each(|(j, x)| *x = f(j));
    } else {
        for (j, x) in dst.iter_mut().enumerate() {
            *x = f(j);
        }
    }
}

fn argmax<T: Elem>(i: &Instr, s: &mut [Buf]) {
    let v = *i.arg_shapes[0].last().expect("checked by infer");
    let mut db = take(s, i.dst);
    let a = T::slice(&s[i.args[0]]);
    for (o, r) in i32::slice_mut(&mut db)[..i.n].iter_mut().zip(a.chunks_exact(v)) {
        let mut best = 0;
        for (j, &x) in r.iter().enumerate() {
            if x > r[best] {
                best = j;
            }
        }
        *o = best as i32;
    }
    s[i.dst] = db;
}

fn matmul<T: Elem>(i: &Instr, s: &[Buf], ta: bool, tb: bool, dst: &mut [T]) {
    let (sa, sb) = (&i.arg_shapes[0], &i.arg_shapes[1]);
    let (a, b) = (T::slice(&s[i.args[0]]), T::slice(&s[i.args[1]]));
    let (ar, br) = (sa.len(), sb.len());
    let (m, k) = match (ar >= 2, ta) {
        (true, false) => (sa[ar - 2], sa[ar - 1]),
        (true, true) => (sa[ar - 1], sa[ar - 2]),
        _ => (1, sa[0]),
    };
    let n = match (br >= 2, tb) {
        (true, false) => sb[br - 1],
        (true, true) => sb[br - 2],
        _ => 1,
    };
    let (ba, bb) = (&sa[..ar.saturating_sub(2)], &sb[..br.saturating_sub(2)]);
    if dst.is_empty() {
        return;
    }
    if bb.is_empty() && !ta {
        // Row-major batches of the left operand are one big matrix: a single BLAS call.
        let big: usize = ba.iter().product::<usize>() * m;
        T::gemm(big, k, n, a, false, b, tb, dst);
        return;
    }
    let batch = crate::graph::broadcast_shapes(ba, bb).expect("checked by infer");
    let (sz_a, sz_b, sz_o) = (m * k, k * n, m * n);
    let scale = |v: Vec<usize>, by: usize| -> Vec<usize> { v.into_iter().map(|x| x * by).collect() };
    let (st_o, st_a, st_b) = (scale(strides(&batch), sz_o), scale(bstrides(ba, batch.len()), sz_a), scale(bstrides(bb, batch.len()), sz_b));
    let count: usize = batch.iter().product();
    if m * k * n * count >= PARALLEL_FLOPS {
        let mut offsets = Vec::with_capacity(count);
        walk(&batch, [&st_a, &st_b], |[p, q]| offsets.push((p, q)));
        dst.par_chunks_mut(sz_o).zip(offsets.par_iter()).for_each(|(out, &(p, q))| {
            T::gemm(m, k, n, &a[p..p + sz_a], ta, &b[q..q + sz_b], tb, out);
        });
    } else {
        walk(&batch, [&st_o, &st_a, &st_b], |[o, p, q]| {
            T::gemm(m, k, n, &a[p..p + sz_a], ta, &b[q..q + sz_b], tb, &mut dst[o..o + sz_o]);
        });
    }
}

/// Arithmetic kernels that are not elementwise.
fn numeric<T: Elem>(i: &Instr, s: &[Buf], dst: &mut [T]) {
    use Op::*;
    match &i.op {
        MatMul => matmul(i, s, false, false, dst),
        MatMulT { ta, tb } => matmul(i, s, *ta, *tb, dst),
        Reduce { kind, axes, .. } => reduce(i, s, *kind, axes, dst),
        TakeGrad { axis, shape } => {
            let ct = T::slice(&s[i.args[0]]);
            let block: usize = shape[*axis + 1..].iter().product();
            let outer: usize = shape[..*axis].iter().product();
            let dim = shape[*axis];
            let n_idx: usize = i.arg_shapes[1].iter().product();
            let idx = &i32::slice(&s[i.args[1]])[..n_idx];
            dst.fill(T::ZERO);
            for o in 0..outer {
                for (j, &ix) in idx.iter().enumerate() {
                    let to = (o * dim + clamp_index(ix, dim)) * block;
                    let from = (o * n_idx + j) * block;
                    for (d, &c) in dst[to..to + block].iter_mut().zip(&ct[from..from + block]) {
                        *d = *d + c;
                    }
                }
            }
        }
        TakeAlongGrad { shape } => {
            let ct = T::slice(&s[i.args[0]]);
            let v = *shape.last().expect("checked by infer");
            let k = *i.arg_shapes[1].last().expect("checked by infer");
            let n_ct: usize = i.arg_shapes[1].iter().product();
            let idx = i32::slice(&s[i.args[1]]);
            dst.fill(T::ZERO);
            for j in 0..n_ct {
                let to = (j / k) * v + clamp_index(idx[j], v);
                dst[to] = dst[to] + ct[j];
            }
        }
        RandUniform { .. } => {
            let key = read_key(s, i.args[0]);
            let (shift, scale) = (64 - T::MANTISSA, 0.5f64.powi(T::MANTISSA as i32));
            fill_random(dst, |j| T::from_f64((mix(key.wrapping_add((j as u64 + 1).wrapping_mul(GAMMA))) >> shift) as f64 * scale));
        }
        RandNormal { .. } => {
            let key = read_key(s, i.args[0]);
            let unit = |c: u64| (mix(key.wrapping_add(c.wrapping_mul(GAMMA))) >> 11) as f64 * 0.5f64.powi(53);
            fill_random(dst, |j| {
                let (u1, u2) = (1.0 - unit(2 * j as u64 + 1), unit(2 * j as u64 + 2));
                T::from_f64((-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos())
            });
        }
        other => unreachable!("{} has no numeric kernel", other.name()),
    }
}

fn reduce<T: Elem>(i: &Instr, s: &[Buf], kind: ReduceKind, axes: &[usize], dst: &mut [T]) {
    let a = T::slice(&s[i.args[0]]);
    let in_shape = &i.arg_shapes[0];
    let in_n: usize = in_shape.iter().product();
    if kind == ReduceKind::Sum && axes.len() == in_shape.len() {
        dst[0] = fast_sum(&a[..in_n]);
        return;
    }
    let mut keep = in_shape.clone();
    for &ax in axes {
        keep[ax] = 1;
    }
    let mut out_st = strides(&keep);
    for &ax in axes {
        out_st[ax] = 0;
    }
    let in_st = strides(in_shape);
    let (outer, len) = split_rows(in_shape);
    let last_reduced = in_shape.last().is_some() && out_st[out_st.len() - 1] == 0;
    let combine = |acc: T, x: T| match kind {
        ReduceKind::Sum => acc + x,
        ReduceKind::Max => nan_max(acc, x),
        ReduceKind::Min => nan_min(acc, x),
    };
    let init = match kind {
        ReduceKind::Sum => T::ZERO,
        ReduceKind::Max => -T::INF,
        ReduceKind::Min => T::INF,
    };
    if in_n >= (if kind == ReduceKind::Sum { PARALLEL_SUM } else { PARALLEL_MINMAX }) && len > 0 {
        if axes == [in_shape.len() - 1] {
            // Each output owns one input row.
            dst.par_iter_mut().zip(a[..in_n].par_chunks_exact(len)).for_each(|(o, row)| {
                *o = if kind == ReduceKind::Sum { fast_sum(row) } else { row.iter().fold(init, |acc, &x| combine(acc, x)) };
            });
            return;
        }
        if axes.len() + 1 == in_shape.len() && !axes.contains(&(in_shape.len() - 1)) {
            // Only the last axis survives: combine partial results in a fixed order.
            let parts: Vec<Vec<T>> = a[..in_n]
                .par_chunks(REDUCE_ROWS * len)
                .map(|chunk| {
                    let mut part = vec![init; len];
                    for row in chunk.chunks_exact(len) {
                        for (o, &x) in part.iter_mut().zip(row) {
                            *o = combine(*o, x);
                        }
                    }
                    part
                })
                .collect();
            dst.fill(init);
            for part in parts {
                for (o, x) in dst.iter_mut().zip(part) {
                    *o = combine(*o, x);
                }
            }
            return;
        }
    }
    dst.fill(init);
    walk(outer, [&in_st, &out_st], |[p, q]| {
        let row = &a[p..p + len];
        if last_reduced && kind == ReduceKind::Sum {
            dst[q] = dst[q] + fast_sum(row);
        } else if last_reduced {
            dst[q] = row.iter().fold(dst[q], |acc, &x| combine(acc, x));
        } else {
            for (o, &x) in dst[q..q + len].iter_mut().zip(row) {
                *o = combine(*o, x);
            }
        }
    });
}

fn cast(i: &Instr, s: &mut [Buf]) {
    let mut db = take(s, i.dst);
    let n = i.n;
    macro_rules! conv {
        ($d:expr, $a:expr, $t:ty) => {
            $d[..n].iter_mut().zip($a).for_each(|(x, &y)| *x = y as $t)
        };
    }
    match (&mut db, &s[i.args[0]]) {
        (Buf::F32(d), Buf::F64(a)) => conv!(d, a, f32),
        (Buf::F64(d), Buf::F32(a)) => conv!(d, a, f64),
        (Buf::F32(d), Buf::I32(a)) => conv!(d, a, f32),
        (Buf::F64(d), Buf::I32(a)) => conv!(d, a, f64),
        (Buf::I32(d), Buf::F32(a)) => conv!(d, a, i32),
        (Buf::I32(d), Buf::F64(a)) => conv!(d, a, i32),
        _ => unreachable!("same-dtype casts are aliases"),
    }
    s[i.dst] = db;
}
