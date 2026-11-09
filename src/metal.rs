//! GPU backend: every plan step becomes a generated MSL kernel over shared memory.
//!
//! Shapes are static, so shapes and strides are compiled into the kernels as constants. Apple
//! Silicon's unified memory means CPU and GPU see the same buffers, which keeps fallback simple:
//! what has no kernel (float64, normal sampling) runs on the CPU after the queue drains.

use crate::graph::{broadcast_shapes, Op, ReduceKind};
use crate::plan::{bstrides, exec_instr, metal_on_cpu, strides, ArgKind, Exec, Fused, Instr, Operand, Plan, Step};
use crate::tensor::{Buf, DType, Tensor};
use objc2::rc::{autoreleasepool, Retained};
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLCompileOptions, MTLComputeCommandEncoder, MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary, MTLMathMode, MTLResourceOptions, MTLSize, MTLBarrierScope, MTLDispatchType,
};
use objc2::AnyThread;
use objc2_metal_performance_shaders::{MPSDataType, MPSMatrix, MPSMatrixDescriptor, MPSMatrixMultiplication};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::Mutex;

#[link(name = "CoreGraphics", kind = "framework")]
extern "C" {}

type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;
type Pipeline = Retained<ProtocolObject<dyn MTLComputePipelineState>>;

const PRELUDE: &str = r#"
#include <metal_stdlib>
using namespace metal;
inline float rg_pow(float x, float y) {
    if (x < 0.0f && y == floor(y)) {
        float r = pow(-x, y);
        return fmod(y, 2.0f) != 0.0f ? -r : r;
    }
    return pow(x, y);
}
inline float rg_max(float a, float b) { return a > b ? a : (b >= a ? b : NAN); }
inline float rg_min(float a, float b) { return a < b ? a : (b <= a ? b : NAN); }
// MSL's tanh(inf) is NaN; the function is exactly +-1 in float32 well before this threshold.
inline float rg_tanh(float x) { return fabs(x) > 20.0f ? copysign(1.0f, x) : tanh(x); }
// MSL's sin(NaN) and cos(NaN) are 0: silently erasing a NaN would destroy diagnostic evidence.
inline float rg_sin(float x) { return isfinite(x) ? sin(x) : NAN; }
inline float rg_cos(float x) { return isfinite(x) ? cos(x) : NAN; }
inline float rg_sign(float x) { return x > 0.0f ? 1.0f : (x < 0.0f ? -1.0f : x); }
inline ulong rg_mix(ulong z) {
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
    return z ^ (z >> 31);
}
inline ulong rg_key(device const int* k) {
    return (ulong(uint(k[0])) << 32) | ulong(uint(k[1]));
}
"#;

pub fn available() -> bool {
    MTLCreateSystemDefaultDevice().is_some()
}

fn ty(d: DType) -> &'static str {
    match d {
        DType::F32 => "float",
        DType::I32 => "int",
        DType::F64 => unreachable!("float64 steps run on the CPU"),
    }
}

/// Flat source index for output index `var`, given the source stride of each output dimension.
fn index(var: &str, shape: &[usize], st: &[usize]) -> String {
    let ds = strides(shape);
    if shape.iter().zip(st).zip(&ds).all(|((&s, a), b)| s == 1 || a == b) {
        return var.to_string();
    }
    let terms: Vec<String> = (0..shape.len()).filter(|&d| shape[d] > 1 && st[d] > 0).map(|d| format!("({}) * {}u", coord(var, ds[d], shape[d]), st[d])).collect();
    if terms.is_empty() {
        "0u".to_string()
    } else {
        terms.join(" + ")
    }
}

fn coord(var: &str, stride: usize, size: usize) -> String {
    if stride == 1 {
        format!("{var} % {size}u")
    } else {
        format!("({var} / {stride}u) % {size}u")
    }
}

fn expr(op: &Op, t: &str, a: &[String]) -> String {
    use Op::*;
    match op {
        Neg => format!("-{}", a[0]),
        Exp => format!("exp({})", a[0]),
        Log => format!("log({})", a[0]),
        Sqrt => format!("sqrt({})", a[0]),
        Sin => format!("rg_sin({})", a[0]),
        Cos => format!("rg_cos({})", a[0]),
        Tanh => format!("rg_tanh({})", a[0]),
        Abs => format!("fabs({})", a[0]),
        Sign => format!("rg_sign({})", a[0]),
        Floor => format!("floor({})", a[0]),
        Add => format!("{} + {}", a[0], a[1]),
        Sub => format!("{} - {}", a[0], a[1]),
        Mul => format!("{} * {}", a[0], a[1]),
        Div => format!("{} / {}", a[0], a[1]),
        Pow => format!("rg_pow({}, {})", a[0], a[1]),
        Max => format!("rg_max({}, {})", a[0], a[1]),
        Min => format!("rg_min({}, {})", a[0], a[1]),
        Gt => format!("{t}({} > {} ? 1 : 0)", a[0], a[1]),
        Ge => format!("{t}({} >= {} ? 1 : 0)", a[0], a[1]),
        Where => format!("{} != {t}(0) ? {} : {}", a[0], a[1], a[2]),
        other => unreachable!("{} is not elementwise", other.name()),
    }
}

/// (buffer element types, kernel body, grid size).
type Generated = (Vec<&'static str>, String, usize);

fn gen_fused(f: &Fused) -> Generated {
    let t = ty(f.dtype);
    let mut sig: Vec<&str> = vec![t; f.outputs.len()];
    sig.extend(std::iter::repeat_n(t, f.inputs.len()));
    let first_in = f.outputs.len();
    let mut body = String::new();
    for (k, &(_, kind, reg)) in f.inputs.iter().enumerate() {
        let b = first_in + k;
        match kind {
            ArgKind::Full => {}
            ArgKind::Splat => body += &format!("{t} r{reg} = b{b}[0];\n"),
            ArgKind::Row(l) => body += &format!("{t} r{reg} = b{b}[i % {l}u];\n"),
            ArgKind::Broadcast => body += &format!("{t} r{reg} = b{b}[{}];\n", index("i", &f.shape, &bstrides(&f.input_shapes[k], f.shape.len()))),
        }
    }
    for m in &f.ops {
        let args: Vec<String> = m
            .args
            .iter()
            .map(|a| match a {
                Operand::Input(k) => format!("b{}[i]", first_in + k),
                Operand::Reg(r) => format!("r{r}"),
            })
            .collect();
        body += &format!("{t} r{} = {};\n", m.reg, expr(&m.op, t, &args));
    }
    for (k, &(reg, _)) in f.outputs.iter().enumerate() {
        body += &format!("b{k}[i] = r{reg};\n");
    }
    (sig, body, f.n)
}

/// `None` when the instruction has no GPU kernel and must run on the CPU.
fn gen_instr(i: &Instr, arg_dtypes: &[DType]) -> Option<Generated> {
    use Op::*;
    if metal_on_cpu(&i.op, i.dtype, i.n, arg_dtypes) {
        return None;
    }
    let t = ty(i.dtype);
    let mut sig: Vec<&str> = vec![t];
    sig.extend(arg_dtypes.iter().map(|&d| ty(d)));
    let shape = &i.shape;
    let rank = shape.len();
    let n = i.n;
    let product = |s: &[usize]| s.iter().product::<usize>();
    let body = match &i.op {
        op if op.is_unary_elementwise() || op.is_binary_elementwise() => {
            let args: Vec<String> = i.arg_shapes.iter().enumerate().map(|(k, s)| format!("b{}[{}]", k + 1, index("i", shape, &bstrides(s, rank)))).collect();
            format!("b0[i] = {};", expr(op, t, &args))
        }
        Where => format!("b0[i] = {};", expr(&Where, t, &["b1[i]".into(), "b2[i]".into(), "b3[i]".into()])),
        Transpose(perm) => {
            let src = strides(&i.arg_shapes[0]);
            let mapped: Vec<usize> = perm.iter().map(|&p| src[p]).collect();
            format!("b0[i] = b1[{}];", index("i", shape, &mapped))
        }
        BroadcastTo(_) => format!("b0[i] = b1[{}];", index("i", shape, &bstrides(&i.arg_shapes[0], rank))),
        Slice { starts, .. } => {
            let src = strides(&i.arg_shapes[0]);
            let base: usize = starts.iter().zip(&src).map(|(a, b)| a * b).sum();
            format!("b0[i] = b1[{base}u + {}];", index("i", shape, &src))
        }
        Pad { starts, .. } => {
            let ds = strides(shape);
            let src = strides(&i.arg_shapes[0]);
            let mut inside = vec!["true".to_string()];
            let mut at = vec!["0u".to_string()];
            for d in 0..rank {
                let c = format!("({})", coord("i", ds[d], shape[d]));
                inside.push(format!("{c} >= {}u && {c} < {}u", starts[d], starts[d] + i.arg_shapes[0][d]));
                at.push(format!("({c} - {}u) * {}u", starts[d], src[d]));
            }
            format!("b0[i] = ({}) ? b1[{}] : {t}(0);", inside.join(" && "), at.join(" + "))
        }
        Concat { axis } => {
            let ds = strides(shape);
            let c = format!("({})", coord("i", ds[*axis], shape[*axis]));
            let mut out = format!("uint c = {c};\n");
            let mut off = 0;
            for (k, s) in i.arg_shapes.iter().enumerate() {
                let st = strides(s);
                let rest: Vec<String> = (0..rank).filter(|&d| d != *axis && shape[d] > 1).map(|d| format!("({}) * {}u", coord("i", ds[d], shape[d]), st[d])).collect();
                let rest = if rest.is_empty() { "0u".to_string() } else { rest.join(" + ") };
                out += &format!("if (c < {}u) {{ b0[i] = b{}[(c - {off}u) * {}u + {rest}]; return; }}\n", off + s[*axis], k + 1, st[*axis]);
                off += s[*axis];
            }
            out
        }
        Take { axis } => {
            let xs = &i.arg_shapes[0];
            let block = product(&xs[*axis + 1..]);
            let dim = xs[*axis];
            let n_idx = product(&i.arg_shapes[1]);
            format!("uint t = i % {block}u; uint q = i / {block}u; uint j = q % {n_idx}u; uint o = q / {n_idx}u;\nb0[i] = b1[(o * {dim}u + uint(clamp(b2[j], 0, {})) ) * {block}u + t];", dim - 1)
        }
        TakeAlong => {
            let v = *i.arg_shapes[0].last()?;
            let k = *i.arg_shapes[1].last()?;
            format!("b0[i] = b1[(i / {k}u) * {v}u + uint(clamp(b2[i], 0, {}))];", v - 1)
        }
        TakeAlongGrad { shape: full } => {
            // Each output element scans its own row of indices, so threads never share a target.
            let v = *full.last()?;
            let k = *i.arg_shapes[1].last()?;
            format!("uint row = i / {v}u; int col = int(i % {v}u); float s = 0.0f;\nfor (uint j = row * {k}u; j < (row + 1u) * {k}u; j++) if (clamp(b2[j], 0, {}) == col) s += b1[j];\nb0[i] = s;", v - 1)
        }
        Reduce { kind, axes, .. } => {
            let in_shape = &i.arg_shapes[0];
            let in_st = strides(in_shape);
            let mut keep = in_shape.clone();
            for &a in axes {
                keep[a] = 1;
            }
            let ks = strides(&keep);
            let mut base: Vec<String> = (0..keep.len()).filter(|&d| keep[d] > 1).map(|d| format!("({}) * {}u", coord("i", ks[d], keep[d]), in_st[d])).collect();
            if base.is_empty() {
                base.push("0u".into());
            }
            let dims: Vec<usize> = axes.iter().map(|&a| in_shape[a]).collect();
            let r_st = strides(&dims);
            let total = product(&dims);
            let mut off: Vec<String> = axes.iter().enumerate().filter(|&(e, _)| dims[e] > 1).map(|(e, &a)| format!("({}) * {}u", coord("r", r_st[e], dims[e]), in_st[a])).collect();
            if off.is_empty() {
                off.push("0u".into());
            }
            let (init, step) = match kind {
                ReduceKind::Sum => ("0.0f", "v + x"),
                ReduceKind::Max => ("-INFINITY", "rg_max(v, x)"),
                ReduceKind::Min => ("INFINITY", "rg_min(v, x)"),
            };
            format!("uint base = {};\nfloat v = {init};\nfor (uint r = 0; r < {total}u; r++) {{ float x = b1[base + {}]; v = {step}; }}\nb0[i] = v;", base.join(" + "), off.join(" + "))
        }
        MatMul => gen_matmul(i, false, false),
        MatMulT { ta, tb } => gen_matmul(i, *ta, *tb),
        Cast(_) => format!("b0[i] = {t}(b1[i]);"),
        Argmax => {
            let v = *i.arg_shapes[0].last()?;
            format!("uint best = 0; float top = b1[i * {v}u];\nfor (uint j = 1; j < {v}u; j++) {{ float x = b1[i * {v}u + j]; if (x > top) {{ top = x; best = j; }} }}\nb0[i] = int(best);")
        }
        RandUniform { .. } => "ulong bits = rg_mix(rg_key(b1) + (ulong(i) + 1UL) * 0x9E3779B97F4A7C15UL);\nb0[i] = float(uint(bits >> 40)) * 5.9604644775390625e-08f;".to_string(),
        KeySplit { .. } => "ulong bits = rg_mix(rg_key(b1) + (ulong(i / 2u) + 1UL) * 0xD1B54A32D192ED03UL);\nb0[i] = int(uint(i % 2u == 0u ? (bits >> 32) : bits));".to_string(),
        _ => return None,
    };
    Some((sig, body, n))
}

fn gen_matmul(i: &Instr, ta: bool, tb: bool) -> String {
    let (sa, sb) = (&i.arg_shapes[0], &i.arg_shapes[1]);
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
    let batch = broadcast_shapes(ba, bb).expect("checked by infer");
    let scale = |v: Vec<usize>, by: usize| -> Vec<usize> { v.into_iter().map(|x| x * by).collect() };
    let (st_a, st_b) = (scale(bstrides(ba, batch.len()), m * k), scale(bstrides(bb, batch.len()), k * n));
    let at_a = if ta { format!("p * {m}u + r") } else { format!("r * {k}u + p") };
    let at_b = if tb { format!("c * {k}u + p") } else { format!("p * {n}u + c") };
    format!(
        "uint b = i / {}u; uint r = (i / {n}u) % {m}u; uint c = i % {n}u;\nuint ba = {}; uint bb = {};\nfloat v = 0.0f;\nfor (uint p = 0; p < {k}u; p++) v += b1[ba + {at_a}] * b2[bb + {at_b}];\nb0[i] = v;",
        m * n,
        batch_index(&batch, &st_a),
        batch_index(&batch, &st_b),
    )
}

/// Batch offset: unlike `index`, the identity shortcut would ignore the matrix size in the strides.
fn batch_index(batch: &[usize], st: &[usize]) -> String {
    let ds = strides(batch);
    let terms: Vec<String> = (0..batch.len()).filter(|&d| batch[d] > 1 && st[d] > 0).map(|d| format!("({}) * {}u", coord("b", ds[d], batch[d]), st[d])).collect();
    if terms.is_empty() {
        "0u".to_string()
    } else {
        terms.join(" + ")
    }
}

/// Below this many multiplications a generated kernel beats MPS's setup (swept on M4 Pro).
const MPS_MIN_FLOPS: usize = 2_000_000;

struct Layout {
    rows: usize,
    cols: usize,
    matrices: usize,
}

fn descriptor(l: &Layout) -> Retained<MPSMatrixDescriptor> {
    // SAFETY: plain value construction; tight rows of float32 are a valid layout.
    unsafe { MPSMatrixDescriptor::matrixDescriptorWithRows_columns_matrices_rowBytes_matrixBytes_dataType(l.rows, l.cols, l.matrices, l.cols * 4, l.rows * l.cols * 4, MPSDataType::Float32) }
}

/// MPS layouts when the batches line up: one flattened matrix, or identically shaped batches.
fn mps_layouts(i: &Instr, ta: bool, tb: bool) -> Option<(Layout, Layout, Layout)> {
    if i.dtype != DType::F32 {
        return None;
    }
    let (sa, sb) = (&i.arg_shapes[0], &i.arg_shapes[1]);
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
    if i.n * k < MPS_MIN_FLOPS {
        return None;
    }
    let stored = |rows: usize, cols: usize, flip: bool, matrices: usize| if flip { Layout { rows: cols, cols: rows, matrices } } else { Layout { rows, cols, matrices } };
    if bb.is_empty() && !ta {
        let big = ba.iter().product::<usize>() * m;
        Some((Layout { rows: big, cols: k, matrices: 1 }, stored(k, n, tb, 1), Layout { rows: big, cols: n, matrices: 1 }))
    } else if ba == bb {
        let count = ba.iter().product();
        Some((stored(m, k, ta, count), stored(k, n, tb, count), Layout { rows: m, cols: n, matrices: count }))
    } else {
        None
    }
}

enum MetalStep {
    /// `writes` is how many leading `slots` are outputs. `barrier`: must see an earlier dispatch's writes.
    Dispatch { pipeline: usize, slots: Vec<usize>, grid: usize, width: usize, writes: usize, barrier: bool },
    /// Apple's matrix multiply; `slots` are left, right, result.
    Mps { kernel: Retained<MPSMatrixMultiplication>, slots: [usize; 3], descriptors: [Retained<MPSMatrixDescriptor>; 3] },
    /// Runs on the CPU with `instr`'s slots renumbered to `slots` order; the last one is the output.
    Cpu { instr: Instr, slots: Vec<usize>, arg_ns: Vec<usize> },
}

pub struct MetalProgram {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pipelines: Vec<Pipeline>,
    steps: Vec<MetalStep>,
    /// MPS kernels are not safe to encode from two threads at once.
    encode_lock: Mutex<()>,
    /// Finished frames. Allocating one costs many steps' worth of time on a large model, and
    /// `run` needs one per chunk. Constants stay in place because swaps only touch other slots.
    pool: Mutex<Vec<Vec<Buffer>>>,
}

// SAFETY: Metal documents devices, queues and pipeline states as safe to share between threads.
// Buffers and command objects live in a `MetalExec`, which one thread owns for its whole run.
unsafe impl Send for MetalProgram {}
unsafe impl Sync for MetalProgram {}

impl MetalProgram {
    pub fn new(plan: &Plan) -> Result<MetalProgram, String> {
        autoreleasepool(|_| MetalProgram::build(plan))
    }

    /// Bytes this process holds on the GPU, to catch leaks.
    pub fn allocated_bytes(&self) -> usize {
        self.device.currentAllocatedSize()
    }

    fn build(plan: &Plan) -> Result<MetalProgram, String> {
        let device = MTLCreateSystemDefaultDevice().ok_or("no Metal device")?;
        let queue = device.newCommandQueue().ok_or("no command queue")?;
        let slot_dtype = |s: usize| plan.slots[s].0;

        let mut kernels: HashMap<String, usize> = HashMap::new();
        let mut sources: Vec<String> = vec![];
        let mut pending = vec![];
        for step in &plan.steps {
            if let Step::One(i) = step {
                let flags = match i.op {
                    Op::MatMul => Some((false, false)),
                    Op::MatMulT { ta, tb } => Some((ta, tb)),
                    _ => None,
                };
                if let Some((ta, tb)) = flags {
                    if let Some((l, r, o)) = mps_layouts(i, ta, tb) {
                        // SAFETY: the dimensions come from shape inference and describe these buffers.
                        let kernel = unsafe {
                            let k = MPSMatrixMultiplication::initWithDevice_transposeLeft_transposeRight_resultRows_resultColumns_interiorColumns_alpha_beta(MPSMatrixMultiplication::alloc(), &device, ta, tb, o.rows, o.cols, if ta { l.rows } else { l.cols }, 1.0, 0.0);
                            k.setBatchStart(0);
                            k.setBatchSize(o.matrices);
                            k
                        };
                        pending.push(MetalStep::Mps { kernel, slots: [i.args[0], i.args[1], i.dst], descriptors: [descriptor(&l), descriptor(&r), descriptor(&o)] });
                        continue;
                    }
                }
            }
            let (generated, slots) = match step {
                Step::Fused(f) => {
                    let mut slots: Vec<usize> = f.outputs.iter().map(|o| o.1).collect();
                    slots.extend(f.inputs.iter().map(|i| i.0));
                    (Some(gen_fused(f)), slots)
                }
                Step::One(i) => {
                    let dtypes: Vec<DType> = i.args.iter().map(|&a| slot_dtype(a)).collect();
                    let mut slots = vec![i.dst];
                    slots.extend(&i.args);
                    (gen_instr(i, &dtypes), slots)
                }
            };
            let Some((sig, body, grid)) = generated else {
                let Step::One(i) = step else { unreachable!("groups always generate") };
                let mut instr = i.clone();
                instr.args = (0..i.args.len()).collect();
                instr.dst = i.args.len();
                let mut slots = i.args.clone();
                slots.push(i.dst);
                let arg_ns = i.arg_shapes.iter().map(|s| s.iter().product()).collect();
                pending.push(MetalStep::Cpu { instr, slots, arg_ns });
                continue;
            };
            let mut params: Vec<String> = sig.iter().enumerate().map(|(k, t)| format!("device {t}* b{k} [[buffer({k})]]")).collect();
            params.push("uint i [[thread_position_in_grid]]".to_string());
            let source = format!("({}) {{\n{body}\n}}", params.join(", "));
            let next = kernels.len();
            let id = *kernels.entry(source.clone()).or_insert_with(|| {
                sources.push(format!("kernel void k{next}{source}\n"));
                next
            });
            let writes = match step {
                Step::Fused(f) => f.outputs.len(),
                Step::One(_) => 1,
            };
            pending.push(MetalStep::Dispatch { pipeline: id, slots, grid, width: 0, writes, barrier: false });
        }

        let options = MTLCompileOptions::new();
        // Fast math would let the compiler assume no NaN or Inf: exactly what diagnostics look for.
        options.setMathMode(MTLMathMode::Safe);
        let source = format!("{PRELUDE}{}", sources.concat());
        let library = device.newLibraryWithSource_options_error(&NSString::from_str(&source), Some(&options)).map_err(|e| format!("kernel compilation failed: {}", e.localizedDescription()))?;
        let mut pipelines = vec![];
        for k in 0..sources.len() {
            let f = library.newFunctionWithName(&NSString::from_str(&format!("k{k}"))).ok_or("generated kernel not found")?;
            pipelines.push(device.newComputePipelineStateWithFunction_error(&f).map_err(|e| format!("pipeline creation failed: {}", e.localizedDescription()))?);
        }
        // Dispatches in one encoder run concurrently, so dependencies need explicit barriers.
        // Tracked buffers cover everything across encoders, which starts a fresh epoch.
        let (mut written, mut read) = (std::collections::HashSet::new(), std::collections::HashSet::new());
        for s in &mut pending {
            match s {
                MetalStep::Dispatch { pipeline, width, grid, slots, writes, barrier } => {
                    *width = pipelines[*pipeline].threadExecutionWidth().min(*grid).max(1);
                    let (outs, ins) = slots.split_at(*writes);
                    *barrier = ins.iter().any(|x| written.contains(x)) || outs.iter().any(|x| written.contains(x) || read.contains(x));
                    if *barrier {
                        written.clear();
                        read.clear();
                    }
                    written.extend(outs.iter().copied());
                    read.extend(ins.iter().copied());
                }
                _ => {
                    written.clear();
                    read.clear();
                }
            }
        }
        Ok(MetalProgram { device, queue, pipelines, steps: pending, encode_lock: Mutex::new(()), pool: Mutex::new(vec![]) })
    }

    pub fn exec<'a>(&'a self, plan: &'a Plan) -> MetalExec<'a> {
        if let Some(buffers) = self.pool.lock().expect("no panic while pooling").pop() {
            return MetalExec { program: self, plan, buffers };
        }
        let buffers: Vec<Buffer> = plan
            .slots
            .iter()
            .map(|&(d, n)| {
                let b = self.device.newBufferWithLength_options((n * d.size()).max(4), MTLResourceOptions::StorageModeShared).expect("buffer allocation failed");
                // SAFETY: freshly created shared buffer of at least that many bytes, not yet in use.
                unsafe { std::ptr::write_bytes(b.contents().as_ptr() as *mut u8, 0, (n * d.size()).max(4)) };
                b
            })
            .collect();
        let mut e = MetalExec { program: self, plan, buffers };
        for (s, t) in &plan.consts {
            e.write_prefix(*s, &t.data);
        }
        e
    }
}

pub struct MetalExec<'a> {
    program: &'a MetalProgram,
    plan: &'a Plan,
    buffers: Vec<Buffer>,
}

fn write_bytes(b: &Buffer, data: &Buf) {
    let dst = b.contents().as_ptr() as *mut u8;
    let (src, bytes) = match data {
        Buf::F32(v) => (v.as_ptr() as *const u8, v.len() * 4),
        Buf::F64(v) => (v.as_ptr() as *const u8, v.len() * 8),
        Buf::I32(v) => (v.as_ptr() as *const u8, v.len() * 4),
    };
    assert!(bytes <= b.length());
    // SAFETY: bounds asserted; the queue is drained whenever the CPU touches a buffer.
    unsafe { std::ptr::copy_nonoverlapping(src, dst, bytes) };
}

fn read_bytes(b: &Buffer, d: DType, n: usize) -> Buf {
    assert!(n * d.size() <= b.length());
    let p = b.contents().as_ptr();
    // SAFETY: bounds asserted; the buffer holds plain numbers of this dtype and the queue is drained.
    unsafe {
        match d {
            DType::F32 => Buf::F32(std::slice::from_raw_parts(p as *const f32, n).to_vec()),
            DType::F64 => Buf::F64(std::slice::from_raw_parts(p as *const f64, n).to_vec()),
            DType::I32 => Buf::I32(std::slice::from_raw_parts(p as *const i32, n).to_vec()),
        }
    }
}

impl Drop for MetalExec<'_> {
    fn drop(&mut self) {
        self.program.pool.lock().expect("no panic while pooling").push(std::mem::take(&mut self.buffers));
    }
}

impl MetalExec<'_> {
    /// Metal hands back autoreleased command objects, and the threads running this have no pool
    /// of their own: without this one they would keep every buffer alive for the whole process.
    fn encode(&self, range: std::ops::Range<usize>, stats: &mut Stats) {
        autoreleasepool(|_| self.encode_inner(range, stats));
    }

    fn encode_inner(&self, range: std::ops::Range<usize>, stats: &mut Stats) {
        let t_start = std::time::Instant::now();
        let mut waited = 0.0;
        let mut cpu = 0.0;
        let _guard = self.program.encode_lock.lock().expect("no panic while encoding");
        type Cb = Retained<ProtocolObject<dyn MTLCommandBuffer>>;
        type Enc = Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>;
        let mut cb: Option<Cb> = None;
        let mut enc: Option<Enc> = None;
        const BIND_CACHE: usize = 8;
        let mut bound_pipeline = usize::MAX;
        let mut bound = [usize::MAX; BIND_CACHE];
        let mut gpu_seconds = 0.0;
        let mut command_buffers = 0;
        let mut finish = |cb: &mut Option<Cb>, enc: &mut Option<Enc>, waited: &mut f64| {
            if let Some(e) = enc.take() {
                e.endEncoding();
            }
            if let Some(c) = cb.take() {
                let t0 = std::time::Instant::now();
                c.commit();
                c.waitUntilCompleted();
                *waited += t0.elapsed().as_secs_f64();
                gpu_seconds += c.GPUEndTime() - c.GPUStartTime();
                command_buffers += 1;
                if let Some(err) = c.error() {
                    // shortcut: a GPU failure aborts the call as a panic, make it a Diag if it proves recoverable
                    panic!("Metal command buffer failed: {}", err.localizedDescription());
                }
            }
        };
        for s in &self.program.steps[range] {
            match s {
                MetalStep::Dispatch { pipeline, slots, grid, width, barrier, .. } => {
                    if *grid == 0 {
                        continue;
                    }
                    // Unretained: `self.buffers` outlives the wait below, so per-binding retains are waste.
                    let c = cb.get_or_insert_with(|| self.program.queue.commandBufferWithUnretainedReferences().expect("command buffer"));
                    if enc.is_none() {
                        bound_pipeline = usize::MAX;
                        bound = [usize::MAX; BIND_CACHE];
                    }
                    let e = enc.get_or_insert_with(|| c.computeCommandEncoderWithDispatchType(MTLDispatchType::Concurrent).expect("compute encoder"));
                    if *barrier {
                        e.memoryBarrierWithScope(MTLBarrierScope::Buffers);
                        stats.barriers += 1;
                    }
                    stats.dispatches += 1;
                    if bound_pipeline != *pipeline {
                        e.setComputePipelineState(&self.program.pipelines[*pipeline]);
                        bound_pipeline = *pipeline;
                    }
                    for (k, &slot) in slots.iter().enumerate() {
                        if k >= BIND_CACHE || bound[k] != slot {
                            // SAFETY: the kernel's index expressions stay inside the planned slot sizes.
                            unsafe { e.setBuffer_offset_atIndex(Some(&self.buffers[slot]), 0, k) };
                            if k < BIND_CACHE {
                                bound[k] = slot;
                            }
                        }
                    }
                    e.dispatchThreads_threadsPerThreadgroup(MTLSize { width: *grid, height: 1, depth: 1 }, MTLSize { width: *width, height: 1, depth: 1 });
                }
                MetalStep::Mps { kernel, slots, descriptors } => {
                    if let Some(e) = enc.take() {
                        e.endEncoding();
                    }
                    stats.mps += 1;
                    let c = cb.get_or_insert_with(|| self.program.queue.commandBufferWithUnretainedReferences().expect("command buffer"));
                    // SAFETY: descriptors were derived from these slots' shapes. Matrix objects are
                    // built per encode because state buffers swap between steps.
                    unsafe {
                        let matrix = |k: usize| MPSMatrix::initWithBuffer_descriptor(MPSMatrix::alloc(), &self.buffers[slots[k]], &descriptors[k]);
                        kernel.encodeToCommandBuffer_leftMatrix_rightMatrix_resultMatrix(c, &matrix(0), &matrix(1), &matrix(2));
                    }
                }
                MetalStep::Cpu { instr, slots, arg_ns } => {
                    finish(&mut cb, &mut enc, &mut waited);
                    let t0 = std::time::Instant::now();
                    stats.cpu_steps += 1;
                    let last = slots.len() - 1;
                    let mut bufs: Vec<Buf> = slots[..last].iter().zip(arg_ns).map(|(&s, &n)| read_bytes(&self.buffers[s], self.plan.slots[s].0, n)).collect();
                    bufs.push(Buf::zeros(instr.dtype, instr.n));
                    exec_instr(instr, &mut bufs);
                    write_bytes(&self.buffers[slots[last]], &bufs[last]);
                    cpu += t0.elapsed().as_secs_f64();
                }
            }
        }
        finish(&mut cb, &mut enc, &mut waited);
        stats.wall += t_start.elapsed().as_secs_f64();
        stats.waited += waited;
        stats.cpu += cpu;
        stats.gpu += gpu_seconds;
        stats.command_buffers += command_buffers;
    }

    /// Each step in its own command buffer, timed by GPU timestamps. Sequential execution removes
    /// concurrency, so this ranks operations; `breakdown` shows what a real step costs.
    pub fn step_timed(&mut self, seconds: &mut [f64]) {
        for k in 0..self.program.steps.len() {
            let mut st = Stats::default();
            self.encode(k..k + 1, &mut st);
            seconds[k] += st.gpu + st.cpu;
        }
    }

    /// Runs whole steps and reports where the wall time went.
    pub fn breakdown(&mut self, repeats: usize) -> Value {
        let mut st = Stats::default();
        for _ in 0..repeats {
            self.encode(0..self.program.steps.len(), &mut st);
        }
        let per = |x: f64| x / repeats as f64 * 1e3;
        json!({
            "wall_ms": per(st.wall), "gpu_ms": per(st.gpu), "encode_ms": per(st.wall - st.waited - st.cpu), "cpu_fallback_ms": per(st.cpu),
            "wait_beyond_gpu_ms": per(st.waited - st.gpu),
            "per_step": { "dispatches": st.dispatches / repeats, "barriers": st.barriers / repeats, "mps_calls": st.mps / repeats, "cpu_steps": st.cpu_steps / repeats, "command_buffers": st.command_buffers / repeats },
        })
    }
}

#[derive(Default)]
struct Stats {
    wall: f64,
    waited: f64,
    gpu: f64,
    cpu: f64,
    dispatches: usize,
    barriers: usize,
    mps: usize,
    cpu_steps: usize,
    command_buffers: usize,
}

impl Exec for MetalExec<'_> {
    fn load(&mut self, k: usize, t: &Tensor) {
        write_bytes(&self.buffers[self.plan.input_slots[k]], &t.data.prefix(self.plan.input_ns[k]));
    }
    fn step(&mut self) {
        self.encode(0..self.program.steps.len(), &mut Stats::default());
    }
    fn swap(&mut self, a: usize, b: usize) {
        self.buffers.swap(a, b);
    }
    fn prefix(&self, slot: usize, n: usize) -> Buf {
        read_bytes(&self.buffers[slot], self.plan.slots[slot].0, n)
    }
    fn write_prefix(&mut self, slot: usize, b: &Buf) {
        write_bytes(&self.buffers[slot], b);
    }
    fn scalar(&self, slot: usize) -> f64 {
        read_bytes(&self.buffers[slot], self.plan.slots[slot].0, 1).to_f64()[0]
    }
}
