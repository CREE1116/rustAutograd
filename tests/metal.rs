//! The GPU backend, checked against the CPU kernels as the reference.
#![cfg(target_os = "macos")]

use rg::autodiff::grad;
use rg::graph::{Graph, Op, ReduceKind};
use rg::metal::{available, MetalProgram};
use rg::optimize::optimize;
use rg::plan::{eval, Exec, Fuse, Plan};
use rg::tensor::{Buf, DType, Tensor};
use rg::validate::{compile_on, Level, RunSpec};

const F: DType = DType::F32;

fn t(shape: &[usize], v: &[f64]) -> Tensor {
    Tensor::from_f64(F, shape, v)
}

fn ints(shape: &[usize], v: &[i32]) -> Tensor {
    Tensor { shape: shape.to_vec(), data: Buf::I32(v.to_vec()) }
}

fn on_gpu(g: &Graph, outs: &[u32], inputs: &[&Tensor]) -> Vec<Tensor> {
    let plan = Plan::build_fused(g, outs, Fuse::METAL);
    let prog = MetalProgram::new(&plan).expect("kernels must compile");
    let mut e = prog.exec(&plan).unwrap();
    for (k, x) in inputs.iter().enumerate() {
        e.load(k, x);
    }
    e.step();
    plan.outputs.iter().map(|o| Tensor { shape: o.shape.clone(), data: e.prefix(o.slot, o.n) }).collect()
}

fn close(a: &Tensor, b: &Tensor, tol: f64, what: &str) {
    assert_eq!(a.shape, b.shape, "{what}");
    let (va, vb) = (a.data.to_f64(), b.data.to_f64());
    let scale = vb.iter().filter(|v| v.is_finite()).fold(1.0f64, |m, v| m.max(v.abs()));
    for (j, (&x, &y)) in va.iter().zip(&vb).enumerate() {
        assert!(x == y || (x.is_nan() && y.is_nan()) || (x - y).abs() <= tol * scale, "{what}[{j}]: gpu {x} cpu {y}");
    }
}

fn agree(build: impl Fn(&mut Graph, &[u32]) -> u32, inputs: &[Tensor], what: &str) {
    let mut g = Graph::new();
    let ids: Vec<u32> = inputs.iter().map(|x| g.input(&x.shape, x.dtype())).collect();
    let out = build(&mut g, &ids);
    let refs: Vec<&Tensor> = inputs.iter().collect();
    close(&on_gpu(&g, &[out], &refs)[0], &eval(&g, &[out], &refs)[0], 2e-6, what);
}

fn seq(shape: &[usize]) -> Tensor {
    let n: usize = shape.iter().product();
    t(shape, &(0..n).map(|i| 0.25 + ((i as f64) * 0.73).sin().abs()).collect::<Vec<_>>())
}

#[test]
fn every_elementwise_op_and_broadcast_pattern() {
    assert!(available());
    let unary = [Op::Neg, Op::Exp, Op::Log, Op::Sqrt, Op::Sin, Op::Cos, Op::Tanh, Op::Abs, Op::Sign, Op::Floor];
    for op in unary {
        agree(|g, i| g.add(op.clone(), &[i[0]]).unwrap(), &[seq(&[5, 7])], op.name());
    }
    let binary = [Op::Add, Op::Sub, Op::Mul, Op::Div, Op::Pow, Op::Max, Op::Min, Op::Gt, Op::Ge];
    let shapes: [(&[usize], &[usize]); 6] = [(&[4, 3], &[4, 3]), (&[4, 3], &[]), (&[], &[4, 3]), (&[4, 3], &[3]), (&[4, 1], &[1, 3]), (&[2, 1, 3], &[4, 1])];
    for op in &binary {
        for (a, b) in &shapes {
            agree(|g, i| g.add(op.clone(), &[i[0], i[1]]).unwrap(), &[seq(a), seq(b)], &format!("{} {a:?} {b:?}", op.name()));
        }
    }
    agree(|g, i| { let c = g.add(Op::Gt, &[i[0], i[1]]).unwrap(); g.add(Op::Where, &[c, i[0], i[1]]).unwrap() }, &[seq(&[3, 3]), t(&[3, 3], &[0.5; 9])], "where");
}

#[test]
fn ieee_edge_cases_match_cpu_with_fast_math_off() {
    let specials = [0.0, -0.0, 1.0, -1.0, 2.5, -2.5, f64::INFINITY, f64::NEG_INFINITY, f64::NAN, 1e30, -1e30, 1e-30];
    let x = t(&[12], &specials);
    for op in [Op::Exp, Op::Log, Op::Sqrt, Op::Tanh, Op::Abs, Op::Sign, Op::Floor, Op::Neg] {
        agree(|g, i| g.add(op.clone(), &[i[0]]).unwrap(), &[x.clone()], &format!("special {}", op.name()));
    }
    // Trigonometry of huge values differs in precision on the GPU, so those are left out.
    let trig = t(&[6], &[0.0, 1.0, -2.5, f64::INFINITY, f64::NEG_INFINITY, f64::NAN]);
    for op in [Op::Sin, Op::Cos] {
        agree(|g, i| g.add(op.clone(), &[i[0]]).unwrap(), &[trig.clone()], &format!("special {}", op.name()));
    }
    let both: Vec<(f64, f64)> = specials.iter().flat_map(|&a| specials.iter().map(move |&b| (a, b))).collect();
    let a = t(&[both.len()], &both.iter().map(|p| p.0).collect::<Vec<_>>());
    let b = t(&[both.len()], &both.iter().map(|p| p.1).collect::<Vec<_>>());
    for op in [Op::Add, Op::Sub, Op::Mul, Op::Div, Op::Max, Op::Min, Op::Gt, Op::Ge] {
        agree(|g, i| g.add(op.clone(), &[i[0], i[1]]).unwrap(), &[a.clone(), b.clone()], &format!("special {}", op.name()));
    }
    // Negative bases with integer exponents, where a plain MSL pow returns NaN.
    let base = t(&[6], &[-2.0, -2.0, -3.0, 2.0, 0.0, -2.0]);
    let exp = t(&[6], &[2.0, 3.0, -1.0, 0.5, 2.0, 0.5]);
    agree(|g, i| g.add(Op::Pow, &[i[0], i[1]]).unwrap(), &[base, exp], "pow");
}

#[test]
fn structural_ops() {
    let x = seq(&[3, 4, 5]);
    agree(|g, i| g.add(Op::Transpose(vec![2, 0, 1]), &[i[0]]).unwrap(), &[x.clone()], "transpose");
    agree(|g, i| g.add(Op::BroadcastTo(vec![2, 3, 4, 5]), &[i[0]]).unwrap(), &[x.clone()], "broadcast");
    agree(|g, i| g.add(Op::Slice { starts: vec![1, 0, 2], stops: vec![3, 4, 4] }, &[i[0]]).unwrap(), &[x.clone()], "slice");
    agree(|g, i| g.add(Op::Pad { starts: vec![1, 2, 0], shape: vec![5, 6, 7] }, &[i[0]]).unwrap(), &[x.clone()], "pad");
    for axis in 0..3 {
        agree(|g, i| g.add(Op::Concat { axis }, &[i[0], i[0], i[0]]).unwrap(), &[x.clone()], &format!("concat {axis}"));
        let idx = ints(&[2, 2], &[1, 0, 2, 2]);
        agree(|g, i| g.add(Op::Take { axis }, &[i[0], i[1]]).unwrap(), &[x.clone(), idx], &format!("take {axis}"));
    }
    let ix = ints(&[3, 4, 2], &(0..24).map(|i| i % 5).collect::<Vec<_>>());
    agree(|g, i| g.add(Op::TakeAlong, &[i[0], i[1]]).unwrap(), &[x.clone(), ix.clone()], "take_along");
    let ct = seq(&[3, 4, 2]);
    agree(|g, i| g.add(Op::TakeAlongGrad { shape: vec![3, 4, 5] }, &[i[0], i[1]]).unwrap(), &[ct, ix], "take_along_grad");
    let rows = ints(&[3], &[1, 1, 0]);
    agree(|g, i| g.add(Op::TakeGrad { axis: 0, shape: vec![2, 5] }, &[i[0], i[1]]).unwrap(), &[seq(&[3, 5]), rows], "take_grad");
    agree(|g, i| g.add(Op::Argmax, &[i[0]]).unwrap(), &[x.clone()], "argmax");
    agree(|g, i| { let c = g.add(Op::Cast(DType::I32), &[i[0]]).unwrap(); g.add(Op::Cast(F), &[c]).unwrap() }, &[t(&[3], &[1.5, -1.5, 3.9])], "cast");
    agree(|g, i| g.add(Op::Add, &[i[0], i[0]]).unwrap(), &[ints(&[3], &[1, -2, 3])], "int add");
    agree(|g, i| g.add(Op::Take { axis: 0 }, &[i[0], i[1]]).unwrap(), &[seq(&[3]), ints(&[3], &[9, -4, 1])], "clamped indices");
}

#[test]
fn reductions() {
    let x = seq(&[3, 4, 5]);
    for kind in [ReduceKind::Sum, ReduceKind::Max, ReduceKind::Min] {
        for axes in [vec![0], vec![1], vec![2], vec![0, 2], vec![1, 2], vec![0, 1, 2], vec![]] {
            for keepdims in [true, false] {
                agree(|g, i| g.add(Op::Reduce { kind, axes: axes.clone(), keepdims }, &[i[0]]).unwrap(), &[x.clone()], &format!("{kind:?} {axes:?} {keepdims}"));
            }
        }
    }
}

#[test]
fn matmul_ranks_flags_and_batches() {
    let cases: [(&[usize], &[usize]); 9] = [(&[3], &[3]), (&[2, 3], &[3]), (&[3], &[3, 2]), (&[2, 3], &[3, 4]), (&[5, 2, 3], &[3, 4]), (&[5, 2, 3], &[5, 3, 4]), (&[2, 3], &[7, 3, 4]), (&[1, 2, 3], &[6, 3, 4]), (&[5, 1, 2, 3], &[1, 4, 3, 2])];
    for (a, b) in cases {
        agree(|g, i| g.add(Op::MatMul, &[i[0], i[1]]).unwrap(), &[seq(a), seq(b)], &format!("matmul {a:?} {b:?}"));
    }
    let swap = |g: &mut Graph, x: u32| { let r = g.shape(x).len(); let mut p: Vec<usize> = (0..r).collect(); p.swap(r - 2, r - 1); g.add(Op::Transpose(p), &[x]).unwrap() };
    let flag_cases: [(&[usize], &[usize], bool, bool); 4] = [(&[3, 2], &[3, 4], true, false), (&[2, 3], &[4, 3], false, true), (&[3, 2], &[4, 3], true, true), (&[6, 3, 2], &[6, 4, 3], true, true)];
    for (a, b, ta, tb) in flag_cases {
        let mut g = Graph::new();
        let (x, y) = (g.input(a, F), g.input(b, F));
        let (l, r) = (if ta { swap(&mut g, x) } else { x }, if tb { swap(&mut g, y) } else { y });
        let m = g.add(Op::MatMul, &[l, r]).unwrap();
        let (o, ids, _) = optimize(&g, &[m]);
        let (ia, ib) = (seq(a), seq(b));
        let plan = Plan::build_fused(&o, &ids, Fuse::METAL);
        assert!(plan.instrs().iter().any(|i| matches!(i.op, Op::MatMulT { .. })));
        close(&on_gpu(&o, &ids, &[&ia, &ib])[0], &eval(&g, &[m], &[&ia, &ib])[0], 2e-6, "matmul_t");
    }
}

#[test]
fn random_bits_are_identical_to_cpu_and_normal_falls_back() {
    let key = ints(&[2], &[7, -11]);
    let mut g = Graph::new();
    let k = g.input(&[2], DType::I32);
    let u = g.add(Op::RandUniform { shape: vec![1000], dtype: F }, &[k]).unwrap();
    let s = g.add(Op::KeySplit { n: 5 }, &[k]).unwrap();
    let n = g.add(Op::RandNormal { shape: vec![100], dtype: F }, &[k]).unwrap();
    let wide = g.add(Op::RandUniform { shape: vec![10], dtype: DType::F64 }, &[k]).unwrap();
    let cpu = eval(&g, &[u, s, n, wide], &[&key]);
    let gpu = on_gpu(&g, &[u, s, n, wide], &[&key]);
    assert_eq!(gpu, cpu, "the same key must give the same bits on both devices");
}

#[test]
fn float64_programs_run_through_cpu_fallback_between_gpu_steps() {
    let mut g = Graph::new();
    let x = g.input(&[4], F);
    let e = g.add(Op::Exp, &[x]).unwrap();
    let w = g.add(Op::Cast(DType::F64), &[e]).unwrap();
    let l = g.add(Op::Log, &[w]).unwrap();
    let back = g.add(Op::Cast(F), &[l]).unwrap();
    let m = g.add(Op::Mul, &[back, x]).unwrap();
    let input = seq(&[4]);
    close(&on_gpu(&g, &[m, l], &[&input])[0], &eval(&g, &[m, l], &[&input])[0], 2e-6, "fallback");
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// Fusion, buffer reuse and generated kernels together, including backward graphs.
#[test]
fn random_graphs_agree_with_cpu() {
    let mut r = Rng(0xFEED_FACE_CAFE_BEEF);
    let mut fused = 0;
    for case in 0..80 {
        let mut g = Graph::new();
        let (a, b) = ([1, 3, 8, 40][r.below(4)], [1, 2, 5, 17][r.below(4)]);
        let pool_shapes = [vec![], vec![b], vec![a, b], vec![1, b], vec![a, 1]];
        let mut inputs = vec![];
        let mut pool: Vec<u32> = vec![];
        for s in [vec![a, b], pool_shapes[r.below(5)].clone(), pool_shapes[r.below(5)].clone()] {
            let n: usize = s.iter().product();
            let v: Vec<f64> = (0..n).map(|_| 0.2 + (r.next() % 1000) as f64 / 1000.0).collect();
            pool.push(g.input(&s, F));
            inputs.push(t(&s, &v));
        }
        for _ in 0..2 + r.below(20) {
            let x = pool[pool.len() - 1 - r.below(pool.len().min(4))];
            let y = pool[r.below(pool.len())];
            let res = match r.below(13) {
                0 => g.add(Op::Add, &[x, y]),
                1 => g.add(Op::Sub, &[x, y]),
                2 => g.add(Op::Mul, &[x, y]),
                3 => g.add(Op::Div, &[x, y]),
                4 => g.add(Op::Max, &[x, y]),
                5 => g.add(Op::Tanh, &[x]),
                6 => g.add(Op::Abs, &[x]),
                7 => g.add(Op::Sin, &[x]),
                8 => g.add(Op::StopGradient, &[x]),
                9 => { let n = g.nodes[x as usize].n(); g.add(Op::Reshape(vec![n]), &[x]) }
                10 => { let axes = (0..g.shape(x).len()).filter(|_| r.next() % 2 == 0).collect(); g.add(Op::Reduce { kind: ReduceKind::Sum, axes, keepdims: true }, &[x]) }
                11 => g.add(Op::MatMul, &[x, y]),
                _ => g.add(Op::Gt, &[x, y]).and_then(|c| g.add(Op::Where, &[c, x, y])),
            };
            if let Ok(id) = res {
                pool.push(id);
            }
        }
        let last = *pool.last().unwrap();
        let axes = (0..g.shape(last).len()).collect();
        let y = g.add(Op::Reduce { kind: ReduceKind::Sum, axes, keepdims: false }, &[last]).unwrap();
        let mut outs: Vec<u32> = (0..2).map(|_| pool[r.below(pool.len())]).collect();
        outs.extend(grad(&mut g, y, &[pool[0]]).unwrap());
        let (o, ids, _) = optimize(&g, &outs);
        let args: Vec<&Tensor> = inputs.iter().collect();
        fused += Plan::build_fused(&o, &ids, Fuse::METAL).fused().len();
        let (gpu, cpu) = (on_gpu(&o, &ids, &args), eval(&g, &outs, &args));
        for k in 0..outs.len() {
            close(&gpu[k], &cpu[k], 5e-3, &format!("case {case} output {k}"));
        }
    }
    assert!(fused > 30, "fuzz must exercise fused kernels: {fused}");
}

#[test]
fn run_loop_trains_on_the_gpu_and_matches_cpu() {
    let build = |metal: bool| {
        let mut g = Graph::new();
        let (w, x, y) = (g.input(&[3], F), g.input(&[8, 3], F), g.input(&[8], F));
        let p = g.add(Op::MatMul, &[x, w]).unwrap();
        let e = g.add(Op::Sub, &[p, y]).unwrap();
        let sq = g.add(Op::Mul, &[e, e]).unwrap();
        let l = g.add(Op::Reduce { kind: ReduceKind::Sum, axes: vec![0], keepdims: false }, &[sq]).unwrap();
        let gr = grad(&mut g, l, &[w]).unwrap()[0];
        let lr = g.scalar(F, 0.01);
        let d = g.add(Op::Mul, &[gr, lr]).unwrap();
        let nw = g.add(Op::Sub, &[w, d]).unwrap();
        let inputs = [t(&[3], &[0.0; 3]), seq(&[8, 3]), t(&[8], &[1., 2., 3., 4., 5., 6., 7., 8.])];
        (compile_on(g, vec![nw, l], &inputs, Level::Standard, metal).unwrap(), inputs)
    };
    let spec = RunSpec { state: vec![(0, 0)], data: vec![1, 2], history: vec![1] };
    let ((gpu, inputs), (cpu, _)) = (build(true), build(false));
    assert_eq!(gpu.report["status"], "ok", "{}", gpu.report);
    assert_eq!(gpu.report["device"], "metal");
    assert_eq!(cpu.report["device"], "cpu");
    let batch = vec![vec![inputs[1].clone(), inputs[2].clone()]];
    let (a, b) = (gpu.run(&spec, &inputs[..1], &batch, 200), cpu.run(&spec, &inputs[..1], &batch, 200));
    assert_eq!(a.steps, 200);
    close(&a.state[0], &b.state[0], 1e-3, "state after 200 steps");
    assert!(a.history[0][199] < a.history[0][0]);
    let p = gpu.profile(&inputs, 2);
    assert!(!p["operations"].as_array().unwrap().is_empty());
}

#[test]
fn repeated_runs_do_not_leak_gpu_memory() {
    let mut g = Graph::new();
    let x = g.input(&[256, 256], F);
    let m = g.add(Op::MatMul, &[x, x]).unwrap();
    let y = g.add(Op::Tanh, &[m]).unwrap();
    let plan = Plan::build_fused(&g, &[y], Fuse::METAL);
    let prog = MetalProgram::new(&plan).unwrap();
    let input = seq(&[256, 256]);
    let cycle = || {
        let mut e = prog.exec(&plan).unwrap();
        e.load(0, &input);
        for _ in 0..5 {
            e.step();
        }
    };
    cycle();
    let before = prog.allocated_bytes();
    for _ in 0..200 {
        cycle(); // each cycle allocates, and must release, about 1.5 MB
    }
    let after = prog.allocated_bytes();
    assert!(after < before + 32 * 1024 * 1024, "GPU memory grew from {before} to {after}");
}

#[test]
fn cpu_bookkeeping_is_hoisted_and_take_along_grad_runs_on_the_gpu() {
    // The key split is written last, but must run before any GPU work starts.
    let mut g = Graph::new();
    let (x, k) = (g.input(&[64, 64], F), g.input(&[2], DType::I32));
    let y = g.add(Op::Tanh, &[x]).unwrap();
    let keys = g.add(Op::KeySplit { n: 2 }, &[k]).unwrap();
    let plan = Plan::build_fused(&g, &[y, keys], Fuse::METAL);
    assert!(matches!(&plan.instrs()[0].op, Op::KeySplit { .. }), "{:?}", plan.instrs()[0].op.name());
    let (xs, key) = (seq(&[64, 64]), ints(&[2], &[3, 4]));
    let (gpu, cpu) = (on_gpu(&g, &[y, keys], &[&xs, &key]), eval(&g, &[y, keys], &[&xs, &key]));
    close(&gpu[0], &cpu[0], 2e-6, "tanh");
    assert_eq!(gpu[1], cpu[1]);

    // Repeated and clamped indices in several rows.
    let ct = seq(&[2, 3, 2]);
    let idx = ints(&[2, 3, 2], &[1, 1, 0, 4, 9, -3, 2, 2, 3, 3, 4, 0]);
    agree(|g, i| g.add(Op::TakeAlongGrad { shape: vec![2, 3, 5] }, &[i[0], i[1]]).unwrap(), &[ct, idx], "take_along_grad repeated");
}

#[test]
fn consecutive_runs_with_swapped_state_stay_correct_with_unretained_buffers() {
    let mut g = Graph::new();
    let w = g.input(&[512], F);
    let a = g.add(Op::Tanh, &[w]).unwrap();
    let b = g.add(Op::Add, &[a, w]).unwrap();
    let input = seq(&[512]);
    let c = compile_on(g.clone(), vec![b], std::slice::from_ref(&input), Level::Basic, true).unwrap();
    let cpu = compile_on(g, vec![b], std::slice::from_ref(&input), Level::Basic, false).unwrap();
    let spec = RunSpec { state: vec![(0, 0)], data: vec![], history: vec![] };
    let (x, y) = (c.run(&spec, std::slice::from_ref(&input), &[], 50), cpu.run(&spec, std::slice::from_ref(&input), &[], 50));
    close(&x.state[0], &y.state[0], 1e-4, "state after swaps");
}

#[test]
fn pooled_frames_keep_constants_and_do_not_leak_state_between_runs() {
    let mut g = Graph::new();
    let w = g.input(&[4], F);
    let c = g.constant(t(&[4], &[1.0, 2.0, 3.0, 4.0]));
    let m = g.add(Op::Mul, &[w, c]).unwrap();
    let input = t(&[4], &[1.0, 1.0, 1.0, 1.0]);
    let prog = compile_on(g, vec![m], std::slice::from_ref(&input), Level::Basic, true).unwrap();
    let spec = RunSpec { state: vec![(0, 0)], data: vec![], history: vec![] };
    let first = prog.run(&spec, std::slice::from_ref(&input), &[], 3);
    for _ in 0..4 {
        // Same start, same result, however many times the frame was recycled and swapped.
        assert_eq!(prog.run(&spec, std::slice::from_ref(&input), &[], 3).state, first.state);
    }
    assert_eq!(first.state[0].data.to_f64(), [1.0, 8.0, 27.0, 64.0]);
}

#[test]
fn a_buffer_beyond_the_device_limit_fails_at_compile_time_with_a_diagnostic() {
    let huge = 1usize << 42; // 16 TiB of float32, and nothing is allocated
    let mut g = Graph::new();
    let x = g.input(&[huge], F);
    let y = g.add(Op::Tanh, &[x]).unwrap();
    let plan = Plan::build_fused(&g, &[y], Fuse::METAL);
    let err = MetalProgram::new(&plan).err().expect("must refuse");
    assert!(err.contains("exceeds"), "{err}");
}

#[test]
fn gpu_failure_reaches_the_caller_as_a_diagnostic_and_stops_the_run() {
    let mut g = Graph::new();
    let w = g.input(&[4], F);
    let two = g.scalar(F, 0.5);
    let m = g.add(Op::Mul, &[w, two]).unwrap();
    let input = t(&[4], &[1.0, 2.0, 3.0, 4.0]);
    let c = compile_on(g, vec![m], std::slice::from_ref(&input), Level::Basic, true).unwrap();
    let gpu = c.gpu.as_ref().unwrap();

    gpu.inject_failure_for_test();
    let d = c.try_call(std::slice::from_ref(&input)).unwrap_err();
    assert_eq!(d["code"], "DEVICE_EXECUTION_FAILED");
    assert_eq!(d["observation"]["detail"], "injected failure");

    let spec = RunSpec { state: vec![(0, 0)], data: vec![], history: vec![] };
    gpu.inject_failure_for_test();
    let out = c.run(&spec, std::slice::from_ref(&input), &[], 10);
    assert_eq!(out.steps, 0);
    assert_eq!(out.device_error.as_deref(), Some("injected failure"));

    // The program recovers: a later run on a recycled frame computes correctly.
    let ok = c.run(&spec, std::slice::from_ref(&input), &[], 1);
    assert!(ok.device_error.is_none());
    assert_eq!(ok.state[0].data.to_f64(), [0.5, 1.0, 1.5, 2.0]);
}

#[test]
fn sort_and_control_flow_run_through_the_cpu_fallback_on_metal() {
    let input = seq(&[3, 8]);
    agree(|g, i| g.add(Op::ArgTopK { k: 3 }, &[i[0]]).unwrap(), std::slice::from_ref(&input), "arg_top_k");
    agree(|g, i| g.add(Op::Cumsum { reverse: true }, &[i[0]]).unwrap(), std::slice::from_ref(&input), "cumsum");

    let mut b = Graph::new();
    let (c, x) = (b.input(&[8], F), b.input(&[8], F));
    let m = b.add(Op::Mul, &[c, x]).unwrap();
    let next = b.add(Op::Tanh, &[m]).unwrap();
    let body = rg::graph::Body::new(b, vec![next, m]);
    let mut g = Graph::new();
    let (c0, xs) = (g.input(&[8], F), g.input(&[3, 8], F));
    let d = g.add(Op::Tanh, &[c0]).unwrap(); // GPU work before and after the loop
    let node = g.add(Op::Scan { body, n_carry: 1, n_x: 1, length: 3, reverse: false }, &[d, xs]).unwrap();
    let last = g.add(Op::Output { index: 0, shape: vec![8], dtype: F }, &[node]).unwrap();
    let ys = g.add(Op::Output { index: 1, shape: vec![3, 8], dtype: F }, &[node]).unwrap();
    let e = g.add(Op::Exp, &[last]).unwrap();
    let axes = vec![0];
    let s = g.add(Op::Reduce { kind: ReduceKind::Sum, axes, keepdims: false }, &[e]).unwrap();
    let gr = grad(&mut g, s, &[c0, xs]).unwrap();
    let c0v = seq(&[8]);
    let args = [&c0v, &input];
    let mut outs = vec![last, ys];
    outs.extend(gr);
    let (o, ids, _) = optimize(&g, &outs);
    let (gpu, cpu) = (on_gpu(&o, &ids, &args), eval(&g, &outs, &args));
    for k in 0..outs.len() {
        close(&gpu[k], &cpu[k], 1e-3, &format!("scan output {k}"));
    }
}
