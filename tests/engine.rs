use rg::autodiff::grad;
use rg::graph::{Graph, GradRecord, Op, ReduceKind};
use rg::optimize::optimize;
use rg::plan::{eval, ArgKind, Fuse, Plan};
use rg::tensor::{Buf, DType, Tensor};
use rg::validate::{compile, Level, RunSpec};

const F: DType = DType::F64;

fn t(shape: &[usize], v: &[f64]) -> Tensor {
    Tensor::from_f64(F, shape, v)
}

fn vals(t: &Tensor) -> Vec<f64> {
    t.data.to_f64()
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
    fn unit(&mut self) -> f64 {
        (self.next() % 1_000_000) as f64 / 1_000_000.0
    }
}

fn bin(g: &mut Graph, op: Op, a: u32, b: u32) -> u32 {
    g.add(op, &[a, b]).unwrap()
}

fn un(g: &mut Graph, op: Op, a: u32) -> u32 {
    g.add(op, &[a]).unwrap()
}

fn sum_all(g: &mut Graph, x: u32) -> u32 {
    let axes = (0..g.shape(x).len()).collect();
    un(g, Op::Reduce { kind: ReduceKind::Sum, axes, keepdims: false }, x)
}

#[test]
fn broadcasting_and_kernels() {
    let mut g = Graph::new();
    let a = g.constant(t(&[2, 3], &[1., 2., 3., 4., 5., 6.]));
    let b = g.constant(t(&[3], &[10., 20., 30.]));
    let c = g.constant(t(&[2, 1], &[100., 200.]));
    let s = g.scalar(F, 2.0);
    let ab = bin(&mut g, Op::Add, a, b);
    let ac = bin(&mut g, Op::Mul, a, c);
    let sa = bin(&mut g, Op::Sub, s, a);
    let tr = un(&mut g, Op::Transpose(vec![1, 0]), a);
    let sm = un(&mut g, Op::Reduce { kind: ReduceKind::Sum, axes: vec![0], keepdims: false }, a);
    let mx = un(&mut g, Op::Reduce { kind: ReduceKind::Max, axes: vec![1], keepdims: true }, a);
    let sl = un(&mut g, Op::Slice { starts: vec![1, 1], stops: vec![2, 3] }, a);
    let pad = un(&mut g, Op::Pad { starts: vec![1], shape: vec![5] }, b);
    let mm = bin(&mut g, Op::MatMul, a, b);
    let ids = [ab, ac, sa, tr, sm, mx, sl, pad, mm];
    let r = eval(&g, &ids, &[]);
    assert_eq!(vals(&r[0]), [11., 22., 33., 14., 25., 36.]);
    assert_eq!(vals(&r[1]), [100., 200., 300., 800., 1000., 1200.]);
    assert_eq!(vals(&r[2]), [1., 0., -1., -2., -3., -4.]);
    assert_eq!(r[3].shape, [3, 2]);
    assert_eq!(vals(&r[3]), [1., 4., 2., 5., 3., 6.]);
    assert_eq!(vals(&r[4]), [5., 7., 9.]);
    assert_eq!(r[5].shape, [2, 1]);
    assert_eq!(vals(&r[5]), [3., 6.]);
    assert_eq!(vals(&r[6]), [5., 6.]);
    assert_eq!(vals(&r[7]), [0., 10., 20., 30., 0.]);
    assert_eq!(vals(&r[8]), [140., 320.]);
}

#[test]
fn static_errors_are_structured() {
    let mut g = Graph::new();
    let a = g.constant(t(&[2, 3], &[0.; 6]));
    let b = g.constant(t(&[2], &[0.; 2]));
    let e = g.add(Op::Add, &[a, b]).unwrap_err();
    assert_eq!(e["code"], "SHAPE_MISMATCH");
    assert_eq!(e["observation"]["lhs_shape"], serde_json_arr(&[2, 3]));
    assert_eq!(g.add(Op::MatMul, &[a, a]).unwrap_err()["code"], "SHAPE_MISMATCH");
    let f = g.constant(Tensor::from_f64(DType::F32, &[2], &[0.; 2]));
    assert_eq!(g.add(Op::Add, &[b, f]).unwrap_err()["code"], "DTYPE_MISMATCH");
    assert_eq!(grad(&mut g, a, &[a]).unwrap_err()["code"], "GRAD_NON_SCALAR");
}

fn serde_json_arr(v: &[usize]) -> serde_json::Value {
    serde_json::json!(v)
}

/// Every differentiable op is compared against f64 central differences.
#[test]
fn every_vjp_matches_finite_differences() {
    let x0 = t(&[2, 3], &[0.3, 0.7, 1.1, 1.3, 0.2, 0.9]);
    let w = t(&[3], &[0.5, 1.5, 0.25]);
    type Build = fn(&mut Graph, u32, u32) -> u32;
    let cases: &[(&str, Build)] = &[
        ("neg", |g, x, _| un(g, Op::Neg, x)),
        ("exp", |g, x, _| un(g, Op::Exp, x)),
        ("log", |g, x, _| un(g, Op::Log, x)),
        ("sqrt", |g, x, _| un(g, Op::Sqrt, x)),
        ("sin", |g, x, _| un(g, Op::Sin, x)),
        ("cos", |g, x, _| un(g, Op::Cos, x)),
        ("tanh", |g, x, _| un(g, Op::Tanh, x)),
        ("abs", |g, x, _| un(g, Op::Abs, x)),
        ("add_bcast", |g, x, w| bin(g, Op::Add, x, w)),
        ("sub_bcast", |g, x, w| bin(g, Op::Sub, w, x)),
        ("mul_bcast", |g, x, w| bin(g, Op::Mul, x, w)),
        ("div_bcast", |g, x, w| bin(g, Op::Div, x, w)),
        ("div_bcast_rev", |g, x, w| bin(g, Op::Div, w, x)),
        ("pow_const", |g, x, _| { let c = g.scalar(F, 3.0); bin(g, Op::Pow, x, c) }),
        ("pow_var", |g, x, w| bin(g, Op::Pow, x, w)),
        ("pow_exponent", |g, x, w| bin(g, Op::Pow, w, x)),
        ("maximum", |g, x, w| bin(g, Op::Max, x, w)),
        ("minimum", |g, x, w| bin(g, Op::Min, x, w)),
        ("where", |g, x, w| { let c = bin(g, Op::Gt, x, w); g.add(Op::Where, &[c, x, w]).unwrap() }),
        ("matmul_mv", |g, x, w| bin(g, Op::MatMul, x, w)),
        ("matmul_vm", |g, x, _| { let v = g.constant(t(&[2], &[0.5, -1.0])); bin(g, Op::MatMul, v, x) }),
        ("matmul_mm", |g, x, _| { let tx = un(g, Op::Transpose(vec![1, 0]), x); let m = bin(g, Op::MatMul, x, tx); bin(g, Op::Mul, m, m) }),
        ("reduce_axis", |g, x, _| { let s = un(g, Op::Reduce { kind: ReduceKind::Sum, axes: vec![1], keepdims: false }, x); bin(g, Op::Mul, s, s) }),
        ("reduce_max", |g, x, _| { let s = un(g, Op::Reduce { kind: ReduceKind::Max, axes: vec![0], keepdims: false }, x); bin(g, Op::Mul, s, s) }),
        ("reduce_min_all", |g, x, _| un(g, Op::Reduce { kind: ReduceKind::Min, axes: vec![0, 1], keepdims: false }, x)),
        ("reshape", |g, x, _| { let r = un(g, Op::Reshape(vec![3, 2]), x); bin(g, Op::Mul, r, r) }),
        ("transpose", |g, x, _| { let r = un(g, Op::Transpose(vec![1, 0]), x); bin(g, Op::Mul, r, r) }),
        ("broadcast_to", |g, x, _| { let r = un(g, Op::BroadcastTo(vec![4, 2, 3]), x); bin(g, Op::Mul, r, r) }),
        ("slice", |g, x, _| { let r = un(g, Op::Slice { starts: vec![0, 1], stops: vec![2, 3] }, x); bin(g, Op::Mul, r, r) }),
        ("pad", |g, x, _| { let r = un(g, Op::Pad { starts: vec![1, 0], shape: vec![4, 3] }, x); bin(g, Op::Mul, r, r) }),
        ("cast_roundtrip", |g, x, _| { let r = un(g, Op::Cast(DType::F32), x); let b = un(g, Op::Cast(DType::F64), r); bin(g, Op::Mul, b, b) }),
    ];
    for (name, build) in cases {
        let mut g = Graph::new();
        let (x, w_id) = (g.input(&x0.shape, F), g.input(&w.shape, F));
        let out = build(&mut g, x, w_id);
        let y = sum_all(&mut g, out);
        let grads = grad(&mut g, y, &[x, w_id]).unwrap();
        let inputs = [&x0, &w];
        let analytic = eval(&g, &grads, &inputs);
        for (which, base) in inputs.iter().enumerate() {
            if name == &"cast_roundtrip" {
                continue; // f32 rounding makes the numeric side meaningless here
            }
            let b = vals(base);
            for e in 0..b.len() {
                let probe = |d: f64| {
                    let mut v = b.clone();
                    v[e] += d;
                    let moved = t(&base.shape, &v);
                    let args: Vec<&Tensor> = (0..2).map(|k| if k == which { &moved } else { inputs[k] }).collect();
                    vals(&eval(&g, &[y], &args)[0])[0]
                };
                let h = 1e-6;
                let numeric = (probe(h) - probe(-h)) / (2.0 * h);
                let a = vals(&analytic[which])[e];
                assert!((a - numeric).abs() < 1e-5 * a.abs().max(1.0), "{name} input {which} element {e}: analytic {a} numeric {numeric}");
            }
        }
    }
}

#[test]
fn higher_order_gradient() {
    // d3/dx3 x^4 = 24x
    let mut g = Graph::new();
    let x = g.input(&[], F);
    let c = g.scalar(F, 4.0);
    let y = bin(&mut g, Op::Pow, x, c);
    let d1 = grad(&mut g, y, &[x]).unwrap()[0];
    let d2 = grad(&mut g, d1, &[x]).unwrap()[0];
    let d3 = grad(&mut g, d2, &[x]).unwrap()[0];
    let r = eval(&g, &[d1, d2, d3], &[&t(&[], &[2.0])]);
    assert_eq!(vals(&r[0]), [32.0]);
    assert_eq!(vals(&r[1]), [48.0]);
    assert_eq!(vals(&r[2]), [48.0]);
}

#[test]
fn stop_gradient_and_disconnected() {
    let mut g = Graph::new();
    let (x, z) = (g.input(&[], F), g.input(&[], F));
    let s = un(&mut g, Op::StopGradient, x);
    let y = bin(&mut g, Op::Mul, s, x);
    let gr = grad(&mut g, y, &[x, z]).unwrap();
    assert_eq!(vals(&eval(&g, &gr, &[&t(&[], &[3.0]), &t(&[], &[1.0])])[0]), [3.0]);
    assert_eq!(g.notes[0]["code"], "GRAD_DISCONNECTED");
}

#[test]
fn optimizer_simplifies_and_keeps_semantics() {
    let mut g = Graph::new();
    let x = g.input(&[3], F);
    let (zero, one, two, two_b) = (g.scalar(F, 0.0), g.scalar(F, 1.0), g.scalar(F, 2.0), g.scalar(F, 2.0));
    let a = bin(&mut g, Op::Add, x, zero);
    let b = bin(&mut g, Op::Mul, a, one);
    let four = bin(&mut g, Op::Add, two, two_b);
    let p = bin(&mut g, Op::Pow, b, two);
    let q = bin(&mut g, Op::Pow, b, two_b);
    let m = bin(&mut g, Op::Mul, p, four);
    let n = bin(&mut g, Op::Mul, q, four);
    let dead = un(&mut g, Op::Exp, x);
    let _ = dead;
    let out = bin(&mut g, Op::Add, m, n);
    let (o, ids, st) = optimize(&g, &[out]);
    assert!(st.folded >= 1 && st.simplified >= 3 && st.cse >= 1, "{st:?}");
    // input, const 4, x*x, *4, add
    assert_eq!(st.nodes_after, 5, "{st:?}");
    let input = t(&[3], &[1., 2., 3.]);
    assert_eq!(vals(&eval(&o, &ids, &[&input])[0]), vals(&eval(&g, &[out], &[&input])[0]));
}

#[test]
fn optimizer_never_erases_nan_through_mul_zero() {
    let mut g = Graph::new();
    let x = g.input(&[], F);
    let zero = g.scalar(F, 0.0);
    let out = bin(&mut g, Op::Mul, x, zero);
    let (o, ids, _) = optimize(&g, &[out]);
    assert!(vals(&eval(&o, &ids, &[&t(&[], &[f64::NAN])])[0])[0].is_nan());
}

/// Fuzz: buffer reuse and optimization must never change results.
#[test]
fn random_graphs_agree_across_plans() {
    let mut r = Rng(0x1234_5678_9ABC_DEF1);
    for case in 0..400 {
        let mut g = Graph::new();
        let dim_a = 1 + r.below(4);
        let dim_b = 1 + r.below(4);
        let shape_pool = [vec![], vec![1], vec![dim_a], vec![dim_b], vec![dim_a, dim_b], vec![1, dim_b], vec![dim_a, 1]];
        let mut inputs = vec![];
        let mut vals_pool: Vec<u32> = vec![];
        for _ in 0..1 + r.below(3) {
            let s = shape_pool[r.below(shape_pool.len())].clone();
            let n: usize = s.iter().product();
            let v: Vec<f64> = (0..n).map(|_| 0.2 + r.unit()).collect();
            vals_pool.push(g.input(&s, F));
            inputs.push(t(&s, &v));
        }
        for _ in 0..1 + r.below(30) {
            let a = vals_pool[r.below(vals_pool.len())];
            let b = vals_pool[r.below(vals_pool.len())];
            let sa = g.shape(a).to_vec();
            let op_id = r.below(18);
            let res = match op_id {
                0 => g.add(Op::Add, &[a, b]),
                1 => g.add(Op::Sub, &[a, b]),
                2 => g.add(Op::Mul, &[a, b]),
                3 => g.add(Op::Div, &[a, b]),
                4 => g.add(Op::Max, &[a, b]),
                5 => g.add(Op::Neg, &[a]),
                6 => g.add(Op::Tanh, &[a]),
                7 => g.add(Op::Abs, &[a]),
                8 => g.add(Op::Sin, &[a]),
                9 => g.add(Op::Reshape(vec![sa.iter().product()]), &[a]),
                10 => g.add(Op::StopGradient, &[a]),
                11 => g.add(Op::Transpose((0..sa.len()).rev().collect()), &[a]),
                12 => g.add(Op::Reduce { kind: ReduceKind::Sum, axes: (0..sa.len()).filter(|_| r.next() % 2 == 0).collect(), keepdims: r.next() % 2 == 0 }, &[a]),
                13 => g.add(Op::MatMul, &[a, b]),
                14 => {
                    let c = bin_try(&mut g, a, b);
                    match c {
                        Some(c) => g.add(Op::Where, &[c, a, b]),
                        None => continue,
                    }
                }
                15 => {
                    let k = g.scalar(F, [0.0, 1.0, 2.0, 0.5][r.below(4)]);
                    g.add(Op::Add, &[a, k])
                }
                16 => {
                    let k = g.scalar(F, [1.0, 2.0, 0.5][r.below(3)]);
                    g.add(Op::Mul, &[a, k])
                }
                _ => g.add(Op::Slice { starts: vec![0; sa.len()], stops: sa.clone() }, &[a]),
            };
            if let Ok(id) = res {
                vals_pool.push(id);
            }
        }
        let n_out = 1 + r.below(3);
        let mut outs: Vec<u32> = (0..n_out).map(|_| vals_pool[r.below(vals_pool.len())]).collect();
        // Include a gradient when the last value allows one, to fuzz backward graphs too.
        let last = *vals_pool.last().unwrap();
        let y = sum_all(&mut g, last);
        let first_input = vals_pool[0];
        outs.extend(grad(&mut g, y, &[first_input]).unwrap());

        let args: Vec<&Tensor> = inputs.iter().collect();
        let reference = {
            let p = Plan::build(&g, &outs, false);
            let mut f = p.frame();
            for (k, x) in args.iter().enumerate() {
                p.load_input(&mut f, k, x);
            }
            p.exec(&mut f, &mut ());
            (0..outs.len()).map(|k| p.read_output(&f, k)).collect::<Vec<_>>()
        };
        let reused = eval(&g, &outs, &args);
        let (o, ids, _) = optimize(&g, &outs);
        let optimized = eval(&o, &ids, &args);
        for k in 0..outs.len() {
            let want = vals(&reference[k]);
            assert_eq!(reference[k].shape, reused[k].shape);
            assert_eq!(reference[k].shape, optimized[k].shape);
            for (j, &w) in want.iter().enumerate() {
                for (label, got) in [("reuse", vals(&reused[k])[j]), ("optimized", vals(&optimized[k])[j])] {
                    assert!((w - got).abs() <= 1e-9 * w.abs().max(1.0) || (w.is_nan() && got.is_nan()), "case {case} {label} output {k}[{j}]: {w} vs {got}");
                }
            }
        }
    }
}

fn run_plan(p: &Plan, inputs: &[&Tensor], n_out: usize) -> Vec<Tensor> {
    let mut f = p.frame();
    for (k, x) in inputs.iter().enumerate() {
        p.load_input(&mut f, k, x);
    }
    p.exec(&mut f, &mut ());
    (0..n_out).map(|k| p.read_output(&f, k)).collect()
}

#[test]
fn transposed_matmul_rewrite_covers_every_flag_combination() {
    let a = t(&[2, 3], &[1., 2., 3., 4., 5., 6.]);
    let b = t(&[2, 3], &[7., 8., 9., 10., 11., 12.]);
    let c = t(&[3, 3], &[1., 0., 2., 0., 3., 0., 4., 0., 5.]);
    let v = t(&[2], &[0.5, -2.0]);
    type Case = fn(&mut Graph, u32, u32, u32, u32) -> u32;
    let cases: Vec<Case> = vec![
        |g, a, b, _, _| { let x = un(g, Op::Transpose(vec![1, 0]), a); bin(g, Op::MatMul, x, b) },
        |g, a, b, _, _| { let y = un(g, Op::Transpose(vec![1, 0]), b); bin(g, Op::MatMul, a, y) },
        |g, a, _, c, _| { let x = un(g, Op::Transpose(vec![1, 0]), c); bin(g, Op::MatMul, a, x) },
        |g, a, _, c, _| { let x = un(g, Op::Transpose(vec![1, 0]), c); let y = un(g, Op::Transpose(vec![1, 0]), a); bin(g, Op::MatMul, x, y) },
        |g, a, _, _, v| { let x = un(g, Op::Transpose(vec![1, 0]), a); bin(g, Op::MatMul, x, v) },
    ];
    for (k, build) in cases.iter().enumerate() {
        let mut g = Graph::new();
        let (ia, ib, ic, iv) = (g.input(&[2, 3], F), g.input(&[2, 3], F), g.input(&[3, 3], F), g.input(&[2], F));
        let out = build(&mut g, ia, ib, ic, iv);
        let args = [&a, &b, &c, &v];
        let want = eval(&g, &[out], &args);
        let (o, ids, _) = optimize(&g, &[out]);
        let plan = Plan::build(&o, &ids, true);
        assert!(plan.instrs().iter().all(|i| !matches!(i.op, Op::Transpose(_))), "case {k} kept a transpose copy");
        assert!(plan.instrs().iter().any(|i| matches!(i.op, Op::MatMulT { .. })), "case {k} was not rewritten");
        let got = eval(&o, &ids, &args);
        assert_eq!(want, got, "case {k}");
    }
    // A transpose with another consumer must survive the rewrite.
    let mut g = Graph::new();
    let x = g.input(&[2, 3], F);
    let tx = un(&mut g, Op::Transpose(vec![1, 0]), x);
    let m = bin(&mut g, Op::MatMul, tx, x);
    let (o, ids, _) = optimize(&g, &[m, tx]);
    assert_eq!(eval(&o, &ids, &[&a]), eval(&g, &[m, tx], &[&a]));
}

/// Fuzz with sizes that cross block (256) and task (4096) boundaries, forcing every group through threads.
#[test]
fn fused_plans_agree_with_reference_on_random_graphs() {
    let mut r = Rng(0xDEAD_BEEF_0BAD_F00D);
    let (mut groups, mut rows) = (0, 0);
    for case in 0..300 {
        let mut g = Graph::new();
        let dim_a = [1, 3, 40, 300][r.below(4)];
        let dim_b = [1, 7, 257, 600][r.below(4)];
        let shape_pool = [vec![], vec![dim_b], vec![dim_a, dim_b], vec![1, dim_b], vec![dim_a, 1]];
        let mut inputs = vec![];
        let mut pool: Vec<u32> = vec![];
        for s in [vec![dim_a, dim_b], shape_pool[r.below(5)].clone(), shape_pool[r.below(5)].clone()] {
            let n: usize = s.iter().product();
            let v: Vec<f64> = (0..n).map(|_| 0.2 + r.unit()).collect();
            pool.push(g.input(&s, F));
            inputs.push(t(&s, &v));
        }
        for _ in 0..2 + r.below(25) {
            // Biased to recent values so long chains form.
            let a = pool[pool.len() - 1 - r.below(pool.len().min(4))];
            let b = pool[r.below(pool.len())];
            let res = match r.below(14) {
                0 => g.add(Op::Add, &[a, b]),
                1 => g.add(Op::Sub, &[a, b]),
                2 => g.add(Op::Mul, &[a, b]),
                3 => g.add(Op::Div, &[a, b]),
                4 => g.add(Op::Max, &[a, b]),
                5 => g.add(Op::Tanh, &[a]),
                6 => g.add(Op::Exp, &[a]),
                7 => g.add(Op::Abs, &[a]),
                8 => g.add(Op::Sqrt, &[a]),
                9 => g.add(Op::Sin, &[a]),
                10 => g.add(Op::StopGradient, &[a]),
                11 => {
                    let n = g.nodes[a as usize].n();
                    g.add(Op::Reshape(vec![n]), &[a])
                }
                12 => {
                    let axes = (0..g.shape(a).len()).filter(|_| r.next() % 2 == 0).collect();
                    g.add(Op::Reduce { kind: ReduceKind::Sum, axes, keepdims: true }, &[a])
                }
                _ => match g.add(Op::Gt, &[a, b]) {
                    Ok(c) => g.add(Op::Where, &[c, a, b]),
                    Err(e) => Err(e),
                },
            };
            if let Ok(id) = res {
                pool.push(id);
            }
        }
        let last = *pool.last().unwrap();
        let y = sum_all(&mut g, last);
        let mut outs: Vec<u32> = (0..2).map(|_| pool[r.below(pool.len())]).collect();
        outs.extend(grad(&mut g, y, &[pool[0]]).unwrap());

        let args: Vec<&Tensor> = inputs.iter().collect();
        let reference = run_plan(&Plan::build(&g, &outs, false), &args, outs.len());
        let p = Plan::build_fused(&g, &outs, Fuse { parallel_ns: 0.0, metal: false, levels: false });
        for threads in [1, 4] {
            groups += p.fused().len();
            rows += p.fused().iter().flat_map(|f| &f.inputs).filter(|i| matches!(i.1, ArgKind::Row(_))).count();
            let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
            let got = pool.install(|| run_plan(&p, &args, outs.len()));
            for k in 0..outs.len() {
                assert_eq!(reference[k].shape, got[k].shape);
                for (j, (&w, &v)) in vals(&reference[k]).iter().zip(&vals(&got[k])).enumerate() {
                    // vForce tails may differ by an ulp, so this is a tolerance and not equality.
                    assert!(w == v || (w - v).abs() <= 1e-12 * w.abs().max(1.0) || (w.is_nan() && v.is_nan()), "case {case} threads {threads} output {k}[{j}]: {w} vs {v}");
                }
            }
        }
    }
    assert!(groups > 300 && rows > 50, "fuzz must really exercise fusion: {groups} groups, {rows} row inputs");
}

#[test]
fn groups_only_form_when_threads_pay_and_results_ignore_thread_count() {
    let mut small = Graph::new();
    let x = small.input(&[1000], DType::F32);
    let (a, b) = (un(&mut small, Op::Neg, x), un(&mut small, Op::Abs, x));
    let m = bin(&mut small, Op::Mul, a, b);
    assert!(Plan::build_fused(&small, &[m], Fuse::DEFAULT).fused().is_empty());

    let mut g = Graph::new();
    let x = g.input(&[1000, 64], DType::F32);
    let b = g.input(&[64], DType::F32);
    let a = bin(&mut g, Op::Add, x, b);
    let h = un(&mut g, Op::Tanh, a);
    let sq = bin(&mut g, Op::Mul, h, h);
    let e = un(&mut g, Op::Exp, sq);
    let p = Plan::build_fused(&g, &[e], Fuse { parallel_ns: 0.0, metal: false, levels: false });
    assert_eq!(p.fused().len(), 1);
    assert_eq!(p.fused()[0].ops.len(), 4);
    // Only the group's single output needs a buffer.
    assert_eq!(p.planned_bytes(), 64_000 * 4);
    assert_eq!(p.naive_bytes, 4 * 64_000 * 4);

    let xs: Vec<f64> = (0..64_000).map(|i| (i as f64 * 0.37).sin()).collect();
    let bs: Vec<f64> = (0..64).map(|i| i as f64 * 0.01).collect();
    let (ix, ib) = (Tensor::from_f64(DType::F32, &[1000, 64], &xs), Tensor::from_f64(DType::F32, &[64], &bs));
    let at = |threads: usize| {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
        pool.install(|| run_plan(&p, &[&ix, &ib], 1))
    };
    let one = at(1);
    assert_eq!(one, at(2));
    assert_eq!(one, at(7));
}

fn bin_try(g: &mut Graph, a: u32, b: u32) -> Option<u32> {
    g.add(Op::Gt, &[a, b]).ok()
}

#[test]
fn memory_plan_reuses_buffers_and_works_in_place() {
    let mut g = Graph::new();
    let x = g.input(&[1000], F);
    let mut v = x;
    for _ in 0..20 {
        v = un(&mut g, Op::Sin, v);
    }
    let p = Plan::build(&g, &[v], true);
    assert_eq!(p.naive_bytes, 20 * 8000);
    assert_eq!(p.planned_bytes(), 8000);
    assert_eq!(Plan::build(&g, &[v], false).planned_bytes(), 20 * 8000);
}

#[test]
fn vector_math_in_place_blocks_and_ieee_edges_match_std() {
    // 1500 elements crosses the 512-element block boundary of the in-place path.
    let v: Vec<f64> = (0..1500).map(|i| 0.001 + i as f64 * 0.01).collect();
    let input = t(&[1500], &v);
    type Row = (Op, fn(f64) -> f64);
    let rows: Vec<Row> = vec![(Op::Exp, f64::exp), (Op::Log, f64::ln), (Op::Sin, f64::sin), (Op::Cos, f64::cos), (Op::Tanh, f64::tanh)];
    for (op, reference) in rows {
        let mut g = Graph::new();
        let x = g.input(&[1500], F);
        let two = g.scalar(F, 0.5);
        let scaled = bin(&mut g, Op::Mul, x, two); // a private buffer, overwritten in place by `op`
        let out = un(&mut g, op.clone(), scaled);
        let plan = Plan::build(&g, &[out], true);
        assert_eq!(plan.instrs()[1].dst, plan.instrs()[1].args[0], "{op:?} must run in place");
        let got = vals(&eval(&g, &[out], &[&input])[0]);
        for (a, &b) in got.iter().zip(&v) {
            let want = reference(b * 0.5);
            assert!((a - want).abs() <= 1e-14 * want.abs().max(1.0), "{op:?}({b}): {a} vs {want}");
        }
    }
    let edges = t(&[4], &[-1.0, 0.0, f64::INFINITY, f64::NAN]);
    let mut g = Graph::new();
    let x = g.input(&[4], F);
    let l = un(&mut g, Op::Log, x);
    let r = vals(&eval(&g, &[l], &[&edges])[0]);
    assert!(r[0].is_nan() && r[1] == f64::NEG_INFINITY && r[2] == f64::INFINITY && r[3].is_nan(), "{r:?}");
}

fn compile_one(build: impl Fn(&mut Graph, u32) -> u32, input: Tensor, level: Level) -> rg::validate::Compiled {
    let mut g = Graph::new();
    g.cur_source = Some("model.py:42".into());
    let x = g.input(&input.shape, input.dtype());
    let o = build(&mut g, x);
    compile(g, vec![o], &[input], level).unwrap()
}

#[test]
fn design_example_exp_overflow_in_f32_only() {
    let f32_input = Tensor::from_f64(DType::F32, &[1], &[92.7]);
    let c = compile_one(|g, x| un(g, Op::Exp, x), f32_input, Level::Standard);
    let d = &c.report["diagnostics"][0];
    assert_eq!(c.report["status"], "validation_failed");
    assert_eq!(d["code"], "NUMERIC_OVERFLOW");
    assert_eq!(d["node"]["operation"], "exp");
    assert_eq!(d["node"]["source"], "model.py:42");
    assert_eq!(d["observation"]["dtype"], "float32");
    assert_eq!(d["observation"]["input_max"], 92.7);
    assert_eq!(d["observation"]["output_contains_inf"], true);
    assert_eq!(d["diagnosis"]["cause"], "exponential_overflow");
    assert_eq!(d["diagnosis"]["certainty"], "verified");
    // The evidence is real: the same value is fine in f64.
    let f64_input = t(&[1], &[92.7]);
    assert_eq!(compile_one(|g, x| un(g, Op::Exp, x), f64_input, Level::Standard).report["status"], "ok");
}

#[test]
fn diagnostics_classify_root_causes() {
    let cases: Vec<(&str, f64, &str, &str)> = vec![("log", -1.0, "DOMAIN_ERROR", "log_of_negative"), ("log", 0.0, "DOMAIN_ERROR", "log_of_zero"), ("sqrt", -1.0, "DOMAIN_ERROR", "sqrt_of_negative"), ("div", 0.0, "DIVIDE_BY_ZERO", "division_by_zero"), ("pow", -2.0, "DOMAIN_ERROR", "negative_base_fractional_exponent")];
    for (op, x, code, cause) in cases {
        let c = compile_one(
            |g, v| match op {
                "log" => un(g, Op::Log, v),
                "sqrt" => un(g, Op::Sqrt, v),
                "div" => {
                    let one = g.scalar(F, 1.0);
                    bin(g, Op::Div, one, v)
                }
                _ => {
                    let e = g.scalar(F, 0.5);
                    bin(g, Op::Pow, v, e)
                }
            },
            t(&[], &[x]),
            Level::Standard,
        );
        let d = &c.report["diagnostics"][0];
        assert_eq!(d["code"], code, "{op}");
        assert_eq!(d["diagnosis"]["cause"], cause, "{op}");
    }
}

#[test]
fn propagated_nodes_are_not_blamed() {
    // x -> log(x) [root] -> *2 -> +1: only the log node is a root.
    let c = compile_one(
        |g, x| {
            let l = un(g, Op::Log, x);
            let two = g.scalar(F, 2.0);
            let m = bin(g, Op::Mul, l, two);
            let e = un(g, Op::Exp, m);
            bin(g, Op::Add, e, two)
        },
        t(&[], &[-1.0]),
        Level::Standard,
    );
    assert_eq!(c.report["diagnostics"].as_array().unwrap().len(), 1);
    assert_eq!(c.report["diagnostics"][0]["node"]["operation"], "log");
    assert_eq!(c.report["checks"][2]["nonfinite"]["propagated_nodes"], 3);
}

#[test]
fn masked_nan_is_warning_and_gradient_nan_is_error_with_backward_role() {
    // where(x > 0, sqrt(x), 0): the classic trap. The value is fine, the gradient at 0 is not.
    let mut g = Graph::new();
    g.cur_source = Some("model.py:7".into());
    let x = g.input(&[], F);
    let zero = g.scalar(F, 0.0);
    let s = un(&mut g, Op::Sqrt, x);
    let c = bin(&mut g, Op::Gt, x, zero);
    let w = g.add(Op::Where, &[c, s, zero]).unwrap();
    let gr = grad(&mut g, w, &[x]).unwrap()[0];

    let value_only = compile(g.clone(), vec![w], &[t(&[], &[-1.0])], Level::Standard).unwrap();
    assert_eq!(value_only.report["status"], "warning");
    assert_eq!(value_only.report["diagnostics"][0]["masked"], true);
    assert_eq!(value_only.report["diagnostics"][0]["node"]["operation"], "sqrt");

    let with_grad = compile(g, vec![gr], &[t(&[], &[0.0])], Level::Standard).unwrap();
    assert_eq!(with_grad.report["status"], "validation_failed");
    let d = with_grad.report["diagnostics"].as_array().unwrap().iter().find(|d| d["severity"] == "error").unwrap();
    assert_eq!(d["code"], "DIVIDE_BY_ZERO");
    assert_eq!(d["node"]["role"], "backward");
    assert_eq!(d["node"]["forward_node"]["operation"], "sqrt");
    assert_eq!(d["node"]["forward_node"]["source"], "model.py:7");
}

#[test]
fn grad_check_passes_and_catches_a_wrong_gradient() {
    let build = |bad: bool| {
        let mut g = Graph::new();
        let x = g.input(&[3], DType::F32);
        let e = un(&mut g, Op::Exp, x);
        let y = sum_all(&mut g, e);
        let gr = grad(&mut g, y, &[x]).unwrap()[0];
        if bad {
            // Deliberately corrupt the record: claim d/dx = 2*grad.
            let two = g.scalar(DType::F32, 2.0);
            let wrong = bin(&mut g, Op::Mul, gr, two);
            g.grads[0] = GradRecord { y, wrt: x, grad: wrong };
        }
        let input = Tensor::from_f64(DType::F32, &[3], &[0.1, 0.2, 0.3]);
        compile(g, vec![y], &[input], Level::Thorough).unwrap()
    };
    let good = build(false);
    assert_eq!(good.report["status"], "ok", "{}", good.report);
    assert_eq!(good.report["checks"][4]["detail"]["elements_checked"], 3);

    let bad = build(true);
    assert_eq!(bad.report["status"], "validation_failed");
    let d = &bad.report["diagnostics"][0];
    assert_eq!(d["code"], "GRADIENT_MISMATCH");
    assert_eq!(d["diagnosis"]["certainty"], "verified");
}

#[test]
fn perturbation_finds_instability_a_clean_sample_hides() {
    // At exactly 1, sqrt(x - 1) is fine; a slightly smaller x is not.
    let c = compile_one(
        |g, x| {
            let one = g.scalar(F, 1.0);
            let d = bin(g, Op::Sub, x, one);
            un(g, Op::Sqrt, d)
        },
        t(&[1], &[1.0]),
        Level::Thorough,
    );
    let codes: Vec<_> = c.report["diagnostics"].as_array().unwrap().iter().map(|d| (d["code"].clone(), d["trigger"]["kind"].clone())).collect();
    assert!(codes.iter().any(|(code, trig)| code == "DOMAIN_ERROR" && trig == "input_perturbation"), "{codes:?}");
    assert_eq!(c.report["status"], "warning");
}

fn gd_program(lr: f64) -> (rg::validate::Compiled, RunSpec) {
    // state w, data x, y: minimize (w*x - y)^2 / 2
    let mut g = Graph::new();
    let (w, x, y) = (g.input(&[], F), g.input(&[], F), g.input(&[], F));
    let p = bin(&mut g, Op::Mul, w, x);
    let e = bin(&mut g, Op::Sub, p, y);
    let half = g.scalar(F, 0.5);
    let sq = bin(&mut g, Op::Mul, e, e);
    let loss = bin(&mut g, Op::Mul, sq, half);
    let gr = grad(&mut g, loss, &[w]).unwrap()[0];
    let rate = g.scalar(F, lr);
    let d = bin(&mut g, Op::Mul, gr, rate);
    let nw = bin(&mut g, Op::Sub, w, d);
    let inputs = [t(&[], &[0.0]), t(&[], &[2.0]), t(&[], &[6.0])];
    let c = compile(g, vec![nw, loss], &inputs, Level::Standard).unwrap();
    (c, RunSpec { state: vec![(0, 0)], data: vec![1, 2], history: vec![1] })
}

#[test]
fn run_trains_and_stops_on_nonfinite_with_exact_replay() {
    let batch = vec![vec![t(&[], &[2.0]), t(&[], &[6.0])]];
    let (c, spec) = gd_program(0.1);
    let out = c.run(&spec, &[t(&[], &[0.0])], &batch, 200);
    assert_eq!(out.steps, 200);
    assert!((vals(&out.state[0])[0] - 3.0).abs() < 1e-9);
    assert!(out.history[0][199] < 1e-15);
    assert_eq!(c.replay(&spec, &[t(&[], &[0.0])], &batch, 200), None);

    // A learning rate of 1000 diverges to Inf: the run stops and replay finds the same region.
    let (c, spec) = gd_program(1000.0);
    let out = c.run(&spec, &[t(&[], &[0.0])], &batch, 10_000);
    let stop = out.nonfinite_step.expect("must diverge");
    assert!(stop < 10_000);
    let (k, _, state) = c.replay(&spec, &[t(&[], &[0.0])], &batch, 10_000).unwrap();
    assert!(k <= stop);
    assert!(state[0].is_finite());
    let report = c.check(&[state[0].clone(), batch[0][0].clone(), batch[0][1].clone()]);
    assert_eq!(report["status"], "validation_failed", "{report}");
}

#[test]
fn run_matches_repeated_calls_exactly_and_cycles_batches() {
    let (c, spec) = gd_program(0.01);
    let batches = vec![vec![t(&[], &[1.0]), t(&[], &[2.0])], vec![t(&[], &[3.0]), t(&[], &[1.0])], vec![t(&[], &[-2.0]), t(&[], &[0.5])]];
    let out = c.run(&spec, &[t(&[], &[0.5])], &batches, 10);
    let mut w = t(&[], &[0.5]);
    for k in 0..10 {
        let b = &batches[k % 3];
        let r = c.call(&[w.clone(), b[0].clone(), b[1].clone()]);
        assert_eq!(vals(&r[1])[0], out.history[0][k], "history at step {k}");
        w = r[0].clone();
    }
    assert_eq!(out.state[0], w);
}

#[test]
fn run_handles_state_passthrough_and_swap_between_inputs() {
    // new_a = b, new_b = a: both state outputs are pinned input slots, so copies must not clobber.
    let mut g = Graph::new();
    let (a, b) = (g.input(&[2], F), g.input(&[2], F));
    let inputs = [t(&[2], &[1., 2.]), t(&[2], &[3., 4.])];
    let c = compile(g, vec![b, a], &inputs, Level::Basic).unwrap();
    let spec = RunSpec { state: vec![(0, 0), (1, 1)], data: vec![], history: vec![] };
    let out = c.run(&spec, &inputs, &[], 3);
    assert_eq!(vals(&out.state[0]), [3., 4.]);
    assert_eq!(vals(&out.state[1]), [1., 2.]);
}

#[test]
fn input_mismatch_is_reported_not_recompiled() {
    let (c, spec) = gd_program(0.1);
    let e = c.check_inputs(&[t(&[2], &[0., 0.]), t(&[], &[1.]), t(&[], &[1.])]).unwrap_err();
    assert_eq!(e["code"], "INPUT_SIGNATURE_MISMATCH");
    assert_eq!(e["observation"]["recompile_required"], true);
    assert!(c.validate_spec(&spec).is_ok());

    let mut g = Graph::new();
    let x = g.input(&[2], F);
    let s = sum_all(&mut g, x);
    let c = compile(g, vec![s], &[t(&[2], &[1., 2.])], Level::Basic).unwrap();
    let e = c.validate_spec(&RunSpec { state: vec![(0, 0)], data: vec![], history: vec![] }).unwrap_err();
    assert_eq!(e["code"], "STATE_CONTRACT_VIOLATION");
}

#[test]
fn f32_precision_is_real() {
    let mut g = Graph::new();
    let x = g.input(&[2], DType::F32);
    let o = bin(&mut g, Op::Add, x, x);
    let r = eval(&g, &[o], &[&Tensor::from_f64(DType::F32, &[2], &[0.1, 0.2])]);
    assert!(matches!(r[0].data, Buf::F32(_)));
}

fn ints(shape: &[usize], v: &[i32]) -> Tensor {
    Tensor { shape: shape.to_vec(), data: Buf::I32(v.to_vec()) }
}

#[test]
fn take_concat_argmax_and_int_arithmetic() {
    let mut g = Graph::new();
    let x = g.constant(t(&[3, 2], &[1., 2., 3., 4., 5., 6.]));
    let idx = g.constant(ints(&[2, 2], &[2, 0, 1, 1]));
    let a0 = g.add(Op::Take { axis: 0 }, &[x, idx]).unwrap();
    let col = g.constant(ints(&[1], &[1]));
    let a1 = g.add(Op::Take { axis: 1 }, &[x, col]).unwrap();
    let along = g.constant(ints(&[3, 1], &[1, 0, 1]));
    let ta = g.add(Op::TakeAlong, &[x, along]).unwrap();
    let c = g.add(Op::Concat { axis: 1 }, &[x, x]).unwrap();
    let m = g.add(Op::Argmax, &[x]).unwrap();
    let one = g.constant(ints(&[], &[1]));
    let shifted = g.add(Op::Add, &[col, one]).unwrap();
    let r = eval(&g, &[a0, a1, ta, c, m, shifted], &[]);
    assert_eq!(r[0].shape, [2, 2, 2]);
    assert_eq!(vals(&r[0]), [5., 6., 1., 2., 3., 4., 3., 4.]);
    assert_eq!(vals(&r[1]), [2., 4., 6.]);
    assert_eq!(vals(&r[2]), [2., 3., 6.]);
    assert_eq!(vals(&r[3]), [1., 2., 1., 2., 3., 4., 3., 4., 5., 6., 5., 6.]);
    assert_eq!(r[4], ints(&[3], &[1, 1, 1]));
    assert_eq!(r[5], ints(&[1], &[2]));
    assert_eq!(g.add(Op::Exp, &[col]).unwrap_err()["code"], "UNSUPPORTED_DTYPE");
    assert_eq!(g.add(Op::Take { axis: 0 }, &[x, x]).unwrap_err()["code"], "DTYPE_MISMATCH");
}

/// Same finite-difference harness as the elementwise ops, for what a language model needs.
#[test]
fn new_op_gradients_match_finite_differences() {
    let x0 = t(&[2, 3, 4], &(0..24).map(|i| 0.1 + (i as f64 * 0.7).sin() * 0.4 + 0.5).collect::<Vec<_>>());
    type Build = fn(&mut Graph, u32) -> u32;
    let cases: &[(&str, Build)] = &[
        ("take_embedding", |g, x| { let i = g.constant(ints(&[3, 2], &[1, 0, 1, 1, 0, 0])); let r = g.add(Op::Take { axis: 0 }, &[x, i]).unwrap(); bin(g, Op::Mul, r, r) }),
        ("take_axis2", |g, x| { let i = g.constant(ints(&[2], &[3, 3])); let r = g.add(Op::Take { axis: 2 }, &[x, i]).unwrap(); bin(g, Op::Mul, r, r) }),
        ("take_along", |g, x| { let i = g.constant(ints(&[2, 3, 2], &[0, 3, 1, 1, 2, 0, 3, 3, 0, 1, 2, 2])); let r = g.add(Op::TakeAlong, &[x, i]).unwrap(); un(g, Op::Log, r) }),
        ("concat", |g, x| { let e = un(g, Op::Exp, x); let r = g.add(Op::Concat { axis: 1 }, &[x, e, x]).unwrap(); bin(g, Op::Mul, r, r) }),
        ("batched_matmul", |g, x| { let tx = un(g, Op::Transpose(vec![0, 2, 1]), x); let m = bin(g, Op::MatMul, x, tx); bin(g, Op::Mul, m, m) }),
        ("batched_broadcast", |g, x| { let w = g.constant(t(&[4, 2], &[0.5, 1., -1., 2., 3., 0.25, 1., 1.])); let m = bin(g, Op::MatMul, x, w); bin(g, Op::Mul, m, m) }),
        ("rhs_batched", |g, x| { let w = g.constant(t(&[5, 3], &(0..15).map(|i| i as f64 * 0.1).collect::<Vec<_>>())); let m = bin(g, Op::MatMul, w, x); bin(g, Op::Mul, m, m) }),
        ("vector_rhs", |g, x| { let v = g.constant(t(&[4], &[1., 2., 3., 4.])); let m = bin(g, Op::MatMul, x, v); bin(g, Op::Mul, m, m) }),
        ("vector_lhs", |g, x| { let v = g.constant(t(&[3], &[1., 2., 3.])); let m = bin(g, Op::MatMul, v, x); bin(g, Op::Mul, m, m) }),
        ("floor_blocks", |g, x| { let f = un(g, Op::Floor, x); bin(g, Op::Mul, f, x) }),
    ];
    for (name, build) in cases {
        let mut g = Graph::new();
        let x = g.input(&x0.shape, F);
        let out = build(&mut g, x);
        let y = sum_all(&mut g, out);
        let gx = grad(&mut g, y, &[x]).unwrap()[0];
        let analytic = vals(&eval(&g, &[gx], &[&x0])[0]);
        let (opt, ids, _) = optimize(&g, &[gx]);
        let fast = Plan::build_fused(&opt, &ids, Fuse { parallel_ns: 0.0, metal: false, levels: false });
        let got = vals(&run_plan(&fast, &[&x0], 1)[0]);
        let b = vals(&x0);
        for e in 0..b.len() {
            let probe = |d: f64| {
                let mut v = b.clone();
                v[e] += d;
                vals(&eval(&g, &[y], &[&t(&x0.shape, &v)])[0])[0]
            };
            let h = 1e-6;
            let numeric = (probe(h) - probe(-h)) / (2.0 * h);
            assert!((analytic[e] - numeric).abs() < 1e-4 * analytic[e].abs().max(1.0), "{name}[{e}]: analytic {} numeric {numeric}", analytic[e]);
            assert!((analytic[e] - got[e]).abs() < 1e-9 * analytic[e].abs().max(1.0), "{name}[{e}] optimized plan: {} vs {}", got[e], analytic[e]);
        }
    }
}

#[test]
fn second_order_through_take_and_concat() {
    // f = sum(x[idx]^3) with repeats; f'' along the diagonal is 6*x*count.
    let x0 = t(&[3], &[1., 2., 3.]);
    let mut g = Graph::new();
    let x = g.input(&[3], F);
    let i = g.constant(ints(&[4], &[0, 2, 2, 1]));
    let tk = g.add(Op::Take { axis: 0 }, &[x, i]).unwrap();
    let c = g.scalar(F, 3.0);
    let p = bin(&mut g, Op::Pow, tk, c);
    let y = sum_all(&mut g, p);
    let g1 = grad(&mut g, y, &[x]).unwrap()[0];
    let s = sum_all(&mut g, g1);
    let g2 = grad(&mut g, s, &[x]).unwrap()[0];
    assert_eq!(vals(&eval(&g, &[g2], &[&x0])[0]), [6., 12., 36.]);
}

#[test]
fn random_streams_are_deterministic_uniform_and_normal() {
    let key = ints(&[2], &[7, 11]);
    let mut g = Graph::new();
    let k = g.input(&[2], DType::I32);
    let u = g.add(Op::RandUniform { shape: vec![100_000], dtype: DType::F32 }, &[k]).unwrap();
    let n = g.add(Op::RandNormal { shape: vec![100_000], dtype: F }, &[k]).unwrap();
    let keys = g.add(Op::KeySplit { n: 3 }, &[k]).unwrap();
    let serial = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap().install(|| eval(&g, &[u, n, keys], &[&key]));
    assert_eq!(serial, eval(&g, &[u, n, keys], &[&key]), "stream must not depend on threads");
    let (us, ns) = (vals(&serial[0]), vals(&serial[1]));
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let var = |v: &[f64]| { let m = mean(v); v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / v.len() as f64 };
    assert!(us.iter().all(|&x| (0.0..1.0).contains(&x)));
    assert!((mean(&us) - 0.5).abs() < 0.005 && (var(&us) - 1.0 / 12.0).abs() < 0.002);
    assert!(mean(&ns).abs() < 0.02 && (var(&ns) - 1.0).abs() < 0.02);
    let tail = ns.iter().filter(|x| x.abs() > 1.96).count() as f64 / ns.len() as f64;
    assert!((tail - 0.05).abs() < 0.003, "tail mass {tail}");
    let split = vals(&serial[2]);
    assert_eq!(serial[2].shape, [3, 2]);
    assert!(split[0] != split[2] && split[2] != split[4]);
    let other = eval(&g, &[u], &[&ints(&[2], &[7, 12])]);
    assert_ne!(other[0], serial[0]);
}

#[test]
fn custom_vjp_rule_replaces_the_body_and_composes() {
    // Identity forward, gradient scaled by 3: x -> custom(x) * x, y = sum(custom(x)*custom(x))
    let mut g = Graph::new();
    let x = g.input(&[2], F);
    let ph = g.placeholder(&[2], F);
    let three = g.scalar(F, 3.0);
    let rule = bin(&mut g, Op::Mul, ph, three);
    let c = g.add(Op::Custom { placeholder: ph, rules: vec![(x, rule)] }, &[x]).unwrap();
    let sq = bin(&mut g, Op::Mul, c, c);
    let y = sum_all(&mut g, sq);
    let gx = grad(&mut g, y, &[x]).unwrap()[0];
    let x0 = t(&[2], &[1., 2.]);
    assert_eq!(vals(&eval(&g, &[y], &[&x0])[0]), [5.]);
    assert_eq!(vals(&eval(&g, &[gx], &[&x0])[0]), [6., 12.]); // 3 * 2x
    let (o, ids, _) = optimize(&g, &[gx, y]);
    assert_eq!(vals(&eval(&o, &ids, &[&x0])[0]), [6., 12.]);
    let c = compile(g, vec![y], &[x0], Level::Thorough).unwrap();
    let d = c.report["diagnostics"].as_array().unwrap().iter().find(|d| d["code"] == "GRADIENT_MISMATCH").expect("the wrong rule must be flagged");
    assert_eq!(d["diagnosis"]["certainty"], "suspected", "a custom rule may be intentional");
}

#[test]
fn out_of_range_index_is_clamped_at_runtime_and_a_hard_validation_error() {
    let mut g = Graph::new();
    let x = g.input(&[3], F);
    let i = g.constant(ints(&[2], &[5, -1]));
    let o = g.add(Op::Take { axis: 0 }, &[x, i]).unwrap();
    let x0 = t(&[3], &[1., 2., 3.]);
    assert_eq!(vals(&eval(&g, &[o], &[&x0])[0]), [3., 1.]);
    let c = compile(g, vec![o], &[x0], Level::Standard).unwrap();
    assert_eq!(c.report["status"], "validation_failed");
    assert_eq!(c.report["diagnostics"][0]["code"], "INDEX_OUT_OF_RANGE");
    assert_eq!(c.report["diagnostics"][0]["observation"]["index_max"], 5.0);
}

#[test]
fn thorough_validation_skips_integers_and_passes_on_an_embedding_model() {
    let mut g = Graph::new();
    let e = g.input(&[4, 3], DType::F32);
    let idx = g.input(&[2, 2], DType::I32);
    let key = g.input(&[2], DType::I32);
    let v = g.add(Op::Take { axis: 0 }, &[e, idx]).unwrap();
    let noise = g.add(Op::RandNormal { shape: vec![2, 2, 3], dtype: DType::F32 }, &[key]).unwrap();
    let a = bin(&mut g, Op::Add, v, noise);
    let sq = bin(&mut g, Op::Mul, a, a);
    let y = sum_all(&mut g, sq);
    let _ = grad(&mut g, y, &[e]).unwrap();
    let inputs = [Tensor::from_f64(DType::F32, &[4, 3], &(0..12).map(|i| i as f64 * 0.1).collect::<Vec<_>>()), ints(&[2, 2], &[0, 3, 1, 1]), ints(&[2], &[1, 2])];
    let c = compile(g, vec![y], &inputs, Level::Thorough).unwrap();
    assert_eq!(c.report["status"], "ok", "{}", c.report);
    assert_eq!(c.check(&inputs)["status"], "ok");
    assert_eq!(c.report["checks"][4]["detail"]["elements_checked"], 6);
    assert_eq!(c.call(&inputs)[0], c.call(&inputs)[0], "RNG with a fixed key is a pure function");
}

#[test]
fn rounding_amplified_by_normalization_is_not_reported_as_a_miscompile() {
    // Adam-like normalization of a near-cancelled value: huge amplification, no bug.
    let mut g = Graph::new();
    let x = g.input(&[2], DType::F32);
    let m = bin(&mut g, Op::Sub, x, x);
    let eps = g.scalar(DType::F32, 1e-8);
    let m2 = bin(&mut g, Op::Add, m, eps);
    let n = bin(&mut g, Op::Mul, m2, m2);
    let root = un(&mut g, Op::Sqrt, n);
    let r = bin(&mut g, Op::Add, root, eps);
    let o = bin(&mut g, Op::Div, m2, r);
    let input = Tensor::from_f64(DType::F32, &[2], &[0.1, 0.2]);
    let c = compile(g, vec![o], &[input], Level::Standard).unwrap();
    assert_eq!(c.report["status"], "ok", "{}", c.report);
    assert_eq!(c.report["checks"][3]["name"], "optimizer_equivalence");
    assert_eq!(c.report["checks"][3]["detail"]["precision"], "float64");
}

#[test]
fn parallel_structural_kernels_match_serial_and_ignore_thread_count() {
    let (a, b, c) = (128, 100, 100); // 1.28M elements, above every parallel threshold in play
    let n = a * b * c;
    let xs: Vec<f64> = (0..n).map(|i| ((i as f64) * 0.37).sin()).collect();
    let x0 = Tensor::from_f64(DType::F32, &[a, b, c], &xs);
    let mut g = Graph::new();
    let x = g.input(&[a, b, c], DType::F32);
    let t1 = un(&mut g, Op::Transpose(vec![2, 0, 1]), x);
    let t2 = un(&mut g, Op::Transpose(vec![1, 0, 2]), x);
    let mx = un(&mut g, Op::Reduce { kind: ReduceKind::Max, axes: vec![2], keepdims: true }, x);
    let sm = un(&mut g, Op::Reduce { kind: ReduceKind::Sum, axes: vec![2], keepdims: false }, x);
    let lead = un(&mut g, Op::Reduce { kind: ReduceKind::Sum, axes: vec![0, 1], keepdims: true }, x);
    let lmin = un(&mut g, Op::Reduce { kind: ReduceKind::Min, axes: vec![0, 1], keepdims: false }, x);
    let outs = [t1, t2, mx, sm, lead, lmin];
    let at = |threads: usize| rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap().install(|| eval(&g, &outs, &[&x0]));
    let one = at(1);
    assert_eq!(one, at(3));
    assert_eq!(one, at(8));
    // Reference by hand, in f64.
    for ci in [0, 7, c - 1] {
        let want: f64 = (0..a * b).map(|r| xs[r * c + ci]).sum();
        assert!((vals(&one[4])[ci] - want).abs() < 1e-2, "{} vs {want}", vals(&one[4])[ci]);
        let want = (0..a * b).map(|r| xs[r * c + ci] as f32).fold(f32::INFINITY, f32::min) as f64;
        assert_eq!(vals(&one[5])[ci], want);
    }
    let row = 5 * b + 3;
    let want = xs[row * c..(row + 1) * c].iter().map(|&v| v as f32).fold(f32::NEG_INFINITY, f32::max) as f64;
    assert_eq!(vals(&one[2])[row], want);
    assert_eq!(one[0].shape, [c, a, b]);
    assert_eq!(vals(&one[0])[a * b + 2], (xs[2 * c + 1] as f32) as f64); // out[1][0][2] = x[0][2][1]
}

#[test]
fn rematerialization_trades_storage_for_recomputation_and_keeps_results() {
    // A GELU-like chain: its intermediates are all used by the backward pass.
    let build = || {
        let mut g = Graph::new();
        let x = g.input(&[300, 200], F);
        let w = g.input(&[200, 200], F);
        let (c, one) = (g.scalar(F, 0.05), g.scalar(F, 1.0));
        // Stacked layers: without recomputation all their chains are alive at the end of the
        // forward pass; with it, each chain exists only while its own layer is differentiated.
        let mut y0 = x;
        for _ in 0..4 {
            let m = bin(&mut g, Op::MatMul, y0, w);
            let m2 = bin(&mut g, Op::Mul, m, m);
            let m3 = bin(&mut g, Op::Mul, m2, m);
            let cub = bin(&mut g, Op::Mul, m3, c);
            let s = bin(&mut g, Op::Add, m, cub);
            let t_ = un(&mut g, Op::Tanh, s);
            let o = bin(&mut g, Op::Add, t_, one);
            y0 = bin(&mut g, Op::Mul, m, o);
        }
        let y = sum_all(&mut g, y0);
        let gs = grad(&mut g, y, &[w]).unwrap();
        (g, gs)
    };
    let (g, gs) = build();
    let xs = t(&[300, 200], &(0..60000).map(|i| (i as f64 * 0.1).sin() * 0.1).collect::<Vec<_>>());
    let ws = t(&[200, 200], &(0..40000).map(|i| (i as f64 * 0.3).cos() * 0.01).collect::<Vec<_>>());
    let (with, ids_with, st) = rg::optimize::optimize_with(&g, &gs, 1000);
    let (without, ids_without, st0) = rg::optimize::optimize_with(&g, &gs, usize::MAX);
    assert!(st.rematerialized >= 4, "{st:?}");
    assert_eq!(st0.rematerialized, 0);
    let (p1, p0) = (Plan::build(&with, &ids_with, true), Plan::build(&without, &ids_without, true));
    assert!(p1.peak_live_bytes * 3 < p0.peak_live_bytes * 2, "peak {} vs {}", p1.peak_live_bytes, p0.peak_live_bytes);
    let (a, b) = (eval(&with, &ids_with, &[&xs, &ws]), eval(&g, &gs, &[&xs, &ws]));
    for (u, v) in vals(&a[0]).iter().zip(&vals(&b[0])) {
        assert!((u - v).abs() <= 1e-10 * v.abs().max(1.0), "{u} vs {v}");
    }
}

/// Same fuzz idea as above, with recomputation forced on for tiny values.
#[test]
fn rematerialized_random_graphs_agree_with_reference() {
    let mut r = Rng(0xABAD_1DEA_F00D_0001);
    let mut total = 0;
    for case in 0..300 {
        let mut g = Graph::new();
        let (a, b) = (1 + r.below(4), 1 + r.below(4));
        let shapes = [vec![], vec![b], vec![a, b], vec![a, 1]];
        let mut inputs = vec![];
        let mut pool: Vec<u32> = vec![];
        for s in [vec![a, b], shapes[r.below(4)].clone()] {
            let n: usize = s.iter().product();
            pool.push(g.input(&s, F));
            inputs.push(t(&s, &(0..n).map(|_| 0.2 + r.unit()).collect::<Vec<_>>()));
        }
        for _ in 0..2 + r.below(25) {
            let x = pool[pool.len() - 1 - r.below(pool.len().min(3))];
            let y = pool[r.below(pool.len())];
            let res = match r.below(10) {
                0 => g.add(Op::Add, &[x, y]),
                1 => g.add(Op::Sub, &[x, y]),
                2 => g.add(Op::Mul, &[x, y]),
                3 => g.add(Op::Div, &[x, y]),
                4 => g.add(Op::Max, &[x, y]),
                5 => g.add(Op::Tanh, &[x]),
                6 => g.add(Op::Exp, &[x]),
                7 => g.add(Op::StopGradient, &[x]),
                8 => g.add(Op::MatMul, &[x, y]),
                _ => g.add(Op::Gt, &[x, y]).and_then(|c| g.add(Op::Where, &[c, x, y])),
            };
            if let Ok(id) = res {
                pool.push(id);
            }
        }
        let last = *pool.last().unwrap();
        let y = sum_all(&mut g, last);
        let mut outs: Vec<u32> = (0..2).map(|_| pool[r.below(pool.len())]).collect();
        outs.extend(grad(&mut g, y, &[pool[0]]).unwrap());
        let args: Vec<&Tensor> = inputs.iter().collect();
        let (o, ids, st) = rg::optimize::optimize_with(&g, &outs, 0);
        total += st.rematerialized;
        let want = run_plan(&Plan::build(&g, &outs, false), &args, outs.len());
        let got = run_plan(&Plan::build_fused(&o, &ids, Fuse::DEFAULT), &args, outs.len());
        for k in 0..outs.len() {
            for (j, (&w, &v)) in vals(&want[k]).iter().zip(&vals(&got[k])).enumerate() {
                assert!(w == v || (w - v).abs() <= 1e-10 * w.abs().max(1.0) || (w.is_nan() && v.is_nan()), "case {case} output {k}[{j}]: {w} vs {v}");
            }
        }
    }
    assert!(total > 300, "fuzz must exercise recomputation: {total}");
}

#[test]
fn recycled_cpu_frames_give_identical_runs() {
    let (c, spec) = gd_program(0.01);
    let batch = vec![vec![t(&[], &[2.0]), t(&[], &[6.0])]];
    let first = c.run(&spec, &[t(&[], &[0.5])], &batch, 20);
    for _ in 0..3 {
        let again = c.run(&spec, &[t(&[], &[0.5])], &batch, 20);
        assert_eq!(again.state, first.state);
        assert_eq!(again.history, first.history);
    }
    let other = c.run(&spec, &[t(&[], &[1.5])], &[vec![t(&[], &[1.0]), t(&[], &[1.0])]], 5);
    assert_ne!(other.state, first.state);
}
