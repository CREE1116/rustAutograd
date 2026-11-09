use crate::graph::{diag, Diag, GradRecord, Graph, Op, ReduceKind};
use crate::tensor::{DType, Tensor};
use std::collections::HashMap;
use serde_json::json;

/// Sums `ct` back down to `target` after a broadcast.
fn unbroadcast(g: &mut Graph, ct: u32, target: &[usize]) -> Result<u32, Diag> {
    let shape = g.shape(ct).to_vec();
    if shape == target {
        return Ok(ct);
    }
    let lead = shape.len() - target.len();
    let axes: Vec<usize> = (0..shape.len()).filter(|&i| i < lead || (target[i - lead] == 1 && shape[i] != 1)).collect();
    let summed = g.add(Op::Reduce { kind: ReduceKind::Sum, axes, keepdims: true }, &[ct])?;
    g.add(Op::Reshape(target.to_vec()), &[summed])
}

fn blocks_gradient(op: &Op) -> bool {
    matches!(op, Op::StopGradient | Op::Sign | Op::Gt | Op::Ge | Op::Floor | Op::Argmax | Op::RandUniform { .. } | Op::RandNormal { .. } | Op::KeySplit { .. } | Op::Placeholder)
}

/// Transposes the last two axes.
fn swap(g: &mut Graph, v: u32) -> Result<u32, Diag> {
    let r = g.shape(v).len();
    let mut p: Vec<usize> = (0..r).collect();
    p.swap(r - 2, r - 1);
    g.add(Op::Transpose(p), &[v])
}

/// Appends d(y)/d(wrts) to the graph. Backward nodes are ordinary IR, so grad-of-grad needs nothing extra.
pub fn grad(g: &mut Graph, y: u32, wrts: &[u32]) -> Result<Vec<u32>, Diag> {
    if g.nodes[y as usize].n() != 1 {
        let obs = json!({ "output_shape": g.shape(y), "required": "single element" });
        return Err(diag("GRAD_NON_SCALAR", "error", g.node_json(y), obs, "gradient_of_non_scalar_output", "verified"));
    }
    if g.dtype(y) == DType::I32 || wrts.iter().any(|&w| g.dtype(w) == DType::I32) {
        let obs = json!({ "output_dtype": g.dtype(y).name(), "wrt_dtypes": wrts.iter().map(|&w| g.dtype(w).name()).collect::<Vec<_>>() });
        return Err(diag("GRAD_OF_INTEGER", "error", g.node_json(y), obs, "integers_are_not_differentiable", "verified"));
    }
    let end = y as usize;
    let saved = (g.cur_source.clone(), g.cur_origin);

    let mut rel = vec![false; end + 1];
    for i in 0..=end {
        let n = &g.nodes[i];
        let flows = match &n.op {
            // The rule, not the traced body, defines how gradients reach the inputs.
            Op::Custom { rules, .. } => rules.iter().any(|&(t, _)| rel[t as usize]),
            op => !blocks_gradient(op) && n.inputs.iter().any(|&x| rel[x as usize]),
        };
        rel[i] = n.dtype != DType::I32 && (wrts.contains(&(i as u32)) || flows);
    }

    let mut cot: Vec<Option<u32>> = vec![None; end + 1];
    if rel[end] {
        cot[end] = Some(g.constant(Tensor::full(g.dtype(y), &g.shape(y).to_vec(), 1.0)));
    }
    for i in (0..=end).rev() {
        let Some(ct) = cot[i] else { continue };
        g.cur_source = g.nodes[i].source.clone();
        g.cur_origin = Some(i as u32);
        for (input, c) in vjp(g, i, ct, &rel)? {
            cot[input as usize] = Some(match cot[input as usize] {
                Some(prev) => g.add(Op::Add, &[prev, c])?,
                None => c,
            });
        }
    }

    (g.cur_source, g.cur_origin) = saved;
    let mut out = vec![];
    for &w in wrts {
        let found = cot.get(w as usize).copied().flatten();
        let id = match found {
            Some(c) => c,
            None => {
                let obs = json!({ "output_node": y, "wrt_node": w });
                g.notes.push(diag("GRAD_DISCONNECTED", "warning", g.node_json(w), obs, "output_does_not_depend_on_input", "verified"));
                g.constant(Tensor::full(g.dtype(w), &g.shape(w).to_vec(), 0.0))
            }
        };
        g.grads.push(GradRecord { y, wrt: w, grad: id });
        out.push(id);
    }
    Ok(out)
}

fn vjp(g: &mut Graph, i: usize, ct: u32, rel: &[bool]) -> Result<Vec<(u32, u32)>, Diag> {
    use Op::*;
    let node = g.nodes[i].clone();
    let me = i as u32;
    let d = node.dtype;
    let x = node.inputs.first().copied().unwrap_or(0);
    let y = node.inputs.get(1).copied().unwrap_or(0);
    let mut out: Vec<(u32, u32)> = vec![];

    // Unary rules: dy/dx times the cotangent.
    let unary_factor: Option<u32> = match &node.op {
        Exp => Some(me),
        Cos => {
            let s = g.add(Sin, &[x])?;
            Some(g.add(Neg, &[s])?)
        }
        Sin => Some(g.add(Cos, &[x])?),
        Tanh => {
            let (one, sq) = (g.scalar(d, 1.0), g.add(Mul, &[me, me])?);
            Some(g.add(Sub, &[one, sq])?)
        }
        Abs => Some(g.add(Sign, &[x])?),
        _ => None,
    };
    if let Some(f) = unary_factor {
        out.push((x, g.add(Mul, &[ct, f])?));
        return Ok(out);
    }

    match &node.op {
        Input(_) | Const(_) | StopGradient | Sign | Gt | Ge => {}
        Neg => out.push((x, g.add(Neg, &[ct])?)),
        Log => out.push((x, g.add(Div, &[ct, x])?)),
        Sqrt => {
            let two = g.scalar(d, 2.0);
            let denom = g.add(Mul, &[two, me])?;
            out.push((x, g.add(Div, &[ct, denom])?));
        }
        Add | Sub | Mul | Div | Pow | Max | Min => {
            let (sx, sy) = (g.shape(x).to_vec(), g.shape(y).to_vec());
            if rel[x as usize] {
                let c = match node.op {
                    Add | Sub => ct,
                    Mul => g.add(Mul, &[ct, y])?,
                    Div => g.add(Div, &[ct, y])?,
                    Pow => {
                        let one = g.scalar(d, 1.0);
                        let e = g.add(Sub, &[y, one])?;
                        let p = g.add(Pow, &[x, e])?;
                        let f = g.add(Mul, &[y, p])?;
                        g.add(Mul, &[ct, f])?
                    }
                    _ => {
                        // Ties send the whole gradient to x, the convention `where(x >= y, x, y)` implies.
                        let zero = g.scalar(d, 0.0);
                        let m = if matches!(node.op, Max) { g.add(Ge, &[x, y])? } else { g.add(Ge, &[y, x])? };
                        g.add(Where, &[m, ct, zero])?
                    }
                };
                out.push((x, unbroadcast(g, c, &sx)?));
            }
            if rel[y as usize] {
                let c = match node.op {
                    Add => ct,
                    Sub => g.add(Neg, &[ct])?,
                    Mul => g.add(Mul, &[ct, x])?,
                    Div => {
                        let n = g.add(Neg, &[ct])?;
                        let nx = g.add(Mul, &[n, x])?;
                        let yy = g.add(Mul, &[y, y])?;
                        g.add(Div, &[nx, yy])?
                    }
                    Pow => {
                        let l = g.add(Log, &[x])?;
                        let f = g.add(Mul, &[me, l])?;
                        g.add(Mul, &[ct, f])?
                    }
                    _ => {
                        let zero = g.scalar(d, 0.0);
                        let m = if matches!(node.op, Max) { g.add(Ge, &[x, y])? } else { g.add(Ge, &[y, x])? };
                        g.add(Where, &[m, zero, ct])?
                    }
                };
                out.push((y, unbroadcast(g, c, &sy)?));
            }
        }
        Where => {
            let zero = g.scalar(d, 0.0);
            let (c, a, b) = (node.inputs[0], node.inputs[1], node.inputs[2]);
            if rel[a as usize] {
                out.push((a, g.add(Where, &[c, ct, zero])?));
            }
            if rel[b as usize] {
                out.push((b, g.add(Where, &[c, zero, ct])?));
            }
        }
        MatMul => {
            // 1-D operands are promoted so one rule covers every rank, batched or not.
            let (sa, sb) = (g.shape(x).to_vec(), g.shape(y).to_vec());
            let a2 = if sa.len() == 1 { g.add(Reshape(vec![1, sa[0]]), &[x])? } else { x };
            let b2 = if sb.len() == 1 { g.add(Reshape(vec![sb[0], 1]), &[y])? } else { y };
            let mut cs = g.shape(ct).to_vec();
            if sb.len() == 1 {
                cs.push(1);
            }
            if sa.len() == 1 {
                cs.insert(cs.len() - 1, 1);
            }
            let ct2 = g.add(Reshape(cs), &[ct])?;
            if rel[x as usize] {
                let by = swap(g, b2)?;
                let m = g.add(MatMul, &[ct2, by])?;
                let target = g.shape(a2).to_vec();
                let c = unbroadcast(g, m, &target)?;
                out.push((x, g.add(Reshape(sa.clone()), &[c])?));
            }
            if rel[y as usize] {
                let c = if sa.len() > 2 && sb.len() == 2 {
                    // A shared matrix under a batched operand: flattening the batch gives the
                    // summed gradient in one multiply, instead of many followed by a reduction.
                    let (k, n) = (sb[0], sb[1]);
                    let m_total = g.nodes[x as usize].n() / k;
                    let af = g.add(Reshape(vec![m_total, k]), &[x])?;
                    let cf = g.add(Reshape(vec![m_total, n]), &[ct2])?;
                    let ax = swap(g, af)?;
                    g.add(MatMul, &[ax, cf])?
                } else {
                    let ax = swap(g, a2)?;
                    let m = g.add(MatMul, &[ax, ct2])?;
                    let target = g.shape(b2).to_vec();
                    unbroadcast(g, m, &target)?
                };
                out.push((y, g.add(Reshape(sb.clone()), &[c])?));
            }
        }
        Take { axis } => {
            let shape = g.shape(x).to_vec();
            out.push((x, g.add(TakeGrad { axis: *axis, shape }, &[ct, y])?));
        }
        TakeGrad { axis, .. } => out.push((x, g.add(Take { axis: *axis }, &[ct, y])?)),
        TakeAlong => {
            let shape = g.shape(x).to_vec();
            out.push((x, g.add(TakeAlongGrad { shape }, &[ct, y])?));
        }
        TakeAlongGrad { .. } => out.push((x, g.add(TakeAlong, &[ct, y])?)),
        Concat { axis } => {
            let shape = node.shape.clone();
            let mut at = 0;
            for &input in &node.inputs {
                let len = g.shape(input)[*axis];
                if rel[input as usize] {
                    let (mut starts, mut stops) = (vec![0; shape.len()], shape.clone());
                    starts[*axis] = at;
                    stops[*axis] = at + len;
                    out.push((input, g.add(Slice { starts, stops }, &[ct])?));
                }
                at += len;
            }
        }
        Custom { placeholder, rules } => {
            // Re-instantiate the rule with the real cotangent in place of the placeholder.
            let mut memo: HashMap<u32, u32> = HashMap::from([(*placeholder, ct)]);
            let last = rules.iter().map(|r| r.1).max().unwrap_or(*placeholder);
            for id in *placeholder + 1..=last {
                let n = g.nodes[id as usize].clone();
                if n.inputs.iter().any(|v| memo.contains_key(v)) {
                    let ins: Vec<u32> = n.inputs.iter().map(|v| *memo.get(v).unwrap_or(v)).collect();
                    memo.insert(id, g.add(n.op, &ins)?);
                }
            }
            for &(target, grad) in rules {
                if rel[target as usize] {
                    out.push((target, *memo.get(&grad).unwrap_or(&grad)));
                }
            }
        }
        Floor | Argmax | RandUniform { .. } | RandNormal { .. } | KeySplit { .. } | Placeholder => {}
        Reduce { kind, axes, .. } => {
            let in_shape = g.shape(x).to_vec();
            let mut keep = in_shape.clone();
            for &a in axes {
                keep[a] = 1;
            }
            let ct_keep = g.add(Reshape(keep.clone()), &[ct])?;
            let c = match kind {
                ReduceKind::Sum => g.add(BroadcastTo(in_shape), &[ct_keep])?,
                ReduceKind::Max | ReduceKind::Min => {
                    // Ties share the gradient equally.
                    let m = g.add(Reshape(keep), &[me])?;
                    let mask = if *kind == ReduceKind::Max { g.add(Ge, &[x, m])? } else { g.add(Ge, &[m, x])? };
                    let count = g.add(Reduce { kind: ReduceKind::Sum, axes: axes.clone(), keepdims: true }, &[mask])?;
                    let share = g.add(Div, &[mask, count])?;
                    g.add(Mul, &[ct_keep, share])?
                }
            };
            out.push((x, c));
        }
        Reshape(_) => {
            let s = g.shape(x).to_vec();
            out.push((x, g.add(Reshape(s), &[ct])?));
        }
        Transpose(p) => {
            let mut inv = vec![0; p.len()];
            for (k, &v) in p.iter().enumerate() {
                inv[v] = k;
            }
            out.push((x, g.add(Transpose(inv), &[ct])?));
        }
        BroadcastTo(_) => {
            let s = g.shape(x).to_vec();
            out.push((x, unbroadcast(g, ct, &s)?));
        }
        Slice { starts, .. } => {
            let shape = g.shape(x).to_vec();
            out.push((x, g.add(Pad { starts: starts.clone(), shape }, &[ct])?));
        }
        Pad { starts, .. } => {
            let stops: Vec<usize> = starts.iter().zip(g.shape(x)).map(|(s, n)| s + n).collect();
            out.push((x, g.add(Slice { starts: starts.clone(), stops }, &[ct])?));
        }
        Cast(_) => {
            let src = g.dtype(x);
            out.push((x, g.add(Cast(src), &[ct])?));
        }
        Exp | Sin | Cos | Tanh | Abs => unreachable!("handled by unary_factor"),
        MatMulT { .. } => unreachable!("optimizer output is never differentiated"),
    }
    Ok(out)
}
