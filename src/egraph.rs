//! Equality saturation: keep every equivalent form of the math at once, then pick the cheapest.
//!
//! The usual pass order commits to each rewrite, so an early one can block a better later one. Here
//! rewrites only add alternatives, and a cost model chooses at the end. This module is one proposal
//! among others: `search` returns `None` unless the whole graph really comes out cheaper.
//!
//! Limits are node and iteration counts and never time, so that a program compiles to the same plan
//! on every run.

use crate::graph::{broadcast_shapes, Graph, Op};
use crate::tensor::{DType, Tensor};
use egg::{Analysis, DidMerge, EGraph, ENodeOrVar, Id, Pattern, PatternAst, Rewrite, Runner, Subst, Symbol, SymbolLang, Var};
use std::collections::HashMap;
use std::sync::Arc;

/// How far a rule is from exact equality in IEEE 754.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RuleSet {
    /// Equal for every input, including NaN and infinity.
    Exact,
    /// Exact plus reassociating and factoring matrix products: few candidates, large wins.
    Linear,
    /// Exact plus rules that only change rounding on ordinary values (associativity, factoring).
    /// Results still pass the equivalence check against the original program.
    Rounding,
}

#[derive(Clone, Debug)]
pub struct Info {
    shape: Vec<usize>,
    dtype: DType,
    /// Set when every element is this constant.
    splat: Option<f64>,
    /// No valid node exists: rule guards keep these out of real classes.
    valid: bool,
}

#[derive(Default)]
pub struct Table {
    ops: HashMap<Symbol, Op>,
    /// Inputs and constants, which have no children to derive information from.
    leaves: HashMap<Symbol, Info>,
}

impl Analysis<SymbolLang> for Table {
    type Data = Info;

    fn make(egraph: &mut EGraph<SymbolLang, Self>, enode: &SymbolLang, _id: Id) -> Info {
        let table = &egraph.analysis;
        if let Some(info) = table.leaves.get(&enode.op) {
            return info.clone();
        }
        let children: Vec<&Info> = enode.children.iter().map(|&c| &egraph[c].data).collect();
        derive(table, enode.op, &children)
    }

    fn merge(&mut self, a: &mut Info, b: Info) -> DidMerge {
        // Equal classes have equal shapes, which guards enforce. Constants can only gain information.
        if a.splat.is_none() && b.splat.is_some() {
            a.splat = b.splat;
            return DidMerge(true, false);
        }
        DidMerge(false, a.splat.is_none() && b.splat.is_some())
    }
}

fn derive(table: &Table, op: Symbol, children: &[&Info]) -> Info {
    let invalid = Info { shape: vec![], dtype: DType::F32, splat: None, valid: false };
    if children.iter().any(|c| !c.valid) {
        return invalid;
    }
    let Some(known) = table.ops.get(&op) else { return invalid };
    // Fast path: this runs for every candidate node, and most candidates are elementwise.
    if known.is_binary_elementwise() && children.len() == 2 && children[0].dtype == children[1].dtype {
        return match broadcast_shapes(&children[0].shape, &children[1].shape) {
            Some(shape) => Info { shape, dtype: children[0].dtype, splat: None, valid: true },
            None => invalid,
        };
    }
    if known.is_unary_elementwise() && children.len() == 1 {
        return Info { shape: children[0].shape.clone(), dtype: children[0].dtype, splat: None, valid: true };
    }
    let mut probe = Graph::new();
    let ids: Vec<u32> = children.iter().map(|c| probe.input(&c.shape, c.dtype)).collect();
    match probe.add(known.clone(), &ids) {
        Ok(id) => Info { shape: probe.shape(id).to_vec(), dtype: probe.dtype(id), splat: None, valid: true },
        Err(_) => invalid,
    }
}

/// Applies a rule only when its result has exactly the matched value's shape and dtype.
struct Guarded {
    rhs: Pattern<SymbolLang>,
}

impl egg::Applier<SymbolLang, Table> for Guarded {
    fn apply_one(&self, egraph: &mut EGraph<SymbolLang, Table>, id: Id, subst: &Subst, _ast: Option<&PatternAst<SymbolLang>>, name: Symbol) -> Vec<Id> {
        let mut infos: Vec<Info> = vec![];
        for node in self.rhs.ast.as_ref() {
            let info = match node {
                ENodeOrVar::Var(v) => egraph[subst[*v]].data.clone(),
                ENodeOrVar::ENode(n) => {
                    let kids: Vec<&Info> = n.children.iter().map(|c| &infos[usize::from(*c)]).collect();
                    derive(&egraph.analysis, n.op, &kids)
                }
            };
            infos.push(info);
        }
        let result = infos.last().expect("patterns are not empty");
        let current = &egraph[id].data;
        if !result.valid || result.shape != current.shape || result.dtype != current.dtype {
            return vec![];
        }
        self.rhs.apply_one(egraph, id, subst, None, name)
    }
    fn vars(&self) -> Vec<Var> {
        self.rhs.vars()
    }
}

fn rule(name: &str, lhs: &str, rhs: &str) -> Rewrite<SymbolLang, Table> {
    let (l, r): (Pattern<SymbolLang>, Pattern<SymbolLang>) = (lhs.parse().expect("rule syntax"), rhs.parse().expect("rule syntax"));
    Rewrite::new(name, l, Guarded { rhs: r }).expect("rule definition")
}

pub fn rules(set: RuleSet) -> Vec<Rewrite<SymbolLang, Table>> {
    let mut r = vec![
        // Exact: these hold for every value, NaN and infinity included.
        rule("add-comm", "(add ?a ?b)", "(add ?b ?a)"),
        rule("mul-comm", "(mul ?a ?b)", "(mul ?b ?a)"),
        rule("neg-neg", "(neg (neg ?a))", "?a"),
        rule("sub-to-add", "(sub ?a ?b)", "(add ?a (neg ?b))"),
        rule("add-to-sub", "(add ?a (neg ?b))", "(sub ?a ?b)"),
        rule("neg-mul", "(mul (neg ?a) ?b)", "(neg (mul ?a ?b))"),
        rule("mul-neg", "(neg (mul ?a ?b))", "(mul (neg ?a) ?b)"),
        rule("neg-div", "(div (neg ?a) ?b)", "(neg (div ?a ?b))"),
        rule("t-t", "(t (t ?a))", "?a"),
        // A transposed operand needs no copy: BLAS and MPS read it through a flag.
        rule("mm-ta", "(matmul (t ?a) ?b)", "(mm-ta ?a ?b)"),
        rule("mm-tb", "(matmul ?a (t ?b))", "(mm-tb ?a ?b)"),
        rule("mm-tt", "(mm-ta ?a (t ?b))", "(mm-tt ?a ?b)"),
        rule("mm-tt2", "(mm-tb (t ?a) ?b)", "(mm-tt ?a ?b)"),
        rule("where-same", "(where ?c ?a ?a)", "?a"),
    ];
    if set != RuleSet::Exact {
        r.extend([
            rule("matmul-assoc", "(matmul (matmul ?a ?b) ?c)", "(matmul ?a (matmul ?b ?c))"),
            rule("matmul-assoc-r", "(matmul ?a (matmul ?b ?c))", "(matmul (matmul ?a ?b) ?c)"),
            rule("matmul-factor-l", "(add (matmul ?a ?x) (matmul ?a ?y))", "(matmul ?a (add ?x ?y))"),
            rule("matmul-factor-r", "(add (matmul ?x ?a) (matmul ?y ?a))", "(matmul (add ?x ?y) ?a)"),
            rule("t-matmul", "(t (matmul ?a ?b))", "(matmul (t ?b) (t ?a))"),
            rule("matmul-t", "(matmul (t ?b) (t ?a))", "(t (matmul ?a ?b))"),
        ]);
    }
    if set == RuleSet::Rounding {
        r.extend([
            rule("add-assoc", "(add (add ?a ?b) ?c)", "(add ?a (add ?b ?c))"),
            rule("add-assoc-r", "(add ?a (add ?b ?c))", "(add (add ?a ?b) ?c)"),
            rule("mul-assoc", "(mul (mul ?a ?b) ?c)", "(mul ?a (mul ?b ?c))"),
            rule("mul-assoc-r", "(mul ?a (mul ?b ?c))", "(mul (mul ?a ?b) ?c)"),
            rule("factor", "(add (mul ?a ?x) (mul ?a ?y))", "(mul ?a (add ?x ?y))"),
            rule("distribute", "(mul ?a (add ?x ?y))", "(add (mul ?a ?x) (mul ?a ?y))"),
            rule("div-div", "(div (div ?x ?y) ?z)", "(div ?x (mul ?y ?z))"),
            rule("mul-div", "(div (mul ?x ?y) ?z)", "(mul ?x (div ?y ?z))"),
        ]);
    }
    r
}

/// Estimated nanoseconds, from kernel measurements on an Apple M4 Pro. Only ratios matter.
#[derive(Clone, Copy, Debug)]
pub struct CostModel {
    /// Fixed cost per operation: dispatch and barrier on a GPU, nearly nothing on a CPU.
    per_op: f64,
    /// Per element written, for memory traffic.
    per_element: f64,
    /// Per multiplication in a matrix product.
    per_flop: f64,
    /// Extra per element for functions evaluated with a math library.
    per_math_element: f64,
    per_transpose_element: f64,
}

impl CostModel {
    pub const CPU: CostModel = CostModel { per_op: 200.0, per_element: 0.06, per_flop: 0.0015, per_math_element: 0.6, per_transpose_element: 1.4 };
    pub const METAL: CostModel = CostModel { per_op: 30_000.0, per_element: 0.1, per_flop: 0.004, per_math_element: 0.05, per_transpose_element: 0.15 };

    fn node(&self, op: &Op, shape: &[usize], args: &[&[usize]]) -> f64 {
        use Op::*;
        let n: f64 = shape.iter().product::<usize>() as f64;
        let base = self.per_op + self.per_element * n;
        match op {
            Input(_) | Const(_) => 0.0,
            // Aliases on every backend: no kernel runs.
            Reshape(_) | StopGradient | Output { .. } | Custom { .. } => 0.0,
            MatMul => {
                let k = args[0].last().copied().unwrap_or(1) as f64;
                base + self.per_flop * n * k
            }
            MatMulT { ta, .. } => {
                let a = args[0];
                let k = if *ta { a[a.len() - 2] } else { a[a.len() - 1] } as f64;
                base + self.per_flop * n * k
            }
            // Elements move one at a time through a permutation.
            Transpose(_) => self.per_op + self.per_transpose_element * n,
            Exp | Log | Sin | Cos | Tanh => base + self.per_math_element * n,
            Pow => base + 16.0 * self.per_math_element * n,
            Div | Sqrt => base + 0.2 * self.per_math_element * n,
            _ => base,
        }
    }

    /// Whole-graph cost, counting shared values once. This, not the extraction's estimate, decides.
    pub fn graph_cost(&self, g: &Graph, outputs: &[u32]) -> f64 {
        let keep = g.reachable(outputs);
        g.nodes
            .iter()
            .enumerate()
            .filter(|(i, _)| keep[*i])
            .map(|(_, n)| {
                let args: Vec<&[usize]> = n.inputs.iter().map(|&x| g.shape(x)).collect();
                // Everything constant folds away.
                let constant = !n.inputs.is_empty() && n.inputs.iter().all(|&x| matches!(g.nodes[x as usize].op, Op::Const(_)));
                if constant {
                    0.0
                } else {
                    self.node(&n.op, &n.shape, &args)
                }
            })
            .sum()
    }
}

struct Cost<'a> {
    model: CostModel,
    egraph: &'a EGraph<SymbolLang, Table>,
}

impl egg::CostFunction<SymbolLang> for Cost<'_> {
    type Cost = f64;
    fn cost<C: FnMut(Id) -> f64>(&mut self, enode: &SymbolLang, mut costs: C) -> f64 {
        let table = &self.egraph.analysis;
        if table.leaves.contains_key(&enode.op) {
            return 0.0;
        }
        let own = if let Some(op) = table.ops.get(&enode.op) {
            let kids: Vec<&Info> = enode.children.iter().map(|&c| &self.egraph[c].data).collect();
            if kids.iter().any(|k| !k.valid) {
                return f64::INFINITY;
            }
            let info = derive(table, enode.op, &kids);
            if !info.valid {
                return f64::INFINITY;
            }
            let args: Vec<&[usize]> = kids.iter().map(|k| k.shape.as_slice()).collect();
            let all_const = !kids.is_empty() && kids.iter().all(|k| k.splat.is_some());
            if all_const {
                0.0
            } else {
                self.model.node(op, &info.shape, &args)
            }
        } else {
            0.0
        };
        enode.children.iter().fold(own, |sum, &c| sum + costs(c))
    }
}

fn symbol_of(op: &Op, id: usize) -> String {
    use Op::*;
    match op {
        Input(k) => format!("input{k}"),
        Const(_) => format!("const{id}"),
        Add | Sub | Mul | Div | Neg | Where => op.name().to_string(),
        MatMul => "matmul".to_string(),
        Transpose(p) if p.as_slice() == [1, 0] => "t".to_string(),
        // Everything else is opaque to the rules. Its debug form keeps attributes distinct,
        // and control-flow bodies print by identity.
        other => format!("opaque-{:?}", other).replace(|c: char| c.is_whitespace() || c == '(' || c == ')', "_"),
    }
}

#[derive(Debug, Default, Clone)]
pub struct SearchStats {
    pub iterations: usize,
    pub egraph_nodes: usize,
    pub cost_before: f64,
    pub cost_after: f64,
}

struct Rebuild<'a> {
    g: &'a Graph,
    class: &'a [Id],
    egraph: &'a EGraph<SymbolLang, Table>,
    info: &'a HashMap<Id, (Option<Arc<str>>, bool)>,
    roots: &'a [u32],
    outputs: &'a [u32],
}

impl Rebuild<'_> {
    /// Builds the graph a choice of node per class describes. `None` if the choice has a cycle.
    fn build(&self, choose: impl Fn(Id) -> SymbolLang) -> Option<(Graph, Vec<u32>)> {
        let egraph = self.egraph;
        let mut h = Graph::new();
        let mut built: HashMap<Id, u32> = HashMap::new();
        // Inputs first, so their numbering survives.
        for (i, n) in self.g.nodes.iter().enumerate() {
            if let Op::Input(_) = n.op {
                h.cur_source = n.source.clone();
                let id = h.input(&n.shape, n.dtype);
                h.nodes[id as usize].op = n.op.clone();
                built.insert(egraph.find(self.class[i]), id);
            }
        }
        // Explicit stack: real graphs are deep enough to overflow a recursive walk.
        let mut on_path = std::collections::HashSet::new();
        for &root in self.roots {
            let mut stack = vec![egraph.find(self.class[root as usize])];
            while let Some(&c) = stack.last() {
                if built.contains_key(&c) {
                    stack.pop();
                    on_path.remove(&c);
                    continue;
                }
                let node = choose(c);
                let missing: Vec<Id> = node.children.iter().map(|&k| egraph.find(k)).filter(|k| !built.contains_key(k)).collect();
                if !missing.is_empty() {
                    if !on_path.insert(c) || missing.iter().any(|k| on_path.contains(k)) {
                        return None;
                    }
                    stack.extend(missing);
                    continue;
                }
                let op = egraph.analysis.ops.get(&node.op).expect("every symbol has an op").clone();
                let ins: Vec<u32> = node.children.iter().map(|&k| built[&egraph.find(k)]).collect();
                let (source, role) = self.info.get(&c).cloned().unwrap_or_else(|| {
                    // A form that did not exist in the source inherits from its operands.
                    let any_backward = node.children.iter().any(|&k| self.info.get(&egraph.find(k)).is_some_and(|x| x.1));
                    (node.children.iter().find_map(|&k| self.info.get(&egraph.find(k)).and_then(|x| x.0.clone())), any_backward)
                });
                h.cur_source = source;
                let id = match op {
                    Op::Const(t) => h.constant(Tensor::clone(&t)),
                    op => h.add(op, &ins).ok()?,
                };
                h.nodes[id as usize].origin = role.then_some(0);
                built.insert(c, id);
                on_path.remove(&c);
                stack.pop();
            }
        }
        let out = self.outputs.iter().map(|&o| built[&egraph.find(self.class[o as usize])]).collect();
        Some((h, out))
    }
}

pub struct Limits {
    pub iterations: usize,
    pub nodes: usize,
}

pub const DEFAULT_LIMITS: Limits = Limits { iterations: 6, nodes: 20_000 };

/// A cheaper graph computing the same outputs, if the search finds one. `backward` marks nodes of
/// the backward pass; the result carries the same marks through `origin`.
pub fn search(g: &Graph, outputs: &[u32], model: CostModel, set: RuleSet, limits: &Limits) -> (Option<(Graph, Vec<u32>)>, SearchStats) {
    let keep = g.reachable(outputs);
    let mut egraph: EGraph<SymbolLang, Table> = EGraph::new(Table::default());
    let mut class = vec![Id::from(0usize); g.nodes.len()];
    let mut stats = SearchStats { cost_before: model.graph_cost(g, outputs), ..Default::default() };
    let mut roots_needed: Vec<u32> = outputs.to_vec();
    for (i, n) in g.nodes.iter().enumerate() {
        if !keep[i] && !matches!(n.op, Op::Input(_)) {
            continue;
        }
        let name = symbol_of(&n.op, i);
        let sym = Symbol::from(&name);
        match &n.op {
            Op::Input(_) | Op::Const(_) => {
                let splat = if let Op::Const(t) = &n.op {
                    let v = t.data.to_f64();
                    v.first().copied().filter(|f| t.n() <= 4096 && v.iter().all(|x| x == f))
                } else {
                    None
                };
                egraph.analysis.leaves.insert(sym, Info { shape: n.shape.clone(), dtype: n.dtype, splat, valid: true });
                egraph.analysis.ops.insert(sym, n.op.clone());
            }
            op => {
                egraph.analysis.ops.insert(sym, op.clone());
            }
        }
        let children: Vec<Id> = n.inputs.iter().map(|&x| class[x as usize]).collect();
        class[i] = egraph.add(SymbolLang::new(sym, children));
        if matches!(n.op, Op::Input(_)) {
            roots_needed.push(i as u32);
        }
    }
    // Rules build arithmetic nodes that may not occur in the source.
    for (name, op) in [("add", Op::Add), ("mul", Op::Mul), ("sub", Op::Sub), ("div", Op::Div), ("neg", Op::Neg), ("matmul", Op::MatMul), ("t", Op::Transpose(vec![1, 0])), ("where", Op::Where), ("mm-ta", Op::MatMulT { ta: true, tb: false }), ("mm-tb", Op::MatMulT { ta: false, tb: true }), ("mm-tt", Op::MatMulT { ta: true, tb: true })] {
        egraph.analysis.ops.insert(Symbol::from(name), op);
    }

    let runner: Runner<SymbolLang, Table> = Runner::default().with_egraph(egraph).with_iter_limit(limits.iterations).with_node_limit(limits.nodes).with_time_limit(std::time::Duration::from_secs(3600)).with_scheduler(egg::BackoffScheduler::default()).run(&rules(set));
    stats.iterations = runner.iterations.len();
    stats.egraph_nodes = runner.egraph.total_size();

    // Rebuild from the cheapest form of each class, keeping sources and backward marks.
    let egraph = &runner.egraph;
    let mut info: HashMap<Id, (Option<Arc<str>>, bool)> = HashMap::new();
    for (i, n) in g.nodes.iter().enumerate() {
        if keep[i] || matches!(n.op, Op::Input(_)) {
            let e = info.entry(egraph.find(class[i])).or_insert((n.source.clone(), n.origin.is_some()));
            // A class computed in the forward pass stays there, even if the backward pass also has it.
            e.1 &= n.origin.is_some();
        }
    }
    let ctx = Rebuild { g, class: &class, egraph, info: &info, roots: &roots_needed, outputs };

    // Tree extraction counts a shared value once per use. Greedy DAG extraction and hill climbing
    // were measured against it: under 0.1% better on a GPT, for quadratic compile time. Not kept.
    let extractor = egg::Extractor::new(egraph, Cost { model, egraph });
    let Some((h, out)) = ctx.build(|c| extractor.find_best_node(c).clone()) else { return (None, stats) };
    stats.cost_after = model.graph_cost(&h, &out);
    (Some((h, out)), stats)
}

