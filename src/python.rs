use crate::autodiff;
use crate::graph::{Diag, Graph, Op, ReduceKind};
use crate::plan;
use crate::tensor::{Buf, DType, Tensor};
use crate::validate::{self, Compiled, Level, RunSpec};
use pyo3::create_exception;
use pyo3::exceptions::{PyException, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use std::sync::Arc;

create_exception!(_rg, DiagnosticError, PyException);

fn raise(py: Python<'_>, d: Diag) -> PyErr {
    let text = d.to_string();
    let err = DiagnosticError::new_err(text.clone());
    let parsed = py.import("json").and_then(|j| j.call_method1("loads", (text,)));
    match parsed.and_then(|v| err.value(py).setattr("diagnostic", v)) {
        Ok(()) => err,
        Err(e) => e,
    }
}

fn parse_dtype(s: &str) -> PyResult<DType> {
    match s {
        "float32" | "f32" => Ok(DType::F32),
        "float64" | "f64" => Ok(DType::F64),
        "int32" | "i32" => Ok(DType::I32),
        _ => Err(PyValueError::new_err(format!("unsupported dtype {s}"))),
    }
}

#[pyclass(name = "Tensor", frozen, from_py_object)]
#[derive(Clone)]
struct PyTensor {
    t: Tensor,
}

#[pymethods]
impl PyTensor {
    #[new]
    fn new(data: &[u8], shape: Vec<usize>, dtype: &str) -> PyResult<PyTensor> {
        let d = parse_dtype(dtype)?;
        let n: usize = shape.iter().product();
        if data.len() != n * d.size() {
            return Err(PyValueError::new_err(format!("{} bytes do not fit shape {:?} of {}", data.len(), shape, d.name())));
        }
        let buf = match d {
            DType::F32 => Buf::F32(data.chunks_exact(4).map(|c| f32::from_ne_bytes(c.try_into().unwrap())).collect()),
            DType::F64 => Buf::F64(data.chunks_exact(8).map(|c| f64::from_ne_bytes(c.try_into().unwrap())).collect()),
            DType::I32 => Buf::I32(data.chunks_exact(4).map(|c| i32::from_ne_bytes(c.try_into().unwrap())).collect()),
        };
        Ok(PyTensor { t: Tensor { shape, data: buf } })
    }
    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.t.shape.clone()
    }
    #[getter]
    fn dtype(&self) -> &'static str {
        self.t.dtype().name()
    }
    fn is_finite(&self) -> bool {
        self.t.is_finite()
    }
    fn tobytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        match &self.t.data {
            Buf::F32(v) => PyBytes::new(py, &v.iter().flat_map(|x| x.to_ne_bytes()).collect::<Vec<u8>>()),
            Buf::F64(v) => PyBytes::new(py, &v.iter().flat_map(|x| x.to_ne_bytes()).collect::<Vec<u8>>()),
            Buf::I32(v) => PyBytes::new(py, &v.iter().flat_map(|x| x.to_ne_bytes()).collect::<Vec<u8>>()),
        }
    }
}

#[pyclass(name = "Graph", from_py_object)]
#[derive(Clone)]
struct PyGraph {
    g: Graph,
}

fn attr<'py, T: FromPyObjectOwned<'py>>(attrs: Option<&Bound<'py, PyDict>>, key: &str) -> PyResult<T> {
    let v = attrs.and_then(|a| a.get_item(key).ok().flatten()).ok_or_else(|| PyValueError::new_err(format!("missing attribute {key}")))?;
    v.extract::<T>().map_err(|_| PyValueError::new_err(format!("invalid attribute {key}")))
}

use pyo3::conversion::FromPyObjectOwned;

fn parse_op(name: &str, a: Option<&Bound<'_, PyDict>>) -> PyResult<Op> {
    use Op::*;
    let reduce = |kind| -> PyResult<Op> { Ok(Reduce { kind, axes: attr(a, "axes")?, keepdims: attr(a, "keepdims")? }) };
    Ok(match name {
        "neg" => Neg,
        "exp" => Exp,
        "log" => Log,
        "sqrt" => Sqrt,
        "sin" => Sin,
        "cos" => Cos,
        "tanh" => Tanh,
        "abs" => Abs,
        "sign" => Sign,
        "add" => Add,
        "sub" => Sub,
        "mul" => Mul,
        "div" => Div,
        "pow" => Pow,
        "maximum" => Max,
        "minimum" => Min,
        "gt" => Gt,
        "ge" => Ge,
        "where" => Where,
        "matmul" => MatMul,
        "sum" => reduce(ReduceKind::Sum)?,
        "max" => reduce(ReduceKind::Max)?,
        "min" => reduce(ReduceKind::Min)?,
        "reshape" => Reshape(attr(a, "shape")?),
        "transpose" => Transpose(attr(a, "perm")?),
        "broadcast_to" => BroadcastTo(attr(a, "shape")?),
        "slice" => Slice { starts: attr(a, "starts")?, stops: attr(a, "stops")? },
        "cast" => Cast(parse_dtype(&attr::<String>(a, "dtype")?)?),
        "stop_gradient" => StopGradient,
        "floor" => Floor,
        "take" => Take { axis: attr(a, "axis")? },
        "take_along" => TakeAlong,
        "concat" => Concat { axis: attr(a, "axis")? },
        "argmax" => Argmax,
        "random_uniform" => RandUniform { shape: attr(a, "shape")?, dtype: parse_dtype(&attr::<String>(a, "dtype")?)? },
        "random_normal" => RandNormal { shape: attr(a, "shape")?, dtype: parse_dtype(&attr::<String>(a, "dtype")?)? },
        "key_split" => KeySplit { n: attr(a, "n")? },
        _ => return Err(PyValueError::new_err(format!("unknown op {name}"))),
    })
}

#[pymethods]
impl PyGraph {
    #[new]
    fn new() -> PyGraph {
        PyGraph { g: Graph::new() }
    }
    fn input(&mut self, shape: Vec<usize>, dtype: &str, source: Option<&str>) -> PyResult<u32> {
        self.g.cur_source = source.map(Arc::from);
        Ok(self.g.input(&shape, parse_dtype(dtype)?))
    }
    fn constant(&mut self, t: &PyTensor, source: Option<&str>) -> u32 {
        self.g.cur_source = source.map(Arc::from);
        self.g.constant(t.t.clone())
    }
    #[pyo3(signature = (name, inputs, attrs=None, source=None))]
    fn op(&mut self, py: Python<'_>, name: &str, inputs: Vec<u32>, attrs: Option<&Bound<'_, PyDict>>, source: Option<&str>) -> PyResult<u32> {
        let op = parse_op(name, attrs)?;
        self.g.cur_source = source.map(Arc::from);
        self.g.add(op, &inputs).map_err(|d| raise(py, d))
    }
    fn grad(&mut self, py: Python<'_>, y: u32, wrts: Vec<u32>, source: Option<&str>) -> PyResult<Vec<u32>> {
        self.g.cur_source = source.map(Arc::from);
        autodiff::grad(&mut self.g, y, &wrts).map_err(|d| raise(py, d))
    }
    fn placeholder(&mut self, shape: Vec<usize>, dtype: &str) -> PyResult<u32> {
        Ok(self.g.placeholder(&shape, parse_dtype(dtype)?))
    }
    fn custom(&mut self, py: Python<'_>, y: u32, placeholder: u32, rules: Vec<(u32, u32)>, source: Option<&str>) -> PyResult<u32> {
        self.g.cur_source = source.map(Arc::from);
        self.g.add(Op::Custom { placeholder, rules }, &[y]).map_err(|d| raise(py, d))
    }
    fn shape(&self, id: u32) -> Vec<usize> {
        self.g.shape(id).to_vec()
    }
    fn dtype(&self, id: u32) -> &'static str {
        self.g.dtype(id).name()
    }
    /// One-off evaluation for eager operations; the graph must have no inputs.
    fn eval(&self, ids: Vec<u32>) -> Vec<PyTensor> {
        plan::eval(&self.g, &ids, &[]).into_iter().map(|t| PyTensor { t }).collect()
    }
}

#[pyclass(name = "Compiled", frozen)]
struct PyCompiled {
    c: Compiled,
}

fn tensors(v: Vec<PyTensor>) -> Vec<Tensor> {
    v.into_iter().map(|p| p.t).collect()
}

fn wrap(v: Vec<Tensor>) -> Vec<PyTensor> {
    v.into_iter().map(|t| PyTensor { t }).collect()
}

#[pymethods]
impl PyCompiled {
    #[getter]
    fn report_json(&self) -> String {
        self.c.report.to_string()
    }
    #[getter]
    fn output_shapes(&self) -> Vec<Vec<usize>> {
        self.c.plan.outputs.iter().map(|o| o.shape.clone()).collect()
    }
    fn call(&self, py: Python<'_>, inputs: Vec<PyTensor>) -> PyResult<Vec<PyTensor>> {
        let inputs = tensors(inputs);
        self.c.check_inputs(&inputs).map_err(|d| raise(py, d))?;
        py.detach(|| self.c.try_call(&inputs)).map(wrap).map_err(|d| raise(py, d))
    }
    fn check(&self, py: Python<'_>, inputs: Vec<PyTensor>) -> PyResult<String> {
        let inputs = tensors(inputs);
        self.c.check_inputs(&inputs).map_err(|d| raise(py, d))?;
        Ok(py.detach(|| self.c.check(&inputs)).to_string())
    }
    fn profile(&self, py: Python<'_>, inputs: Vec<PyTensor>, repeats: usize) -> PyResult<String> {
        let inputs = tensors(inputs);
        self.c.check_inputs(&inputs).map_err(|d| raise(py, d))?;
        Ok(py.detach(|| self.c.profile(&inputs, repeats)).to_string())
    }
    /// Returns (state, outputs, history, steps, nonfinite_step, elapsed_seconds).
    #[allow(clippy::type_complexity)]
    fn run(&self, py: Python<'_>, state_map: Vec<(usize, usize)>, data_inputs: Vec<usize>, history: Vec<usize>, state: Vec<PyTensor>, batches: Vec<Vec<PyTensor>>, steps: usize) -> PyResult<(Vec<PyTensor>, Vec<PyTensor>, Vec<Vec<f64>>, usize, Option<usize>, f64)> {
        let spec = RunSpec { state: state_map, data: data_inputs, history };
        self.c.validate_spec(&spec).map_err(|d| raise(py, d))?;
        let state = tensors(state);
        let batches: Vec<Vec<Tensor>> = batches.into_iter().map(tensors).collect();
        let out = py.detach(|| self.c.run(&spec, &state, &batches, steps));
        if let Some(error) = &out.device_error {
            let mut d = validate::device_failure(error);
            d["observation"]["completed_steps"] = serde_json::json!(out.steps);
            return Err(raise(py, d));
        }
        let seconds = out.seconds;
        Ok((wrap(out.state), wrap(out.outputs), out.history, out.steps, out.nonfinite_step, seconds))
    }
    /// First step at which the state or an output turns non-finite: (step, batch index, state before it).
    fn replay(&self, py: Python<'_>, state_map: Vec<(usize, usize)>, data_inputs: Vec<usize>, history: Vec<usize>, state: Vec<PyTensor>, batches: Vec<Vec<PyTensor>>, steps: usize) -> Option<(usize, usize, Vec<PyTensor>)> {
        let spec = RunSpec { state: state_map, data: data_inputs, history };
        let state = tensors(state);
        let batches: Vec<Vec<Tensor>> = batches.into_iter().map(tensors).collect();
        py.detach(|| self.c.replay(&spec, &state, &batches, steps)).map(|(k, b, s)| (k, b, wrap(s)))
    }
}

#[pyfunction]
fn compile(py: Python<'_>, graph: &PyGraph, outputs: Vec<u32>, examples: Vec<PyTensor>, level: &str, device: &str) -> PyResult<PyCompiled> {
    let use_metal = match device {
        "cpu" => false,
        "metal" => true,
        other => {
            let d = crate::graph::diag("DEVICE_UNAVAILABLE", "error", serde_json::Value::Null, serde_json::json!({ "device": other, "available": available_devices() }), "unknown_device", "verified");
            return Err(raise(py, d));
        }
    };
    let level = Level::parse(level).ok_or_else(|| PyValueError::new_err(format!("unknown validation level {level}")))?;
    let examples = tensors(examples);
    let raw = graph.g.clone();
    let c = py.detach(|| validate::compile_on(raw, outputs, &examples, level, use_metal)).map_err(|d| raise(py, d))?;
    Ok(PyCompiled { c })
}

#[pyfunction]
fn available_devices() -> Vec<&'static str> {
    if crate::metal::available() {
        vec!["cpu", "metal"]
    } else {
        vec!["cpu"]
    }
}

#[pymodule]
fn _rg(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTensor>()?;
    m.add_class::<PyGraph>()?;
    m.add_class::<PyCompiled>()?;
    m.add_function(wrap_pyfunction!(compile, m)?)?;
    m.add_function(wrap_pyfunction!(available_devices, m)?)?;
    m.add("DiagnosticError", m.py().get_type::<DiagnosticError>())?;
    Ok(())
}
