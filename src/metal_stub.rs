//! Platforms without Metal: the device reports itself unavailable and is never constructed.

use crate::plan::{Exec, Plan};
use crate::tensor::{Buf, Tensor};

pub fn available() -> bool {
    false
}

pub struct MetalProgram(());
pub struct MetalExec(());

impl MetalProgram {
    pub fn new(_: &Plan) -> Result<MetalProgram, String> {
        Err("Metal exists only on Apple platforms".into())
    }
    pub fn exec(&self, _: &Plan) -> MetalExec {
        unreachable!("MetalProgram cannot be constructed here")
    }
}

impl MetalExec {
    pub fn step_timed(&mut self, _: &mut [f64]) {}
    pub fn breakdown(&mut self, _: usize) -> serde_json::Value {
        serde_json::Value::Null
    }
}

impl Exec for MetalExec {
    fn load(&mut self, _: usize, _: &Tensor) {}
    fn step(&mut self) {}
    fn swap(&mut self, _: usize, _: usize) {}
    fn prefix(&self, _: usize, _: usize) -> Buf {
        unreachable!("MetalProgram cannot be constructed here")
    }
    fn write_prefix(&mut self, _: usize, _: &Buf) {}
    fn scalar(&self, _: usize) -> f64 {
        0.0
    }
}
