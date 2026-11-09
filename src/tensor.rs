use std::ops::{Add, Div, Mul, Neg, Sub};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DType {
    F32,
    F64,
    I32,
}

impl DType {
    pub fn name(self) -> &'static str {
        match self {
            DType::F32 => "float32",
            DType::F64 => "float64",
            DType::I32 => "int32",
        }
    }
    pub fn size(self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F64 => 8,
            DType::I32 => 4,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Buf {
    F32(Vec<f32>),
    F64(Vec<f64>),
    I32(Vec<i32>),
}

impl Buf {
    pub fn zeros(d: DType, n: usize) -> Buf {
        match d {
            DType::F32 => Buf::F32(vec![0.0; n]),
            DType::F64 => Buf::F64(vec![0.0; n]),
            DType::I32 => Buf::I32(vec![0; n]),
        }
    }
    pub fn len(&self) -> usize {
        match self {
            Buf::F32(v) => v.len(),
            Buf::F64(v) => v.len(),
            Buf::I32(v) => v.len(),
        }
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn dtype(&self) -> DType {
        match self {
            Buf::F32(_) => DType::F32,
            Buf::F64(_) => DType::F64,
            Buf::I32(_) => DType::I32,
        }
    }
    /// Copies the first `n` elements of `src`; panics on dtype mismatch (callers validate signatures).
    pub fn copy_prefix_from(&mut self, src: &Buf, n: usize) {
        match (self, src) {
            (Buf::F32(d), Buf::F32(s)) => d[..n].copy_from_slice(&s[..n]),
            (Buf::F64(d), Buf::F64(s)) => d[..n].copy_from_slice(&s[..n]),
            (Buf::I32(d), Buf::I32(s)) => d[..n].copy_from_slice(&s[..n]),
            _ => panic!("dtype mismatch"),
        }
    }
    pub fn prefix(&self, n: usize) -> Buf {
        match self {
            Buf::F32(v) => Buf::F32(v[..n].to_vec()),
            Buf::F64(v) => Buf::F64(v[..n].to_vec()),
            Buf::I32(v) => Buf::I32(v[..n].to_vec()),
        }
    }
    pub fn to_f64(&self) -> Vec<f64> {
        match self {
            Buf::F32(v) => v.iter().map(|&x| x as f64).collect(),
            Buf::F64(v) => v.clone(),
            Buf::I32(v) => v.iter().map(|&x| x as f64).collect(),
        }
    }
    pub fn from_f64(d: DType, v: &[f64]) -> Buf {
        match d {
            DType::F32 => Buf::F32(v.iter().map(|&x| x as f32).collect()),
            DType::F64 => Buf::F64(v.to_vec()),
            DType::I32 => Buf::I32(v.iter().map(|&x| x as i32).collect()),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Buf,
}

impl Tensor {
    pub fn from_f64(d: DType, shape: &[usize], v: &[f64]) -> Tensor {
        assert_eq!(shape.iter().product::<usize>(), v.len());
        Tensor { shape: shape.to_vec(), data: Buf::from_f64(d, v) }
    }
    pub fn full(d: DType, shape: &[usize], x: f64) -> Tensor {
        Tensor::from_f64(d, shape, &vec![x; shape.iter().product()])
    }
    pub fn dtype(&self) -> DType {
        self.data.dtype()
    }
    pub fn n(&self) -> usize {
        self.data.len()
    }
    pub fn is_finite(&self) -> bool {
        match &self.data {
            Buf::F32(v) => v.iter().all(|x| x.is_finite()),
            Buf::F64(v) => v.iter().all(|x| x.is_finite()),
            Buf::I32(_) => true,
        }
    }
}

/// What structural kernels (copy, slice, take, concat) need: any stored element type.
pub trait Storage: Copy + PartialOrd + Send + Sync {
    const ZERO: Self;
    fn slice(b: &Buf) -> &[Self];
    fn slice_mut(b: &mut Buf) -> &mut [Self];
}

impl Storage for i32 {
    const ZERO: Self = 0;
    #[inline(always)]
    fn slice(b: &Buf) -> &[Self] {
        match b {
            Buf::I32(v) => v,
            _ => panic!("dtype mismatch in plan"),
        }
    }
    #[inline(always)]
    fn slice_mut(b: &mut Buf) -> &mut [Self] {
        match b {
            Buf::I32(v) => v,
            _ => panic!("dtype mismatch in plan"),
        }
    }
}

/// Floating-point elements: everything arithmetic and differentiable.
pub trait Elem:
    Storage
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
{
    const ONE: Self;
    /// Bits of precision, so uniform samples are exactly representable and stay below 1.
    const MANTISSA: u32;
    const NAN: Self;
    const INF: Self;
    /// Largest `x` for which `exp(x)` stays finite; used as diagnostic evidence.
    const EXP_MAX: f64;
    fn from_f64(x: f64) -> Self;
    fn to_f64(self) -> f64;
    /// Shortest decimal form, so f32 92.7 reports as 92.7 and not 92.69999694824219.
    fn to_json_f64(self) -> f64;
    fn exp(self) -> Self;
    fn ln(self) -> Self;
    fn sqrt(self) -> Self;
    fn sin(self) -> Self;
    fn cos(self) -> Self;
    fn tanh(self) -> Self;
    fn abs(self) -> Self;
    fn powf(self, e: Self) -> Self;
    fn finite(self) -> bool;
    fn floor(self) -> Self;
    /// out[m*n] = A[m*k] @ B[k*n], row-major. `ta`/`tb`: the operand is stored transposed.
    #[allow(clippy::too_many_arguments)]
    fn gemm(m: usize, k: usize, n: usize, a: &[Self], ta: bool, b: &[Self], tb: bool, out: &mut [Self]);
    /// y[i] = f(x[i]) with the platform's vector math library when it has one.
    fn vmath(f: VFn, x: &[Self], y: &mut [Self]);
}

#[derive(Clone, Copy, Debug)]
pub enum VFn {
    Exp,
    Log,
    Sin,
    Cos,
    Tanh,
}

#[cfg(all(target_os = "macos", not(feature = "portable")))]
#[link(name = "Accelerate", kind = "framework")]
extern "C" {
    fn vvexpf(y: *mut f32, x: *const f32, n: *const i32);
    fn vvlogf(y: *mut f32, x: *const f32, n: *const i32);
    fn vvsinf(y: *mut f32, x: *const f32, n: *const i32);
    fn vvcosf(y: *mut f32, x: *const f32, n: *const i32);
    fn vvtanhf(y: *mut f32, x: *const f32, n: *const i32);
    fn vvexp(y: *mut f64, x: *const f64, n: *const i32);
    fn vvlog(y: *mut f64, x: *const f64, n: *const i32);
    fn vvsin(y: *mut f64, x: *const f64, n: *const i32);
    fn vvcos(y: *mut f64, x: *const f64, n: *const i32);
    fn vvtanh(y: *mut f64, x: *const f64, n: *const i32);
}

#[cfg(any(not(target_os = "macos"), feature = "portable"))]
fn vmath_fallback<T: Elem>(f: VFn, x: &[T], y: &mut [T]) {
    let g: fn(T) -> T = match f {
        VFn::Exp => T::exp,
        VFn::Log => T::ln,
        VFn::Sin => T::sin,
        VFn::Cos => T::cos,
        VFn::Tanh => T::tanh,
    };
    for (o, &a) in y.iter_mut().zip(x) {
        *o = g(a);
    }
}

#[cfg(all(target_os = "macos", not(feature = "portable")))]
#[link(name = "Accelerate", kind = "framework")]
extern "C" {
    #[allow(clippy::too_many_arguments)]
    fn cblas_sgemm(o: i32, ta: i32, tb: i32, m: i32, n: i32, k: i32, alpha: f32, a: *const f32, lda: i32, b: *const f32, ldb: i32, beta: f32, c: *mut f32, ldc: i32);
    #[allow(clippy::too_many_arguments)]
    fn cblas_dgemm(o: i32, ta: i32, tb: i32, m: i32, n: i32, k: i32, alpha: f64, a: *const f64, lda: i32, b: *const f64, ldb: i32, beta: f64, c: *mut f64, ldc: i32);
}

#[cfg(any(not(target_os = "macos"), feature = "portable"))]
#[allow(clippy::too_many_arguments)]
fn gemm_fallback<T: Elem>(m: usize, k: usize, n: usize, a: &[T], ta: bool, b: &[T], tb: bool, out: &mut [T]) {
    out[..m * n].fill(T::ZERO);
    for i in 0..m {
        let row = &mut out[i * n..(i + 1) * n];
        for p in 0..k {
            let x = if ta { a[p * m + i] } else { a[i * k + p] };
            for (j, o) in row.iter_mut().enumerate() {
                *o = *o + x * if tb { b[j * k + p] } else { b[p * n + j] };
            }
        }
    }
}

macro_rules! impl_elem {
    ($t:ty, $variant:ident, $exp_max:expr, $blas:ident, $vexp:ident, $vlog:ident, $vsin:ident, $vcos:ident, $vtanh:ident) => {
        impl Storage for $t {
            const ZERO: Self = 0.0;
            #[inline(always)]
            fn slice(b: &Buf) -> &[Self] {
                match b {
                    Buf::$variant(v) => v,
                    _ => panic!("dtype mismatch in plan"),
                }
            }
            #[inline(always)]
            fn slice_mut(b: &mut Buf) -> &mut [Self] {
                match b {
                    Buf::$variant(v) => v,
                    _ => panic!("dtype mismatch in plan"),
                }
            }
        }

        impl Elem for $t {
            const ONE: Self = 1.0;
            const MANTISSA: u32 = <$t>::MANTISSA_DIGITS;
            const NAN: Self = <$t>::NAN;
            const INF: Self = <$t>::INFINITY;
            const EXP_MAX: f64 = $exp_max;
            #[inline(always)]
            fn from_f64(x: f64) -> Self {
                x as $t
            }
            #[inline(always)]
            fn to_f64(self) -> f64 {
                self as f64
            }
            fn to_json_f64(self) -> f64 {
                format!("{}", self).parse().unwrap_or(f64::NAN)
            }
            #[inline(always)]
            fn exp(self) -> Self {
                <$t>::exp(self)
            }
            #[inline(always)]
            fn ln(self) -> Self {
                <$t>::ln(self)
            }
            #[inline(always)]
            fn sqrt(self) -> Self {
                <$t>::sqrt(self)
            }
            #[inline(always)]
            fn sin(self) -> Self {
                <$t>::sin(self)
            }
            #[inline(always)]
            fn cos(self) -> Self {
                <$t>::cos(self)
            }
            #[inline(always)]
            fn tanh(self) -> Self {
                <$t>::tanh(self)
            }
            #[inline(always)]
            fn abs(self) -> Self {
                <$t>::abs(self)
            }
            #[inline(always)]
            fn powf(self, e: Self) -> Self {
                <$t>::powf(self, e)
            }
            #[inline(always)]
            fn finite(self) -> bool {
                self.is_finite()
            }
            #[inline(always)]
            fn floor(self) -> Self {
                <$t>::floor(self)
            }
            fn gemm(m: usize, k: usize, n: usize, a: &[Self], ta: bool, b: &[Self], tb: bool, out: &mut [Self]) {
                #[cfg(all(target_os = "macos", not(feature = "portable")))]
                {
                    assert!(a.len() >= m * k && b.len() >= k * n && out.len() >= m * n);
                    const ROW_MAJOR: i32 = 101;
                    const NO_TRANS: i32 = 111;
                    const TRANS: i32 = 112;
                    let (fa, lda) = if ta { (TRANS, m) } else { (NO_TRANS, k) };
                    let (fb, ldb) = if tb { (TRANS, k) } else { (NO_TRANS, n) };
                    // SAFETY: bounds asserted above; the planner keeps `out` disjoint from `a` and `b`.
                    unsafe {
                        $blas(ROW_MAJOR, fa, fb, m as i32, n as i32, k as i32, 1.0, a.as_ptr(), lda as i32, b.as_ptr(), ldb as i32, 0.0, out.as_mut_ptr(), n as i32);
                    }
                }
                #[cfg(any(not(target_os = "macos"), feature = "portable"))]
                gemm_fallback(m, k, n, a, ta, b, tb, out);
            }
            fn vmath(f: VFn, x: &[Self], y: &mut [Self]) {
                #[cfg(all(target_os = "macos", not(feature = "portable")))]
                {
                    assert_eq!(x.len(), y.len());
                    let g = match f {
                        VFn::Exp => $vexp,
                        VFn::Log => $vlog,
                        VFn::Sin => $vsin,
                        VFn::Cos => $vcos,
                        VFn::Tanh => $vtanh,
                    };
                    let n = x.len() as i32;
                    // SAFETY: both slices hold exactly `n` elements and cannot overlap (&mut).
                    unsafe { g(y.as_mut_ptr(), x.as_ptr(), &n) };
                }
                #[cfg(any(not(target_os = "macos"), feature = "portable"))]
                vmath_fallback(f, x, y);
            }
        }
    };
}

impl_elem!(f32, F32, 88.722839, cblas_sgemm, vvexpf, vvlogf, vvsinf, vvcosf, vvtanhf);
impl_elem!(f64, F64, 709.782712, cblas_dgemm, vvexp, vvlog, vvsin, vvcos, vvtanh);
