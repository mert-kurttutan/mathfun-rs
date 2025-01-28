// Implementations below are inspired by the avx_mathfun project
// and changed according to several factors, (e.g. different simd extensions, correctness for large values of exp)
// avx_mathfun project: https://github.com/reyoung/avx_mathfun

#[cfg(target_arch = "aarch64")]
pub(crate) use crate::simd::Neon;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub(crate) use crate::simd::Sse2;

#[cfg(target_arch = "x86_64")]
pub(crate) use crate::simd::{Avx2Fma, Avx512f, AvxSse2};

#[cfg(target_arch = "wasm32")]
pub(crate) use crate::simd::Wasm32;

#[allow(unused)]
use crate::RUNTIME_HW_CONFIG;

#[cfg(all(feature = "std", not(target_arch = "wasm32")))]
#[derive(Clone, Copy)]
pub(crate) struct InternalPtr<T> {
    pub(crate) ptr: *const T,
}

#[cfg(all(feature = "std", not(target_arch = "wasm32")))]
// need to implement to share between threads
unsafe impl<T> Send for InternalPtr<T> {}
#[cfg(all(feature = "std", not(target_arch = "wasm32")))]
#[derive(Clone, Copy)]
pub(crate) struct InternalPtrMut<T> {
    pub(crate) ptr: *mut T,
}
#[cfg(all(feature = "std", not(target_arch = "wasm32")))]
// need to implement to share between threads
unsafe impl<T> Send for InternalPtrMut<T> {}

macro_rules! impl_unary {
    (
        $docs:tt,
        $name:ident, $dispatch_fn:ident, $ta:ty, $tb:ty,
        $($target_arch:tt | $extension:expr => $fn_simd:ident,)*

    ) => {
        fn $dispatch_fn() -> unsafe fn(usize, *const $ta, *mut $tb) {
            $(
                #[cfg(target_arch = $target_arch)]
                if $extension {
                    return $fn_simd::$name;
                }
            )*
            Fallback::$name
        }
        #[doc = $docs]
        pub unsafe fn $name(n: usize, a: *const $ta, b: *mut $tb) {
            if n == 0 {
                return;
            }
            let fn_sequantial = $dispatch_fn();
            // wasm32 does not have proper multithreading support, yet
            #[cfg(all(feature = "std", not(target_arch = "wasm32")))]
            {
                let a_ptr = InternalPtr { ptr: a };
                let b_ptr = InternalPtrMut { ptr: b };
                let host_num_threads = std::thread::available_parallelism().map_or(1, |x| x.get());
                let num_per_thread_min = 100000;
                let num_thread_max = std::env::var("MATHFUN_NUM_THREADS")
                    .ok()
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(host_num_threads);
                let n_per_thread = (n + num_thread_max - 1) / num_thread_max;
                let num_thread = if n_per_thread >= num_per_thread_min {
                    num_thread_max
                } else {
                    (n + num_per_thread_min - 1) / num_per_thread_min
                };
                let num_thread = num_thread.min(2);
                let num_per_thread = (n + num_thread - 1) / num_thread;
                std::thread::scope(|s| {
                    for i in 1..num_thread {
                        let n_cur = num_per_thread.min(n - i * num_per_thread);
                        s.spawn(move || {
                            let a_ptr = a_ptr;
                            let b_ptr = b_ptr;
                            let a_c = a_ptr.ptr.add(i * num_per_thread);
                            let b_c = b_ptr.ptr.add(i * num_per_thread);
                            fn_sequantial(n_cur, a_c, b_c);
                        });
                    }
                });
                fn_sequantial(n.min(num_per_thread), a, b);
            }
            #[cfg(any(not(feature = "std"), target_arch = "wasm32"))]
            {
                fn_sequantial(n, a, b);
            }

        }
    };
}

// add docs for vd_exp

impl_unary!(
    r"Calculates the exponential of each element in an array of f64 for length n, with the precision of 1e-6",
    vd_exp,
    dispatch_vd_exp,
    f64,
    f64,
);

impl_unary!(
    r"Calculates the exponential of each element in an array of f32 for length n, with the precision of 1e-6",
    vs_exp, dispatch_vs_exp, f32, f32,
    "x86_64" | RUNTIME_HW_CONFIG.avx512f => Avx512f,
    "x86_64" | RUNTIME_HW_CONFIG.avx2 && RUNTIME_HW_CONFIG.fma => Avx2Fma,
    "x86_64" | RUNTIME_HW_CONFIG.avx && RUNTIME_HW_CONFIG.sse2 => AvxSse2,
    "x86_64" | RUNTIME_HW_CONFIG.sse2 && RUNTIME_HW_CONFIG.sse => Sse2,
    "x86" | RUNTIME_HW_CONFIG.sse2 && RUNTIME_HW_CONFIG.sse => Sse2,
    "aarch64" | RUNTIME_HW_CONFIG.neon => Neon,
    "wasm32" | RUNTIME_HW_CONFIG.simd128 => Wasm32,
);

impl_unary!(
    r"Calculates the natural logarithm of each element in an array of f64 for length n, with the precision of 1e-6",
    vd_ln,
    dispatch_vd_ln,
    f64,
    f64,
);
impl_unary!(
    r"Calculates the natural logarithm of each element in an array of f32 for length n, with the precision of 1e-6",
    vs_ln, dispatch_vs_ln, f32, f32,
    "x86_64" | RUNTIME_HW_CONFIG.avx512f => Avx512f,
    "x86_64" | RUNTIME_HW_CONFIG.avx2 && RUNTIME_HW_CONFIG.fma => Avx2Fma,
    "x86_64" | RUNTIME_HW_CONFIG.avx && RUNTIME_HW_CONFIG.sse2 => AvxSse2,
    "x86_64" | RUNTIME_HW_CONFIG.sse2 && RUNTIME_HW_CONFIG.sse => Sse2,
    "x86" | RUNTIME_HW_CONFIG.sse2 && RUNTIME_HW_CONFIG.sse => Sse2,
    "aarch64" | RUNTIME_HW_CONFIG.neon => Neon,
    "wasm32" | RUNTIME_HW_CONFIG.simd128 => Wasm32,
);

impl_unary!(
    r"Calculates the tanh of each element in an array of f64 for length n, with the precision of 1e-6",
    vd_tanh,
    dispatch_vd_tanh,
    f64,
    f64,
);

impl_unary!(
    r"Calculates the tanh of each element in an array of f32 for length n, with the precision of 1e-6",
    vs_tanh, dispatch_vs_tanh, f32, f32,
    "x86_64" | RUNTIME_HW_CONFIG.avx512f => Avx512f,
    "x86_64" | RUNTIME_HW_CONFIG.avx2 && RUNTIME_HW_CONFIG.fma => Avx2Fma,
    "x86_64" | RUNTIME_HW_CONFIG.avx && RUNTIME_HW_CONFIG.sse2 => AvxSse2,
    "x86_64" | RUNTIME_HW_CONFIG.sse2 && RUNTIME_HW_CONFIG.sse => Sse2,
    "x86" | RUNTIME_HW_CONFIG.sse2 && RUNTIME_HW_CONFIG.sse => Sse2,
    "aarch64" | RUNTIME_HW_CONFIG.neon => Neon,
    "wasm32" | RUNTIME_HW_CONFIG.simd128 => Wasm32,
);

impl_unary!(
    r"Calculates the square root of each element in an array of f64 for length n, with the precision of 1e-6",
    vd_sqrt, dispatch_vd_sqrt, f64, f64,
    "x86_64" | RUNTIME_HW_CONFIG.avx512f => Avx512f,
    "x86_64" | RUNTIME_HW_CONFIG.avx => AvxSse2,
    "x86_64" | RUNTIME_HW_CONFIG.sse2 && RUNTIME_HW_CONFIG.sse => Sse2,
    "x86" | RUNTIME_HW_CONFIG.sse2 && RUNTIME_HW_CONFIG.sse => Sse2,
    "aarch64" | RUNTIME_HW_CONFIG.neon => Neon,
    "wasm32" | RUNTIME_HW_CONFIG.simd128 => Wasm32,
);
impl_unary!(
    r"Calculates the square root of each element in an array of f32 for length n, with the precision of 1e-6",
    vs_sqrt, dispatch_vs_sqrt, f32, f32,
    "x86_64" | RUNTIME_HW_CONFIG.avx512f => Avx512f,
    "x86_64" | RUNTIME_HW_CONFIG.avx =>  AvxSse2,
    "x86_64" | RUNTIME_HW_CONFIG.sse2 && RUNTIME_HW_CONFIG.sse => Sse2,
    "x86" | RUNTIME_HW_CONFIG.sse2 && RUNTIME_HW_CONFIG.sse => Sse2,
    "aarch64" | RUNTIME_HW_CONFIG.neon => Neon,
    "wasm32" | RUNTIME_HW_CONFIG.simd128 => Wasm32,
);

impl_unary!(
    r"Calculates the sin of each element in an array of f64 for length n, with the precision of 1e-6",
    vd_sin,
    dispatch_vd_sin,
    f64,
    f64,
);

impl_unary!(
    r"Calculates the sin of each element in an array of f32 for length n, with the precision of 1e-6",
    vs_sin, dispatch_vs_sin, f32, f32,
    "x86_64" | RUNTIME_HW_CONFIG.avx2 && RUNTIME_HW_CONFIG.fma => Avx2Fma,
    "x86_64" | RUNTIME_HW_CONFIG.avx && RUNTIME_HW_CONFIG.sse2 => AvxSse2,
    "x86_64" | RUNTIME_HW_CONFIG.sse2 && RUNTIME_HW_CONFIG.sse => Sse2,
    "x86" | RUNTIME_HW_CONFIG.sse2 && RUNTIME_HW_CONFIG.sse => Sse2,
    "aarch64" | RUNTIME_HW_CONFIG.neon => Neon,
    "wasm32" | RUNTIME_HW_CONFIG.simd128 => Wasm32,
);

impl_unary!(
    r"Calculates the cosine of each element in an array of f64 for length n, with the precision of 1e-6",
    vd_cos,
    dispatch_vd_cos,
    f64,
    f64,
);
impl_unary!(
    r"Calculates the cosine of each element in an array of f32 for length n, with the precision of 1e-6",
    vs_cos, dispatch_vs_cos, f32, f32,
    "x86_64" | RUNTIME_HW_CONFIG.avx2 && RUNTIME_HW_CONFIG.fma => Avx2Fma,
    "x86_64" | RUNTIME_HW_CONFIG.avx && RUNTIME_HW_CONFIG.sse2 => AvxSse2,
    "x86_64" | RUNTIME_HW_CONFIG.sse2 && RUNTIME_HW_CONFIG.sse => Sse2,
    "x86" | RUNTIME_HW_CONFIG.sse2 && RUNTIME_HW_CONFIG.sse => Sse2,
    "aarch64" | RUNTIME_HW_CONFIG.neon => Neon,
    "wasm32" | RUNTIME_HW_CONFIG.simd128 => Wasm32,
);

pub(crate) struct Fallback {}

macro_rules! impl_fallback {
    ($name:ident, $n:ident, $t:ty) => {
        impl Fallback {
            pub(crate) unsafe fn $name(n: usize, a: *const $t, b: *mut $t) {
                let mut i = 0;
                while i < n {
                    *b.offset(i as isize) = (*a.offset(i as isize)).$n();
                    i += 1;
                }
            }
        }
    };
}

impl_fallback!(vd_exp, exp, f64);
impl_fallback!(vs_exp, exp, f32);

impl_fallback!(vd_ln, ln, f64);
impl_fallback!(vs_ln, ln, f32);

impl_fallback!(vd_tanh, tanh, f64);
impl_fallback!(vs_tanh, tanh, f32);

impl_fallback!(vd_sqrt, sqrt, f64);
impl_fallback!(vs_sqrt, sqrt, f32);

impl_fallback!(vd_sin, sin, f64);
impl_fallback!(vs_sin, sin, f32);

impl_fallback!(vd_cos, cos, f64);
impl_fallback!(vs_cos, cos, f32);

#[cfg(test)]
mod tests {
    use super::*;
    extern crate std;

    use std::{vec, vec::Vec};

    struct F32Pool {
        a: Vec<f32>,
        b: Vec<f32>,
        // n: usize,
    }

    impl F32Pool {
        pub(crate) fn new() -> Self {
            Self {
                a: vec![0.0; 1 << 20],
                b: vec![0.0; 1 << 20],
                // n: 12,
            }
        }

        pub(crate) fn set_interval(&mut self, n: usize) {
            assert!(n <= 1 << 12);
            let bit_start = n * self.a.len();
            for (i, a) in self.a.iter_mut().enumerate() {
                *a = f32::from_bits(bit_start as u32 + i as u32);
            }
        }

        pub(crate) fn get_interval(&mut self) -> (&[f32], &mut [f32]) {
            (&self.a, &mut self.b)
        }

        pub(crate) fn len(&self) -> usize {
            assert_eq!(self.a.len(), self.b.len());
            self.a.len()
        }
    }

    fn check_nan_inf(y_result: f64, y: f64) -> bool {
        // nan check
        if y_result.is_nan() {
            return y.is_nan();
        }
        // inf check
        if y_result > f32::MAX as f64 {
            return y == f64::INFINITY;
        }
        if y_result < f32::MIN as f64 {
            return y == f64::NEG_INFINITY;
        }
        return false;
    }
    fn check_exp(x: f32, y: f32) {
        let x = x as f64;
        let y = y as f64;

        let y_result = x.exp();

        let nan_inf = check_nan_inf(y_result, y);
        let diff = (y_result - y).abs();
        let rel_diff = diff / y.abs().max(1.0);
        let rel_diff = (rel_diff * 1e8).round() / 1e8;
        assert!(rel_diff <= 1e-6 || nan_inf, "{} != {}, x: {}, diff: {}", y_result, y, x, rel_diff);
    }

    fn check_tanh(x: f32, y: f32) {
        let x = x as f64;
        let y = y as f64;

        let y_result = x.tanh();

        let nan_inf = check_nan_inf(y_result, y);
        let diff = (y_result - y).abs();
        let rel_diff = (diff * 1e8).round() / 1e8;
        assert!(rel_diff <= 1e-6 || nan_inf, "{} != {}, x: {}, diff: {}", y_result, y, x, diff);
    }

    fn check_ln(x: f32, y: f32) {
        let x = x as f64;
        let y = y as f64;

        let y_result = x.ln();

        let nan_inf = check_nan_inf(y_result, y);
        let diff = (y_result - y).abs();
        let rel_diff = diff / y.abs().max(1.0);
        assert!(rel_diff <= 1e-6 || nan_inf, "{} != {}, x: {}, diff: {}", y_result, y, x, rel_diff);
    }

    fn check_sin(x: f32, y: f32) {
        let x = x as f64;
        let y = y as f64;

        let y_result = x.sin();
        let too_large = x.abs() > 2f64.powi(16);

        let nan_inf = check_nan_inf(y_result, y);
        let diff = (y_result - y).abs();
        let rel_diff = diff;
        assert!(rel_diff < 1e-6 || nan_inf || too_large, "{} != {}, x: {}, diff: {}", y_result, y, x, diff);
        // let x_64 = x as f64;
        // // let y = x.sin();
        // // let y = y as f64;
        // if x.is_nan() {
        //     assert!(y.is_nan(), "{} != {}, x: {}, x_tanh: {}", x, y, x, x.sin());
        // } else if x.is_infinite() {
        //     assert!(y.is_nan());
        // } else {
        //     let x_exp = x_64.sin();
        //     if x_exp > f32::MAX as f64 {
        //         assert!(y.is_nan(), "{} != {}, x: {}", x_exp, y, x);
        //     } else {
        //         let diff = (x_exp - y as f64).abs();
        //         let rel_diff = diff;
        //         // round rel_diff to 1e-7 precision
        //         let rel_diff_round = (rel_diff * 1e7).round() / 1e7;
        //         // pass if x is bigger than 2**16
        //         let pass = x_64.abs() > 2f64.powi(16);
        //         assert!(rel_diff_round <= 1e-6 ||pass, "{} != {}, x: {}, diff: {}", x_64.sin(), y, x, rel_diff);
        //     };

        // }
    }
    fn check_cos(x: f32, y: f32) {
        let x = x as f64;
        let y = y as f64;
        let too_large = x.abs() > 2f64.powi(16);
        let y_result = x.cos();

        let nan_inf = check_nan_inf(y_result, y);
        let diff = (y_result - y).abs();
        let rel_diff = diff;
        assert!(rel_diff < 1e-6 || nan_inf || too_large, "{} != {}, x: {}, diff: {}", y_result, y, x, diff);
    }

    fn check_sqrt(x: f32, y: f32) {
        let x = x as f64;
        let y = y as f64;

        let y_result = x.sqrt();

        let is_x_neg = x < 0.0;

        let nan_inf = check_nan_inf(y_result, y);
        let diff = (y_result - y).abs();
        let rel_diff = diff / (y.abs().max(1.0));
        assert!(rel_diff < 1e-6 || is_x_neg || nan_inf, "{} != {}, x: {}, diff: {}", y_result, y, x, diff);
    }

    #[test]
    fn accuracy_sqrt() {
        let mut pool = F32Pool::new();
        let a_len = pool.len();

        for a_i in 0..1 << 12 {
            pool.set_interval(a_i);
            let (a, b) = pool.get_interval();
            unsafe {
                vs_sqrt(a_len, a.as_ptr(), b.as_mut_ptr());
            }
            for i in 0..a_len {
                check_sqrt(a[i], b[i]);
            }
        }
    }

    #[test]
    fn accuracy_sin() {
        let mut pool = F32Pool::new();
        let a_len = pool.len();

        for a_i in 0..1 << 12 {
            pool.set_interval(a_i);
            let (a, b) = pool.get_interval();
            unsafe {
                vs_sin(a_len, a.as_ptr(), b.as_mut_ptr());
            }
            for i in 0..a_len {
                check_sin(a[i], b[i]);
            }
        }
    }

    #[test]
    fn accuracy_cos() {
        let mut pool = F32Pool::new();
        let a_len = pool.len();

        for a_i in 0..1 << 12 {
            pool.set_interval(a_i);
            let (a, b) = pool.get_interval();
            unsafe {
                vs_cos(a_len, a.as_ptr(), b.as_mut_ptr());
            }
            for i in 0..a_len {
                check_cos(a[i], b[i]);
            }
        }
    }

    #[test]
    fn accuracy_exp() {
        let mut pool = F32Pool::new();
        let a_len = pool.len();

        for a_i in 0..1 << 12 {
            pool.set_interval(a_i);
            let (a, b) = pool.get_interval();
            unsafe {
                vs_exp(a_len, a.as_ptr(), b.as_mut_ptr());
            }
            for i in 0..a_len {
                check_exp(a[i], b[i]);
            }
        }
    }

    #[test]
    fn accuracy_tanh() {
        let mut pool = F32Pool::new();
        let a_len = pool.len();

        for a_i in 0..1 << 12 {
            pool.set_interval(a_i);
            let (a, b) = pool.get_interval();
            unsafe {
                vs_tanh(a_len, a.as_ptr(), b.as_mut_ptr());
            }
            for i in 0..a_len {
                check_tanh(a[i], b[i]);
            }
        }
    }
    #[test]
    fn accuracy_ln() {
        let mut pool = F32Pool::new();
        let a_len = pool.len();

        for a_i in 0..1 << 12 {
            pool.set_interval(a_i);
            let (a, b) = pool.get_interval();
            unsafe {
                vs_ln(a_len, a.as_ptr(), b.as_mut_ptr());
            }
            for i in 0..a_len {
                check_ln(a[i], b[i]);
            }
        }
    }
    // use libc::{c_float, c_int};

    // use once_cell::sync::Lazy;
    // use rand::rngs::StdRng;
    // use rand::{Rng, SeedableRng};
    // static MATHFUN_LIBRARY: Lazy<libloading::Library> = Lazy::new(|| unsafe {
    //     #[cfg(target_os = "windows")]
    //     let default_cblas_path = format!("{PROJECT_DIR}/mkl_rt.2.dll");
    //     #[cfg(target_os = "linux")]
    //     let default_cblas_path = format!("{PROJECT_DIR}/../../.venv/lib/libmkl_rt.so.2");
    //     let cblas_path = std::env::var("PIRE_MATHFUN_PATH").unwrap_or(default_cblas_path);
    //     libloading::Library::new(cblas_path).unwrap()
    // });
    // type UnaryFnF32 = unsafe extern "C" fn(c_int, *const c_float, *mut c_float);

    // fn random_matrix_std<T>(arr: &mut [T])
    // where
    //     rand::distributions::Standard: rand::prelude::Distribution<T>,
    // {
    //     let mut x = StdRng::seed_from_u64(43);
    //     arr.iter_mut().for_each(|p| *p = x.gen::<T>());
    // }
    // static VS_COS_MKL: Lazy<libloading::Symbol<'static, UnaryFnF32>> = Lazy::new(|| unsafe {
    //     let unary_fn = MATHFUN_LIBRARY.get(b"vsExp").unwrap();
    //     unary_fn
    // });

    // unsafe fn vs_cos_mkl(n: c_int, a: *const c_float, y: *mut c_float) {
    //     VS_COS_MKL(n, a, y);
    // }
    // #[test]
    // fn mkl_test() {
    //     let m = 1 << 31 - 1;
    //     let mut a = vec![1.0; m];
    //     let mut b = vec![1.0; m];
    //     random_matrix_std(&mut a);
    //     random_matrix_std(&mut b);
    //     let a_len = a.len();
    //     let a_len_i32 = a_len as i32;
    //     println!("a_len_i32: {}", a_len_i32);
    //     let t0 = std::time::Instant::now();
    //     unsafe {
    //         vs_cos_mkl(a_len_i32, a.as_ptr(), b.as_mut_ptr());
    //         // vs_cos(a_len, a.as_ptr(), b.as_mut_ptr());
    //     }
    //     let t1 = std::time::Instant::now();
    //     println!("time: {:?}", t1 - t0);
    // }
}
