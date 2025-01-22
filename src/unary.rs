// Implementations below are inspired by the avx_mathfun project
// and changed according to several factors, (e.g. different simd extensions, correctness for large values of exp)
// avx_mathfun project: https://github.com/reyoung/avx_mathfun

use crate::RUNTIME_HW_CONFIG;

macro_rules! impl_unary {
    (
        $name:ident, $dispatch_fn:ident, $ta:ty, $tb:ty,
        $fallback:ident, $($target_arch:tt | $extension:expr => $fn_simd:ident,)*

    ) => {
        fn $dispatch_fn() -> unsafe fn(usize, *const $ta, *mut $tb) {
            $(
                #[cfg(target_arch = $target_arch)]
                if $extension {
                    return $fn_simd;
                }
            )*
            $fallback
        }
        pub unsafe fn $name(n: usize, a: *const $ta, y: *mut $tb) {
            if n == 0 {
                return;
            }
            let fn_sequantial = $dispatch_fn();
            let num_per_thread_min = 100000;
            let host_num_threads = std::thread::available_parallelism().map_or(1, |x| x.get());
            // read MATHFUN_NUM_THREADS env variable
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
            // upper bound by 2
            let num_thread = num_thread.min(2);
            let num_per_thread = (n + num_thread - 1) / num_thread;
            let a_usize = a as usize;
            let y_usize = y as usize;
            let x = std::thread::spawn(move || {
                for i in 1..num_thread {
                    let a = a_usize as *const $ta;
                    let y = y_usize as *mut $tb;
                    let n_cur = num_per_thread.min(n - i * num_per_thread);
                    fn_sequantial(
                        n_cur,
                        a.offset((i * num_per_thread) as isize),
                        y.offset((i * num_per_thread) as isize),
                    );
                }
            });
            fn_sequantial(n.min(num_per_thread), a, y);
            x.join().unwrap();
        }
    };
}

macro_rules! impl_cos {
    ($simd:tt, $features:tt, $fn_name:ident) => {
        impl $simd {
            #[target_feature(enable = $features)]
            unsafe fn vs_cos(n: usize, a: *const f32, b: *mut f32) {
                const DP1: f32 = -0.78515625;
                const DP2: f32 = -2.4187564849853515625e-4;
                const DP3: f32 = -3.77489497744594108e-8;

                const P0: f32 = 1.0;
                const P1: f32 = -1.6666654611E-1;
                const P2: f32 = 8.3321608736E-3;
                const P3: f32 = -1.9515295891E-4;
                // const FOPI: f32 = 0.63661977236*2.0;
                const FOPI: f32 = 1.27323954474;
                const Q1: f32 = -0.5;
                const Q2: f32 = 4.1666645683E-2;
                const Q3: f32 = -1.3887316255E-3;
                const Q4: f32 = 2.443315711809948E-5;

                let dp1 = Self::set1_f32(DP1);
                let dp2 = Self::set1_f32(DP2);
                let dp3 = Self::set1_f32(DP3);

                let fopi = Self::set1_f32(FOPI);

                let p0 = Self::set1_f32(P0);
                let p1 = Self::set1_f32(P1);
                let p2 = Self::set1_f32(P2);
                let p3 = Self::set1_f32(P3);

                let q1 = Self::set1_f32(Q1);
                let q2 = Self::set1_f32(Q2);
                let q3 = Self::set1_f32(Q3);
                let q4 = Self::set1_f32(Q4);

                let inv_sign_mask = Self::set1_i32(!0x80000000u32 as i32);

                let inv_sign_mask = Self::cast_i32_f32(inv_sign_mask);

                let mut i = 0;
                while (i + Self::F32_WIDTH) <= n {
                    let x = Self::loadu_f32(a.offset(i as isize));
                    let x = Self::and_f32(x, inv_sign_mask);
                    let mut y = Self::mul_f32(x, fopi);
                    let mut imm2 = Self::cvt_f32_i32(y);
                    imm2 = Self::add_i32(imm2, Self::set1_i32(1));
                    imm2 = Self::and_i32(imm2, Self::set1_i32(!1));
                    y = Self::cvt_i32_f32(imm2);
                    imm2 = Self::sub_i32(imm2, Self::set1_i32(2));
                    let imm0 = Self::andnot_i32(imm2, Self::set1_i32(4));
                    let imm0 = Self::slli_i32::<29>(imm0);
                    let imm2 = Self::and_i32(imm2, Self::set1_i32(2));
                    let imm2 = Self::cmpeq_i32(imm2, Self::set1_i32(0));
                    let sign_bit = Self::cast_i32_f32(imm0);
                    let poly_mask = Self::cast_i32_f32(imm2);
                    let x = Self::fmadd_f32(y, dp1, x);
                    let x = Self::fmadd_f32(y, dp2, x);
                    let x = Self::fmadd_f32(y, dp3, x);
                    let mut y = q4;
                    let z = Self::mul_f32(x, x);
                    y = Self::fmadd_f32(y, z, q3);
                    y = Self::fmadd_f32(y, z, q2);
                    y = Self::fmadd_f32(y, z, q1);
                    y = Self::fmadd_f32(y, z, p0);
                    let mut y2 = p3;
                    y2 = Self::fmadd_f32(y2, z, p2);
                    y2 = Self::fmadd_f32(y2, z, p1);
                    y2 = Self::fmadd_f32(y2, z, p0);
                    y2 = Self::mul_f32(y2, x);
                    let y2 = Self::and_f32(poly_mask, y2);
                    let y = Self::andnot_f32(poly_mask, y);
                    let y = Self::add_f32(y, y2);
                    let y = Self::xor_f32(y, sign_bit);
                    Self::storeu_f32(b.offset(i as isize), y);
                    i += Self::F32_WIDTH;
                }
                while i < n {
                    *b.offset(i as isize) = (*a.offset(i as isize)).cos();
                    i += 1;
                }
            }
        }

        pub(crate) unsafe fn $fn_name(n: usize, a: *const f32, b: *mut f32) {
            $simd::vs_cos(n, a, b);
        }
    };
}

macro_rules! impl_sin {
    ($simd:tt, $features:tt, $fn_name:ident) => {
        impl $simd {
            #[target_feature(enable = $features)]
            unsafe fn vs_sin(n: usize, a: *const f32, b: *mut f32) {
                // define constants

                const DP1: f32 = -0.78515625;
                const DP2: f32 = -2.4187564849853515625e-4;
                const DP3: f32 = -3.77489497744594108e-8;

                const P0: f32 = 1.0;
                const P1: f32 = -1.6666654611E-1;
                const P2: f32 = 8.3321608736E-3;
                const P3: f32 = -1.9515295891E-4;
                // const FOPI: f32 = 0.63661977236*2.0;
                const FOPI: f32 = 1.27323954474;
                const Q1: f32 = -0.5;
                const Q2: f32 = 4.1666645683E-2;
                const Q3: f32 = -1.3887316255E-3;
                const Q4: f32 = 2.443315711809948E-5;

                let dp1 = Self::set1_f32(DP1);
                let dp2 = Self::set1_f32(DP2);
                let dp3 = Self::set1_f32(DP3);

                let fopi = Self::set1_f32(FOPI);

                let p0 = Self::set1_f32(P0);
                let p1 = Self::set1_f32(P1);
                let p2 = Self::set1_f32(P2);
                let p3 = Self::set1_f32(P3);

                let q1 = Self::set1_f32(Q1);
                let q2 = Self::set1_f32(Q2);
                let q3 = Self::set1_f32(Q3);
                let q4 = Self::set1_f32(Q4);

                let sign_mask = Self::set1_i32(0x80000000u32 as i32);
                let inv_sign_mask = Self::set1_i32(!0x80000000u32 as i32);

                let sign_mask = Self::cast_i32_f32(sign_mask);
                let inv_sign_mask = Self::cast_i32_f32(inv_sign_mask);

                let mut i = 0;
                while (i + Self::F32_WIDTH) <= n {
                    let x = Self::loadu_f32(a.offset(i as isize));
                    let mut sign_bit = x;
                    let x = Self::and_f32(x, inv_sign_mask);
                    // extract the sign bit (upper one)
                    sign_bit = Self::and_f32(sign_bit, sign_mask);

                    //scale by 4/Pi
                    let mut y = Self::mul_f32(x, fopi);

                    // store the integer part of y in mm0
                    let mut imm2 = Self::cvt_f32_i32(y);
                    // j=(j+1) & (~1) (see the cephes sources)
                    // another two AVX2 instruction
                    imm2 = Self::add_i32(imm2, Self::set1_i32(1));
                    imm2 = Self::and_i32(imm2, Self::set1_i32(!1));
                    y = Self::cvt_i32_f32(imm2);

                    // get the swap sign flag
                    let imm0 = Self::and_i32(imm2, Self::set1_i32(4));
                    let imm0 = Self::slli_i32::<29>(imm0);
                    // get the polynom selection mask
                    // there is one polynom for 0 <= x <= Pi/4
                    // and another one for Pi/4<x<=Pi/2
                    let imm2 = Self::and_i32(imm2, Self::set1_i32(2));
                    let imm2 = Self::cmpeq_i32(imm2, Self::set1_i32(0));
                    let swap_sign_bit = Self::cast_i32_f32(imm0);
                    let poly_mask = Self::cast_i32_f32(imm2);
                    let sign_bit = Self::xor_f32(sign_bit, swap_sign_bit);

                    // The magic pass: "Extended precision modular arithmetic"
                    // x = ((x - y * DP1) - y * DP2) - y * DP3;
                    let x = Self::fmadd_f32(y, dp1, x);
                    let x = Self::fmadd_f32(y, dp2, x);
                    let x = Self::fmadd_f32(y, dp3, x);

                    // Evaluate the first polynom  (0 <= x <= Pi/4)
                    let mut y = q4;
                    let z = Self::mul_f32(x, x);
                    y = Self::fmadd_f32(y, z, q3);
                    y = Self::fmadd_f32(y, z, q2);
                    y = Self::fmadd_f32(y, z, q1);
                    y = Self::fmadd_f32(y, z, p0);

                    // evaluate the second polynom (Pi/4 <= x <= 0)
                    let mut y2 = p3;
                    y2 = Self::fmadd_f32(y2, z, p2);
                    y2 = Self::fmadd_f32(y2, z, p1);
                    y2 = Self::fmadd_f32(y2, z, p0);
                    y2 = Self::mul_f32(y2, x);

                    // select the correct result from the two polynoms
                    let y2 = Self::and_f32(poly_mask, y2);
                    let y = Self::andnot_f32(poly_mask, y);
                    let y = Self::add_f32(y, y2);
                    // update the sign
                    let y = Self::xor_f32(y, sign_bit);

                    Self::storeu_f32(b.offset(i as isize), y);
                    i += Self::F32_WIDTH;
                }
                while i < n {
                    *b.offset(i as isize) = (*a.offset(i as isize)).sin();
                    i += 1;
                }
            }
        }

        pub(crate) unsafe fn $fn_name(n: usize, a: *const f32, b: *mut f32) {
            $simd::vs_sin(n, a, b);
        }
    };
}

macro_rules! impl_ln {
    ($simd:tt, $features:tt, $fn_name:ident) => {
        impl $simd {
            #[target_feature(enable = $features)]
            unsafe fn vs_ln(n: usize, a: *const f32, b: *mut f32) {
                // define constants
                const LN2F_HI: f32 = 0.693359375;
                const LN2F_LO: f32 = -2.12194440E-4;
                const P0LOGF: f32 = -0.5;
                const P1LOGF: f32 = 3.3333331174E-1;
                const P2LOGF: f32 = -2.4999993993E-1;
                const P3LOGF: f32 = 2.0000714765E-1;
                const P4LOGF: f32 = -1.6666657665E-1;
                const P5LOGF: f32 = 1.4249322787E-1;
                const P6LOGF: f32 = -1.250000140846E-1;
                const P7LOGF: f32 = 1.1676998740E-1;
                // const P8LOGF: f32 = -1.1514610310E-1;
                // const P9LOGF: f32 = 7.0376836292E-2;

                let ln2f_hi = Self::set1_f32(LN2F_HI + LN2F_LO);
                let p0logf = Self::set1_f32(P0LOGF);
                let p1logf = Self::set1_f32(P1LOGF);
                let p2logf = Self::set1_f32(P2LOGF);
                let p3logf = Self::set1_f32(P3LOGF);
                let p4logf = Self::set1_f32(P4LOGF);
                let p5logf = Self::set1_f32(P5LOGF);
                let p6logf = Self::set1_f32(P6LOGF);
                let p7logf = Self::set1_f32(P7LOGF);
                // let p8logf = Self::set1_f32(P8LOGF);
                // let p9logf = Self::set1_f32(P9LOGF);
                // let mantissa_th = Self::set1_f32(1.41421356237*0.5);
                let mantissa_th = Self::set1_f32(0.73337);
                let one = Self::set1_f32(1.0);
                let onehalf = Self::set1_f32(1.5);
                let log2onehalf = Self::set1_f32(0.58496250072);

                // ops name always end with _f32 not _ps
                let mut i = 0;
                while (i + Self::F32_WIDTH) <= n {
                    let x = Self::loadu_f32(a.offset(i as isize));

                    let (mut exp, mut x) = Self::get_exp_mant_f32(x);

                    let mantissa_mask = Self::cmp_f32::<_CMP_LT_OQ>(x, mantissa_th);
                    // masked doubling only for mantissa < mantissa_th (by adding to itself)
                    x = Self::mask_mul_f32(mantissa_mask, x, onehalf);
                    exp = Self::add_f32(exp, one);
                    exp = Self::mask_sub_f32(mantissa_mask, exp, log2onehalf);

                    x = Self::sub_f32(x, one);

                    let mut y = p7logf;
                    y = Self::fmadd_f32(y, x, p6logf);
                    y = Self::fmadd_f32(y, x, p5logf);
                    y = Self::fmadd_f32(y, x, p4logf);
                    y = Self::fmadd_f32(y, x, p3logf);
                    y = Self::fmadd_f32(y, x, p2logf);
                    y = Self::fmadd_f32(y, x, p1logf);
                    y = Self::fmadd_f32(y, x, p0logf);
                    y = Self::fmadd_f32(y, x, one);
                    y = Self::mul_f32(y, x);

                    let r = Self::fmadd_f32(ln2f_hi, exp, y);

                    Self::storeu_f32(b.offset(i as isize), r);
                    i += Self::F32_WIDTH;
                }

                while i < n {
                    *b.offset(i as isize) = (*a.offset(i as isize)).ln();
                    i += 1;
                }
            }
        }

        pub(crate) unsafe fn $fn_name(n: usize, a: *const f32, b: *mut f32) {
            $simd::vs_ln(n, a, b);
        }
    };
}

macro_rules! impl_exp {
    ($simd:tt, $features:tt, $fn_name:ident, FMA) => {
        impl $simd {
            #[target_feature(enable = $features)]
            unsafe fn vs_exp(n: usize, a: *const f32, b: *mut f32) {
                // define constants
                const EXP_HI: f32 = 88.7228391117 * 1.0;
                const EXP_LO: f32 = -88.7228391117 * 1.0;
                const LOG2EF: f32 = 1.44269504088896341;
                const INV_LOG2EF: f32 = 0.69314718056;
                const EXP_P0: f32 = 0.00032712723;
                const EXP_P1: f32 = 0.00228989065 + 1e-6;
                // const EXP_P1: f32 = 0.00138888888;
                const EXP_P2: f32 = 0.01373934392;
                const EXP_P3: f32 = 0.06869671961;
                const EXP_P4: f32 = 0.27478687845;
                const EXP_P5: f32 = 0.82436063535;

                let exp_hi = Self::set1_f32(EXP_HI);
                let exp_lo = Self::set1_f32(EXP_LO);
                let log2ef = Self::set1_f32(LOG2EF);

                let half = Self::set1_f32(0.5);

                let exp_p0 = Self::set1_f32(EXP_P0);
                let exp_p1 = Self::set1_f32(EXP_P1);
                let exp_p2 = Self::set1_f32(EXP_P2);
                let exp_p3 = Self::set1_f32(EXP_P3);
                let exp_p4 = Self::set1_f32(EXP_P4);
                let exp_p5 = Self::set1_f32(EXP_P5);
                let min_exponent = Self::set1_f32(-127.0);
                let e_sqrt = Self::set1_f32(1.6487212707);

                let inv_c2 = Self::set1_f32(-INV_LOG2EF);

                let mut i = 0;
                while (i + Self::F32_WIDTH) <= n {
                    let x = Self::loadu_f32(a.offset(i as isize));

                    // Clamp x
                    let x = Self::min_f32(exp_hi, x);
                    let x = Self::max_f32(exp_lo, x);

                    // Compute fx = floor(x * log2ef + 0.5)
                    let mut fx = Self::mul_f32(x, log2ef);
                    // use to zero rounding since nearest int is problematic for near overflow and underflowing values
                    fx = Self::round_f32::<{ _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC }>(fx);
                    // to prevent denormalized values
                    fx = Self::max_f32(fx, min_exponent);

                    // Approximation for exp(x)
                    let x = Self::fmadd_f32(fx, inv_c2, x);

                    let x = Self::sub_f32(x, half);

                    let mut y = exp_p0;
                    y = Self::fmadd_f32(y, x, exp_p1);
                    y = Self::fmadd_f32(y, x, exp_p2);
                    y = Self::fmadd_f32(y, x, exp_p3);
                    y = Self::fmadd_f32(y, x, exp_p4);
                    y = Self::fmadd_f32(y, x, exp_p5);
                    y = Self::fmadd_f32(y, x, e_sqrt);
                    y = Self::fmadd_f32(y, x, e_sqrt);

                    // Compute 2^fx
                    let mut imm0 = Self::cvt_f32_i32(fx);
                    imm0 = Self::add_i32(imm0, Self::set1_i32(0x7f));
                    imm0 = Self::slli_i32::<23>(imm0);
                    let pow2n = Self::cast_i32_f32(imm0);

                    // Final result
                    let r = Self::mul_f32(y, pow2n);
                    Self::storeu_f32(b.offset(i as isize), r);
                    i += Self::F32_WIDTH;
                }
                while i < n {
                    *b.offset(i as isize) = (*a.offset(i as isize)).exp();
                    i += 1;
                }
            }
        }

        pub(crate) unsafe fn $fn_name(n: usize, a: *const f32, b: *mut f32) {
            $simd::vs_exp(n, a, b);
        }
    };
    ($simd:tt, $features:tt, $fn_name:ident, NO_FMA) => {
        impl $simd {
            #[target_feature(enable = $features)]
            unsafe fn vs_exp(n: usize, a: *const f32, b: *mut f32) {
                // define constants
                const EXP_HI: f32 = 88.7228391117 * 1.0;
                const EXP_LO: f32 = -88.7228391117 * 1.0;
                const LOG2EF: f32 = 1.44269504088896341;
                const EXP_P0: f32 = 0.00032712723;
                const EXP_P1: f32 = 0.00228989065 + 1e-6;
                // const EXP_P1: f32 = 0.00138888888;
                const EXP_P2: f32 = 0.01373934392;
                const EXP_P3: f32 = 0.06869671961;
                const EXP_P4: f32 = 0.27478687845;
                const EXP_P5: f32 = 0.82436063535;
                const L2_U: f32 = -0.693_145_751_953_125;
                const L2_L: f32 = -1.428_606_765_330_187_045_e-6;

                let exp_hi = Self::set1_f32(EXP_HI);
                let exp_lo = Self::set1_f32(EXP_LO);
                let log2ef = Self::set1_f32(LOG2EF);

                let half = Self::set1_f32(0.5);

                let exp_p0 = Self::set1_f32(EXP_P0);
                let exp_p1 = Self::set1_f32(EXP_P1);
                let exp_p2 = Self::set1_f32(EXP_P2);
                let exp_p3 = Self::set1_f32(EXP_P3);
                let exp_p4 = Self::set1_f32(EXP_P4);
                let exp_p5 = Self::set1_f32(EXP_P5);
                let min_exponent = Self::set1_f32(-127.0);
                let e_sqrt = Self::set1_f32(1.6487212707);

                let l2_u = Self::set1_f32(L2_U);
                let l2_l = Self::set1_f32(L2_L);

                let mut i = 0;
                while (i + Self::F32_WIDTH) <= n {
                    let x = Self::loadu_f32(a.offset(i as isize));

                    // Clamp x
                    let x = Self::min_f32(exp_hi, x);
                    let x = Self::max_f32(exp_lo, x);

                    // Compute fx = floor(x * log2ef + 0.5)
                    let mut fx = Self::mul_f32(x, log2ef);
                    // use to zero rounding since nearest int is problematic for near overflow and underflowing values
                    fx = Self::round_f32::<{ _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC }>(fx);
                    // to prevent denormalized values
                    fx = Self::max_f32(fx, min_exponent);

                    // Approximation for exp(x)
                    let x = Self::fmadd_f32(fx, l2_u, x);
                    let x = Self::fmadd_f32(fx, l2_l, x);

                    let x = Self::sub_f32(x, half);

                    let mut y = exp_p0;
                    y = Self::fmadd_f32(y, x, exp_p1);
                    y = Self::fmadd_f32(y, x, exp_p2);
                    y = Self::fmadd_f32(y, x, exp_p3);
                    y = Self::fmadd_f32(y, x, exp_p4);
                    y = Self::fmadd_f32(y, x, exp_p5);
                    y = Self::fmadd_f32(y, x, e_sqrt);
                    y = Self::fmadd_f32(y, x, e_sqrt);

                    // Compute 2^fx
                    let mut imm0 = Self::cvt_f32_i32(fx);
                    imm0 = Self::add_i32(imm0, Self::set1_i32(0x7f));
                    imm0 = Self::slli_i32::<23>(imm0);
                    let pow2n = Self::cast_i32_f32(imm0);

                    // Final result
                    let r = Self::mul_f32(y, pow2n);
                    Self::storeu_f32(b.offset(i as isize), r);
                    i += Self::F32_WIDTH;
                }
                while i < n {
                    *b.offset(i as isize) = (*a.offset(i as isize)).exp();
                    i += 1;
                }
            }
        }

        pub(crate) unsafe fn $fn_name(n: usize, a: *const f32, b: *mut f32) {
            $simd::vs_exp(n, a, b);
        }
    };
}

macro_rules! impl_tanh {
    ($simd:tt, $features:tt, $fn_name:ident) => {
        impl $simd {
            #[target_feature(enable = $features)]
            unsafe fn vs_tanh(n: usize, a: *const f32, b: *mut f32) {
                // define constants
                const EXP_HI: f32 = 88.7228391117 * 1.0;
                const EXP_LO: f32 = -88.7228391117 * 1.0;
                const LOG2EF: f32 = 1.44269504088896341;
                const INV_LOG2EF: f32 = 0.69314718056;
                const EXP_P0: f32 = 0.00032712723;
                const EXP_P1: f32 = 0.00228989065;
                // const EXP_P1: f32 = 0.00138888888;
                const EXP_P2: f32 = 0.01373934392;
                const EXP_P3: f32 = 0.06869671961;
                const EXP_P4: f32 = 0.27478687845;
                const EXP_P5: f32 = 0.82436063535;

                let exp_hi = Self::set1_f32(EXP_HI);
                let exp_lo = Self::set1_f32(EXP_LO);
                let log2ef = Self::set1_f32(LOG2EF);
                let inv_c2 = Self::set1_f32(-INV_LOG2EF);

                let half = Self::set1_f32(0.5);

                let one = Self::set1_f32(1.0);
                let one_neg = Self::set1_f32(-1.0);

                let exp_p0 = Self::set1_f32(EXP_P0);
                let exp_p1 = Self::set1_f32(EXP_P1);
                let exp_p2 = Self::set1_f32(EXP_P2);
                let exp_p3 = Self::set1_f32(EXP_P3);
                let exp_p4 = Self::set1_f32(EXP_P4);
                let exp_p5 = Self::set1_f32(EXP_P5);
                let min_exponent = Self::set1_f32(-127.0);
                let e_sqrt = Self::set1_f32(1.6487212707);

                let mut i = 0;
                while (i + Self::F32_WIDTH) <= n {
                    let x = Self::loadu_f32(a.offset(i as isize));
                    let x0 = x;
                    let x0 = Self::max_f32(exp_hi, x0);
                    let x = Self::min_f32(exp_hi, x);
                    let x = Self::max_f32(exp_lo, x);

                    let fx = Self::mul_f32(x, log2ef);
                    let fx = Self::round_f32::<{ _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC }>(fx);
                    let fx = Self::max_f32(fx, min_exponent);

                    let x = Self::fmadd_f32(fx, inv_c2, x);
                    let x = Self::sub_f32(x, half);

                    let mut y = exp_p0;
                    y = Self::fmadd_f32(y, x, exp_p1);
                    y = Self::fmadd_f32(y, x, exp_p2);
                    y = Self::fmadd_f32(y, x, exp_p3);
                    y = Self::fmadd_f32(y, x, exp_p4);
                    y = Self::fmadd_f32(y, x, exp_p5);
                    y = Self::fmadd_f32(y, x, e_sqrt);
                    y = Self::fmadd_f32(y, x, e_sqrt);

                    let imm0 = Self::cvt_f32_i32(fx);
                    let imm0 = Self::add_i32(imm0, Self::set1_i32(0x7f));
                    let imm0 = Self::slli_i32::<23>(imm0);
                    let pow2n = Self::cast_i32_f32(imm0);

                    let r = Self::mul_f32(y, pow2n);
                    let r = Self::mul_f32(r, r);

                    let numerator = Self::sub_f32(r, one);
                    let denominator = Self::add_f32(r, one);
                    let mut r = Self::div_f32(numerator, denominator);
                    r = Self::min_f32(r, one);
                    r = Self::max_f32(r, one_neg);
                    r = Self::min_f32(r, x0);

                    Self::storeu_f32(b.offset(i as isize), r);
                    i += Self::F32_WIDTH;
                }
                while i < n {
                    *b.offset(i as isize) = (*a.offset(i as isize)).tanh();
                    i += 1;
                }
            }
        }

        pub(crate) unsafe fn $fn_name(n: usize, a: *const f32, b: *mut f32) {
            $simd::vs_tanh(n, a, b);
        }
    };
}

impl_unary!(vd_exp, dispatch_vd_exp, f64, f64, vd_exp_fallback,);

impl_unary!(
    vs_exp, dispatch_vs_exp, f32, f32,
    vs_exp_fallback,
    "x86_64" | RUNTIME_HW_CONFIG.avx512f => vs_exp_avx512f_asm,
    "x86_64" | RUNTIME_HW_CONFIG.avx2 && RUNTIME_HW_CONFIG.fma => vs_exp_avx2_fma,
    "x86_64" | RUNTIME_HW_CONFIG.avx && RUNTIME_HW_CONFIG.sse2 => vs_exp_avx_sse2,
    "x86_64" | RUNTIME_HW_CONFIG.sse2 && RUNTIME_HW_CONFIG.sse => vs_exp_sse2,
    "x86" | RUNTIME_HW_CONFIG.sse => vs_exp_sse2,
);

impl_unary!(vd_ln, dispatch_vd_ln, f64, f64, vd_ln_fallback,);
impl_unary!(
    vs_ln, dispatch_vs_ln, f32, f32,
    vs_ln_fallback,
    "x86_64" | RUNTIME_HW_CONFIG.avx512f => vs_ln_avx512f_asm,
    "x86_64" | RUNTIME_HW_CONFIG.avx2 && RUNTIME_HW_CONFIG.fma => vs_ln_avx2_fma,
    "x86_64" | RUNTIME_HW_CONFIG.avx && RUNTIME_HW_CONFIG.sse2 => vs_ln_avx_sse2,
    "x86_64" | RUNTIME_HW_CONFIG.sse2 && RUNTIME_HW_CONFIG.sse => vs_ln_sse2,
    "x86" | RUNTIME_HW_CONFIG.sse => vs_ln_sse2,
);

impl_unary!(vd_tanh, dispatch_vd_tanh, f64, f64, vd_tanh_fallback,);

impl_unary!(
    vs_tanh, dispatch_vs_tanh, f32, f32,
    vs_tanh_fallback,
    "x86_64" | RUNTIME_HW_CONFIG.avx512f => vs_tanh_avx512f_asm,
    "x86_64" | RUNTIME_HW_CONFIG.avx2 && RUNTIME_HW_CONFIG.fma => vs_tanh_avx2_fma,
    "x86_64" | RUNTIME_HW_CONFIG.avx && RUNTIME_HW_CONFIG.sse2 => vs_tanh_avx_sse2,
    "x86_64" | RUNTIME_HW_CONFIG.sse2 && RUNTIME_HW_CONFIG.sse => vs_tanh_sse2,
    "x86" | RUNTIME_HW_CONFIG.sse => vs_tanh_sse2,
);

impl_unary!(
    vd_sqrt, dispatch_vd_sqrt, f64, f64,
    vd_sqrt_fallback,
    "x86_64" | RUNTIME_HW_CONFIG.avx512f => vd_sqrt_avx512f_asm,
    "x86_64" | RUNTIME_HW_CONFIG.avx => vd_sqrt_avx,
);
impl_unary!(
    vs_sqrt, dispatch_vs_sqrt, f32, f32,
    vs_sqrt_fallback,
    "x86_64" | RUNTIME_HW_CONFIG.avx512f => vs_sqrt_avx512f_asm,
    "x86_64" | RUNTIME_HW_CONFIG.avx => vs_sqrt_avx,
);

impl_unary!(vd_sin, dispatch_vd_sin, f64, f64, vd_sin_fallback,);
impl_unary!(
    vs_sin, dispatch_vs_sin, f32, f32,
    vs_sin_fallback,
    "x86_64" | RUNTIME_HW_CONFIG.avx2 && RUNTIME_HW_CONFIG.fma => vs_sin_avx2_fma,
    "x86_64" | RUNTIME_HW_CONFIG.avx && RUNTIME_HW_CONFIG.sse2 => vs_sin_avx_sse2,
    "x86_64" | RUNTIME_HW_CONFIG.sse2 && RUNTIME_HW_CONFIG.sse => vs_sin_sse2,
    "x86" | RUNTIME_HW_CONFIG.sse => vs_sin_sse2,
);

impl_unary!(vd_cos, dispatch_vd_cos, f64, f64, vd_cos_fallback,);
impl_unary!(
    vs_cos, dispatch_vs_cos, f32, f32,
    vs_cos_fallback,
    "x86_64" | RUNTIME_HW_CONFIG.avx2 && RUNTIME_HW_CONFIG.fma => vs_cos_avx2_fma,
    "x86_64" | RUNTIME_HW_CONFIG.avx && RUNTIME_HW_CONFIG.sse2 => vs_cos_avx_sse2,
    "x86_64" | RUNTIME_HW_CONFIG.sse2 && RUNTIME_HW_CONFIG.sse => vs_cos_sse2,
    "x86" | RUNTIME_HW_CONFIG.sse => vs_cos_sse2,
);

#[cfg(target_arch = "x86_64")]
mod avx2_fma_mod {
    use core::arch::x86_64::*;
    pub(crate) struct Avx2Fma {}

    impl Avx2Fma {
        const F32_WIDTH: usize = 8;
        #[inline(always)]
        unsafe fn set1_f32(x: f32) -> __m256 {
            _mm256_set1_ps(x)
        }
        #[inline(always)]
        unsafe fn set1_i32(x: i32) -> __m256i {
            _mm256_set1_epi32(x)
        }

        #[inline(always)]
        unsafe fn loadu_f32(ptr: *const f32) -> __m256 {
            _mm256_loadu_ps(ptr)
        }

        #[inline(always)]
        unsafe fn and_f32(a: __m256, b: __m256) -> __m256 {
            _mm256_and_ps(a, b)
        }

        #[inline(always)]
        unsafe fn mul_f32(a: __m256, b: __m256) -> __m256 {
            _mm256_mul_ps(a, b)
        }

        #[inline(always)]
        unsafe fn add_i32(a: __m256i, b: __m256i) -> __m256i {
            _mm256_add_epi32(a, b)
        }

        #[inline(always)]
        unsafe fn and_i32(a: __m256i, b: __m256i) -> __m256i {
            _mm256_and_si256(a, b)
        }
        #[inline(always)]
        unsafe fn cvt_i32_f32(a: __m256i) -> __m256 {
            _mm256_cvtepi32_ps(a)
        }
        #[inline(always)]
        unsafe fn cvt_f32_i32(a: __m256) -> __m256i {
            _mm256_cvtps_epi32(a)
        }

        #[inline(always)]
        unsafe fn sub_i32(a: __m256i, b: __m256i) -> __m256i {
            _mm256_sub_epi32(a, b)
        }

        #[inline(always)]
        unsafe fn andnot_i32(a: __m256i, b: __m256i) -> __m256i {
            _mm256_andnot_si256(a, b)
        }

        #[inline(always)]
        unsafe fn slli_i32<const IMM8: i32>(a: __m256i) -> __m256i {
            _mm256_slli_epi32(a, IMM8)
        }

        #[inline(always)]
        unsafe fn cmpeq_i32(a: __m256i, b: __m256i) -> __m256i {
            _mm256_cmpeq_epi32(a, b)
        }

        #[inline(always)]
        unsafe fn cast_i32_f32(a: __m256i) -> __m256 {
            _mm256_castsi256_ps(a)
        }

        #[inline(always)]
        unsafe fn fmadd_f32(a: __m256, b: __m256, c: __m256) -> __m256 {
            _mm256_fmadd_ps(a, b, c)
        }

        #[inline(always)]
        unsafe fn andnot_f32(a: __m256, b: __m256) -> __m256 {
            _mm256_andnot_ps(a, b)
        }

        #[inline(always)]
        unsafe fn add_f32(a: __m256, b: __m256) -> __m256 {
            _mm256_add_ps(a, b)
        }

        #[inline(always)]
        unsafe fn xor_f32(a: __m256, b: __m256) -> __m256 {
            _mm256_xor_ps(a, b)
        }

        #[inline(always)]
        unsafe fn storeu_f32(ptr: *mut f32, a: __m256) {
            _mm256_storeu_ps(ptr, a)
        }

        #[inline(always)]
        unsafe fn sub_f32(a: __m256, b: __m256) -> __m256 {
            _mm256_sub_ps(a, b)
        }

        #[inline(always)]
        unsafe fn cmp_f32<const IMM8: i32>(a: __m256, b: __m256) -> __m256 {
            _mm256_cmp_ps(a, b, IMM8)
        }

        #[inline(always)]
        unsafe fn mask_mul_f32(mask: __m256, a: __m256, b: __m256) -> __m256 {
            let one = _mm256_set1_ps(1.0);
            let one = _mm256_andnot_ps(mask, one);
            let masked_one = _mm256_and_ps(b, mask);
            let masked_b = _mm256_or_ps(masked_one, one);
            let c = _mm256_mul_ps(a, masked_b);
            c
        }

        #[inline(always)]
        unsafe fn mask_sub_f32(mask: __m256, a: __m256, b: __m256) -> __m256 {
            let masked_b = _mm256_and_ps(b, mask);
            let c = _mm256_sub_ps(a, masked_b);
            c
        }

        #[inline(always)]
        unsafe fn or_f32(a: __m256, b: __m256) -> __m256 {
            _mm256_or_ps(a, b)
        }

        #[inline(always)]
        unsafe fn mask_add_f32(mask: __m256, a: __m256, b: __m256) -> __m256 {
            let masked_b = _mm256_and_ps(b, mask);
            let c = _mm256_add_ps(a, masked_b);
            c
        }

        #[inline(always)]
        unsafe fn cast_f32_i32(a: __m256) -> __m256i {
            _mm256_castps_si256(a)
        }

        #[inline(always)]
        unsafe fn srli_i32<const IMM8: i32>(a: __m256i) -> __m256i {
            _mm256_srli_epi32(a, IMM8)
        }

        #[inline(always)]
        unsafe fn min_f32(a: __m256, b: __m256) -> __m256 {
            _mm256_min_ps(a, b)
        }

        #[inline(always)]
        unsafe fn round_f32<const IMM8: i32>(a: __m256) -> __m256 {
            _mm256_round_ps(a, IMM8)
        }

        #[inline(always)]
        unsafe fn max_f32(a: __m256, b: __m256) -> __m256 {
            _mm256_max_ps(a, b)
        }

        #[inline(always)]
        unsafe fn div_f32(a: __m256, b: __m256) -> __m256 {
            _mm256_div_ps(a, b)
        }

        #[inline(always)]
        unsafe fn get_exp_mant_f32(a: __m256) -> (__m256, __m256) {
            let a_0 = a;
            let zero_mask = Self::cmp_f32::<_CMP_EQ_OS>(a, Self::set1_f32(0.0));
            let nan_mask = Self::cmp_f32::<_CMP_LT_OS>(a, Self::set1_f32(0.0));
            let inv_mant_mask = Self::cast_i32_f32(Self::set1_i32(!0x7f800000));
            let inf_mask = Self::cmp_f32::<_CMP_EQ_OS>(a, Self::set1_f32(f32::INFINITY));
            let denorm_mul = Self::set1_f32(134217730.);
            let denorm_th = Self::set1_f32(1.1754945e-38);
            let denorm_mask = Self::cmp_f32::<_CMP_LT_OS>(a, denorm_th);
            let mut a = Self::mask_mul_f32(denorm_mask, a, denorm_mul);

            let mut imm0 = Self::srli_i32::<23>(Self::cast_f32_i32(a));

            /* keep only the fractional part */
            a = Self::and_f32(a, inv_mant_mask);
            a = Self::or_f32(a, Self::set1_f32(0.5));

            // this is again another AVX2 instruction
            imm0 = Self::sub_i32(imm0, Self::set1_i32(0x7f));

            let e = Self::cvt_i32_f32(imm0);

            let e = Self::mask_sub_f32(denorm_mask, e, Self::set1_f32(27.0));
            let e = Self::mask_sub_f32(zero_mask, e, Self::set1_f32(f32::INFINITY));
            let e = Self::mask_add_f32(inf_mask, e, Self::set1_f32(f32::INFINITY));
            let e = Self::min_f32(e, a_0);
            let e = Self::mask_add_f32(nan_mask, e, Self::set1_f32(f32::NAN));

            (e, a)
        }
    }

    impl_cos!(Avx2Fma, "avx,avx2,fma", vs_cos_avx2_fma);
    impl_sin!(Avx2Fma, "avx,avx2,fma", vs_sin_avx2_fma);
    impl_ln!(Avx2Fma, "avx,avx2,fma", vs_ln_avx2_fma);
    impl_exp!(Avx2Fma, "avx,avx2,fma", vs_exp_avx2_fma, FMA);
    impl_tanh!(Avx2Fma, "avx,avx2,fma", vs_tanh_avx2_fma);
}
#[cfg(target_arch = "x86_64")]
pub(crate) use avx2_fma_mod::*;

#[cfg(target_arch = "x86_64")]
mod avx_sse2_mod {
    use core::arch::x86_64::*;
    pub(crate) struct AvxSse2 {}

    impl AvxSse2 {
        const F32_WIDTH: usize = 8;
        #[inline(always)]
        unsafe fn set1_f32(x: f32) -> __m256 {
            _mm256_set1_ps(x)
        }
        #[inline(always)]
        unsafe fn set1_i32(x: i32) -> __m256i {
            _mm256_set1_epi32(x)
        }

        #[inline(always)]
        unsafe fn loadu_f32(ptr: *const f32) -> __m256 {
            _mm256_loadu_ps(ptr)
        }

        #[inline(always)]
        unsafe fn and_f32(a: __m256, b: __m256) -> __m256 {
            _mm256_and_ps(a, b)
        }

        #[inline(always)]
        unsafe fn mul_f32(a: __m256, b: __m256) -> __m256 {
            _mm256_mul_ps(a, b)
        }

        #[inline(always)]
        unsafe fn add_i32(a: __m256i, b: __m256i) -> __m256i {
            // extract second half of a and b
            let a1 = _mm256_extractf128_si256(a, 1);
            let b1 = _mm256_extractf128_si256(b, 1);
            let a0 = _mm256_castsi256_si128(a);
            let b0 = _mm256_castsi256_si128(b);
            let c0 = _mm_add_epi32(a0, b0);
            let c1 = _mm_add_epi32(a1, b1);
            _mm256_insertf128_si256(_mm256_castsi128_si256(c0), c1, 1)
        }

        #[inline(always)]
        unsafe fn and_i32(a: __m256i, b: __m256i) -> __m256i {
            // extract second half of a and b
            let a1 = _mm256_extractf128_si256(a, 1);
            let b1 = _mm256_extractf128_si256(b, 1);
            let a0 = _mm256_castsi256_si128(a);
            let b0 = _mm256_castsi256_si128(b);
            let c0 = _mm_and_si128(a0, b0);
            let c1 = _mm_and_si128(a1, b1);
            _mm256_insertf128_si256(_mm256_castsi128_si256(c0), c1, 1)
        }
        #[inline(always)]
        unsafe fn cvt_i32_f32(a: __m256i) -> __m256 {
            _mm256_cvtepi32_ps(a)
        }
        #[inline(always)]
        unsafe fn cvt_f32_i32(a: __m256) -> __m256i {
            _mm256_cvtps_epi32(a)
        }

        #[inline(always)]
        unsafe fn sub_i32(a: __m256i, b: __m256i) -> __m256i {
            // extract second half of a and b
            let a1 = _mm256_extractf128_si256(a, 1);
            let b1 = _mm256_extractf128_si256(b, 1);
            let a0 = _mm256_castsi256_si128(a);
            let b0 = _mm256_castsi256_si128(b);
            let c0 = _mm_sub_epi32(a0, b0);
            let c1 = _mm_sub_epi32(a1, b1);
            _mm256_insertf128_si256(_mm256_castsi128_si256(c0), c1, 1)
        }

        #[inline(always)]
        unsafe fn andnot_i32(a: __m256i, b: __m256i) -> __m256i {
            // extract second half of a and b
            let a1 = _mm256_extractf128_si256(a, 1);
            let b1 = _mm256_extractf128_si256(b, 1);
            let a0 = _mm256_castsi256_si128(a);
            let b0 = _mm256_castsi256_si128(b);
            let c0 = _mm_andnot_si128(a0, b0);
            let c1 = _mm_andnot_si128(a1, b1);
            _mm256_insertf128_si256(_mm256_castsi128_si256(c0), c1, 1)
        }

        #[inline(always)]
        unsafe fn slli_i32<const IMM8: i32>(a: __m256i) -> __m256i {
            // extract second half of a and b
            let a1 = _mm256_extractf128_si256(a, 1);
            let a0 = _mm256_castsi256_si128(a);
            let c0 = _mm_slli_epi32(a0, IMM8);
            let c1 = _mm_slli_epi32(a1, IMM8);
            _mm256_insertf128_si256(_mm256_castsi128_si256(c0), c1, 1)
        }

        #[inline(always)]
        unsafe fn cmpeq_i32(a: __m256i, b: __m256i) -> __m256i {
            // extract second half of a and b
            let a1 = _mm256_extractf128_si256(a, 1);
            let b1 = _mm256_extractf128_si256(b, 1);
            let a0 = _mm256_castsi256_si128(a);
            let b0 = _mm256_castsi256_si128(b);
            let c0 = _mm_cmpeq_epi32(a0, b0);
            let c1 = _mm_cmpeq_epi32(a1, b1);
            _mm256_insertf128_si256(_mm256_castsi128_si256(c0), c1, 1)
        }

        #[inline(always)]
        unsafe fn cast_i32_f32(a: __m256i) -> __m256 {
            _mm256_castsi256_ps(a)
        }

        #[inline(always)]
        unsafe fn fmadd_f32(a: __m256, b: __m256, c: __m256) -> __m256 {
            let mul = _mm256_mul_ps(a, b);
            _mm256_add_ps(mul, c)
        }

        #[inline(always)]
        unsafe fn andnot_f32(a: __m256, b: __m256) -> __m256 {
            _mm256_andnot_ps(a, b)
        }

        #[inline(always)]
        unsafe fn add_f32(a: __m256, b: __m256) -> __m256 {
            _mm256_add_ps(a, b)
        }

        #[inline(always)]
        unsafe fn xor_f32(a: __m256, b: __m256) -> __m256 {
            _mm256_xor_ps(a, b)
        }

        #[inline(always)]
        unsafe fn storeu_f32(ptr: *mut f32, a: __m256) {
            _mm256_storeu_ps(ptr, a)
        }

        #[inline(always)]
        unsafe fn sub_f32(a: __m256, b: __m256) -> __m256 {
            _mm256_sub_ps(a, b)
        }

        #[inline(always)]
        unsafe fn cmp_f32<const IMM8: i32>(a: __m256, b: __m256) -> __m256 {
            _mm256_cmp_ps(a, b, IMM8)
        }

        #[inline(always)]
        unsafe fn mask_mul_f32(mask: __m256, a: __m256, b: __m256) -> __m256 {
            let one = _mm256_set1_ps(1.0);
            let one = _mm256_andnot_ps(mask, one);
            let masked_one = _mm256_and_ps(b, mask);
            let masked_b = _mm256_or_ps(masked_one, one);
            let c = _mm256_mul_ps(a, masked_b);
            c
        }

        #[inline(always)]
        unsafe fn mask_sub_f32(mask: __m256, a: __m256, b: __m256) -> __m256 {
            let masked_b = _mm256_and_ps(b, mask);
            let c = _mm256_sub_ps(a, masked_b);
            c
        }

        #[inline(always)]
        unsafe fn or_f32(a: __m256, b: __m256) -> __m256 {
            _mm256_or_ps(a, b)
        }

        #[inline(always)]
        unsafe fn mask_add_f32(mask: __m256, a: __m256, b: __m256) -> __m256 {
            let masked_b = _mm256_and_ps(b, mask);
            let c = _mm256_add_ps(a, masked_b);
            c
        }

        #[inline(always)]
        unsafe fn cast_f32_i32(a: __m256) -> __m256i {
            _mm256_castps_si256(a)
        }

        #[inline(always)]
        unsafe fn srli_i32<const IMM8: i32>(a: __m256i) -> __m256i {
            // extract second half of a
            let a1 = _mm256_extractf128_si256(a, 1);
            let a0 = _mm256_castsi256_si128(a);
            let c0 = _mm_srli_epi32(a0, IMM8);
            let c1 = _mm_srli_epi32(a1, IMM8);
            _mm256_insertf128_si256(_mm256_castsi128_si256(c0), c1, 1)
        }

        #[inline(always)]
        unsafe fn min_f32(a: __m256, b: __m256) -> __m256 {
            _mm256_min_ps(a, b)
        }

        #[inline(always)]
        unsafe fn round_f32<const IMM8: i32>(a: __m256) -> __m256 {
            _mm256_round_ps(a, IMM8)
        }

        #[inline(always)]
        unsafe fn max_f32(a: __m256, b: __m256) -> __m256 {
            _mm256_max_ps(a, b)
        }

        #[inline(always)]
        unsafe fn div_f32(a: __m256, b: __m256) -> __m256 {
            _mm256_div_ps(a, b)
        }

        #[inline(always)]
        unsafe fn get_exp_mant_f32(a: __m256) -> (__m256, __m256) {
            let a_0 = a;
            let zero_mask = Self::cmp_f32::<_CMP_EQ_OS>(a, Self::set1_f32(0.0));
            let nan_mask = Self::cmp_f32::<_CMP_LT_OS>(a, Self::set1_f32(0.0));
            let inv_mant_mask = Self::cast_i32_f32(Self::set1_i32(!0x7f800000));
            let inf_mask = Self::cmp_f32::<_CMP_EQ_OS>(a, Self::set1_f32(f32::INFINITY));
            let denorm_mul = Self::set1_f32(134217730.);
            let denorm_th = Self::set1_f32(1.1754945e-38);
            let denorm_mask = Self::cmp_f32::<_CMP_LT_OS>(a, denorm_th);
            let mut a = Self::mask_mul_f32(denorm_mask, a, denorm_mul);

            let mut imm0 = Self::srli_i32::<23>(Self::cast_f32_i32(a));

            /* keep only the fractional part */
            a = Self::and_f32(a, inv_mant_mask);
            a = Self::or_f32(a, Self::set1_f32(0.5));

            // this is again another AVX2 instruction
            imm0 = Self::sub_i32(imm0, Self::set1_i32(0x7f));

            let e = Self::cvt_i32_f32(imm0);

            let e = Self::mask_sub_f32(denorm_mask, e, Self::set1_f32(27.0));
            let e = Self::mask_sub_f32(zero_mask, e, Self::set1_f32(f32::INFINITY));
            let e = Self::mask_add_f32(inf_mask, e, Self::set1_f32(f32::INFINITY));
            let e = Self::min_f32(e, a_0);
            let e = Self::mask_add_f32(nan_mask, e, Self::set1_f32(f32::NAN));

            (e, a)
        }
    }

    impl_cos!(AvxSse2, "avx,sse2", vs_cos_avx_sse2);
    impl_sin!(AvxSse2, "avx,sse2", vs_sin_avx_sse2);
    impl_ln!(AvxSse2, "avx,sse2", vs_ln_avx_sse2);
    impl_exp!(AvxSse2, "avx,sse2", vs_exp_avx_sse2, NO_FMA);
    impl_tanh!(AvxSse2, "avx,sse2", vs_tanh_avx_sse2);

    #[target_feature(enable = "avx")]
    pub(crate) unsafe fn vs_sqrt_avx(n: usize, a: *const f32, b: *mut f32) {
        const NR: usize = 8;
        let mut i = 0;
        while (i + NR) <= n {
            let x = _mm256_loadu_ps(a.offset(i as isize));
            let y = _mm256_sqrt_ps(x);
            _mm256_storeu_ps(b.offset(i as isize), y);
            i += NR;
        }
        while i < n {
            *b.offset(i as isize) = (*a.offset(i as isize)).sqrt();
            i += 1;
        }
    }

    #[target_feature(enable = "avx")]
    pub(crate) unsafe fn vd_sqrt_avx(n: usize, a: *const f64, b: *mut f64) {
        const NR: usize = 4;
        let mut i = 0;
        while (i + NR) <= n {
            let x = _mm256_loadu_pd(a.offset(i as isize));
            let y = _mm256_sqrt_pd(x);
            _mm256_storeu_pd(b.offset(i as isize), y);
            i += NR;
        }
        while i < n {
            *b.offset(i as isize) = (*a.offset(i as isize)).sqrt();
            i += 1;
        }
    }
}
#[cfg(target_arch = "x86_64")]
pub(crate) use avx_sse2_mod::*;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod sse2_mod {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::*;

    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::*;
    pub(crate) struct Sse2 {}

    impl Sse2 {
        const F32_WIDTH: usize = 4;
        #[inline(always)]
        unsafe fn set1_f32(x: f32) -> __m128 {
            _mm_set1_ps(x)
        }
        #[inline(always)]
        unsafe fn set1_i32(x: i32) -> __m128i {
            _mm_set1_epi32(x)
        }

        #[inline(always)]
        unsafe fn loadu_f32(ptr: *const f32) -> __m128 {
            _mm_loadu_ps(ptr)
        }

        #[inline(always)]
        unsafe fn and_f32(a: __m128, b: __m128) -> __m128 {
            _mm_and_ps(a, b)
        }

        #[inline(always)]
        unsafe fn mul_f32(a: __m128, b: __m128) -> __m128 {
            _mm_mul_ps(a, b)
        }

        #[inline(always)]
        unsafe fn add_i32(a: __m128i, b: __m128i) -> __m128i {
            _mm_add_epi32(a, b)
        }

        #[inline(always)]
        unsafe fn sub_i32(a: __m128i, b: __m128i) -> __m128i {
            _mm_sub_epi32(a, b)
        }

        #[inline(always)]
        unsafe fn and_i32(a: __m128i, b: __m128i) -> __m128i {
            _mm_and_si128(a, b)
        }
        #[inline(always)]
        unsafe fn cvt_i32_f32(a: __m128i) -> __m128 {
            _mm_cvtepi32_ps(a)
        }
        #[inline(always)]
        unsafe fn cvt_f32_i32(a: __m128) -> __m128i {
            _mm_cvtps_epi32(a)
        }

        #[inline(always)]
        unsafe fn andnot_i32(a: __m128i, b: __m128i) -> __m128i {
            _mm_andnot_si128(a, b)
        }

        #[inline(always)]
        unsafe fn slli_i32<const IMM8: i32>(a: __m128i) -> __m128i {
            _mm_slli_epi32(a, IMM8)
        }

        #[inline(always)]
        unsafe fn cmpeq_i32(a: __m128i, b: __m128i) -> __m128i {
            _mm_cmpeq_epi32(a, b)
        }

        #[inline(always)]
        unsafe fn cast_i32_f32(a: __m128i) -> __m128 {
            _mm_castsi128_ps(a)
        }

        #[inline(always)]
        unsafe fn fmadd_f32(a: __m128, b: __m128, c: __m128) -> __m128 {
            let mul = _mm_mul_ps(a, b);
            _mm_add_ps(mul, c)
        }

        #[inline(always)]
        unsafe fn andnot_f32(a: __m128, b: __m128) -> __m128 {
            _mm_andnot_ps(a, b)
        }

        #[inline(always)]
        unsafe fn add_f32(a: __m128, b: __m128) -> __m128 {
            _mm_add_ps(a, b)
        }

        #[inline(always)]
        unsafe fn xor_f32(a: __m128, b: __m128) -> __m128 {
            _mm_xor_ps(a, b)
        }

        #[inline(always)]
        unsafe fn storeu_f32(ptr: *mut f32, a: __m128) {
            _mm_storeu_ps(ptr, a)
        }

        #[inline(always)]
        unsafe fn sub_f32(a: __m128, b: __m128) -> __m128 {
            _mm_sub_ps(a, b)
        }

        #[inline(always)]
        unsafe fn cmp_f32<const IMM8: i32>(a: __m128, b: __m128) -> __m128 {
            _mm_cmp_ps(a, b, IMM8)
        }

        #[inline(always)]
        unsafe fn mask_mul_f32(mask: __m128, a: __m128, b: __m128) -> __m128 {
            let one = _mm_set1_ps(1.0);
            let one = _mm_andnot_ps(mask, one);
            let masked_one = _mm_and_ps(b, mask);
            let masked_b = _mm_or_ps(masked_one, one);
            let c = _mm_mul_ps(a, masked_b);
            c
        }

        #[inline(always)]
        unsafe fn mask_sub_f32(mask: __m128, a: __m128, b: __m128) -> __m128 {
            let masked_b = _mm_and_ps(b, mask);
            let c = _mm_sub_ps(a, masked_b);
            c
        }

        #[inline(always)]
        unsafe fn or_f32(a: __m128, b: __m128) -> __m128 {
            _mm_or_ps(a, b)
        }

        #[inline(always)]
        unsafe fn mask_add_f32(mask: __m128, a: __m128, b: __m128) -> __m128 {
            let masked_b = _mm_and_ps(b, mask);
            let c = _mm_add_ps(a, masked_b);
            c
        }

        #[inline(always)]
        unsafe fn cast_f32_i32(a: __m128) -> __m128i {
            _mm_castps_si128(a)
        }

        #[inline(always)]
        unsafe fn srli_i32<const IMM8: i32>(a: __m128i) -> __m128i {
            _mm_srli_epi32(a, IMM8)
        }

        #[inline(always)]
        unsafe fn min_f32(a: __m128, b: __m128) -> __m128 {
            _mm_min_ps(a, b)
        }

        #[inline(always)]
        unsafe fn round_f32<const IMM8: i32>(a: __m128) -> __m128 {
            _mm_round_ps(a, IMM8)
        }

        #[inline(always)]
        unsafe fn max_f32(a: __m128, b: __m128) -> __m128 {
            _mm_max_ps(a, b)
        }

        #[inline(always)]
        unsafe fn div_f32(a: __m128, b: __m128) -> __m128 {
            _mm_div_ps(a, b)
        }

        #[inline(always)]
        unsafe fn get_exp_mant_f32(a: __m128) -> (__m128, __m128) {
            let a_0 = a;
            let zero_mask = Self::cmp_f32::<_CMP_EQ_OS>(a, Self::set1_f32(0.0));
            let nan_mask = Self::cmp_f32::<_CMP_LT_OS>(a, Self::set1_f32(0.0));
            let inv_mant_mask = Self::cast_i32_f32(Self::set1_i32(!0x7f800000));
            let inf_mask = Self::cmp_f32::<_CMP_EQ_OS>(a, Self::set1_f32(f32::INFINITY));
            let denorm_mul = Self::set1_f32(134217730.);
            let denorm_th = Self::set1_f32(1.1754945e-38);
            let denorm_mask = Self::cmp_f32::<_CMP_LT_OS>(a, denorm_th);
            let mut a = Self::mask_mul_f32(denorm_mask, a, denorm_mul);

            let mut imm0 = Self::srli_i32::<23>(Self::cast_f32_i32(a));

            /* keep only the fractional part */
            a = Self::and_f32(a, inv_mant_mask);
            a = Self::or_f32(a, Self::set1_f32(0.5));

            // this is again another AVX2 instruction
            imm0 = Self::sub_i32(imm0, Self::set1_i32(0x7f));

            let e = Self::cvt_i32_f32(imm0);

            let e = Self::mask_sub_f32(denorm_mask, e, Self::set1_f32(27.0));
            let e = Self::mask_sub_f32(zero_mask, e, Self::set1_f32(f32::INFINITY));
            let e = Self::mask_add_f32(inf_mask, e, Self::set1_f32(f32::INFINITY));
            let e = Self::min_f32(e, a_0);
            let e = Self::mask_add_f32(nan_mask, e, Self::set1_f32(f32::NAN));

            (e, a)
        }
    }

    impl_cos!(Sse2, "sse,sse2", vs_cos_sse2);
    impl_sin!(Sse2, "sse,sse2", vs_sin_sse2);
    impl_ln!(Sse2, "sse,sse2", vs_ln_sse2);
    impl_exp!(Sse2, "sse,sse2", vs_exp_sse2, NO_FMA);
    impl_tanh!(Sse2, "sse,sse2", vs_tanh_sse2);
}
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub(crate) use sse2_mod::*;

#[cfg(target_arch = "x86_64")]
mod avx512f_mod {
    use core::arch::asm;
    pub(crate) unsafe fn vs_ln_avx512f_asm(n: usize, a: *const f32, b: *mut f32) {
        const NR: usize = 16;
        // define constants
        const LN2F_HI: f32 = 0.693359375;
        const LN2F_LO: f32 = -2.12194440E-4;
        const P0LOGF: f32 = -0.5;
        const P1LOGF: f32 = 3.3333331174E-1;
        const P2LOGF: f32 = -2.4999993993E-1;
        const P3LOGF: f32 = 2.0000714765E-1;
        const P4LOGF: f32 = -1.6666657665E-1;
        const P5LOGF: f32 = 1.4249322787E-1;
        const P6LOGF: f32 = -1.250000140846E-1;
        const P7LOGF: f32 = 1.1676998740E-1;
        // const P8LOGF: f32 = -1.1514610310E-1;
        // const P9LOGF: f32 = 7.0376836292E-2;

        let constant_arr = [
            0.73337,
            1.5,
            1.0,
            -0.58496250072,
            -1.0,
            P7LOGF,
            P6LOGF,
            P5LOGF,
            P4LOGF,
            P3LOGF,
            P2LOGF,
            P1LOGF,
            P0LOGF,
            LN2F_HI + LN2F_LO,
            f32::NAN,
        ];

        let mut i = 0;
        asm!(
            "vxorps %zmm0, %zmm0, %zmm0",
            "vbroadcastss 0({constant_arrx}), %zmm1",
            "vbroadcastss 4({constant_arrx}), %zmm2",
            "vbroadcastss 8({constant_arrx}), %zmm3",
            "vbroadcastss 12({constant_arrx}), %zmm4",
            "vbroadcastss 16({constant_arrx}), %zmm5",

            "vbroadcastss 20({constant_arrx}), %zmm6",
            "vbroadcastss 24({constant_arrx}), %zmm7",
            "vbroadcastss 28({constant_arrx}), %zmm8",
            "vbroadcastss 32({constant_arrx}), %zmm9",
            "vbroadcastss 36({constant_arrx}), %zmm10",
            "vbroadcastss 40({constant_arrx}), %zmm11",
            "vbroadcastss 44({constant_arrx}), %zmm12",
            "vbroadcastss 48({constant_arrx}), %zmm13",

            "vbroadcastss 52({constant_arrx}), %zmm14",

            "test {nx:e}, {nx:e}",
            "je 3f",

            "2:",
            "vmovups ({ax}), %zmm15",
            "vcmpltps        %zmm0, %zmm15, %k1",
            "vgetexpps       %zmm15, %zmm16",
            "vgetmantps      $2, %zmm15, %zmm15",
            "vcmplt_oqps     %zmm1, %zmm15, %k2",
            "vmulps  %zmm2, %zmm15, %zmm15 {{%k2}}",
            "vaddps  %zmm3, %zmm16, %zmm16",
            "vaddps  %zmm4, %zmm16, %zmm16 {{%k2}}",
            "vaddps  %zmm5, %zmm15, %zmm15",
            "vmovaps %zmm6, %zmm17",
            "vfmadd213ps     %zmm7, %zmm15, %zmm17",
            "vfmadd213ps     %zmm8, %zmm15, %zmm17",
            "vfmadd213ps     %zmm9, %zmm15, %zmm17",
            "vfmadd213ps     %zmm10, %zmm15, %zmm17",
            "vfmadd213ps     %zmm11, %zmm15, %zmm17",
            "vfmadd213ps     %zmm12, %zmm15, %zmm17",
            "vfmadd213ps     %zmm13, %zmm15, %zmm17",
            "vfmadd213ps     %zmm3, %zmm15, %zmm17",
            "vmulps  %zmm17, %zmm15, %zmm15",
            "vfmadd231ps     %zmm14, %zmm16, %zmm15",
            "vbroadcastss    56({constant_arrx}), %zmm15 {{%k1}}",
            "vmovups %zmm15, ({bx})",
            "add  $64, {ax}",
            "add  $64, {bx}",
            "add $16, {ix:e}",
            "cmp {nx:e}, {ix:e}",
            "jl 2b",

            "3:",
            "vzeroupper",

            constant_arrx = in(reg) &constant_arr,
            ax = inout(reg) a => _,
            bx = inout(reg) b => _,
            ix = inout(reg) i => i,
            nx = inout(reg) n / NR * NR => _,
            out("zmm0") _, out("zmm1") _, out("zmm2") _, out("zmm3") _, out("zmm4") _, out("zmm5") _, out("zmm6") _, out("zmm7") _, out("zmm8") _, out("zmm9") _,
            out("zmm10") _, out("zmm11") _, out("zmm12") _, out("zmm13") _, out("zmm14") _, out("zmm15") _, out("zmm16") _, out("zmm17") _, out("zmm18") _, out("zmm19") _,
            out("zmm20") _, out("zmm21") _, out("zmm22") _, out("zmm23") _, out("zmm24") _, out("zmm25") _, out("zmm26") _, out("zmm27") _, out("zmm28") _, out("zmm29") _,
            out("zmm30") _, out("zmm31") _, out("k1") _, out("k2") _,
            options(att_syntax)
        );
        while i < n {
            *b.offset(i as isize) = (*a.offset(i as isize)).ln();
            i += 1;
        }
    }

    pub(crate) unsafe fn vs_exp_avx512f_asm(n: usize, a: *const f32, b: *mut f32) {
        const NR: usize = 16;
        // Constants
        // use asm until avx512f is stabilized
        const EXP_HI: f32 = 88.3762626647949 * 2.0;
        const EXP_LO: f32 = -88.3762626647949 * 2.0;
        const LOG2EF: f32 = 1.44269504088896341;
        const INV_LOG2EF: f32 = 0.693359375;
        const CEHPES_EXP_C2: f32 = -2.12194440e-4;
        const EXP_P0: f32 = 1.9875691500E-4;
        const EXP_P1: f32 = 1.3981999507E-3;
        const EXP_P2: f32 = 8.3334519073E-3;
        const EXP_P3: f32 = 4.1665795894E-2;
        const EXP_P4: f32 = 1.6666665459E-1;
        const EXP_P5: f32 = 5.0000001201E-1;

        let constant_arr =
            [LOG2EF, -CEHPES_EXP_C2 - INV_LOG2EF, EXP_P0, EXP_P1, EXP_P2, EXP_P3, EXP_P4, EXP_P5, 1.0, EXP_HI, EXP_LO];
        let mut i = 0;
        asm!(
            "vbroadcastss ({constant_arrx}), %zmm0",
            "vbroadcastss 4({constant_arrx}), %zmm1",
            "vbroadcastss 8({constant_arrx}), %zmm2",
            "vbroadcastss 12({constant_arrx}), %zmm3",
            "vbroadcastss 16({constant_arrx}), %zmm4",
            "vbroadcastss 20({constant_arrx}), %zmm5",
            "vbroadcastss 24({constant_arrx}), %zmm6",
            "vbroadcastss 28({constant_arrx}), %zmm7",
            "vbroadcastss 32({constant_arrx}), %zmm8",

            "vbroadcastss 36({constant_arrx}), %zmm13",
            "vbroadcastss 40({constant_arrx}), %zmm14",

            "test {nx:e}, {nx:e}",
            "je 3f",

            "2:",
            "vmovups ({ax}), %zmm9",
            // order of input for max and min is important
            // since it leads to correct NaN propagation
            "vminps %zmm9, %zmm13, %zmm9",
            "vmaxps %zmm9, %zmm14, %zmm9",
            "vmulps  %zmm0, %zmm9, %zmm10",
            "vrndscaleps     $8, %zmm10, %zmm10",
            "vfmadd231ps     %zmm1, %zmm10, %zmm9",
            "vmovaps %zmm2, %zmm11",
            "vfmadd213ps     %zmm3, %zmm9, %zmm11",
            "vfmadd213ps     %zmm4, %zmm9, %zmm11",
            "vfmadd213ps     %zmm5, %zmm9, %zmm11",
            "vfmadd213ps     %zmm6, %zmm9, %zmm11",
            "vmulps  %zmm9, %zmm9, %zmm12",
            "vfmadd213ps     %zmm7, %zmm9, %zmm11",
            "vfmadd213ps     %zmm9, %zmm12, %zmm11",
            "vaddps  %zmm8, %zmm11, %zmm9",
            "vscalefps       %zmm10, %zmm9, %zmm9",
            "vmovups %zmm9, ({bx})",
            "add  $64, {ax}",
            "add  $64, {bx}",
            "add $16, {ix:e}",
            "cmp {nx:e}, {ix:e}",
            "jl 2b",

            "3:",
            "vzeroupper",

            constant_arrx = in(reg) &constant_arr,
            ax = inout(reg) a => _,
            bx = inout(reg) b => _,
            ix = inout(reg) i => i,
            nx = inout(reg) n / NR * NR => _,
            out("zmm0") _, out("zmm1") _, out("zmm2") _, out("zmm3") _, out("zmm4") _, out("zmm5") _, out("zmm6") _, out("zmm7") _, out("zmm8") _, out("zmm9") _,
            out("zmm10") _, out("zmm11") _, out("zmm12") _, out("zmm13") _, out("zmm14") _, out("zmm15") _, out("zmm16") _, out("zmm17") _, out("zmm18") _, out("zmm19") _,
            out("zmm20") _, out("zmm21") _, out("zmm22") _, out("zmm23") _, out("zmm24") _, out("zmm25") _, out("zmm26") _, out("zmm27") _, out("zmm28") _, out("zmm29") _,
            out("zmm30") _, out("zmm31") _,
            options(att_syntax)
        );
        while i < n {
            *b.offset(i as isize) = (*a.offset(i as isize)).exp();
            i += 1;
        }
    }

    pub(crate) unsafe fn vs_tanh_avx512f_asm(n: usize, a: *const f32, b: *mut f32) {
        const NR: usize = 16;
        // Constants
        // use asm until avx512f is stabilized
        const EXP_HI: f32 = 88.3762626647949 * 1.0;
        const EXP_LO: f32 = -88.3762626647949 * 1.0;
        const LOG2EF: f32 = 1.44269504088896341;
        const INV_LOG2EF: f32 = 0.693359375;
        const CEHPES_EXP_C2: f32 = -2.12194440e-4;
        const EXP_P0: f32 = 1.9875691500E-4;
        const EXP_P1: f32 = 1.3981999507E-3;
        const EXP_P2: f32 = 8.3334519073E-3;
        const EXP_P3: f32 = 4.1665795894E-2;
        const EXP_P4: f32 = 1.6666665459E-1;
        const EXP_P5: f32 = 5.0000001201E-1;

        let constant_arr = [
            LOG2EF,
            -CEHPES_EXP_C2 - INV_LOG2EF,
            EXP_P0,
            EXP_P1,
            EXP_P2,
            EXP_P3,
            EXP_P4,
            EXP_P5,
            1.0,
            EXP_HI,
            EXP_LO,
            -1.0,
        ];
        let mut i = 0;
        asm!(
            "vbroadcastss ({constant_arrx}), %zmm0",
            "vbroadcastss 4({constant_arrx}), %zmm1",
            "vbroadcastss 8({constant_arrx}), %zmm2",
            "vbroadcastss 12({constant_arrx}), %zmm3",
            "vbroadcastss 16({constant_arrx}), %zmm4",
            "vbroadcastss 20({constant_arrx}), %zmm5",
            "vbroadcastss 24({constant_arrx}), %zmm6",
            "vbroadcastss 28({constant_arrx}), %zmm7",
            "vbroadcastss 32({constant_arrx}), %zmm8",

            "vbroadcastss 36({constant_arrx}), %zmm13",
            "vbroadcastss 40({constant_arrx}), %zmm14",
            "vbroadcastss 44({constant_arrx}), %zmm15",

            "test {nx:e}, {nx:e}",
            "je 3f",

            "2:",
            "vmovups ({ax}), %zmm31",
            // order of input for max and min is important
            // since it leads to correct NaN propagation
            "vminps %zmm31, %zmm13, %zmm9",
            "vmaxps %zmm9, %zmm14, %zmm9",
            "vmulps  %zmm0, %zmm9, %zmm10",
            "vrndscaleps     $8, %zmm10, %zmm10",
            "vfmadd231ps     %zmm1, %zmm10, %zmm9",
            "vmovaps %zmm2, %zmm11",
            "vfmadd213ps     %zmm3, %zmm9, %zmm11",
            "vfmadd213ps     %zmm4, %zmm9, %zmm11",
            "vfmadd213ps     %zmm5, %zmm9, %zmm11",
            "vfmadd213ps     %zmm6, %zmm9, %zmm11",
            "vmulps  %zmm9, %zmm9, %zmm12",
            "vfmadd213ps     %zmm7, %zmm9, %zmm11",
            "vfmadd213ps     %zmm9, %zmm12, %zmm11",
            "vaddps  %zmm8, %zmm11, %zmm9",
            "vscalefps       %zmm10, %zmm9, %zmm9",
            "vmulps %zmm9, %zmm9, %zmm9",

            "vmaxps %zmm31, %zmm13, %zmm31",

            "vaddps %zmm9, %zmm15, %zmm16",
            "vaddps %zmm9, %zmm8, %zmm9",
            "vdivps %zmm9, %zmm16, %zmm9",
            "vminps %zmm8, %zmm9, %zmm9",
            "vmaxps %zmm15, %zmm9, %zmm9",

            "vminps %zmm31, %zmm9, %zmm9",
            "vmovups %zmm9, ({bx})",
            "add  $64, {ax}",
            "add  $64, {bx}",
            "add $16, {ix:e}",
            "cmp {nx:e}, {ix:e}",
            "jl 2b",

            "3:",
            "vzeroupper",

            constant_arrx = in(reg) &constant_arr,
            ax = inout(reg) a => _,
            bx = inout(reg) b => _,
            ix = inout(reg) i => i,
            nx = inout(reg) n / NR * NR => _,
            out("zmm0") _, out("zmm1") _, out("zmm2") _, out("zmm3") _, out("zmm4") _, out("zmm5") _, out("zmm6") _, out("zmm7") _, out("zmm8") _, out("zmm9") _,
            out("zmm10") _, out("zmm11") _, out("zmm12") _, out("zmm13") _, out("zmm14") _, out("zmm15") _, out("zmm16") _, out("zmm17") _, out("zmm18") _, out("zmm19") _,
            out("zmm20") _, out("zmm21") _, out("zmm22") _, out("zmm23") _, out("zmm24") _, out("zmm25") _, out("zmm26") _, out("zmm27") _, out("zmm28") _, out("zmm29") _,
            out("zmm30") _, out("zmm31") _,
            options(att_syntax)
        );
        while i < n {
            *b.offset(i as isize) = (*a.offset(i as isize)).tanh();
            i += 1;
        }
    }

    pub(crate) unsafe fn vs_sqrt_avx512f_asm(n: usize, a: *const f32, b: *mut f32) {
        const NR: usize = 16;
        // Constants
        // use asm until avx512f is stabilized
        let mut i = 0;
        asm!(
            "test {nx:e}, {nx:e}",
            "je 3f",

            "2:",
            "vmovups ({ax}), %zmm9",
            "vsqrtps %zmm9, %zmm9",
            "vmovups %zmm9, ({bx})",
            "add  $64, {ax}",
            "add  $64, {bx}",
            "add $16, {ix:e}",
            "cmp {nx:e}, {ix:e}",
            "jl 2b",

            "3:",
            "vzeroupper",

            ax = inout(reg) a => _,
            bx = inout(reg) b => _,
            ix = inout(reg) i => i,
            nx = inout(reg) n / NR * NR => _,
            out("zmm9") _,
            options(att_syntax)
        );
        while i < n {
            *b.offset(i as isize) = (*a.offset(i as isize)).sqrt();
            i += 1;
        }
    }

    pub(crate) unsafe fn vd_sqrt_avx512f_asm(n: usize, a: *const f64, b: *mut f64) {
        const NR: usize = 8;
        // Constants
        // use asm until avx512f is stabilized
        let mut i = 0;
        asm!(
            "test {nx:e}, {nx:e}",
            "je 3f",

            "2:",
            "vmovupd ({ax}), %zmm9",
            "vsqrtpd %zmm9, %zmm9",
            "vmovupd %zmm9, ({bx})",
            "add  $64, {ax}",
            "add  $64, {bx}",
            "add $16, {ix:e}",
            "cmp {nx:e}, {ix:e}",
            "jl 2b",

            "3:",
            "vzeroupper",

            ax = inout(reg) a => _,
            bx = inout(reg) b => _,
            ix = inout(reg) i => i,
            nx = inout(reg) n / NR * NR => _,
            out("zmm9") _,
            options(att_syntax)
        );
        while i < n {
            *b.offset(i as isize) = (*a.offset(i as isize)).sqrt();
            i += 1;
        }
    }
}

#[cfg(target_arch = "x86_64")]
pub(crate) use avx512f_mod::*;

macro_rules! impl_fallback {
    ($name:ident, $n:ident, $t:ty) => {
        pub(crate) unsafe fn $name(n: usize, a: *const $t, b: *mut $t) {
            let mut i = 0;
            while i < n {
                *b.offset(i as isize) = (*a.offset(i as isize)).$n();
                i += 1;
            }
        }
    };
}

impl_fallback!(vd_exp_fallback, exp, f64);
impl_fallback!(vs_exp_fallback, exp, f32);

impl_fallback!(vd_ln_fallback, ln, f64);
impl_fallback!(vs_ln_fallback, ln, f32);

impl_fallback!(vd_tanh_fallback, tanh, f64);
impl_fallback!(vs_tanh_fallback, tanh, f32);

impl_fallback!(vd_sqrt_fallback, sqrt, f64);
impl_fallback!(vs_sqrt_fallback, sqrt, f32);

impl_fallback!(vd_sin_fallback, sin, f64);
impl_fallback!(vs_sin_fallback, sin, f32);

impl_fallback!(vd_cos_fallback, cos, f64);
impl_fallback!(vs_cos_fallback, cos, f32);

#[cfg(test)]
mod tests {
    use super::*;

    fn full_range_f32_pair() -> (Vec<f32>, Vec<f32>) {
        let v_len = 1 << 32;
        // let v_len = 16;
        println!("v_len log2: {}", (v_len as f64).log2());
        let mut a = vec![0f32; v_len];
        let b = vec![0f32; v_len];
        // collect the max value and print
        let mut max = 0.;
        for i in 0..v_len {
            a[i] = f32::from_bits(i as u32);
            if a[i] > max {
                max = a[i];
            }
        }
        println!("max: {}", max);

        (a, b)
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
        // println!("x.exp(): {}", x.exp());
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
        let (a, mut b) = full_range_f32_pair();
        let a_len = a.len();
        unsafe {
            vs_sqrt(a_len, a.as_ptr(), b.as_mut_ptr());
        }
        for i in 0..a_len {
            check_sqrt(a[i], b[i]);
        }
    }

    #[test]
    fn accuracy_sin() {
        let (a, mut b) = full_range_f32_pair();
        let a_len = a.len();
        unsafe {
            vs_sin(a_len, a.as_ptr(), b.as_mut_ptr());
        }
        for i in 0..a_len {
            check_sin(a[i], b[i]);
        }
    }

    #[test]
    fn accuracy_cos() {
        let (a, mut b) = full_range_f32_pair();
        let a_len = a.len();
        unsafe {
            vs_cos(a_len, a.as_ptr(), b.as_mut_ptr());
        }
        for i in 0..a_len {
            check_cos(a[i], b[i]);
        }
    }

    #[test]
    fn accuracy_exp() {
        let (a, mut b) = full_range_f32_pair();
        let a_len = a.len();
        unsafe {
            vs_exp(a_len, a.as_ptr(), b.as_mut_ptr());
        }
        for i in 0..a_len {
            check_exp(a[i], b[i]);
        }
    }

    #[test]
    fn accuracy_tanh() {
        let (a, mut b) = full_range_f32_pair();
        let a_len = a.len();
        unsafe {
            vs_tanh(a_len, a.as_ptr(), b.as_mut_ptr());
        }
        for i in 0..a_len {
            check_tanh(a[i], b[i]);
        }
    }
    #[test]
    fn accuracy_ln() {
        let (a, mut b) = full_range_f32_pair();
        let a_len = a.len();
        unsafe {
            vs_ln(a_len, a.as_ptr(), b.as_mut_ptr());
        }
        for i in 0..a_len {
            check_ln(a[i], b[i]);
        }
    }
    const PROJECT_DIR: &str = "C:\\Users\\I011745\\Desktop\\corenum\\pire\\.venv\\Library\\bin";
    use libc::{c_float, c_int};

    use once_cell::sync::Lazy;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    static MATHFUN_LIBRARY: Lazy<libloading::Library> = Lazy::new(|| unsafe {
        #[cfg(target_os = "windows")]
        let default_cblas_path = format!("{PROJECT_DIR}/mkl_rt.2.dll");
        #[cfg(target_os = "linux")]
        let default_cblas_path = format!("{PROJECT_DIR}/../../.venv/lib/libmkl_rt.so.2");
        let cblas_path = std::env::var("PIRE_MATHFUN_PATH").unwrap_or(default_cblas_path);
        libloading::Library::new(cblas_path).unwrap()
    });
    type UnaryFnF32 = unsafe extern "C" fn(c_int, *const c_float, *mut c_float);

    fn random_matrix_std<T>(arr: &mut [T])
    where
        rand::distributions::Standard: rand::prelude::Distribution<T>,
    {
        let mut x = StdRng::seed_from_u64(43);
        arr.iter_mut().for_each(|p| *p = x.gen::<T>());
    }
    static VS_COS_MKL: Lazy<libloading::Symbol<'static, UnaryFnF32>> = Lazy::new(|| unsafe {
        let unary_fn = MATHFUN_LIBRARY.get(b"vsExp").unwrap();
        unary_fn
    });

    unsafe fn vs_cos_mkl(n: c_int, a: *const c_float, y: *mut c_float) {
        VS_COS_MKL(n, a, y);
    }
    #[test]
    fn mkl_test() {
        let m = 1 << 31 - 1;
        let mut a = vec![1.0; m];
        let mut b = vec![1.0; m];
        random_matrix_std(&mut a);
        random_matrix_std(&mut b);
        let a_len = a.len();
        let a_len_i32 = a_len as i32;
        println!("a_len_i32: {}", a_len_i32);
        let t0 = std::time::Instant::now();
        unsafe {
            vs_cos_mkl(a_len_i32, a.as_ptr(), b.as_mut_ptr());
            // vs_cos(a_len, a.as_ptr(), b.as_mut_ptr());
        }
        let t1 = std::time::Instant::now();
        println!("time: {:?}", t1 - t0);
    }
}
