// Implementations below are inspired by the avx_mathfun project
// and changed according to several factors, (e.g. different simd extensions, correctness for large values of exp)
// avx_mathfun project: https://github.com/reyoung/avx_mathfun
use once_cell::sync::Lazy;

use core::arch::x86_64::*;

use crate::RUNTIME_HW_CONFIG;

pub static VD_EXP: Lazy<unsafe fn(i64, *const f64, *mut f64)> = Lazy::new(|| {
    vd_exp_fallback
});

pub static VS_EXP: Lazy<unsafe fn(i64, *const f32, *mut f32)> = Lazy::new(|| {
    if RUNTIME_HW_CONFIG.avx2 && RUNTIME_HW_CONFIG.fma {
        vs_exp_avx2_fma
    } else {
        vs_exp_fallback
    }
});

pub static VD_LN: Lazy<unsafe fn(i64, *const f64, *mut f64)> = Lazy::new(|| {
    vd_ln_fallback
});

pub static VS_LN: Lazy<unsafe fn(i64, *const f32, *mut f32)> = Lazy::new(|| {
    vs_ln_fallback
});

pub static VD_TANH: Lazy<unsafe fn(i64, *const f64, *mut f64)> = Lazy::new(|| {
    vd_tanh_fallback
});

pub static VS_TANH: Lazy<unsafe fn(i64, *const f32, *mut f32)> = Lazy::new(|| {
    if RUNTIME_HW_CONFIG.avx2 && RUNTIME_HW_CONFIG.fma {
        vs_tanh_avx2_fma
    } else {
        vs_tanh_fallback
    }
});


pub static VD_SQRT: Lazy<unsafe fn(i64, *const f64, *mut f64)> = Lazy::new(|| {
    vd_sqrt_fallback
});

pub static VS_SQRT: Lazy<unsafe fn(i64, *const f32, *mut f32)> = Lazy::new(|| {
    vs_sqrt_fallback
});

pub static VD_SIN: Lazy<unsafe fn(i64, *const f64, *mut f64)> = Lazy::new(|| {
    vd_sin_fallback
});

pub static VS_SIN: Lazy<unsafe fn(i64, *const f32, *mut f32)> = Lazy::new(|| {
    if RUNTIME_HW_CONFIG.avx2 && RUNTIME_HW_CONFIG.fma {
        vs_sin_avx2_fma
    } else {
        vs_sin_fallback
    }
});

pub static VD_COS: Lazy<unsafe fn(i64, *const f64, *mut f64)> = Lazy::new(|| {
    vd_cos_fallback
});

pub static VS_COS: Lazy<unsafe fn(i64, *const f32, *mut f32)> = Lazy::new(|| {
    if RUNTIME_HW_CONFIG.avx2 && RUNTIME_HW_CONFIG.fma {
        vs_cos_avx2_fma
    } else {
        vs_cos_fallback
    }
});

macro_rules! impl_unary {
    ($name:ident, $fn:ident, $ta:ty, $tb:ty) => {
        pub unsafe fn $name(n: i64, a: *const $ta, y: *mut $tb) {
            assert!(n > 0);
            let fn_sequantial = *$fn;
            let num_per_thread_min = 1000000;
            let num_thread_max = 8;
            let n_per_thread = (n + num_thread_max - 1) / num_thread_max;
            let num_thread = if n_per_thread >= num_per_thread_min {
                num_thread_max
            } else {
                (n + num_per_thread_min - 1) / num_per_thread_min
            };
            let num_per_thread = (n + num_thread - 1) / num_thread;
            let a_usize = a as usize;
            let y_usize = y as usize;
            let x = std::thread::spawn(move || {
                for i in 1..num_thread {
                    let a = a_usize as *const $ta;
                    let y = y_usize as *mut $tb;
                    let n_cur = num_per_thread.min(n - i * num_per_thread);
                    fn_sequantial(n_cur, a.offset((i * num_per_thread) as isize), y.offset((i * num_per_thread) as isize));
                }
            });
            fn_sequantial(n.min(num_per_thread), a, y);
            x.join().unwrap();
        }
    };
}




#[target_feature(enable = "avx,avx2,fma")]
unsafe fn vs_exp_avx2_fma(n: i64, a: *const f32, b: *mut f32) {
    const NR: i64 = 8;
    // Constants
    const EXP_HI: f32 = 88.7228391117*1.0;
    const EXP_LO: f32 = -88.7228391117*1.0;
    const LOG2EF: f32 = 1.44269504088896341;
    const INV_LOG2EF: f32 = 0.69314718056;
    const EXP_P0: f32 = 0.00032712723;
    const EXP_P1: f32 = 0.00228989065;
    // const EXP_P1: f32 = 0.00138888888;
    const EXP_P2: f32 = 0.01373934392;
    const EXP_P3: f32 = 0.06869671961;
    const EXP_P4: f32 = 0.27478687845;
    const EXP_P5: f32 = 0.82436063535;

    // Load constants as __m256
    let exp_hi = _mm256_set1_ps(EXP_HI);
    let exp_lo = _mm256_set1_ps(EXP_LO);
    let log2ef = _mm256_set1_ps(LOG2EF);
    let inv_c2 = _mm256_set1_ps(-INV_LOG2EF);

    let half = _mm256_set1_ps(0.5);

    let exp_p0 = _mm256_set1_ps(EXP_P0);
    let exp_p1 = _mm256_set1_ps(EXP_P1);
    let exp_p2 = _mm256_set1_ps(EXP_P2);
    let exp_p3 = _mm256_set1_ps(EXP_P3);
    let exp_p4 = _mm256_set1_ps(EXP_P4);
    let exp_p5 = _mm256_set1_ps(EXP_P5);
    let min_exponent = _mm256_set1_ps(-127.0);
    let e_sqrt = _mm256_set1_ps(1.6487212707);

    let mut i = 0;
    while i <= (n-NR) {
        let x = _mm256_loadu_ps(a.offset(i as isize));

        // Clamp x
        let x = _mm256_min_ps(exp_hi, x);
        let x = _mm256_max_ps(exp_lo, x);

        // Compute fx = floor(x * log2ef + 0.5)
        let mut fx = _mm256_mul_ps(x, log2ef);
        // use to zero rounding since nearest int is problematic for near overflow and underflowing values
        fx = _mm256_round_ps(fx, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
        // to prevent denormalized values
        fx = _mm256_max_ps(fx, min_exponent);

        // Approximation for exp(x)
        let x = _mm256_fmadd_ps(fx, inv_c2, x);

        let x = _mm256_sub_ps(x, half);

        let mut y = exp_p0;
        y = _mm256_fmadd_ps(y, x, exp_p1);
        y = _mm256_fmadd_ps(y, x, exp_p2);
        y = _mm256_fmadd_ps(y, x, exp_p3);
        y = _mm256_fmadd_ps(y, x, exp_p4);
        y = _mm256_fmadd_ps(y, x, exp_p5);
        y = _mm256_fmadd_ps(y, x, e_sqrt);
        y = _mm256_fmadd_ps(y, x, e_sqrt);

        // Compute 2^fx
        let mut imm0 = _mm256_cvttps_epi32(fx);
        imm0 = _mm256_add_epi32(imm0, _mm256_set1_epi32(0x7f));
        imm0 = _mm256_slli_epi32(imm0, 23);
        let pow2n = _mm256_castsi256_ps(imm0);

        // Final result
        let r = _mm256_mul_ps(y, pow2n);
        _mm256_storeu_ps(b.offset(i as isize), r);
        i += NR;
    }
    while i < n {
        *b.offset(i as isize) = (*a.offset(i as isize)).exp();
        i += 1;
    }
}


#[target_feature(enable = "avx,avx2,fma")]
unsafe fn vs_tanh_avx2_fma(n: i64, a: *const f32, b: *mut f32) {
    const NR: i64 = 8;
    // Constants
    const EXP_HI: f32 = 88.7228391117*1.0;
    const EXP_LO: f32 = -88.7228391117*1.0;
    const LOG2EF: f32 = 1.44269504088896341;
    const INV_LOG2EF: f32 = 0.69314718056;
    const EXP_P0: f32 = 0.00032712723;
    const EXP_P1: f32 = 0.00228989065;
    // const EXP_P1: f32 = 0.00138888888;
    const EXP_P2: f32 = 0.01373934392;
    const EXP_P3: f32 = 0.06869671961;
    const EXP_P4: f32 = 0.27478687845;
    const EXP_P5: f32 = 0.82436063535;

    // Load constants as __m256
    let exp_hi = _mm256_set1_ps(EXP_HI);
    let exp_lo = _mm256_set1_ps(EXP_LO);
    let log2ef = _mm256_set1_ps(LOG2EF);
    let inv_c2 = _mm256_set1_ps(-INV_LOG2EF);

    let half = _mm256_set1_ps(0.5);

    let one = _mm256_set1_ps(1.0);
    let one_neg = _mm256_set1_ps(-1.0);

    let exp_p0 = _mm256_set1_ps(EXP_P0);
    let exp_p1 = _mm256_set1_ps(EXP_P1);
    let exp_p2 = _mm256_set1_ps(EXP_P2);
    let exp_p3 = _mm256_set1_ps(EXP_P3);
    let exp_p4 = _mm256_set1_ps(EXP_P4);
    let exp_p5 = _mm256_set1_ps(EXP_P5);
    let min_exponent = _mm256_set1_ps(-127.0);
    let e_sqrt = _mm256_set1_ps(1.6487212707);

    let mut i = 0;
    while i <= (n-NR) {
        let x = _mm256_loadu_ps(a.offset(i as isize));
        let x0 = x;
        let x0 = _mm256_max_ps(exp_hi,x0);
        // Clamp x
        let x = _mm256_min_ps(exp_hi,x);
        let x = _mm256_max_ps(exp_lo,x);

        // Compute fx = floor(x * log2ef + 0.5)
        let mut fx = _mm256_mul_ps(x, log2ef);
        // use to zero rounding since nearest int is problematic for near overflow and underflowing values
        fx = _mm256_round_ps(fx, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
        // to prevent denormalized values
        fx = _mm256_max_ps(fx, min_exponent);

        // Approximation for exp(x)
        let x = _mm256_fmadd_ps(fx, inv_c2, x);

        let x = _mm256_sub_ps(x, half);

        let mut y = exp_p0;
        y = _mm256_fmadd_ps(y, x, exp_p1);
        y = _mm256_fmadd_ps(y, x, exp_p2);
        y = _mm256_fmadd_ps(y, x, exp_p3);
        y = _mm256_fmadd_ps(y, x, exp_p4);
        y = _mm256_fmadd_ps(y, x, exp_p5);
        y = _mm256_fmadd_ps(y, x, e_sqrt);
        y = _mm256_fmadd_ps(y, x, e_sqrt);

        // Compute 2^fx
        let mut imm0 = _mm256_cvttps_epi32(fx);
        imm0 = _mm256_add_epi32(imm0, _mm256_set1_epi32(0x7f));
        imm0 = _mm256_slli_epi32(imm0, 23);
        let pow2n = _mm256_castsi256_ps(imm0);

        // Final result
        let r = _mm256_mul_ps(y, pow2n);
        let r = _mm256_mul_ps(r, r);

        let numerator = _mm256_sub_ps(r, one);
        let denominator = _mm256_add_ps(r, one);
        let mut r = _mm256_div_ps(numerator, denominator);
        r = _mm256_min_ps(r,one);
        r = _mm256_max_ps(r,one_neg);
        // restore nan using min_ps
        r = _mm256_min_ps(r,x0);

        _mm256_storeu_ps(b.offset(i as isize), r);
        i += NR;
    }
    while i < n {
        *b.offset(i as isize) = (*a.offset(i as isize)).tanh();
        i += 1;
    }
}





#[target_feature(enable = "avx,avx2,fma")]
unsafe fn vs_sin_avx2_fma(n: i64, a: *const f32, b: *mut f32) {
    const NR: i64 = 8;
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

    let dp1 = _mm256_set1_ps(DP1);
    let dp2 = _mm256_set1_ps(DP2);
    let dp3 = _mm256_set1_ps(DP3);

    let fopi = _mm256_set1_ps(FOPI);


    let p0 = _mm256_set1_ps(P0);
    let p1 = _mm256_set1_ps(P1);
    let p2 = _mm256_set1_ps(P2);
    let p3 = _mm256_set1_ps(P3);

    // let q0 = _mm256_set1_ps(Q0);
    let q1 = _mm256_set1_ps(Q1);
    let q2 = _mm256_set1_ps(Q2);
    let q3 = _mm256_set1_ps(Q3);
    let q4 = _mm256_set1_ps(Q4);

    let sign_mask = _mm256_set1_epi32(0x80000000u32 as i32);
    let inv_sign_mask = _mm256_set1_epi32(!0x80000000u32 as i32);

    // cast to mm256
    let sign_mask = _mm256_castsi256_ps(sign_mask);
    let inv_sign_mask = _mm256_castsi256_ps(inv_sign_mask);

    let mut i = 0;
    while i <= (n-NR) {
        let x = _mm256_loadu_ps(a.offset(i as isize));
        let mut sign_bit = x;
        // take the absolute value
        let x = _mm256_and_ps(x, inv_sign_mask);
        // extract the sign bit (upper one)
        sign_bit = _mm256_and_ps(sign_bit, sign_mask);

        // scale by 4/Pi
        let mut y = _mm256_mul_ps(x, fopi);

        // store the integer part of y in mm0
        let mut imm2 = _mm256_cvttps_epi32(y);
        // j=(j+1) & (~1) (see the cephes sources)
        // another two AVX2 instruction
        imm2 = _mm256_add_epi32(imm2, _mm256_set1_epi32(1));
        imm2 = _mm256_and_si256(imm2, _mm256_set1_epi32(!1));
        y = _mm256_cvtepi32_ps(imm2);

        // get the swap sign flag
        let imm0 = _mm256_and_si256(imm2, _mm256_set1_epi32(4));
        let imm0 = _mm256_slli_epi32(imm0, 29);
        // get the polynom selection mask
        // there is one polynom for 0 <= x <= Pi/4
        // and another one for Pi/4<x<=Pi/2
        let imm2 = _mm256_and_si256(imm2, _mm256_set1_epi32(2));
        let imm2 = _mm256_cmpeq_epi32(imm2, _mm256_set1_epi32(0));



        let swap_sign_bit = _mm256_castsi256_ps(imm0);
        let poly_mask = _mm256_castsi256_ps(imm2);
        let sign_bit = _mm256_xor_ps(sign_bit, swap_sign_bit);

        // The magic pass: "Extended precision modular arithmetic"
        // x = ((x - y * DP1) - y * DP2) - y * DP3;
        let xmm1 = _mm256_mul_ps(y, dp1);
        let xmm2 = _mm256_mul_ps(y, dp2);
        let xmm3 = _mm256_mul_ps(y, dp3);
        let x = _mm256_add_ps(x, xmm1);
        let x = _mm256_add_ps(x, xmm2);
        let x = _mm256_add_ps(x, xmm3);


        // Evaluate the first polynom  (0 <= x <= Pi/4)
        let mut y = q4;
        let z = _mm256_mul_ps(x, x);
        y = _mm256_fmadd_ps(y, z, q3);
        y = _mm256_fmadd_ps(y, z, q2);
        y = _mm256_fmadd_ps(y, z, q1);
        y = _mm256_fmadd_ps(y, z, p0);

        // evaluate the second polynom (Pi/4 <= x <= 0)
        let mut y2 = p3;
        y2 = _mm256_fmadd_ps(y2, z, p2);
        y2 = _mm256_fmadd_ps(y2, z, p1);
        y2 = _mm256_fmadd_ps(y2, z, p0);
        y2 = _mm256_mul_ps(y2, x);

        // select the correct result from the two polynoms
        let y2 = _mm256_and_ps(poly_mask, y2);
        let y = _mm256_andnot_ps(poly_mask, y);
        let y = _mm256_add_ps(y, y2);
        // update the sign
        let y = _mm256_xor_ps(y, sign_bit);

        _mm256_storeu_ps(b.offset(i as isize), y);
        i += NR;
    }

    while i < n {
        *b.offset(i as isize) = (*a.offset(i as isize)).sin();
        i += 1;
    }
}



#[target_feature(enable = "avx,avx2,fma")]
unsafe fn vs_cos_avx2_fma(n: i64, a: *const f32, b: *mut f32) {
    const NR: i64 = 8;
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

    let dp1 = _mm256_set1_ps(DP1);
    let dp2 = _mm256_set1_ps(DP2);
    let dp3 = _mm256_set1_ps(DP3);

    let fopi = _mm256_set1_ps(FOPI);


    let p0 = _mm256_set1_ps(P0);
    let p1 = _mm256_set1_ps(P1);
    let p2 = _mm256_set1_ps(P2);
    let p3 = _mm256_set1_ps(P3);

    // let q0 = _mm256_set1_ps(Q0);
    let q1 = _mm256_set1_ps(Q1);
    let q2 = _mm256_set1_ps(Q2);
    let q3 = _mm256_set1_ps(Q3);
    let q4 = _mm256_set1_ps(Q4);

    let inv_sign_mask = _mm256_set1_epi32(!0x80000000u32 as i32);

    // cast to mm256
    let inv_sign_mask = _mm256_castsi256_ps(inv_sign_mask);

    let mut i = 0;
    while i <= (n-NR) {
        let x = _mm256_loadu_ps(a.offset(i as isize));
        // take the absolute value
        let x = _mm256_and_ps(x, inv_sign_mask);
        // extract the sign bit (upper one)

        // scale by 4/Pi
        let mut y = _mm256_mul_ps(x, fopi);

        // store the integer part of y in mm0
        let mut imm2 = _mm256_cvttps_epi32(y);
        // j=(j+1) & (~1) (see the cephes sources)
        // another two AVX2 instruction
        imm2 = _mm256_add_epi32(imm2, _mm256_set1_epi32(1));
        imm2 = _mm256_and_si256(imm2, _mm256_set1_epi32(!1));
        y = _mm256_cvtepi32_ps(imm2);
        imm2 = _mm256_sub_epi32(imm2, _mm256_set1_epi32(2));

        // get the swap sign flag
        let imm0 = _mm256_andnot_si256(imm2, _mm256_set1_epi32(4));
        let imm0 = _mm256_slli_epi32(imm0, 29);
        // get the polynom selection mask
        // there is one polynom for 0 <= x <= Pi/4
        // and another one for Pi/4<x<=Pi/2
        let imm2 = _mm256_and_si256(imm2, _mm256_set1_epi32(2));
        let imm2 = _mm256_cmpeq_epi32(imm2, _mm256_set1_epi32(0));

        let sign_bit = _mm256_castsi256_ps(imm0);
        let poly_mask = _mm256_castsi256_ps(imm2);

        // The magic pass: "Extended precision modular arithmetic"
        // x = ((x - y * DP1) - y * DP2) - y * DP3;
        let xmm1 = _mm256_mul_ps(y, dp1);
        let xmm2 = _mm256_mul_ps(y, dp2);
        let xmm3 = _mm256_mul_ps(y, dp3);
        let x = _mm256_add_ps(x, xmm1);
        let x = _mm256_add_ps(x, xmm2);
        let x = _mm256_add_ps(x, xmm3);


        // Evaluate the first polynom  (0 <= x <= Pi/4)
        let mut y = q4;
        let z = _mm256_mul_ps(x, x);
        y = _mm256_fmadd_ps(y, z, q3);
        y = _mm256_fmadd_ps(y, z, q2);
        y = _mm256_fmadd_ps(y, z, q1);
        y = _mm256_fmadd_ps(y, z, p0);

        // evaluate the second polynom (Pi/4 <= x <= 0)
        let mut y2 = p3;
        y2 = _mm256_fmadd_ps(y2, z, p2);
        y2 = _mm256_fmadd_ps(y2, z, p1);
        y2 = _mm256_fmadd_ps(y2, z, p0);
        y2 = _mm256_mul_ps(y2, x);

        // select the correct result from the two polynoms
        let y2 = _mm256_and_ps(poly_mask, y2);
        let y = _mm256_andnot_ps(poly_mask, y);
        let y = _mm256_add_ps(y, y2);
        // update the sign
        let y = _mm256_xor_ps(y, sign_bit);

        _mm256_storeu_ps(b.offset(i as isize), y);
        i += NR;
    }

    while i < n {
        *b.offset(i as isize) = (*a.offset(i as isize)).sin();
        i += 1;
    }
}




pub(crate) unsafe fn vd_exp_fallback(n: i64, a: *const f64, y: *mut f64) {
    for i in 0..n {
        *y.offset(i as isize) = (*a.offset(i as isize)).exp();
    }
}

pub(crate) unsafe fn vs_exp_fallback(n: i64, a: *const f32, y: *mut f32) {
    for i in 0..n {
        *y.offset(i as isize) = (*a.offset(i as isize)).exp();
    }
}

pub(crate) unsafe fn vd_ln_fallback(n: i64, a: *const f64, y: *mut f64) {
    for i in 0..n {
        *y.offset(i as isize) = (*a.offset(i as isize)).ln();
    }
}

pub(crate) unsafe fn vs_ln_fallback(n: i64, a: *const f32, y: *mut f32) {
    for i in 0..n {
        *y.offset(i as isize) = (*a.offset(i as isize)).ln();
    }
}

pub(crate) unsafe fn vd_tanh_fallback(n: i64, a: *const f64, y: *mut f64) {
    for i in 0..n {
        *y.offset(i as isize) = (*a.offset(i as isize)).tanh();
    }
}

pub(crate) unsafe fn vs_tanh_fallback(n: i64, a: *const f32, y: *mut f32) {
    for i in 0..n {
        *y.offset(i as isize) = (*a.offset(i as isize)).tanh();
    }
}

pub(crate) unsafe fn vd_sqrt_fallback(n: i64, a: *const f64, y: *mut f64) {
    for i in 0..n {
        *y.offset(i as isize) = (*a.offset(i as isize)).sqrt();
    }
}

pub(crate) unsafe fn vs_sqrt_fallback(n: i64, a: *const f32, y: *mut f32) {
    for i in 0..n {
        *y.offset(i as isize) = (*a.offset(i as isize)).sqrt();
    }
}

pub(crate) unsafe fn vd_sin_fallback(n: i64, a: *const f64, y: *mut f64) {
    for i in 0..n {
        *y.offset(i as isize) = (*a.offset(i as isize)).sin();
    }
}

pub(crate) unsafe fn vs_sin_fallback(n: i64, a: *const f32, y: *mut f32) {
    for i in 0..n {
        *y.offset(i as isize) = (*a.offset(i as isize)).sin();
    }
}

pub(crate) unsafe fn vd_cos_fallback(n: i64, a: *const f64, y: *mut f64) {
    for i in 0..n {
        *y.offset(i as isize) = (*a.offset(i as isize)).cos();
    }
}

pub(crate) unsafe fn vs_cos_fallback(n: i64, a: *const f32, y: *mut f32) {
    for i in 0..n {
        *y.offset(i as isize) = (*a.offset(i as isize)).cos();
    }
}

impl_unary!(vd_exp, VD_EXP, f64, f64);
impl_unary!(vs_exp, VS_EXP, f32, f32);

impl_unary!(vd_ln, VD_LN, f64, f64);
impl_unary!(vs_ln, VS_LN, f32, f32);

impl_unary!(vd_tanh, VD_TANH, f64, f64);
impl_unary!(vs_tanh, VS_TANH, f32, f32);

impl_unary!(vd_sqrt, VD_SQRT, f64, f64);
impl_unary!(vs_sqrt, VS_SQRT, f32, f32);

impl_unary!(vd_sin, VD_SIN, f64, f64);
impl_unary!(vs_sin, VS_SIN, f32, f32);

impl_unary!(vd_cos, VD_COS, f64, f64);
impl_unary!(vs_cos, VS_COS, f32, f32);
