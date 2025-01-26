#[cfg(target_arch = "aarch64")]
pub(crate) mod aarch64;
#[cfg(target_arch = "wasm32")]
pub(crate) mod wasm32;
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
pub(crate) mod x86;
#[cfg(target_arch = "x86_64")]
pub(crate) mod x86_64;

#[cfg(target_arch = "aarch64")]
pub(crate) use aarch64::Neon;
#[cfg(target_arch = "wasm32")]
pub(crate) use wasm32::Wasm32;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub(crate) use x86::Sse2;
#[cfg(target_arch = "x86_64")]
pub(crate) use x86_64::{Avx2Fma, Avx512f, AvxSse2};

pub(crate) trait Simd {
    type Vf64: Copy;
    type Vf32: Copy;

    type Vi32: Copy;

    const F32_WIDTH: usize;
    const F64_WIDTH: usize;

    unsafe fn set1_f32(x: f32) -> Self::Vf32;
    unsafe fn set1_i32(x: i32) -> Self::Vi32;
    unsafe fn loadu_f32(ptr: *const f32) -> Self::Vf32;
    unsafe fn loadu_f64(ptr: *const f64) -> Self::Vf64;
    unsafe fn and_f32(a: Self::Vf32, b: Self::Vf32) -> Self::Vf32;
    unsafe fn mul_f32(a: Self::Vf32, b: Self::Vf32) -> Self::Vf32;
    unsafe fn add_i32(a: Self::Vi32, b: Self::Vi32) -> Self::Vi32;
    unsafe fn and_i32(a: Self::Vi32, b: Self::Vi32) -> Self::Vi32;
    unsafe fn cvt_i32_f32(a: Self::Vi32) -> Self::Vf32;
    unsafe fn cvt_f32_i32(a: Self::Vf32) -> Self::Vi32;
    unsafe fn sqrt_f32(a: Self::Vf32) -> Self::Vf32;
    unsafe fn sqrt_f64(a: Self::Vf64) -> Self::Vf64;
    unsafe fn sub_i32(a: Self::Vi32, b: Self::Vi32) -> Self::Vi32;
    unsafe fn andnot_i32(a: Self::Vi32, b: Self::Vi32) -> Self::Vi32;
    unsafe fn slli_i32<const IMM8: i32>(a: Self::Vi32) -> Self::Vi32;
    unsafe fn cmpeq_i32(a: Self::Vi32, b: Self::Vi32) -> Self::Vi32;
    unsafe fn cast_i32_f32(a: Self::Vi32) -> Self::Vf32;
    unsafe fn fmadd_f32(a: Self::Vf32, b: Self::Vf32, c: Self::Vf32) -> Self::Vf32;
    unsafe fn andnot_f32(a: Self::Vf32, b: Self::Vf32) -> Self::Vf32;
    unsafe fn add_f32(a: Self::Vf32, b: Self::Vf32) -> Self::Vf32;
    unsafe fn xor_f32(a: Self::Vf32, b: Self::Vf32) -> Self::Vf32;
    unsafe fn storeu_f32(ptr: *mut f32, a: Self::Vf32);
    unsafe fn storeu_f64(ptr: *mut f64, a: Self::Vf64);
    unsafe fn sub_f32(a: Self::Vf32, b: Self::Vf32) -> Self::Vf32;
    unsafe fn cmp_eq_f32(a: Self::Vf32, b: Self::Vf32) -> Self::Vf32;
    unsafe fn cmp_lt_f32(a: Self::Vf32, b: Self::Vf32) -> Self::Vf32;
    unsafe fn mask_mul_f32(mask: Self::Vf32, a: Self::Vf32, b: Self::Vf32) -> Self::Vf32;
    unsafe fn mask_sub_f32(mask: Self::Vf32, a: Self::Vf32, b: Self::Vf32) -> Self::Vf32;
    unsafe fn mask_add_f32(mask: Self::Vf32, a: Self::Vf32, b: Self::Vf32) -> Self::Vf32;
    unsafe fn or_f32(a: Self::Vf32, b: Self::Vf32) -> Self::Vf32;
    unsafe fn cast_f32_i32(a: Self::Vf32) -> Self::Vi32;
    unsafe fn srli_i32<const IMM8: i32>(a: Self::Vi32) -> Self::Vi32;
    unsafe fn min_f32(a: Self::Vf32, b: Self::Vf32) -> Self::Vf32;
    unsafe fn floor_f32(a: Self::Vf32) -> Self::Vf32;
    unsafe fn max_f32(a: Self::Vf32, b: Self::Vf32) -> Self::Vf32;
    unsafe fn div_f32(a: Self::Vf32, b: Self::Vf32) -> Self::Vf32;
    unsafe fn get_exp_mant_f32(a: Self::Vf32) -> (Self::Vf32, Self::Vf32);
}

pub(crate) trait UnaryFn1: Simd {
    unsafe fn vs_exp_0(n: usize, a: *const f32, b: *mut f32) {
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
            fx = Self::floor_f32(fx);
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
    unsafe fn vs_ln_0(n: usize, a: *const f32, b: *mut f32) {
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

            let mantissa_mask = Self::cmp_lt_f32(x, mantissa_th);
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
    unsafe fn vs_tanh_0(n: usize, a: *const f32, b: *mut f32) {
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
            let fx = Self::floor_f32(fx);
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

    unsafe fn vs_sqrt_0(n: usize, a: *const f32, b: *mut f32) {
        let mut i = 0;
        while (i + Self::F32_WIDTH) <= n {
            let x = Self::loadu_f32(a.offset(i as isize));
            let r = Self::sqrt_f32(x);
            Self::storeu_f32(b.offset(i as isize), r);
            i += Self::F32_WIDTH;
        }
        while i < n {
            *b.offset(i as isize) = (*a.offset(i as isize)).sqrt();
            i += 1;
        }
    }

    unsafe fn vd_sqrt_0(n: usize, a: *const f64, b: *mut f64) {
        let mut i = 0;
        while (i + Self::F64_WIDTH) <= n {
            let x = Self::loadu_f64(a.offset(i as isize));
            let r = Self::sqrt_f64(x);
            Self::storeu_f64(b.offset(i as isize), r);
            i += Self::F64_WIDTH;
        }
        while i < n {
            *b.offset(i as isize) = (*a.offset(i as isize)).sqrt();
            i += 1;
        }
    }
}

pub(crate) trait UnaryFn2: Simd {
    unsafe fn vs_sin_0(n: usize, a: *const f32, b: *mut f32) {
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
    unsafe fn vs_cos_0(n: usize, a: *const f32, b: *mut f32) {
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
