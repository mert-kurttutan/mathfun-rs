#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use super::{Simd, UnaryFn1, UnaryFn2};

pub(crate) struct Sse2 {}

impl Simd for Sse2 {
    const F32_WIDTH: usize = 4;
    const F64_WIDTH: usize = 2;

    type Vf32 = __m128;
    type Vf64 = __m128d;
    type Vi32 = __m128i;

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
    unsafe fn loadu_f64(ptr: *const f64) -> __m128d {
        _mm_loadu_pd(ptr)
    }

    #[inline(always)]
    unsafe fn storeu_f64(ptr: *mut f64, a: __m128d) {
        _mm_storeu_pd(ptr, a)
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
    unsafe fn sqrt_f32(a: __m128) -> __m128 {
        _mm_sqrt_ps(a)
    }

    #[inline(always)]
    unsafe fn sqrt_f64(a: __m128d) -> __m128d {
        _mm_sqrt_pd(a)
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
    unsafe fn cmp_eq_f32(a: __m128, b: __m128) -> __m128 {
        _mm_cmp_ps(a, b, _CMP_EQ_OS)
    }

    #[inline(always)]
    unsafe fn cmp_lt_f32(a: __m128, b: __m128) -> __m128 {
        _mm_cmp_ps(a, b, _CMP_LT_OS)
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
    unsafe fn floor_f32(a: __m128) -> __m128 {
        _mm_round_ps(a, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC)
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
        let zero_mask = Self::cmp_eq_f32(a, Self::set1_f32(0.0));
        let nan_mask = Self::cmp_lt_f32(a, Self::set1_f32(0.0));
        let inv_mant_mask = Self::cast_i32_f32(Self::set1_i32(!0x7f800000));
        let inf_mask = Self::cmp_eq_f32(a, Self::set1_f32(f32::INFINITY));
        let denorm_mul = Self::set1_f32(134217730.);
        let denorm_th = Self::set1_f32(1.1754945e-38);
        let denorm_mask = Self::cmp_lt_f32(a, denorm_th);
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

impl UnaryFn1 for Sse2 {
    unsafe fn vs_exp_0(n: usize, a: *const f32, b: *mut f32) {
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
            fx = Self::floor_f32(fx);
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

impl UnaryFn2 for Sse2 {}

impl Sse2 {
    #[target_feature(enable = "sse2")]
    pub(crate) unsafe fn vs_exp(n: usize, a: *const f32, b: *mut f32) {
        Self::vs_exp_0(n, a, b)
    }

    #[target_feature(enable = "sse2")]
    pub(crate) unsafe fn vs_ln(n: usize, a: *const f32, b: *mut f32) {
        Self::vs_ln_0(n, a, b)
    }

    #[target_feature(enable = "sse2")]
    pub(crate) unsafe fn vs_tanh(n: usize, a: *const f32, b: *mut f32) {
        Self::vs_tanh_0(n, a, b)
    }

    #[target_feature(enable = "sse2")]
    pub(crate) unsafe fn vs_cos(n: usize, a: *const f32, b: *mut f32) {
        Self::vs_cos_0(n, a, b)
    }

    #[target_feature(enable = "sse2")]
    pub(crate) unsafe fn vs_sin(n: usize, a: *const f32, b: *mut f32) {
        Self::vs_sin_0(n, a, b)
    }

    #[target_feature(enable = "sse2")]
    pub(crate) unsafe fn vs_sqrt(n: usize, a: *const f32, b: *mut f32) {
        Self::vs_sqrt_0(n, a, b)
    }

    #[target_feature(enable = "sse2")]
    pub(crate) unsafe fn vd_sqrt(n: usize, a: *const f64, b: *mut f64) {
        Self::vd_sqrt_0(n, a, b)
    }
}
