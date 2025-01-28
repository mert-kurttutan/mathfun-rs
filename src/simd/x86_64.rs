use super::{Simd, UnaryFn1, UnaryFn2};
use core::arch::asm;
use core::arch::x86_64::*;

pub(crate) struct Avx2Fma {}

impl Simd for Avx2Fma {
    const F32_WIDTH: usize = 8;
    const F64_WIDTH: usize = 4;

    type Vf32 = __m256;
    type Vf64 = __m256d;
    type Vi32 = __m256i;
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
    unsafe fn loadu_f64(ptr: *const f64) -> __m256d {
        _mm256_loadu_pd(ptr)
    }

    #[inline(always)]
    unsafe fn sqrt_f32(a: __m256) -> __m256 {
        _mm256_sqrt_ps(a)
    }

    #[inline(always)]
    unsafe fn sqrt_f64(a: __m256d) -> __m256d {
        _mm256_sqrt_pd(a)
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
    unsafe fn cvtt_f32_i32(a: __m256) -> __m256i {
        _mm256_cvttps_epi32(a)
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
    unsafe fn storeu_f64(ptr: *mut f64, a: __m256d) {
        _mm256_storeu_pd(ptr, a)
    }

    #[inline(always)]
    unsafe fn sub_f32(a: __m256, b: __m256) -> __m256 {
        _mm256_sub_ps(a, b)
    }

    #[inline(always)]
    unsafe fn cmp_eq_f32(a: __m256, b: __m256) -> __m256 {
        _mm256_cmp_ps(a, b, _CMP_EQ_OS)
    }

    #[inline(always)]
    unsafe fn cmp_lt_f32(a: __m256, b: __m256) -> __m256 {
        _mm256_cmp_ps(a, b, _CMP_LT_OS)
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
    unsafe fn floor_f32(a: __m256) -> __m256 {
        _mm256_round_ps(a, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC)
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

impl UnaryFn1 for Avx2Fma {}
impl UnaryFn2 for Avx2Fma {}

impl Avx2Fma {
    #[target_feature(enable = "avx2,avx,fma")]
    pub(crate) unsafe fn vs_exp(n: usize, a: *const f32, b: *mut f32) {
        Self::vs_exp_0(n, a, b)
    }

    #[target_feature(enable = "avx2,avx,fma")]
    pub(crate) unsafe fn vs_ln(n: usize, a: *const f32, b: *mut f32) {
        Self::vs_ln_0(n, a, b)
    }

    #[target_feature(enable = "avx2,avx,fma")]
    pub(crate) unsafe fn vs_tanh(n: usize, a: *const f32, b: *mut f32) {
        Self::vs_tanh_0(n, a, b)
    }

    #[target_feature(enable = "avx2,avx,fma")]
    pub(crate) unsafe fn vs_sin(n: usize, a: *const f32, b: *mut f32) {
        Self::vs_sin_0(n, a, b)
    }

    #[target_feature(enable = "avx2,avx,fma")]
    pub(crate) unsafe fn vs_cos(n: usize, a: *const f32, b: *mut f32) {
        Self::vs_cos_0(n, a, b)
    }
}

pub(crate) struct AvxSse2 {}

impl Simd for AvxSse2 {
    const F32_WIDTH: usize = 8;
    const F64_WIDTH: usize = 4;

    type Vf32 = __m256;
    type Vf64 = __m256d;
    type Vi32 = __m256i;
    #[inline(always)]
    unsafe fn set1_f32(x: f32) -> __m256 {
        _mm256_set1_ps(x)
    }
    #[inline(always)]
    unsafe fn set1_i32(x: i32) -> __m256i {
        _mm256_set1_epi32(x)
    }
    #[inline(always)]
    unsafe fn loadu_f64(ptr: *const f64) -> __m256d {
        _mm256_loadu_pd(ptr)
    }

    #[inline(always)]
    unsafe fn storeu_f64(ptr: *mut f64, a: __m256d) {
        _mm256_storeu_pd(ptr, a)
    }

    #[inline(always)]
    unsafe fn sqrt_f32(a: __m256) -> __m256 {
        _mm256_sqrt_ps(a)
    }

    #[inline(always)]
    unsafe fn sqrt_f64(a: __m256d) -> __m256d {
        _mm256_sqrt_pd(a)
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
    unsafe fn cvtt_f32_i32(a: __m256) -> __m256i {
        _mm256_cvttps_epi32(a)
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
    unsafe fn cmp_eq_f32(a: __m256, b: __m256) -> __m256 {
        _mm256_cmp_ps(a, b, _CMP_EQ_OS)
    }

    #[inline(always)]
    unsafe fn cmp_lt_f32(a: __m256, b: __m256) -> __m256 {
        _mm256_cmp_ps(a, b, _CMP_LT_OS)
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
    unsafe fn floor_f32(a: __m256) -> __m256 {
        _mm256_round_ps(a, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC)
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

impl UnaryFn1 for AvxSse2 {
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

impl UnaryFn2 for AvxSse2 {}

impl AvxSse2 {
    #[target_feature(enable = "avx,sse2,sse")]
    pub(crate) unsafe fn vs_exp(n: usize, a: *const f32, b: *mut f32) {
        Self::vs_exp_0(n, a, b)
    }

    #[target_feature(enable = "avx,sse2,sse")]
    pub(crate) unsafe fn vs_ln(n: usize, a: *const f32, b: *mut f32) {
        Self::vs_ln_0(n, a, b)
    }

    #[target_feature(enable = "avx,sse2,sse")]
    pub(crate) unsafe fn vs_tanh(n: usize, a: *const f32, b: *mut f32) {
        Self::vs_tanh_0(n, a, b)
    }

    #[target_feature(enable = "avx,sse2,sse")]
    pub(crate) unsafe fn vs_sin(n: usize, a: *const f32, b: *mut f32) {
        Self::vs_sin_0(n, a, b)
    }

    #[target_feature(enable = "avx,sse2,sse")]
    pub(crate) unsafe fn vs_cos(n: usize, a: *const f32, b: *mut f32) {
        Self::vs_cos_0(n, a, b)
    }

    #[target_feature(enable = "avx")]
    pub(crate) unsafe fn vs_sqrt(n: usize, a: *const f32, b: *mut f32) {
        Self::vs_sqrt_0(n, a, b)
    }

    #[target_feature(enable = "avx")]
    pub(crate) unsafe fn vd_sqrt(n: usize, a: *const f64, b: *mut f64) {
        Self::vd_sqrt_0(n, a, b)
    }
}

pub(crate) struct Avx512f {}
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
    const EXP_HI: f32 = 88.3762626647949 * 0.5;
    const EXP_LO: f32 = -88.3762626647949 * 0.5;
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

        "vaddps %zmm9, %zmm15, %zmm16",
        "vaddps %zmm9, %zmm8, %zmm9",
        "vdivps %zmm9, %zmm16, %zmm9",
        "vminps %zmm9, %zmm8, %zmm9",
        "vmaxps %zmm9, %zmm15, %zmm9",

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

impl Avx512f {
    pub(crate) unsafe fn vs_exp(n: usize, a: *const f32, b: *mut f32) {
        vs_exp_avx512f_asm(n, a, b);
    }

    pub(crate) unsafe fn vs_ln(n: usize, a: *const f32, b: *mut f32) {
        vs_ln_avx512f_asm(n, a, b);
    }

    pub(crate) unsafe fn vs_tanh(n: usize, a: *const f32, b: *mut f32) {
        vs_tanh_avx512f_asm(n, a, b);
    }

    pub(crate) unsafe fn vs_sqrt(n: usize, a: *const f32, b: *mut f32) {
        vs_sqrt_avx512f_asm(n, a, b);
    }

    pub(crate) unsafe fn vd_sqrt(n: usize, a: *const f64, b: *mut f64) {
        vd_sqrt_avx512f_asm(n, a, b);
    }
}
