use core::arch::aarch64::*;
pub(crate) struct Neon {}

use super::{Simd, UnaryFn1, UnaryFn2};
impl Simd for Neon {
    const F32_WIDTH: usize = 4;
    const F64_WIDTH: usize = 2;
    type Vf32 = float32x4_t;
    type Vf64 = float64x2_t;
    type Vi32 = int32x4_t;
    #[inline(always)]
    unsafe fn set1_f32(x: f32) -> Self::Vf32 {
        vdupq_n_f32(x)
    }
    #[inline(always)]
    unsafe fn set1_i32(x: i32) -> Self::Vi32 {
        vdupq_n_s32(x)
    }

    #[inline(always)]
    unsafe fn sqrt_f32(x: Self::Vf32) -> Self::Vf32 {
        vsqrtq_f32(x)
    }

    #[inline(always)]
    unsafe fn sqrt_f64(x: Self::Vf64) -> Self::Vf64 {
        vsqrtq_f64(x)
    }

    #[inline(always)]
    unsafe fn loadu_f32(ptr: *const f32) -> Self::Vf32 {
        vld1q_f32(ptr)
    }
    #[inline(always)]
    unsafe fn loadu_f64(ptr: *const f64) -> Self::Vf64 {
        vld1q_f64(ptr)
    }

    #[inline(always)]
    unsafe fn storeu_f32(ptr: *mut f32, a: Self::Vf32) {
        vst1q_f32(ptr, a)
    }

    #[inline(always)]
    unsafe fn storeu_f64(ptr: *mut f64, a: Self::Vf64) {
        vst1q_f64(ptr, a)
    }

    #[inline(always)]
    unsafe fn and_f32(a: Self::Vf32, b: Self::Vf32) -> Self::Vf32 {
        vreinterpretq_f32_s32(vandq_s32(vreinterpretq_s32_f32(a), vreinterpretq_s32_f32(b)))
    }

    #[inline(always)]
    unsafe fn mul_f32(a: Self::Vf32, b: Self::Vf32) -> Self::Vf32 {
        vmulq_f32(a, b)
    }

    #[inline(always)]
    unsafe fn add_i32(a: Self::Vi32, b: Self::Vi32) -> Self::Vi32 {
        vaddq_s32(a, b)
    }

    #[inline(always)]
    unsafe fn and_i32(a: Self::Vi32, b: Self::Vi32) -> Self::Vi32 {
        vandq_s32(a, b)
    }
    #[inline(always)]
    unsafe fn cvt_i32_f32(a: Self::Vi32) -> Self::Vf32 {
        vcvtq_f32_s32(a)
    }
    #[inline(always)]
    unsafe fn cvt_f32_i32(a: Self::Vf32) -> Self::Vi32 {
        vcvtaq_s32_f32(a)
    }

    #[inline(always)]
    unsafe fn sub_i32(a: Self::Vi32, b: Self::Vi32) -> Self::Vi32 {
        vaddq_s32(a, b)
    }

    #[inline(always)]
    unsafe fn andnot_i32(a: Self::Vi32, b: Self::Vi32) -> Self::Vi32 {
        let b_not = vmvnq_s32(b);
        vandq_s32(a, b_not)
    }

    #[inline(always)]
    unsafe fn slli_i32<const IMM8: i32>(a: Self::Vi32) -> Self::Vi32 {
        vshlq_n_s32(a, IMM8)
    }

    #[inline(always)]
    unsafe fn srli_i32<const IMM8: i32>(a: Self::Vi32) -> Self::Vi32 {
        vshrq_n_s32(a, IMM8)
    }

    #[inline(always)]
    unsafe fn cmpeq_i32(a: Self::Vi32, b: Self::Vi32) -> Self::Vi32 {
        vreinterpretq_s32_u32(vceqq_s32(a, b))
    }

    #[inline(always)]
    unsafe fn cast_i32_f32(a: Self::Vi32) -> Self::Vf32 {
        vreinterpretq_f32_s32(a)
    }

    #[inline(always)]
    unsafe fn fmadd_f32(a: Self::Vf32, b: Self::Vf32, c: Self::Vf32) -> Self::Vf32 {
        vfmaq_f32(a, b, c)
    }

    #[inline(always)]
    unsafe fn andnot_f32(a: Self::Vf32, b: Self::Vf32) -> Self::Vf32 {
        let a_i32 = vreinterpretq_s32_f32(a);
        let b_i32 = vreinterpretq_s32_f32(b);
        let c_i32 = Self::andnot_i32(a_i32, b_i32);
        vreinterpretq_f32_s32(c_i32)
    }

    #[inline(always)]
    unsafe fn add_f32(a: Self::Vf32, b: Self::Vf32) -> Self::Vf32 {
        vaddq_f32(a, b)
    }

    #[inline(always)]
    unsafe fn xor_f32(a: Self::Vf32, b: Self::Vf32) -> Self::Vf32 {
        vreinterpretq_f32_s32(veorq_s32(vreinterpretq_s32_f32(a), vreinterpretq_s32_f32(b)))
    }

    #[inline(always)]
    unsafe fn sub_f32(a: Self::Vf32, b: Self::Vf32) -> Self::Vf32 {
        vsubq_f32(a, b)
    }

    #[inline(always)]
    unsafe fn cmp_eq_f32(a: Self::Vf32, b: Self::Vf32) -> Self::Vf32 {
        vreinterpretq_f32_u32(vceqq_f32(a, b))
    }

    #[inline(always)]
    unsafe fn cmp_lt_f32(a: Self::Vf32, b: Self::Vf32) -> Self::Vf32 {
        vreinterpretq_f32_u32(vcltq_f32(a, b))
    }

    #[inline(always)]
    unsafe fn mask_mul_f32(mask: Self::Vf32, a: Self::Vf32, b: Self::Vf32) -> Self::Vf32 {
        let one = Self::set1_f32(1.0);
        let one = Self::andnot_f32(mask, one);
        let masked_one = Self::and_f32(b, mask);
        let masked_b = Self::or_f32(masked_one, one);
        let c = vmulq_f32(a, masked_b);
        c
    }

    #[inline(always)]
    unsafe fn mask_sub_f32(mask: Self::Vf32, a: Self::Vf32, b: Self::Vf32) -> Self::Vf32 {
        let masked_b = Self::and_f32(b, mask);
        vsubq_f32(a, masked_b)
    }

    #[inline(always)]
    unsafe fn or_f32(a: Self::Vf32, b: Self::Vf32) -> Self::Vf32 {
        vreinterpretq_f32_s32(vorrq_s32(vreinterpretq_s32_f32(a), vreinterpretq_s32_f32(b)))
    }

    #[inline(always)]
    unsafe fn mask_add_f32(mask: Self::Vf32, a: Self::Vf32, b: Self::Vf32) -> Self::Vf32 {
        let masked_b = vreinterpretq_f32_s32(vandq_s32(vreinterpretq_s32_f32(b), vreinterpretq_s32_f32(mask)));
        vaddq_f32(a, masked_b)
    }

    #[inline(always)]
    unsafe fn cast_f32_i32(a: Self::Vf32) -> Self::Vi32 {
        vreinterpretq_s32_f32(a)
    }

    #[inline(always)]
    unsafe fn min_f32(a: Self::Vf32, b: Self::Vf32) -> Self::Vf32 {
        vminq_f32(a, b)
    }

    #[inline(always)]
    unsafe fn max_f32(a: Self::Vf32, b: Self::Vf32) -> Self::Vf32 {
        vmaxq_f32(a, b)
    }

    #[inline(always)]
    unsafe fn floor_f32(a: Self::Vf32) -> Self::Vf32 {
        vrndmq_f32(a)
    }

    #[inline(always)]
    unsafe fn div_f32(a: Self::Vf32, b: Self::Vf32) -> Self::Vf32 {
        vdivq_f32(a, b)
    }

    #[inline(always)]
    unsafe fn get_exp_mant_f32(a: Self::Vf32) -> (Self::Vf32, Self::Vf32) {
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

impl UnaryFn1 for Neon {}
impl UnaryFn2 for Neon {}

impl Neon {
    #[target_feature(enable = "neon")]
    pub(crate) unsafe fn vs_exp(n: usize, a: *const f32, b: *mut f32) {
        Self::vs_exp_0(n, a, b)
    }

    #[target_feature(enable = "neon")]
    pub(crate) unsafe fn vs_ln(n: usize, a: *const f32, b: *mut f32) {
        Self::vs_ln_0(n, a, b)
    }

    #[target_feature(enable = "neon")]
    pub(crate) unsafe fn vs_tanh(n: usize, a: *const f32, b: *mut f32) {
        Self::vs_tanh_0(n, a, b)
    }

    #[target_feature(enable = "neon")]
    pub(crate) unsafe fn vs_sin(n: usize, a: *const f32, b: *mut f32) {
        Self::vs_sin_0(n, a, b)
    }

    #[target_feature(enable = "neon")]
    pub(crate) unsafe fn vs_cos(n: usize, a: *const f32, b: *mut f32) {
        Self::vs_cos_0(n, a, b)
    }

    #[target_feature(enable = "neon")]
    pub(crate) unsafe fn vs_sqrt(n: usize, a: *const f32, b: *mut f32) {
        Self::vs_sqrt_0(n, a, b)
    }

    #[target_feature(enable = "neon")]
    pub(crate) unsafe fn vd_sqrt(n: usize, a: *const f64, b: *mut f64) {
        Self::vd_sqrt_0(n, a, b)
    }
}
