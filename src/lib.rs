#[cfg(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64", target_arch = "wasm32"))]
pub(crate) mod simd;
pub(crate) mod unary;
use once_cell::sync::Lazy;

pub(crate) static RUNTIME_HW_CONFIG: Lazy<CpuFeatures> = Lazy::new(|| detect_hw_config());

pub use unary::{vd_cos, vd_exp, vd_ln, vd_sin, vd_sqrt, vd_tanh, vs_cos, vs_exp, vs_ln, vs_sin, vs_sqrt, vs_tanh};

#[allow(unused)]
#[cfg(target_arch = "x86_64")]
#[derive(Copy, Clone)]
pub(crate) struct CpuFeatures {
    pub(crate) sse: bool,
    pub(crate) sse2: bool,
    pub(crate) sse3: bool,
    pub(crate) ssse3: bool,
    pub(crate) avx: bool,
    pub(crate) avx2: bool,
    pub(crate) avx512f: bool,
    pub(crate) avx512f16: bool,
    // pub(crate) avx512bf16: bool,
    pub(crate) avx512bw: bool,
    pub(crate) avx512_vnni: bool,
    pub(crate) fma: bool,
    pub(crate) fma4: bool,
    pub(crate) f16c: bool,
}

#[allow(unused)]
#[cfg(target_arch = "x86")]
#[derive(Copy, Clone)]
pub struct CpuFeatures {
    pub sse: bool,
    pub sse2: bool,
    pub sse3: bool,
    pub ssse3: bool,
}

#[cfg(target_arch = "aarch64")]
#[derive(Copy, Clone)]
pub struct CpuFeatures {
    pub sve: bool,
    pub neon: bool,
}

#[cfg(target_arch = "wasm32")]
#[derive(Copy, Clone)]
pub struct CpuFeatures {
    pub simd128: bool,
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "x86", target_arch = "aarch64", target_arch = "wasm32")))]
#[derive(Copy, Clone)]
pub struct CpuFeatures {}

#[inline]
fn detect_hw_config() -> CpuFeatures {
    #[cfg(target_arch = "x86_64")]
    {
        let cpuid = raw_cpuid::CpuId::new();
        let feature_info = cpuid.get_feature_info().unwrap();
        let extended_feature_info = cpuid.get_extended_feature_info().unwrap();
        let sse = feature_info.has_sse();
        let sse2 = feature_info.has_sse2();
        let sse3 = feature_info.has_sse3();
        let ssse3 = feature_info.has_ssse3();
        let avx = feature_info.has_avx();
        let fma = feature_info.has_fma();
        let avx2 = extended_feature_info.has_avx2();
        let avx512f16 = extended_feature_info.has_avx512_fp16();
        // let avx512bf16 = extended_feature_info.has_avx512_bf16();
        let avx512f = extended_feature_info.has_avx512f();
        let avx512bw = extended_feature_info.has_avx512bw();
        let avx512_vnni = extended_feature_info.has_avx512vnni();
        let f16c = feature_info.has_f16c();
        let extended_processor_info = cpuid.get_extended_processor_and_feature_identifiers().unwrap();
        let fma4 = extended_processor_info.has_fma4();
        let cpu_ft = CpuFeatures {
            sse,
            sse2,
            sse3,
            ssse3,
            avx,
            avx2,
            avx512f,
            avx512f16,
            avx512bw,
            avx512_vnni,
            fma,
            fma4,
            f16c,
        };
        cpu_ft
    }
    #[cfg(target_arch = "x86")]
    {
        let cpuid = raw_cpuid::CpuId::new();
        let feature_info = cpuid.get_feature_info().unwrap();
        let sse = feature_info.has_sse();
        let sse2 = feature_info.has_sse2();
        let sse3 = feature_info.has_sse3();
        let ssse3 = feature_info.has_ssse3();
        let cpu_ft = CpuFeatures { sse, sse2, sse3, ssse3 };
        return cpu_ft;
    }
    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::is_aarch64_feature_detected;
        let neon = is_aarch64_feature_detected!("neon");
        let sve = is_aarch64_feature_detected!("sve");

        return CpuFeatures { neon, sve };
    }

    #[cfg(target_arch = "wasm32")]
    {
        #[cfg(target_feature = "simd128")]
        let simd128 = true;
        #[cfg(not(target_feature = "simd128"))]
        let simd128 = false;
        return CpuFeatures { simd128 };
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "x86", target_arch = "aarch64", target_arch = "wasm32")))]
    {
        CpuFeatures {}
    }
}
