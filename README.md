# mathfun-rs
Prvoides simd-enabled implementation of several mathfunctions to help deep learning and high performance computing in rust have high performance.

# Threading control
You can control the number of threads used by setting the `MATHFUN_NUM_THREADS` environment variable. Max number of thrading is 2 due to higher number of threads causing performance degradation.

# Hardware support:

Feature detection is done at runtime. The library will use the best available implementation for the current hardware, **except for the wasm32 target.**
The dynamic feature detection for wasm target is not possible, see [this issue](https://github.com/rust-lang/rust/issues/74372#issuecomment-889458705) for more information.
To enable simd on wasm32 target, you need to use tthe flag to enable the feature at compile time, e.g `-C target-feature=+simd128`, and make sure the wasm engine has support for the simd feature.

# Functions

| Function | SSE2 | AVX,SSE2 | AVX2,FMA | AVX512F | NEON | WASM |
|----------|-----|-----|-----|--------|------|------|
| vs_exp      |  ✅   |   ✅  |   ✅  |    ✅    |   ✅   |   ✅  |
| vs_ln       |  ✅   |   ✅  |  ✅   |   ✅      |   ✅   |   ✅   |
| vs_sin      |  ✅   |   ✅  |  ✅   |    ❌     |  ✅    |   ✅   |
| vs_cos      |  ✅   |  ✅   |   ✅  |    ❌     |  ✅    |   ✅   |
| vs_tanh     |  ✅   |   ✅  |  ✅   |    ✅    |    ✅  |    ✅  |
| vs_sqrt    |  ✅   |   ✅  |  ✅   |    ✅    |    ✅  |    ✅  |
| vd_sqrt    | ✅   |   ✅  |  ✅   |    ✅    |    ✅  |    ✅  |
