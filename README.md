# mathfun-rs
Prvoides simd-enabled implementation of several mathfunctions to help deep learning and high performance computing in rust have high performance.

# Threading control
The library uses rayon for parallelism. You can control the number of threads used by setting the `MATHFUN_NUM_THREADS` environment variable. Max number of thrading is 2 due to higher number of threads causing performance degradation.

# Hardware support:

# Functions

| Function | SSE2 | AVX,SSE2 | AVX2,FMA | AVX512F | NEON | WASM |
|----------|-----|-----|-----|--------|------|------|
| vs_exp      |  ✅   |   ✅  |   ✅  |    ✅    |   ✅   |   ❌   |
| vs_ln       |  ✅   |   ✅  |  ✅   |   ✅      |   ✅   |   ❌   |
| vs_sin      |  ✅   |   ✅  |  ✅   |    ❌     |  ✅    |   ❌   |
| vs_cos      |  ✅   |  ✅   |   ✅  |    ❌     |  ✅    |   ❌   |
| vs_tanh     |  ✅   |   ✅  |  ✅   |    ✅    |    ✅  |    ❌  |
