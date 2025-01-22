# mathfun-rs
Prvoides simd-enabled implementation of several mathfunctions to help deep learning and high performance computing in rust have state of the art performance.

# Hardware support:
// make table where first row is simd extension: sse, avx, fma, avx512f, neon, wasm and row is the name of funciton (exp, ln, sin, cos, tanh), each cell green tik or red cross to show its availability

# Functions

| Function | SSE2 | AVX,SSE2 | AVX2,FMA | AVX512F | NEON | WASM |
|----------|-----|-----|-----|--------|------|------|
| vs_exp      |  ❌  |   ✅  |   ✅  |    ✅     |   ❌   |   ❌   |
| vs_ln       |  ❌   |   ✅  |  ✅   |   ✅      |   ❌   |   ❌   |
| vs_sin      |  ❌   |   ✅  |  ✅   |    ❌     |  ❌    |   ❌   |
| vs_cos      |   ❌  |  ✅   |   ✅  |    ❌     |  ❌    |   ❌   |
| vs_tanh     |  ❌   |   ✅  |  ✅   |    ✅    |    ❌  |    ❌  |
