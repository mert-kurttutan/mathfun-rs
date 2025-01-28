use criterion::{criterion_group, criterion_main, Criterion};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::Duration;

pub fn random_matrix_std<T>(arr: &mut [T])
where
    rand::distr::StandardUniform: rand::prelude::Distribution<T>,
{
    let mut x = StdRng::seed_from_u64(43);
    arr.iter_mut().for_each(|p| *p = x.random::<T>());
}

use criterion::BenchmarkId;
use std::any::type_name;

pub fn bench_blas_group3<M: criterion::measurement::Measurement, TA: 'static, TB: 'static>(
    bench_c: &mut BenchmarkGroup<M>,
    m: usize,
    a: *const TA,
    b: *const TB,
) {
    let a = a as *const f32;
    let b = b as *mut f32;
    let type_name = type_name::<TA>();
    bench_c.bench_with_input(BenchmarkId::new(format!("{}-mathfun-vs_exp", type_name), m), &m, |bench_b, _x| {
        bench_b.iter(|| unsafe {
            mathfun::vs_exp(m, a, b);
        })
    });
}

use criterion::BenchmarkGroup;
fn bench_exp(c: &mut Criterion) {
    let mut group = c.benchmark_group("bbb");
    for i in 12..30 {
        let m = 1 << i;
        let mut a = vec![1.0; m];
        let mut b = vec![1.0; m];
        random_matrix_std(&mut a);
        random_matrix_std(&mut b);
        bench_blas_group3(&mut group, m, a.as_ptr(), b.as_mut_ptr());
    }
    group.finish();
}

criterion_group!(
    name = benches;
    config = Criterion::default()
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(2))
        .sample_size(10);
    targets = bench_exp
);
criterion_main!(benches);
