use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::prelude::*;
use ultraloglog::{Estimator, MaximumLikelihoodEstimator, OptimalFGRAEstimator, UltraLogLog};

fn bench_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("UltraLogLog Insert");
    let mut rng = StdRng::seed_from_u64(42);

    // Test different precision values
    for p in 8..=16 {
        // Create a vector of random numbers to insert
        let numbers: Vec<u64> = (0..10_000).map(|_| rng.gen()).collect();

        group.bench_with_input(BenchmarkId::from_parameter(p), &p, |b, &p| {
            b.iter(|| {
                let mut ull = UltraLogLog::new(p).unwrap();
                for &num in numbers.iter() {
                    ull.add(black_box(num));
                }
            });
        });
    }
    group.finish();
}

fn bench_estimate_fgra(c: &mut Criterion) {
    let mut group = c.benchmark_group("UltraLogLog Estimate (FGRA)");
    let mut rng = StdRng::seed_from_u64(42);
    let estimator = OptimalFGRAEstimator;

    // Test different precision values
    for p in 8..=16 {
        // Create and populate UltraLogLog
        let mut ull = UltraLogLog::new(p).unwrap();
        for _ in 0..10_000 {
            ull.add(rng.gen());
        }

        group.bench_with_input(BenchmarkId::from_parameter(p), &p, |b, _| {
            b.iter(|| {
                black_box(estimator.estimate(&ull));
            });
        });
    }
    group.finish();
}

fn bench_estimate_ml(c: &mut Criterion) {
    let mut group = c.benchmark_group("UltraLogLog Estimate (ML)");
    let mut rng = StdRng::seed_from_u64(42);
    let estimator = MaximumLikelihoodEstimator;

    // Test different precision values
    for p in 8..=12 {
        // Create and populate UltraLogLog
        let mut ull = UltraLogLog::new(p).unwrap();
        for _ in 0..10_000 {
            ull.add(rng.gen());
        }

        group.bench_with_input(BenchmarkId::from_parameter(p), &p, |b, _| {
            b.iter(|| {
                black_box(estimator.estimate(&ull));
            });
        });
    }
    group.finish();
}

fn bench_estimator_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("Estimator Comparison");
    let mut rng = StdRng::seed_from_u64(42);
    let fgra = OptimalFGRAEstimator;
    let ml = MaximumLikelihoodEstimator;

    // Test with different data sizes for p=12
    let p = 12;
    for &size in &[1000, 10_000, 100_000] {
        let mut ull = UltraLogLog::new(p).unwrap();
        for _ in 0..size {
            ull.add(rng.gen());
        }

        group.bench_with_input(BenchmarkId::new("FGRA", size), &size, |b, _| {
            b.iter(|| {
                black_box(fgra.estimate(&ull));
            });
        });

        group.bench_with_input(BenchmarkId::new("ML", size), &size, |b, _| {
            b.iter(|| {
                black_box(ml.estimate(&ull));
            });
        });
    }
    group.finish();
}

fn bench_combined_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("UltraLogLog Combined");
    let mut rng = StdRng::seed_from_u64(42);

    // Test different precision values
    for p in 8..=16 {
        // Create a vector of random numbers
        let numbers: Vec<u64> = (0..1_000).map(|_| rng.gen()).collect();

        group.bench_with_input(BenchmarkId::from_parameter(p), &p, |b, &p| {
            b.iter(|| {
                let mut ull = UltraLogLog::new(p).unwrap();
                // Insert numbers in batches and estimate periodically
                for chunk in numbers.chunks(100) {
                    for &num in chunk {
                        ull.add(black_box(num));
                    }
                    black_box(ull.get_distinct_count_estimate());
                }
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_insert,
    bench_estimate_fgra,
    bench_estimate_ml,
    bench_estimator_comparison,
    bench_combined_operations
);
criterion_main!(benches);
