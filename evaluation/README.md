# UltraLogLog Evaluation Suite

This directory contains tools for evaluating the accuracy of the UltraLogLog implementation under different parameters and scenarios.

## Overview

The evaluation suite consists of:
1. A Rust binary (`evl`) for running the experiments
2. A Python script (`exp.py`) for orchestrating experiments and generating plots
3. Support for testing both OptimalFGRA and MaximumLikelihood estimators
4. Evaluation of different precision parameters (3-26)
5. Two types of experiments:
   - Hash-based: Adding random hashes one by one and measuring error
   - Cardinality-based: Testing error at specific cardinality points

## Requirements

- Rust toolchain
- Python 3.x with numpy and matplotlib
- Cargo build tools

## Building

```bash
cargo build --release
```

## Running Experiments

The evaluation process has three steps:

1. Prepare input data:
```bash
# Generate hash files for hash-based experiments
python exp.py prepare -t hashes -c 100000 -r 500 data/

# Generate cardinality points for cardinality-based experiments
python exp.py prepare -t cardinalities -a 1000000000 data/
```

2. Run experiments:
```bash
# Run hash-based experiments with OptimalFGRA estimator
python exp.py run -m hashes -p 14 -e optimal -i data/ -o results/

# Run cardinality-based experiments with MaximumLikelihood estimator
python exp.py run -m cardinalities -p 14 -e ml -r 500 -c data/cardinalities.dat -o results/
```

3. Generate plots:
```bash
# Plot hash-based experiment results
python exp.py plot -m hashes -p 14 -e optimal -a 100000 results/ plots/

# Plot cardinality-based experiment results
python exp.py plot -m cardinalities -p 14 -e ml -a 1000000000 results/ plots/
```

## Parameters

- Precision (`-p`): Integer in range [3, 26]
- Estimator (`-e`): 
  - `optimal`: OptimalFGRA estimator
  - `ml`: MaximumLikelihood estimator
- Mode (`-m`):
  - `hashes`: Test accuracy as hashes are added one by one
  - `cardinalities`: Test accuracy at specific cardinality points
- Jobs (`-j`): Number of parallel jobs to run

## Example Results

The evaluation generates plots showing:
- Mean relative error
- Median relative error
- Standard deviation bands
- Error distribution across different cardinalities

Example plot interpretation:
- X-axis: Number of distinct elements (cardinality)
- Y-axis: Relative error |estimate - true|/true
- Shaded area: ±1 standard deviation from mean
- Lines: Mean and median relative error

## Notes

1. Higher precision parameters use more memory but generally provide better accuracy
2. The OptimalFGRA estimator is optimized for general use cases
3. The MaximumLikelihood estimator may provide better results in some scenarios
4. The cardinality-based experiments use a geometric progression (ratio 1.007) to test across orders of magnitude 
