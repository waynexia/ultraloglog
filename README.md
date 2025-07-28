# UltraLogLog

[![Crates.io](https://img.shields.io/crates/v/ultraloglog.svg)](https://crates.io/crates/ultraloglog)
[![PyPI](https://img.shields.io/pypi/v/ultraloglog.svg)](https://pypi.org/project/ultraloglog/)
[![Documentation](https://docs.rs/ultraloglog/badge.svg)](https://docs.rs/ultraloglog)

Rust implementation of the [UltraLogLog algorithm](https://arxiv.org/pdf/2308.16862). Ultraloglog is more space efficient than the widely used HyperLogLog, but can be slower. FGRA estimator or MLE estimator can be used. 

## Usage

```rust
use ultraloglog::{Estimator, MaximumLikelihoodEstimator, OptimalFGRAEstimator, UltraLogLog};

let mut ull = UltraLogLog::new(6).unwrap();

ull.add_value("apple")
    .add_value("banana")
    .add_value("cherry")
    .add_value("033");
let est = ull.get_distinct_count_estimate();
```

The serde feature can be activated so that the sketch can be saved to disk and then loaded.
```rust
use ultraloglog::{Estimator, MaximumLikelihoodEstimator, OptimalFGRAEstimator, UltraLogLog};
use std::fs::{remove_file, File};
use std::io::{BufReader, BufWriter};

let file_path = "test_ultraloglog.bin";

// Create UltraLogLog and add data
let mut ull = UltraLogLog::new(5).expect("Failed to create ULL");
ull.add(123456789);
ull.add(987654321);
let original_estimate = ull.get_distinct_count_estimate();

// Save to file using writer
let file = File::create(file_path).expect("Failed to create file");
let writer = BufWriter::new(file);
ull.save(writer).expect("Failed to save UltraLogLog");

// Load from file using reader
let file = File::open(file_path).expect("Failed to open file");
let reader = BufReader::new(file);
let loaded_ull = UltraLogLog::load(reader).expect("Failed to load UltraLogLog");
let loaded_estimate = loaded_ull.get_distinct_count_estimate();
```

## Python Bindings
This crate also provides Python bindings for the UltraLogLog algorithm using [PyO3](https://pyo3.rs/). See [example.py](./example.py) for usage.

```python
import ultraloglog

# Create a new UltraLogLog sketch
ull = ultraloglog.PyUltraLogLog(12)  # precision parameter

# Add values
ull.add_str("hello")
ull.add_int(42)
ull.add_float(3.14)

# Get estimated count
print(f"Estimated distinct count: {ull.count()}")
```

### Installation

#### Using pip

This package is available as [`ultraloglog`](https://pypi.org/project/ultraloglog/) in PyPI. You can install it using:

```bash
pip install ultraloglog
``` 

#### From Source

*`uv` is recommended to manage virtual environments.*

1. Install Rust, and maturin `pip install maturin`
2. Build and install: `maturin develop --release`


## 64-bit hash function
As mentioned in the paper, high quality 64-bit hash function is key to ultraloglog algorithm. We tested several modern 64-bit hash libraries and found that xxhash-rust (default) and wyhash-rs worked well. However, users can easily replace the default xxhash-rust with polymurhash, komihash, ahash and t1ha et.al. See testing section for details. 

## Reference
Ertl, O., 2024. UltraLogLog: A Practical and More Space-Efficient Alternative to HyperLogLog for Approximate Distinct Counting. Proceedings of the VLDB Endowment, 17(7), pp.1655-1668.