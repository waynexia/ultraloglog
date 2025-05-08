# UltraLogLog

Rust implementation of the [UltraLogLog algorithm](https://arxiv.org/pdf/2308.16862). Ultraloglog is more space efficient than the widely used HyperLogLog, but can be slower. FGRA estimator and MLE estimator can be used. 

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

## Reference
Ertl, O., 2024. UltraLogLog: A Practical and More Space-Efficient Alternative to HyperLogLog for Approximate Distinct Counting. Proceedings of the VLDB Endowment, 17(7), pp.1655-1668.