//! Python bindings

use crate::*;
use pyo3::prelude::*;
use pyo3::types::PyModule;

/// Python wrapper for UltraLogLog sketch
#[pyclass]
pub struct PyUltraLogLog {
    inner: UltraLogLog,
}

#[pymethods]
impl PyUltraLogLog {
    /// Create a new UltraLogLog sketch with the given precision parameter.
    /// The precision parameter p must be in the range [3, 26].
    /// It defines the size of the internal state, which is a byte array of length 2^p.
    #[new]
    fn new(p: u32) -> PyResult<Self> {
        match UltraLogLog::new(p) {
            Ok(ull) => Ok(PyUltraLogLog { inner: ull }),
            Err(e) => Err(pyo3::exceptions::PyValueError::new_err(e)),
        }
    }

    /// Add a value to the sketch using hash of the value
    fn add(&mut self, value: u64) {
        self.inner.add(value);
    }

    /// Add a string value to the sketch
    fn add_str(&mut self, value: &str) {
        self.inner.add_value(value);
    }

    /// Add an integer value to the sketch
    fn add_int(&mut self, value: i64) {
        self.inner.add_value(value);
    }

    /// Add a float value to the sketch
    fn add_float(&mut self, value: f64) {
        // Convert f64 to u64 bits for hashing since f64 doesn't implement Hash
        self.inner.add_value(value.to_bits());
    }

    /// Get the estimated count of distinct elements
    fn count(&self) -> f64 {
        self.inner.get_distinct_count_estimate()
    }

    /// Get the precision parameter of this sketch
    fn get_p(&self) -> u32 {
        self.inner.get_p()
    }

    /// Check if the sketch is empty
    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Reset the sketch to its initial state
    fn reset(&mut self) {
        self.inner.reset();
    }

    /// Create a copy of this sketch
    fn copy(&self) -> Self {
        PyUltraLogLog {
            inner: self.inner.copy(),
        }
    }

    /// Merge this sketch with another sketch
    fn merge(&mut self, other: &PyUltraLogLog) -> PyResult<()> {
        match self.inner.add_sketch(&other.inner) {
            Ok(_) => Ok(()),
            Err(e) => Err(pyo3::exceptions::PyValueError::new_err(e)),
        }
    }

    /// Create a new sketch that is the result of merging two sketches
    #[staticmethod]
    fn merge_sketches(sketch1: &PyUltraLogLog, sketch2: &PyUltraLogLog) -> PyResult<PyUltraLogLog> {
        match UltraLogLog::merge(&sketch1.inner, &sketch2.inner) {
            Ok(merged) => Ok(PyUltraLogLog { inner: merged }),
            Err(e) => Err(pyo3::exceptions::PyValueError::new_err(e)),
        }
    }

    /// Downsize the sketch to a smaller precision
    fn downsize(&self, p: u32) -> PyResult<PyUltraLogLog> {
        match self.inner.downsize(p) {
            Ok(downsized) => Ok(PyUltraLogLog { inner: downsized }),
            Err(e) => Err(pyo3::exceptions::PyValueError::new_err(e)),
        }
    }

    /// String representation of the sketch
    fn __repr__(&self) -> String {
        format!(
            "UltraLogLog(p={}, estimate={:.2})",
            self.get_p(),
            self.count()
        )
    }

    /// Length returns the estimated count
    fn __len__(&self) -> usize {
        self.count() as usize
    }
}

/// Python module definition
#[pymodule]
fn ultraloglog(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyUltraLogLog>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
