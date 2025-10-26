// This implementation is based on the Java version
// https://github.com/dynatrace-oss/hash4j/
// See the paper "UltraLogLog: A Practical and More Space-Efficient Alternative to HyperLogLog for Approximate Distinct Counting"

#[cfg(feature = "python")]
mod python;
#[cfg(feature = "python")]
pub use python::*;

// Constants for UltraLogLog implementation
const MIN_P: u32 = 3;
const MAX_P: u32 = 26; // 32 - 6 (same as Java implementation)
const MIN_STATE_SIZE: usize = 1 << MIN_P;
const MAX_STATE_SIZE: usize = 1 << MAX_P;

// Constants for OptimalFGRAEstimator
const ETA_0: f64 = 4.663135422063788;
const ETA_1: f64 = 2.1378502137958524;
const ETA_2: f64 = 2.781144650979996;
const ETA_3: f64 = 0.9824082545153715;
const TAU: f64 = 0.8194911375910897;
// const V: f64 = 0.6118931496978437;

const POW_2_TAU: f64 = 1.7631258657688563; // pre-computed 2.0f64.powf(TAU)
const POW_2_MINUS_TAU: f64 = 0.5670918786435586; // pre-computed 2.0f64.powf(-TAU)
const POW_4_MINUS_TAU: f64 = 0.3216030842364037; // pre-computed 4.0f64.powf(-TAU)

// Additional constants for OptimalFGRAEstimator
const MINUS_INV_TAU: f64 = -1.0 / TAU;
const ETA_X: f64 = ETA_0 - ETA_1 - ETA_2 + ETA_3;
const ETA23X: f64 = (ETA_2 - ETA_3) / ETA_X;
const ETA13X: f64 = (ETA_1 - ETA_3) / ETA_X;
const ETA3012XX: f64 = (ETA_3 * ETA_0 - ETA_1 * ETA_2) / (ETA_X * ETA_X);
const POW_4_MINUS_TAU_ETA_23: f64 = POW_4_MINUS_TAU * (ETA_2 - ETA_3);
const POW_4_MINUS_TAU_ETA_01: f64 = POW_4_MINUS_TAU * (ETA_0 - ETA_1);
const POW_4_MINUS_TAU_ETA_3: f64 = POW_4_MINUS_TAU * ETA_3;
const POW_4_MINUS_TAU_ETA_1: f64 = POW_4_MINUS_TAU * ETA_1;
const POW_2_MINUS_TAU_ETA_X: f64 = POW_2_MINUS_TAU * ETA_X;
const PHI_1: f64 = ETA_0 / (POW_2_TAU * (2.0 * POW_2_TAU - 1.0));
const P_INITIAL: f64 = ETA_X * (POW_4_MINUS_TAU / (2.0 - POW_2_MINUS_TAU));

// Constants for MaximumLikelihoodEstimator
const INV_SQRT_FISHER_INFORMATION: f64 = 0.7608621002725182;
const ML_EQUATION_SOLVER_EPS: f64 = 0.001 * INV_SQRT_FISHER_INFORMATION;
const ML_BIAS_CORRECTION_CONSTANT: f64 = 0.48147376527720065;
const C0: f64 = -1.0 / 3.0;
const C1: f64 = 1.0 / 45.0;
const C2: f64 = 1.0 / 472.5;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::hash::{BuildHasher, Hash, Hasher};
use xxhash_rust::xxh3::Xxh3;

/// A trait for observing state changes in the UltraLogLog sketch
pub trait StateChangeObserver {
    fn state_changed(&mut self, probability_decrement: f64);
}

/// UltraLogLog is a sketch for approximate distinct counting that is more space efficient than HyperLogLog
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct UltraLogLog {
    state: Vec<u8>,
}

impl UltraLogLog {
    /// Creates a new UltraLogLog sketch with the given precision parameter.
    ///
    /// The precision parameter `p` must be in the range {3, 4, 5, ..., 25, 26}.
    /// It defines the size of the internal state, which is a byte array of length 2^p.
    pub fn new(p: u32) -> Result<Self, &'static str> {
        if !(MIN_P..=MAX_P).contains(&p) {
            return Err("Invalid precision parameter");
        }
        Ok(Self {
            state: vec![0; 1 << p],
        })
    }

    /// Returns the precision parameter of this sketch.
    pub fn get_p(&self) -> u32 {
        31 - (self.state.len() as u32).leading_zeros()
    }

    /// Returns true if the sketch is empty (initial state).
    pub fn is_empty(&self) -> bool {
        self.state.iter().all(|&x| x == 0)
    }

    /// Resets this sketch to its initial state representing an empty set.
    pub fn reset(&mut self) {
        self.state.fill(0);
    }

    fn unpack(register: u8) -> u64 {
        // Java: (4L | (register & 3)) << ((register >>> 2) - 2)
        // BUT: Java shifts are mod 64. Mirror that explicitly.
        if register == 0 {
            return 0;
        }
        let sh = ((((register as u32) >> 2).wrapping_sub(2)) & 63) as u32;
        (4u64 | ((register & 3) as u64)).wrapping_shl(sh)
    }

    fn pack(hash_prefix: u64) -> u8 {
        // Java: nlz = Long.numberOfLeadingZeros(hashPrefix) + 1
        //        return (byte)((-nlz << 2) | ((hashPrefix << nlz) >>> 62));
        // Again, enforce mod-64 on the left shift to match Java semantics.
        debug_assert!(hash_prefix != 0, "pack() must be called with nonzero hash_prefix");
        let nlz = hash_prefix.leading_zeros() + 1; // 1..=64
        let s = (nlz & 63) as u32;
        let y = (hash_prefix.wrapping_shl(s) >> 62) as u8; // top 2 bits after masked shift
        (((-(nlz as i32)) << 2) | (y as i32)) as u8
    }

    /// Get the scaled register change probability
    fn get_scaled_register_change_probability(reg: u8, p: u32) -> u64 {
        if reg == 0 {
            return 1u64 << (64 - p);
        }
        let k = 1i32.wrapping_sub(p as i32) + ((reg >> 2) as i32);

        // Java’s `<< ~k`  ==>  shift by ((!k as u32) & 63)
        let shift = ((!k) as u32) & 63;
        let head = (((reg & 2) | ((reg & 1) << 2)) ^ 7u8) as u64;

        (head << shift) >> p
    }

    pub fn add_value_with_build_hasher<T, S>(&mut self, value: T, build: &S) -> &mut Self
    where
        T: Hash,
        S: BuildHasher + ?Sized,
    {
        // one fresh hasher per call
        let mut h = build.build_hasher();
        value.hash(&mut h);
        // 64‑bit digest
        let hash = h.finish();
        self.add(hash)
    }

    /// If `build_hasher` is `None` the method falls back to xxh3‑64 hash function.
    pub fn add_value_with<T, S>(&mut self, value: T, build_hasher: Option<&S>) -> &mut Self
    where
        T: Hash,
        S: BuildHasher + ?Sized,
    {
        match build_hasher {
            Some(b) => self.add_value_with_build_hasher(value, b),
            None => self.add_value(value),
        }
    }

    pub fn add_value<T: Hash>(&mut self, value: T) -> &mut Self {
        // 64‑bit xxh3 hash
        let mut h = Xxh3::default();
        value.hash(&mut h);
        let hash = h.finish();
        self.add(hash)
    }
    /// Same as [`Self::add_value`] but notifies a `StateChangeObserver`.
    pub fn add_value_with_observer<T, O>(
        &mut self,
        value: T,
        mut observer: Option<&mut O>,
    ) -> &mut Self
    where
        T: Hash,
        O: StateChangeObserver,
    {
        let mut h = Xxh3::default();
        value.hash(&mut h);
        let hash = h.finish();
        self.add_with_observer(hash, observer.take())
    }

    /// Adds a new element represented by a 64-bit hash value to this sketch.
    ///
    /// In order to get good estimates, it is important that the hash value is calculated using a
    /// high-quality hash algorithm.
    pub fn add(&mut self, hash_value: u64) -> &mut Self {
        struct NoopObserver;
        impl StateChangeObserver for NoopObserver {
            fn state_changed(&mut self, _: f64) {}
        }
        self.add_with_observer::<NoopObserver>(hash_value, None)
    }

    /// Adds a new element with a state change observer
    pub fn add_with_observer<T: StateChangeObserver>(
        &mut self,
        hash_value: u64,
        mut observer: Option<&mut T>,
    ) -> &mut Self {
        // q = Long.numberOfLeadingZeros(state.length - 1L)  (i.e., 64 - p)
        let q: u32 = (self.state.len() as u64 - 1).leading_zeros();
        let p: u32 = 64 - q;

        let idx = (hash_value >> q) as usize;

        // nlz = Long.numberOfLeadingZeros(~(~hash << p)) with p masked like Java
        let nlz: u32 = (!((!hash_value).wrapping_shl(p & 63))).leading_zeros();

        let old = self.state[idx];
        let mut hp = Self::unpack(old);

        // Java: 1L << (nlz + ~q) ; since shifts are mod 64, (nlz + ~q) ≡ (nlz + p - 1) mod 64
        let bitpos = ((nlz + p - 1) & 63) as u32;
        hp |= 1u64.wrapping_shl(bitpos);

        let new = Self::pack(hp);

        if let Some(obs) = observer.as_deref_mut() {
            if new != old {
                let pu = p as u32;
                let dec = (Self::get_scaled_register_change_probability(old, pu)
                    .wrapping_sub(Self::get_scaled_register_change_probability(new, pu))) as f64
                    * 2f64.powi(-64);
                obs.state_changed(dec);
            }
        }

        self.state[idx] = new;
        self
    }

    /// Computes a token from a given 64-bit hash value.
    pub fn compute_token(hash_value: u64) -> u32 {
        let idx = (hash_value >> 38) as u32; // top 26 bits
        let nlz = (!((!hash_value) << 26)).leading_zeros(); // in 0..=38
        (idx << 6) | (nlz & 0x3f)
    }

    /// Matches DistinctCountUtil.reconstructHash
    pub fn reconstruct_hash_from_token(token: u32) -> u64 {
        let idx = (token & 0xFFFF_FFC0) as u64; // bits 31..6
        let nlz = (token & 0x3f) as u32; // bits 5..0
                                         // 38 ones shifted logically by nlz (use only nlz for the >>> on a 64-bit value)
        let low = (0x3FFF_FFFF_FFu64) >> nlz; // 0x3FFFFFFFFF = (1<<38) - 1      // 38 ones
        (idx << 32) | low
    }

    pub fn add_token(&mut self, token: u32) -> &mut Self {
        let hash = Self::reconstruct_hash_from_token(token);
        self.add(hash)
    }

    pub fn add_token_with_observer<T: StateChangeObserver>(
        &mut self,
        token: u32,
        observer: Option<&mut T>,
    ) -> &mut Self {
        let hash = Self::reconstruct_hash_from_token(token);
        self.add_with_observer(hash, observer)
    }

    /// Returns an estimate of the number of distinct elements added to this sketch.
    pub fn get_distinct_count_estimate(&self) -> f64 {
        OptimalFGRAEstimator.estimate(self)
    }

    /// Creates a copy of this sketch.
    pub fn copy(&self) -> Self {
        Self {
            state: self.state.clone(),
        }
    }

    /// Returns a downsized copy of this sketch with a precision that is not larger than the given
    /// precision parameter.
    pub fn downsize(&self, p: u32) -> Result<Self, &'static str> {
        if !(MIN_P..=MAX_P).contains(&p) {
            return Err("Invalid precision parameter");
        }
        if (1 << p) >= self.state.len() {
            Ok(self.copy())
        } else {
            let mut downsized = Self::new(p)?;
            downsized.add_sketch(self)?;
            Ok(downsized)
        }
    }

    /// Adds another sketch.
    ///
    /// The precision parameter of the added sketch must not be smaller than the precision parameter
    /// of this sketch. Otherwise, an error will be returned.
    pub fn add_sketch(&mut self, other: &UltraLogLog) -> Result<&mut Self, &'static str> {
        let other_data = &other.state;
        match other_data.len().cmp(&self.state.len()) {
            std::cmp::Ordering::Less => {
                return Err("other has smaller precision");
            }
            std::cmp::Ordering::Equal => {
                for i in 0..self.state.len() {
                    let other_r = other_data[i];
                    if other_r != 0 {
                        self.state[i] =
                            Self::pack(Self::unpack(self.state[i]) | Self::unpack(other_r));
                    }
                }
            }
            std::cmp::Ordering::Greater => {
                let p = self.get_p();
                let other_p = other.get_p();
                let other_p_minus_one = other_p - 1;
                let k_upper_bound = 1u64 << (other_p - p);
                let mut j = 0;
                for i in 0..self.state.len() {
                    let mut hash_prefix = Self::unpack(self.state[i]) | Self::unpack(other_data[j]);
                    j += 1;
                    for k in 1..k_upper_bound {
                        if other_data[j] != 0 {
                            let shift =
                                ((k.leading_zeros() as i32 + other_p_minus_one as i32) as u32) & 63;
                            hash_prefix |= 1u64.wrapping_shl(shift);
                        }
                        j += 1;
                    }
                    if hash_prefix != 0 {
                        self.state[i] = Self::pack(hash_prefix);
                    }
                }
            }
        }
        Ok(self)
    }

    /// Merges two UltraLogLog sketches into a new sketch.
    ///
    /// The precision of the merged sketch is given by the smaller precision of both sketches.
    pub fn merge(sketch1: &UltraLogLog, sketch2: &UltraLogLog) -> Result<Self, &'static str> {
        if sketch1.state.len() <= sketch2.state.len() {
            let mut result = sketch1.copy();
            result.add_sketch(sketch2)?;
            Ok(result)
        } else {
            let mut result = sketch2.copy();
            result.add_sketch(sketch1)?;
            Ok(result)
        }
    }

    /// Returns a UltraLogLog sketch whose state is kept in the given byte array.
    ///
    /// The array must have a length that is a power of two of a valid precision parameter.
    /// If the state is not valid (it was not retrieved using get_state()) the behavior will be undefined.
    pub fn wrap(state: Vec<u8>) -> Result<Self, &'static str> {
        if state.len() > MAX_STATE_SIZE
            || state.len() < MIN_STATE_SIZE
            || !state.len().is_power_of_two()
        {
            return Err("Invalid state length");
        }
        Ok(Self { state })
    }

    /// Returns a reference to the internal state of this sketch.
    pub fn get_state(&self) -> &[u8] {
        &self.state
    }

    #[cfg(feature = "serde")]
    /// Serializes UltraLogLog to a file using bincode
    /// the serde feature must be enabled
    pub fn save<W: std::io::Write>(&self, mut writer: W) -> std::io::Result<()> {
        bincode::serialize_into(&mut writer, &self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
    }

    #[cfg(feature = "serde")]
    /// Loads an UltraLogLog from a bincode file
    /// the serde feature must be enabled
    pub fn load<R: std::io::Read>(mut reader: R) -> std::io::Result<Self> {
        let sketch: UltraLogLog = bincode::deserialize_from(&mut reader)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

        if sketch.state.len() < (1 << crate::MIN_P)
            || sketch.state.len() > (1 << crate::MAX_P)
            || !sketch.state.len().is_power_of_two()
        {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Invalid UltraLogLog state length!",
            ));
        }

        Ok(sketch)
    }
}

// Trait for estimators
pub trait Estimator {
    fn estimate(&self, ull: &UltraLogLog) -> f64;
}

/// The Optimal FGRA (Further Generalized Remaining Area) Estimator
pub struct OptimalFGRAEstimator;

impl Estimator for OptimalFGRAEstimator {
    fn estimate(&self, ull: &UltraLogLog) -> f64 {
        let state = &ull.state;
        let m = state.len();
        let p = ull.get_p();
        let off = (p << 2) + 4;

        let mut sum = 0.0;
        let mut c0 = 0;
        let mut c4 = 0;
        let mut c8 = 0;
        let mut c10 = 0;
        let mut c4w0 = 0;
        let mut c4w1 = 0;
        let mut c4w2 = 0;
        let mut c4w3 = 0;

        // Process each register
        for &reg in state {
            let r = reg as i32;
            let r2 = r - off as i32;
            if r2 < 0 {
                if r2 < -8 {
                    c0 += 1;
                }
                if r2 == -8 {
                    c4 += 1;
                }
                if r2 == -4 {
                    c8 += 1;
                }
                if r2 == -2 {
                    c10 += 1;
                }
            } else if r < 252 {
                sum += REGISTER_CONTRIBUTIONS[r2 as usize];
            } else {
                match r {
                    252 => c4w0 += 1,
                    253 => c4w1 += 1,
                    254 => c4w2 += 1,
                    255 => c4w3 += 1,
                    _ => unreachable!(),
                }
            }
        }

        // Handle small range estimates if needed
        if c0 > 0 || c4 > 0 || c8 > 0 || c10 > 0 {
            let z = Self::small_range_estimate(c0, c4, c8, c10, m as i32);
            if c0 > 0 {
                sum += Self::calculate_contribution0(c0, z);
            }
            if c4 > 0 {
                sum += Self::calculate_contribution4(c4, z);
            }
            if c8 > 0 {
                sum += Self::calculate_contribution8(c8, z);
            }
            if c10 > 0 {
                sum += Self::calculate_contribution10(c10, z);
            }
        }

        // Handle large range estimates if needed
        if c4w0 > 0 || c4w1 > 0 || c4w2 > 0 || c4w3 > 0 {
            sum += Self::calculate_large_range_contribution(
                c4w0,
                c4w1,
                c4w2,
                c4w3,
                m as i32,
                65 - p as i32,
            );
        }

        // Return final estimate
        ESTIMATION_FACTORS[(p - MIN_P) as usize] * sum.powf(MINUS_INV_TAU)
    }
}

impl OptimalFGRAEstimator {
    fn small_range_estimate(c0: i32, c4: i32, c8: i32, c10: i32, m: i32) -> f64 {
        let alpha = m + 3 * (c0 + c4 + c8 + c10);
        let beta = m - c0 - c4;
        let gamma = 4 * c0 + 2 * c4 + 3 * c8 + c10;

        // Do math in f64 to avoid i32 overflow in debug builds.
        let alpha_f = alpha as f64;
        let beta_f = beta as f64;
        let gamma_f = gamma as f64;

        let quad_root_z =
            (((beta_f * beta_f) + 4.0 * (alpha_f * gamma_f)).sqrt() - beta_f) / (2.0 * alpha_f);
        let root_z = quad_root_z * quad_root_z;
        root_z * root_z
    }

    fn calculate_contribution0(c0: i32, z: f64) -> f64 {
        c0 as f64 * Self::sigma(z)
    }

    fn calculate_contribution4(c4: i32, z: f64) -> f64 {
        c4 as f64 * POW_2_MINUS_TAU_ETA_X * Self::psi_prime(z, z * z)
    }

    fn calculate_contribution8(c8: i32, z: f64) -> f64 {
        c8 as f64 * (z * POW_4_MINUS_TAU_ETA_01 + POW_4_MINUS_TAU_ETA_1)
    }

    fn calculate_contribution10(c10: i32, z: f64) -> f64 {
        c10 as f64 * (z * POW_4_MINUS_TAU_ETA_23 + POW_4_MINUS_TAU_ETA_3)
    }

    fn psi_prime(z: f64, z_square: f64) -> f64 {
        (z + ETA23X) * (z_square + ETA13X) + ETA3012XX
    }

    fn sigma(z: f64) -> f64 {
        if z <= 0.0 {
            return ETA_3;
        }
        if z >= 1.0 {
            return f64::INFINITY;
        }

        let mut pow_z = z;
        let mut next_pow_z = pow_z * pow_z;
        let mut s = 0.0;
        let mut pow_tau = ETA_X;

        loop {
            let old_s = s;
            let next_next_pow_z = next_pow_z * next_pow_z;
            s += pow_tau * (pow_z - next_pow_z) * Self::psi_prime(next_pow_z, next_next_pow_z);
            if !(s > old_s) {
                return s / z;
            }
            pow_z = next_pow_z;
            next_pow_z = next_next_pow_z;
            pow_tau *= POW_2_TAU;
        }
    }

    fn calculate_large_range_contribution(
        c4w0: i32,
        c4w1: i32,
        c4w2: i32,
        c4w3: i32,
        m: i32,
        w: i32,
    ) -> f64 {
        let z = Self::large_range_estimate(c4w0, c4w1, c4w2, c4w3, m);
        let root_z = z.sqrt();
        let mut s = Self::phi(root_z, z) * (c4w0 + c4w1 + c4w2 + c4w3) as f64;
        s += z
            * (1.0 + root_z)
            * (c4w0 as f64 * ETA_0
                + c4w1 as f64 * ETA_1
                + c4w2 as f64 * ETA_2
                + c4w3 as f64 * ETA_3);
        s += root_z
            * ((c4w0 + c4w1) as f64
                * (z * POW_2_MINUS_TAU * (ETA_0 - ETA_2) + POW_2_MINUS_TAU * ETA_2)
                + (c4w2 + c4w3) as f64
                    * (z * POW_2_MINUS_TAU * (ETA_1 - ETA_3) + POW_2_MINUS_TAU * ETA_3));
        s * POW_2_MINUS_TAU.powi(w) / ((1.0 + root_z) * (1.0 + z))
    }

    fn large_range_estimate(c4w0: i32, c4w1: i32, c4w2: i32, c4w3: i32, m: i32) -> f64 {
        let alpha = m + 3 * (c4w0 + c4w1 + c4w2 + c4w3);
        let beta = c4w0 + c4w1 + 2 * (c4w2 + c4w3);
        let gamma = m + 2 * c4w0 + c4w2 - c4w3;

        // Same widening to f64 here.
        let alpha_f = alpha as f64;
        let beta_f = beta as f64;
        let gamma_f = gamma as f64;

        let inner =
            (((beta_f * beta_f) + 4.0 * (alpha_f * gamma_f)).sqrt() - beta_f) / (2.0 * alpha_f);
        inner.sqrt()
    }

    fn phi(z: f64, z_square: f64) -> f64 {
        if z <= 0.0 {
            return 0.0;
        }
        if z >= 1.0 {
            return PHI_1;
        }

        let mut previous_pow_z = z_square;
        let mut pow_z = z;
        let mut next_pow_z = z.sqrt();
        let mut p = P_INITIAL / (1.0 + next_pow_z);
        let mut ps = Self::psi_prime(pow_z, previous_pow_z);
        let mut s = next_pow_z * (ps + ps) * p;

        loop {
            previous_pow_z = pow_z;
            pow_z = next_pow_z;
            let old_s = s;
            next_pow_z = pow_z.sqrt();
            let next_ps = Self::psi_prime(pow_z, previous_pow_z);
            p *= POW_2_MINUS_TAU / (1.0 + next_pow_z);
            s += next_pow_z * ((next_ps + next_ps) - (pow_z + next_pow_z) * ps) * p;
            if !(s > old_s) {
                return s;
            }
            ps = next_ps;
        }
    }
}

// Pre-computed estimation factors and register contributions
const ESTIMATION_FACTORS: [f64; 24] = [
    94.59941722950778,
    455.6358404615186,
    2159.476860400962,
    10149.51036338182,
    47499.52712820488,
    221818.76564766388,
    1034754.6840013304,
    4824374.384717942,
    2.2486750611989766e7,
    1.0479810199493326e8,
    4.8837185623048025e8,
    2.275794725435168e9,
    1.0604938814719946e10,
    4.9417362104242645e10,
    2.30276227770117e11,
    1.0730444972228585e12,
    5.0001829613164e12,
    2.329988778511272e13,
    1.0857295240912981e14,
    5.059288069986326e14,
    2.3575295235667005e15,
    1.0985627213141412e16,
    5.119087674515589e16,
    2.3853948339571715e17,
];

const REGISTER_CONTRIBUTIONS: [f64; 236] = [
    0.8484061093359406,
    0.38895829052007685,
    0.5059986252327467,
    0.17873835725405993,
    0.48074234060273024,
    0.22040001471443574,
    0.2867199572932749,
    0.10128061935935387,
    0.2724086914332655,
    0.12488785473931466,
    0.16246750447680292,
    0.057389829555353204,
    0.15435814343988866,
    0.0707666752272979,
    0.09206087452057209,
    0.03251947467566813,
    0.08746577181824695,
    0.0400993542020493,
    0.05216553700867983,
    0.018426892732996067,
    0.04956175987398336,
    0.022721969094305374,
    0.029559172293066274,
    0.01044144713836362,
    0.02808376340530896,
    0.012875216815740723,
    0.01674946174724118,
    0.005916560101748389,
    0.015913433441643893,
    0.0072956356627506685,
    0.009490944673308844,
    0.0033525700962450116,
    0.009017216113341773,
    0.004134011914931561,
    0.0053779657012946284,
    0.0018997062578498703,
    0.005109531310944485,
    0.002342503834183061,
    0.00304738001114257,
    0.001076452918957914,
    0.0028952738727082267,
    0.0013273605219527246,
    0.0017267728074345586,
    6.09963188753462E-4,
    0.0016405831157217021,
    7.521379173550258E-4,
    9.78461602292084E-4,
    3.4563062172237723E-4,
    9.2962292270938E-4,
    4.2619276177576713E-4,
    5.544372155028133E-4,
    1.958487477192352E-4,
    5.267631795945699E-4,
    2.4149862146135835E-4,
    3.141672858847145E-4,
    1.1097608132071735E-4,
    2.9848602115777116E-4,
    1.3684320663902123E-4,
    1.7802030736817869E-4,
    6.288368329501905E-5,
    1.6913464774658265E-4,
    7.754107700464113E-5,
    1.0087374230011362E-4,
    3.563252169014952E-5,
    9.583875639268212E-5,
    4.393801322487549E-5,
    5.715927601779108E-5,
    2.0190875207520577E-5,
    5.430624268457414E-5,
    2.4897113642537945E-5,
    3.2388833410757184E-5,
    1.144099329232623E-5,
    3.0772185549154786E-5,
    1.4107744575453657E-5,
    1.8352865935237916E-5,
    6.482944704957522E-6,
    1.7436805727319977E-5,
    7.99403737572986E-6,
    1.0399500462555932E-5,
    3.67350727106242E-6,
    9.880422483694849E-6,
    4.529755498675165E-6,
    5.892791363067244E-6,
    2.081562667074589E-6,
    5.5986600976661345E-6,
    2.5667486794686803E-6,
    3.339101736056405E-6,
    1.1795003568090263E-6,
    3.1724346748254955E-6,
    1.4544270182973653E-6,
    1.8920745223756656E-6,
    6.683541714686068E-7,
    1.7976340035771381E-6,
    8.241391019206623E-7,
    1.072128458850476E-6,
    3.7871739159788393E-7,
    1.0186145159929963E-6,
    4.6699164053601817E-7,
    6.075127690181302E-7,
    2.1459709360913574E-7,
    5.77189533646426E-7,
    2.6461697039041317E-7,
    3.442421115430427E-7,
    1.2159967724530947E-7,
    3.27059699739513E-7,
    1.4994302882644454E-7,
    1.9506195985170504E-7,
    6.890345650764188E-8,
    1.853256875916027E-7,
    8.49639834530526E-8,
    1.1053025444979778E-7,
    3.904357664636507E-8,
    1.0501327589016596E-7,
    4.814414208323267E-8,
    6.263105916717392E-8,
    2.2123721430020238E-8,
    5.9504908663745294E-8,
    2.7280481949286693E-8,
    3.548937430686624E-8,
    1.2536224699555158E-8,
    3.371796684815404E-8,
    1.545826061452554E-8,
    2.0109761920695445E-8,
    7.103548569567803E-9,
    1.910600846054063E-8,
    8.759296176321385E-9,
    1.139503111580109E-8,
    4.0251673442004705E-9,
    1.082626247715867E-8,
    4.963383100969499E-9,
    6.456900615837058E-9,
    2.28082795382416E-9,
    6.134612546958812E-9,
    2.812460192131048E-9,
    3.65874960227048E-9,
    1.292412391857717E-9,
    3.476127720042246E-9,
    1.5936574250689536E-9,
    2.0732003554895977E-9,
    7.323348470132607E-10,
    1.9697191686598677E-9,
    9.030328662369446E-10,
    1.1747619217600795E-9,
    4.1497151491950363E-10,
    1.1161251587553774E-9,
    5.116961428952198E-10,
    6.656691762391315E-10,
    2.351401942661752E-10,
    6.324431369849931E-10,
    2.899484087937328E-10,
    3.771959611450379E-10,
    1.3324025619025952E-10,
    3.5836869940773545E-10,
    1.6429687995368037E-10,
    2.1373498756237659E-10,
    7.549949478033437E-11,
    2.0306667462222755E-10,
    9.309747508122088E-11,
    1.2111117194789844E-10,
    4.2781167456975155E-11,
    1.1506606020637118E-10,
    5.275291818652914E-11,
    6.86266490006118E-11,
    2.424159650745726E-11,
    6.520123617549523E-11,
    2.9892007004129765E-11,
    3.888672595026375E-11,
    1.3736301184893309E-11,
    3.6945743959497274E-11,
    1.693805979747882E-11,
    2.2034843273746723E-11,
    7.783562034953282E-12,
    2.093500180037604E-11,
    9.597812206565218E-12,
    1.248586262365167E-11,
    4.4104913787558985E-12,
    1.186264650299681E-11,
    5.4385213096368525E-12,
    7.075011313669894E-12,
    2.499168647301308E-12,
    6.721871027139603E-12,
    3.081693348317683E-12,
    4.008996942969544E-12,
    1.4161333491633975E-12,
    3.808892905481426E-12,
    1.7462161775917615E-12,
    2.271665129027518E-12,
    8.024403094117999E-13,
    2.1582778227746425E-12,
    9.89479027998621E-13,
    1.2872203525845489E-12,
    4.54696198313039E-13,
    1.2229703685228866E-12,
    5.606801491206791E-13,
    7.293928206826874E-13,
    2.5764985922987735E-13,
    6.92986095905959E-13,
    3.1770479284824887E-13,
    4.1330443990824427E-13,
    1.4599517261737423E-13,
    3.926748688923721E-13,
    1.8002480658009348E-13,
    2.3419555992885186E-13,
    8.272696321778206E-14,
    2.225059832666067E-13,
    1.0200957528418621E-13,
    1.327049869160979E-13,
    4.687655297461429E-14,
    1.2608118449008524E-13,
    5.780288643182276E-14,
    7.519618885068399E-14,
    2.656221301145837E-14,
    7.144286571105751E-14,
    3.2753529955811655E-14,
    4.2609301647742677E-14,
    1.5051259431302017E-14,
    4.0482511975524363E-14,
    1.8559518231526075E-14,
    2.4144210160882415E-14,
    8.528672304925501E-15,
    2.293908229376684E-14,
    1.0516598285774437E-14,
    1.3681118012966618E-14,
    4.832701981970378E-15,
    1.2998242223663023E-14,
    5.959143881034847E-15,
    7.752292944665042E-15,
    2.7384108113817744E-15,
    7.365346997814574E-15,
    3.376699844369893E-15,
    4.392773006047039E-15,
    1.5516979527951759E-15,
    4.173513269314059E-15,
    1.9133791810691354E-15,
    2.4891286772044455E-15,
    8.792568765435867E-16,
];

/// The Maximum Likelihood Estimator
pub struct MaximumLikelihoodEstimator;

impl Estimator for MaximumLikelihoodEstimator {
    fn estimate(&self, ull: &UltraLogLog) -> f64 {
        let state = &ull.state;
        let p_i32 = ull.get_p() as i32;
        let m = state.len();

        // Use unsigned arithmetic and logical shifts to mirror Java's `>>>`.
        let mut sum: u64 = 0;
        let mut b = vec![0i32; 64];

        for &r_u8 in state {
            let r = r_u8 as i32; // Java uses Byte.toUnsignedInt
            let r2 = r - ((p_i32 << 2) + 4);

            if r2 < 0 {
                // Small-range bucket
                let mut ret_u: u64 = 4;
                if r2 == -2 || r2 == -8 {
                    b[0] += 1;
                    ret_u = ret_u.wrapping_sub(2);
                }
                if r2 == -2 || r2 == -4 {
                    b[1] += 1;
                    ret_u = ret_u.wrapping_sub(1);
                }
                // ret << (62 - p)
                let sh = (62 - p_i32) as u32;
                sum = sum.wrapping_add(ret_u << sh);
            } else {
                // Large-range bucket
                let k = r2 >> 2;
                let y0 = (r & 1) as u64;
                let y1 = ((r >> 1) & 1) as u64;

                // 0xE000_0000_0000_0000L minus top-bit masks; Java then uses `>>>`
                let mut ret_u: u64 = 0xE000_0000_0000_0000u64;
                ret_u = ret_u.wrapping_sub(y0 << 63);
                ret_u = ret_u.wrapping_sub(y1 << 62);

                // shift count is masked in Java: (k + p) & 63
                let sh: u32 = ((k + p_i32) & 63) as u32;
                sum = sum.wrapping_add(ret_u >> sh);

                // tallies for the Newton solver
                b[k as usize] += y0 as i32;
                b[(k + 1) as usize] += y1 as i32;
                b[(k + 2) as usize] += 1;
            }
        }

        if sum == 0 {
            return if state.get(0).copied().unwrap_or(0) == 0 {
                0.0
            } else {
                f64::INFINITY
            };
        }

        b[(63 - p_i32) as usize] += b[(64 - p_i32) as usize];

        let factor = (m << 1) as f64;
        let a = (sum as f64) * factor * 2f64.powi(-64);

        factor
            * solve_maximum_likelihood_equation(
                a,
                &b,
                63 - p_i32,
                ML_EQUATION_SOLVER_EPS / (m as f64).sqrt(),
            )
            / (1.0 + ML_BIAS_CORRECTION_CONSTANT / m as f64)
    }
}


// Helper function for MaximumLikelihoodEstimator
fn solve_maximum_likelihood_equation(a: f64, b: &[i32], n: i32, relative_error_limit: f64) -> f64 {
    // Java: if (a == 0.) return +INF;
    if a == 0.0 {
        return f64::INFINITY;
    }

    // Find kMax = largest index <= n with b[k] > 0
    let mut k_max: i32 = n;
    while k_max >= 0 && b[k_max as usize] == 0 {
        k_max -= 1;
    }
    // If all b are zero -> return 0
    if k_max < 0 {
        return 0.0;
    }

    // Compute kMin, s1 = Σ b[k], s2 = Σ b[k] * 2^k  (over nonzero entries)
    let mut k_min: i32 = k_max;
    let mut s1: i64 = b[k_max as usize] as i64;
    let mut s2: f64 = (b[k_max as usize] as f64) * (2f64).powi(k_max);

    for k in (0..k_max as usize).rev() {
        let t = b[k];
        if t > 0 {
            s1 += t as i64;
            s2 += (t as f64) * (2f64).powi(k as i32);
            k_min = k as i32;
        }
    }

    let s1_f = s1 as f64;

    // Initial x (exactly as in Java)
    let mut x = if s2 <= 1.5 * a {
        s1_f / (0.5 * s2 + a)
    } else {
        // ln1p(s2/a) * (s1/s2)
        (1.0 + s2 / a).ln() * (s1_f / s2)
    };

    let mut delta_x = x;
    let mut g_prev = 0.0;

    // Iterate until relative change is small
    while delta_x > x * relative_error_limit {
        // kappa = exponent(x) - 1021 (bit-exact port)
        let raw_x = x.to_bits();
        let exp_bits = ((raw_x & 0x7FF0_0000_0000_0000u64) >> 52) as i32;
        let kappa = exp_bits - 1021;

        // xPrime = x / 2^(max(kMax, kappa)+1)  ∈ [0, 0.25]
        let max_k = if k_max > kappa { k_max } else { kappa };
        let xprime_bits = raw_x.wrapping_sub(((max_k + 1) as u64) << 52);
        let mut x_prime = f64::from_bits(xprime_bits);

        // h(xPrime) ≈ xPrime + xPrime^2 * (C0 + xPrime^2 * (C1 - xPrime^2 * C2))
        let x_prime2 = x_prime * x_prime;
        let mut h = x_prime + x_prime2 * (C0 + x_prime2 * (C1 - x_prime2 * C2));

        // First loop: for k = kappa-1 down to kMax
        //   h = (xPrime + h*(1-h)) / (xPrime + (1-h)); xPrime *= 2
        if kappa - 1 >= k_max {
            let mut k = kappa - 1;
            loop {
                let h_prime = 1.0 - h;
                h = (x_prime + h * h_prime) / (x_prime + h_prime);
                x_prime += x_prime; // *= 2
                if k == k_max { break; }
                k -= 1;
            }
        }

        // g accumulation
        let mut g = (b[k_max as usize] as f64) * h;

        // Second loop: for k = kMax-1 down to kMin
        if k_max - 1 >= k_min {
            let mut k = k_max - 1;
            loop {
                let h_prime = 1.0 - h;
                h = (x_prime + h * h_prime) / (x_prime + h_prime);
                x_prime += x_prime; // *= 2
                g += (b[k as usize] as f64) * h;
                if k == k_min { break; }
                k -= 1;
            }
        }

        // g += x * a
        g += x * a;

        // Step update (exactly as Java)
        if g_prev < g && g <= s1_f {
            delta_x *= (g - s1_f) / (g_prev - g);
        } else {
            delta_x = 0.0;
        }
        x += delta_x;
        g_prev = g;
    }

    x
}

#[cfg(test)]
mod tests {
    use super::*;
    use ahash::RandomState;

    #[test]
    fn test_create_ultraloglog() {
        assert!(UltraLogLog::new(2).is_err()); // Too small p
        assert!(UltraLogLog::new(27).is_err()); // Too large p
        assert!(UltraLogLog::new(4).is_ok()); // Valid p
    }

    #[test]
    fn test_empty_state() {
        let ull = UltraLogLog::new(4).unwrap();
        assert!(ull.is_empty());
    }

    #[test]
    fn test_add_and_reset() {
        let mut ull = UltraLogLog::new(4).unwrap();
        assert!(ull.is_empty());

        ull.add(123456789);
        assert!(!ull.is_empty());

        ull.reset();
        assert!(ull.is_empty());
    }

    #[test]
    fn test_estimation() {
        let mut ull = UltraLogLog::new(4).unwrap();
        assert_eq!(ull.get_distinct_count_estimate(), 0.0);

        ull.add(123456789);
        assert!(ull.get_distinct_count_estimate() > 0.0);
    }

    #[test]
    fn test_wrap_and_get_state() {
        // Test valid state
        let valid_state = vec![0; 16]; // 2^4 = 16
        let ull = UltraLogLog::wrap(valid_state).unwrap();
        assert_eq!(ull.get_state().len(), 16);
        assert!(ull.is_empty());

        // Test invalid state lengths
        assert!(UltraLogLog::wrap(vec![0; 7]).is_err()); // Not power of 2
        assert!(UltraLogLog::wrap(vec![0; 2]).is_err()); // Too small
        assert!(UltraLogLog::wrap(vec![0; MAX_STATE_SIZE + 1]).is_err()); // Too large

        // Test state preservation
        let mut original = UltraLogLog::new(4).unwrap();
        original.add(12345);
        let state = original.get_state().to_vec();
        let wrapped = UltraLogLog::wrap(state).unwrap();
        assert_eq!(
            original.get_distinct_count_estimate(),
            wrapped.get_distinct_count_estimate()
        );
    }
    #[cfg(feature = "serde")]
    #[test]
    fn test_save_and_load() {
        use std::fs::{remove_file, File};
        use std::io::{BufReader, BufWriter};

        let file_path = "test_ultraloglog.bin";

        // Create UltraLogLog and add data
        let mut ull = UltraLogLog::new(5).expect("Failed to create ULL");
        ull.add(123456789);
        ull.add(987654321);
        let original_estimate = ull.get_distinct_count_estimate();
        assert!(original_estimate > 0.0);

        // Save to file using writer
        let file = File::create(file_path).expect("Failed to create file");
        let writer = BufWriter::new(file);
        ull.save(writer).expect("Failed to save UltraLogLog");

        // Load from file using reader
        let file = File::open(file_path).expect("Failed to open file");
        let reader = BufReader::new(file);
        let loaded_ull = UltraLogLog::load(reader).expect("Failed to load UltraLogLog");

        let loaded_estimate = loaded_ull.get_distinct_count_estimate();
        assert!(
            (loaded_estimate - original_estimate).abs() < f64::EPSILON,
            "Loaded estimate ({}) differs from original ({})",
            loaded_estimate,
            original_estimate
        );

        // Cleanup
        remove_file(file_path).ok();
    }
    #[cfg(feature = "serde")]
    #[test]
    fn test_two_sketch_in_one_file() {
        use std::fs::{remove_file, File};
        use std::io::Write;
        use std::io::{BufReader, BufWriter};
        const P: u32 = 8;
        // build set 1
        let mut s1 = UltraLogLog::new(P).expect("alloc sketch");
        for word in ["apple", "banana", "cherry", "dragonfruit"] {
            s1.add_value(word);
        }
        // build set 2
        let mut s2 = UltraLogLog::new(P).expect("alloc sketch");
        for word in ["alpha", "beta", "delta"] {
            s2.add_value(word);
        }

        // stream-save both sketches into ONE file
        let bin_path = "sets_stream.bin";
        {
            let f = File::create(bin_path).expect("create file");
            let mut w = BufWriter::new(f);
            s1.save(&mut w).expect("save s1");
            s2.save(&mut w).expect("save s2");
            w.flush().unwrap();
        }

        // reopen & stream-load them back
        let f = File::open(bin_path).expect("open file");
        let mut r = BufReader::new(f);

        let s1_loaded = UltraLogLog::load(&mut r).expect("load first sketch");
        let s2_loaded = UltraLogLog::load(&mut r).expect("load second sketch");

        // verify order + estimates
        let est1 = s1_loaded.get_distinct_count_estimate();
        let est2 = s2_loaded.get_distinct_count_estimate();

        println!("Set 1 (fruit)  estimate ≈ {:.3}", est1);
        println!("Set 2 (alpha…) estimate ≈ {:.3}", est2);

        assert!(
            (est1 - 4.0).abs() < 0.1,
            "set 1 estimate {est1} not close to 4"
        );
        assert!(
            (est2 - 3.0).abs() < 0.1,
            "set 2 estimate {est2} not close to 3"
        );

        remove_file(bin_path).ok();
    }

    #[test]
    fn test_xxhash3() {
        let mut ull = UltraLogLog::new(8).unwrap();

        ull.add_value("apple")
            .add_value("banana")
            .add_value("cherry")
            .add_value("033");

        let est = ull.get_distinct_count_estimate();
        assert!(
            (est - 4.0).abs() < 0.1,
            "estimate {:.3} deviates from true count 3",
            est
        );
    }
    #[test]
    fn test_custom_ahash_hasher() {
        let ahash = RandomState::with_seeds(1, 2, 3, 4);

        let mut ull = UltraLogLog::new(8).unwrap();

        ull.add_value_with_build_hasher("apple", &ahash)
            .add_value_with_build_hasher("banana", &ahash)
            .add_value_with_build_hasher("cherry", &ahash);

        let est = ull.get_distinct_count_estimate();
        assert!(
            (est - 3.0).abs() < 0.1,
            "estimate {:.3} deviates from true count 3",
            est
        );
    }
    #[test]
    fn test_custom_komihash_hasher() {
        use std::hash::BuildHasher;
        // Komihash v5 one-shot hasher
        use komihash::v5::KomiHasher;

        /// Simple BuildHasher that spawns a fresh Komihash with a fixed seed
        #[derive(Clone)]
        struct KomiBuildHasher {
            seed: u64,
        }
        impl BuildHasher for KomiBuildHasher {
            type Hasher = KomiHasher;
            #[inline]
            fn build_hasher(&self) -> Self::Hasher {
                KomiHasher::new(self.seed)
            }
        }

        // use deterministic seed
        let komi_builder = KomiBuildHasher {
            seed: 0x1234_5678_9abc_def0,
        };

        let mut ull = UltraLogLog::new(8).unwrap();

        // feed four distinct strings through the Komihash builder
        ull.add_value_with_build_hasher("apple", &komi_builder)
            .add_value_with_build_hasher("banana", &komi_builder)
            .add_value_with_build_hasher("cherry", &komi_builder)
            .add_value_with_build_hasher("dragonfruit", &komi_builder);

        let est = ull.get_distinct_count_estimate();
        assert!(
            (est - 4.0).abs() < 0.1, // tolerate ≈5 % relative error at p = 6
            "estimate {:.3} deviates from true count 4",
            est
        );
    }
    #[test]
    fn test_custom_hasher_polymurhash() {
        use polymur_hash::PolymurHasher;
        use std::hash::BuildHasherDefault;

        // BuildHasherDefault turns our one-shot PolymurHasher into a BuildHasher
        type PolymurBuild = BuildHasherDefault<PolymurHasher>;
        let polymur_builder = PolymurBuild::default();

        let mut ull = UltraLogLog::new(8).unwrap();

        ull.add_value_with_build_hasher("apple", &polymur_builder)
            .add_value_with_build_hasher("banana", &polymur_builder)
            .add_value_with_build_hasher("cherry", &polymur_builder)
            .add_value_with_build_hasher("dragonfruit", &polymur_builder);

        let est = ull.get_distinct_count_estimate();
        assert!(
            (est - 4.0).abs() < 0.1,
            "estimate {:.3} deviates from true count 4",
            est
        );
    }
    #[test]
    fn test_custom_hasher_wyhash() {
        // The default builder already gives us seed = 0 and the stock 256-bit secret.
        use wyhash::WyHasherBuilder;

        let wyhash_builder = WyHasherBuilder::default();

        let mut ull = UltraLogLog::new(8).unwrap();

        ull.add_value_with_build_hasher("apple", &wyhash_builder)
            .add_value_with_build_hasher("banana", &wyhash_builder)
            .add_value_with_build_hasher("cherry", &wyhash_builder)
            .add_value_with_build_hasher("dragonfruit", &wyhash_builder);

        let est = ull.get_distinct_count_estimate();
        assert!(
            (est - 4.0).abs() < 0.1,
            "wyhash estimate {:.3} deviates too much from true count 4",
            est
        );
    }
    #[test]
    fn test_custom_hasher_t1ha2_atonce() {
        use std::hash::{BuildHasher, Hasher};
        use t1ha;
        use t1ha::t1ha2_atonce;
        // ── Hasher wrapper ────────────────────────────────────────────────────────
        #[derive(Default)]
        struct T1ha2AtonceHasher {
            seed: u64,
            buf: Vec<u8>,
        }

        impl Hasher for T1ha2AtonceHasher {
            fn write(&mut self, bytes: &[u8]) {
                self.buf.extend_from_slice(bytes);
            }
            fn finish(&self) -> u64 {
                t1ha2_atonce(&self.buf, self.seed)
            }
        }

        #[derive(Clone, Default)]
        struct T1ha2AtonceBuild;
        impl BuildHasher for T1ha2AtonceBuild {
            type Hasher = T1ha2AtonceHasher;
            fn build_hasher(&self) -> Self::Hasher {
                T1ha2AtonceHasher::default()
            }
        }

        let builder = T1ha2AtonceBuild::default();
        let mut ull = UltraLogLog::new(8).unwrap();

        ull.add_value_with_build_hasher("alpha", &builder)
            .add_value_with_build_hasher("beta", &builder)
            .add_value_with_build_hasher("gamma", &builder)
            .add_value_with_build_hasher("delta", &builder);

        let est = ull.get_distinct_count_estimate();
        assert!(
            (est - 4.0).abs() < 0.1,
            "t1ha2-atonce estimate {:.3} deviates too much from 4",
            est
        );
    }
    // run test like this: RUSTFLAGS="-C target-cpu=native" cargo test test_custom_hasher_t1ha0_avx2 -- --nocapture
    #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature = "avx2"
    ))]
    #[test]
    fn test_custom_hasher_t1ha0_avx2() {
        use std::hash::{BuildHasher, Hasher};
        use t1ha;
        use t1ha::t1ha0_ia32aes_avx2;

        // ── Hasher wrapper ────────────────────────────────────────────────────────
        #[derive(Default)]
        struct T1ha0Avx2Hasher {
            seed: u64,
            buf: Vec<u8>,
        }

        impl Hasher for T1ha0Avx2Hasher {
            fn write(&mut self, bytes: &[u8]) {
                self.buf.extend_from_slice(bytes);
            }
            fn finish(&self) -> u64 {
                t1ha0_ia32aes_avx2(&self.buf, self.seed)
            }
        }

        #[derive(Clone, Default)]
        struct T1ha0Avx2Build;
        impl BuildHasher for T1ha0Avx2Build {
            type Hasher = T1ha0Avx2Hasher;
            fn build_hasher(&self) -> Self::Hasher {
                T1ha0Avx2Hasher::default()
            }
        }

        // Sketch four keys and check the estimate
        let builder = T1ha0Avx2Build::default();
        let mut ull = UltraLogLog::new(8).unwrap();

        ull.add_value_with_build_hasher("apple", &builder)
            .add_value_with_build_hasher("banana", &builder)
            .add_value_with_build_hasher("cherry", &builder)
            .add_value_with_build_hasher("dragonfruit", &builder);

        let est = ull.get_distinct_count_estimate();
        assert!(
            (est - 4.0).abs() < 0.1,
            "t1ha0-avx2 estimate {:.3} deviates too much from 4",
            est
        );
    }
    #[test]
    fn ull_vs_hll_space() {
        use streaming_algorithms::HyperLogLog as HLL;

        const N: u64 = 100_000;
        const TRIALS: usize = 51; // odd ⇒ clean medians; bump for more stability
        const TARGET_REL_ERR: f64 = 0.015; // 1.5% absolute relative error target
        const HLL_MIN_P: u8 = 4;
        const HLL_MAX_P: u8 = 16;

        // deterministic 64-bit “hash” stream (SplitMix64)
        #[inline]
        fn splitmix64(mut x: u64) -> u64 {
            x = x.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = x;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^ (z >> 31)
        }
        fn make_hashes(seed: u64) -> Vec<u64> {
            (0..N).map(|i| splitmix64(seed.wrapping_add(i))).collect()
        }

        // single-run error helpers
        fn err_ull(p: u32, keys: &[u64]) -> f64 {
            let mut ull = crate::UltraLogLog::new(p).expect("valid ULL p");
            for &h in keys {
                ull.add(h);
            } // ULL expects 64-bit hash
            let est = ull.get_distinct_count_estimate();
            (est - N as f64).abs() / (N as f64)
        }
        fn err_hll(p: u8, keys: &[u64]) -> f64 {
            let mut hll = HLL::<u64>::with_p(p);
            for &h in keys {
                hll.push_hash64(h);
            } // avoid rehash inside HLL
            let est = hll.len();
            (est - N as f64).abs() / (N as f64)
        }

        // generate deterministic per-trial seeds
        let mut seeds = Vec::with_capacity(TRIALS);
        let mut s = 0x1234_5678_9ABC_DEF0u64;
        for _ in 0..TRIALS {
            s = s
                .wrapping_mul(0x9E3779B97F4A7C15)
                .wrapping_add(0xD1B54A32D192ED03);
            seeds.push(s);
        }

        // per-trial minimal p and Δp
        let mut deltas: Vec<i32> = Vec::with_capacity(TRIALS);
        let mut per_trial_rows: Vec<(usize, u32, u8, i32)> = Vec::with_capacity(TRIALS);

        'trial: for (ti, &seed) in seeds.iter().enumerate() {
            let keys = make_hashes(seed);

            // find minimal p for ULL
            let mut p_ull_opt = None;
            for p in crate::MIN_P..=crate::MAX_P {
                if err_ull(p, &keys) <= TARGET_REL_ERR {
                    p_ull_opt = Some(p);
                    break;
                }
            }
            let p_ull = match p_ull_opt {
                Some(p) => p,
                None => continue 'trial, // skip if not achievable (shouldn’t happen for these N/targets)
            };

            // find minimal p for HLL
            let mut p_hll_opt = None;
            for p in HLL_MIN_P..=HLL_MAX_P {
                if err_hll(p, &keys) <= TARGET_REL_ERR {
                    p_hll_opt = Some(p);
                    break;
                }
            }
            let p_hll = match p_hll_opt {
                Some(p) => p,
                None => continue 'trial,
            };

            let dp = p_hll as i32 - p_ull as i32; // Δp = HLL − ULL
            deltas.push(dp);
            per_trial_rows.push((ti, p_ull, p_hll, dp));
        }

        if deltas.is_empty() {
            panic!("no successful trials; try relaxing TARGET_REL_ERR or increasing p ranges");
        }

        // Δp histogram
        use std::collections::BTreeMap;
        let mut hist: BTreeMap<i32, usize> = BTreeMap::new();
        for &dp in &deltas {
            *hist.entry(dp).or_default() += 1;
        }

        // medians & means
        let mut d_sorted = deltas.clone();
        d_sorted.sort_unstable();
        let median_dp = d_sorted[d_sorted.len() / 2] as f64;
        let mean_dp = deltas.iter().copied().map(|k| k as f64).sum::<f64>() / deltas.len() as f64;

        // Geometric-mean ratios via Δp:
        // naive  = 2^{-Δp}, packed = (4/3) * 2^{-Δp}
        let gm_naive = 2f64.powf(-mean_dp);
        let gm_packed = (4.0 / 3.0) * gm_naive;

        // Median ratios via median Δp (handy intuition):
        let med_naive = 2f64.powf(-median_dp);
        let med_packed = (4.0 / 3.0) * med_naive;

        // Pretty print
        println!("Trials: {}", deltas.len());
        println!("N = {}", N);
        println!(
            "Target absolute relative error: {:.2}%",
            TARGET_REL_ERR * 100.0
        );
        println!();
        println!("Δp histogram (HLL − ULL):");
        for (dp, cnt) in hist {
            println!("  Δp={:>+2}: {:>2} trial(s)", dp, cnt);
        }
        println!();

        println!(
            "Geometric-mean ratios over trials:\n  \
            packed (ULL=8b, HLL=6b): {:.3}  → savings ≈ {:.1}%\n  \
            naive  (1B/reg)        : {:.3}  → savings ≈ {:.1}%",
            gm_packed,
            (1.0 - gm_packed) * 100.0,
            gm_naive,
            (1.0 - gm_naive) * 100.0
        );
        println!(
            "Median ratios (from median Δp={}):\n  \
            packed: {:.3}  → savings ≈ {:.1}%\n  \
            naive : {:.3}  → savings ≈ {:.1}%",
            median_dp as i32,
            med_packed,
            (1.0 - med_packed) * 100.0,
            med_naive,
            (1.0 - med_naive) * 100.0
        );
        assert!(
            gm_packed <= 0.80,
            "expected ≲ 0.80 (≈≥20% saving), got {:.3}",
            gm_packed
        );
    }
    #[test]
    fn mle_vs_hll_accuracy_large_n() {
        use streaming_algorithms::HyperLogLog as HLL;

        // --- config ---
        const P_ULL: u32 = 16;      // ULL precision (2^16 regs)
        const P_HLL: u8  = 16;      // HLL precision to compare
        const N: u64     = 1_000_000; // large cardinality
        const TRIALS: usize = 10;   // “10 or so” runs

        // --- accumulators ---
        let mut sum_rel_err_mle = 0.0f64;
        let mut sum_rel_err_hll = 0.0f64;
        let mut sumsq_rel_err_mle = 0.0f64;
        let mut sumsq_rel_err_hll = 0.0f64;

        // deterministic trial seeds
        let mut seed = 0x1234_5678_9ABC_DEF0u64;

        println!(
            "Comparing MLE vs HLL over {} trials | N={} | p_ull={} p_hll={}",
            TRIALS, N, P_ULL, P_HLL
        );

        for t in 0..TRIALS {
            // bump seed (simple LCG-ish hop)
            seed = seed
                .wrapping_mul(0x9E3779B97F4A7C15)
                .wrapping_add(0xD1B54A32D192ED03);

            // ---------- generate N pseudorandom 64-bit values (SplitMix64 inline) ----------
            let mut keys = Vec::with_capacity(N as usize);
            let mut x = seed;
            for _ in 0..N {
                x = x.wrapping_add(0x9E3779B97F4A7C15);
                let mut z = x;
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
                z ^= z >> 31;
                keys.push(z);
            }

            // ---------- feed ULL ----------
            let mut ull = UltraLogLog::new(P_ULL).expect("valid p");
            for &h in &keys {
                ull.add(h); // already 64-bit hash values
            }
            let mle = MaximumLikelihoodEstimator;
            let est_mle = mle.estimate(&ull);

            // ---------- feed HLL (use push_hash64 to avoid a rehash) ----------
            let mut hll = HLL::<u64>::with_p(P_HLL);
            for &h in &keys {
                hll.push_hash64(h);
            }
            let est_hll = hll.len();

            // ---------- relative errors ----------
            let n_f = N as f64;
            let rel_mle = (est_mle - n_f) / n_f;
            let rel_hll = (est_hll - n_f) / n_f;

            sum_rel_err_mle += rel_mle;
            sum_rel_err_hll += rel_hll;
            sumsq_rel_err_mle += rel_mle * rel_mle;
            sumsq_rel_err_hll += rel_hll * rel_hll;

            println!(
                "[trial {:02}] MLE rel.err = {:+.4}%   HLL rel.err = {:+.4}%",
                t,
                rel_mle * 100.0,
                rel_hll * 100.0
            );
        }

        // ---------- RMSE (relative) & mean bias ----------
        let t_f = TRIALS as f64;
        let rmse_mle = (sumsq_rel_err_mle / t_f).sqrt();
        let rmse_hll = (sumsq_rel_err_hll / t_f).sqrt();
        let mean_mle = sum_rel_err_mle / t_f;
        let mean_hll = sum_rel_err_hll / t_f;

        println!(
            "\nRelative RMSE:  MLE = {:.4}%   HLL = {:.4}%",
            rmse_mle * 100.0,
            rmse_hll * 100.0
        );
        println!(
            "Mean bias:      MLE = {:+.4}%  HLL = {:+.4}%",
            mean_mle * 100.0,
            mean_hll * 100.0
        );

        // sanity: with p=16, expected SE around ~0.4% for HLL; allow slack in debug
        assert!(rmse_mle < 0.01, "MLE RMSE too large: {:.4}%", rmse_mle * 100.0);
        assert!(rmse_hll < 0.01, "HLL RMSE too large: {:.4}%", rmse_hll * 100.0);
    }
}
