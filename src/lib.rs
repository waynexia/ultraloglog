// This implementation is based on the Java version
// https://github.com/dynatrace-oss/hash4j/
// See the paper "UltraLogLog: A Practical and More Space-Efficient Alternative to HyperLogLog for Approximate Distinct Counting"

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

    // Helper functions for bit manipulation
    fn unpack(register: u8) -> u64 {
        if register == 0 {
            return 0;
        }
        (4u64 | (register & 3) as u64) << ((register >> 2).wrapping_sub(2))
    }

    fn pack(hash_prefix: u64) -> u8 {
        let nlz = hash_prefix.leading_zeros() + 1;
        ((((nlz as i32).wrapping_neg()) << 2) | ((hash_prefix << nlz) >> 62) as i32) as u8
    }

    /// Get the scaled register change probability
    fn get_scaled_register_change_probability(reg: u8, p: u32) -> u64 {
        if reg == 0 {
            return 1u64 << (64 - p);
        }
        let k = 1i32.wrapping_sub(p as i32) + ((reg >> 2) as i32);
        ((((reg & 2) | ((reg & 1) << 2)) ^ 7u8) as u64) << (63 - k) >> p
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
        observer: Option<&mut T>,
    ) -> &mut Self {
        let q = (self.state.len() as u64 - 1).leading_zeros(); // q = 64 - p
        let idx = (hash_value >> q) as usize;
        let nlz = (!hash_value << q).leading_zeros(); // nlz in {0, 1, ..., 64-p}

        let old_state = self.state[idx];
        let mut hash_prefix = Self::unpack(old_state);
        let exp = (nlz + (64 - q)) as u32; // exp is 0‑64
        hash_prefix |= 1u64.wrapping_shl(exp & 63); // shift modulo 64
        let new_state = Self::pack(hash_prefix);

        if let Some(obs) = observer {
            if new_state != old_state {
                let p = 64 - q;
                obs.state_changed(
                    (Self::get_scaled_register_change_probability(old_state, p)
                        - Self::get_scaled_register_change_probability(new_state, p))
                        as f64
                        * 2f64.powi(-64),
                );
            }
        }

        self.state[idx] = new_state;
        self
    }

    /// Computes a token from a given 64-bit hash value.
    pub fn compute_token(hash_value: u64) -> u32 {
        // In the Java implementation this uses DistinctCountUtil.computeToken1
        // For now we'll implement a simple version
        (hash_value >> 32) as u32
    }

    /// Adds a new element represented by a 32-bit token
    pub fn add_token(&mut self, token: u32) -> &mut Self {
        // Reconstruct hash from token (simplified version)
        let hash = (token as u64) << 32;
        self.add(hash)
    }

    /// Adds a new element represented by a 32-bit token with observer
    pub fn add_token_with_observer<T: StateChangeObserver>(
        &mut self,
        token: u32,
        observer: Option<&mut T>,
    ) -> &mut Self {
        let hash = (token as u64) << 32;
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
                            hash_prefix |= 1u64
                                << (k.leading_zeros() as i32 + other_p_minus_one as i32) as u32;
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
        let quad_root_z = (((beta * beta) as f64 + 4.0 * (alpha * gamma) as f64).sqrt()
            - beta as f64)
            / (2.0 * alpha as f64);
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
        ((((beta * beta) as f64 + 4.0 * (alpha * gamma) as f64).sqrt() - beta as f64)
            / (2.0 * alpha as f64))
            .sqrt()
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
        let p = ull.get_p();
        let m = state.len();

        let mut sum = 0i64;
        let mut b = vec![0i32; 64];

        for &r in state {
            let r2 = r as i32 - ((p << 2) as i32 + 4);
            if r2 < 0 {
                let mut ret = 4i64;
                if r2 == -2 || r2 == -8 {
                    b[0] += 1;
                    ret -= 2;
                }
                if r2 == -2 || r2 == -4 {
                    b[1] += 1;
                    ret -= 1;
                }
                sum += ret << (62 - p);
            } else {
                let k = r2 >> 2;
                let mut ret = i64::MIN >> 2; // Equivalent to 0xE000000000000000i64
                let y0 = (r & 1) as i32;
                let y1 = ((r >> 1) & 1) as i32;
                ret -= (y0 as i64) << 63;
                ret -= (y1 as i64) << 62;
                b[k as usize] += y0;
                b[(k + 1) as usize] += y1;
                b[(k + 2) as usize] += 1;
                sum += ret >> (k + p as i32);
            }
        }

        if sum == 0 {
            return if state[0] == 0 { 0.0 } else { f64::INFINITY };
        }

        b[63 - p as usize] += b[64 - p as usize];
        let factor = (m << 1) as f64;
        let a = (sum as f64) * factor * 2f64.powi(-64);

        factor
            * solve_maximum_likelihood_equation(
                a,
                &b,
                63 - p as i32,
                ML_EQUATION_SOLVER_EPS / (m as f64).sqrt(),
            )
            / (1.0 + ML_BIAS_CORRECTION_CONSTANT / m as f64)
    }
}

// Helper function for MaximumLikelihoodEstimator
fn solve_maximum_likelihood_equation(a: f64, b: &[i32], max_k: i32, eps: f64) -> f64 {
    let mut x = 1.0;
    let mut dx;
    loop {
        let mut s = 0.0;
        let mut ds = 0.0;
        for k in 0..=max_k {
            if b[k as usize] != 0 {
                let t = 2f64.powi(-k);
                let y = x * t;
                let z = 1.0 + y;
                s += b[k as usize] as f64 * y / z;
                ds += b[k as usize] as f64 * t / (z * z);
            }
        }
        dx = (a - s) / ds;
        x += dx;
        if dx.abs() <= eps * x {
            break;
        }
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

        //use deterministic seed
        let komi_builder = KomiBuildHasher { seed: 0x1234_5678_9abc_def0 };

        let mut ull = UltraLogLog::new(8).unwrap();

        // feed four distinct strings through the Komihash builder
        ull.add_value_with_build_hasher("apple",  &komi_builder)
            .add_value_with_build_hasher("banana", &komi_builder)
            .add_value_with_build_hasher("cherry", &komi_builder)
            .add_value_with_build_hasher("dragonfruit",   &komi_builder);

        let est = ull.get_distinct_count_estimate();
        assert!(
            (est - 4.0).abs() < 0.1,          // tolerate ≈5 % relative error at p = 6
            "estimate {:.3} deviates from true count 4",
            est
        );
    }
    #[test]
    fn test_custom_hasher_polymurhash() {
        use std::hash::BuildHasherDefault;
        use polymur_hash::PolymurHasher;

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
}
