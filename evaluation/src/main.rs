use std::fmt;
use std::fs::File;
use std::hash::{BuildHasher, Hasher};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use std::str;

use clap::{App, Arg};
use rand::prelude::*;
use rayon::prelude::*;

use ultraloglog::{Estimator, MaximumLikelihoodEstimator, OptimalFGRAEstimator, UltraLogLog};

// PassThrough hasher for direct testing
struct PassThroughHasher(u64);

impl Hasher for PassThroughHasher {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, _: &[u8]) {}
    fn write_u64(&mut self, i: u64) {
        self.0 = i
    }
}

struct PassThroughHasherBuilder;

impl BuildHasher for PassThroughHasherBuilder {
    type Hasher = PassThroughHasher;
    fn build_hasher(&self) -> Self::Hasher {
        PassThroughHasher(0)
    }
}

struct Estimation(u64, u64, String);

impl fmt::Display for Estimation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {} {}", self.0, self.1, self.2)
    }
}

// Generates random 64-bit hash values and saves them in files
fn generate(args: &clap::ArgMatches) {
    let (count, runs, output) = (
        args.value_of("count").unwrap().parse::<usize>().unwrap(),
        args.value_of("runs").unwrap().parse::<usize>().unwrap(),
        args.value_of("output").unwrap(),
    );

    (0..runs).into_par_iter().for_each(|r| {
        let values: Vec<u64> = (0..count).map(|_| rand::random::<u64>()).collect();
        let filename = format!("hashes-r{}.dat", r);
        save(&values, filename.as_str(), output);
    });
}

// Runs evaluation experiments for UltraLogLog
fn run(args: &clap::ArgMatches) {
    let (mode, precision, estimator, output) = (
        args.value_of("mode").unwrap(),
        args.value_of("precision").unwrap().parse::<u32>().unwrap(),
        args.value_of("estimator").unwrap(),
        args.value_of("output").unwrap(),
    );

    match mode {
        "hashes" => {
            let files: Vec<&str> = args.values_of("input").unwrap().collect();

            files.par_iter().for_each(|file| {
                let mut ull = UltraLogLog::new(precision).unwrap();
                let hashes = load::<u64>(file);

                let estimations = hashes
                    .iter()
                    .enumerate()
                    .map(|(i, num)| {
                        ull.add(*num);
                        let est = match estimator {
                            "optimal" => OptimalFGRAEstimator.estimate(&ull),
                            "ml" => MaximumLikelihoodEstimator.estimate(&ull),
                            _ => unreachable!(),
                        };

                        Estimation((i + 1) as u64, est as u64, estimator.to_string())
                    })
                    .collect();

                let basename = Path::new(file).file_name().unwrap().to_str().unwrap();

                let filename = format!("est-p{}-{}-{}", precision, estimator, basename);

                save(&estimations, filename.as_str(), output);
            });
        }
        "cardinalities" => {
            let runs = args.value_of("runs").unwrap().parse::<usize>().unwrap();
            let cardinalities = args.value_of("cardinalities").unwrap();

            (0..runs).into_par_iter().for_each(|r| {
                let mut ull = UltraLogLog::new(precision).unwrap();
                let cardinalities = load::<usize>(cardinalities);
                let mut rng = rand::thread_rng();
                let mut c = 0;

                let estimations = cardinalities
                    .iter()
                    .map(|cardinality| {
                        while c < *cardinality {
                            let num = rng.gen::<u64>();
                            ull.add(num);
                            c += 1;
                        }

                        let est = match estimator {
                            "optimal" => OptimalFGRAEstimator.estimate(&ull),
                            "ml" => MaximumLikelihoodEstimator.estimate(&ull),
                            _ => unreachable!(),
                        };

                        Estimation(c as u64, est as u64, estimator.to_string())
                    })
                    .collect();

                let filename = format!("est-p{}-{}-cards-r{}.dat", precision, estimator, r);

                save(&estimations, filename.as_str(), output);
            });
        }
        _ => {}
    }
}

// Loads values from a file
fn load<T>(filepath: &str) -> Vec<T>
where
    T: str::FromStr + fmt::Debug,
{
    let reader = BufReader::new(File::open(filepath).unwrap());
    let mut nums = Vec::with_capacity(10000);

    for line in reader.lines() {
        nums.push(
            line.unwrap()
                .parse::<T>()
                .map_err(|_| "Parsing line failed")
                .unwrap(),
        );
    }

    nums
}

// Saves values to a file
fn save<T>(values: &Vec<T>, filename: &str, output: &str)
where
    T: fmt::Display,
{
    let mut writer = BufWriter::new(File::create(Path::new(output).join(filename)).unwrap());

    for val in values {
        write!(writer, "{}\n", val).unwrap();
    }

    writer.flush().unwrap();
}

fn main() {
    let gen_app = App::new("gen")
        .about("generate random hash values")
        .arg(
            Arg::with_name("count")
                .short('c')
                .long("count")
                .required(true)
                .takes_value(true),
        )
        .arg(
            Arg::with_name("runs")
                .short('r')
                .long("runs")
                .required(true)
                .takes_value(true),
        )
        .arg(
            Arg::with_name("output")
                .short('o')
                .long("output")
                .required(true)
                .takes_value(true),
        );

    let run_app = App::new("run")
        .about("run UltraLogLog evaluation experiments")
        .arg(
            Arg::with_name("mode")
                .short('m')
                .long("mode")
                .required(true)
                .takes_value(true)
                .possible_values(&["hashes", "cardinalities"]),
        )
        .arg(
            Arg::with_name("precision")
                .short('p')
                .long("precision")
                .required(true)
                .takes_value(true),
        )
        .arg(
            Arg::with_name("estimator")
                .short('e')
                .long("estimator")
                .required(true)
                .takes_value(true)
                .possible_values(&["optimal", "ml"]),
        )
        .arg(
            Arg::with_name("input")
                .multiple(true)
                .short('i')
                .long("input")
                .takes_value(true)
                .required_if("mode", "hashes"),
        )
        .arg(
            Arg::with_name("runs")
                .short('r')
                .long("runs")
                .takes_value(true)
                .required_if("mode", "cardinalities"),
        )
        .arg(
            Arg::with_name("cardinalities")
                .short('c')
                .long("cardinalities")
                .takes_value(true)
                .required_if("mode", "cardinalities"),
        )
        .arg(
            Arg::with_name("output")
                .short('o')
                .long("output")
                .required(true)
                .takes_value(true),
        );

    let matches = App::new("evl")
        .about("run UltraLogLog evaluation experiments")
        .arg(
            Arg::with_name("jobs")
                .short('j')
                .long("jobs")
                .takes_value(true),
        )
        .subcommand(gen_app)
        .subcommand(run_app)
        .get_matches();

    let jobs = matches
        .value_of("jobs")
        .unwrap_or("1")
        .parse::<usize>()
        .unwrap();

    rayon::ThreadPoolBuilder::new()
        .num_threads(jobs)
        .build_global()
        .unwrap();

    match matches.subcommand() {
        ("gen", Some(sub_matches)) => generate(sub_matches),
        ("run", Some(sub_matches)) => run(sub_matches),
        _ => {}
    }
}
