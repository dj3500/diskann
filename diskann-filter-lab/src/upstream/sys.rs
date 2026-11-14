/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Clocks and host information for timed runs (Linux; other platforms report `None`).

use std::time::{SystemTime, UNIX_EPOCH};

use diskann_vector::{distance::SquaredL2, PureDistanceFunction};
use serde_json::{json, Value};

/// CPU time consumed by the calling thread, in nanoseconds.
#[cfg(target_os = "linux")]
pub fn thread_cpu_ns() -> u64 {
    let mut ts = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: `ts` is a valid, writable `timespec`, and the clock id is supported on Linux.
    let rc = unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut ts) };
    if rc != 0 {
        return 0;
    }
    ts.tv_sec as u64 * 1_000_000_000 + ts.tv_nsec as u64
}

/// CPU time consumed by the calling thread (unsupported platform: always 0).
#[cfg(not(target_os = "linux"))]
pub fn thread_cpu_ns() -> u64 {
    0
}

/// The CPU the calling thread runs on.
#[cfg(target_os = "linux")]
pub fn current_cpu() -> Option<u32> {
    // SAFETY: `sched_getcpu` has no preconditions.
    let cpu = unsafe { libc::sched_getcpu() };
    u32::try_from(cpu).ok()
}

#[cfg(not(target_os = "linux"))]
pub fn current_cpu() -> Option<u32> {
    None
}

/// Scheduling niceness of the process.
#[cfg(target_os = "linux")]
pub fn nice_value() -> Option<i32> {
    // SAFETY: `getpriority` has no memory-safety preconditions.
    let nice = unsafe { libc::getpriority(libc::PRIO_PROCESS, 0) };
    Some(nice)
}

#[cfg(not(target_os = "linux"))]
pub fn nice_value() -> Option<i32> {
    None
}

/// The CPUs the calling thread may run on.
#[cfg(target_os = "linux")]
pub fn affinity() -> Option<Vec<u32>> {
    // SAFETY: an all-zero `cpu_set_t` is a valid (empty) set.
    let mut set: libc::cpu_set_t = unsafe { std::mem::zeroed() };
    // SAFETY: `set` is a valid, writable `cpu_set_t` of the size passed.
    let rc =
        unsafe { libc::sched_getaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &mut set) };
    if rc != 0 {
        return None;
    }
    let max = 8 * std::mem::size_of::<libc::cpu_set_t>();
    let is_set = |cpu: usize| {
        // SAFETY: `cpu < max`, the capacity of `set`.
        unsafe { libc::CPU_ISSET(cpu, &set) }
    };
    Some(
        (0..max)
            .filter(|&cpu| is_set(cpu))
            .map(|cpu| cpu as u32)
            .collect(),
    )
}

#[cfg(not(target_os = "linux"))]
pub fn affinity() -> Option<Vec<u32>> {
    None
}

/// Compresses a sorted list of CPU ids into `a-b,c` ranges.
pub fn cpu_ranges(cpus: &[u32]) -> String {
    let mut parts = Vec::new();
    let mut i = 0;
    while i < cpus.len() {
        let mut j = i;
        while j + 1 < cpus.len() && cpus[j + 1] == cpus[j] + 1 {
            j += 1;
        }
        parts.push(if i == j {
            cpus[i].to_string()
        } else {
            format!("{}-{}", cpus[i], cpus[j])
        });
        i = j + 1;
    }
    parts.join(",")
}

fn parse_cpu_list(text: &str) -> Vec<u32> {
    text.trim()
        .split(',')
        .filter(|s| !s.is_empty())
        .flat_map(|part| match part.split_once('-') {
            Some((a, b)) => match (a.parse::<u32>(), b.parse::<u32>()) {
                (Ok(a), Ok(b)) => (a..=b).collect(),
                _ => Vec::new(),
            },
            None => part.parse().into_iter().collect(),
        })
        .collect()
}

/// The NUMA node of `cpu` (from sysfs).
pub fn numa_node(cpu: u32) -> Option<u32> {
    let entries = std::fs::read_dir("/sys/devices/system/node").ok()?;
    entries.flatten().find_map(|entry| {
        let name = entry.file_name().into_string().ok()?;
        let node: u32 = name.strip_prefix("node")?.parse().ok()?;
        let cpus = std::fs::read_to_string(entry.path().join("cpulist")).ok()?;
        parse_cpu_list(&cpus).contains(&cpu).then_some(node)
    })
}

/// The 1, 5 and 15 minute load averages.
pub fn load_average() -> Option<[f64; 3]> {
    let text = std::fs::read_to_string("/proc/loadavg").ok()?;
    let mut it = text.split_whitespace().map(|s| s.parse::<f64>());
    Some([it.next()?.ok()?, it.next()?.ok()?, it.next()?.ok()?])
}

fn proc_status_field(name: &str) -> Option<String> {
    let text = std::fs::read_to_string("/proc/self/status").ok()?;
    text.lines().find_map(|line| {
        line.strip_prefix(name)
            .and_then(|rest| rest.strip_prefix(':'))
            .map(|v| v.trim().to_owned())
    })
}

/// NUMA memory policy of the heap (from `/proc/self/numa_maps`, e.g. `bind:1` or `default`).
pub fn heap_mempolicy() -> Option<String> {
    let text = std::fs::read_to_string("/proc/self/numa_maps").ok()?;
    let line = text
        .lines()
        .find(|l| l.contains(" heap"))
        .or_else(|| text.lines().next())?;
    line.split_whitespace().nth(1).map(str::to_owned)
}

/// Peak resident set size of this process in bytes.
pub fn peak_rss_bytes() -> Option<u64> {
    let v = proc_status_field("VmHWM")?;
    let kb: u64 = v.trim_end_matches("kB").trim().parse().ok()?;
    Some(kb * 1024)
}

pub fn hostname() -> Option<String> {
    std::fs::read_to_string("/proc/sys/kernel/hostname")
        .ok()
        .map(|s| s.trim().to_owned())
}

/// Host and placement information recorded with every run.
pub fn host_info() -> Value {
    let cpu = current_cpu();
    json!({
        "hostname": hostname(),
        "pid": std::process::id(),
        "cpu": cpu,
        "numa_node": cpu.and_then(numa_node),
        "affinity": affinity().map(|a| cpu_ranges(&a)),
        "mems_allowed": proc_status_field("Mems_allowed_list"),
        "mempolicy": heap_mempolicy(),
        "nice": nice_value(),
        "available_parallelism": std::thread::available_parallelism().ok().map(|n| n.get()),
    })
}

/// Seconds since the Unix epoch.
pub fn unix_time() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

/// UTC time as `YYYY-MM-DDTHH:MM:SSZ` (`compact`: `YYYYMMDD-HHMMSS`).
pub fn utc_timestamp(compact: bool) -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0) as i64;
    let (days, rem) = (secs.div_euclid(86_400), secs.rem_euclid(86_400));
    // Civil date from days since 1970-01-01 (proleptic Gregorian calendar).
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    let (h, m, s) = (rem / 3600, rem % 3600 / 60, rem % 60);
    if compact {
        format!("{year:04}{month:02}{day:02}-{h:02}{m:02}{s:02}")
    } else {
        format!("{year:04}-{month:02}-{day:02}T{h:02}:{m:02}:{s:02}Z")
    }
}

/// A fixed single-threaded CPU calibration: nanoseconds per squared-L2 distance between
/// 64-dimensional `u8` vectors held in a 256 KiB buffer (best of five passes).
pub fn calibrate() -> f64 {
    const DIM: usize = 64;
    const N: usize = 4096;
    let mut state = 0x2545_f491_4f6c_dd1du64;
    let data: Vec<u8> = (0..N * DIM)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 56) as u8
        })
        .collect();
    let query = &data[..DIM];
    let mut best = f64::INFINITY;
    let mut sink = 0f32;
    for _ in 0..5 {
        let start = std::time::Instant::now();
        for _ in 0..16 {
            for row in data.chunks_exact(DIM) {
                sink += <SquaredL2 as PureDistanceFunction<&[u8], &[u8], f32>>::evaluate(
                    std::hint::black_box(query),
                    row,
                );
            }
        }
        let ns = start.elapsed().as_nanos() as f64 / (16 * N) as f64;
        best = best.min(ns);
    }
    std::hint::black_box(sink);
    best
}

/// FNV-1a 64 of `bytes`, as 16 hex digits.
pub fn fnv1a_hex(bytes: &[u8]) -> String {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h = (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3);
    }
    format!("{h:016x}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_lists() {
        assert_eq!(parse_cpu_list("0-3,8,10-11\n"), [0, 1, 2, 3, 8, 10, 11]);
        assert_eq!(cpu_ranges(&[0, 1, 2, 3, 8, 10, 11]), "0-3,8,10-11");
        assert_eq!(cpu_ranges(&[]), "");
    }

    #[test]
    fn timestamps_have_fixed_width() {
        assert_eq!(utc_timestamp(false).len(), 20);
        assert_eq!(utc_timestamp(true).len(), 15);
    }
}
