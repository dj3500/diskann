/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Clocks, CPU placement and host information for timed runs (Linux; elsewhere the values
//! are `None` and timed runs refuse to start).

use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{json, Value};

/// CPU time consumed by the calling thread, in nanoseconds (`CLOCK_THREAD_CPUTIME_ID`).
#[cfg(target_os = "linux")]
#[inline]
pub fn thread_cpu_ns() -> u64 {
    let mut ts = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: `ts` is a valid, writable `timespec`, and the clock id exists on Linux.
    let rc = unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut ts) };
    if rc != 0 {
        return 0;
    }
    ts.tv_sec as u64 * 1_000_000_000 + ts.tv_nsec as u64
}

#[cfg(not(target_os = "linux"))]
#[inline]
pub fn thread_cpu_ns() -> u64 {
    0
}

/// CPU time consumed by the whole process (all threads), in seconds
/// (`CLOCK_PROCESS_CPUTIME_ID`).
#[cfg(target_os = "linux")]
pub fn process_cpu_s() -> Option<f64> {
    let mut ts = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: `ts` is a valid, writable `timespec`, and the clock id exists on Linux.
    let rc = unsafe { libc::clock_gettime(libc::CLOCK_PROCESS_CPUTIME_ID, &mut ts) };
    (rc == 0).then_some(ts.tv_sec as f64 + ts.tv_nsec as f64 * 1e-9)
}

#[cfg(not(target_os = "linux"))]
pub fn process_cpu_s() -> Option<f64> {
    None
}

/// The CPU the calling thread runs on.
#[cfg(target_os = "linux")]
pub fn current_cpu() -> Option<u32> {
    // SAFETY: `sched_getcpu` has no preconditions.
    u32::try_from(unsafe { libc::sched_getcpu() }).ok()
}

#[cfg(not(target_os = "linux"))]
pub fn current_cpu() -> Option<u32> {
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
    Some(
        (0..max)
            // SAFETY: `cpu < max`, the capacity of `set`.
            .filter(|&cpu| unsafe { libc::CPU_ISSET(cpu, &set) })
            .map(|cpu| cpu as u32)
            .collect(),
    )
}

#[cfg(not(target_os = "linux"))]
pub fn affinity() -> Option<Vec<u32>> {
    None
}

/// Scheduling niceness of the process.
#[cfg(target_os = "linux")]
pub fn nice() -> Option<i32> {
    // SAFETY: `getpriority` has no memory-safety preconditions.
    Some(unsafe { libc::getpriority(libc::PRIO_PROCESS, 0) })
}

#[cfg(not(target_os = "linux"))]
pub fn nice() -> Option<i32> {
    None
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

/// Peak resident set size of the process in bytes.
pub fn peak_rss_bytes() -> Option<u64> {
    let v = proc_status_field("VmHWM")?;
    let kb: u64 = v.trim_end_matches("kB").trim().parse().ok()?;
    Some(kb * 1024)
}

/// NUMA policy of the heap (from `/proc/self/numa_maps`, e.g. `bind:1` or `default`).
pub fn heap_mempolicy() -> Option<String> {
    let text = std::fs::read_to_string("/proc/self/numa_maps").ok()?;
    let line = text
        .lines()
        .find(|l| l.contains(" heap"))
        .or_else(|| text.lines().next())?;
    line.split_whitespace().nth(1).map(str::to_owned)
}

/// Anonymous memory of the process backed by transparent huge pages, in kB (from
/// `/proc/self/smaps_rollup`).
pub fn anon_huge_kb() -> Option<u64> {
    let text = std::fs::read_to_string("/proc/self/smaps_rollup").ok()?;
    let line = text.lines().find(|l| l.starts_with("AnonHugePages:"))?;
    line.split_whitespace().nth(1)?.parse().ok()
}

/// Placement and host information recorded with every run.
pub fn host_info() -> Value {
    let cpu = current_cpu();
    json!({
        "hugepages_advised": crate::hugepages::enabled(),
        "anon_huge_kb": anon_huge_kb(),
        "hostname": std::fs::read_to_string("/proc/sys/kernel/hostname").ok().map(|s| s.trim().to_owned()),
        "pid": std::process::id(),
        "cpu": cpu,
        "numa_node": cpu.and_then(numa_node),
        "affinity": affinity(),
        "mems_allowed": proc_status_field("Mems_allowed_list"),
        "mempolicy": heap_mempolicy(),
        "nice": nice(),
        "load_avg": load_average(),
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
        .unwrap_or(0);
    let (days, rem) = (secs / 86_400, secs % 86_400);
    let (h, m, s) = (rem / 3600, rem % 3600 / 60, rem % 60);
    // Civil date from days since 1970-01-01 (Howard Hinnant's algorithm).
    let z = days as i64 + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let mo = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(mo <= 2);
    if compact {
        format!("{y:04}{mo:02}{d:02}-{h:02}{m:02}{s:02}")
    } else {
        format!("{y:04}-{mo:02}-{d:02}T{h:02}:{m:02}:{s:02}Z")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_lists_and_clocks() {
        assert_eq!(parse_cpu_list("0-2,7,9-10\n"), vec![0, 1, 2, 7, 9, 10]);
        assert!(parse_cpu_list("").is_empty());
        let a = thread_cpu_ns();
        let mut x = 0u64;
        for i in 0..200_000u64 {
            x = x.wrapping_add(std::hint::black_box(i));
        }
        std::hint::black_box(x);
        assert!(thread_cpu_ns() >= a);
        let ts = utc_timestamp(false);
        assert_eq!(ts.len(), 20);
        assert!(ts.ends_with('Z'));
        assert_eq!(utc_timestamp(true).len(), 15);
    }
}
