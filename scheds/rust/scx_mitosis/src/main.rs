// Copyright (c) Meta Platforms, Inc. and affiliates.

// This software may be used and distributed according to the terms of the
// GNU General Public License version 2.
mod bpf_skel;
pub use bpf_skel::*;
pub mod bpf_intf;

use std::collections::BTreeMap;
use std::collections::HashMap;
use std::mem::MaybeUninit;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Instant;

use anyhow::bail;
use anyhow::Context;
use anyhow::Result;
use clap::Parser;
use libbpf_rs::MapCore as _;
use libbpf_rs::OpenObject;
use log::debug;
use log::info;
use log::trace;
use resctrlfs::ResctrlReader;
use scx_utils::init_libbpf_logging;
use scx_utils::scx_enums;
use scx_utils::scx_ops_attach;
use scx_utils::scx_ops_load;
use scx_utils::scx_ops_open;
use scx_utils::uei_exited;
use scx_utils::uei_report;
use scx_utils::Cpumask;
use scx_utils::Topology;
use scx_utils::UserExitInfo;
use scx_utils::NR_CPUS_POSSIBLE;
use scx_utils::NR_CPU_IDS;

const MAX_CELLS: usize = bpf_intf::consts_MAX_CELLS as usize;

/// scx_mitosis: A dynamic affinity scheduler
///
/// Cgroups are assigned to a dynamic number of Cells which are assigned to a
/// dynamic set of CPUs. The BPF part does simple vtime scheduling for each cell.
///
/// Userspace makes the dynamic decisions of which Cells should be merged or
/// split and which cpus they should be assigned to.
#[derive(Debug, Parser)]
struct Opts {
    /// Enable verbose output, including libbpf details. Specify multiple
    /// times to increase verbosity.
    #[clap(short = 'v', long, action = clap::ArgAction::Count)]
    verbose: u8,

    /// Exit debug dump buffer length. 0 indicates default.
    #[clap(long, default_value = "0")]
    exit_dump_len: u32,

    /// Interval to consider reconfiguring the Cells (e.g. merge or split)
    #[clap(long, default_value = "10")]
    reconfiguration_interval_s: u64,

    /// Interval to consider rebalancing CPUs to Cells
    #[clap(long, default_value = "5")]
    rebalance_cpus_interval_s: u64,

    /// Interval to report monitoring information
    #[clap(long, default_value = "1")]
    monitor_interval_s: u64,
}

unsafe fn any_as_u8_slice<T: Sized>(p: &T) -> &[u8] {
    unsafe {
        ::std::slice::from_raw_parts((p as *const T) as *const u8, ::std::mem::size_of::<T>())
    }
}

#[derive(Debug)]
struct CellStats {
    collected_time: Instant,
    l3_to_mbm_total: BTreeMap<usize, u64>,
}

impl CellStats {
    fn gen_l3_to_mbm_total(
        cell: &Cell,
        mbm: &HashMap<Cpumask, BTreeMap<usize, u64>>,
    ) -> Result<BTreeMap<usize, u64>> {
        // assume that a mon group is at least exclusive to a cell.

        let mut l3_to_mbm_total = BTreeMap::new();
        let mut assumed_cpus = Cpumask::new();
        // group cpus -> usage per l3
        for (cpus, l3_mbm_total_bytes) in mbm {
            if cell.cpus.and(cpus).eq(cpus) {
                for (&l3, &mbm_total_bytes) in l3_mbm_total_bytes.iter() {
                    l3_to_mbm_total.insert(l3, mbm_total_bytes);
                }

                // accounting
                assumed_cpus = assumed_cpus.or(cpus);
            }
        }

        if assumed_cpus != cell.cpus {
            bail!("assumed_cpus != cell.cpus");
        }

        Ok(l3_to_mbm_total)
    }
}

// Per cell book-keeping
#[derive(Debug)]
struct Cell {
    cpus: Cpumask,
}

struct Scheduler<'a> {
    skel: BpfSkel<'a>,
    prev_percpu_cell_cycles: Vec<[u64; MAX_CELLS]>,
    monitor_interval: std::time::Duration,
    cells: HashMap<u32, Cell>,
    cell_stats: BTreeMap<u32, Arc<CellStats>>,
    prev_cell_stats: BTreeMap<u32, Arc<CellStats>>,
}

impl<'a> Scheduler<'a> {
    fn init(opts: &Opts, open_object: &'a mut MaybeUninit<OpenObject>) -> Result<Self> {
        let topology = Topology::new()?;

        let mut skel_builder = BpfSkelBuilder::default();
        skel_builder.obj_builder.debug(opts.verbose > 1);
        init_libbpf_logging(None);
        let mut skel = scx_ops_open!(skel_builder, open_object, mitosis)?;

        skel.struct_ops.mitosis_mut().exit_dump_len = opts.exit_dump_len;

        skel.maps.rodata_data.slice_ns = scx_enums.SCX_SLICE_DFL;

        skel.maps.rodata_data.nr_possible_cpus = *NR_CPUS_POSSIBLE as u32;
        for cpu in topology.all_cores.keys() {
            skel.maps.rodata_data.all_cpus[cpu / 8] |= 1 << (cpu % 8);
        }

        let skel = scx_ops_load!(skel, mitosis, uei)?;

        Ok(Self {
            skel,
            prev_percpu_cell_cycles: vec![[0; MAX_CELLS]; *NR_CPU_IDS],
            monitor_interval: std::time::Duration::from_secs(opts.monitor_interval_s),
            cells: HashMap::new(),
            cell_stats: BTreeMap::new(),
            prev_cell_stats: BTreeMap::new(),
        })
    }

    fn run(&mut self, shutdown: Arc<AtomicBool>) -> Result<UserExitInfo> {
        let struct_ops = scx_ops_attach!(self.skel, mitosis)?;
        info!("Mitosis Scheduler Attached");
        while !shutdown.load(Ordering::Relaxed) && !uei_exited!(&self.skel, uei) {
            std::thread::sleep(self.monitor_interval);
            self.refresh_bpf_cells()?;
            self.collect_cell_stats()?;
            self.debug()?;
        }
        drop(struct_ops);
        uei_report!(&self.skel, uei)
    }

    /// Output various debugging data like per cell stats, per-cpu stats, etc.
    fn debug(&mut self) -> Result<()> {
        let zero = 0 as libc::__u32;
        let zero_slice = unsafe { any_as_u8_slice(&zero) };
        if let Some(v) = self
            .skel
            .maps
            .cpu_ctxs
            .lookup_percpu(zero_slice, libbpf_rs::MapFlags::ANY)
            .context("Failed to lookup cpu_ctxs map")?
        {
            for (cpu, ctx) in v.iter().enumerate() {
                let cpu_ctx = unsafe {
                    let ptr = ctx.as_slice().as_ptr() as *const bpf_intf::cpu_ctx;
                    &*ptr
                };
                let diff_cycles: Vec<i64> = self.prev_percpu_cell_cycles[cpu]
                    .iter()
                    .zip(cpu_ctx.cell_cycles.iter())
                    .map(|(a, b)| (b - a) as i64)
                    .collect();
                self.prev_percpu_cell_cycles[cpu] = cpu_ctx.cell_cycles;
                trace!("CPU {}: {:?}", cpu, diff_cycles);
            }
        }

        for (cell_id, cell_stats) in &self.cell_stats {
            trace!("CELL {}: {:?}", cell_id, cell_stats.l3_to_mbm_total);
        }

        Ok(())
    }

    fn refresh_bpf_cells(&mut self) -> Result<()> {
        // collect all cpus per cell.
        let mut cell_to_cpus: HashMap<u32, Cpumask> = HashMap::new();
        let cpu_ctxs = read_cpu_ctxs(&self.skel)?;
        for (i, cpu_ctx) in cpu_ctxs.iter().enumerate() {
            cell_to_cpus
                .entry(cpu_ctx.cell)
                .and_modify(|mask| mask.set_cpu(i).expect("set cpu in existing mask"))
                .or_insert_with(|| {
                    let mut mask = Cpumask::new();
                    mask.set_cpu(i).expect("set cpu in new mask");
                    mask
                });
        }

        // create cells we don't have yet, drop cells that are no longer in use.
        let cells = &self.skel.maps.bss_data.cells;
        for i in 0..MAX_CELLS {
            let cell_idx = i as u32;
            let bpf_cell = cells[i];
            if bpf_cell.in_use > 0 {
                self.cells.entry(cell_idx).or_insert(Cell {
                    cpus: cell_to_cpus
                        .get(&cell_idx)
                        .expect("missing cell in cpu map")
                        .clone(),
                });
            } else {
                self.cells.remove(&cell_idx);
            }
        }

        Ok(())
    }

    fn collect_cell_stats(&mut self) -> Result<()> {
        let collected_time = Instant::now();
        // move prev stats. check for dead cells.
        while let Some((cell_id, stats)) = self.cell_stats.pop_first() {
            if self.cells.contains_key(&cell_id) {
                self.prev_cell_stats.insert(cell_id, stats);
            }
        }

        let resctrl_membw = read_resctrl_membw()?;
        for (&cell_id, cell) in &self.cells {
            // no need to collect stats for root
            if cell_id == 0 {
                continue;
            }

            let stats = Arc::new(CellStats {
                collected_time,
                l3_to_mbm_total: CellStats::gen_l3_to_mbm_total(&cell, &resctrl_membw)?,
            });
            self.cell_stats.insert(cell_id, stats);
        }

        Ok(())
    }
}

fn read_cpu_ctxs(skel: &BpfSkel) -> Result<Vec<bpf_intf::cpu_ctx>> {
    let mut cpu_ctxs = vec![];
    let cpu_ctxs_vec = skel
        .maps
        .cpu_ctxs
        .lookup_percpu(&0u32.to_ne_bytes(), libbpf_rs::MapFlags::ANY)
        .context("Failed to lookup cpu_ctx")?
        .unwrap();
    for cpu in 0..*NR_CPUS_POSSIBLE {
        cpu_ctxs.push(*unsafe {
            &*(cpu_ctxs_vec[cpu].as_slice().as_ptr() as *const bpf_intf::cpu_ctx)
        });
    }
    Ok(cpu_ctxs)
}

fn cpuset_to_cpumask(cpuset: &resctrlfs::Cpuset) -> Cpumask {
    let mut cpumask = Cpumask::new();
    for x in &cpuset.cpus {
        cpumask.set_cpu(*x as usize).expect("cast to usize");
    }
    return cpumask;
}

fn read_resctrl_membw() -> Result<HashMap<Cpumask, BTreeMap<usize, u64>>> {
    // map ctrl mon groups (by cpu) to mbm bytes per l3
    let reader = ResctrlReader::new("/sys/fs/resctrl".into(), true)?;
    let sample = reader.read_all()?;

    let mut mon_grp_to_l3_usage = HashMap::new();

    // im sorry.
    if let Some(mon_groups) = &sample.mon_groups {
        for (mon_group_id, mon_group) in mon_groups {
            let mut l3_usage = BTreeMap::new();
            if let Some(cpuset) = &mon_group.cpuset {
                if let Some(mon_stat) = &mon_group.mon_stat {
                    if let Some(l3_mon_stat) = &mon_stat.l3_mon_stat {
                        for (&l3, stat) in l3_mon_stat.iter() {
                            if let Some(rmid_bytes) = &stat.mbm_total_bytes {
                                if let resctrlfs::RmidBytes::Bytes(b) = rmid_bytes {
                                    l3_usage.insert(l3 as usize, *b);
                                }
                            }
                        }
                    }
                }

                let cpumask = cpuset_to_cpumask(cpuset);
                mon_grp_to_l3_usage.insert(cpumask, l3_usage);
            }
        }
    }

    if let Some(ctrl_mon_groups) = &sample.ctrl_mon_groups {
        for (ctrl_mon_group_id, ctrl_mon_group) in ctrl_mon_groups.iter() {
            if let Some(mon_groups) = &ctrl_mon_group.mon_groups {
                for (mon_group_id, mon_group) in mon_groups.iter() {
                    if let Some(cpuset) = &mon_group.cpuset {
                        let mut l3_usage = BTreeMap::new();
                        if let Some(mon_stat) = &mon_group.mon_stat {
                            if let Some(l3_mon_stat) = &mon_stat.l3_mon_stat {
                                for (&l3, stat) in l3_mon_stat.iter() {
                                    if let Some(rmid_bytes) = &stat.mbm_total_bytes {
                                        if let resctrlfs::RmidBytes::Bytes(b) = &rmid_bytes {
                                            l3_usage.insert(l3 as usize, *b);
                                        }
                                    }
                                }
                            }
                        }

                        let cpumask = cpuset_to_cpumask(cpuset);
                        mon_grp_to_l3_usage.insert(cpumask, l3_usage);
                    }
                }
            }
        }
    }

    Ok(mon_grp_to_l3_usage)
}

fn main() -> Result<()> {
    let opts = Opts::parse();

    let llv = match opts.verbose {
        0 => simplelog::LevelFilter::Info,
        1 => simplelog::LevelFilter::Debug,
        _ => simplelog::LevelFilter::Trace,
    };
    let mut lcfg = simplelog::ConfigBuilder::new();
    lcfg.set_time_level(simplelog::LevelFilter::Error)
        .set_location_level(simplelog::LevelFilter::Off)
        .set_target_level(simplelog::LevelFilter::Off)
        .set_thread_level(simplelog::LevelFilter::Off);
    simplelog::TermLogger::init(
        llv,
        lcfg.build(),
        simplelog::TerminalMode::Stderr,
        simplelog::ColorChoice::Auto,
    )?;

    debug!("opts={:?}", &opts);

    let shutdown = Arc::new(AtomicBool::new(false));
    let shutdown_clone = shutdown.clone();
    ctrlc::set_handler(move || {
        shutdown_clone.store(true, Ordering::Relaxed);
    })
    .context("Error setting Ctrl-C handler")?;

    let mut open_object = MaybeUninit::uninit();
    loop {
        let mut sched = Scheduler::init(&opts, &mut open_object)?;
        if !sched.run(shutdown.clone())?.should_restart() {
            break;
        }
    }

    Ok(())
}
