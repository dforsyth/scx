// Copyright (c) Meta Platforms, Inc. and affiliates.

// This software may be used and distributed according to the terms of the
// GNU General Public License version 2.
mod bpf_skel;
pub use bpf_skel::*;
pub mod bpf_intf;

use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::collections::HashMap;
use std::collections::VecDeque;
use std::mem::MaybeUninit;
use std::os::unix::fs::MetadataExt;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use std::sync::Arc;

use anyhow::bail;
use anyhow::Context;
use anyhow::Result;
use cgroupfs::CgroupReader;
use clap::Parser;
use itertools::multizip;
use libbpf_rs::skel::OpenSkel;
use libbpf_rs::skel::Skel;
use libbpf_rs::skel::SkelBuilder;
use libbpf_rs::MapCore as _;
use libbpf_rs::OpenObject;
use log::debug;
use log::info;
use log::trace;
use maplit::hashmap;
use scx_utils::import_enums;
use scx_utils::init_libbpf_logging;
use scx_utils::scx_enums;
use scx_utils::scx_ops_attach;
use scx_utils::scx_ops_load;
use scx_utils::scx_ops_open;
use scx_utils::uei_exited;
use scx_utils::uei_report;
use scx_utils::Topology;
use scx_utils::UserExitInfo;
use scx_utils::NR_CPUS_POSSIBLE;
use scx_utils::NR_CPU_IDS;

const MAX_CELLS: usize = bpf_intf::consts_MAX_CELLS as usize;
const CSTAT_LOCAL: usize = bpf_intf::cell_stat_idx_CSTAT_LOCAL as usize;
const CSTAT_GLOBAL: usize = bpf_intf::cell_stat_idx_CSTAT_GLOBAL as usize;
const CSTAT_AFFN_VIOL: usize = bpf_intf::cell_stat_idx_CSTAT_AFFN_VIOL as usize;
const NR_CSTATS: usize = bpf_intf::cell_stat_idx_NR_CSTATS as usize;

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

    /// Run mitosis even if we only have a single (root) cell.
    #[clap(long, default_value = "false")]
    allow_root_cell_only: bool,
}

unsafe fn any_as_u8_slice<T: Sized>(p: &T) -> &[u8] {
    ::std::slice::from_raw_parts((p as *const T) as *const u8, ::std::mem::size_of::<T>())
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

#[derive(Clone, Debug, Default)]
struct CellCost {
    percpu_cycles: Vec<u64>,

    stats_local: u64,
    stats_global: u64,
    stats_affn_viol: u64,
}

impl CellCost {
    fn new() -> Self {
        Self {
            percpu_cycles: vec![0; *NR_CPU_IDS],
            ..Default::default()
        }
    }
}

#[derive(Debug)]
// A `Cell` is a dynamic collection of CPUs. There is no limitation of the
// number of tasks that can be assigned to a given `Cell`. The bpf layer
// generates cpu bitmask using the assignments described by the cell. Cells are
// an accounting domain, and we expect the union of all non-root cell cpus to
// be the cpus of the root cell.
struct Cell {
    cpu_assignment: BTreeSet<u32>,

    cost: CellCost,
    prev_cost: CellCost,
}

struct Scheduler<'a> {
    skel: BpfSkel<'a>,
    cells: BTreeMap<u32, Cell>,
    cgroup_to_cell: HashMap<String, u32>,
    monitor_interval: std::time::Duration,
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

        if opts.verbose >= 1 {
            skel.maps.rodata_data.debug = true;
        }
        if opts.allow_root_cell_only {
            skel.maps.rodata_data.allow_root_cell_only = true;
        }
        skel.maps.rodata_data.nr_possible_cpus = *NR_CPUS_POSSIBLE as u32;
        for cpu in topology.all_cores.keys() {
            skel.maps.rodata_data.all_cpus[cpu / 8] |= 1 << (cpu % 8);
        }

        let skel = scx_ops_load!(skel, mitosis, uei)?;

        // Initial configuration
        let mut cells: BTreeMap<u32, Cell> = BTreeMap::new();
        let mut num_cells = 1;
        let mut root_cell = Cell {
            cpu_assignment: topology
                .all_cpus
                .keys()
                .cloned()
                .map(|x| x as u32)
                .collect(),
            cost: CellCost::new(),
            prev_cost: CellCost::new(),
        };
        let mut cgroup_to_cell = hashmap! {
            "".to_string() => 0
        };

        let mut stack = VecDeque::new();
        let root = CgroupReader::root()?;
        for child in root.child_cgroup_iter()? {
            stack.push_back(child);
        }
        while let Some(reader) = stack.pop_back() {
            for child in reader.child_cgroup_iter()? {
                stack.push_back(child);
            }
            let cpuset = match reader.read_cpuset_cpus() {
                Err(cgroupfs::Error::IoError(_, e)) if e.kind() == std::io::ErrorKind::NotFound => {
                    continue;
                }
                r => r,
            }
            .with_context(|| {
                format!(
                    "Error while reading cpuset from {}",
                    reader.name().display()
                )
            })?;
            if !cpuset.cpus.is_empty() {
                trace!(
                    "Cgroup {} has non-empty cpuset: {}",
                    reader.name().display(),
                    cpuset
                );
                let cg_name = reader.name().to_string_lossy().to_string();
                if !cpuset.cpus.is_subset(&root_cell.cpu_assignment) {
                    bail!(
                        "cpuset of cgroup {} ({:?}) is not a subset of the root cell cpus ({:?})",
                        reader.name().display(),
                        cpuset.cpus,
                        root_cell.cpu_assignment
                    );
                }
                root_cell
                    .cpu_assignment
                    .retain(|x| !cpuset.cpus.contains(x));
                cells.insert(
                    num_cells,
                    Cell {
                        cpu_assignment: cpuset.cpus,
                        cost: CellCost::new(),
                        prev_cost: CellCost::new(),
                    },
                );
                cgroup_to_cell.insert(cg_name, num_cells);
                num_cells += 1;
            }
        }
        cells.insert(0, root_cell);

        Ok(Self {
            skel,
            cells,
            cgroup_to_cell,
            monitor_interval: std::time::Duration::from_secs(opts.monitor_interval_s),
        })
    }

    fn run(&mut self, shutdown: Arc<AtomicBool>) -> Result<UserExitInfo> {
        self.update_bpf_cells()?;
        self.update_cgroup_to_cell_assignment()?;
        let _struct_ops = scx_ops_attach!(self.skel, mitosis)?;
        info!("Mitosis Scheduler Attached");
        while !shutdown.load(Ordering::Relaxed) && !uei_exited!(&self.skel, uei) {
            std::thread::sleep(self.monitor_interval);
            self.refresh_cell_costs()?;
            self.debug()?;
            if self.skel.maps.bss_data.user_global_seq != self.skel.maps.bss_data.global_seq {
                trace!("BPF reconfiguration still in progress, skipping further changes");
                continue;
            }
        }
        uei_report!(&self.skel, uei)
    }

    fn update_cgroup_to_cell_assignment(&mut self) -> Result<()> {
        for (cgroup, cell_idx) in self.cgroup_to_cell.iter() {
            let mut cg_path = String::from("/sys/fs/cgroup/");
            cg_path.push_str(cgroup);
            let cg_inode = std::fs::metadata(&cg_path)?.ino();
            let cg_inode_slice = unsafe { any_as_u8_slice(&cg_inode) };
            let cell_idx_u32 = *cell_idx as libc::__u32;
            let cell_idx_slice = unsafe { any_as_u8_slice(&cell_idx_u32) };

            self.skel
                .maps
                .cgrp_init_cell_assignment
                .update(
                    cg_inode_slice,
                    cell_idx_slice,
                    libbpf_rs::MapFlags::NO_EXIST,
                )
                .with_context(|| {
                    format!("Failed to update cgroup cell assignment for: {}", cg_path)
                })?;
            trace!(
                "Assigned {} with inode {} to {}",
                cgroup,
                cg_inode,
                cell_idx
            );
        }
        self.skel.maps.bss_data.update_cell_assignment = true;
        Ok(())
    }

    fn update_bpf_cells(&mut self) -> Result<()> {
        for (cell_idx, cell) in self.cells.iter() {
            for cpu in cell.cpu_assignment.iter() {
                trace!("Assigned CPU {} to Cell {}", cpu, cell_idx);
                self.skel.maps.bss_data.cells[*cell_idx as usize].cpus[(cpu / 8) as usize] |=
                    1 << cpu % 8;
            }
        }
        Ok(())
    }

    fn refresh_cell_costs(&mut self) -> Result<()> {
        let cpu_ctxs = read_cpu_ctxs(&self.skel)
            .context("Failed to read cpu ctxs")
            .unwrap();

        let nr_cpus = cpu_ctxs.len();

        let mut cell_percpu_cycles = vec![vec![0u64; nr_cpus]; MAX_CELLS];
        let mut cell_stats = vec![vec![0u64; NR_CSTATS]; MAX_CELLS];

        for (cpu, ctx) in cpu_ctxs.iter().enumerate() {
            for (cell_id, _) in &self.cells {
                let cell_idx = *cell_id as usize;
                cell_percpu_cycles[cell_idx][cpu] = ctx.cell_cycles[cell_idx];

                for stat in 0..NR_CSTATS {
                    cell_stats[cell_idx][stat] = ctx.cstats[cell_idx][stat];
                }
            }
        }

        for ((_, cell), percpu_cycles, stats) in
            multizip((&mut self.cells, cell_percpu_cycles.iter(), &cell_stats))
        {
            // XXX
            cell.prev_cost = cell.cost.clone();
            cell.cost = CellCost {
                percpu_cycles: percpu_cycles.to_vec(),
                stats_local: stats[CSTAT_LOCAL],
                stats_global: stats[CSTAT_GLOBAL],
                stats_affn_viol: stats[CSTAT_AFFN_VIOL],
            };
        }

        Ok(())
    }

    /// Output various debugging data like per cell stats, per-cpu stats, etc.
    fn debug(&mut self) -> Result<()> {
        let mut diff_cycles = vec![[0u64; MAX_CELLS]; *NR_CPU_IDS];
        for (cell_id, cell) in self.cells.iter() {
            let percpu_cycles_deltas: Vec<u64> = cell
                .cost
                .percpu_cycles
                .iter()
                .zip(cell.prev_cost.percpu_cycles.iter())
                .map(|(a, b)| a - b)
                .collect();

            for cpu in 0..*NR_CPU_IDS {
                diff_cycles[cpu][*cell_id as usize] = percpu_cycles_deltas[cpu];
            }

            // trace!("Cell {}: {:?}", cell_id, cell);
        }

        for cpu in 0..*NR_CPU_IDS {
            trace!("CPU {}: {:?}", cpu, diff_cycles[cpu]);
        }

        Ok(())
    }
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
