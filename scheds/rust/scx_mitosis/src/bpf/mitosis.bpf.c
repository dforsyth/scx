/* Copyright (c) Meta Platforms, Inc. and affiliates. */
/*
 * This software may be used and distributed according to the terms of the
 * GNU General Public License version 2.
 *
 * scx_mitosis is a dynamic affinity scheduler. Cgroups (and their tasks) are
 * assigned to Cells which are affinitized to discrete sets of CPUs. The number
 * of cells is dynamic, as is cgroup to cell assignment and cell to CPU
 * assignment (all are determined by userspace).
 *
 * Each cell has one or more DSQs for vtime scheduling. With LLC-awareness
 * enabled, each cell has a DSQ per LLC domain; otherwise a single flat DSQ.
 */

#ifdef LSP
#define __bpf__
#include "../../../../include/scx/common.bpf.h"
#else
#include <scx/common.bpf.h>
#endif

/*
 * When LLC awareness is disabled, we use a single "fake" LLC index to flatten
 * the entire cell's topology into one scheduling domain. All CPUs in the cell
 * share the same DSQ and vtime, ignoring actual LLC cache boundaries.
 */
#define FAKE_FLAT_CELL_LLC 0

#include "mitosis.bpf.h"
#include "dsq.bpf.h"
#include "llc_aware.bpf.h"

char _license[] SEC("license") = "GPL";

/*
 * Variables populated by userspace
 */
const volatile u32	     nr_possible_cpus = 1;
const volatile bool	     smt_enabled      = true;
const volatile unsigned char all_cpus[MAX_CPUS_U8];

const volatile u64	     slice_ns;
const volatile u64	     root_cgid			     = 1;
const volatile bool	     debug_events_enabled	     = false;
const volatile bool	     exiting_task_workaround_enabled = true;
const volatile bool	     cpu_controller_disabled	     = false;
const volatile bool	     reject_multicpu_pinning	     = false;
const volatile bool	     userspace_managed_cell_mode     = false;
const volatile bool	     enable_borrowing		     = false;

/*
 * Global arrays for LLC topology, populated by userspace before load.
 * Declared in llc_aware.bpf.h as extern.
 */
u32		   cpu_to_llc[MAX_CPUS];
struct llc_cpumask llc_to_cpus[MAX_LLCS];

/* Userspace bumps applied_configuration_seq after publishing a full config. */
u32 applied_configuration_seq;
/* Cpuset writes bump cpuset_seq so userspace can refresh cell cpusets. */
u32 cpuset_seq;

/*
 * Debug events circular buffer
 */
u32 debug_event_pos;

struct {
	__uint(type, BPF_MAP_TYPE_ARRAY);
	__uint(max_entries, DEBUG_EVENTS_BUF_SIZE);
	__type(key, u32);
	__type(value, struct debug_event);
} debug_events SEC(".maps");

/* Configuration struct for apply_cell_config, populated by userspace. */
struct cell_config cell_config;

private(all_cpumask) struct bpf_cpumask __kptr *all_cpumask;
private(root_cgrp) struct cgroup __kptr *root_cgrp;

UEI_DEFINE(uei);

struct cell_map cells SEC(".maps");

/* Forward declaration for init_cgrp_ctx_with_ancestors (defined later) */
static int init_cgrp_ctx_with_ancestors(struct cgroup *cgrp);

/*
 * We store per-cpu values along with per-cell values. Helper functions to
 * translate.
 */

static inline struct cgroup *lookup_cgrp_ancestor(struct cgroup *cgrp,
						  u32		 ancestor)
{
	struct cgroup *cg;

	if (!(cg = bpf_cgroup_ancestor(cgrp, ancestor))) {
		scx_bpf_error("Failed to get ancestor level %d for cgid %llu",
			      ancestor, cgrp->kn->id);
		return NULL;
	}

	return cg;
}

struct {
	__uint(type, BPF_MAP_TYPE_CGRP_STORAGE);
	__uint(map_flags, BPF_F_NO_PREALLOC);
	__type(key, int);
	__type(value, struct cgrp_ctx);
} cgrp_ctxs		       SEC(".maps");

static inline struct cgrp_ctx *lookup_cgrp_ctx_fallible(struct cgroup *cgrp)
{
	struct cgrp_ctx *cgc;

	if (!(cgc = bpf_cgrp_storage_get(&cgrp_ctxs, cgrp, 0, 0))) {
		return NULL;
	}

	return cgc;
}

static inline struct cgrp_ctx *lookup_cgrp_ctx(struct cgroup *cgrp)
{
	struct cgrp_ctx *cgc = lookup_cgrp_ctx_fallible(cgrp);

	if (!cgc)
		scx_bpf_error("cgrp_ctx lookup failed for cgid %llu",
			      cgrp->kn->id);

	return cgc;
}

static inline struct cgroup *task_cgroup(struct task_struct *p)
{
	struct cgroup *cgrp;

	if (!cpu_controller_disabled) {
		cgrp = scx_bpf_task_cgroup(p);
	} else {
		/*
		 * When CPU controller is disabled, scx_bpf_task_cgroup() returns
		 * root. Use p->cgroups->dfl_cgrp to get the task's actual cgroup
		 * in the default (unified) hierarchy.
		 *
		 * p->cgroups is RCU-protected, so we need RCU lock.
		 */
		scoped_guard(rcu)
		{
			cgrp = bpf_cgroup_acquire(p->cgroups->dfl_cgrp);
		}
	}

	if (!cgrp)
		scx_bpf_error("Failed to get cgroup for task %d", p->pid);

	return cgrp;
}

struct {
	__uint(type, BPF_MAP_TYPE_TASK_STORAGE);
	__uint(map_flags, BPF_F_NO_PREALLOC);
	__type(key, int);
	__type(value, struct task_ctx);
} task_ctxs		       SEC(".maps");

static inline struct task_ctx *lookup_task_ctx(struct task_struct *p)
{
	struct task_ctx *tctx;

	if ((tctx = bpf_task_storage_get(&task_ctxs, p, 0, 0))) {
		return tctx;
	}

	scx_bpf_error("task_ctx lookup failed");
	return NULL;
}

struct {
	__uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
	__type(key, u32);
	__type(value, struct cpu_ctx);
	__uint(max_entries, 1);
} cpu_ctxs		      SEC(".maps");

static inline struct cpu_ctx *lookup_cpu_ctx(int cpu)
{
	struct cpu_ctx *cctx;
	u32		zero = 0;

	if (cpu < 0)
		cctx = bpf_map_lookup_elem(&cpu_ctxs, &zero);
	else
		cctx = bpf_map_lookup_percpu_elem(&cpu_ctxs, &zero, cpu);

	if (!cctx) {
		scx_bpf_error("no cpu_ctx for cpu %d", cpu);
		return NULL;
	}

	return cctx;
}

/* Record debug events to the circular buffer. */
static inline void record_cgroup_init(u64 cgid)
{
	struct debug_event *event;
	u32		    pos, idx;

	if (likely(!debug_events_enabled))
		return;

	pos   = __sync_fetch_and_add(&debug_event_pos, 1);
	idx   = pos % DEBUG_EVENTS_BUF_SIZE;

	event = bpf_map_lookup_elem(&debug_events, &idx);
	if (unlikely(!event))
		return;

	event->timestamp	= scx_bpf_now();
	event->event_type	= DEBUG_EVENT_CGROUP_INIT;
	event->cgroup_init.cgid = cgid;
}

static inline void record_init_task(u64 cgid, u32 pid)
{
	struct debug_event *event;
	u32		    pos, idx;

	if (likely(!debug_events_enabled))
		return;

	pos   = __sync_fetch_and_add(&debug_event_pos, 1);
	idx   = pos % DEBUG_EVENTS_BUF_SIZE;

	event = bpf_map_lookup_elem(&debug_events, &idx);
	if (unlikely(!event))
		return;

	event->timestamp      = scx_bpf_now();
	event->event_type     = DEBUG_EVENT_INIT_TASK;
	event->init_task.cgid = cgid;
	event->init_task.pid  = pid;
}

/*
 * Store the cpumask for each cell (owned by BPF logic). We need this in an
 * explicit map to allow for these to be kptrs.
 */
struct cell_cpumask_wrapper {
	struct bpf_cpumask __kptr *cpumask;
	/*
	 * To avoid allocation on the reconfiguration path, have a second cpumask we
	 * can just do an xchg on.
	 */
	struct bpf_cpumask __kptr *tmp_cpumask;
	/* Borrowable cpumask: CPUs this cell can borrow from other cells */
	struct bpf_cpumask __kptr *borrowable_cpumask;
	struct bpf_cpumask __kptr *borrowable_tmp_cpumask;
};

struct {
	__uint(type, BPF_MAP_TYPE_ARRAY);
	__type(key, u32);
	__type(value, struct cell_cpumask_wrapper);
	__uint(max_entries, MAX_CELLS);
	__uint(map_flags, 0);
} cell_cpumasks			    SEC(".maps");

static inline const struct cpumask *lookup_cell_cpumask(int idx)
{
	struct cell_cpumask_wrapper *cpumaskw;

	if (!(cpumaskw = bpf_map_lookup_elem(&cell_cpumasks, &idx))) {
		scx_bpf_error("no cell cpumask");
		return NULL;
	}

	if (!cpumaskw->cpumask) {
		scx_bpf_error("cell cpumask is NULL");
		return NULL;
	}

	return (const struct cpumask *)cpumaskw->cpumask;
}

static inline const struct cpumask *lookup_cell_borrowable_cpumask(int idx)
{
	struct cell_cpumask_wrapper *cpumaskw;

	if (!(cpumaskw = bpf_map_lookup_elem(&cell_cpumasks, &idx))) {
		scx_bpf_error("no cell cpumask wrapper for cell %d", idx);
		return NULL;
	}

	return (const struct cpumask *)cpumaskw->borrowable_cpumask;
}

/* Determine whether the task can use the cell-wide shared DSQ path. */
static __always_inline bool
task_can_use_shared_cell_domain(struct task_struct *p, u32 cell_id)
{
	const struct cpumask *cell_cpumask;

	cell_cpumask = lookup_cell_cpumask(cell_id);
	if (!cell_cpumask)
		return false;

	if (!bpf_cpumask_subset(cell_cpumask, p->cpus_ptr))
		return false;

	if (enable_borrowing) {
		const struct cpumask *borrowable_cpumask;

		borrowable_cpumask = lookup_cell_borrowable_cpumask(cell_id);
		if (!borrowable_cpumask)
			return false;

		if (!bpf_cpumask_subset(borrowable_cpumask, p->cpus_ptr))
			return false;
	}

	return true;
}

/* Build the task's current primary eligibility mask from cell ownership. */
static __always_inline int task_primary_mask(struct task_struct *p,
					     struct task_ctx	*tctx,
					     struct bpf_cpumask *dst)
{
	const struct cpumask *cell_cpumask;

	cell_cpumask = lookup_cell_cpumask(tctx->cell);
	if (!cell_cpumask)
		return -ENOENT;

	bpf_cpumask_and(dst, cell_cpumask, p->cpus_ptr);
	return 0;
}

/* Build the task's current borrowable eligibility mask from cell borrowing state. */
static __always_inline int task_borrowable_mask(struct task_struct *p,
						struct task_ctx	   *tctx,
						struct bpf_cpumask *dst)
{
	const struct cpumask *borrowable_cpumask;

	borrowable_cpumask = lookup_cell_borrowable_cpumask(tctx->cell);
	if (!borrowable_cpumask)
		return -ENOENT;

	bpf_cpumask_and(dst, borrowable_cpumask, p->cpus_ptr);
	return 0;
}

/*
 * Helper functions for bumping per-cell stats
 */
static void cstat_add(enum cell_stat_idx idx, u32 cell, struct cpu_ctx *cctx,
		      s64 delta)
{
	u64 *vptr;

	if ((vptr = MEMBER_VPTR(*cctx, .cstats[cell][idx])))
		(*vptr) += delta;
	else
		scx_bpf_error("invalid cell or stat idxs: %d, %d", idx, cell);
}

static void cstat_inc(enum cell_stat_idx idx, u32 cell, struct cpu_ctx *cctx)
{
	cstat_add(idx, cell, cctx, 1);
}

/* Reject unsupported multi-CPU pinning within non-root cells. */
static __always_inline int validate_task_affinity(struct task_struct *p,
						  struct task_ctx    *tctx,
						  bool shared_cell_domain)
{
	if (tctx->cell != 0 && reject_multicpu_pinning && !shared_cell_domain &&
	    bpf_cpumask_weight(p->cpus_ptr) > 1) {
		scx_bpf_error("multi-CPU pinning within cell %d not supported",
			      tctx->cell);
		return -EINVAL;
	}

	return 0;
}

/* Route a constrained task to its per-CPU DSQ and baseline vtime. */
static __always_inline int route_task_to_cpu_dsq(struct task_struct *p,
						 struct task_ctx    *tctx)
{
	struct cpu_ctx *cpu_ctx;
	u32		cpu;

	cpu = bpf_cpumask_any_distribute(p->cpus_ptr);
	if (!(cpu_ctx = lookup_cpu_ctx(cpu)))
		return -ENOENT;

	tctx->dsq = get_cpu_dsq_id(cpu);
	if (dsq_is_invalid(tctx->dsq))
		return -EINVAL;

	p->scx.dsq_vtime = READ_ONCE(cpu_ctx->vtime_now);
	return 0;
}

/* Route a shared-cell task to its cell DSQ and baseline vtime. */
static __always_inline int route_task_to_cell_dsq(struct task_struct *p,
						  struct task_ctx    *tctx)
{
	struct bpf_cpumask *primary_mask __free(bpf_cpumask) = NULL;
	struct cell			*cell;

	if (enable_llc_awareness) {
		primary_mask = bpf_cpumask_create();
		if (!primary_mask)
			return -ENOMEM;
		if (task_primary_mask(p, tctx, primary_mask))
			return -ENOENT;
		return update_task_llc_assignment(
			p, tctx, (const struct cpumask *)primary_mask);
	}

	tctx->dsq = get_cell_llc_dsq_id(tctx->cell, FAKE_FLAT_CELL_LLC);
	if (dsq_is_invalid(tctx->dsq))
		return -EINVAL;

	if (!(cell = lookup_cell(tctx->cell)))
		return -ENOENT;

	p->scx.dsq_vtime = READ_ONCE(cell->llcs[FAKE_FLAT_CELL_LLC].vtime_now);
	return 0;
}

/* Pick a fallback CPU from the task's current primary mask. */
static __always_inline s32 task_primary_fallback_cpu(struct task_struct *p,
						     s32	      prev_cpu,
						     struct task_ctx *tctx)
{
	struct bpf_cpumask *fallback_mask __free(bpf_cpumask) =
		bpf_cpumask_create();

	if (!fallback_mask) {
		scx_bpf_error("Failed to allocate task fallback cpumask");
		return -1;
	}
	if (task_primary_mask(p, tctx, fallback_mask))
		return -1;

	if (!bpf_cpumask_test_cpu(prev_cpu,
				  (const struct cpumask *)fallback_mask))
		return bpf_cpumask_any_distribute(
			(const struct cpumask *)fallback_mask);

	return prev_cpu;
}

/* Read the task's vtime baseline from its active dispatch domain. */
static __always_inline int task_basis_vtime(struct task_ctx *tctx, s32 cpu,
					    bool shared_cell_domain,
					    u64 *basis_vtime)
{
	struct cpu_ctx *cpu_ctx;
	struct cell    *cell;

	if (shared_cell_domain) {
		if (!(cell = lookup_cell(tctx->cell)))
			return -ENOENT;

		if (enable_llc_awareness) {
			if (!llc_is_valid(tctx->llc)) {
				scx_bpf_error("Invalid LLC ID: %d", tctx->llc);
				return -EINVAL;
			}

			*basis_vtime =
				READ_ONCE(cell->llcs[tctx->llc].vtime_now);
		} else {
			*basis_vtime = READ_ONCE(
				cell->llcs[FAKE_FLAT_CELL_LLC].vtime_now);
		}

		return 0;
	}

	if (!(cpu_ctx = lookup_cpu_ctx(cpu)))
		return -ENOENT;

	*basis_vtime = READ_ONCE(cpu_ctx->vtime_now);
	return 0;
}

/* Refresh the task's DSQ and vtime baseline from current cell state. */
static inline int update_task_routing(struct task_struct *p,
				      struct task_ctx	 *tctx)
{
	bool shared_cell_domain;

	shared_cell_domain = task_can_use_shared_cell_domain(p, tctx->cell);

	/*
	* Single-CPU pinning is fine (even if outside this cell).
	* However, multi-CPU pinning that doesn't cover the entire
	* cell is not supported - the scheduler can't efficiently
	 * handle partial affinity restrictions.
	 */
	if (validate_task_affinity(p, tctx, shared_cell_domain))
		return -EINVAL;

	/*
	 * XXX - To be correct, we'd need to calculate the vtime
	 * delta in the previous dsq, scale it by the load
	 * fraction difference and then offset from the new
	 * dsq's vtime_now. For now, just do the simple thing
	 * and assume the offset to be zero.
	 *
	 * Revisit if high frequency dynamic cell switching
	 * needs to be supported.
	 */

	/* Per-CPU pinned path */
	if (!shared_cell_domain)
		return route_task_to_cpu_dsq(p, tctx);

	return route_task_to_cell_dsq(p, tctx);
}

/* Resolve the task's current cgroup ownership into a live cell id. */
static inline int resolve_task_cell(struct task_struct *p,
				    struct task_ctx *tctx, struct cgroup *cg,
				    u32 *cell, u64 *cgid)
{
	struct cgrp_ctx *cgc;

	cgc = lookup_cgrp_ctx_fallible(cg);

	if (!cgc) {
		/*
		 * Cgroup lookup failed - this can happen during scheduler load
		 * for tasks that were forked before the scheduler was loaded,
		 * whose cgroups went offline before scx_cgroup_init() ran.
		 * Only fall back to root cgroup if the workaround is enabled
		 * and the task is exiting.
		 */
		if (exiting_task_workaround_enabled &&
		    (p->flags & PF_EXITING)) {
			struct cgroup *rootcg = READ_ONCE(root_cgrp);
			if (!rootcg) {
				scx_bpf_error(
					"Unexpected uninitialized rootcg");
				return -ENOENT;
			}

			cgc = lookup_cgrp_ctx(rootcg);
		}

		if (!cgc) {
			scx_bpf_error(
				"cgrp_ctx lookup failed for cgid %llu (task %d, flags 0x%x, tctx->cgid %llu)",
				cg->kn->id, p->pid, p->flags, tctx->cgid);
			return -ENOENT;
		}
	}

	*cell = cgc->cell;
	*cgid = cg->kn->id;
	return 0;
}

/* Figure out the task's cell and refresh its routing state. */
static inline int update_task_cell(struct task_struct *p, struct task_ctx *tctx,
				   struct cgroup *cg)
{
	u32 cell;
	u64 cgid;

	if (resolve_task_cell(p, tctx, cg, &cell, &cgid))
		return -ENOENT;

	tctx->cell = cell;
	tctx->cgid = cgid;

	/* Task ownership changed, so rebuild its DSQ routing from live cell state. */
	return update_task_routing(p, tctx);
}

/* Helper function for picking an idle cpu out of a candidate set */
static s32 pick_idle_cpu_from(struct task_struct   *p,
			      const struct cpumask *cand_cpumask, s32 prev_cpu,
			      const struct cpumask *idle_smtmask)
{
	bool prev_in_cand = bpf_cpumask_test_cpu(prev_cpu, cand_cpumask);
	s32  cpu;

	/*
	 * If CPU has SMT, any wholly idle CPU is likely a better pick than
	 * partially idle @prev_cpu.
	 */
	if (smt_enabled) {
		if (prev_in_cand &&
		    bpf_cpumask_test_cpu(prev_cpu, idle_smtmask) &&
		    scx_bpf_test_and_clear_cpu_idle(prev_cpu))
			return prev_cpu;

		cpu = scx_bpf_pick_idle_cpu(cand_cpumask, SCX_PICK_IDLE_CORE);
		if (cpu >= 0)
			return cpu;
	}

	if (prev_in_cand && scx_bpf_test_and_clear_cpu_idle(prev_cpu))
		return prev_cpu;

	return scx_bpf_pick_idle_cpu(cand_cpumask, 0);
}

/* Refresh task cell ownership when the active cgroup state may have changed. */
static __always_inline int maybe_refresh_cell(struct task_struct *p,
					      struct task_ctx	 *tctx)
{
	struct cgroup *cgrp __free(cgroup) = task_cgroup(p);
	u32		    cell;
	u64		    cgid;

	if (!cgrp)
		return -1;

	/* Userspace owns cell assignment, so hot paths re-read the live cgroup cell. */
	if (resolve_task_cell(p, tctx, cgrp, &cell, &cgid))
		return -1;

	if (cell != tctx->cell || cgid != tctx->cgid) {
		tctx->cell = cell;
		tctx->cgid = cgid;
		return update_task_routing(p, tctx);
	}

	return 0;
}

static __always_inline s32 pick_idle_cpu(struct task_struct *p, s32 prev_cpu,
					 struct cpu_ctx	 *cctx,
					 struct task_ctx *tctx)
{
	struct bpf_cpumask *task_cpumask __free(bpf_cpumask) =
		bpf_cpumask_create();

	if (!task_cpumask) {
		scx_bpf_error("Failed to allocate task primary cpumask");
		return -1;
	}

	if (task_primary_mask(p, tctx, task_cpumask)) {
		scx_bpf_error("Failed to derive task primary cpumask");
		return -1;
	}

	const struct cpumask *idle_smtmask __free(idle_cpumask) =
		scx_bpf_get_idle_smtmask();
	if (!idle_smtmask) {
		scx_bpf_error("Failed to get idle smtmask");
		return -1;
	}

	/* No overlap between cell cpus and task cpus, just find some idle cpu */
	if (bpf_cpumask_empty((const struct cpumask *)task_cpumask)) {
		cstat_inc(CSTAT_AFFN_VIOL, tctx->cell, cctx);
		return pick_idle_cpu_from(p, p->cpus_ptr, prev_cpu,
					  idle_smtmask);
	}

	return pick_idle_cpu_from(p, (const struct cpumask *)task_cpumask,
				  prev_cpu, idle_smtmask);
}

/* Insert the task directly onto an idle CPU and optionally kick it. */
static __always_inline s32 dispatch_to_idle_cpu(struct task_struct *p, s32 cpu,
						bool kick)
{
	/*
	 * Use SCX_DSQ_LOCAL_ON to target the specific idle CPU we picked.
	 * From enqueue(), plain SCX_DSQ_LOCAL would resolve to task_rq(p)
	 * instead of that idle CPU.
	 */
	scx_bpf_dsq_insert(p, SCX_DSQ_LOCAL_ON | cpu, slice_ns, 0);
	if (kick)
		scx_bpf_kick_cpu(cpu, SCX_KICK_IDLE);
	return cpu;
}

/* Try to find an idle CPU from the task's borrowable mask. */
static __always_inline s32 pick_idle_borrowed_cpu(struct task_struct *p,
						  s32		      prev_cpu,
						  struct task_ctx    *tctx)
{
	struct bpf_cpumask *borrowable __free(bpf_cpumask) =
		bpf_cpumask_create();
	const struct cpumask *idle_smtmask __free(idle_cpumask) =
		scx_bpf_get_idle_smtmask();

	if (!borrowable) {
		scx_bpf_error("Failed to allocate task borrowable cpumask");
		return -1;
	}
	if (task_borrowable_mask(p, tctx, borrowable)) {
		scx_bpf_error("Failed to derive task borrowable cpumask");
		return -1;
	}
	if (!idle_smtmask) {
		scx_bpf_error("Failed to get idle smtmask");
		return -1;
	}

	return pick_idle_cpu_from(p, (const struct cpumask *)borrowable,
				  prev_cpu, idle_smtmask);
}

/*
 * Try to find an idle CPU for a task. First searches within the cell's
 * own CPUs, then tries borrowing from other cells if enabled.
 *
 * On success, bumps CSTAT_LOCAL or CSTAT_BORROWED as appropriate and
 * dispatches the task to SCX_DSQ_LOCAL. If @kick is true, the idle CPU
 * is also kicked.
 *
 * Returns: CPU number >= 0 on success, -1 on error, -EBUSY if no idle CPU found.
 */
static __always_inline s32 try_pick_idle_cpu(struct task_struct *p,
					     s32 prev_cpu, struct cpu_ctx *cctx,
					     struct task_ctx *tctx, bool kick)
{
	s32 cpu;

	cpu = pick_idle_cpu(p, prev_cpu, cctx, tctx);
	if (cpu >= 0) {
		cstat_inc(CSTAT_LOCAL, tctx->cell, cctx);
		return dispatch_to_idle_cpu(p, cpu, kick);
	}
	if (cpu == -1)
		return -1; /* error from pick_idle_cpu, propagate */

	/* cpu == -EBUSY: no idle CPU in cell, try borrowing */
	if (enable_borrowing) {
		cpu = pick_idle_borrowed_cpu(p, prev_cpu, tctx);
		if (cpu >= 0) {
			tctx->borrowed = true;
			cstat_inc(CSTAT_BORROWED, tctx->cell, cctx);
			return dispatch_to_idle_cpu(p, cpu, kick);
		}
	}

	return -EBUSY;
}

/*
 * select_cpu is where we update each task's cell assignment and then try to
 * dispatch to an idle core in the cell if possible
 */
s32 BPF_STRUCT_OPS(mitosis_select_cpu, struct task_struct *p, s32 prev_cpu,
		   u64 wake_flags)
{
	s32		 cpu;
	struct cpu_ctx	*cctx;
	struct task_ctx *tctx;
	bool		 shared_cell_domain;

	if (!(cctx = lookup_cpu_ctx(-1)) || !(tctx = lookup_task_ctx(p)))
		return prev_cpu;

	if (maybe_refresh_cell(p, tctx) < 0)
		return prev_cpu;

	/* Recheck against the live cell masks in case userspace just reconfigured. */
	shared_cell_domain = task_can_use_shared_cell_domain(p, tctx->cell);

	if (!shared_cell_domain) {
		cstat_inc(CSTAT_AFFN_VIOL, tctx->cell, cctx);
		cpu = get_cpu_from_dsq(tctx->dsq);
		if (cpu < 0)
			return prev_cpu;

		if (scx_bpf_test_and_clear_cpu_idle(cpu))
			scx_bpf_dsq_insert(p, SCX_DSQ_LOCAL, slice_ns, 0);
		return cpu;
	}

	if ((cpu = try_pick_idle_cpu(p, prev_cpu, cctx, tctx, false)) >= 0)
		return cpu;

	/* No idle CPU was available, so fall back to the current primary mask. */
	cpu = task_primary_fallback_cpu(p, prev_cpu, tctx);
	if (cpu < 0)
		return prev_cpu;

	return cpu;
}

void BPF_STRUCT_OPS(mitosis_enqueue, struct task_struct *p, u64 enq_flags)
{
	struct cpu_ctx	*cctx;
	struct task_ctx *tctx;
	s32		 task_cpu = scx_bpf_task_cpu(p);
	u64		 vtime;
	s32		 cpu = -1;
	u64		 basis_vtime;
	bool		 shared_cell_domain;

	if (!(tctx = lookup_task_ctx(p)) || !(cctx = lookup_cpu_ctx(-1)))
		return;

	if (maybe_refresh_cell(p, tctx) < 0)
		return;

	/* Ensure this is done *AFTER* refreshing cell which might manipulate vtime */
	vtime = p->scx.dsq_vtime;

	/* Recheck against the live cell masks before choosing the queueing path. */
	shared_cell_domain = task_can_use_shared_cell_domain(p, tctx->cell);

	if (!shared_cell_domain) {
		cpu = get_cpu_from_dsq(tctx->dsq);
		if (cpu < 0)
			return;
	} else if (!__COMPAT_is_enq_cpu_selected(enq_flags)) {
		/*
		 * If we haven't selected a cpu, then we haven't looked for and kicked an
		 * idle CPU. Let's do the lookup now.
		 */
		if (!(cctx = lookup_cpu_ctx(-1)))
			return;
		cpu = try_pick_idle_cpu(p, task_cpu, cctx, tctx, true);
		if (cpu >= 0)
			return;
		if (cpu == -1)
			return;
		if (cpu == -EBUSY) {
			/* No idle CPU was available, so fall back to the current primary mask. */
			cpu = task_primary_fallback_cpu(p, task_cpu, tctx);
			if (cpu < 0)
				return;
		}
	}

	if (shared_cell_domain)
		cstat_inc(CSTAT_CELL_DSQ, tctx->cell, cctx);
	else
		cstat_inc(CSTAT_CPU_DSQ, tctx->cell, cctx);

	/* Read the baseline from the queueing domain we just selected. */
	if (task_basis_vtime(tctx, cpu, shared_cell_domain, &basis_vtime))
		return;

	tctx->basis_vtime = basis_vtime;

	if (time_after(vtime, basis_vtime + 8192 * slice_ns)) {
		scx_bpf_error(
			"vtime too far ahead: pid=%d vtime=%llu basis=%llu diff=%llu cell=%u",
			p->pid, p->scx.dsq_vtime, basis_vtime,
			p->scx.dsq_vtime - basis_vtime, tctx->cell);
		return;
	}
	/*
	 * Limit the amount of budget that an idling task can accumulate
	 * to one slice.
	 */
	if (time_before(vtime, basis_vtime - slice_ns))
		vtime = basis_vtime - slice_ns;

	scx_bpf_dsq_insert_vtime(p, tctx->dsq.raw, slice_ns, vtime, enq_flags);

	/* Kick the CPU if needed */
	if (!__COMPAT_is_enq_cpu_selected(enq_flags) && cpu >= 0)
		scx_bpf_kick_cpu(cpu, SCX_KICK_IDLE);
}

void BPF_STRUCT_OPS(mitosis_dispatch, s32 cpu, struct task_struct *prev)
{
	struct cpu_ctx *cctx;
	u32		cell;

	if (!(cctx = lookup_cpu_ctx(-1)))
		return;

	cell				  = READ_ONCE(cctx->cell);

	bool		    found	  = false;
	dsq_id_t	    min_vtime_dsq = DSQ_INVALID;
	u64		    min_vtime	  = 0;

	struct task_struct *p;

	/* Check the cell-LLC DSQ (use FAKE_FLAT_CELL_LLC when not LLC-aware) */
	u32	 llc = enable_llc_awareness ? cctx->llc : FAKE_FLAT_CELL_LLC;
	dsq_id_t cell_dsq = get_cell_llc_dsq_id(cell, llc);
	dsq_id_t cpu_dsq  = get_cpu_dsq_id(cpu);

	if (dsq_is_invalid(cell_dsq) || dsq_is_invalid(cpu_dsq)) {
		return;
	}

	/* Peek at cell-LLC DSQ head */
	p = __COMPAT_scx_bpf_dsq_peek(cell_dsq.raw);
	if (p) {
		min_vtime     = p->scx.dsq_vtime;
		min_vtime_dsq = cell_dsq;
		found	      = true;
	}

	/* Peek at CPU DSQ head, prefer if lower vtime */
	p = __COMPAT_scx_bpf_dsq_peek(cpu_dsq.raw);
	if (p && (!found || time_before(p->scx.dsq_vtime, min_vtime))) {
		min_vtime     = p->scx.dsq_vtime;
		min_vtime_dsq = cpu_dsq;
		found	      = true;
	}

	/*
	 * If we failed to find an eligible task, try work stealing if enabled.
	 * Otherwise, scx will keep running prev if prev->scx.flags &
	 * SCX_TASK_QUEUED (we don't set SCX_OPS_ENQ_LAST), and otherwise go idle.
	 */
	if (!found) {
		/* Try work stealing if enabled */
		if (enable_llc_awareness && enable_work_stealing) {
			/* Returns: <0 error, 0 no steal, >0 stole work */
			s32 ret = try_stealing_work(cell, llc);
			if (ret < 0)
				return;
			if (ret > 0) {
				cstat_inc(CSTAT_STEAL, cell, cctx);
			}
		}
		return;
	}

	/*
	 * The move_to_local can fail if we raced with some other cpu in the cell
	 * and now the cell is empty. We have to ensure to try the cpu_dsq or else
	 * we might never wakeup.
	 */

	/* Try the winner first */
	if (scx_bpf_dsq_move_to_local(min_vtime_dsq.raw, 0))
		return;

	/* Winner was cell DSQ but failed - try the CPU DSQ */
	if (min_vtime_dsq.raw == cell_dsq.raw)
		scx_bpf_dsq_move_to_local(cpu_dsq.raw, 0);
}

/*
 * This array keeps track of the cgroup ancestor's cell as we iterate over the
 * cgroup hierarchy.
 */
u32		   level_cells[MAX_CG_DEPTH];

static inline void advance_cell_llc_vtime(struct cell	  *cell,
					  struct task_ctx *tctx, u64 task_vtime)
{
	u32 llc_idx = enable_llc_awareness && llc_is_valid(tctx->llc) ?
			      tctx->llc :
			      FAKE_FLAT_CELL_LLC;

	if (time_before(READ_ONCE(cell->llcs[llc_idx].vtime_now), task_vtime))
		WRITE_ONCE(cell->llcs[llc_idx].vtime_now, task_vtime);
}

void BPF_STRUCT_OPS(mitosis_running, struct task_struct *p)
{
	struct cpu_ctx	*cctx;
	struct task_ctx *tctx;

	if (!(cctx = lookup_cpu_ctx(-1)) || !(tctx = lookup_task_ctx(p)))
		return;

	/* Handle stolen task retag (LLC-aware mode only) */
	if (enable_llc_awareness && enable_work_stealing) {
		if (maybe_retag_stolen_task(p, tctx, cctx) < 0)
			return;
	}

	tctx->started_running_at = scx_bpf_now();
}

void BPF_STRUCT_OPS(mitosis_stopping, struct task_struct *p, bool runnable)
{
	struct cpu_ctx	*cctx;
	struct task_ctx *tctx;
	struct cell	*cell;
	u64		 now, used;
	u32		 cidx;

	if (!(cctx = lookup_cpu_ctx(-1)) || !(tctx = lookup_task_ctx(p)))
		return;

	/*
	 * Use CPU's cell (not task's cell) to match dispatch() logic.
	 * Prevents starvation when a task is pinned outside its cell.
	 * E.g. a cell 0 kworker pinned to a cell 1 CPU.
	 */
	cidx = cctx->cell;
	if (!(cell = lookup_cell(cidx)))
		return;

	now			 = scx_bpf_now();
	used			 = now - tctx->started_running_at;
	tctx->started_running_at = now;
	/* scale the execution time by the inverse of the weight and charge */
	if (p->scx.weight == 0) {
		scx_bpf_error("Task %d has zero weight", p->pid);
		return;
	}
	p->scx.dsq_vtime += used * 100 / p->scx.weight;

	/*
	 * Advance this CPU's per-CPU DSQ vtime, UNLESS the task was
	 * genuinely borrowed from another cell. Borrowed tasks' vtime
	 * is in the borrowing cell's domain — writing it to the lending
	 * CPU's vtime_now would contaminate that domain.
	 *
	 * For cell-reassigned tasks (tctx->cell != cidx but not borrowed),
	 * the vtime was initialized in this CPU's cell domain, so
	 * advancing cctx->vtime_now is correct and prevents staleness.
	 */
	if (!tctx->borrowed) {
		if (time_before(READ_ONCE(cctx->vtime_now), p->scx.dsq_vtime))
			WRITE_ONCE(cctx->vtime_now, p->scx.dsq_vtime);
	}

	/* Clear the borrowed flag — it is one-shot, consumed above */
	tctx->borrowed = false;

	struct cell *vtime_cell;
	if (tctx->cell != cidx) {
		vtime_cell = lookup_cell(tctx->cell);
		if (!vtime_cell)
			return;
	} else {
		vtime_cell = cell;
	}
	advance_cell_llc_vtime(vtime_cell, tctx, p->scx.dsq_vtime);

	{
		u64 *running = MEMBER_VPTR(cctx->running_ns, [tctx->cell]);
		if (!running) {
			scx_bpf_error("Task cell index too large: %d",
				      tctx->cell);
			return;
		}
		*running += used;
	}
}

SEC("fentry/cpuset_write_resmask")
int BPF_PROG(fentry_cpuset_write_resmask, struct kernfs_open_file *of,
	     char *buf, size_t nbytes, loff_t off, ssize_t retval)
{
	/* Cpuset writes are handled in userspace and applied back through apply_cell_config(). */
	__atomic_add_fetch(&cpuset_seq, 1, __ATOMIC_RELEASE);
	return 0;
}

/* From linux/percpu-refcount.h */
#define __PERCPU_REF_DEAD (1LU << 1)

/*
 * Check if a cgroup is dying (being destroyed).
 */
static bool cgrp_is_dying(struct cgroup *cgrp)
{
	unsigned long refcnt_ptr;
	bpf_core_read(&refcnt_ptr, sizeof(refcnt_ptr),
		      &cgrp->self.refcnt.percpu_count_ptr);
	return refcnt_ptr & __PERCPU_REF_DEAD;
}

/* Create cgrp_ctx and inherit the current parent cell. */
static int init_cgrp_ctx(struct cgroup *cgrp)
{
	struct cgrp_ctx *cgc;

	record_cgroup_init(cgrp->kn->id);

	if (!(cgc = bpf_cgrp_storage_get(&cgrp_ctxs, cgrp, 0,
					 BPF_LOCAL_STORAGE_GET_F_CREATE))) {
		scx_bpf_error("cgrp_ctx creation failed for cgid %llu",
			      cgrp->kn->id);
		return -ENOENT;
	}

	if (cgrp->kn->id == root_cgid) {
		WRITE_ONCE(cgc->cell, 0);
		return 0;
	}

	/* Initialize to parent's cell */
	struct cgroup *parent_cg __free(cgroup) =
		lookup_cgrp_ancestor(cgrp, cgrp->level - 1);
	if (!parent_cg)
		return -ENOENT;

	struct cgrp_ctx *parent_cgc;
	if (!(parent_cgc = lookup_cgrp_ctx(parent_cg)))
		return -ENOENT;

	cgc->cell = parent_cgc->cell;
	return 0;
}

/*
 * Initialize cgroup and all its ancestors. Handles dying cgroups gracefully.
 * Used when CPU controller is disabled since SCX cgroup callbacks won't fire.
 */
static int init_cgrp_ctx_with_ancestors(struct cgroup *cgrp)
{
	u32 target_level = cgrp->level;
	u32 level;
	int ret;

	/* Skip dying cgroups */
	if (cgrp_is_dying(cgrp))
		return 0;

	/* Initialize ancestors first (replicates SCX cgroup_init order) */
	bpf_for(level, 1, target_level)
	{
		struct cgroup *ancestor __free(cgroup) =
			lookup_cgrp_ancestor(cgrp, level);
		if (!ancestor)
			return -ENOENT;

		/* Skip if dying or already initialized */
		if (!cgrp_is_dying(ancestor) &&
		    !lookup_cgrp_ctx_fallible(ancestor)) {
			ret = init_cgrp_ctx(ancestor);
			if (ret)
				return ret;
		}
	}

	/* Skip if already initialized */
	if (lookup_cgrp_ctx_fallible(cgrp))
		return 0;

	return init_cgrp_ctx(cgrp);
}

/*
 * SCX cgroup callbacks - called by the SCX framework when the CPU controller
 * is enabled.
 */
s32 BPF_STRUCT_OPS(mitosis_cgroup_init, struct cgroup *cgrp,
		   struct scx_cgroup_init_args *args)
{
	if (cpu_controller_disabled)
		return 0;
	return init_cgrp_ctx(cgrp);
}

s32 BPF_STRUCT_OPS(mitosis_cgroup_exit, struct cgroup *cgrp)
{
	if (cpu_controller_disabled)
		return 0;

	/* Userspace tears down cell ownership and republishes config on cgroup removal. */
	return 0;
}

void BPF_STRUCT_OPS(mitosis_cgroup_move, struct task_struct *p,
		    struct cgroup *from, struct cgroup *to)
{
	struct task_ctx *tctx;

	if (cpu_controller_disabled)
		return;

	if (!(tctx = lookup_task_ctx(p)))
		return;

	update_task_cell(p, tctx, to);
}

/*
 * Tracepoint fallbacks - only active when CPU controller is disabled.
 * These provide cgroup tracking when SCX cgroup callbacks don't fire.
 */
SEC("tp_btf/cgroup_mkdir")
int BPF_PROG(tp_cgroup_mkdir, struct cgroup *cgrp, const char *cgrp_path)
{
	int ret;
	if (!cpu_controller_disabled)
		return 0;

	ret = init_cgrp_ctx_with_ancestors(cgrp);
	if (ret) {
		scx_bpf_error(
			"tp_cgroup_mkdir: init_cgrp_ctx_with_ancestors failed for cgid %llu: %d",
			cgrp->kn->id, ret);
	}
	return 0;
}

SEC("tp_btf/cgroup_rmdir")
int BPF_PROG(tp_cgroup_rmdir, struct cgroup *cgrp, const char *cgrp_path)
{
	if (!cpu_controller_disabled)
		return 0;

	/* Userspace handles cell teardown; the tracepoint only remains for symmetry with mkdir. */
	return 0;
}

void BPF_STRUCT_OPS(mitosis_set_cpumask, struct task_struct *p,
		    const struct cpumask *cpumask)
{
	struct task_ctx *tctx;

	if (!(tctx = lookup_task_ctx(p)))
		return;

	/* Affinity changes can move the task between constrained and shared routing. */
	update_task_routing(p, tctx);
}

s32 validate_flags()
{
	/* Need valid llc */
	if (enable_llc_awareness && (nr_llc < 1 || nr_llc > MAX_LLCS)) {
		scx_bpf_error(
			"LLC-aware mode requires nr_llc between 1 and %d inclusive, got %d",
			MAX_LLCS, nr_llc);
		return -EINVAL;
	}

	/* Work stealing only makes sense when enable_llc_awareness. */
	if (enable_work_stealing && (!enable_llc_awareness)) {
		scx_bpf_error(
			"Work stealing requires LLC-aware mode to be enabled");
		return -EINVAL;
	}

	return 0;
}

s32 validate_userspace_data()
{
	if (nr_possible_cpus > MAX_CPUS) {
		scx_bpf_error("nr_possible_cpus %d exceeds MAX_CPUS %d",
			      nr_possible_cpus, MAX_CPUS);
		return -EINVAL;
	}
	return 0;
}

static int init_task_impl(struct task_struct *p, struct cgroup *cgrp)
{
	struct task_ctx *tctx;

	record_init_task(cgrp->kn->id, p->pid);

	tctx = bpf_task_storage_get(&task_ctxs, p, 0,
				    BPF_LOCAL_STORAGE_GET_F_CREATE);
	if (!tctx) {
		scx_bpf_error("task_ctx allocation failure");
		return -ENOMEM;
	}

	/* Initialize LLC assignment fields */
	if (enable_llc_awareness)
		init_task_llc(tctx);

	return update_task_cell(p, tctx, cgrp);
}

s32 BPF_STRUCT_OPS(mitosis_init_task, struct task_struct *p,
		   struct scx_init_task_args *args)
{
	/*
	 * When CPU controller is disabled, args->cgroup is root, so we need
	 * to get the task's actual cgroup for both logging and cell assignment.
	 * We also need to ensure the cgroup hierarchy is initialized since
	 * SCX cgroup callbacks won't fire.
	 */
	if (cpu_controller_disabled) {
		struct cgroup *cgrp __free(cgroup) = task_cgroup(p);
		if (!cgrp)
			return -ENOENT;

		/* Ensure cgroup hierarchy is initialized (handles ancestors + this cgroup) */
		int ret = init_cgrp_ctx_with_ancestors(cgrp);
		if (ret)
			return ret;

		return init_task_impl(p, cgrp);
	}

	return init_task_impl(p, args->cgroup);
}

__hidden void dump_cpumask_word(s32 word, const struct cpumask *cpumask)
{
	u32 u, v = 0;

	bpf_for(u, 0, 32)
	{
		s32 cpu = 32 * word + u;
		if (cpu < nr_possible_cpus &&
		    bpf_cpumask_test_cpu(cpu, cpumask))
			v |= 1 << u;
	}
	scx_bpf_dump("%08x", v);
}

static void dump_cpumask(const struct cpumask *cpumask)
{
	u32 word, nr_words = (nr_possible_cpus + 31) / 32;

	bpf_for(word, 0, nr_words)
	{
		if (word)
			scx_bpf_dump(",");
		dump_cpumask_word(nr_words - word - 1, cpumask);
	}
}

static void dump_cell_cpumask(int id)
{
	const struct cpumask *cell_cpumask;

	if (!(cell_cpumask = lookup_cell_cpumask(id)))
		return;

	dump_cpumask(cell_cpumask);
}

void BPF_STRUCT_OPS(mitosis_dump, struct scx_dump_ctx *dctx)
{
	dsq_id_t	dsq_id;
	int		i;
	struct cell    *cell;
	struct cpu_ctx *cpu_ctx;

	scx_bpf_dump_header();

	bpf_for(i, 0, MAX_CELLS)
	{
		if (!(cell = lookup_cell(i)))
			return;

		if (!cell->in_use)
			continue;

		scx_bpf_dump("CELL[%d] CPUS=", i);
		dump_cell_cpumask(i);
		scx_bpf_dump("\n");
		/* Per-LLC stats deferred: FAKE_FLAT_CELL_LLC used for now */
		dsq_id_t dsq_id = get_cell_llc_dsq_id(i, FAKE_FLAT_CELL_LLC);
		if (dsq_is_invalid(dsq_id))
			return;

		scx_bpf_dump(
			"CELL[%d] vtime=%llu nr_queued=%d\n", i,
			READ_ONCE(cell->llcs[FAKE_FLAT_CELL_LLC].vtime_now),
			scx_bpf_dsq_nr_queued(dsq_id.raw));
	}

	bpf_for(i, 0, nr_possible_cpus)
	{
		if (!(cpu_ctx = lookup_cpu_ctx(i)))
			return;

		dsq_id = get_cpu_dsq_id(i);
		if (dsq_is_invalid(dsq_id))
			return;
		scx_bpf_dump("CPU[%d] cell=%d vtime=%llu nr_queued=%d\n", i,
			     cpu_ctx->cell, READ_ONCE(cpu_ctx->vtime_now),
			     scx_bpf_dsq_nr_queued(dsq_id.raw));
	}

	if (!debug_events_enabled)
		return;

	/* Dump debug events */
	scx_bpf_dump("\n");
	scx_bpf_dump("DEBUG EVENTS (last %d):\n", DEBUG_EVENTS_BUF_SIZE);

	u32 total_events = READ_ONCE(debug_event_pos);
	u32 start_idx	 = total_events > DEBUG_EVENTS_BUF_SIZE ?
				   total_events - DEBUG_EVENTS_BUF_SIZE :
				   0;

	bpf_for(i, 0, DEBUG_EVENTS_BUF_SIZE)
	{
		u32 event_num = start_idx + i;
		if (event_num >= total_events)
			break;

		u32		    idx = event_num % DEBUG_EVENTS_BUF_SIZE;
		struct debug_event *event =
			bpf_map_lookup_elem(&debug_events, &idx);
		if (!event)
			continue;

		switch (event->event_type) {
		case DEBUG_EVENT_CGROUP_INIT:
			scx_bpf_dump("[%3d] CGROUP_INIT cgid=%llu ts=%llu\n",
				     event_num, event->cgroup_init.cgid,
				     event->timestamp);
			break;
		case DEBUG_EVENT_INIT_TASK:
			scx_bpf_dump(
				"[%3d] INIT_TASK   cgid=%llu pid=%u ts=%llu\n",
				event_num, event->init_task.cgid,
				event->init_task.pid, event->timestamp);
			break;
		case DEBUG_EVENT_CGROUP_EXIT:
			scx_bpf_dump("[%3d] CGROUP_EXIT cgid=%llu ts=%llu\n",
				     event_num, event->cgroup_exit.cgid,
				     event->timestamp);
			break;
		default:
			scx_bpf_dump("[%3d] UNKNOWN     type=%u ts=%llu\n",
				     event_num, event->event_type,
				     event->timestamp);
			break;
		}
	}
}

void BPF_STRUCT_OPS(mitosis_dump_task, struct scx_dump_ctx *dctx,
		    struct task_struct *p)
{
	struct task_ctx *tctx;
	bool		 shared_cell_domain;

	if (!(tctx = lookup_task_ctx(p)))
		return;

	shared_cell_domain = task_can_use_shared_cell_domain(p, tctx->cell);

	scx_bpf_dump(
		"Task[%d] vtime=%llu basis_vtime=%llu cell=%u dsq=%llx all_cell_cpus_allowed=%d\n",
		p->pid, p->scx.dsq_vtime, tctx->basis_vtime, tctx->cell,
		tctx->dsq.raw, shared_cell_domain);
	scx_bpf_dump("Task[%d] CPUS=", p->pid);
	dump_cpumask(p->cpus_ptr);
	scx_bpf_dump("\n");
}

s32 BPF_STRUCT_OPS_SLEEPABLE(mitosis_init)
{
	struct bpf_cpumask *cpumask;
	u32		    i;
	s32		    ret;

	/* Sanity check the flags we get from userspace. */
	if ((ret = validate_flags()))
		return ret;

	/* Check data from userspace. */
	if ((ret = validate_userspace_data()))
		return ret;

	if (!userspace_managed_cell_mode) {
		scx_bpf_exit(0,
			     "scx_mitosis requires userspace cell management");
		return -EINVAL;
	}

	struct cgroup *rootcg __free(cgroup) = bpf_cgroup_from_id(root_cgid);
	if (!rootcg)
		return -ENOENT;

	/* Root cgroup storage backs inherited cell assignment from userspace config. */
	if (!bpf_cgrp_storage_get(&cgrp_ctxs, rootcg, 0,
				  BPF_LOCAL_STORAGE_GET_F_CREATE)) {
		scx_bpf_error("cgrp_ctx creation failed for rootcg");
		return -ENOENT;
	}

	struct cgroup *old __free(cgroup) =
		bpf_kptr_xchg(&root_cgrp, no_free_ptr(rootcg));

	/* setup all_cpumask - must be done before cgroup iteration */
	cpumask = bpf_cpumask_create();
	if (!cpumask)
		return -ENOMEM;

	bpf_for(i, 0, nr_possible_cpus)
	{
		const volatile u8 *u8_ptr;

		if ((u8_ptr = MEMBER_VPTR(all_cpus, [i / 8]))) {
			if (*u8_ptr & (1 << (i % 8))) {
				bpf_cpumask_set_cpu(i, cpumask);
				dsq_id_t dsq_id = get_cpu_dsq_id(i);
				if (dsq_is_invalid(dsq_id)) {
					bpf_cpumask_release(cpumask);
					scx_bpf_error(
						"Invalid dsq_id for cpu %d, dsq_id: %llx",
						i, dsq_id.raw);
					return -EINVAL;
				}
				ret = scx_bpf_create_dsq(dsq_id.raw, ANY_NUMA);
				if (ret < 0) {
					bpf_cpumask_release(cpumask);
					scx_bpf_error(
						"Failed to create dsq for cpu %d, dsq_id: %llx, ret: %d",
						i, dsq_id.raw, ret);
					return ret;
				}
			}
		} else {
			return -EINVAL;
		}

		/* Store the LLC that each cpu belongs to. Used in Dispatch. */
		struct cpu_ctx *cpu_ctx = lookup_cpu_ctx(i);
		if (!cpu_ctx) {
			bpf_cpumask_release(cpumask);
			return -EINVAL;
		}

		if (enable_llc_awareness) {
			if (i < MAX_CPUS) // explicit bounds check for verifier
				cpu_ctx->llc = cpu_to_llc[i];
		} else {
			cpu_ctx->llc = FAKE_FLAT_CELL_LLC;
		}
	}

	cpumask = bpf_kptr_xchg(&all_cpumask, cpumask);
	if (cpumask)
		bpf_cpumask_release(cpumask);

	/*
	 * When CPU controller is disabled, initialize cgrp_ctx for all existing
	 * cgroups. This replicates SCX cgroup_init callback behavior - all
	 * cgroups get initialized in hierarchical order during scheduler attach.
	 * The tracepoint handles new cgroups created after attach.
	 */
	if (cpu_controller_disabled) {
		struct cgroup *iter_root __free(cgroup) = NULL;

		scoped_guard(rcu)
		{
			if (root_cgrp)
				iter_root = bpf_cgroup_acquire(root_cgrp);
		}

		if (!iter_root) {
			scx_bpf_error(
				"Failed to acquire root cgroup for initialization");
			return -ENOENT;
		}

		struct cgroup_subsys_state *root_css = &iter_root->self;
		struct cgroup_subsys_state *pos;

		scoped_guard(rcu)
		{
			bpf_for_each(css, pos, root_css,
				     BPF_CGROUP_ITER_DESCENDANTS_PRE) {
				/*
				 * pos->cgroup dereference loses RCU tracking in verifier,
				 * so we can't use it directly with bpf_cgroup_acquire or
				 * pass it to functions that call bpf_cgroup_ancestor.
				 * Instead, read the cgroup ID and use bpf_cgroup_from_id
				 * to get a trusted, acquired reference.
				 */
				u64		    cgid = pos->cgroup->kn->id;
				struct cgroup *cgrp __free(cgroup) =
					bpf_cgroup_from_id(cgid);
				if (cgrp)
					init_cgrp_ctx(cgrp);
			}
		}
	}

	bpf_for(i, 0, MAX_CELLS)
	{
		struct cell_cpumask_wrapper *cpumaskw;

		if (enable_llc_awareness) {
			u32 llc;
			bpf_for(llc, 0, nr_llc)
			{
				dsq_id_t dsq_id = get_cell_llc_dsq_id(i, llc);
				if (dsq_is_invalid(dsq_id))
					return -EINVAL; // scx_bpf_error called in get_cell_llc_dsq_id

				ret = scx_bpf_create_dsq(dsq_id.raw, ANY_NUMA);
				if (ret < 0)
					return ret;
			}
		} else {
			dsq_id_t dsq_id =
				get_cell_llc_dsq_id(i, FAKE_FLAT_CELL_LLC);
			if (dsq_is_invalid(dsq_id))
				return -EINVAL; // scx_bpf_error called in get_cell_llc_dsq_id

			ret = scx_bpf_create_dsq(dsq_id.raw, ANY_NUMA);
			if (ret < 0)
				return ret;
		}

		if (!(cpumaskw = bpf_map_lookup_elem(&cell_cpumasks, &i)))
			return -ENOENT;

		cpumask = bpf_cpumask_create();
		if (!cpumask)
			return -ENOMEM;

		/*
		 * Start with full cpumask for all cells. The timer will set up
		 * the correct cpumasks based on cgroup configuration.
		 */
		bpf_cpumask_setall(cpumask);

		cpumask = bpf_kptr_xchg(&cpumaskw->cpumask, cpumask);
		if (cpumask) {
			/* Should be impossible, we just initialized the cell cpumask */
			bpf_cpumask_release(cpumask);
			return -EINVAL;
		}

		cpumask = bpf_cpumask_create();
		if (!cpumask)
			return -ENOMEM;
		cpumask = bpf_kptr_xchg(&cpumaskw->tmp_cpumask, cpumask);
		if (cpumask) {
			/* Should be impossible, we just initialized the cell tmp_cpumask */
			bpf_cpumask_release(cpumask);
			return -EINVAL;
		}

		if (enable_borrowing) {
			cpumask = bpf_cpumask_create();
			if (!cpumask)
				return -ENOMEM;

			/* Start with empty borrowable masks */
			cpumask = bpf_kptr_xchg(&cpumaskw->borrowable_cpumask,
						cpumask);
			if (cpumask) {
				bpf_cpumask_release(cpumask);
				return -EINVAL;
			}

			cpumask = bpf_cpumask_create();
			if (!cpumask)
				return -ENOMEM;
			cpumask = bpf_kptr_xchg(
				&cpumaskw->borrowable_tmp_cpumask, cpumask);
			if (cpumask) {
				bpf_cpumask_release(cpumask);
				return -EINVAL;
			}
		}
	}

	if (enable_llc_awareness) {
		{
			guard(rcu)();
			if (recalc_cell_llc_counts(ROOT_CELL_ID, NULL))
				return -EINVAL;
		}
	}

	{
		struct cell *cell = lookup_cell(0);
		if (!cell)
			return -ENOENT;

		cell->in_use = true;
	}

	return 0;
}

void BPF_STRUCT_OPS(mitosis_exit, struct scx_exit_info *ei)
{
	UEI_RECORD(uei, ei);
}

/*
 * Apply a complete cell configuration.
 *
 * Configuration data is read from the cell_config global struct,
 * which is populated by userspace before invoking this program.
 *
 * The function operates in ordered phases:
 * 1. Clear old non-root cell ownership
 * 2. Publish primary cell cpumasks and CPU ownership
 * 3. Publish borrowable cell cpumasks
 * 4. Recompute LLC counts for active cells
 * 5. Bind owner cgroups to configured cells
 * 6. Propagate inherited cell ownership
 * 7. Bump applied_configuration_seq to signal completion
 *
 * Note: This is not atomic - tasks may observe intermediate states during
 * execution. On error, the scheduler may be left in a partially-configured
 * state. This is acceptable because userspace treats errors as fatal and
 * exits, causing the scheduler to be unloaded.
 */
/* Read one CPU bit out of a userspace-populated cell cpumask blob. */
static __always_inline bool
config_cpumask_test_cpu(const struct cell_cpumask_data *cpumask_data, u32 cpu,
			int *err)
{
	u32		     byte_idx = cpu / 8;
	const unsigned char *bytep =
		MEMBER_VPTR(cpumask_data->mask, [byte_idx]);

	if (!bytep) {
		scx_bpf_error("byte_idx %d out of bounds", byte_idx);
		*err = -EINVAL;
		return false;
	}

	return *bytep & (1 << (cpu % 8));
}

/* Clear owner state for all non-root cells before applying a new config. */
static __always_inline int reset_non_root_cells_for_config(void)
{
	u32 i;

	bpf_for(i, 1, MAX_CELLS)
	{
		struct cell *cell = lookup_cell(i);
		if (!cell)
			return -EINVAL;

		WRITE_ONCE(cell->in_use, 0);
		cell->owner_cgid = 0;
	}

	return 0;
}

/* Publish a cell's primary cpumask and update per-CPU cell ownership. */
static __always_inline int
publish_primary_cell_state(u32				   cell_id,
			   const struct cell_cpumask_data *cpumask_data)
{
	struct cell		       *cell;
	struct cpu_ctx		       *cctx;
	struct cell_cpumask_wrapper    *cpumaskw;
	struct bpf_cpumask *new_cpumask __free(bpf_cpumask) = NULL;
	int				err		    = 0;
	u32				cpu;

	cpumaskw = bpf_map_lookup_elem(&cell_cpumasks, &cell_id);
	if (!cpumaskw)
		return 0;

	new_cpumask = bpf_kptr_xchg(&cpumaskw->tmp_cpumask, NULL);
	if (!new_cpumask) {
		scx_bpf_error("tmp_cpumask is NULL for cell %d", cell_id);
		return -EINVAL;
	}

	bpf_cpumask_clear(new_cpumask);

	bpf_for(cpu, 0, nr_possible_cpus)
	{
		if (err)
			return err;
		if (!config_cpumask_test_cpu(cpumask_data, cpu, &err)) {
			if (err)
				return err;
			continue;
		}

		bpf_cpumask_set_cpu(cpu, new_cpumask);
		cctx = bpf_map_lookup_percpu_elem(&cpu_ctxs, &(u32){ 0 }, cpu);
		if (!cctx)
			return -ENOENT;

		if (cctx->cell != cell_id) {
			u32 llc_idx;

			cell = lookup_cell(cell_id);
			if (!cell)
				return -ENOENT;

			llc_idx = enable_llc_awareness &&
						  llc_is_valid(cctx->llc) ?
					  cctx->llc :
					  FAKE_FLAT_CELL_LLC;

			if (time_before(READ_ONCE(cell->llcs[llc_idx].vtime_now),
					cctx->vtime_now))
				WRITE_ONCE(cell->llcs[llc_idx].vtime_now,
					   cctx->vtime_now);
		}

		cctx->cell = cell_id;
	}

	new_cpumask =
		bpf_kptr_xchg(&cpumaskw->cpumask, no_free_ptr(new_cpumask));
	if (!new_cpumask) {
		scx_bpf_error("cpumask should never be null");
		return -EINVAL;
	}

	struct bpf_cpumask *stale __free(bpf_cpumask) =
		bpf_kptr_xchg(&cpumaskw->tmp_cpumask, no_free_ptr(new_cpumask));
	if (stale) {
		scx_bpf_error("tmp_cpumask should be null");
		return -EINVAL;
	}

	return 0;
}

/* Publish a cell's borrowable cpumask used for cross-cell idle borrowing. */
static __always_inline int
publish_borrowable_cell_state(u32			      cell_id,
			      const struct cell_cpumask_data *borrowable_data)
{
	struct cell_cpumask_wrapper *cpumaskw;
	struct bpf_cpumask *bmask    __free(bpf_cpumask) = NULL;
	int			     err		 = 0;
	u32			     cpu;

	cpumaskw = bpf_map_lookup_elem(&cell_cpumasks, &cell_id);
	if (!cpumaskw)
		return 0;

	bmask = bpf_kptr_xchg(&cpumaskw->borrowable_tmp_cpumask, NULL);
	if (!bmask) {
		scx_bpf_error("borrowable_tmp_cpumask is NULL for cell %d",
			      cell_id);
		return -EINVAL;
	}

	bpf_cpumask_clear(bmask);

	bpf_for(cpu, 0, nr_possible_cpus)
	{
		if (err)
			return err;
		if (!config_cpumask_test_cpu(borrowable_data, cpu, &err)) {
			if (err)
				return err;
			continue;
		}

		bpf_cpumask_set_cpu(cpu, bmask);
	}

	bmask = bpf_kptr_xchg(&cpumaskw->borrowable_cpumask,
			      no_free_ptr(bmask));
	if (!bmask) {
		scx_bpf_error("borrowable cpumask should never be null");
		return -EINVAL;
	}

	struct bpf_cpumask *stale __free(bpf_cpumask) = bpf_kptr_xchg(
		&cpumaskw->borrowable_tmp_cpumask, no_free_ptr(bmask));
	if (stale) {
		scx_bpf_error("borrowable tmp_cpumask should be null");
		return -EINVAL;
	}

	return 0;
}

/* Bind one configured owner cgroup to its cell. */
static __always_inline int
apply_owner_assignment(const struct cell_assignment *assignment)
{
	struct cgroup *cg __free(cgroup) = NULL;
	struct cgrp_ctx	 *cgc;
	struct cell	 *cell;
	u32		  cell_id = assignment->cell_id;
	u64		  cgid	  = assignment->cgid;

	if (cell_id >= MAX_CELLS)
		return -EINVAL;

	cg = bpf_cgroup_from_id(cgid);
	if (!cg)
		return 0;

	cgc = lookup_cgrp_ctx(cg);
	if (!cgc)
		return -ENOENT;

	cell = lookup_cell(cell_id);
	if (!cell)
		return -EINVAL;

	cell->in_use	 = 1;
	cell->owner_cgid = cgid;

	cgc->cell	 = cell_id;
	cgc->cell_owner	 = true;
	return 0;
}

/* Walk the cgroup tree so non-owner cgroups inherit their parent's cell. */
static __always_inline int propagate_inherited_cells_from_root(void)
{
	struct cgroup_subsys_state *root_css, *pos;
	struct cgroup		   *cur_cgrp;

	scoped_guard(rcu)
	{
		if (!root_cgrp) {
			scx_bpf_error("root_cgrp should not be null");
			return -EINVAL;
		}

		struct cgroup *root_cgrp_ref __free(cgroup) =
			bpf_cgroup_acquire(root_cgrp);
		if (!root_cgrp_ref) {
			scx_bpf_error(
				"Failed to acquire reference to root_cgrp");
			return -EINVAL;
		}

		root_css       = &root_cgrp_ref->self;
		level_cells[0] = 0;

		bpf_for_each(css, pos, root_css,
			     BPF_CGROUP_ITER_DESCENDANTS_PRE) {
			struct cgrp_ctx *cgrp_ctx;
			struct cell	*cell;
			u32		 level;

			cur_cgrp = pos->cgroup;
			cgrp_ctx = lookup_cgrp_ctx_fallible(cur_cgrp);
			if (!cgrp_ctx)
				continue;

			level = cur_cgrp->level;
			if (level >= MAX_CG_DEPTH) {
				scx_bpf_error("Cgroup hierarchy too deep: %d",
					      level);
				return -EINVAL;
			}

			if (cgrp_ctx->cell_owner) {
				cell = lookup_cell(cgrp_ctx->cell);
				if (!cell)
					return -EINVAL;

				if (cell->in_use &&
				    cell->owner_cgid == cur_cgrp->kn->id) {
					level_cells[level] = cgrp_ctx->cell;
					continue;
				}

				cgrp_ctx->cell_owner = false;
			}

			u32 parent_cell = level > 0 ? level_cells[level - 1] :
						      0;
			WRITE_ONCE(cgrp_ctx->cell, parent_cell);
			level_cells[level] = parent_cell;
		}
	}

	return 0;
}

SEC("syscall")
int apply_cell_config(void *ctx)
{
	struct cell_config *config = &cell_config;
	u32		    i, cell_id;
	int		    ret;

	ret = reset_non_root_cells_for_config();
	if (ret)
		return ret;

	if (config->num_cells > MAX_CELLS)
		return -EINVAL;

	/* Phase 2: publish primary cpumasks and CPU ownership */
	bpf_for(cell_id, 0, MAX_CELLS)
	{
		struct cell_cpumask_data *cpumask_data;

		if (cell_id >= config->num_cells)
			break;

		cpumask_data = MEMBER_VPTR(config->cpumasks, [cell_id]);
		if (!cpumask_data)
			return -EINVAL;

		ret = publish_primary_cell_state(cell_id, cpumask_data);
		if (ret)
			return ret;
	}

	/* Phase 3: publish borrowable cpumasks */
	if (enable_borrowing) {
		bpf_for(cell_id, 0, MAX_CELLS)
		{
			struct cell_cpumask_data *borrowable_data;

			if (cell_id >= config->num_cells)
				break;

			borrowable_data = MEMBER_VPTR(
				config->borrowable_cpumasks, [cell_id]);
			if (!borrowable_data)
				return -EINVAL;

			ret = publish_borrowable_cell_state(cell_id,
							    borrowable_data);
			if (ret)
				return ret;
		}
	}

	/* Phase 4: recompute per-LLC CPU counts for all active cells */
	if (enable_llc_awareness) {
		scoped_guard(rcu)
		{
			bpf_for(cell_id, 0, MAX_CELLS)
			{
				if (cell_id >= config->num_cells)
					break;
				if (recalc_cell_llc_counts(cell_id, NULL))
					return -EINVAL;
			}
		}
	}

	/* Phase 5: bind owner cgroups to their configured cells */
	if (config->num_cell_assignments > MAX_CELLS)
		return -EINVAL;

	bpf_for(i, 0, MAX_CELLS)
	{
		if (i >= config->num_cell_assignments)
			break;

		ret = apply_owner_assignment(&config->assignments[i]);
		if (ret)
			return ret;
	}

	/* Phase 6: propagate inherited cell ownership through the tree */
	ret = propagate_inherited_cells_from_root();
	if (ret)
		return ret;

	/* Phase 7: publish the new configuration sequence */
	__atomic_add_fetch(&applied_configuration_seq, 1, __ATOMIC_RELEASE);

	return 0;
}

// clang-format off
SCX_OPS_DEFINE(mitosis,
	       .select_cpu		= (void *)mitosis_select_cpu,
	       .enqueue			= (void *)mitosis_enqueue,
	       .dispatch		= (void *)mitosis_dispatch,
	       .running			= (void *)mitosis_running,
	       .stopping		= (void *)mitosis_stopping,
	       .set_cpumask		= (void *)mitosis_set_cpumask,
	       .init_task		= (void *)mitosis_init_task,
	       .cgroup_init		= (void *)mitosis_cgroup_init,
	       .cgroup_exit		= (void *)mitosis_cgroup_exit,
	       .cgroup_move		= (void *)mitosis_cgroup_move,
	       .dump 			= (void *)mitosis_dump,
	       .dump_task		= (void *)mitosis_dump_task,
	       .init			= (void *)mitosis_init,
	       .exit			= (void *)mitosis_exit,
	       .name			= "mitosis");
// clang-format on
