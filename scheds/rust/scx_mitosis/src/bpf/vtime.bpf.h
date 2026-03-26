/* Copyright (c) Meta Platforms, Inc. and affiliates. */
/*
 * This software may be used and distributed according to the terms of the
 * GNU General Public License version 2.
 *
 * Centralized vtime mutation API for scx_mitosis. Every vtime write goes
 * through one of the vt_* functions. When enable_vtime_ledger is true,
 * mutations are recorded to a circular buffer for post-mortem analysis.
 * When false, the BPF compiler eliminates all ledger code.
 *
 * Function name = WHAT (charge, set, advance, zero, enqueue)
 * Domain        = WHICH FIELD changed (task, cpu, cell-llc)
 * Reason        = WHY (code context that triggered the mutation)
 */
#pragma once

#include "mitosis.bpf.h"

const volatile bool enable_vtime_ledger = false;
const volatile bool enable_vtime_recording = false;
const volatile u64  vtime_ledger_event_mask = ((1ULL << NR_VT_REASONS) - 1);
const volatile u32  vtime_ledger_size = VTIME_LEDGER_DEFAULT_SIZE;

u32 vtime_ledger_pos;

struct {
	__uint(type, BPF_MAP_TYPE_ARRAY);
	__uint(max_entries, VTIME_LEDGER_DEFAULT_SIZE);
	__type(key, u32);
	__type(value, struct vtime_event);
} vtime_ledger SEC(".maps");

static inline void record_vtime(enum vt_domain domain,
				enum vt_reason reason,
				u64 old_vtime, u64 new_vtime,
				u32 pid, u32 cpu, u32 cell,
				s32 llc, u32 weight)
{
	if (!enable_vtime_recording)
		return;

	if (!(vtime_ledger_event_mask & (1ULL << reason)))
		return;

	u32 pos = __sync_fetch_and_add(&vtime_ledger_pos, 1);
	u32 idx = pos % vtime_ledger_size;

	struct vtime_event *ev = bpf_map_lookup_elem(&vtime_ledger, &idx);
	if (unlikely(!ev))
		return;

	ev->timestamp  = scx_bpf_now();
	ev->old_vtime  = old_vtime;
	ev->new_vtime  = new_vtime;
	ev->domain     = domain;
	ev->reason     = reason;
	ev->pid        = pid;
	ev->cpu        = cpu;
	ev->cell       = cell;
	ev->llc        = llc;
	ev->weight     = weight;
}

static inline void vt_task_set(struct task_struct *p,
			       struct task_ctx *tctx,
			       u64 new_vtime, enum vt_reason reason)
{
	u64 old = p->scx.dsq_vtime;
	p->scx.dsq_vtime = new_vtime;

	record_vtime(VT_DOMAIN_TASK, reason, old, new_vtime,
		     p->pid, bpf_get_smp_processor_id(),
		     tctx->cell, tctx->llc, p->scx.weight);
}

static inline void vt_task_charge(struct task_struct *p,
				  struct task_ctx *tctx, u64 used,
				  enum vt_reason reason)
{
	if (p->scx.weight == 0) {
		scx_bpf_error("Task %d has zero weight", p->pid);
		return;
	}
	vt_task_set(p, tctx,
		    p->scx.dsq_vtime + used * 100 / p->scx.weight,
		    reason);
}

static inline void vt_enqueue(struct task_struct *p,
			      struct task_ctx *tctx,
			      u64 vtime, u64 enq_flags,
			      enum vt_reason reason)
{
	u64 basis = tctx->basis_vtime;
	u64 old_vtime = vtime;

	if (time_before(vtime, basis - slice_ns))
		vtime = basis - slice_ns;

	if (old_vtime != vtime)
		record_vtime(VT_DOMAIN_TASK, reason, old_vtime, vtime,
			     p->pid, bpf_get_smp_processor_id(),
			     tctx->cell, tctx->llc, p->scx.weight);

	scx_bpf_dsq_insert_vtime(p, tctx->dsq.raw, slice_ns, vtime, enq_flags);
}

static inline void vt_cpu_try_advance(struct cpu_ctx *cctx, u64 new_vtime,
				      enum vt_reason reason)
{
	u64 old = READ_ONCE(cctx->vtime_now);
	if (!time_before(old, new_vtime))
		return;

	WRITE_ONCE(cctx->vtime_now, new_vtime);

	record_vtime(VT_DOMAIN_CPU, reason, old, new_vtime,
		     0, bpf_get_smp_processor_id(),
		     cctx->cell, cctx->llc, 0);
}

static inline void vt_cell_llc_try_advance(struct cell *cell, u32 cell_idx,
					   u32 llc, u64 new_vtime,
					   enum vt_reason reason)
{
	u64 old = READ_ONCE(cell->llcs[llc].vtime_now);
	if (!time_before(old, new_vtime))
		return;

	WRITE_ONCE(cell->llcs[llc].vtime_now, new_vtime);

	record_vtime(VT_DOMAIN_CELL_LLC, reason, old, new_vtime,
		     0, bpf_get_smp_processor_id(),
		     cell_idx, (s32)llc, 0);
}

static inline void vt_cell_zero(struct cell *cell, u32 cell_idx,
				enum vt_reason reason)
{
	if (enable_llc_awareness) {
		u32 llc_idx;
		bpf_for(llc_idx, 0, MAX_LLCS)
		{
			u64 old = READ_ONCE(cell->llcs[llc_idx].vtime_now);
			WRITE_ONCE(cell->llcs[llc_idx].vtime_now, 0);
			if (old != 0)
				record_vtime(VT_DOMAIN_CELL_LLC, reason, old, 0,
					     0, bpf_get_smp_processor_id(),
					     cell_idx, (s32)llc_idx, 0);
		}
	} else {
		u64 old = READ_ONCE(cell->llcs[FAKE_FLAT_CELL_LLC].vtime_now);
		WRITE_ONCE(cell->llcs[FAKE_FLAT_CELL_LLC].vtime_now, 0);
		if (old != 0)
			record_vtime(VT_DOMAIN_CELL_LLC, reason, old, 0,
				     0, bpf_get_smp_processor_id(),
				     cell_idx, FAKE_FLAT_CELL_LLC, 0);
	}
}
