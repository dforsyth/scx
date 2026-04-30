// Copyright (c) Meta Platforms, Inc. and affiliates.

// This software may be used and distributed according to the terms of the
// GNU General Public License version 2.

pub mod sched {
    pub const START: &str = "sched:start";
    pub const ATTACH: &str = "sched:attach";
    pub const DETACH: &str = "sched:detach";
    pub const RUN_ID: &str = "sched:run-id";
    pub const QUEUED_WAKEUP_UNSUPPORTED: &str = "sched:queued-wakeup-unsupported";
    pub const VERBOSE_FLAG_DEPRECATED: &str = "sched:verbose-flag-deprecated";
    pub const STATS_MONITOR_EXIT: &str = "sched:stats-monitor-exit";
    pub const STATS_MONITOR_ERROR: &str = "sched:stats-monitor-error";
}

pub mod cell {
    pub const INIT: &str = "cell:init";
    pub const UPDATE: &str = "cell:update";
    pub const CREATE: &str = "cell:create";
    pub const DESTROY: &str = "cell:destroy";
    pub const REBAL: &str = "cell:rebal";
    pub const LAYOUT: &str = "cell:layout";
    pub const CPUSET: &str = "cell:cpuset";
    pub const CPUSET_CHANGE: &str = "cell:cpuset-change";
    pub const SEED_DEMAND: &str = "cell:seed-demand";
}

pub mod queue {
    pub const IDLE_GLOBAL: &str = "queue:idle-global";
}
