# scx_mitosis

A cgroup-aware scheduler that isolates workloads into *cells*. The eventual goal is to enable overcomitting workloads on datacenter servers.

## How it works

Cgroups that restrict their parent's cpuset get their own *cell*—a dedicated CPU set with a shared dispatch queue. Tasks within a cell are scheduled using weighted vtime. CPU-pinned tasks (typically system threads) use per-CPU queues. Cell and CPU tasks compete for dispatch based on their vtime.

On multi-LLC systems, LLC-awareness keeps tasks on cache-sharing CPUs. In this case, the single cell queue is split into multiple queues, one per LLC.

## Usage

```bash
# Create a parent cgroup for cells and enable the CPU controller
sudo mkdir -p /sys/fs/cgroup/workloads.slice
echo +cpu | sudo tee /sys/fs/cgroup/workloads.slice/cgroup.subtree_control

# Start mitosis in userspace-managed mode
sudo scx_mitosis --cell-parent-cgroup /workloads.slice

# With LLC-awareness
sudo scx_mitosis --cell-parent-cgroup /workloads.slice --enable-llc-awareness
```
