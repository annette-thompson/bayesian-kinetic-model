"""Summarize draws/hour by GPU model from progress_log.jsonl files.

Usage:
    python gpu_throughput_report.py "Results/Chain Scaling Tests"

Scans every progress_log.jsonl under the given root, groups consecutive
entries by device (jax device_kind), and reports draws/hour within each
same-device run. Intervals that cross a device change (a preempt/requeue
landing on different hardware) are excluded from the rate calc -- that
interval's wall time is contaminated by queue wait + a fresh JIT compile
on the new device, not sustained throughput on either device.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path


def load_log(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def intervals_by_device(rows: list[dict]) -> list[tuple[str, float, float]]:
    """(device, wall_seconds, draws_advanced) for each consecutive same-device pair."""
    out = []
    for prev, cur in zip(rows, rows[1:]):
        if prev["device"] != cur["device"]:
            continue  # cross-restart / cross-device boundary -- not attributable
        dt = cur["t"] - prev["t"]
        d_draws = (cur["warmup_done"] + cur["sampling_done"]) - (
            prev["warmup_done"] + prev["sampling_done"]
        )
        if dt > 0 and d_draws >= 0:
            out.append((cur["device"], dt, d_draws))
    return out


def main(root: str) -> None:
    root_path = Path(root)
    logs = sorted(root_path.glob("*/checkpoint/progress_log.jsonl"))
    if not logs:
        print(f"No progress_log.jsonl files found under {root_path}")
        return

    per_device_time = defaultdict(float)
    per_device_draws = defaultdict(int)
    per_system_summary = []

    for log_path in logs:
        system = log_path.parent.parent.name
        rows = load_log(log_path)
        if len(rows) < 2:
            per_system_summary.append((system, len(rows), "not enough data yet"))
            continue

        devices_seen = [r["device"] for r in rows]
        n_restarts = sum(1 for a, b in zip(devices_seen, devices_seen[1:]) if a != b)

        ivals = intervals_by_device(rows)
        sys_time = defaultdict(float)
        sys_draws = defaultdict(int)
        for device, dt, d_draws in ivals:
            per_device_time[device] += dt
            per_device_draws[device] += d_draws
            sys_time[device] += dt
            sys_draws[device] += d_draws

        rate_str = ", ".join(
            f"{d}: {sys_draws[d] / (sys_time[d] / 3600):.0f} draws/hr"
            for d in sys_time
            if sys_time[d] > 0
        )
        per_system_summary.append(
            (system, len(rows), f"{rate_str or 'no within-device interval yet'}"
             f"{f' ({n_restarts} restart(s))' if n_restarts else ''}")
        )

    print("=== Per-system ===")
    for system, n_rows, detail in per_system_summary:
        print(f"  {system}: {n_rows} checkpoint(s) -- {detail}")

    print("\n=== Aggregate draws/hour by GPU model (within-device intervals only) ===")
    if not per_device_time:
        print("  Not enough same-device consecutive checkpoints yet to compute any rate.")
        return
    for device, total_time in sorted(per_device_time.items(), key=lambda kv: -per_device_draws[kv[0]]):
        draws = per_device_draws[device]
        hours = total_time / 3600
        print(f"  {device or '(unknown)'}: {draws} draws over {hours:.2f} hr -> {draws / hours:.0f} draws/hr"
              if hours > 0 else f"  {device}: insufficient time")


if __name__ == "__main__":
    root_arg = sys.argv[1] if len(sys.argv) > 1 else "Results/Chain Scaling Tests"
    main(root_arg)
