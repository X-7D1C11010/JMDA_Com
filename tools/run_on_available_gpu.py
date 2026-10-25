"""Wait for a sufficiently idle GPU, then bind and run one command.

The selected physical GPU is exposed to the child as CUDA device 0 through
``CUDA_VISIBLE_DEVICES``.  Selection requires both enough free memory and a
low utilization reading for several consecutive polls, preventing a brief dip
between another job's kernels from being mistaken for an idle GPU.
"""

from __future__ import annotations

import argparse
import csv
import os
import signal
import subprocess
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class GpuState:
    index: str
    uuid: str
    free_mib: int
    utilization_percent: int


def parse_nvidia_smi(output: str) -> list[GpuState]:
    states = []
    for row in csv.reader(line for line in output.splitlines() if line.strip()):
        if len(row) != 4:
            raise ValueError(f"unexpected nvidia-smi row: {row!r}")
        index, uuid, free_mib, utilization = (value.strip() for value in row)
        states.append(
            GpuState(
                index=index,
                uuid=uuid,
                free_mib=int(free_mib),
                utilization_percent=int(utilization),
            )
        )
    if not states:
        raise RuntimeError("nvidia-smi returned no GPUs")
    return states


def query_gpus() -> list[GpuState]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return parse_nvidia_smi(completed.stdout)


def eligible_gpus(
    states: list[GpuState],
    min_free_mib: int,
    max_utilization_percent: int,
    candidates: set[str] | None = None,
) -> list[GpuState]:
    eligible = [
        state
        for state in states
        if (candidates is None or state.index in candidates or state.uuid in candidates)
        and state.free_mib >= min_free_mib
        and state.utilization_percent <= max_utilization_percent
    ]
    return sorted(
        eligible,
        key=lambda state: (state.free_mib, -state.utilization_percent),
        reverse=True,
    )


def format_states(states: list[GpuState]) -> str:
    return "; ".join(
        f"GPU {state.index}: free={state.free_mib} MiB, "
        f"util={state.utilization_percent}%"
        for state in states
    )


def scheduler_lock(path: Path):
    if os.name == "nt":
        return nullcontext(None)
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open("a+", encoding="utf-8")

    class _Lock:
        def __enter__(self):
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            return stream

        def __exit__(self, exc_type, exc_value, traceback):
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
            stream.close()

    return _Lock()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Wait for an idle GPU with enough free memory, then run a command"
    )
    parser.add_argument("--min-free-mib", type=int, required=True)
    parser.add_argument("--max-util-percent", type=int, default=10)
    parser.add_argument("--stable-checks", type=int, default=3)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--timeout-seconds", type=float, default=0.0)
    parser.add_argument("--launch-grace-seconds", type=float, default=15.0)
    parser.add_argument("--gpus", default="")
    parser.add_argument("--tag", default="training")
    parser.add_argument("--lock-file", default="/tmp/jmda_gpu_scheduler.lock")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("a command is required after --")
    if args.min_free_mib <= 0:
        parser.error("--min-free-mib must be positive")
    if not 0 <= args.max_util_percent <= 100:
        parser.error("--max-util-percent must be in [0, 100]")
    if args.stable_checks <= 0 or args.poll_seconds <= 0:
        parser.error("stable-checks and poll-seconds must be positive")
    return args


def main():
    args = parse_args()
    candidates = {item.strip() for item in args.gpus.split(",") if item.strip()}
    candidates = candidates or None
    stable_counts: dict[str, int] = {}
    started = time.monotonic()
    lock_path = Path(args.lock_file).expanduser()

    print(
        f"[GPU-WAIT] {args.tag}: require free >= {args.min_free_mib} MiB, "
        f"util <= {args.max_util_percent}% for {args.stable_checks} checks; "
        f"candidates={sorted(candidates) if candidates else 'all'}",
        flush=True,
    )

    while True:
        with scheduler_lock(lock_path):
            states = query_gpus()
            eligible = eligible_gpus(
                states,
                args.min_free_mib,
                args.max_util_percent,
                candidates,
            )
            eligible_ids = {state.index for state in eligible}
            for state in states:
                stable_counts[state.index] = (
                    stable_counts.get(state.index, 0) + 1
                    if state.index in eligible_ids
                    else 0
                )
            ready = [
                state
                for state in eligible
                if stable_counts.get(state.index, 0) >= args.stable_checks
            ]
            if ready:
                selected = ready[0]
                environment = os.environ.copy()
                environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
                environment["CUDA_VISIBLE_DEVICES"] = selected.index
                environment["JMDA_SELECTED_PHYSICAL_GPU"] = selected.index
                print(
                    f"[GPU-SELECT] {args.tag}: physical GPU {selected.index}; "
                    f"free={selected.free_mib} MiB, "
                    f"util={selected.utilization_percent}%; launching: "
                    + " ".join(args.command),
                    flush=True,
                )
                child = subprocess.Popen(args.command, env=environment)

                def forward_signal(signum, _frame, process=child):
                    if process.poll() is None:
                        process.send_signal(signum)

                signal.signal(signal.SIGTERM, forward_signal)
                signal.signal(signal.SIGINT, forward_signal)
                grace_deadline = time.monotonic() + args.launch_grace_seconds
                while time.monotonic() < grace_deadline and child.poll() is None:
                    time.sleep(0.25)
                if child.poll() is not None:
                    return child.returncode
                break

            stamp = datetime.now().astimezone().isoformat(timespec="seconds")
            print(
                f"[GPU-WAIT] {stamp} {args.tag}: {format_states(states)}",
                flush=True,
            )

        if args.timeout_seconds and time.monotonic() - started >= args.timeout_seconds:
            print(f"[GPU-WAIT] {args.tag}: timed out", file=sys.stderr, flush=True)
            return 124
        time.sleep(args.poll_seconds)

    return child.wait()


if __name__ == "__main__":
    raise SystemExit(main())
