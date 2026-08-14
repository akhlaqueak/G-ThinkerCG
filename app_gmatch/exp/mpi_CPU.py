#!/bin/bash
#SBATCH --job-name=GM-MPI
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=10G
#SBATCH --partition=medium
#SBATCH --time=20:00:00
#SBATCH --no-requeue
#SBATCH --output=%x.out
#SBATCH --error=%x.err
""":"
module load OpenMPI
module load mpi4py

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
SOURCE_SCRIPT="${BASH_SOURCE[0]:-$0}"
RUNTIME_SCRIPT="${PWD}/.mpi_runtime_${SLURM_JOB_ID:-$$}.py"
cp "$SOURCE_SCRIPT" "$RUNTIME_SCRIPT"

echo "Runtime script: $RUNTIME_SCRIPT"
echo "Job started at $(date)"
srun --mpi=pmix --cpu-bind=cores python3 -u "$RUNTIME_SCRIPT" "$@"
rc=$?
echo "Job finished at $(date)"
exit "$rc"
":"""

from mpi4py import MPI
from pathlib import Path
import os
import shlex
import subprocess
import sys

TAG_READY = 1
TAG_TASK = 2
TAG_DONE = 3
TAG_STOP = 4

DS_FILE = Path(os.environ.get("DS_FILE", "ds.txt"))
LOG_DIR = Path(os.environ.get("LOG_DIR", "logs"))
FAILED_LOG = LOG_DIR / "failed.log"

TIMEOUT_THRESHOLD = os.environ.get("TIMEOUT_THRESHOLD", "30m")
SKIP_EXISTING_LOGS = os.environ.get("SKIP_EXISTING_LOGS", "0") == "1"
SKIP_COMPLETED_LOGS = os.environ.get("SKIP_COMPLETED_LOGS", "1") == "1"
SKIP_TIMEOUT_LOGS = os.environ.get("SKIP_TIMEOUT_LOGS", "1") == "1"

RUN_EXE = os.environ.get("RUN_EXE", "./run")
LOG_SUFFIX = os.environ.get("LOG_SUFFIX", "nogpu")
DEFAULT_ARGS = os.environ.get(
    "RUN_ARGS",
    "-gpu 0",
)


def ensure_log_dir() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def parse_ds_file() -> list[tuple[str, str]]:
    tasks = []
    with DS_FILE.open() as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                print(f"[MASTER] Skipping malformed ds.txt line: {raw_line.rstrip()}", flush=True)
                continue
            tasks.append((parts[0], parts[1]))
    return tasks


def log_has(logfile: Path, needle: str) -> bool:
    if not logfile.is_file():
        return False
    try:
        with logfile.open(errors="ignore") as fh:
            return any(needle in line for line in fh)
    except OSError:
        return False


def build_case(ds: str, q: str) -> tuple[Path, list[str]]:
    logfile = LOG_DIR / f"{ds}-{q}-{LOG_SUFFIX}.log"
    cmd = [
        *shlex.split(RUN_EXE),
        "-dg",
        f"ds/{ds}.bin",
        "-q",
        q,
        *shlex.split(DEFAULT_ARGS),
    ]
    return logfile, cmd


def should_skip(logfile: Path) -> bool:
    if SKIP_EXISTING_LOGS and logfile.is_file():
        return True
    if SKIP_COMPLETED_LOGS and log_has(logfile, "Total time"):
        return True
    if SKIP_TIMEOUT_LOGS and log_has(logfile, "STATUS: TIMEOUT"):
        return True
    return False


def generate_tasks() -> list[tuple[str, str, str, list[str]]]:
    ensure_log_dir()
    tasks = []
    for ds, q in parse_ds_file():
        logfile, cmd = build_case(ds, q)
        if should_skip(logfile):
            print(f"[MASTER] Skipping {logfile}", flush=True)
            continue
        tasks.append((ds, q, str(logfile), cmd))
    return tasks


def format_cmd(cmd: list[str]) -> str:
    return " ".join(shlex.quote(x) for x in cmd)


def run_case(task: tuple[str, str, str, list[str]]) -> tuple[int, str]:
    ds, q, logfile_raw, cmd = task
    rank = MPI.COMM_WORLD.Get_rank()
    logfile = Path(logfile_raw)
    cmd_str = format_cmd(cmd)
    timeout_cmd = ["timeout", TIMEOUT_THRESHOLD, *cmd]

    print(f"[rank {rank}] Running {ds} q={q}: timeout {TIMEOUT_THRESHOLD} {cmd_str}", flush=True)

    with logfile.open("w") as out:
        out.write(f"Command: timeout {TIMEOUT_THRESHOLD} {cmd_str}\n\n")
        out.flush()
        proc = subprocess.run(
            timeout_cmd,
            stdout=out,
            stderr=subprocess.STDOUT,
            check=False,
        )

    rc = proc.returncode
    with logfile.open("a") as out:
        if rc == 0:
            out.write("\nSTATUS: OK\n")
        elif rc == 124:
            out.write(f"\nSTATUS: TIMEOUT after {TIMEOUT_THRESHOLD}\n")
        else:
            out.write(f"\nSTATUS: FAILED (exit code {rc})\n")

    if rc != 0:
        with FAILED_LOG.open("a") as failed:
            failed.write(f"Run failed (exit code: {rc}): timeout {TIMEOUT_THRESHOLD} {cmd_str}\n")

    return rc, str(logfile)


def master(comm, size: int) -> None:
    tasks = generate_tasks()
    next_task = 0
    stopped = 0
    inflight = 0
    nworkers = size - 1
    status = MPI.Status()

    print(f"[MASTER] Loaded {len(tasks)} runnable tasks from {DS_FILE}", flush=True)

    while stopped < nworkers:
        msg = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        src = status.Get_source()
        tag = status.Get_tag()

        if tag == TAG_READY:
            if next_task < len(tasks):
                comm.send(tasks[next_task], dest=src, tag=TAG_TASK)
                next_task += 1
                inflight += 1
            else:
                comm.send(None, dest=src, tag=TAG_STOP)

        elif tag == TAG_DONE:
            ds, q, rc, logfile = msg
            inflight -= 1
            state = "OK" if rc == 0 else f"FAILED rc={rc}"
            print(f"[MASTER] {state}: {ds} q={q} -> {logfile}", flush=True)

        elif tag == TAG_STOP:
            stopped += 1

    print(f"[MASTER] All workers stopped; inflight={inflight}", flush=True)


def worker(comm, rank: int) -> None:
    while True:
        comm.send(None, dest=0, tag=TAG_READY)
        status = MPI.Status()
        task = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)

        if status.Get_tag() == TAG_STOP or task is None:
            print(f"[rank {rank}] STOP", flush=True)
            comm.send(None, dest=0, tag=TAG_STOP)
            return

        ds, q, _, _ = task
        rc, logfile = run_case(task)
        comm.send((ds, q, rc, logfile), dest=0, tag=TAG_DONE)


def main() -> None:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if size < 2:
        print("Please run with at least 2 MPI ranks: 1 master + workers.", file=sys.stderr)
        sys.exit(1)

    if rank == 0:
        master(comm, size)
    else:
        worker(comm, rank)


if __name__ == "__main__":
    main()
