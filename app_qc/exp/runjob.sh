#!/usr/bin/env bash
#SBATCH --job-name=Gthinker
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=4G
#SBATCH --partition=amperenodes
#SBATCH --time=12:00:00
#SBATCH --no-requeue
#SBATCH --gres=gpu:1
#SBATCH --output=%x.out
#SBATCH --error=%x.err

ds_path="/home/akhlaque.ak@gmail.com/G-ThinkerCG/datasets"
set -euo pipefail

SCRIPT_DIR="."
LOG_DIR="$SCRIPT_DIR/logs"
CONFIG_FILE="$SCRIPT_DIR/configs.txt"

mkdir -p "$LOG_DIR"
: > "$LOG_DIR/failed.log"

while read -r ds k g; do
    # for sol in cg quick fastqc; do
    for sol in cg; do
        rc=0
        fname="$LOG_DIR/$sol-$ds-$k-$g"
        timeout 10m "$SCRIPT_DIR/$sol" -f "$ds_path/$ds.sbin" -k "$k" -g "$g" > "$fname-hybrid.log" 2>&1 || rc=$?
        timeout 10m "$SCRIPT_DIR/$sol" -f "$ds_path/$ds.sbin" -k "$k" -g "$g" -gpu 0 > "$fname-cpu.log" 2>&1 || rc=$?
        timeout 10m "$SCRIPT_DIR/$sol" -f "$ds_path/$ds.sbin" -k "$k" -g "$g" -cpu 0 > "$fname-gpu.log" 2>&1 || rc=$?
        timeout 10m "$SCRIPT_DIR/quick" -f "$ds_path/$ds.sbin" -k "$k" -g "$g"> "$fname-quick.log" 2>&1 || rc=$?
        timeout 10m "$SCRIPT_DIR/fastqc" -f "$ds_path/$ds.sbin" -k "$k" -g "$g"> "$fname-fastqc.log" 2>&1 || rc=$?

        if [ "$rc" -ne 0 ]; then
            echo "Failed $fname (exit code: $rc)" | tee -a "$LOG_DIR/failed.log"
        fi
    done
done < "$CONFIG_FILE"


