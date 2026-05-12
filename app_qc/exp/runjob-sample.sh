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

TIMEOUT=10m
ds=WordNet
k=5
g=0.8
sol=cg
# for sol in cg quick fastqc; do
for chunk in 100 1000 10000 20000; do
    rc=0
    fname="$LOG_DIR/$sol-$ds-$k-$g-$chunk"
    timeout $TIMEOUT "$SCRIPT_DIR/$sol" -f "$ds_path/$ds.sbin" -k "$k" -g "$g" -gpuchunk $chunk -cpuchunk 10 > "$fname-gpuchunk.log" 2>&1 || rc=$?
done

for chunk in 1 10 50 100; do
    rc=0
    fname="$LOG_DIR/$sol-$ds-$k-$g-$chunk"
    timeout $TIMEOUT "$SCRIPT_DIR/$sol" -f "$ds_path/$ds.sbin" -k "$k" -g "$g" -cpuchunk $chunk > "$fname-cpuchunk.log" 2>&1 || rc=$?
done

