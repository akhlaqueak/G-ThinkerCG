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

for d in $ds_path/*.txt; do
    ./preprocess -f $d
    ./kc -f ${d/txt/bin}
    ./binToSer ${d/txt/bin} ${d/txt/sbin}
done

while read -r ds k g; do
    for sol in cg quick fastqc; do
        rc=0
        fname="$LOG_DIR/$sol-$ds-$k-$g.log"
        timeout 10m "$SCRIPT_DIR/$sol" -f "$ds_path/$ds.sbin" -k "$k" -g "$g" > "$fname" 2>&1 || rc=$?

        if [ "$rc" -ne 0 ]; then
            echo "Failed $fname (exit code: $rc)" | tee -a "$LOG_DIR/failed.log"
        fi
    done
done < "$CONFIG_FILE"



# output="results.txt"
# : > "$output"

# for ds in $datasets; do
#     for chunk in 200; do
#         for tau in 1000; do 
#             for gpuchunk in 1000 10000 50000; do
#                 fname="logs/$ds-cpuchunk-$chunk-tau-$tau-gpuchunk-$gpuchunk.log"
#     # for chunk in 1 10 100 200 500 1000; do
#     #     for tau in 1 10 100 500 1000; do 
#                 cliques=$(grep "Total count" "$fname" 2>/dev/null | awk '{print $NF}' | tail -n 1 || true)
#                 time_taken=$(grep "Total time" "$fname" 2>/dev/null | awk '{print $NF}' | tail -n 1 || true)
#                 cliques=${cliques:-NA}
#                 time_taken=${time_taken:-NA}
#                 printf "%s " "$time_taken" >> "$output"
#             done
#         done
#     done
#     printf "\n" >> "$output"
# done

# output="results-cpuonly.txt"
# : > "$output"

# for ds in $datasets; do 
#     rc_no_cpu=0
#     fname="logs/$ds-cpuonly.log"
#     timeout 10m ./run -dg "$ds_path/$ds.bin" -gpu 0 -cpu 32 -cpuchunk 200 -tau 1000 > "$fname" 2>&1 || rc_no_cpu=$?
#     cliques=$(grep "Total count" "$fname" 2>/dev/null | awk '{print $NF}' | tail -n 1 || true)
#     time_taken=$(grep "Total time" "$fname" 2>/dev/null | awk '{print $NF}' | tail -n 1 || true)
#     cliques=${cliques:-NA}
#     time_taken=${time_taken:-NA}
#     printf "%s " "$time_taken" >> "$output"
# done
