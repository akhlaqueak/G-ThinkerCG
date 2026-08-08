#!/usr/bin/env bash
#SBATCH --job-name=g2aimd
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
#SBATCH --partition=amperenodes
#SBATCH --time=12:00:00
#SBATCH --no-requeue
#SBATCH --gres=gpu:1
#SBATCH --output=%x.out
#SBATCH --error=%x.err

if [ ! -f ds.txt ]; then
    echo "Missing dataset list: ds.txt" >&2
    exit 1
fi

quer="2 5 7 8 9"
timeout_threshold="30m"
skip_existing_logs=0
skip_completed_logs=1
mkdir -p logs
: > logs/failed.log

run_case() {
    local logfile="logs/$1"
    shift
    local cmd_str
    printf -v cmd_str '%q ' "$@"
    cmd_str=${cmd_str% }

    if [ "$skip_existing_logs" -eq 1 ] && [ -f "$logfile" ]; then
        echo "Skipping existing $logfile"
        return
    fi

    if [ "$skip_completed_logs" -eq 1 ] && [ -f "$logfile" ] && grep -Eq "Total time|OOM" "$logfile"; then
        echo "Skipping $logfile"
        return
    fi

    sleep 5s # sleeping because sometimes device is not ready from previous experiment
    echo "Running: timeout $timeout_threshold $cmd_str"
    {
        echo "Command: timeout $timeout_threshold $cmd_str"
        echo
    } > "$logfile"

    timeout "$timeout_threshold" "$@" >> "$logfile" 2>&1
    local rc=$?

    if [ "$rc" -ne 0 ]; then
        echo "Run failed (exit code: $rc): timeout $timeout_threshold $cmd_str" >> logs/failed.log
        echo "Run failed (exit code: $rc): timeout $timeout_threshold $cmd_str" >> "$logfile"
    fi
}

while IFS= read -r d || [ -n "$d" ]; do
    d=${d%$'\r'}
    if [ -z "$d" ] || [[ "$d" == \#* ]]; then
        continue
    fi
    for q in $quer; do
        # run_case "$d-$q-nogpu.log" ./run-exp -dg "ds/$d.bin" -q "$q" -gpu 0
        run_case "$d-$q-g2aimd.log" ./g2aimd -dg "ds/$d.bin" -q "$q" -cpu 0
        run_case "$d-$q-nocpu.log" ./run -dg "ds/$d.bin" -q "$q" -cpu 0 -pingpong 0 -gpuchunk 100000
        # run_case "$d-$q-cpugpu.log" ./run -dg "ds/$d.bin" -q "$q" -cpuchunk 1 -pingpong 0 -gpuchunk 100000

    done
done < ds.txt


get_results(){
exps="nocpu_pingpong_abort with_cpu_gpu_pingpong_abort"
exps="g2aimd nocpu"

while IFS= read -r d || [ -n "$d" ]; do
    d=${d%$'\r'}
    if [ -z "$d" ] || [[ "$d" == \#* ]]; then
        continue
    fi
    for q in $quer; do
        echo -en "$d $q "
        for exp in $exps; do
            fname="logs/$d-$q-$exp.log"

            if grep -q "OOM" "$fname" 2>/dev/null; then
                echo -en "OOM "
            elif grep -q "Total time" "$fname" 2>/dev/null; then
                grep "Total time" "$fname" | awk '{printf "%s ", $NF}'
            else
                echo -en "X "
            fi
        done
        echo 
    done
done < ds.txt
}
get_results
