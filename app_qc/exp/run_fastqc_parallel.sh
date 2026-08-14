#!/bin/bash
#SBATCH --job-name=FastQC
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=11
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=10G
#SBATCH --partition=medium
#SBATCH --time=20:00:00
#SBATCH --no-requeue
#SBATCH --output=%x.out
#SBATCH --error=%x.err

set -u

mkdir -p logs

run_case() {
    local logfile="$1"
    shift

    echo "Running: $*"
    {
        echo "Command: $*"
        echo "START: $(date)"
        echo
    } > "$logfile"

    (
        local start_time end_time rc
        start_time=$(date +%s)
        srun --exclusive -N1 -n1 -c1 timeout 30m "$@" >> "$logfile" 2>&1
        rc=$?
        end_time=$(date +%s)
        {
            echo
            if [ "$rc" -eq 0 ]; then
                echo "STATUS: OK"
            elif [ "$rc" -eq 124 ]; then
                echo "STATUS: TIMEOUT after 30m"
            else
                echo "STATUS: FAILED (exit code $rc)"
            fi
            echo "END: $(date)"
            echo "Elapsed time (s): $((end_time - start_time))"
        } >> "$logfile"
    ) &
}

run_case logs/Flixster-fastqc.log ./fastqc -f ~/gt/datasets/Flixster.sbin -k 30 -g 0.95
run_case logs/FB-Pages-fastqc.log ./fastqc -f ~/gt/datasets/FB-Pages.sbin -k 23 -g 0.9
run_case logs/WordNet-fastqc.log ./fastqc -f ~/gt/datasets/WordNet.sbin -k 5 -g 0.8
run_case logs/Hyves-fastqc.log ./fastqc -f ~/gt/datasets/Hyves.sbin -k 20 -g 0.8
run_case logs/socfb-B-anon-fastqc.log ./fastqc -f ~/gt/datasets/socfb-B-anon.sbin -k 20 -g 0.95
run_case logs/soc-pokec-fastqc.log ./fastqc -f ~/gt/datasets/soc-pokec.sbin -k 20 -g 0.9
run_case logs/wikilens-ratings-fastqc.log ./fastqc -f ~/gt/datasets/wikilens-ratings.sbin -k 10 -g 0.72
run_case logs/edit-sswiki-fastqc.log ./fastqc -f ~/gt/datasets/edit-sswiki.sbin -k 10 -g 0.7
run_case logs/edit-kywiktionary-fastqc.log ./fastqc -f ~/gt/datasets/edit-kywiktionary.sbin -k 7 -g 0.6
run_case logs/slashdot-threads-fastqc.log ./fastqc -f ~/gt/datasets/slashdot-threads.sbin -k 6 -g 0.7
run_case logs/Douban-fastqc.log ./fastqc -f ~/gt/datasets/Douban.sbin -k 8 -g 0.5

run_case logs/Flixster-quick.log ./quick -f ~/gt/datasets/Flixster.sbin -k 30 -g 0.95
run_case logs/FB-Pages-quick.log ./quick -f ~/gt/datasets/FB-Pages.sbin -k 23 -g 0.9
run_case logs/WordNet-quick.log ./quick -f ~/gt/datasets/WordNet.sbin -k 5 -g 0.8
run_case logs/Hyves-quick.log ./quick -f ~/gt/datasets/Hyves.sbin -k 20 -g 0.8
run_case logs/socfb-B-anon-quick.log ./quick -f ~/gt/datasets/socfb-B-anon.sbin -k 20 -g 0.95
run_case logs/soc-pokec-quick.log ./quick -f ~/gt/datasets/soc-pokec.sbin -k 20 -g 0.9
run_case logs/wikilens-ratings-quick.log ./quick -f ~/gt/datasets/wikilens-ratings.sbin -k 10 -g 0.72
run_case logs/edit-sswiki-quick.log ./quick -f ~/gt/datasets/edit-sswiki.sbin -k 10 -g 0.7
run_case logs/edit-kywiktionary-quick.log ./quick -f ~/gt/datasets/edit-kywiktionary.sbin -k 7 -g 0.6
run_case logs/slashdot-threads-quick.log ./quick -f ~/gt/datasets/slashdot-threads.sbin -k 6 -g 0.7
run_case logs/Douban-quick.log ./quick -f ~/gt/datasets/Douban.sbin -k 8 -g 0.5

wait
