#!/usr/bin/env bash
#SBATCH --job-name=CPU
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=4G
#SBATCH --partition=medium
#SBATCH --time=40:00:00
#SBATCH --no-requeue
#SBATCH --output=%x.out
#SBATCH --error=%x.err


run_case() {
    local logfile="logs/$1"
    shift

    if [ -f "$logfile" ] && grep -q "Total time" "$logfile"; then
        echo "Skipping $logfile (already completed)"
        return
    fi

    {
        echo "CMD: $*"
        echo "START: $(date)"
        echo
    } > "$logfile"

    if timeout 30m "$@" >> "$logfile" 2>&1; then
        {
            echo
            echo "STATUS: OK"
            echo "END: $(date)"
        } >> "$logfile"
    else
        rc=$?
        {
            echo
            if [ "$rc" -eq 124 ]; then
                echo "STATUS: TIMEOUT after 30m"
            else
                echo "STATUS: FAILED (exit code $rc)"
            fi
            echo "END: $(date)"
        } >> "$logfile"
    fi
}

while IFS=$'\t' read -r ds q; do

    [ -z "$ds" ] && continue

    run_case "$ds-$q-nogpu.log" \
        ./run -dg "$HOME/graphs/data/kcore/$ds.bin" -q "$q" -gpu 0 
done < ds.txt