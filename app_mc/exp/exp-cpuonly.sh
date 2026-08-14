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
    local logfile="$1"
    shift

    echo "Running $logfile"

    if [ -f "$logfile" ] && grep -q "Total time" "$logfile"; then
        echo "Skipping $logfile (already completed)"
        return
    fi

    {
        echo "CMD: $*"
        echo "START: $(date)"
        echo
    } > "$logfile"
    sleep 5s
    if timeout 10m "$@" >> "$logfile" 2>&1; then
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
                echo "STATUS: TIMEOUT after 10m"
            else
                echo "STATUS: FAILED (exit code $rc)"
            fi
            echo "END: $(date)"
        } >> "$logfile"
    fi
}
run=0
while IFS=$'\t' read -r ds; do
    # run=1
    [ -z "$ds" ] && continue
    fname="$ds-pp2-cpugpu-upd.log"
    if [ $run -eq 1 ]; then 
    run_case "$fname" \
        ./run -dg "$HOME/graphs/data/kcore/$ds.bin" -pingpong 2 -tau 500 -gpuchunk 100000
    else
        if grep -q "Total time" "$fname" 2>/dev/null; then
            grep "Total time" "$fname" | awk '{printf "%s ", $NF}'
        else
            echo -en "X "
        fi
    fi
    echo
done < ds.txt


while IFS=$'\t' read -r ds ; do
        fname=logs/$ds.log
        if grep -q "Total time" "$fname" 2>/dev/null; then
            grep "Total time" "$fname" | awk '{printf "%s\n", $NF}'
        else
            echo "X"
        fi
done < ds.txt


while IFS=$'\t' read -r ds; do
for eta in 1000 2000 5000 10000 20000; do 

        [ -z "$ds" ] && continue
        fname="logs/$ds-eta-$eta.log"


        if grep -q "Total time" "$fname" 2>/dev/null; then
            grep "Total time" "$fname" | awk '{printf "%s ", $NF}'
        else
            echo -en "X "
        fi
        # run_case "$fname" \
        #     ./run -dg "$HOME/graphs/data/kcore/$ds.bin" -cpu 0 -gpuchunk 1000000 -eta $eta -pingpong 0
done
        echo
done < ds.txt
