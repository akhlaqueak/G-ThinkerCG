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

while IFS=$'\t' read -r ds k g; do
    [ -z "$ds" ] && continue
# for pp in 0 1 2; do

#     run_case "$ds-$k-$g-pp$pp-gpuchunk-1m.log" \
#         ./run -f "../ds/$ds.sbin" -k $k -g $g -cpu 0 -pingpong $pp -drop_oversized_tasks 1 -gpuchunk 1000000
# done
#     run_case "$ds-$k-$g-cpu.log" \
#         ./run -f "../ds/$ds.sbin" -k $k -g $g -gpu 0 
for c in 3 4; do
sleep 5s
    run_case "$ds-$k-$g-hybrid-c$c.log" \
        ./run -f "../ds/$ds.sbin" -k $k -g $g -c $c
done
    # run_case "$ds-$k-$g-fastqc.log" \
    #     ./fastqc -f "../ds/$ds.sbin" -k $k -g $g 
    # run_case "$ds-$k-$g-quick.log" \
    #     ./quick -f "../ds/$ds.sbin" -k $k -g $g 

done < ds.txt


while IFS=$'\t' read -r ds k g; do
for c in 3 4; do
        fname=logs/$ds-$k-$g-hybrid-c$c.log
        if grep -q "Total time" "$fname" 2>/dev/null; then
            grep "Total time" "$fname" | awk '{printf "%s ", $NF}'
        else
            echo -en "X "
        fi
done

echo
        # fname=logs/$ds-$k-$g-hybrid-c$c.log
        # if grep -q "Total time" "$fname" 2>/dev/null; then
        #     grep "Total time" "$fname" | awk '{printf "%s\n", $NF}'
        # else
        #     echo "X"
        # fi
done < ds.txt