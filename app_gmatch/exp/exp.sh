#!/usr/bin/env bash
#SBATCH --job-name=Qset1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
#SBATCH --partition=amperenodes
#SBATCH --time=12:00:00
#SBATCH --no-requeue
#SBATCH --gres=gpu:1
#SBATCH --output=%x.out
#SBATCH --error=%x.err

# ds="wikipedia_link_sr
# wikipedia_link_sh
# wikipedia_link_fr
# web-wikipedia_link_en13-all
# wikipedia_link_de
# edit-frwiki
# orkut-links
# wikipedia_link_ru
# link-dynamic-frwiki
ds="
edit-shwiki
edit-cebwiki
edit-mgwiktionary
edit-svwiki
edit-zhwiki
edit-viwiki
edit-frwiktionary
delicious-ti
edit-enwiktionary
trackers
edit-eswiki
edit-ruwiki
edit-nlwiki
edit-frwiki
as-skitter
edit-itwiki
soc-livejournal
dbpedia-link
socfb-B-anon
edit-jawiki
zhishi-baidu-internallink
zhishi-all
edit-plwiki
wikipedia_link_ceb
"


quer="2 5 9"
timeout_threshold="10m"
skip_existing_logs=0
skip_completed_logs=1
mkdir -p logs
: > logs/failed.log

run_case() {
    local logfile="$1"
    shift
    local cmd_str
    printf -v cmd_str '%q ' "$@"
    cmd_str=${cmd_str% }

    if [ "$skip_existing_logs" -eq 1 ] && [ -f "$logfile" ]; then
        echo "Skipping existing $logfile"
        return
    fi

    if [ "$skip_completed_logs" -eq 1 ] && [ -f "$logfile" ] && grep -q "Total time" "$logfile"; then
        echo "Skipping $logfile"
        return
    fi

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


run_experiments() {
    local run="$1"
    local fname

    for cp in 1 10 50 100 200; do
        while IFS=$'\t' read -r d q; do
            fname="logs/$d-$q-cp-$cp"

            if [ "$run" -eq 1 ]; then
                run_case "$fname" ./run -dg "ds/$d.bin" -q "$q" \
                    -cpuchunk "$cp" -gh_steal 5000
            else
                if grep -q "Total time" "$fname" 2>/dev/null; then
                    grep "Total time" "$fname" | awk '{printf "%s ", $NF}'
                else
                    echo -n "X "
                fi
                echo
            fi
        done < ds.txt
        echo
    done
}

run_experiments 1
run_experiments 0

# script of EGSM, since it has OOM and Errors
while IFS=$'\t' read -r d q; do
        run=0
        fname="logs/$d-$q-GAMMA.log"

            if grep -q "Total time" "$fname" 2>/dev/null; then
                grep "Total time" "$fname" | awk '{printf "%s ", $NF}'
            elif grep -q "out of memory" "$fname" 2>/dev/null; then
                echo -en "OOM "
            elif grep -q "illegal memory" "$fname" 2>/dev/null; then
                echo -en "OOM "
            else #OOT
                echo -en "X "
            fi
        echo
done < ds.txt

cp run run-exp
for d in $ds; do 
    for q in $quer; do
        # run_case "logs/$d-$q-nogpu.log" ./run-exp -dg "ds/$d.bin" -q "$q" -gpu 0
        # run_case "logs/$d-$q-g2aimd.log" ./g2aimd -dg "ds/$d.bin" -q "$q" -cpu 0
        # run_case "logs/$d-$q-nocpu.log" ./run-exp -dg "ds/$d.bin" -q "$q" -cpu 0 -pingpong 0 -gpuchunk 100000
        # run_case "logs/$d-$q-with_cpu_gpu.log" ./run-exp -dg "ds/$d.bin" -q "$q" -tau 100000 -pingpong 0 -gpuchunk 100000 -cpuchunk 1
        run_case "logs/$d-$q-with_cpu_gpu_pingpong_abort.log" ./run-exp -dg "ds/$d.bin" -q "$q" -tau 100000 -pingpong 1 -gpuchunk 100000 -cpuchunk 1

    done
done

get_results(){
exps="nocpu_pingpong_abort with_cpu_gpu_pingpong_abort"
# exps="g2aimd nocpu-pingpong nocpu nogpu with_cpu_gpu"

while IFS=$'\t' read -r d q; do


            fname="logs/$d-$q-cpugpu-chunk-10.log"

            if grep -q "Total time" "$fname" 2>/dev/null; then
                grep "Total time" "$fname" | awk '{printf "%s ", $NF}'
            else
                echo -en "X "
            fi
        echo 
done < ds.txt
}
get_results
