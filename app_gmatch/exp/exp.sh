#!/usr/bin/env bash
#SBATCH --job-name=Qset1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
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


quer="4 5 6 7 8 9 18 24 25 26 27"
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
exps="nocpu with_cpu_gpu"
# exps="g2aimd nocpu-pingpong nocpu nogpu with_cpu_gpu"

for d in $ds; do 
    for q in $quer; do
        echo -en "$d $q "
        for exp in $exps; do
            fname="logs/$d-$q-$exp.log"

            if grep -q "Total time" "$fname" 2>/dev/null; then
                grep "Total time" "$fname" | awk '{printf "%s ", $NF}'
            else
                echo -en "X "
            fi
        done
        echo 
    done
done
}
get_results
