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
edit-mgwiktionary
edit-cebwiki
edit-shwiki
edit-svwiki
wikipedia_link_ceb
wikipedia_link_sr
edit-frwiktionary
trackers
edit-viwiki
wikipedia_link_sh
link-dynamic-frwiki
edit-frwiki
edit-eswiki
edit-enwiktionary
edit-arwiki
web-wikipedia_link_en13-all
wikipedia_link_ru
wikipedia_link_de
orkut-links
edit-dewiki
wikipedia_link_it
sx-stackoverflow
wikipedia_link_fr
wikipedia_link_nl
wikipedia_link_war
wikipedia_link_sv
edit-zhwiki
soc-sinaweibo
"


quer="2 5 7 8 9"
timeout_threshold="10m"
skip_existing_logs=0
skip_completed_logs=0
mkdir -p logs
: > logs/failed.log

run_case() {
    sleep 5s # sleeping because sometimes device is not ready from previous experiment
    local logfile="logs/$1"
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

for d in $ds; do 
    for q in $quer; do
        # run_case "$d-$q-nogpu.log" ./run-exp -dg "ds/$d.bin" -q "$q" -gpu 0
        run_case "$d-$q-g2aimd.log" ./g2aimd -dg "ds/$d.bin" -q "$q" -cpu 0
        run_case "$d-$q-nocpu.log" ./run -dg "ds/$d.bin" -q "$q" -cpu 0 -pingpong 0 -gpuchunk 100000
        # run_case "logs/$d-$q-with_cpu_gpu.log" ./run-exp -dg "ds/$d.bin" -q "$q" -tau 100000 -pingpong 0 -gpuchunk 100000 -cpuchunk 1
        # run_case "logs/$d-$q-with_cpu_gpu_pingpong_abort.log" ./run-exp -dg "ds/$d.bin" -q "$q" -tau 100000 -pingpong 1 -gpuchunk 100000 -cpuchunk 1
        # run_case "logs/$d-$q-nocpu_pingpong_abort.log" ./run-exp -dg "ds/$d.bin" -q "$q" -pingpong 1 -gpuchunk 100000 -cpu 0

    done
done
get_results(){
exps="nocpu_pingpong_abort with_cpu_gpu_pingpong_abort"
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
