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
soc-livejournal
socfb-B-anon
soc-pokec
delicious-ui
orkut-links
as-skitter
sx-stackoverflow
link-dynamic-itwiki
zhishi-baidu-internallink
wiki-Talk
link-dynamic-frwiki
zhishi-all
wiki-topcats
soc-sinaweibo
"


quer="2 5 6 7 8 9"
timeout_threshold="30m"
mkdir -p logs
: > logs/failed.log

run_case() {
    local logfile="$1"
    shift

    # if [ -f "$logfile" ] && grep -q "Total time" "$logfile"; then
    #     echo "Skipping $logfile"
    #     return
    # fi

    timeout "$timeout_threshold" "$@" > "$logfile" 2>&1
    local rc=$?

    if [ "$rc" -ne 0 ]; then
        echo "Run failed (exit code: $rc): timeout $timeout_threshold $*" >> logs/failed.log
        echo "Run failed (exit code: $rc): timeout $timeout_threshold $*" >> $logfile
    fi
}

cp run run-exp
for d in $ds; do 
    for q in $quer; do
        # run_case "logs/$d-$q-nogpu.log" ./run-exp -dg "ds/$d.bin" -q "$q" -gpu 0
        # run_case "logs/$d-$q-g2aimd.log" ./g2aimd -dg "ds/$d.bin" -q "$q" -cpu 0
        # run_case "logs/$d-$q-nocpu.log" ./run-exp -dg "ds/$d.bin" -q "$q" -cpu 0 -pingpong 0 -gpuchunk 100000
        run_case "logs/$d-$q-nocpu-pingpong.log" ./run-exp -dg "ds/$d.bin" -q "$q" -cpu 0 -pingpong 1 -gpuchunk 100000 
        # run_case "logs/$d-$q-with_cpu_gpu.log" ./run-exp -dg "ds/$d.bin" -q "$q" -tau 100000 -pingpong 0 -gpuchunk 100000 -cpuchunk 1

    done
done

get_results(){
for d in $ds; do 
    for q in $quer; do
        fname="logs/$d-$q-$exp.log"

        if grep -q "Total time" "$fname" 2>/dev/null; then
            grep "Total time" "$fname" | awk '{print $NF}'
        else
            echo "NA"
        fi
    done
done
}

exp=with_cpu_gpu
