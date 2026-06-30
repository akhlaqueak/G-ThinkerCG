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
mkdir -p logs
: > logs/failed.log
TIME_THRESH=1h
run_case() {
    local logfile="$1"
    shift

    timeout $TIME_THRESH "$@" > "$logfile" 2>&1
    local rc=$?

    if [ "$rc" -ne 0 ]; then
        if [ "$rc" -eq 124 ]; then
            echo "Run timed out after $TIME_THRESH (exit code: $rc): $*" >> logs/failed.log
            echo "Run timed out after $TIME_THRESH (exit code: $rc): $*" >> "$logfile"
        elif [ "$rc" -eq 134 ]; then
            echo "Run aborted (SIGABRT, exit code: $rc): $*" >> logs/failed.log
            echo "Run aborted (SIGABRT, exit code: $rc): $*" >> "$logfile"
        else
            echo "Run failed (exit code: $rc): $*" >> logs/failed.log
            echo "Run failed (exit code: $rc): $*" >> "$logfile"
        fi
    fi
}

cp run run-exp
for d in $ds; do 
    for q in $quer; do
    #    run_case "logs/$d-$q-g2aimd.log" ./g2aimd -dg "ds/$d.bin" -q "$q" -cpu 0
    #    run_case "logs/$d-$q-nogpu.log" ./run-exp -dg "ds/$d.bin" -q "$q" -gpu 0  -tau 100000
    #    run_case "logs/$d-$q-nocpu.log" ./run-exp -dg "ds/$d.bin" -q "$q" -cpu 0 -pingpong 0 -gpuchunk 100000
    #    run_case "logs/$d-$q-with_cpu_gpu.log" ./run-exp -dg "ds/$d.bin" -q "$q" -tau 100000 -pingpong 0 -gpuchunk 100000 -cpuchunk 10
    #    run_case "logs/$d-$q-nocpu-expand.log" ./run-exp -dg "ds/$d.bin" -q "$q" -cpu 0 -s expand
    
    if [ -f "logs/$d-$q-nogpu.log" ] && grep -q "Total time" "logs/$d-$q-nogpu.log"; then
        echo "Skipping logs/$d-$q-nogpu.log"
    else
        run_case "logs/$d-$q-nogpu.log" ./run-exp -dg "ds/$d.bin" -q "$q" -tau 100000 -gpu 0
    fi
    done
done

get_results(){
exps="g2aimd nocpu nogpu with_cpu_gpu"

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
