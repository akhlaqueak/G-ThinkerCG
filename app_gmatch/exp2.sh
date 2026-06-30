#!/usr/bin/env bash
#SBATCH --job-name=Gthinker2
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
mkdir -p logs
: > logs/failed.log

run_case() {
    local logfile="$1"
    shift

    timeout 10m "$@" > "$logfile" 2>&1
    local rc=$?

    if [ "$rc" -ne 0 ]; then
        if [ "$rc" -eq 124 ]; then
            echo "Run timed out after 10m (exit code: $rc): $*" >> logs/failed.log
        elif [ "$rc" -eq 134 ]; then
            echo "Run aborted (SIGABRT, exit code: $rc): $*" >> logs/failed.log
        else
            echo "Run failed (exit code: $rc): $*" >> logs/failed.log
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
    
    if [ -f "logs/$d-$q-with_cpu_gpu.log" ] && grep -q "Total time" "logs/$d-$q-with_cpu_gpu.log"; then
        run_case "logs/$d-$q-with_cpu_gpu.log" ./run-exp -dg "ds/$d.bin" -q "$q" -tau 100000 -pingpong 0 -gpuchunk 100000 -cpuchunk 10
    else
        echo "Skipping logs/$d-$q-with_cpu_gpu.log"
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
