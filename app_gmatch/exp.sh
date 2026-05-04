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
edit-dewiki
wikipedia_link_sv
sx-stackoverflow
soc-sinaweibo
edit-eswiki
wikipedia_link_it
wikipedia_link_nl
trackers
edit-nlwiki
orkut-groupmemberships
wikipedia_link_war
edit-ptwiki
dbpedia-link
wikipedia_link_es
edit-svwiki
wikipedia_link_ceb
edit-itwiki
link-dynamic-itwiki
livejournal-groupmemberships
edit-enwiktionary
delicious-ti
edit-plwiki
edit-cebwiki
edit-ruwiki
wiki-Talk
zhishi-all
delicious-ui
edit-shwiki
edit-zhwiki
edit-frwiktionary
edit-arwiki
soc-livejournal
as-skitter
edit-viwiki
wiki-topcats
edit-jawiki
edit-ukwiki
zhishi-baidu-internallink
socfb-B-anon
edit-mgwiktionary
soc-pokec"


quer="2 5 8"
mkdir -p logs
: > logs/failed.log

run_case() {
    local logfile="$1"
    shift

    if ! timeout 10m "$@" > "$logfile" 2>&1; then
        local rc=$?
        echo "Run failed (exit code: $rc): $*" >> logs/failed.log
    fi
}

cp run run-exp
for d in $ds; do 
    for q in $quer; do
    #    run_case "logs/$d-$q-g2aimd.log" ./g2aimd -dg "ds/$d.bin" -q "$q" -cpu 0
    #    run_case "logs/$d-$q-nocpu.log" ./run-exp -dg "ds/$d.bin" -q "$q" -cpu 0
       run_case "logs/$d-$q-nogpu.log" ./run-exp -dg "ds/$d.bin" -q "$q" -gpu 0  -tau 100000
       run_case "logs/$d-$q-with_cpu_gpu.log" ./run-exp -dg "ds/$d.bin" -q "$q" -tau 100000
    #    run_case "logs/$d-$q-nocpu-expand.log" ./run-exp -dg "ds/$d.bin" -q "$q" -cpu 0 -s expand
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