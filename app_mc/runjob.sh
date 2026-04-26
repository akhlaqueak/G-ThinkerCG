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

set -euo pipefail

mkdir -p "logs"

datasets="wikipedia_link_sr
wikipedia_link_sh
wikipedia_link_fr
web-wikipedia_link_en13-all
wikipedia_link_de
edit-frwiki
orkut-links
wikipedia_link_ru
link-dynamic-frwiki
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

ds_path="/home/akhlaque.ak@gmail.com/graphs/data/kcore"
mkdir -p logs
: > logs/failed.log

for ds in $datasets; do
    for chunk in 200; do
        for tau in 1000; do 
            for gpuchunk in 1000 10000 50000 ; do
                rc_no_cpu=0
                sleep 5
                fname="logs/$ds-cpuchunk-$chunk-tau-$tau-gpuchunk-$gpuchunk.log"
                timeout 10m ./run -dg "$ds_path/$ds.bin" -eta 2000 -cpu 32 -cpuchunk $chunk -gpuchunk $gpuchunk -pingpong 1 -tau $tau\
                    > "$fname" 2>&1 || rc_no_cpu=$?

                if [ "$rc_no_cpu" -ne 0 ]; then
                    echo "Dataset failed: $ds cpuchunk=$chunk (exit code: $rc_no_cpu)" | tee -a "logs/failed.log"
                fi
            done
        done
    done
done


output="results.txt"
: > "$output"

for ds in $datasets; do
    for chunk in 200; do
        for tau in 1000; do 
            for gpuchunk in 1000 10000 50000; do
                fname="logs/$ds-cpuchunk-$chunk-tau-$tau-gpuchunk-$gpuchunk.log"
    # for chunk in 1 10 100 200 500 1000; do
    #     for tau in 1 10 100 500 1000; do 
                cliques=$(grep "Total count" "$fname" 2>/dev/null | awk '{print $NF}' | tail -n 1 || true)
                time_taken=$(grep "Total time" "$fname" 2>/dev/null | awk '{print $NF}' | tail -n 1 || true)
                cliques=${cliques:-NA}
                time_taken=${time_taken:-NA}
                printf "%s " "$time_taken" >> "$output"
            done
        done
    done
    printf "\n" >> "$output"
done

output="results-cpuonly.txt"
: > "$output"

for ds in $datasets; do 
    rc_no_cpu=0
    fname="logs/$ds-cpuonly.log"
    timeout 10m ./run -dg "$ds_path/$ds.bin" -gpu 0 -cpu 32 -cpuchunk 200 -tau 1000 > "$fname" 2>&1 || rc_no_cpu=$?
    cliques=$(grep "Total count" "$fname" 2>/dev/null | awk '{print $NF}' | tail -n 1 || true)
    time_taken=$(grep "Total time" "$fname" 2>/dev/null | awk '{print $NF}' | tail -n 1 || true)
    cliques=${cliques:-NA}
    time_taken=${time_taken:-NA}
    printf "%s " "$time_taken" >> "$output"
done