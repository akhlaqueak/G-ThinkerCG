#!/usr/bin/env bash
#SBATCH --job-name=Gthinker
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=10G
#SBATCH --partition=amperenodes
#SBATCH --time=12:00:00
#SBATCH --no-requeue
#SBATCH --gres=gpu:2
#SBATCH --output=%x.out
#SBATCH --error=%x.err

set -euo pipefail

mkdir -p "logs"

datasets="wikipedia_link_sr
wikipedia_link_sh
link-dynamic-dewiki
wiki_talk_en
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
aff-orkut-user2groups
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
for ds in $datasets; do 
    ./g2aimd -dg "$ds_path/$ds.bin" > "$script_dir/logs/$ds.log"
done
