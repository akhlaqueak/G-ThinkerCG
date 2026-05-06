#!/usr/bin/env bash
#SBATCH --job-name=data-prep
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=10G
#SBATCH --partition=medium
#SBATCH --time=48:00:00
#SBATCH --no-requeue
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

ds_path="/home/akhlaque.ak@gmail.com/G-ThinkerCG/datasets"

readarray -t datasets <<'EOF'
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
soc-pokec
EOF

cd "$ds_path"

for ((i=${#datasets[@]}-1; i>=0; i--)); do
    d="${datasets[$i]}"
    ./binToSer "/home/akhlaque.ak@gmail.com/graphs/data/kcore/$d.bin" "$d.sbin"
    echo "$d done"
done