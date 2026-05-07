#!/usr/bin/env bash
#SBATCH --job-name=data-prep
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=192G
#SBATCH --partition=medium
#SBATCH --time=48:00:00
#SBATCH --no-requeue
#SBATCH --output=%x.out
#SBATCH --error=%x.err

set -euo pipefail

ds_path="/home/akhlaque.ak@gmail.com/G-ThinkerCG/datasets"
APP_DIR="/home/akhlaque.ak@gmail.com/G-ThinkerCG/app_qc"
BIN_TO_SER="$APP_DIR/binToSer"

if [[ ! -x "$BIN_TO_SER" ]]; then
    echo "Missing converter: $BIN_TO_SER" >&2
    echo "Build it with: make -C \"$APP_DIR\" tools" >&2
    exit 1
fi

readarray -t datasets <<'EOF'
orkut-links
as-skitter
sx-stackoverflow
link-dynamic-itwiki
zhishi-baidu-internallink
wiki-Talk
link-dynamic-frwiki
zhishi-all
wiki-topcats
wikipedia_link_sh
wikipedia_link_es
soc-sinaweibo
wikipedia_link_ru
wikipedia_link_it
edit-ukwiki
orkut-groupmemberships
EOF

cd "$ds_path"

format_duration() {
    local total_seconds="$1"
    printf '%02d:%02d:%02d' \
        $((total_seconds / 3600)) \
        $(((total_seconds % 3600) / 60)) \
        $((total_seconds % 60))
}

failures=0
total_start=$SECONDS

for d in "${datasets[@]}"; do
    start=$SECONDS
    echo "Processing $d"

    if timeout 15m "$BIN_TO_SER" "/home/akhlaque.ak@gmail.com/graphs/data/kcore/$d.bin" "$d.sbin"; then
        elapsed=$((SECONDS - start))
        echo "$d done in $(format_duration "$elapsed")"
    else
        rc=$?
        elapsed=$((SECONDS - start))
        failures=$((failures + 1))
        if [[ "$rc" -eq 124 ]]; then
            echo "$d timed out after 15 minutes; continuing" >&2
        else
            echo "$d failed with exit code $rc after $(format_duration "$elapsed"); continuing" >&2
        fi
    fi
done

total_elapsed=$((SECONDS - total_start))
echo "Processed ${#datasets[@]} datasets in $(format_duration "$total_elapsed") with $failures failure(s)"
