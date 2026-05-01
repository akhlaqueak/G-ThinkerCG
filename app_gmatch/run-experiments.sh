ds="soc-pokec
socfb-B-anon
soc-livejournal
edit-mgwiktionary
edit-shwiki
edit-cebwiki
soc-sinaweibo"

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
       run_case "logs/$d-$q-nocpu.log" ./run-exp -dg "ds/$d.bin" -q "$q" -cpu 0
       run_case "logs/$d-$q-nogpu.log" ./run-exp -dg "ds/$d.bin" -q "$q" -gpu 0  -tau 100000
       run_case "logs/$d-$q-with_cpu_gpu.log" ./run-exp -dg "ds/$d.bin" -q "$q" -tau 100000
       run_case "logs/$d-$q-nocpu-expand.log" ./run-exp -dg "ds/$d.bin" -q "$q" -cpu 0 -s expand
    done
done


for d in $ds; do 
    for q in $quer; do
        fname="logs/$d-$q-nocpu-expand.log"

        if grep -q "Total time" "$fname" 2>/dev/null; then
            grep "Total time" "$fname" | awk '{print $NF}'
        else
            echo "NA"
        fi
    done
done
