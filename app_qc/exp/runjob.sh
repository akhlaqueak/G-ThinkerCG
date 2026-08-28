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

    run_case() {
        local logfile="$1"
        shift

        echo "Running $logfile"

        if [ -f "$logfile" ] && grep -q "Total time" "$logfile"; then
            echo "Skipping $logfile (already completed)"
            return
        fi

        {
            echo "CMD: $*"
            echo "START: $(date)"
            echo
        } > "$logfile"
        sleep 5s
        if timeout 30m "$@" >> "$logfile" 2>&1; then
            {
                echo
                echo "STATUS: OK"
                echo "END: $(date)"
            } >> "$logfile"
        else
            rc=$?
            {
                echo
                if [ "$rc" -eq 124 ]; then
                    echo "STATUS: TIMEOUT after 10m"
                else
                    echo "STATUS: FAILED (exit code $rc)"
                fi
                echo "END: $(date)"
            } >> "$logfile"
        fi
    }
    run_experiments() {
        local run="$1"

        for gh in 100 500 1000 5000 10000; do
            while IFS=$'\t' read -r ds k g; do
                [ -z "$ds" ] && continue
                fname="$ds-$k-$g-gh-$gh.log"

                if [ "$run" -eq 1 ]; then
                    run_case "$fname" \
                        ./run -f "/home/akhlaque.ak@gmail.com/G-ThinkerCG/datasets/$ds.sbin" \
                        -k "$k" -g "$g" -gh_steal "$gh"
                else
                    if grep -q "Total time" "$fname" 2>/dev/null; then
                        grep "Total time" "$fname" | awk '{printf "%s ", $NF}'
                    else
                        echo -n "X "
                    fi
                fi

                echo
            done < ds.txt
            echo "----"
        done
    }

    run_experiments 1
    run_experiments 0
