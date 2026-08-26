# G-ThinkerCG

G-ThinkerCG is a hybrid CPU-GPU framework containing three graph-search applications:

- `app_mc`: maximal clique enumeration
- `app_gmatch`: subgraph matching using built-in query patterns
- `app_qc`: maximal quasi-clique enumeration

## Requirements

- A CUDA-capable NVIDIA GPU for GPU execution
- CUDA Toolkit with `nvcc`
- A C++20 compiler
- OpenMP and POSIX threads
- GNU Make

The supplied Makefiles target NVIDIA A100 GPUs (`sm_80`) by default. The application Makefiles also provide P100 (`sm_60`) targets where applicable.

## Build

From the repository root, build the Bliss dependency and each application:

```bash
make -C common/bliss lib
make -C app_mc
make -C app_gmatch
make -C app_qc
```

Each application produces an executable named `run` in its own directory.

To build a P100 version instead of the default A100 version:

```bash
make -C app_mc p100
make -C app_gmatch p100
make -C app_qc p100
```

## Graph files

`app_mc` and `app_gmatch` read the framework's one-hop binary graph format (`.bin`). `app_qc` reads an expanded graph (`.sbin`) containing both one-hop and two-hop adjacency data.

To create an `.sbin` file from an existing framework `.bin` file:

```bash
cd app_qc
make tools
./binToSer ../graphs/soc-amazon.bin 16
```

The optional final argument is the number of conversion threads. The converter writes `soc-amazon.sbin` in the current directory.

## Shared runtime options

The applications share the following CPU-GPU runtime controls. Defaults that differ by application are shown as QC / MC / GMatch.

| Option | Description | Default |
|---|---|---:|
| `-cpu <n>` | Number of CPU workers | `32` |
| `-gpu <n>` | Number of GPU workers | `1` |
| `-cpuchunk <n>` | Tasks assigned to a CPU worker per fetch | `10 / 200 / 200` |
| `-gpuchunk <n>` | Initial GPU tasks per fetch; QC computes its GPU root chunk separately | `QC: dynamic / MC: 1000000 / GMatch: 100000` |
| `-hg_steal <n>` | Maximum tasks transferred from the shared host queue to a GPU in one steal | `1000000` |
| `-gh_steal <n>` | Maximum tasks transferred from the GPU host buffer to the shared CPU queue in one spill | `1000` |
| `-min_hg_steal <n>` | Minimum shared-queue size required before a GPU steals CPU-generated work | `1000` |
| `-idle_worker_divisor <n>` | Divides the CPU-worker count to determine the GPU-to-host idle-worker threshold | `2` |
| `-eta <n>` | GPU ETA limit per warp | `2000` |
| `-tau <microseconds>` | CPU task-decomposition time threshold | `10 / 1000 / 100000` |
| `-pingpong <mode>` | GPU buffer mode: `0` disabled, `1` enabled with abort, `2` enabled without abort | `1 / 2 / 1` |

The stealing controls are independent. For example:

```bash
-min_hg_steal 2000 -hg_steal 500000
```

This configuration waits until the shared queue contains at least 2,000 tasks, then transfers at most 500,000 tasks to the GPU. `-min_hg_steal` must be greater than zero.

The GPU-to-host idle threshold is `floor(cpu_workers / idle_worker_divisor)`. For example, `-cpu 20 -idle_worker_divisor 2` produces a threshold of 10, while divisor 4 produces a threshold of 5. The divisor must be greater than zero. The spill condition uses a strict comparison, `workers_list.size() > threshold`. The worker list can also contain idle GPU workers, so this is an idle-worker threshold rather than an exact idle-CPU count.

Use `-gpu 0` for CPU-only execution. Use `-cpu 0` for GPU-only execution when the selected application and graph do not require CPU handling of oversized tasks.

## Maximal clique enumeration (`app_mc`)

### Run

```bash
cd app_mc
./run -dg ../graphs/soc-amazon.bin
```

Example with explicit scheduling parameters:

```bash
./run \
  -dg ../graphs/soc-amazon.bin \
  -cpu 32 \
  -gpu 1 \
  -cpuchunk 200 \
  -gpuchunk 1000000 \
  -min_hg_steal 1000 \
  -hg_steal 1000000 \
  -gh_steal 1000 \
  -tau 1000 \
  -pingpong 2
```

### Application options

| Option | Description | Default |
|---|---|---:|
| `-dg <path>` | Input one-hop binary graph | `./data/com-friendster.ungraph.txt.bin` |

The application reports total runtime, the number of maximal cliques, the largest clique size, and the number of GPU tasks spilled to the CPU.

## Subgraph matching (`app_gmatch`)

### Run

Both the graph and query ID are required:

```bash
cd app_gmatch
./run -dg ../graphs/soc-amazon.bin -q 0
```

Query `0` is a triangle. Query `24` is a 4-clique:

```bash
./run -dg ../graphs/soc-amazon.bin -q 24
```

### Application options

| Option | Description | Default |
|---|---|---:|
| `-dg <path>` | Input one-hop binary graph | Required |
| `-q <id>` | Built-in query-pattern ID | Required |
| `-prefixbatch <n>` | Candidate-prefix batch size used by CPU and GPU processing | `100` |
| `-s <strategy>` | Storage strategy; `hybrid` selects prefix storage where useful and `expand` forces expansion | `hybrid` |

### Built-in query IDs

| IDs | Patterns |
|---|---|
| `0` | Triangle |
| `1` | Square |
| `2` | Chordal square |
| `3` | Two-tail triangle |
| `4` | House |
| `5` | Chordal house |
| `6` | Chordal roof |
| `7` | Three triangles |
| `8` | Solar square |
| `9` | Near 5-clique |
| `10` | Four triangles |
| `11` | One-in-three triangles |
| `12` | Near 6-clique |
| `13` | Square on top |
| `14` | Near 7-clique |
| `15` | 5-clique on top |
| `16`, `17` | 5-cycle, 6-cycle |
| `18` | Hourglass |
| `23` | Triangle |
| `24`-`27` | 4-clique through 7-clique |

Query IDs `19`-`22` are placeholders and are not implemented.

The application prints total runtime and the number of matched embeddings.

## Maximal quasi-clique enumeration (`app_qc`)

### Prepare and run

```bash
cd app_qc
make tools
./binToSer ../graphs/soc-amazon.bin 16
./run -f soc-amazon.sbin -g 0.5 -k 10
```

By default, QC counts quasi-cliques without performing the final maximality check. To write maximal quasi-cliques to a file, enable `-rmnonmax`:

```bash
./run \
  -f soc-amazon.sbin \
  -g 0.7 \
  -k 10 \
  -rmnonmax 1 \
  -o qc-output.txt \
  -min_hg_steal 1000
```

### Application options

| Option | Description | Default |
|---|---|---:|
| `-f <path>` | Input expanded binary graph (`.sbin`) | Required |
| `-g <ratio>` | Minimum degree ratio in `[0.5, 1.0]` | `0.5` |
| `-k <n>` | Minimum quasi-clique size; must be greater than 1 | `10` |
| `-o <path>` | Maximal quasi-clique output file when `-rmnonmax` is enabled | `output.txt` |
| `-rmnonmax [0\|1]` | Run the maximality-removal pass and write final results | `0` |
| `-sched <0\|1>` | QC scheduling mode: `0` dynamic, `1` static | `0` |
| `-drop_oversized_tasks <0\|1>` | Drop top-level tasks too large for the GPU task buffer instead of routing them to CPU workers | `0` |
| `-c <n>` | Divisor used to compute the initial GPU root chunk | `4` |
| `-min_gpuchunk <n>` | Lower bound for the computed initial GPU root chunk | `1000` |

QC reports preprocessing and search time, the count before maximality checking, the largest result size, oversized-root instrumentation, and—when enabled—the final maximal quasi-clique count.

Oversized roots are normally sent to CPU workers. A QC run with `-cpu 0` fails if such roots exist. Enabling `-drop_oversized_tasks 1` discards them and therefore changes the result set.

## Capturing output

All applications write progress and summary information to standard output. Redirect it to a log when running experiments:

```bash
./run [options] > run.log 2>&1
```

For the complete options recognized by an application, consult its startup help and the corresponding `main.cu`.
