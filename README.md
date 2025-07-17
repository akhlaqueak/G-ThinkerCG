# G-ThinkerCG: A Hybrid CPU-GPU Framework for Efficient Parallel Subgraph Finding

## Requirements

* C++11
* NVIDA GPU A100/P100

## Running Maximal Clique Application

Go to app_mc
`make`
`./ru -dg ../graphs/soc-amazon.bin`


## Running Maximal Subgraph Matching

Go to app_gmatch
`make`
`./ru -dg ../graphs/soc-amazon.bin -q 0`
