#ifndef GLOBAL_H
#define GLOBAL_H

#include <vector>
#include <queue>
#include <chrono>
#include <atomic>
#include <algorithm>
#include <map>
#include <iostream>
#include <set>
#include <fstream>
#include <unordered_map>
#include <unordered_set>
#include <stack>
#include <mutex>
#include <thread>
#include <condition_variable>
#include <iomanip>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <omp.h>
#include <cuda_runtime.h>
#include <shared_mutex>

using namespace std;
using namespace std::chrono;

typedef unsigned int Index;
typedef unsigned int VertexID;
typedef char Label;
typedef unsigned long long int ULL;
typedef unsigned int ui;
typedef unsigned long long int ull;
typedef unsigned int uintV;
typedef unsigned long long int uintE;

#include "common/meta.h"
#include "device/cuda_context.h"
#include "util.h"
#include "graph.h"
#include "task.h"

#define BLK_NUMS 56
#define BLK_DIM 1024
// // A100
// #define BLK_NUMS 108
// #define BLK_DIM 1024

#define WARPS_EACH_BLK (BLK_DIM >> 5)
#define N_THREADS (BLK_DIM * BLK_NUMS)
#define N_WARPS (BLK_NUMS * WARPS_EACH_BLK)
#define GLBUFFER_SIZE 1000000
#define THID threadIdx.x
#define WARP_SIZE 32


#define WARPID (THID >> 5)
#define LANEID (THID & 31)
#define BLKID blockIdx.x
#define FULL 0xFFFFFFFF
#define GLWARPID (BLKID * WARPS_EACH_BLK + WARPID)
#define GTHID (BLKID * N_THREADS + THID)

__device__ ui eta=1000*N_WARPS;

#define HOST_BUFF_SZ 20'000'000'000ULL
#define HOST_OFFSET_SZ 2'000'000'000ULL

#define DEV __device__
#define DEVHOST __device__ __host__


// #define SRC
#define DST
#define LO_SPILL_THRESH 1000

// Global task queue
void *global_SC;

// std::shared_mutex SC_mtx;
// share_mutex is not available on CUDA at Polaris
std::shared_timed_mutex SC_mtx;

// Number of compers. Compers means threads
size_t num_cpu_workers;
size_t num_gpu_workers;

// Number of tasks assigned to each comper
size_t tasks_per_fetch_g = 50;
size_t tasks_per_fetch_gpu_worker_g = 50'000;

// no. of tasks moved from gpu to host to be added to Sc
size_t gpu_to_host_transfer_size_g = 1'00'000;

condition_variable cv_master;
bool master_ready = true;
mutex mtx_master;
bool global_end_label;

ThreadSafeQueue<void *> workers_list;

void* global_aggregator = NULL;

#endif