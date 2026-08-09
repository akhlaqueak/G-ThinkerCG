#include <string>
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include "src/gpu_match.cuh"

using namespace std;
using namespace STMatch;

static void print_usage(const char* program) {
  cerr << "Usage:\n";
  cerr << "  " << program << " <stmatch_graph_prefix> <pattern.g>\n";
  cerr << "  " << program << " -dg <graph.bin> -q <app_gmatch_query_id>\n";
}

static char* get_option(int argc, char* argv[], const string& option) {
  for (int i = 1; i < argc; ++i) {
    if (argv[i] != nullptr && string(argv[i]) == option) {
      if (i + 1 >= argc) {
        throw invalid_argument("missing value for " + option);
      }
      return argv[i + 1];
    }
  }
  return nullptr;
}

int main(int argc, char* argv[]) {

  cudaSetDevice(0);

  string graph_file;
  string pattern_file;
  int gmatch_query_id = -1;
  bool use_gmatch_query = false;

  try {
    char* dg = get_option(argc, argv, "-dg");
    char* q = get_option(argc, argv, "-q");

    if (dg != nullptr || q != nullptr) {
      if (dg == nullptr || q == nullptr) {
        print_usage(argv[0]);
        return 1;
      }
      graph_file = dg;
      gmatch_query_id = atoi(q);
      use_gmatch_query = true;
    }
    else {
      if (argc < 3) {
        print_usage(argv[0]);
        return 1;
      }
      graph_file = argv[1];
      pattern_file = argv[2];
    }
  }
  catch (const exception& e) {
    cerr << e.what() << endl;
    print_usage(argv[0]);
    return 1;
  }

  STMatch::GraphPreprocessor g(graph_file);
  STMatch::PatternPreprocessor p = use_gmatch_query
      ? STMatch::PatternPreprocessor(gmatch_query_id)
      : STMatch::PatternPreprocessor(pattern_file);

  // copy graph and pattern to GPU global memory
  Graph* gpu_graph = g.to_gpu();
  Pattern* gpu_pattern = p.to_gpu();
  JobQueue* gpu_queue = JobQueuePreprocessor(g.g, p).to_gpu();
  CallStack* gpu_callstack;

  // allocate the callstack for all warps in global memory
  graph_node_t* slot_storage;
  cudaMalloc(&slot_storage, sizeof(graph_node_t) * NWARPS_TOTAL * MAX_SLOT_NUM * UNROLL * GRAPH_DEGREE);
  //cout << "global memory usage: " << sizeof(graph_node_t) * NWARPS_TOTAL * MAX_SLOT_NUM * UNROLL * GRAPH_DEGREE / 1024.0 / 1024 / 1024 << " GB" << endl;

  std::vector<CallStack> stk(NWARPS_TOTAL);

  for (int i = 0; i < NWARPS_TOTAL; i++) {
    auto& s = stk[i];
    memset(s.iter, 0, sizeof(s.iter));
    memset(s.slot_size, 0, sizeof(s.slot_size));
    s.slot_storage = (graph_node_t(*)[UNROLL][GRAPH_DEGREE])((char*)slot_storage + i * sizeof(graph_node_t) * MAX_SLOT_NUM * UNROLL * GRAPH_DEGREE);
  }
  cudaMalloc(&gpu_callstack, NWARPS_TOTAL * sizeof(CallStack));
  cudaMemcpy(gpu_callstack, stk.data(), sizeof(CallStack) * NWARPS_TOTAL, cudaMemcpyHostToDevice);

  size_t* gpu_res;
  cudaMalloc(&gpu_res, sizeof(size_t) * NWARPS_TOTAL);
  cudaMemset(gpu_res, 0, sizeof(size_t) * NWARPS_TOTAL);
  size_t* res = new size_t[NWARPS_TOTAL];

  int* idle_warps;
  cudaMalloc(&idle_warps, sizeof(int) * GRID_DIM);
  cudaMemset(idle_warps, 0, sizeof(int) * GRID_DIM);

  int* idle_warps_count;
  cudaMalloc(&idle_warps_count, sizeof(int));
  cudaMemset(idle_warps_count, 0, sizeof(int));

  int* global_mutex;
  cudaMalloc(&global_mutex, sizeof(int) * GRID_DIM);
  cudaMemset(global_mutex, 0, sizeof(int) * GRID_DIM);

  bool* stk_valid;
  cudaMalloc(&stk_valid, sizeof(bool) * GRID_DIM);
  cudaMemset(stk_valid, 0, sizeof(bool) * GRID_DIM);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);

  //cout << "shared memory usage: " << sizeof(Graph) << " " << sizeof(Pattern) << " " << sizeof(JobQueue) << " " << sizeof(CallStack) * NWARPS_PER_BLOCK << " " << NWARPS_PER_BLOCK * 33 * sizeof(int) << " Bytes" << endl;

  _parallel_match << <GRID_DIM, BLOCK_DIM >> > (gpu_graph, gpu_pattern, gpu_callstack, gpu_queue, gpu_res, idle_warps, idle_warps_count, global_mutex);


  cudaEventRecord(stop);

  cudaEventSynchronize(stop);

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  //printf("matching time: %f ms\n", milliseconds);

  cudaMemcpy(res, gpu_res, sizeof(size_t) * NWARPS_TOTAL, cudaMemcpyDeviceToHost);

  unsigned long long tot_count = 0;
  for (int i=0; i<NWARPS_TOTAL; i++) tot_count += res[i];

  if(!LABELED) tot_count = tot_count * p.PatternMultiplicity;
  
  if (use_gmatch_query) {
    printf("Q%d\t%f\t%llu\n", gmatch_query_id, milliseconds, tot_count);
  }
  else {
    printf("%s\t%f\t%llu\n", pattern_file.c_str(), milliseconds, tot_count);
  }
  //cout << "count: " << tot_count << endl;
  return 0;
}
