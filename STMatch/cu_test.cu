#include <string>
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include "src/gpu_match.cuh"

using namespace std;
using namespace STMatch;

static void check_cuda(cudaError_t err, const char* call) {
  if (err != cudaSuccess) {
    cerr << "CUDA error after " << call << ": " << cudaGetErrorString(err) << endl;
    exit(1);
  }
}

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

static string format_count(unsigned long long value) {
  string digits = to_string(value);
  string formatted;
  formatted.reserve(digits.size() + digits.size() / 3);
  const size_t first_group = digits.size() % 3;
  for (size_t i = 0; i < digits.size(); ++i) {
    if (i > 0 && (i - first_group) % 3 == 0) {
      formatted.push_back(',');
    }
    formatted.push_back(digits[i]);
  }
  return formatted;
}

int main(int argc, char* argv[]) {

  check_cuda(cudaSetDevice(0), "cudaSetDevice");

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

  JobQueuePreprocessor queue_preprocessor(g.g, p);

  // copy graph and pattern to GPU global memory
  Graph* gpu_graph = g.to_gpu();
  Pattern* gpu_pattern = p.to_gpu();
  JobQueue* gpu_queue = queue_preprocessor.to_gpu();
  CallStack* gpu_callstack;

  // allocate the callstack for all warps in global memory
  graph_node_t* slot_storage;
  const size_t slot_storage_bytes = sizeof(graph_node_t) * NWARPS_TOTAL * MAX_SLOT_NUM * UNROLL * GRAPH_DEGREE;
  check_cuda(cudaMalloc(&slot_storage, slot_storage_bytes), "cudaMalloc(slot_storage)");
  //cout << "global memory usage: " << sizeof(graph_node_t) * NWARPS_TOTAL * MAX_SLOT_NUM * UNROLL * GRAPH_DEGREE / 1024.0 / 1024 / 1024 << " GB" << endl;

  std::vector<CallStack> stk(NWARPS_TOTAL);

  for (int i = 0; i < NWARPS_TOTAL; i++) {
    auto& s = stk[i];
    memset(s.iter, 0, sizeof(s.iter));
    memset(s.slot_size, 0, sizeof(s.slot_size));
    s.slot_storage = (graph_node_t(*)[UNROLL][GRAPH_DEGREE])((char*)slot_storage + i * sizeof(graph_node_t) * MAX_SLOT_NUM * UNROLL * GRAPH_DEGREE);
  }
  const size_t callstack_bytes = NWARPS_TOTAL * sizeof(CallStack);
  check_cuda(cudaMalloc(&gpu_callstack, callstack_bytes), "cudaMalloc(gpu_callstack)");
  check_cuda(cudaMemcpy(gpu_callstack, stk.data(), callstack_bytes, cudaMemcpyHostToDevice),
             "cudaMemcpy(gpu_callstack)");

  size_t* gpu_res;
  check_cuda(cudaMalloc(&gpu_res, sizeof(size_t) * NWARPS_TOTAL), "cudaMalloc(gpu_res)");
  check_cuda(cudaMemset(gpu_res, 0, sizeof(size_t) * NWARPS_TOTAL), "cudaMemset(gpu_res)");
  size_t* res = new size_t[NWARPS_TOTAL];

  int* idle_warps;
  check_cuda(cudaMalloc(&idle_warps, sizeof(int) * GRID_DIM), "cudaMalloc(idle_warps)");
  check_cuda(cudaMemset(idle_warps, 0, sizeof(int) * GRID_DIM), "cudaMemset(idle_warps)");

  int* idle_warps_count;
  check_cuda(cudaMalloc(&idle_warps_count, sizeof(int)), "cudaMalloc(idle_warps_count)");
  check_cuda(cudaMemset(idle_warps_count, 0, sizeof(int)), "cudaMemset(idle_warps_count)");

  int* global_mutex;
  check_cuda(cudaMalloc(&global_mutex, sizeof(int) * GRID_DIM), "cudaMalloc(global_mutex)");
  check_cuda(cudaMemset(global_mutex, 0, sizeof(int) * GRID_DIM), "cudaMemset(global_mutex)");

  bool* stk_valid;
  check_cuda(cudaMalloc(&stk_valid, sizeof(bool) * GRID_DIM), "cudaMalloc(stk_valid)");
  check_cuda(cudaMemset(stk_valid, 0, sizeof(bool) * GRID_DIM), "cudaMemset(stk_valid)");

  cudaEvent_t start, stop;
  check_cuda(cudaEventCreate(&start), "cudaEventCreate(start)");
  check_cuda(cudaEventCreate(&stop), "cudaEventCreate(stop)");

  check_cuda(cudaEventRecord(start), "cudaEventRecord(start)");

  //cout << "shared memory usage: " << sizeof(Graph) << " " << sizeof(Pattern) << " " << sizeof(JobQueue) << " " << sizeof(CallStack) * NWARPS_PER_BLOCK << " " << NWARPS_PER_BLOCK * 33 * sizeof(int) << " Bytes" << endl;

  _parallel_match << <GRID_DIM, BLOCK_DIM >> > (gpu_graph, gpu_pattern, gpu_callstack, gpu_queue, gpu_res, idle_warps, idle_warps_count, global_mutex);
  check_cuda(cudaGetLastError(), "kernel launch");


  check_cuda(cudaEventRecord(stop), "cudaEventRecord(stop)");

  check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");

  float milliseconds = 0;
  check_cuda(cudaEventElapsedTime(&milliseconds, start, stop), "cudaEventElapsedTime");
  //printf("matching time: %f ms\n", milliseconds);

  check_cuda(cudaMemcpy(res, gpu_res, sizeof(size_t) * NWARPS_TOTAL, cudaMemcpyDeviceToHost),
             "cudaMemcpy(gpu_res)");

  unsigned long long tot_count = 0;
  for (int i=0; i<NWARPS_TOTAL; i++) tot_count += res[i];

  if(!LABELED && !use_gmatch_query) {
    tot_count = tot_count * p.PatternMultiplicity;
  }

  cout << "Total time (s): " << fixed << setprecision(6) << milliseconds / 1000.0 << endl;
  cout << "Total count: " << format_count(tot_count) << endl;
  return 0;
}
