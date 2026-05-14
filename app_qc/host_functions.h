#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <set>
#include <iostream>
#include <fstream>
#include <string>
#include <cstdio>
#include <sstream>
#include <cmath>
#include <limits>
#include <time.h>
#include <chrono>
#include <sys/timeb.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <sm_30_intrinsics.h>
#include <device_atomic_functions.h>
using namespace std;



// GPU KERNEL LAUNCH
#define BLOCK_SIZE 1024
#define NUM_OF_BLOCKS 108
#define WARP_SIZE 32

// GPU INFORMATION
#define IDX ((blockIdx.x * blockDim.x) + threadIdx.x)
#define WARP_IDX (IDX / WARP_SIZE)
#define LANE_IDX (IDX % WARP_SIZE)
#define WIB_IDX (threadIdx.x / WARP_SIZE)
#define WARPS_PER_BLOCK (BLOCK_SIZE / WARP_SIZE)
#define NUMBER_OF_WARPS (NUM_OF_BLOCKS * WARPS_PER_BLOCK)
#define NUMBER_OF_THREADS (NUM_OF_BLOCKS * BLOCK_SIZE)

// DATA STRUCTURE SIZE
#define TASKS_SIZE 2500
#define TASKS_PER_WARP 100
#define BUFFER_SIZE 900000000
#define BUFFER_OFFSET_SIZE 90000000
#define CLIQUES_SIZE 1'000'000'000
#define CLIQUES_OFFSET_SIZE 30'000'000
#define CLIQUES_PERCENT 50
// per warp
#define WTASKS_SIZE 150000L
#define WTASKS_OFFSET_SIZE 10000
// global memory vertices, should be a multiple of 32 as to not waste space
#define WVERTICES_SIZE 32000
// shared memory vertices
#define VERTICES_SIZE 70

#define EXPAND_THRESHOLD (TASKS_PER_WARP * NUMBER_OF_WARPS)
#define CLIQUES_DUMP (CLIQUES_SIZE * (CLIQUES_PERCENT / 100.0))
 
// PROGRAM RUN SETTINGS
// cpu settings
#define CPU_LEVELS 1
#define CPU_EXPAND_THRESHOLD 1
// whether the program will run entirely on the CPU or not, 0-CPU/GPU 1-CPU only
#define CPU_MODE 0

// debug toggle 0-normal/1-debug
#define DEBUG_TOGGLE 0

struct Vertex
{
    int vertexid;
    // labels: 0 -> candidate, 1 -> member, 2 -> covered vertex, 3 -> cover vertex, 4 -> critical adjacent vertex
    int label;
    int indeg;
    int exdeg;
    int lvl2adj;
};

// CPU GRAPH / CONSTRUCTOR
class CPU_Graph
{
    public:

    int number_of_vertices;
    uint64_t number_of_edges;
    uint64_t number_of_lvl2adj;

    // one dimentional arrays of 1hop and 2hop neighbors and the offsets for each vertex
    int* onehop_neighbors;
    uint64_t* onehop_offsets;
    int* twohop_neighbors;
    uint64_t* twohop_offsets;

    CPU_Graph(ifstream& graph_stream)
    {
        uint64_t edge_count_u64 = 0;
        graph_stream.read(reinterpret_cast<char*>(&number_of_vertices), sizeof(number_of_vertices));
        graph_stream.read(reinterpret_cast<char*>(&edge_count_u64), sizeof(edge_count_u64));
        graph_stream.read(reinterpret_cast<char*>(&number_of_lvl2adj), sizeof(number_of_lvl2adj));
        if (!graph_stream.good())
        {
            throw runtime_error("Failed to read graph header");
        }
        number_of_edges = edge_count_u64;

        if (number_of_vertices <= 0 ||
            edge_count_u64 > static_cast<uint64_t>(numeric_limits<size_t>::max() / sizeof(int)) ||
            number_of_lvl2adj > static_cast<uint64_t>(numeric_limits<size_t>::max() / sizeof(int)))
        {
            graph_stream.clear();
            graph_stream.seekg(0, ios::beg);

            size_t uintV_size = 0;
            size_t uintE_size = 0;
            size_t vertex_count = 0;
            size_t edge_count = 0;
            graph_stream.read(reinterpret_cast<char*>(&uintV_size), sizeof(uintV_size));
            graph_stream.read(reinterpret_cast<char*>(&uintE_size), sizeof(uintE_size));
            graph_stream.read(reinterpret_cast<char*>(&vertex_count), sizeof(vertex_count));
            graph_stream.read(reinterpret_cast<char*>(&edge_count), sizeof(edge_count));

            if (graph_stream.good() &&
                uintV_size == sizeof(int) &&
                uintE_size == sizeof(uint64_t))
            {
                throw runtime_error("Graph file is in the original 1-hop binary format. Run binToSer.cpp to generate the expanded binary graph first.");
            }

            throw runtime_error("Graph file is not in the expected expanded binary format.");
        }

        onehop_neighbors = new int[number_of_edges];
        onehop_offsets = new uint64_t[number_of_vertices + 1];
        twohop_neighbors = new int[number_of_lvl2adj];
        twohop_offsets = new uint64_t[number_of_vertices + 1];

        graph_stream.read(reinterpret_cast<char*>(onehop_neighbors), sizeof(int) * number_of_edges);
        graph_stream.read(reinterpret_cast<char*>(onehop_offsets), sizeof(uint64_t) * (number_of_vertices + 1));
        graph_stream.read(reinterpret_cast<char*>(twohop_neighbors), sizeof(int) * number_of_lvl2adj);
        graph_stream.read(reinterpret_cast<char*>(twohop_offsets), sizeof(uint64_t) * (number_of_vertices + 1));
        assert(graph_stream.good() || graph_stream.eof());
    }

    ~CPU_Graph() 
    {
        delete[] onehop_neighbors;
        delete[] onehop_offsets;
        delete[] twohop_neighbors;
        delete[] twohop_offsets;
    }
};

// CPU DATA
struct CPU_Data
{
    uint64_t* buffer_count;
    uint64_t* buffer_offset;
    Vertex* buffer_vertices;

    uint64_t* current_level;
    bool* maximal_expansion;
    bool* dumping_cliques;

    int* vertex_order_map;
    int* remaining_candidates;
    int* removed_candidates;
    int* remaining_count;
    int* removed_count;
    int* candidate_indegs;

    Vertex* initial_vertices;
    size_t initial_vertices_count;
    int* initial_order_map;
};

// CPU CLIQUES
struct CPU_Cliques
{
    uint64_t cliques_count = 0;
    std::vector<uint64_t> cliques_offset;
    std::vector<int> cliques_vertex;
};

// DEVICE DATA
struct GPU_Data
{
    // GPU DATA
    uint64_t* current_level;

    uint64_t* buffer_count;
    uint64_t* buffer_offset;
    Vertex* buffer_vertices;

    uint64_t* wtasks_count;
    uint64_t* wtasks_offset;
    Vertex* wtasks_vertices;

    Vertex* global_vertices;
    Vertex* initial_vertices;
    uint64_t* initial_vertices_count;
    int* initial_order_map;

    int* removed_candidates;
    int* lane_removed_candidates;

    Vertex* remaining_candidates;
    int* lane_remaining_candidates;

    int* candidate_indegs;
    int* lane_candidate_indegs;

    int* adjacencies;

    int* total_tasks;

    double* minimum_degree_ratio;
    int* minimum_degrees;
    int* minimum_clique_size;
    int* scheduling_toggle;
    bool* store_cliques;

    uint64_t* buffer_offset_start;
    uint64_t* buffer_start;

    // GPU GRAPH
    int* number_of_vertices;
    uint64_t* number_of_edges;

    int* onehop_neighbors;
    uint64_t* onehop_offsets;
    int* twohop_neighbors;
    uint64_t* twohop_offsets;

    // GPU CLIQUES
    uint64_t* cliques_count;
    uint64_t* cliques_vertex_count;
    // cliques_offset stores each clique's vertex start; cliques_size stores its length.
    uint64_t* cliques_offset;
    uint64_t* cliques_size;
    int* cliques_vertex;

    // task scheduling
    int* current_task;
};

// WARP DATA
struct Warp_Data
{
    uint64_t start[WARPS_PER_BLOCK];
    uint64_t end[WARPS_PER_BLOCK];
    int tot_vert[WARPS_PER_BLOCK];
    int num_mem[WARPS_PER_BLOCK];
    int number_of_covered[WARPS_PER_BLOCK];
    int num_cand[WARPS_PER_BLOCK];
    int expansions[WARPS_PER_BLOCK];

    int number_of_members[WARPS_PER_BLOCK];
    int number_of_candidates[WARPS_PER_BLOCK];
    int total_vertices[WARPS_PER_BLOCK];

    Vertex shared_vertices[VERTICES_SIZE * WARPS_PER_BLOCK];

    int removed_count[WARPS_PER_BLOCK];
    int remaining_count[WARPS_PER_BLOCK];
    int num_val_cands[WARPS_PER_BLOCK];
    int rw_counter[WARPS_PER_BLOCK];

    int min_ext_deg[WARPS_PER_BLOCK];
    int lower_bound[WARPS_PER_BLOCK];
    int upper_bound[WARPS_PER_BLOCK];

    int tightened_upper_bound[WARPS_PER_BLOCK];
    int min_clq_indeg[WARPS_PER_BLOCK];
    int min_indeg_exdeg[WARPS_PER_BLOCK];
    int min_clq_totaldeg[WARPS_PER_BLOCK];
    int sum_clq_indeg[WARPS_PER_BLOCK];
    int sum_candidate_indeg[WARPS_PER_BLOCK];

    bool invalid_bounds[WARPS_PER_BLOCK];
    bool success[WARPS_PER_BLOCK];

    int number_of_crit_adj[WARPS_PER_BLOCK];

    // for dynamic intersection
    int count[WARPS_PER_BLOCK];
};

// LOCAL DATA
struct Local_Data
{
    Vertex* vertices;
};



// METHODS
// general
void calculate_minimum_degrees(CPU_Graph& hg);
void search(CPU_Graph& hg, ofstream& temp_results);
void allocate_memory(CPU_Data& hd, GPU_Data& dd, CPU_Cliques& hc, CPU_Graph& hg);
void initialize_tasks(CPU_Graph& hg, CPU_Data& hd);
void move_to_gpu(CPU_Data& hd, GPU_Data& dd);
uint64_t dump_cliques(CPU_Cliques& hc, GPU_Data& dd, ofstream& output_file);
void flush_cliques(CPU_Cliques& hc, ofstream& temp_results);
void free_memory(CPU_Data& hd, GPU_Data& dd, CPU_Cliques& hc);
int RemoveNonMax(const char* szset_filename, const char* szoutput_filename);

// expansion
void h_write_to_tasks(CPU_Data& hd, Vertex* vertices, int total_vertices, Vertex* write_vertices, uint64_t* write_offsets, uint64_t* write_count);
void h_fill_from_buffer(CPU_Data& hd, Vertex* write_vertices, uint64_t* write_offsets, uint64_t* write_count, int threshold);

// helper
int h_sort_vert_cv(const void* a, const void* b);
int h_sort_vert_Q(const void* a, const void* b);
int h_sort_desc(const void* a, const void* b);
inline int h_get_mindeg(int clique_size);
inline bool h_cand_isvalid_LU(Vertex vertex, int clique_size, int upper_bound, int lower_bound, int min_ext_deg);
inline bool  h_vert_isextendable_LU(Vertex vertex, int clique_size, int upper_bound, int lower_bound, int min_ext_deg);


// debug
void print_CPU_Data(CPU_Data& hd);
void print_GPU_Data(GPU_Data& dd);
void print_CPU_Graph(CPU_Graph& hg);
void print_GPU_Graph(GPU_Data& dd, CPU_Graph& hg);
void print_WTask_Buffers(GPU_Data& dd);
void print_GPU_Cliques(GPU_Data& dd); 
void print_CPU_Cliques(CPU_Cliques& hc);
bool print_Data_Sizes(GPU_Data& dd);
void h_print_Data_Sizes(CPU_Data& hd, CPU_Cliques& hc);
void print_vertices(Vertex* vertices, int size);
bool print_Data_Sizes_Every(GPU_Data& dd, int every);
bool print_Warp_Data_Sizes(GPU_Data& dd);
void print_All_Warp_Data_Sizes(GPU_Data& dd);
bool print_Warp_Data_Sizes_Every(GPU_Data& dd, int every);
void print_All_Warp_Data_Sizes_Every(GPU_Data& dd, int every);
void initialize_maxes();
void print_maxes();



// expansion
__device__ int d_lookahead_pruning(GPU_Data& dd, Warp_Data& wd, Local_Data& ld);
__device__ int d_remove_one_vertex(GPU_Data& dd, Warp_Data& wd, Local_Data& ld);
__device__ int d_add_one_vertex(GPU_Data& dd, Warp_Data& wd, Local_Data& ld);
__device__ int d_critical_vertex_pruning(GPU_Data& dd, Warp_Data& wd, Local_Data& ld);
__device__ void d_check_for_clique(GPU_Data& dd, Warp_Data& wd, Local_Data& ld);
__device__ void d_write_to_tasks(GPU_Data& dd, Warp_Data& wd, Local_Data& ld);
__device__ void d_diameter_pruning(GPU_Data& dd, Warp_Data& wd, Local_Data& ld, int pvertexid);
__device__ void d_diameter_pruning_cv(GPU_Data& dd, Warp_Data& wd, Local_Data& ld, int number_of_crit_adj);
__device__ void d_calculate_LU_bounds(GPU_Data& dd, Warp_Data& wd, Local_Data& ld, int number_of_candidates);
__device__ bool d_degree_pruning(GPU_Data& dd, Warp_Data& wd, Local_Data& ld);

// helper
__device__ void d_sort(Vertex* target, int size, int (*func)(Vertex&, Vertex&));
__device__ void d_sort_i(int* target, int size, int (*func)(int, int));
__device__ int d_sort_vert_Q(Vertex& v1, Vertex& v2);
__device__ int d_sort_vert_cv(Vertex& v1, Vertex& v2);
__device__ int d_sort_degs(int n1, int n2);
__device__ int d_bsearch_array(int* search_array, int array_size, int search_number);
__device__ bool d_cand_isvalid_LU(Vertex& vertex, GPU_Data& dd, Warp_Data& wd, Local_Data& ld);
__device__ bool d_vert_isextendable_LU(Vertex& vertex, GPU_Data& dd, Warp_Data& wd, Local_Data& ld);
__device__ int d_get_mindeg(int number_of_members, GPU_Data& dd);

// debug
__device__ void d_print_vertices(Vertex* vertices, int size);



// TODO (HIGH PRIORITY)
// - 

// TODO (LOW PRIORITY - these are all not worth the time to do)
// - reevaluate and change where uint64_t's are used
// - label for vertices can be a byte rather than int
// - don't need lvl2adj in all places anymore
// - look for places where we can break early
// - examine code for unnecessary syncs on the GPU
// - in degree pruning see if we can remove failed_found by consolidating with success
// - see whether it's possible to parallelize some of calculate_LU_bounds
// - remove device expand level code duplication by using a method



// DEBUG - MAX TRACKER VARIABLES
uint64_t mts, mbs, mbo, mcs, mco, wts, wto, mvs;



// COMMAND LINE INPUT VARIABLES
double minimum_degree_ratio;
int minimum_clique_size;
int* minimum_degrees;
int scheduling_toggle;
bool store_cliques;
std::vector<int> output_vertex_id_map;

inline int output_vertex_id(int vertexid)
{
    if (vertexid >= 0 && static_cast<size_t>(vertexid) < output_vertex_id_map.size())
        return output_vertex_id_map[vertexid];
    return vertexid;
}

inline void load_output_vertex_id_map(const std::string &graph_file)
{
    output_vertex_id_map.clear();

    std::ifstream map_stream(graph_file + ".origmap", std::ios::binary);
    if (!map_stream.is_open())
        return;

    size_t uintV_size = 0;
    size_t vertex_count = 0;
    map_stream.read(reinterpret_cast<char *>(&uintV_size), sizeof(uintV_size));
    map_stream.read(reinterpret_cast<char *>(&vertex_count), sizeof(vertex_count));

    if (!map_stream || uintV_size != sizeof(unsigned int) ||
        vertex_count > static_cast<size_t>(std::numeric_limits<int>::max()))
    {
        output_vertex_id_map.clear();
        return;
    }

    std::vector<unsigned int> raw_map(vertex_count);
    map_stream.read(reinterpret_cast<char *>(raw_map.data()), sizeof(unsigned int) * raw_map.size());
    if (!map_stream)
    {
        output_vertex_id_map.clear();
        return;
    }

    output_vertex_id_map.resize(vertex_count);
    for (size_t i = 0; i < vertex_count; i++)
        output_vertex_id_map[i] = static_cast<int>(raw_map[i]);
}



// --- HOST METHODS --- 

// initializes minimum degrees array 
void calculate_minimum_degrees(CPU_Graph& hg)
{
    minimum_degrees = new int[hg.number_of_vertices + 1];
    minimum_degrees[0] = 0;
    for (int i = 1; i <= hg.number_of_vertices; i++) {
        minimum_degrees[i] = ceil(minimum_degree_ratio * (i - 1));
    }
}



uint64_t dump_cliques(CPU_Cliques& hc, GPU_Data& dd, ofstream& temp_results)
{
    uint64_t dumped_cliques_count = hc.cliques_count;

    if (store_cliques)
    {
        // flush CPU cliques first; otherwise they get overwritten by the GPU copy below
        flush_cliques(hc, temp_results);
    }
    else
    {
        hc.cliques_count = 0;
        hc.cliques_vertex.clear();
        hc.cliques_offset.clear();
    }

    chkerr(cudaDeviceSynchronize());

    uint64_t gpu_cliques_count = 0;
    uint64_t gpu_cliques_size = 0;
    chkerr(cudaMemcpy(&gpu_cliques_count, dd.cliques_count, sizeof(uint64_t), cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(&gpu_cliques_size, dd.cliques_vertex_count, sizeof(uint64_t), cudaMemcpyDeviceToHost));
    chkerr(cudaDeviceSynchronize());

    if (gpu_cliques_count == 0)
    {
        cudaMemset(dd.cliques_count, 0, sizeof(uint64_t));
        cudaMemset(dd.cliques_vertex_count, 0, sizeof(uint64_t));
        return dumped_cliques_count;
    }

    if (!store_cliques)
    {
        cudaMemset(dd.cliques_count, 0, sizeof(uint64_t));
        cudaMemset(dd.cliques_vertex_count, 0, sizeof(uint64_t));
        return dumped_cliques_count + gpu_cliques_count;
    }

    if (gpu_cliques_count > CLIQUES_OFFSET_SIZE)
    {
        throw std::runtime_error("GPU clique count exceeds CLIQUES_OFFSET_SIZE");
    }

    if (gpu_cliques_size > CLIQUES_SIZE)
    {
        throw std::runtime_error("GPU clique vertex size exceeds CLIQUES_SIZE");
    }

    std::vector<uint64_t> gpu_cliques_offset(gpu_cliques_count);
    std::vector<uint64_t> gpu_cliques_sizes(gpu_cliques_count);
    chkerr(cudaMemcpy(gpu_cliques_offset.data(), dd.cliques_offset, sizeof(uint64_t) * gpu_cliques_count, cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(gpu_cliques_sizes.data(), dd.cliques_size, sizeof(uint64_t) * gpu_cliques_count, cudaMemcpyDeviceToHost));

    for (uint64_t i = 0; i < gpu_cliques_count; i++) {
        if (gpu_cliques_offset[i] > gpu_cliques_size || gpu_cliques_sizes[i] > gpu_cliques_size - gpu_cliques_offset[i]) {
            throw std::runtime_error("GPU clique offsets are corrupted");
        }
    }

    std::vector<int> gpu_cliques_vertex(gpu_cliques_size);
    if (!gpu_cliques_vertex.empty())
    {
        chkerr(cudaMemcpy(gpu_cliques_vertex.data(), dd.cliques_vertex, sizeof(int) * gpu_cliques_vertex.size(), cudaMemcpyDeviceToHost));
    }
    chkerr(cudaDeviceSynchronize());

    for (uint64_t i = 0; i < gpu_cliques_count; i++) {
        uint64_t start = gpu_cliques_offset[i];
        uint64_t end = start + gpu_cliques_sizes[i];
        temp_results << end - start << " ";
        for (uint64_t j = start; j < end; j++) {
            temp_results << output_vertex_id(gpu_cliques_vertex[j]) << " ";
        }
        temp_results << "\n";
    }

    cudaMemset(dd.cliques_count, 0, sizeof(uint64_t));
    cudaMemset(dd.cliques_vertex_count, 0, sizeof(uint64_t));
    return dumped_cliques_count + gpu_cliques_count;
}

void flush_cliques(CPU_Cliques& hc, ofstream& temp_results) 
{
    for (size_t i = 0; i + 1 < hc.cliques_offset.size(); i++) {
        uint64_t start = hc.cliques_offset[i];
        uint64_t end = hc.cliques_offset[i + 1];
        temp_results << end - start << " ";
        for (uint64_t j = start; j < end; j++) {
            temp_results << output_vertex_id(hc.cliques_vertex[j]) << " ";
        }
        temp_results << "\n";
    }
    hc.cliques_vertex.clear();
    if (store_cliques)
    {
        hc.cliques_offset.assign(1, 0);
    }
    else
    {
        hc.cliques_offset.clear();
    }
    hc.cliques_count = 0;
}

void free_memory(CPU_Data& hd, GPU_Data& dd, CPU_Cliques& hc)
{
    // GPU GRAPH
    chkerr(cudaFree(dd.number_of_vertices));
    chkerr(cudaFree(dd.number_of_edges));
    chkerr(cudaFree(dd.onehop_neighbors));
    chkerr(cudaFree(dd.onehop_offsets));
    chkerr(cudaFree(dd.twohop_neighbors));
    chkerr(cudaFree(dd.twohop_offsets));

    // CPU DATA
    delete hd.buffer_count;
    delete[] hd.buffer_offset;
    delete[] hd.buffer_vertices;

    delete hd.current_level;
    delete hd.maximal_expansion;
    delete hd.dumping_cliques;

    delete[] hd.vertex_order_map;
    delete[] hd.remaining_candidates;
    delete hd.remaining_count;
    delete[] hd.removed_candidates;
    delete hd.removed_count;
    delete[] hd.candidate_indegs;
    delete[] hd.initial_vertices;
    delete[] hd.initial_order_map;
    hd.initial_vertices = nullptr;
    hd.initial_vertices_count = 0;
    hd.initial_order_map = nullptr;

    // GPU DATA
    chkerr(cudaFree(dd.current_level));

    chkerr(cudaFree(dd.buffer_count));
    chkerr(cudaFree(dd.buffer_offset));
    chkerr(cudaFree(dd.buffer_vertices));

    chkerr(cudaFree(dd.wtasks_count));
    chkerr(cudaFree(dd.wtasks_offset));
    chkerr(cudaFree(dd.wtasks_vertices));

    chkerr(cudaFree(dd.global_vertices));
    chkerr(cudaFree(dd.initial_vertices));
    chkerr(cudaFree(dd.initial_vertices_count));
    chkerr(cudaFree(dd.initial_order_map));

    chkerr(cudaFree(dd.remaining_candidates));
    chkerr(cudaFree(dd.lane_remaining_candidates));

    chkerr(cudaFree(dd.removed_candidates));
    chkerr(cudaFree(dd.lane_removed_candidates));

    chkerr(cudaFree(dd.candidate_indegs));
    chkerr(cudaFree(dd.lane_candidate_indegs));

    chkerr(cudaFree(dd.adjacencies));

    chkerr(cudaFree(dd.minimum_degree_ratio));
    chkerr(cudaFree(dd.minimum_degrees));
    chkerr(cudaFree(dd.minimum_clique_size));
    chkerr(cudaFree(dd.scheduling_toggle));
    chkerr(cudaFree(dd.store_cliques));

    chkerr(cudaFree(dd.total_tasks));

    // CPU CLIQUES
    hc.cliques_count = 0;
    hc.cliques_vertex.clear();
    hc.cliques_vertex.shrink_to_fit();
    hc.cliques_offset.clear();
    hc.cliques_offset.shrink_to_fit();

    // GPU CLIQUES
    chkerr(cudaFree(dd.cliques_count));
    chkerr(cudaFree(dd.cliques_vertex_count));
    if (dd.cliques_vertex)
        chkerr(cudaFree(dd.cliques_vertex));
    if (dd.cliques_offset)
        chkerr(cudaFree(dd.cliques_offset));
    if (dd.cliques_size)
        chkerr(cudaFree(dd.cliques_size));

    chkerr(cudaFree(dd.buffer_offset_start));
    chkerr(cudaFree(dd.buffer_start));
    // tasks scheduling
    chkerr(cudaFree(dd.current_task));
}

void h_write_to_tasks(CPU_Data& hd, Vertex* vertices, int total_vertices, Vertex* write_vertices, uint64_t* write_offsets, uint64_t* write_count)
{
    (*hd.maximal_expansion) = false;

    if ((*write_count) < CPU_EXPAND_THRESHOLD) {
        uint64_t start_write = write_offsets[*write_count];

        for (int k = 0; k < total_vertices; k++) {
            write_vertices[start_write + k].vertexid = vertices[k].vertexid;
            write_vertices[start_write + k].label = vertices[k].label;
            write_vertices[start_write + k].indeg = vertices[k].indeg;
            write_vertices[start_write + k].exdeg = vertices[k].exdeg;
            write_vertices[start_write + k].lvl2adj = 0;
        }
        (*write_count)++;
        write_offsets[*write_count] = start_write + total_vertices;
    }
    else {
        uint64_t start_write = hd.buffer_offset[(*hd.buffer_count)];

        for (int k = 0; k < total_vertices; k++) {
            hd.buffer_vertices[start_write + k].vertexid = vertices[k].vertexid;
            hd.buffer_vertices[start_write + k].label = vertices[k].label;
            hd.buffer_vertices[start_write + k].indeg = vertices[k].indeg;
            hd.buffer_vertices[start_write + k].exdeg = vertices[k].exdeg;
            hd.buffer_vertices[start_write + k].lvl2adj = 0;
        }
        (*hd.buffer_count)++;
        hd.buffer_offset[(*hd.buffer_count)] = start_write + total_vertices;
    }
}

void h_fill_from_buffer(CPU_Data& hd, Vertex* write_vertices, uint64_t* write_offsets, uint64_t* write_count, int threshold)
{
    // read from end of buffer, write to end of tasks, decrement buffer
    (*hd.maximal_expansion) = false;

    // get read and write locations
    int write_amount = ((*hd.buffer_count) >= (threshold - *write_count)) ? threshold - *write_count : (*hd.buffer_count);
    uint64_t start_buffer = hd.buffer_offset[(*hd.buffer_count) - write_amount];
    uint64_t end_buffer = hd.buffer_offset[(*hd.buffer_count)];
    uint64_t size_buffer = end_buffer - start_buffer;
    uint64_t start_write = write_offsets[*write_count];

    // copy tasks data from end of buffer to end of tasks
    memcpy(&write_vertices[start_write], &hd.buffer_vertices[start_buffer], sizeof(Vertex) * size_buffer);

    // handle offsets
    for (int j = 1; j <= write_amount; j++) {
        write_offsets[*write_count + j] = start_write + (hd.buffer_offset[(*hd.buffer_count) - write_amount + j] - start_buffer);
    }

    // update counts
    (*write_count) += write_amount;
    (*hd.buffer_count) -= write_amount;
}



// --- HELPER METHODS ---

// update how this method looks
int h_sort_vert_Q(const void* a, const void* b)
{
    // order is: member -> covered -> cands -> cover
    // keys are: indeg -> exdeg -> lvl2adj -> vertexid
    
    Vertex* v1;
    Vertex* v2;

    v1 = (Vertex*)a;
    v2 = (Vertex*)b;

    if (v1->label == 1 && v2->label != 1)
        return -1;
    else if (v1->label != 1 && v2->label == 1)
        return 1;
    else if (v1->label == 2 && v2->label != 2)
        return -1;
    else if (v1->label != 2 && v2->label == 2)
        return 1;
    else if (v1->label == 0 && v2->label != 0)
        return -1;
    else if (v1->label != 0 && v2->label == 0)
        return 1;
    else if (v1->label == 3 && v2->label != 3)
        return -1;
    else if (v1->label != 3 && v2->label == 3)
        return 1;
    else if (v1->indeg > v2->indeg)
        return -1;
    else if (v1->indeg < v2->indeg)
        return 1;
    else if (v1->exdeg > v2->exdeg)
        return -1;
    else if (v1->exdeg < v2->exdeg)
        return 1;
    else if (v1->lvl2adj > v2->lvl2adj)
        return -1;
    else if (v1->lvl2adj < v2->lvl2adj)
        return 1;
    else if (v1->vertexid > v2->vertexid)
        return -1;
    else if (v1->vertexid < v2->vertexid)
        return 1;
    else
        return 0;
}

int h_sort_vert_cv(const void* a, const void* b)
{
    // but crit adj vertices before candidates

    Vertex* v1;
    Vertex* v2;

    v1 = (Vertex*)a;
    v2 = (Vertex*)b;

    if (v1->label == 4 && v2->label != 4)
        return -1;
    else if (v1->label != 4 && v2->label == 4)
        return 1;
    else
        return 0;
}

// sorts degrees in descending order
int h_sort_desc(const void* a, const void* b) 
{
    int n1;
    int n2;

    n1 = *(int*)a;
    n2 = *(int*)b;

    if (n1 > n2) {
        return -1;
    }
    else if (n1 < n2) {
        return 1;
    }
    else {
        return 0;
    }
}

inline int h_get_mindeg(int clique_size) {
    if (clique_size < minimum_clique_size) {
        return minimum_degrees[minimum_clique_size];
    }
    else {
        return minimum_degrees[clique_size];
    }
}

inline bool h_cand_isvalid_LU(Vertex vertex, int clique_size, int upper_bound, int lower_bound, int min_ext_deg) 
{
    if (vertex.indeg + vertex.exdeg < minimum_degrees[minimum_clique_size]) {
        return false;
    }
    else if (vertex.indeg + vertex.exdeg < h_get_mindeg(clique_size + vertex.exdeg + 1)) {
        return false;
    }
    else if (vertex.indeg + vertex.exdeg < min_ext_deg) {
        return false;
    }
    else if (vertex.indeg + upper_bound - 1 < minimum_degrees[clique_size + lower_bound]) {
        return false;
    }
    else if (vertex.indeg + vertex.exdeg < h_get_mindeg(clique_size + lower_bound)) {
        return false;
    }
    else {
        return true;
    }
}

inline bool h_vert_isextendable_LU(Vertex vertex, int clique_size, int upper_bound, int lower_bound, int min_ext_deg)
{
    if (vertex.indeg + vertex.exdeg < minimum_degrees[minimum_clique_size]) {
        return false;
    }
    else if (vertex.indeg + vertex.exdeg < h_get_mindeg(clique_size + vertex.exdeg)) {
        return false;
    }
    else if (vertex.indeg + vertex.exdeg < min_ext_deg) {
        return false;
    }
    else if (vertex.exdeg == 0 && vertex.indeg < h_get_mindeg(clique_size + vertex.exdeg)) {
        return false;
    }
    else if (vertex.indeg + upper_bound < minimum_degrees[clique_size + upper_bound]) {
        return false;
    }
    else if (vertex.indeg + vertex.exdeg < h_get_mindeg(clique_size + lower_bound)) {
        return false;
    }
    else {
        return true;
    }
}



// --- DEBUG METHODS ---

void print_CPU_Graph(CPU_Graph& hg) {
    cout << endl << " --- (CPU_Graph)host_graph details --- " << endl;
    cout << endl << "|V|: " << hg.number_of_vertices << " |E|: " << hg.number_of_edges << endl;
    cout << endl << "Onehop Offsets:" << endl;
    for (uint64_t i = 0; i <= hg.number_of_vertices; i++) {
        cout << hg.onehop_offsets[i] << " ";
    }
    cout << endl << "Onehop Neighbors:" << endl;
    for (uint64_t i = 0; i < hg.number_of_edges * 2; i++) {
        cout << hg.onehop_neighbors[i] << " ";
    }
    cout << endl << "Twohop Offsets:" << endl;
    for (uint64_t i = 0; i <= hg.number_of_vertices; i++) {
        cout << hg.twohop_offsets[i] << " ";
    }
    cout << endl << "Twohop Neighbors:" << endl;
    for (uint64_t i = 0; i < hg.number_of_lvl2adj; i++) {
        cout << hg.twohop_neighbors[i] << " ";
    }
    cout << endl << endl;
}

void print_GPU_Graph(GPU_Data& dd, CPU_Graph& hg)
{
    int* number_of_vertices = new int;
    int* number_of_edges = new int;

    int* onehop_neighbors = new int[hg.number_of_edges * 2];
    uint64_t * onehop_offsets = new uint64_t[(hg.number_of_vertices)+1];
    int* twohop_neighbors = new int[hg.number_of_lvl2adj];
    uint64_t * twohop_offsets = new uint64_t[(hg.number_of_vertices)+1];

    chkerr(cudaMemcpy(number_of_vertices, dd.number_of_vertices, sizeof(int), cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(number_of_edges, dd.number_of_edges, sizeof(int), cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(onehop_neighbors, dd.onehop_neighbors, sizeof(int)*hg.number_of_edges * 2, cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(onehop_offsets, dd.onehop_offsets, sizeof(uint64_t)*(hg.number_of_vertices+1), cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(twohop_neighbors, dd.twohop_neighbors, sizeof(int)*hg.number_of_lvl2adj, cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(twohop_offsets, dd.twohop_offsets, sizeof(uint64_t)*(hg.number_of_vertices+1), cudaMemcpyDeviceToHost));

    cout << endl << " --- (GPU_Graph)device_graph details --- " << endl;
    cout << endl << "|V|: " << (*number_of_vertices) << " |E|: " << (*number_of_edges) << endl;
    cout << endl << "Onehop Offsets:" << endl;
    for (uint64_t i = 0; i <= (*number_of_vertices); i++) {
        cout << onehop_offsets[i] << " ";
    }
    cout << endl << "Onehop Neighbors:" << endl;
    for (uint64_t i = 0; i < hg.number_of_edges * 2; i++) {
        cout << onehop_neighbors[i] << " ";
    }
    cout << endl << "Twohop Offsets:" << endl;
    for (uint64_t i = 0; i <= (*number_of_vertices); i++) {
        cout << twohop_offsets[i] << " ";
    }
    cout << endl << "Twohop Neighbors:" << endl;
    for (uint64_t i = 0; i < hg.number_of_lvl2adj; i++) {
        cout << twohop_neighbors[i] << " ";
    }
    cout << endl << endl;

    delete number_of_vertices;
    delete number_of_edges;

    delete[] onehop_offsets;
    delete[] onehop_neighbors;
    delete[] twohop_offsets;
    delete[] twohop_neighbors;
}

void print_CPU_Data(CPU_Data& hd)
{
    cout << endl << " --- (CPU_Data)host_data details --- " << endl;
    cout << endl << "Buffer: " << "Size: " << (*(hd.buffer_count)) << endl;
    cout << endl << "Offsets:" << endl;
    for (uint64_t i = 0; i <= (*(hd.buffer_count)); i++) {
        cout << hd.buffer_offset[i] << " ";
    }
    cout << endl << "Vertex:" << endl;
    for (uint64_t i = 0; i < hd.buffer_offset[(*(hd.buffer_count))]; i++) {
        cout << hd.buffer_vertices[i].vertexid << " ";
    }
    cout << endl << "Label:" << endl;
    for (uint64_t i = 0; i < hd.buffer_offset[(*(hd.buffer_count))]; i++) {
        cout << hd.buffer_vertices[i].label << " ";
    }
    cout << endl << "Indeg:" << endl;
    for (uint64_t i = 0; i < hd.buffer_offset[(*(hd.buffer_count))]; i++) {
        cout << hd.buffer_vertices[i].indeg << " ";
    }
    cout << endl << "Exdeg:" << endl;
    for (uint64_t i = 0; i < hd.buffer_offset[(*(hd.buffer_count))]; i++) {
        cout << hd.buffer_vertices[i].exdeg << " ";
    }
    cout << endl << "Lvl2adj:" << endl;
    for (uint64_t i = 0; i < hd.buffer_offset[(*(hd.buffer_count))]; i++) {
        cout << hd.buffer_vertices[i].lvl2adj << " ";
    }
    cout << endl << endl;
}

void print_GPU_Data(GPU_Data& dd)
{
    uint64_t* current_level = new uint64_t;
    uint64_t* buffer_count = new uint64_t;
    uint64_t* buffer_offset = new uint64_t[BUFFER_OFFSET_SIZE];
    Vertex* buffer_vertices = new Vertex[BUFFER_SIZE];

    chkerr(cudaMemcpy(current_level, dd.current_level, sizeof(uint64_t), cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(buffer_count, dd.buffer_count, sizeof(uint64_t), cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(buffer_offset, dd.buffer_offset, (BUFFER_OFFSET_SIZE) * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(buffer_vertices, dd.buffer_vertices, (BUFFER_SIZE) * sizeof(Vertex), cudaMemcpyDeviceToHost));

    cout << " --- (GPU_Data)device_data details --- " << endl;
    cout << endl << "Buffer: Level: " << (*current_level) << " Size: " << (*buffer_count) << endl;
    cout << endl << "Offsets:" << endl;
    for (int i = 0; i <= (*buffer_count); i++) {
        cout << buffer_offset[i] << " " << flush;
    }
    cout << endl << "Vertex:" << endl;
    for (int i = 0; i < buffer_offset[*buffer_count]; i++) {
        cout << buffer_vertices[i].vertexid << " " << flush;
    }
    cout << endl << "Label:" << endl;
    for (int i = 0; i < buffer_offset[*buffer_count]; i++) {
        cout << buffer_vertices[i].label << " " << flush;
    }
    cout << endl << "Indeg:" << endl;
    for (int i = 0; i < buffer_offset[*buffer_count]; i++) {
        cout << buffer_vertices[i].indeg << " " << flush;
    }
    cout << endl << "Exdeg:" << endl;
    for (int i = 0; i < buffer_offset[*buffer_count]; i++) {
        cout << buffer_vertices[i].exdeg << " " << flush;
    }
    cout << endl << "Lvl2adj:" << endl;
    for (int i = 0; i < buffer_offset[*buffer_count]; i++) {
        cout << buffer_vertices[i].lvl2adj << " " << flush;
    }
    cout << endl;

    delete current_level;
    delete buffer_count;
    delete[] buffer_offset;
    delete[] buffer_vertices;
}

// returns true if warp buffer was too small causing error
bool print_Warp_Data_Sizes(GPU_Data& dd)
{
    uint64_t* tasks_counts = new uint64_t[NUMBER_OF_WARPS];
    uint64_t* tasks_sizes = new uint64_t[NUMBER_OF_WARPS];
    int tasks_tcount = 0;
    int tasks_tsize = 0;
    int tasks_mcount = 0;
    int tasks_msize = 0;

    chkerr(cudaMemcpy(tasks_counts, dd.wtasks_count, sizeof(uint64_t) * NUMBER_OF_WARPS, cudaMemcpyDeviceToHost));
    for (int i = 0; i < NUMBER_OF_WARPS; i++) {
        chkerr(cudaMemcpy(tasks_sizes + i, dd.wtasks_offset + (i * WTASKS_OFFSET_SIZE) + tasks_counts[i], sizeof(uint64_t), cudaMemcpyDeviceToHost));
    }

    for (int i = 0; i < NUMBER_OF_WARPS; i++) {
        tasks_tcount += tasks_counts[i];
        if (tasks_counts[i] > tasks_mcount) {
            tasks_mcount = tasks_counts[i];
        }
        tasks_tsize += tasks_sizes[i];
        if (tasks_sizes[i] > tasks_msize) {
            tasks_msize = tasks_sizes[i];
        }
    }

    cout << "WTasks( TC: " << tasks_tcount << " TS: " << tasks_tsize << " MC: " << tasks_mcount << " MS: " << tasks_msize << ")" << endl;

    if (tasks_mcount > wto) {
        wto = tasks_mcount;
    }
    if (tasks_msize > wts) {
        wts = tasks_msize;
    }

    if (tasks_mcount > WTASKS_OFFSET_SIZE || tasks_msize > WTASKS_SIZE) {
        cout << "!!! WBUFFER SIZE ERROR !!!" << endl;
        return true;
    }

    delete[] tasks_counts;
    delete[] tasks_sizes;

    return false;
}

void print_All_Warp_Data_Sizes(GPU_Data& dd)
{
    uint64_t* tasks_counts = new uint64_t[NUMBER_OF_WARPS];
    uint64_t* tasks_sizes = new uint64_t[NUMBER_OF_WARPS];

    chkerr(cudaMemcpy(tasks_counts, dd.wtasks_count, sizeof(uint64_t) * NUMBER_OF_WARPS, cudaMemcpyDeviceToHost));
    for (int i = 0; i < NUMBER_OF_WARPS; i++) {
        chkerr(cudaMemcpy(tasks_sizes + i, dd.wtasks_offset + (i * WTASKS_OFFSET_SIZE) + tasks_counts[i], sizeof(uint64_t), cudaMemcpyDeviceToHost));
    }

    cout << "WTasks Sizes: " << flush;
    for (int i = 0; i < NUMBER_OF_WARPS; i++) {
        cout << i << ":" << tasks_counts[i] << " " << tasks_sizes[i] << " " << flush;
    }

    delete[] tasks_counts;
    delete[] tasks_sizes;
}

bool print_Warp_Data_Sizes_Every(GPU_Data& dd, int every)
{
    bool result = false;
    int level;
    chkerr(cudaMemcpy(&level, dd.current_level, sizeof(int), cudaMemcpyDeviceToHost));
    if (level % every == 0) {
        result = print_Warp_Data_Sizes(dd);
    }
    return result;
}

void print_All_Warp_Data_Sizes_Every(GPU_Data& dd, int every)
{
    int level;
    chkerr(cudaMemcpy(&level, dd.current_level, sizeof(int), cudaMemcpyDeviceToHost));
    if (level % every == 0) {
        print_All_Warp_Data_Sizes(dd);
    }
}

bool print_Data_Sizes_Every(GPU_Data& dd, int every)
{
    bool result = false;
    int level;
    chkerr(cudaMemcpy(&level, dd.current_level, sizeof(int), cudaMemcpyDeviceToHost));
    if (level % every == 0) {
        result = print_Data_Sizes(dd);
    }
    return result;
}

bool print_Data_Sizes(GPU_Data& dd)
{
    uint64_t* current_level = new uint64_t;
    uint64_t* buffer_count = new uint64_t;
    uint64_t* cliques_count = new uint64_t;
    uint64_t* buffer_size = new uint64_t;
    uint64_t* cliques_size = new uint64_t;

    chkerr(cudaMemcpy(current_level, dd.current_level, sizeof(uint64_t), cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(buffer_count, dd.buffer_count, sizeof(uint64_t), cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(cliques_count, dd.cliques_count, sizeof(uint64_t), cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(buffer_size, dd.buffer_offset + (*buffer_count), sizeof(uint64_t), cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(cliques_size, dd.cliques_vertex_count, sizeof(uint64_t), cudaMemcpyDeviceToHost));

    cout << "L: " << (*current_level) << " B: " << (*buffer_count) << " " << (*buffer_size) << " C: " << 
        (*cliques_count) << " " << (*cliques_size) << endl << endl;

    if (*buffer_size > mbs) {
        mbs = *buffer_size;
    }
    if (*buffer_count > mbo) {
        mbo = *buffer_count;
    }
    if (*cliques_size > mcs) {
        mcs = *cliques_size;
    }
    if (*cliques_count > mco) {
        mco = *cliques_count;
    }

    if ((*buffer_count) > BUFFER_OFFSET_SIZE || (*buffer_size) > BUFFER_SIZE ||
        (store_cliques && ((*cliques_count) > CLIQUES_OFFSET_SIZE || (*cliques_size) > CLIQUES_SIZE))) {
        cout << "!!! ARRAY SIZE ERROR !!!" << endl;
        return true;
    }

    delete current_level;
    delete buffer_count;
    delete cliques_count;
    delete buffer_size;
    delete cliques_size;
    
    return false;
}

void h_print_Data_Sizes(CPU_Data& hd, CPU_Cliques& hc)
{
    uint64_t hc_count = hc.cliques_offset.empty() ? 0 : static_cast<uint64_t>(hc.cliques_offset.size() - 1);
    uint64_t hc_size = hc.cliques_offset.empty() ? 0 : hc.cliques_offset.back();
    cout << "L: " << (*hd.current_level) << " B: " << (*hd.buffer_count) << " " << (*(hd.buffer_offset + (*hd.buffer_count))) << " C: " << 
        hc_count << " " << hc_size << endl;

    if ((*(hd.buffer_offset + (*hd.buffer_count))) > mbs) {
        mbs = (*(hd.buffer_offset + (*hd.buffer_count)));
    }
    if ((*hd.buffer_count) > mbo) {
        mbo = (*hd.buffer_count);
    }
    if (hc_size > mcs) {
        mcs = hc_size;
    }
    if (hc_count > mco) {
        mco = hc_count;
    }
}

void print_WTask_Buffers(GPU_Data& dd)
{
    uint64_t* wtasks_count = new uint64_t[NUMBER_OF_WARPS];
    uint64_t* wtasks_offset = new uint64_t[NUMBER_OF_WARPS*WTASKS_OFFSET_SIZE];
    Vertex* wtasks_vertices = new Vertex[NUMBER_OF_WARPS*WTASKS_SIZE];

    chkerr(cudaMemcpy(wtasks_count, dd.wtasks_count, sizeof(uint64_t)*NUMBER_OF_WARPS, cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(wtasks_offset, dd.wtasks_offset, sizeof(uint64_t) * (NUMBER_OF_WARPS*WTASKS_OFFSET_SIZE), cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(wtasks_vertices, dd.wtasks_vertices, sizeof(Vertex) * (NUMBER_OF_WARPS*WTASKS_SIZE), cudaMemcpyDeviceToHost));

    cout << endl << " --- Warp Task Buffers details --- " << endl;
    for (int i = 0; i < NUMBER_OF_WARPS; i++) {
        int wtasks_offset_start = WTASKS_OFFSET_SIZE * i;
        int wtasks_start = WTASKS_SIZE * i;

        cout << endl << "Warp " << i << ": " << "Size : " << wtasks_count[i] << endl;
        if (wtasks_count[i] == 0) {
            continue;
        }
        cout << "Offsets:" << endl;
        for (int j = 0; j <= wtasks_count[i]; j++) {
            cout << wtasks_offset[wtasks_offset_start+j] << " ";
        }
        cout << endl << "Vertex:" << endl;
        for (int j = 0; j < wtasks_offset[wtasks_offset_start+wtasks_count[i]]; j++) {
            cout << wtasks_vertices[wtasks_start+j].vertexid << " ";
        }
        cout << endl << "Label:" << endl;
        for (int j = 0; j < wtasks_offset[wtasks_offset_start + wtasks_count[i]]; j++) {
            cout << wtasks_vertices[wtasks_start + j].label << " ";
        }
        cout << endl << "Indeg:" << endl;
        for (int j = 0; j < wtasks_offset[wtasks_offset_start + wtasks_count[i]]; j++) {
            cout << wtasks_vertices[wtasks_start + j].indeg << " ";
        }
        cout << endl << "Exdeg:" << endl;
        for (int j = 0; j < wtasks_offset[wtasks_offset_start + wtasks_count[i]]; j++) {
            cout << wtasks_vertices[wtasks_start + j].exdeg << " ";
        }
        cout << endl << "Lvl2adj:" << endl;
        for (int j = 0; j < wtasks_offset[wtasks_offset_start + wtasks_count[i]]; j++) {
            cout << wtasks_vertices[wtasks_start + j].lvl2adj << " ";
        }
        cout << endl;
    }
    cout << endl << endl;

    delete[] wtasks_count;
    delete[] wtasks_offset;
    delete[] wtasks_vertices;
}

void print_GPU_Cliques(GPU_Data& dd)
{
    uint64_t* cliques_count = new uint64_t;
    uint64_t* cliques_vertex_count = new uint64_t;

    chkerr(cudaMemcpy(cliques_count, dd.cliques_count, sizeof(uint64_t), cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(cliques_vertex_count, dd.cliques_vertex_count, sizeof(uint64_t), cudaMemcpyDeviceToHost));

    if (!store_cliques)
    {
        cout << endl << " --- (GPU_Cliques)device_cliques details --- " << endl;
        cout << endl << "Cliques: " << "Count: " << (*cliques_count) << " Vertex storage disabled" << endl;
        delete cliques_count;
        delete cliques_vertex_count;
        return;
    }

    uint64_t* cliques_offset = new uint64_t[CLIQUES_OFFSET_SIZE];
    uint64_t* cliques_size = new uint64_t[CLIQUES_OFFSET_SIZE];
    int* cliques_vertex = new int[CLIQUES_SIZE];

    chkerr(cudaMemcpy(cliques_offset, dd.cliques_offset, sizeof(uint64_t) * CLIQUES_OFFSET_SIZE, cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(cliques_size, dd.cliques_size, sizeof(uint64_t) * CLIQUES_OFFSET_SIZE, cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(cliques_vertex, dd.cliques_vertex, sizeof(int) * CLIQUES_SIZE, cudaMemcpyDeviceToHost));

    cout << endl << " --- (GPU_Cliques)device_cliques details --- " << endl;
    cout << endl << "Cliques: " << "Count: " << (*cliques_count) << " Vertex size: " << (*cliques_vertex_count) << endl;
    cout << endl << "Starts:" << endl;
    for (uint64_t i = 0; i < (*cliques_count); i++) {
        cout << cliques_offset[i] << " ";
    }
    cout << endl << "Sizes:" << endl;
    for (uint64_t i = 0; i < (*cliques_count); i++) {
        cout << cliques_size[i] << " ";
    }

    cout << endl << "Vertex:" << endl;
    for (uint64_t i = 0; i < (*cliques_count); i++) {
        uint64_t start = cliques_offset[i];
        uint64_t end = start + cliques_size[i];
        cout << i << " S: " << start << " E: " << end << " " << flush;
        for (uint64_t j = start; j < end; j++) {
            cout << cliques_vertex[j] << " " << flush;
        }
        cout << endl;
    }

    delete cliques_count;
    delete cliques_vertex_count;
    delete[] cliques_offset;
    delete[] cliques_size;
    delete[] cliques_vertex;

    return;

    cout << endl << "Vertex:" << endl;
    for (uint64_t i = 0; i < (*cliques_vertex_count); i++) {
        cout << cliques_vertex[i] << " ";
    }
    cout << endl;
}

void print_CPU_Cliques(CPU_Cliques& hc)
{
    uint64_t hc_count = hc.cliques_offset.empty() ? 0 : static_cast<uint64_t>(hc.cliques_offset.size() - 1);
    cout << endl << " --- (CPU_Cliques)host_cliques details --- " << endl;
    cout << endl << "Cliques: " << "Size: " << hc_count << endl;
    cout << endl << "Offsets:" << endl;
    for (uint64_t i = 0; i < hc.cliques_offset.size(); i++) {
        cout << hc.cliques_offset[i] << " ";
    }
    cout << endl << "Vertex:" << endl;
    for (uint64_t i = 0; i < hc.cliques_vertex.size(); i++) {
        cout << hc.cliques_vertex[i] << " ";
    }
    cout << endl;
}

void print_vertices(Vertex* vertices, int size)
{
    cout << " --- level 0 details --- " << endl;
    cout << endl << "Tasks1: Level: " << 0 << " Size: " << size << endl;
    cout << endl << "Offsets:" << endl;
    cout << "0 " << size << flush;
    cout << endl << "Vertex:" << endl;
    for (int i = 0; i < size; i++) {
        cout << vertices[i].vertexid << " " << flush;
    }
    cout << endl << "Label:" << endl;
    for (int i = 0; i < size; i++) {
        cout << vertices[i].label << " " << flush;
    }
    cout << endl << "Indeg:" << endl;
    for (int i = 0; i < size; i++) {
        cout << vertices[i].indeg << " " << flush;
    }
    cout << endl << "Exdeg:" << endl;
    for (int i = 0; i < size; i++) {
        cout << vertices[i].exdeg << " " << flush;
    }
    cout << endl << "Lvl2adj:" << endl;
    for (int i = 0; i < size; i++) {
        cout << vertices[i].lvl2adj << " " << flush;
    }
    cout << endl;
}

void initialize_maxes()
{
    mts = 0;
    mbs = 0;
    mbo = 0;
    mcs = 0;
    mco = 0;
    wts = 0;
    wto = 0;
    mvs = 0;
}

void print_maxes()
{
    cout << endl
        << "TASKS SIZE: " << mts << endl
        << "BUFFER SIZE: " << mbs << endl
        << "BUFFER OFFSET SIZE: " << mbo << endl
        << "CLIQUES SIZE: " << mcs << endl
        << "CLIQUES OFFSET SIZE: " << mco << endl
        << "WTASKS SIZE: " << wts << endl
        << "WTASKS OFFSET SIZE: " << wto << endl
        << "VERTICES SIZE: " << mvs << endl
        << endl;
}



// Quick enumeration order sort keys
__device__ int d_sort_vert_Q(Vertex& v1, Vertex& v2)
{
    // order is: member -> covered -> cands -> cover
    // keys are: indeg -> exdeg -> lvl2adj -> vertexid

    if (v1.label == 1 && v2.label != 1)
        return -1;
    else if (v1.label != 1 && v2.label == 1)
        return 1;
    else if (v1.label == 2 && v2.label != 2)
        return -1;
    else if (v1.label != 2 && v2.label == 2)
        return 1;
    else if (v1.label == 0 && v2.label != 0)
        return -1;
    else if (v1.label != 0 && v2.label == 0)
        return 1;
    else if (v1.label == 3 && v2.label != 3)
        return -1;
    else if (v1.label != 3 && v2.label == 3)
        return 1;
    else if (v1.indeg > v2.indeg)
        return -1;
    else if (v1.indeg < v2.indeg)
        return 1;
    else if (v1.exdeg > v2.exdeg)
        return -1;
    else if (v1.exdeg < v2.exdeg)
        return 1;
    else if (v1.lvl2adj > v2.lvl2adj)
        return -1;
    else if (v1.lvl2adj < v2.lvl2adj)
        return 1;
    else if (v1.vertexid > v2.vertexid)
        return -1;
    else if (v1.vertexid < v2.vertexid)
        return 1;
    else
        return 0;
}

__device__ int d_sort_vert_cv(Vertex& v1, Vertex& v2)
{
    // put crit adj vertices before candidates

    if (v1.label == 4 && v2.label != 4)
        return -1;
    else if (v1.label != 4 && v2.label == 4)
        return 1;
    else
        return 0;
}

__device__ int d_sort_degs(int n1, int n2)
{
    // descending order

    if (n1 > n2) {
        return -1;
    }
    else if (n1 < n2) {
        return 1;
    }
    else {
        return 0;
    }
}


// // --- DEBUG KERNELS ---

__device__ void d_print_vertices(Vertex* vertices, int size)
{
    printf("\nOffsets:\n0 %i\nVertex:\n", size);
    for (int i = 0; i < size; i++) {
        printf("%i ", vertices[i].vertexid);
    }
    printf("\nLabel:\n");
    for (int i = 0; i < size; i++) {
        printf("%i ", vertices[i].label);
    }
    printf("\nIndeg:\n");
    for (int i = 0; i < size; i++) {
        printf("%i ", vertices[i].indeg);
    }
    printf("\nExdeg:\n");
    for (int i = 0; i < size; i++) {
        printf("%i ", vertices[i].exdeg);
    }
    printf("\nLvl2adj:\n");
    for (int i = 0; i < size; i++) {
        printf("%i ", vertices[i].lvl2adj);
    }
    printf("\n");
}



// --- RM NON-MAX (from Quick) ---

int comp_int(const void* e1, const void* e2)
{
    int n1, n2;
    n1 = *(int*)e1;
    n2 = *(int*)e2;

    if (n1 > n2)
        return 1;
    else if (n1 < n2)
        return -1;
    else
        return 0;
}

extern int gntotal_max_cliques;

struct TREE_NODE
{
    int nid;
    TREE_NODE* pchild;
    TREE_NODE* pright_sib;
    bool bis_max;
};

#define TNODE_PAGE_SIZE (1<<10)

struct TNODE_PAGE
{
    TREE_NODE ptree_nodes[TNODE_PAGE_SIZE];
    TNODE_PAGE* pnext;
};

struct TNODE_BUF
{
    TNODE_PAGE* phead;
    TNODE_PAGE* pcur_page;
    int ncur_pos;
    int ntotal_pages;
};

extern TNODE_BUF gotreenode_buf;

inline TREE_NODE* NewTreeNode()
{
    TREE_NODE* ptnode;
    TNODE_PAGE* pnew_page;

    if (gotreenode_buf.ncur_pos == TNODE_PAGE_SIZE)
    {
        if (gotreenode_buf.pcur_page->pnext == NULL)
        {
            pnew_page = new TNODE_PAGE;
            pnew_page->pnext = NULL;
            gotreenode_buf.pcur_page->pnext = pnew_page;
            gotreenode_buf.pcur_page = pnew_page;
            gotreenode_buf.ntotal_pages++;
        }
        else
            gotreenode_buf.pcur_page = gotreenode_buf.pcur_page->pnext;
        gotreenode_buf.ncur_pos = 0;
    }

    ptnode = &(gotreenode_buf.pcur_page->ptree_nodes[gotreenode_buf.ncur_pos]);
    gotreenode_buf.ncur_pos++;

    ptnode->bis_max = true;

    return ptnode;
}

inline void OutputOneSet(FILE* fp, int* pset, int nlen)
{
    int i;

    gntotal_max_cliques++;

    fprintf(fp, "%d ", nlen);
    for (i = 0; i < nlen; i++)
        fprintf(fp, "%d ", pset[i]);
    fprintf(fp, "\n");

}

int gntotal_max_cliques;

TNODE_BUF gotreenode_buf;

void DelTNodeBuf()
{
    TNODE_PAGE* ppage;

    ppage = gotreenode_buf.phead;
    while (ppage != NULL)
    {
        gotreenode_buf.phead = gotreenode_buf.phead->pnext;
        delete ppage;
        gotreenode_buf.ntotal_pages--;
        ppage = gotreenode_buf.phead;
    }
    if (gotreenode_buf.ntotal_pages != 0)
        printf("Error: inconsistent number of pages\n");
}

void InsertOneSet(int* pset, int nlen, TREE_NODE*& proot)
{
    TREE_NODE* pnode, * pparent, * pleftsib, * pnew_node;
    int i, j;

    i = 0;
    pparent = NULL;
    pnode = proot;
    pleftsib = NULL;

    while (i < nlen)
    {
        while (pnode != NULL && pnode->nid < pset[i])
        {
            pleftsib = pnode;
            pnode = pnode->pright_sib;
        }

        if (pnode == NULL || pnode->nid > pset[i])
        {
            pnew_node = NewTreeNode();
            pnew_node->nid = pset[i];
            pnew_node->pchild = NULL;
            pnew_node->pright_sib = pnode;
            if (pleftsib != NULL)
                pleftsib->pright_sib = pnew_node;
            else if (pparent != NULL)
                pparent->pchild = pnew_node;
            if (i == 0 && pleftsib == NULL)
                proot = pnew_node;
            pparent = pnew_node;
            for (j = i + 1; j < nlen; j++)
            {
                pnew_node = NewTreeNode();
                pnew_node->nid = pset[j];
                pnew_node->pchild = NULL;
                pnew_node->pright_sib = NULL;
                pparent->pchild = pnew_node;
                pparent = pnew_node;
            }
            break;
        }
        else
        {
            pparent = pnode;
            pnode = pnode->pchild;
            pleftsib = NULL;
        }
        i++;
    }
}

int BuildTree(const char* szset_filename, TREE_NODE*& proot)
{
    FILE* fp;
    int nlen, * pset, nset_size, i, nmax_len, num_of_sets;

    fp = fopen(szset_filename, "rt");
    if (fp == NULL)
    {
        printf("Error: cannot open file %s for read\n", szset_filename);
        return 0;
    }

    gotreenode_buf.phead = new TNODE_PAGE;
    gotreenode_buf.phead->pnext = NULL;
    gotreenode_buf.pcur_page = gotreenode_buf.phead;
    gotreenode_buf.ntotal_pages = 1;
    gotreenode_buf.ncur_pos = 0;

    proot = NULL;

    num_of_sets = 0;

    nset_size = 100;
    pset = new int[nset_size];

    nmax_len = 0;
    fscanf(fp, "%d", &nlen);
    while (!feof(fp))
    {
        if (nmax_len < nlen)
            nmax_len = nlen;
        if (nlen > nset_size)
        {
            delete[]pset;
            nset_size *= 2;
            if (nset_size < nlen)
                nset_size = nlen;
            pset = new int[nset_size];
        }
        for (i = 0; i < nlen; i++)
            fscanf(fp, "%d", &pset[i]);
        qsort(pset, nlen, sizeof(int), comp_int);
        InsertOneSet(pset, nlen, proot);

        num_of_sets++;
        fscanf(fp, "%d", &nlen);
    }
    fclose(fp);

    delete[]pset;

    return nmax_len;
}

void SearchSubset(int* pset, int nset_len, TREE_NODE* proot, TREE_NODE** pstack, int* ppos)
{
    TREE_NODE* pnode;
    int ntop, npos;

    if (proot == NULL)
        return;
    ntop = 0;
    npos = 0;
    pnode = proot;

    while (ntop >= 0)
    {
        while (pnode != NULL && npos < nset_len && pnode->nid != pset[npos])
        {
            if (pnode->nid < pset[npos])
                pnode = pnode->pright_sib;
            else
                npos++;
        }
        if (pnode != NULL && npos < nset_len)
        {
            if (pnode->pchild == NULL && pnode->bis_max)
                pnode->bis_max = false;
            pstack[ntop] = pnode;
            ppos[ntop] = npos;
            ntop++;
            pnode = pnode->pchild;
            npos++;
        }
        else
        {
            ntop--;
            if (ntop >= 0)
            {
                pnode = pstack[ntop]->pright_sib;
                npos = ppos[ntop] + 1;
            }
        }
    }

}

void RmNonMax(TREE_NODE* proot, int nmax_len)
{
    TREE_NODE* pnode, ** pstack, ** psearch_stack;
    int* pset, ntop, i, * ppos;

    if (proot == NULL || nmax_len <= 0)
        return;

    pset = new int[nmax_len];
    pstack = new TREE_NODE * [nmax_len];
    psearch_stack = new TREE_NODE * [nmax_len];
    ppos = new int[nmax_len];

    pstack[0] = proot;
    pset[0] = proot->nid;
    ntop = 1;
    pnode = proot;

    while (ntop > 0)
    {
        if (pnode->pchild != NULL)
        {
            pnode = pnode->pchild;
            pstack[ntop] = pnode;
            pset[ntop] = pnode->nid;
            ntop++;
        }
        else
        {
            if (ntop >= 2 && pnode->bis_max)
            {
                for (i = ntop - 1; i >= 1; i--)
                {
                    if (pstack[i - 1]->pright_sib != NULL)
                        SearchSubset(&pset[i], ntop - i, pstack[i - 1]->pright_sib, psearch_stack, ppos);
                }
            }

            while (ntop > 0 && pnode->pright_sib == NULL)
            {
                ntop--;
                if (ntop > 0)
                    pnode = pstack[ntop - 1];
            }
            if (ntop == 0)
                break;
            else //if(pnode->pright_sib!=NULL)
            {
                pnode = pnode->pright_sib;
                pstack[ntop - 1] = pnode;
                pset[ntop - 1] = pnode->nid;
            }
        }
    }

    delete[]pset;
    delete[]pstack;
    delete[]psearch_stack;
    delete[]ppos;
}

void OutputMaxSet(TREE_NODE* proot, int nmax_len, const char* szoutput_filename)
{
    FILE* fp;
    TREE_NODE** pstack, * pnode;
    int* pset, ntop;

    fp = fopen(szoutput_filename, "wt");
    if (fp == NULL)
    {
        printf("Error: cannot open file %s for write\n", szoutput_filename);
        return;
    }

    pstack = new TREE_NODE * [nmax_len];
    pset = new int[nmax_len];

    pstack[0] = proot;
    pset[0] = proot->nid;
    ntop = 1;
    pnode = proot;

    while (ntop > 0)
    {
        if (pnode->pchild != NULL)
        {
            pnode = pnode->pchild;
            pstack[ntop] = pnode;
            pset[ntop] = pnode->nid;
            ntop++;
        }
        else
        {
            if (pnode->bis_max)
                OutputOneSet(fp, pset, ntop);

            while (ntop > 0 && pnode->pright_sib == NULL)
            {
                ntop--;
                if (ntop > 0)
                    pnode = pstack[ntop - 1];
            }
            if (ntop == 0)
                break;
            else //if(pnode->pright_sib!=NULL)
            {
                pnode = pnode->pright_sib;
                pstack[ntop - 1] = pnode;
                pset[ntop - 1] = pnode->nid;
            }
        }
    }

    delete[]pstack;
    delete[]pset;

    fclose(fp);
}

int RemoveNonMax(const char* szset_filename, const char* szoutput_filename)
{
    cout << ">:REMOVING NON-MAXIMAL CLIQUES" << endl;

    TREE_NODE* proot;
    int nmax_len;
    struct timeb start, end;

    ftime(&start);

    gntotal_max_cliques = 0;

    nmax_len = BuildTree(szset_filename, proot);
    RmNonMax(proot, nmax_len);
    OutputMaxSet(proot, nmax_len, szoutput_filename);

    DelTNodeBuf();

    ftime(&end);


    printf(">:NUMBER OF FINAL MAXIMAL QUASI-CLIQUES: %d\n", gntotal_max_cliques);
    return gntotal_max_cliques;
}


__global__ void transfer_cliques(GPU_Data dd)
{
    if (IDX == 0) {
        (*(dd.total_tasks)) = 0;
    }
}
