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
#define CLIQUES_SIZE 10000000
#define CLIQUES_OFFSET_SIZE 1000000
#define CLIQUES_PERCENT 50
// per warp
#define WCLIQUES_SIZE 10000
#define WCLIQUES_OFFSET_SIZE 1000
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
        graph_stream >> number_of_vertices;
        graph_stream >> number_of_edges;
        graph_stream >> number_of_lvl2adj;

        onehop_neighbors = new int[number_of_edges];
        onehop_offsets = new uint64_t[number_of_vertices + 1];
        twohop_neighbors = new int[number_of_lvl2adj];
        twohop_offsets = new uint64_t[number_of_vertices + 1];

        for (int i = 0; i < number_of_edges; i++) {
            graph_stream >> onehop_neighbors[i];
        }

        for (int i = 0; i < number_of_vertices + 1; i++) {
            graph_stream >> onehop_offsets[i];
        }

        for (int i = 0; i < number_of_lvl2adj; i++) {
            graph_stream >> twohop_neighbors[i];
        }

        for (int i = 0; i < number_of_vertices + 1; i++) {
            graph_stream >> twohop_offsets[i];
        }
    }

    ~CPU_Graph() 
    {
        delete onehop_neighbors;
        delete onehop_offsets;
        delete twohop_neighbors;
        delete twohop_offsets;
    }
};

// CPU DATA
struct CPU_Data
{
    uint64_t* tasks1_count;
    uint64_t* tasks1_offset;
    Vertex* tasks1_vertices;

    uint64_t* tasks2_count;
    uint64_t* tasks2_offset;
    Vertex* tasks2_vertices;

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
};

// CPU CLIQUES
struct CPU_Cliques
{
    uint64_t* cliques_count;
    uint64_t* cliques_offset;
    int* cliques_vertex;
};

// DEVICE DATA
struct GPU_Data
{
    // GPU DATA
    uint64_t* current_level;

    uint64_t* tasks1_count;
    uint64_t* tasks1_offset;
    Vertex* tasks1_vertices;

    uint64_t* tasks2_count;
    uint64_t* tasks2_offset;
    Vertex* tasks2_vertices;

    uint64_t* buffer_count;
    uint64_t* buffer_offset;
    Vertex* buffer_vertices;

    uint64_t* wtasks_count;
    uint64_t* wtasks_offset;
    Vertex* wtasks_vertices;

    Vertex* global_vertices;

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

    uint64_t* buffer_offset_start;
    uint64_t* buffer_start;
    uint64_t* cliques_offset_start;
    uint64_t* cliques_start;

    // GPU GRAPH
    int* number_of_vertices;
    uint64_t* number_of_edges;

    int* onehop_neighbors;
    uint64_t* onehop_offsets;
    int* twohop_neighbors;
    uint64_t* twohop_offsets;

    // GPU CLIQUES
    uint64_t* cliques_count;
    uint64_t* cliques_offset;
    int* cliques_vertex;

    uint64_t* wcliques_count;
    uint64_t* wcliques_offset;
    int* wcliques_vertex;

    ull* total_cliques;

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
void dump_cliques(CPU_Cliques& hc, GPU_Data& dd, ofstream& output_file);
void flush_cliques(CPU_Cliques& hc, ofstream& temp_results);
void free_memory(CPU_Data& hd, GPU_Data& dd, CPU_Cliques& hc);
void RemoveNonMax(const char* szset_filename, char* szoutput_filename);

// expansion
void h_expand_level(CPU_Graph& hg, CPU_Data& hd, CPU_Cliques& hc);
int h_lookahead_pruning(CPU_Graph& hg, CPU_Cliques& hc, CPU_Data& hd, Vertex* read_vertices, int tot_vert, int num_mem, int num_cand, uint64_t start);
int h_remove_one_vertex(CPU_Graph& hg, CPU_Data& hd, Vertex* read_vertices, int& tot_vert, int& num_cand, int& num_vert, uint64_t start);
int h_add_one_vertex(CPU_Graph& hg, CPU_Data& hd, Vertex* vertices, int& total_vertices, int& number_of_candidates, int& number_of_members, int& upper_bound, int& lower_bound, int& min_ext_deg);
void h_diameter_pruning(CPU_Graph& hg, CPU_Data& hd, Vertex* vertices, int pvertexid, int& total_vertices, int& number_of_candidates, int number_of_members);
bool h_degree_pruning(CPU_Graph& hg, CPU_Data& hd, Vertex* vertices, int& total_vertices, int& number_of_candidates, int number_of_members, int& upper_bound, int& lower_bound, int& min_ext_deg);
bool h_calculate_LU_bounds(CPU_Data& hd, int& upper_bound, int& lower_bound, int& min_ext_deg, Vertex* vertices, int number_of_members, int number_of_candidates);
void h_check_for_clique(CPU_Cliques& hc, Vertex* vertices, int number_of_members);
int h_critical_vertex_pruning(CPU_Graph& hg, CPU_Data& hd, Vertex* vertices, int& total_vertices, int& number_of_candidates, int& number_of_members, int& upper_bound, int& lower_bound, int& min_ext_deg);
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
void print_WClique_Buffers(GPU_Data& dd);
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
uint64_t mts, mbs, mbo, mcs, mco, wts, wto, wcs, wco, mvs;



// COMMAND LINE INPUT VARIABLES
double minimum_degree_ratio;
int minimum_clique_size;
int* minimum_degrees;
int scheduling_toggle;



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



void dump_cliques(CPU_Cliques& hc, GPU_Data& dd, ofstream& temp_results)
{
    // gpu cliques to cpu cliques
    chkerr(cudaMemcpy(hc.cliques_count, dd.cliques_count, sizeof(uint64_t), cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(hc.cliques_offset, dd.cliques_offset, sizeof(uint64_t) * CLIQUES_OFFSET_SIZE, cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(hc.cliques_vertex, dd.cliques_vertex, sizeof(int) * CLIQUES_SIZE, cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();

    // DEBUG
    //print_CPU_Cliques(hc);

    flush_cliques(hc, temp_results);

    cudaMemset(dd.cliques_count, 0, sizeof(uint64_t));
}

void flush_cliques(CPU_Cliques& hc, ofstream& temp_results) 
{
    for (int i = 0; i < ((*hc.cliques_count)); i++) {
        uint64_t start = hc.cliques_offset[i];
        uint64_t end = hc.cliques_offset[i + 1];
        temp_results << end - start << " ";
        for (uint64_t j = start; j < end; j++) {
            temp_results << hc.cliques_vertex[j] << " ";
        }
        temp_results << "\n";
    }
    ((*hc.cliques_count)) = 0;
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
    delete hd.tasks1_count;
    delete hd.tasks1_offset;
    delete hd.tasks1_vertices;

    delete hd.tasks2_count;
    delete hd.tasks2_offset;
    delete hd.tasks2_vertices;

    delete hd.buffer_count;
    delete hd.buffer_offset;
    delete hd.buffer_vertices;

    delete hd.current_level;
    delete hd.maximal_expansion;
    delete hd.dumping_cliques;

    delete hd.vertex_order_map;
    delete hd.remaining_candidates;
    delete hd.remaining_count;
    delete hd.removed_candidates;
    delete hd.removed_count;
    delete hd.candidate_indegs;

    // GPU DATA
    chkerr(cudaFree(dd.current_level));

    chkerr(cudaFree(dd.tasks1_count));
    chkerr(cudaFree(dd.tasks1_offset));
    chkerr(cudaFree(dd.tasks1_vertices));

    chkerr(cudaFree(dd.tasks2_count));
    chkerr(cudaFree(dd.tasks2_offset));
    chkerr(cudaFree(dd.tasks2_vertices));

    chkerr(cudaFree(dd.buffer_count));
    chkerr(cudaFree(dd.buffer_offset));
    chkerr(cudaFree(dd.buffer_vertices));

    chkerr(cudaFree(dd.wtasks_count));
    chkerr(cudaFree(dd.wtasks_offset));
    chkerr(cudaFree(dd.wtasks_vertices));

    chkerr(cudaFree(dd.global_vertices));

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

    chkerr(cudaFree(dd.total_tasks));

    // CPU CLIQUES
    delete hc.cliques_count;
    delete hc.cliques_vertex;
    delete hc.cliques_offset;

    // GPU CLIQUES
    chkerr(cudaFree(dd.cliques_count));
    chkerr(cudaFree(dd.cliques_vertex));
    chkerr(cudaFree(dd.cliques_offset));

    chkerr(cudaFree(dd.wcliques_count));
    chkerr(cudaFree(dd.wcliques_vertex));
    chkerr(cudaFree(dd.wcliques_offset));

    chkerr(cudaFree(dd.buffer_offset_start));
    chkerr(cudaFree(dd.buffer_start));
    chkerr(cudaFree(dd.cliques_offset_start));
    chkerr(cudaFree(dd.cliques_start));

    // tasks scheduling
    chkerr(cudaFree(dd.current_task));
}



// --- HOST EXPANSION METHODS ---

// returns 1 if lookahead was a success, else 0
int h_lookahead_pruning(CPU_Graph& hg, CPU_Cliques& hc, CPU_Data& hd, Vertex* read_vertices, int tot_vert, int num_mem, int num_cand, uint64_t start)
{
    // intersection
    int pvertexid;
    uint64_t pneighbors_start;
    uint64_t pneighbors_end;
    int phelper1;


    // check if members meet degree requirement, dont need to check 2hop adj as diameter pruning guarentees all members will be within 2hops of eveything
    for (int i = 0; i < num_mem; i++) {
        if (read_vertices[start + i].indeg + read_vertices[start + i].exdeg < minimum_degrees[tot_vert]) {
            return 0;
        }
    }

    // initialize vertex order map
    for (int i = num_mem; i < tot_vert; i++) {
        hd.vertex_order_map[read_vertices[start + i].vertexid] = i;
    }

    // update lvl2adj to candidates for all vertices
    for (int i = num_mem; i < tot_vert; i++) {
        pvertexid = read_vertices[start + i].vertexid;
        pneighbors_start = hg.twohop_offsets[pvertexid];
        pneighbors_end = hg.twohop_offsets[pvertexid + 1];
        for (int j = pneighbors_start; j < pneighbors_end; j++) {
            phelper1 = hd.vertex_order_map[hg.twohop_neighbors[j]];

            if (phelper1 >= num_mem) {
                read_vertices[start + phelper1].lvl2adj++;
            }
        }
    }

    // reset vertex order map
    for (int i = num_mem; i < tot_vert; i++) {
        hd.vertex_order_map[read_vertices[start + i].vertexid] = -1;
    }

    // check for lookahead
    for (int j = num_mem; j < tot_vert; j++) {
        if (read_vertices[start + j].lvl2adj < num_cand - 1 || read_vertices[start + j].indeg + read_vertices[start + j].exdeg < minimum_degrees[tot_vert]) {
            return 0;
        }
    }

    // write to cliques
    uint64_t start_write = hc.cliques_offset[(*hc.cliques_count)];
    for (int j = 0; j < tot_vert; j++) {
        hc.cliques_vertex[start_write + j] = read_vertices[start + j].vertexid;
    }
    (*hc.cliques_count)++;
    hc.cliques_offset[(*hc.cliques_count)] = start_write + tot_vert;

    return 1;
}

// returns 1 is failed found or not enough vertices, else 0
int h_remove_one_vertex(CPU_Graph& hg, CPU_Data& hd, Vertex* read_vertices, int& tot_vert, int& num_cand, int& num_mem, uint64_t start)
{
    // intersection
    int pvertexid;
    uint64_t pneighbors_start;
    uint64_t pneighbors_end;
    int phelper1;

    // helper variables
    int mindeg;
    bool failed_found;



    mindeg = h_get_mindeg(num_mem);

    // remove one vertex
    num_cand--;
    tot_vert--;

    // initialize vertex order map
    for (int i = 0; i < tot_vert; i++) {
        hd.vertex_order_map[read_vertices[start + i].vertexid] = i;
    }

    failed_found = false;

    // update info of vertices connected to removed cand
    pvertexid = read_vertices[start + tot_vert].vertexid;
    pneighbors_start = hg.onehop_offsets[pvertexid];
    pneighbors_end = hg.onehop_offsets[pvertexid + 1];
    for (int i = pneighbors_start; i < pneighbors_end; i++) {
        phelper1 = hd.vertex_order_map[hg.onehop_neighbors[i]];

        if (phelper1 > -1) {
            read_vertices[start + phelper1].exdeg--;

            if (phelper1 < num_mem && read_vertices[start + phelper1].indeg + read_vertices[start + phelper1].exdeg < mindeg) {
                failed_found = true;
                break;
            }
        }
    }

    // reset vertex order map
    for (int i = 0; i < tot_vert; i++) {
        hd.vertex_order_map[read_vertices[start + i].vertexid] = -1;
    }

    if (failed_found) {
        return 1;
    }

    return 0;
}

// returns 1 if failed found or invalid bound, 0 otherwise
int h_add_one_vertex(CPU_Graph& hg, CPU_Data& hd, Vertex* vertices, int& total_vertices, int& number_of_candidates, int& number_of_members, int& upper_bound, int& lower_bound, int& min_ext_deg)
{
    // helper variables
    bool method_return;

    // intersection
    int pvertexid;
    uint64_t pneighbors_start;
    uint64_t pneighbors_end;
    int pneighbors_count;
    int phelper1;



    // ADD ONE VERTEX
    pvertexid = vertices[number_of_members].vertexid;

    vertices[number_of_members].label = 1;
    number_of_members++;
    number_of_candidates--;

    // initialize vertex order map
    for (int i = 0; i < total_vertices; i++) {
        hd.vertex_order_map[vertices[i].vertexid] = i;
    }

    pneighbors_start = hg.onehop_offsets[pvertexid];
    pneighbors_end = hg.onehop_offsets[pvertexid + 1];
    pneighbors_count = pneighbors_end - pneighbors_start;
    for (int i = 0; i < pneighbors_count; i++) {
        phelper1 = hd.vertex_order_map[hg.onehop_neighbors[pneighbors_start + i]];

        if (phelper1 > -1) {
            vertices[phelper1].indeg++;
            vertices[phelper1].exdeg--;
        }
    }



    // DIAMETER PRUNING
    h_diameter_pruning(hg, hd, vertices, pvertexid, total_vertices, number_of_candidates, number_of_members);



    // DEGREE-BASED PRUNING
    method_return = h_degree_pruning(hg, hd, vertices, total_vertices, number_of_candidates, number_of_members, upper_bound, lower_bound, min_ext_deg);

    for (int i = 0; i < hg.number_of_vertices; i++) {
        hd.vertex_order_map[i] = -1;
    }

    // if vertex in x found as not extendable, check if current set is clique and continue to next iteration
    if (method_return) {
        return 1;
    }

    return 0;
}

// returns 2 if too many vertices pruned or a critical vertex fail, returns 1 if failed found or invalid bounds, else 0
int h_critical_vertex_pruning(CPU_Graph& hg, CPU_Data& hd, Vertex* vertices, int& total_vertices, int& number_of_candidates, int& number_of_members, int& upper_bound, int& lower_bound, int& min_ext_deg)
{
    // intersection
    int pvertexid;
    uint64_t pneighbors_start;
    uint64_t pneighbors_end;
    int phelper1;

    bool critical_fail;
    int number_of_crit_adj;
    int* adj_counters;

    bool method_return;



    // initialize vertex order map
    for (int i = 0; i < total_vertices; i++) {
        hd.vertex_order_map[vertices[i].vertexid] = i;
    }

    // CRITICAL VERTEX PRUNING
    // adj_counter[0] = 10, means that the vertex at position 0 in new_vertices has 10 critical vertices neighbors within 2 hops
    adj_counters = new int[total_vertices];
    memset(adj_counters, 0, sizeof(int) * total_vertices);

    // iterate through all vertices in clique
    for (int k = 0; k < number_of_members; k++)
    {
        // if they are a critical vertex
        if (vertices[k].indeg + vertices[k].exdeg == minimum_degrees[number_of_members + lower_bound] && vertices[k].exdeg > 0) {
            pvertexid = vertices[k].vertexid;

            // iterate through all neighbors
            pneighbors_start = hg.onehop_offsets[pvertexid];
            pneighbors_end = hg.onehop_offsets[pvertexid + 1];
            for (uint64_t l = pneighbors_start; l < pneighbors_end; l++) {
                phelper1 = hd.vertex_order_map[hg.onehop_neighbors[l]];

                // if neighbor is cand
                if (phelper1 >= number_of_members) {
                    vertices[phelper1].label = 4;
                }
            }
        }
    }



    // reset vertex order map
    for (int i = 0; i < total_vertices; i++) {
        hd.vertex_order_map[vertices[i].vertexid] = -1;
    }

    // sort vertices so that critical vertex adjacent candidates are immediately after vertices within the clique
    qsort(vertices + number_of_members, number_of_candidates, sizeof(Vertex), h_sort_vert_cv);

    // calculate number of critical adjacent vertices
    number_of_crit_adj = 0;
    for (int i = number_of_members; i < total_vertices; i++) {
        if (vertices[i].label == 4) {
            number_of_crit_adj++;
        }
        else {
            break;
        }
    }



    // if there were any neighbors of critical vertices
    if (number_of_crit_adj > 0)
    {
        // initialize vertex order map
        for (int i = 0; i < total_vertices; i++) {
            hd.vertex_order_map[vertices[i].vertexid] = i;
        }

        // iterate through all neighbors
        for (int i = number_of_members; i < number_of_members + number_of_crit_adj; i++) {
            pvertexid = vertices[i].vertexid;

            // update 1hop adj
            pneighbors_start = hg.onehop_offsets[pvertexid];
            pneighbors_end = hg.onehop_offsets[pvertexid + 1];
            for (uint64_t k = pneighbors_start; k < pneighbors_end; k++) {
                phelper1 = hd.vertex_order_map[hg.onehop_neighbors[k]];

                if (phelper1 > -1) {
                    vertices[phelper1].indeg++;
                    vertices[phelper1].exdeg--;
                }
            }

            // track 2hop adj
            pneighbors_start = hg.twohop_offsets[pvertexid];
            pneighbors_end = hg.twohop_offsets[pvertexid + 1];
            for (uint64_t k = pneighbors_start; k < pneighbors_end; k++) {
                phelper1 = hd.vertex_order_map[hg.twohop_neighbors[k]];

                if (phelper1 > -1) {
                    adj_counters[phelper1]++;
                }
            }
        }

        critical_fail = false;

        // all vertices within the clique must be within 2hops of the newly added critical vertex adj vertices
        for (int k = 0; k < number_of_members; k++) {
            if (adj_counters[k] != number_of_crit_adj) {
                critical_fail = true;
            }
        }

        if (critical_fail) {
            // reset vertex order map
            for (int i = 0; i < total_vertices; i++) {
                hd.vertex_order_map[vertices[i].vertexid] = -1;
            }
            delete adj_counters;
            return 2;
        }

        // all critical adj vertices must all be within 2 hops of each other
        for (int k = number_of_members; k < number_of_members + number_of_crit_adj; k++) {
            if (adj_counters[k] < number_of_crit_adj - 1) {
                critical_fail = true;
            }
        }

        if (critical_fail) {
            // reset vertex order map
            for (int i = 0; i < total_vertices; i++) {
                hd.vertex_order_map[vertices[i].vertexid] = -1;
            }
            delete adj_counters;
            return 2;
        }

        // no failed vertices found so add all critical vertex adj candidates to clique
        for (int k = number_of_members; k < number_of_members + number_of_crit_adj; k++) {
            vertices[k].label = 1;
        }
        number_of_members += number_of_crit_adj;
        number_of_candidates -= number_of_crit_adj;
    }



    // DIAMTER PRUNING
    (*hd.remaining_count) = 0;

    // remove all cands who are not within 2hops of all newly added cands
    for (int k = number_of_members; k < total_vertices; k++) {
        if (adj_counters[k] == number_of_crit_adj) {
            hd.candidate_indegs[(*hd.remaining_count)++] = vertices[k].indeg;
        }
        else {
            vertices[k].label = -1;
        }
    }

    

    // DEGREE-BASED PRUNING
    method_return = h_degree_pruning(hg, hd, vertices, total_vertices, number_of_candidates, number_of_members, upper_bound, lower_bound, min_ext_deg);

    // reset vertex order map
    for (int i = 0; i < total_vertices; i++) {
        hd.vertex_order_map[vertices[i].vertexid] = -1;
    }

    delete adj_counters;

    // if vertex in x found as not extendable, check if current set is clique and continue to next iteration
    if (method_return) {
        return 1;
    }

    return 0;
}

void h_diameter_pruning(CPU_Graph& hg, CPU_Data& hd, Vertex* vertices, int pvertexid, int& total_vertices, int& number_of_candidates, int number_of_members)
{
    // intersection
    uint64_t pneighbors_start;
    uint64_t pneighbors_end;
    int phelper1;

    (*hd.remaining_count) = 0;

    for (int i = number_of_members; i < total_vertices; i++) {
        vertices[i].label = -1;
    }

    pneighbors_start = hg.twohop_offsets[pvertexid];
    pneighbors_end = hg.twohop_offsets[pvertexid + 1];
    for (int i = pneighbors_start; i < pneighbors_end; i++) {
        phelper1 = hd.vertex_order_map[hg.twohop_neighbors[i]];

        if (phelper1 >= number_of_members) {
            vertices[phelper1].label = 0;
            hd.candidate_indegs[(*hd.remaining_count)++] = vertices[phelper1].indeg;
        }
    }
}

// returns true is invalid bounds calculated or a failed vertex was found, else false
bool h_degree_pruning(CPU_Graph& hg, CPU_Data& hd, Vertex* vertices, int& total_vertices, int& number_of_candidates, int number_of_members, int& upper_bound, int& lower_bound, int& min_ext_deg)
{
    // intersection
    int pvertexid;
    uint64_t pneighbors_start;
    uint64_t pneighbors_end;
    int phelper1;

    // helper variables
    int num_val_cands;

    qsort(hd.candidate_indegs, (*hd.remaining_count), sizeof(int), h_sort_desc);

    // if invalid bounds found while calculating lower and upper bounds
    if (h_calculate_LU_bounds(hd, upper_bound, lower_bound, min_ext_deg, vertices, number_of_members, (*hd.remaining_count))) {
        return true;
    }

    // check for failed vertices
    for (int k = 0; k < number_of_members; k++) {
        if (!h_vert_isextendable_LU(vertices[k], number_of_members, upper_bound, lower_bound, min_ext_deg)) {
            return true;
        }
    }

    (*hd.remaining_count) = 0;
    (*hd.removed_count) = 0;

    // check for invalid candidates
    for (int i = number_of_members; i < total_vertices; i++) {
        if (vertices[i].label == 0 && h_cand_isvalid_LU(vertices[i], number_of_members, upper_bound, lower_bound, min_ext_deg)) {
            hd.remaining_candidates[(*hd.remaining_count)++] = i;
        }
        else {
            hd.removed_candidates[(*hd.removed_count)++] = i;
        }
    }

    while ((*hd.remaining_count) > 0 && (*hd.removed_count) > 0) {
        // update degrees
        if ((*hd.remaining_count) < (*hd.removed_count)) {
            // reset exdegs
            for (int i = 0; i < total_vertices; i++) {
                vertices[i].exdeg = 0;
            }

            for (int i = 0; i < (*hd.remaining_count); i++) {
                pvertexid = vertices[hd.remaining_candidates[i]].vertexid;
                pneighbors_start = hg.onehop_offsets[pvertexid];
                pneighbors_end = hg.onehop_offsets[pvertexid + 1];
                for (int j = pneighbors_start; j < pneighbors_end; j++) {
                    phelper1 = hd.vertex_order_map[hg.onehop_neighbors[j]];

                    if (phelper1 > -1) {
                        vertices[phelper1].exdeg++;
                    }
                }
            }
        }
        else {
            for (int i = 0; i < (*hd.removed_count); i++) {
                pvertexid = vertices[hd.removed_candidates[i]].vertexid;
                pneighbors_start = hg.onehop_offsets[pvertexid];
                pneighbors_end = hg.onehop_offsets[pvertexid + 1];
                for (int j = pneighbors_start; j < pneighbors_end; j++) {
                    phelper1 = hd.vertex_order_map[hg.onehop_neighbors[j]];

                    if (phelper1 > -1) {
                        vertices[phelper1].exdeg--;
                    }
                }
            }
        }

        num_val_cands = 0;

        for (int k = 0; k < (*hd.remaining_count); k++) {
            if (h_cand_isvalid_LU(vertices[hd.remaining_candidates[k]], number_of_members, upper_bound, lower_bound, min_ext_deg)) {
                hd.candidate_indegs[num_val_cands++] = vertices[hd.remaining_candidates[k]].indeg;
            }
        }

        qsort(hd.candidate_indegs, num_val_cands, sizeof(int), h_sort_desc);

        // if invalid bounds found while calculating lower and upper bounds
        if (h_calculate_LU_bounds(hd, upper_bound, lower_bound, min_ext_deg, vertices, number_of_members, num_val_cands)) {
            return true;
        }

        // check for failed vertices
        for (int k = 0; k < number_of_members; k++) {
            if (!h_vert_isextendable_LU(vertices[k], number_of_members, upper_bound, lower_bound, min_ext_deg)) {
                return true;
            }
        }

        num_val_cands = 0;
        (*hd.removed_count) = 0;

        // check for invalid candidates
        for (int k = 0; k < (*hd.remaining_count); k++) {
            if (h_cand_isvalid_LU(vertices[hd.remaining_candidates[k]], number_of_members, upper_bound, lower_bound, min_ext_deg)) {
                hd.remaining_candidates[num_val_cands++] = hd.remaining_candidates[k];
            }
            else {
                hd.removed_candidates[(*hd.removed_count)++] = hd.remaining_candidates[k];
            }
        }

        (*hd.remaining_count) = num_val_cands;
    }

    for (int i = 0; i < (*hd.remaining_count); i++) {
        vertices[number_of_members + i] = vertices[hd.remaining_candidates[i]];
    }

    total_vertices = total_vertices - number_of_candidates + (*hd.remaining_count);
    number_of_candidates = (*hd.remaining_count);

    return false;
}

bool h_calculate_LU_bounds(CPU_Data& hd, int& upper_bound, int& lower_bound, int& min_ext_deg, Vertex* vertices, int number_of_members, int number_of_candidates)
{
    bool invalid_bounds = false;
    int index;

    int sum_candidate_indeg = 0;
    int tightened_upper_bound = 0;

    int min_clq_indeg = vertices[0].indeg;
    int min_indeg_exdeg = vertices[0].exdeg;
    int min_clq_totaldeg = vertices[0].indeg + vertices[0].exdeg;
    int sum_clq_indeg = vertices[0].indeg;

    for (index = 1; index < number_of_members; index++) {
        sum_clq_indeg += vertices[index].indeg;

        if (vertices[index].indeg < min_clq_indeg) {
            min_clq_indeg = vertices[index].indeg;
            min_indeg_exdeg = vertices[index].exdeg;
        }
        else if (vertices[index].indeg == min_clq_indeg) {
            if (vertices[index].exdeg < min_indeg_exdeg) {
                min_indeg_exdeg = vertices[index].exdeg;
            }
        }

        if (vertices[index].indeg + vertices[index].exdeg < min_clq_totaldeg) {
            min_clq_totaldeg = vertices[index].indeg + vertices[index].exdeg;
        }
    }

    min_ext_deg = h_get_mindeg(number_of_members + 1);

    if (min_clq_indeg < minimum_degrees[number_of_members])
    {
        // lower
        lower_bound = h_get_mindeg(number_of_members) - min_clq_indeg;

        while (lower_bound <= min_indeg_exdeg && min_clq_indeg + lower_bound < minimum_degrees[number_of_members + lower_bound]) {
            lower_bound++;
        }

        if (min_clq_indeg + lower_bound < minimum_degrees[number_of_members + lower_bound]) {
            lower_bound = number_of_candidates + 1;
            invalid_bounds = true;
        }

        // upper
        upper_bound = floor(min_clq_totaldeg / minimum_degree_ratio) + 1 - number_of_members;

        if (upper_bound > number_of_candidates) {
            upper_bound = number_of_candidates;
        }

        // tighten
        if (lower_bound < upper_bound) {
            // tighten lower
            for (index = 0; index < lower_bound; index++) {
                sum_candidate_indeg += hd.candidate_indegs[index];
            }

            while (index < upper_bound && sum_clq_indeg + sum_candidate_indeg < number_of_members * minimum_degrees[number_of_members + index]) {
                sum_candidate_indeg += hd.candidate_indegs[index];
                index++;
            }

            if (sum_clq_indeg + sum_candidate_indeg < number_of_members * minimum_degrees[number_of_members + index]) {
                lower_bound = upper_bound + 1;
                invalid_bounds = true;
            }
            else {
                lower_bound = index;

                tightened_upper_bound = index;

                while (index < upper_bound) {
                    sum_candidate_indeg += hd.candidate_indegs[index];

                    index++;

                    if (sum_clq_indeg + sum_candidate_indeg >= number_of_members * minimum_degrees[number_of_members + index]) {
                        tightened_upper_bound = index;
                    }
                }

                if (upper_bound > tightened_upper_bound) {
                    upper_bound = tightened_upper_bound;
                }

                if (lower_bound > 1) {
                    min_ext_deg = h_get_mindeg(number_of_members + lower_bound);
                }
            }
        }
    }
    else {
        upper_bound = number_of_candidates;

        if (number_of_members < minimum_clique_size) {
            lower_bound = minimum_clique_size - number_of_members;
        }
        else {
            lower_bound = 0;
        }
    }

    if (number_of_members + upper_bound < minimum_clique_size) {
        invalid_bounds = true;
    }

    if (upper_bound < 0 || upper_bound < lower_bound) {
        invalid_bounds = true;
    }

    return invalid_bounds;
}

void h_check_for_clique(CPU_Cliques& hc, Vertex* vertices, int number_of_members)
{
    bool clique = true;

    int degree_requirement = minimum_degrees[number_of_members];
    for (int k = 0; k < number_of_members; k++) {
        if (vertices[k].indeg < degree_requirement) {
            clique = false;
            break;
        }
    }

    // if clique write to cliques array
    if (clique) {
        uint64_t start_write = hc.cliques_offset[(*hc.cliques_count)];
        for (int k = 0; k < number_of_members; k++) {
            hc.cliques_vertex[start_write + k] = vertices[k].vertexid;
        }
        (*hc.cliques_count)++;
        hc.cliques_offset[(*hc.cliques_count)] = start_write + number_of_members;
    }
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

    delete onehop_offsets;
    delete onehop_neighbors;
    delete twohop_offsets;
    delete twohop_neighbors;
}

void print_CPU_Data(CPU_Data& hd)
{
    cout << endl << " --- (CPU_Data)host_data details --- " << endl;
    cout << endl << "Tasks1: " << "Size: " << (*(hd.tasks1_count)) << endl;
    cout << endl << "Offsets:" << endl;
    for (uint64_t i = 0; i <= (*(hd.tasks1_count)); i++) {
        cout << hd.tasks1_offset[i] << " ";
    }
    cout << endl << "Vertex:" << endl;
    for (uint64_t i = 0; i < hd.tasks1_offset[(*(hd.tasks1_count))]; i++) {
        cout << hd.tasks1_vertices[i].vertexid << " ";
    }
    cout << endl << "Label:" << endl;
    for (uint64_t i = 0; i < hd.tasks1_offset[(*(hd.tasks1_count))]; i++) {
        cout << hd.tasks1_vertices[i].label << " ";
    }
    cout << endl << "Indeg:" << endl;
    for (uint64_t i = 0; i < hd.tasks1_offset[(*(hd.tasks1_count))]; i++) {
        cout << hd.tasks1_vertices[i].indeg << " ";
    }
    cout << endl << "Exdeg:" << endl;
    for (uint64_t i = 0; i < hd.tasks1_offset[(*(hd.tasks1_count))]; i++) {
        cout << hd.tasks1_vertices[i].exdeg << " ";
    }
    cout << endl << "Lvl2adj:" << endl;
    for (uint64_t i = 0; i < hd.tasks1_offset[(*(hd.tasks1_count))]; i++) {
        cout << hd.tasks1_vertices[i].lvl2adj << " ";
    }

    cout << endl << endl << "Tasks2: " << "Size: " << (*(hd.tasks2_count)) << endl;
    cout << endl << "Offsets:" << endl;
    for (uint64_t i = 0; i <= (*(hd.tasks2_count)); i++) {
        cout << hd.tasks2_offset[i] << " ";
    }
    cout << endl << "Vertex:" << endl;
    for (uint64_t i = 0; i < hd.tasks2_offset[(*(hd.tasks2_count))]; i++) {
        cout << hd.tasks2_vertices[i].vertexid << " ";
    }
    cout << endl << "Label:" << endl;
    for (uint64_t i = 0; i < hd.tasks2_offset[(*(hd.tasks2_count))]; i++) {
        cout << hd.tasks2_vertices[i].label << " ";
    }
    cout << endl << "Indeg:" << endl;
    for (uint64_t i = 0; i < hd.tasks2_offset[(*(hd.tasks2_count))]; i++) {
        cout << hd.tasks2_vertices[i].indeg << " ";
    }
    cout << endl << "Exdeg:" << endl;
    for (uint64_t i = 0; i < hd.tasks2_offset[(*(hd.tasks2_count))]; i++) {
        cout << hd.tasks2_vertices[i].exdeg << " ";
    }
    cout << endl << "Lvl2adj:" << endl;
    for (uint64_t i = 0; i < hd.tasks2_offset[(*(hd.tasks2_count))]; i++) {
        cout << hd.tasks2_vertices[i].lvl2adj << " ";
    }

    cout << endl << endl << "Buffer: " << "Size: " << (*(hd.buffer_count)) << endl;
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

    uint64_t* tasks1_count = new uint64_t;
    uint64_t* tasks1_offset = new uint64_t[EXPAND_THRESHOLD + 1];
    Vertex* tasks1_vertices = new Vertex[TASKS_SIZE];

    uint64_t* tasks2_count = new uint64_t;
    uint64_t* tasks2_offset = new uint64_t[EXPAND_THRESHOLD + 1];
    Vertex* tasks2_vertices = new Vertex[TASKS_SIZE];


    uint64_t* buffer_count = new uint64_t;
    uint64_t* buffer_offset = new uint64_t[BUFFER_OFFSET_SIZE];
    Vertex* buffer_vertices = new Vertex[BUFFER_SIZE];


    chkerr(cudaMemcpy(current_level, dd.current_level, sizeof(uint64_t), cudaMemcpyDeviceToHost));

    chkerr(cudaMemcpy(tasks1_count, dd.tasks1_count, sizeof(uint64_t), cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(tasks1_offset, dd.tasks1_offset, (EXPAND_THRESHOLD + 1) * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(tasks1_vertices, dd.tasks1_vertices, (TASKS_SIZE) * sizeof(Vertex), cudaMemcpyDeviceToHost));

    chkerr(cudaMemcpy(tasks2_count, dd.tasks2_count, sizeof(uint64_t), cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(tasks2_offset, dd.tasks2_offset, (EXPAND_THRESHOLD + 1) * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(tasks2_vertices, dd.tasks2_vertices, (TASKS_SIZE) * sizeof(Vertex), cudaMemcpyDeviceToHost));

    chkerr(cudaMemcpy(buffer_count, dd.buffer_count, sizeof(uint64_t), cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(buffer_offset, dd.buffer_offset, (BUFFER_OFFSET_SIZE) * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(buffer_vertices, dd.buffer_vertices, (BUFFER_SIZE) * sizeof(Vertex), cudaMemcpyDeviceToHost));

    cout << " --- (GPU_Data)device_data details --- " << endl;
    cout << endl << "Tasks1: Level: " << (*current_level) << " Size: " << (*tasks1_count) << endl;
    cout << endl << "Offsets:" << endl;
    for (int i = 0; i <= (*tasks1_count); i++) {
        cout << tasks1_offset[i] << " " << flush;
    }
    cout << endl << "Vertex:" << endl;
    for (int i = 0; i < tasks1_offset[*tasks1_count]; i++) {
        cout << tasks1_vertices[i].vertexid << " " << flush;
    }
    cout << endl << "Label:" << endl;
    for (int i = 0; i < tasks1_offset[*tasks1_count]; i++) {
        cout << tasks1_vertices[i].label << " " << flush;
    }
    cout << endl << "Indeg:" << endl;
    for (int i = 0; i < tasks1_offset[*tasks1_count]; i++) {
        cout << tasks1_vertices[i].indeg << " " << flush;
    }
    cout << endl << "Exdeg:" << endl;
    for (int i = 0; i < tasks1_offset[*tasks1_count]; i++) {
        cout << tasks1_vertices[i].exdeg << " " << flush;
    }
    cout << endl << "Lvl2adj:" << endl;
    for (int i = 0; i < tasks1_offset[*tasks1_count]; i++) {
        cout << tasks1_vertices[i].lvl2adj << " " << flush;
    }
    cout << endl;

    cout << endl << "Tasks2: " << "Size: " << (*tasks2_count) << endl;
    cout << endl << "Offsets:" << endl;
    for (int i = 0; i <= (*tasks2_count); i++) {
        cout << tasks2_offset[i] << " " << flush;
    }
    cout << endl << "Vertex:" << endl;
    for (int i = 0; i < tasks2_offset[*tasks2_count]; i++) {
        cout << tasks2_vertices[i].vertexid << " " << flush;
    }
    cout << endl << "Label:" << endl;
    for (int i = 0; i < tasks2_offset[*tasks2_count]; i++) {
        cout << tasks2_vertices[i].label << " " << flush;
    }
    cout << endl << "Indeg:" << endl;
    for (int i = 0; i < tasks2_offset[*tasks2_count]; i++) {
        cout << tasks2_vertices[i].indeg << " " << flush;
    }
    cout << endl << "Exdeg:" << endl;
    for (int i = 0; i < tasks2_offset[*tasks2_count]; i++) {
        cout << tasks2_vertices[i].exdeg << " " << flush;
    }
    cout << endl << "Lvl2adj:" << endl;
    for (int i = 0; i < tasks2_offset[*tasks2_count]; i++) {
        cout << tasks2_vertices[i].lvl2adj << " " << flush;
    }
    cout << endl << endl;

    cout << endl << "Buffer: " << "Size: " << (*buffer_count) << endl;
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

    delete tasks1_count;
    delete tasks1_offset;
    delete tasks1_vertices;

    delete tasks2_count;
    delete tasks2_offset;
    delete tasks2_vertices;

    delete buffer_count;
    delete buffer_offset;
    delete buffer_vertices;
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
    uint64_t* cliques_counts = new uint64_t[NUMBER_OF_WARPS];
    uint64_t* cliques_sizes = new uint64_t[NUMBER_OF_WARPS];
    int cliques_tcount = 0;
    int cliques_tsize = 0;
    int cliques_mcount = 0;
    int cliques_msize = 0;

    chkerr(cudaMemcpy(tasks_counts, dd.wtasks_count, sizeof(uint64_t) * NUMBER_OF_WARPS, cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(cliques_counts, dd.wcliques_count, sizeof(uint64_t) * NUMBER_OF_WARPS, cudaMemcpyDeviceToHost));
    for (int i = 0; i < NUMBER_OF_WARPS; i++) {
        chkerr(cudaMemcpy(tasks_sizes + i, dd.wtasks_offset + (i * WTASKS_OFFSET_SIZE) + tasks_counts[i], sizeof(uint64_t), cudaMemcpyDeviceToHost));
        chkerr(cudaMemcpy(cliques_sizes + i, dd.wcliques_offset + (i * WCLIQUES_OFFSET_SIZE) + cliques_counts[i], sizeof(uint64_t), cudaMemcpyDeviceToHost));
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
        cliques_tcount += cliques_counts[i];
        if (cliques_counts[i] > cliques_mcount) {
            cliques_mcount = cliques_counts[i];
        }
        cliques_tsize += cliques_sizes[i];
        if (cliques_sizes[i] > cliques_msize) {
            cliques_msize = cliques_sizes[i];
        }
    }

    cout << "WTasks( TC: " << tasks_tcount << " TS: " << tasks_tsize << " MC: " << tasks_mcount << " MS: " << tasks_msize << ") WCliques ( TC: " << cliques_tcount << " TS: " << cliques_tsize << " MC: " << cliques_mcount << " MS: " << cliques_msize << ")" << endl;

    if (tasks_mcount > wto) {
        wto = tasks_mcount;
    }
    if (tasks_msize > wts) {
        wts = tasks_msize;
    }
    if (cliques_mcount > wco) {
        wco = cliques_mcount;
    }
    if (cliques_msize > wcs) {
        wcs = cliques_msize;
    }

    if (tasks_mcount > WTASKS_OFFSET_SIZE || tasks_msize > WTASKS_SIZE || cliques_mcount > WCLIQUES_OFFSET_SIZE || cliques_msize > WCLIQUES_SIZE) {
        cout << "!!! WBUFFER SIZE ERROR !!!" << endl;
        return true;
    }

    delete tasks_counts;
    delete tasks_sizes;
    delete cliques_counts;
    delete cliques_sizes;

    return false;
}

void print_All_Warp_Data_Sizes(GPU_Data& dd)
{
    uint64_t* tasks_counts = new uint64_t[NUMBER_OF_WARPS];
    uint64_t* tasks_sizes = new uint64_t[NUMBER_OF_WARPS];
    uint64_t* cliques_counts = new uint64_t[NUMBER_OF_WARPS];
    uint64_t* cliques_sizes = new uint64_t[NUMBER_OF_WARPS];

    chkerr(cudaMemcpy(tasks_counts, dd.wtasks_count, sizeof(uint64_t) * NUMBER_OF_WARPS, cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(cliques_counts, dd.wcliques_count, sizeof(uint64_t) * NUMBER_OF_WARPS, cudaMemcpyDeviceToHost));
    for (int i = 0; i < NUMBER_OF_WARPS; i++) {
        chkerr(cudaMemcpy(tasks_sizes + i, dd.wtasks_offset + (i * WTASKS_OFFSET_SIZE) + tasks_counts[i], sizeof(uint64_t), cudaMemcpyDeviceToHost));
        chkerr(cudaMemcpy(cliques_sizes + i, dd.wcliques_offset + (i * WCLIQUES_OFFSET_SIZE) + cliques_counts[i], sizeof(uint64_t), cudaMemcpyDeviceToHost));
    }

    cout << "WTasks Sizes: " << flush;
    for (int i = 0; i < NUMBER_OF_WARPS; i++) {
        cout << i << ":" << tasks_counts[i] << " " << tasks_sizes[i] << " " << flush;
    }
    cout << "\nWCliques Sizez: " << flush;
    for (int i = 0; i < NUMBER_OF_WARPS; i++) {
        cout << i << ":" << cliques_counts[i] << " " << cliques_sizes[i] << " " << flush;
    }

    delete tasks_counts;
    delete tasks_sizes;
    delete cliques_counts;
    delete cliques_sizes;
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
    uint64_t* tasks1_count = new uint64_t;
    uint64_t* tasks2_count = new uint64_t;
    uint64_t* buffer_count = new uint64_t;
    uint64_t* cliques_count = new uint64_t;
    uint64_t* tasks1_size = new uint64_t;
    uint64_t* tasks2_size = new uint64_t;
    uint64_t* buffer_size = new uint64_t;
    uint64_t* cliques_size = new uint64_t;

    chkerr(cudaMemcpy(current_level, dd.current_level, sizeof(uint64_t), cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(tasks1_count, dd.tasks1_count, sizeof(uint64_t), cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(tasks2_count, dd.tasks2_count, sizeof(uint64_t), cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(buffer_count, dd.buffer_count, sizeof(uint64_t), cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(cliques_count, dd.cliques_count, sizeof(uint64_t), cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(tasks1_size, dd.tasks1_offset + (*tasks1_count), sizeof(uint64_t), cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(tasks2_size, dd.tasks2_offset + (*tasks2_count), sizeof(uint64_t), cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(buffer_size, dd.buffer_offset + (*buffer_count), sizeof(uint64_t), cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(cliques_size, dd.cliques_offset + (*cliques_count), sizeof(uint64_t), cudaMemcpyDeviceToHost));

    cout << "L: " << (*current_level) << " T1: " << (*tasks1_count) << " " << (*tasks1_size) << " T2: " << (*tasks2_count) << " " << (*tasks2_size) << " B: " << (*buffer_count) << " " << (*buffer_size) << " C: " << 
        (*cliques_count) << " " << (*cliques_size) << endl << endl;

    if (*tasks1_size > mts) {
        mts = *tasks1_size;
    }
    if (*tasks2_size > mts) {
        mts = *tasks2_size;
    }
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

    if ((*tasks1_count) > EXPAND_THRESHOLD || (*tasks1_size) > TASKS_SIZE || (*tasks2_count) > EXPAND_THRESHOLD || (*tasks2_size) > TASKS_SIZE || (*buffer_count) > BUFFER_OFFSET_SIZE || (*buffer_size) > BUFFER_SIZE || (*cliques_count) > CLIQUES_OFFSET_SIZE ||
        (*cliques_size) > CLIQUES_SIZE) {
        cout << "!!! ARRAY SIZE ERROR !!!" << endl;
        return true;
    }

    delete current_level;
    delete tasks1_count;
    delete tasks2_count;
    delete buffer_count;
    delete cliques_count;
    delete tasks1_size;
    delete tasks2_size;
    delete buffer_size;
    delete cliques_size;
    
    return false;
}

void h_print_Data_Sizes(CPU_Data& hd, CPU_Cliques& hc)
{
    cout << "L: " << (*hd.current_level) << " T1: " << (*hd.tasks1_count) << " " << (*(hd.tasks1_offset + (*hd.tasks1_count))) << " T2: " << (*hd.tasks2_count) << " " << 
        (*(hd.tasks2_offset + (*hd.tasks2_count))) << " B: " << (*hd.buffer_count) << " " << (*(hd.buffer_offset + (*hd.buffer_count))) << " C: " << 
        (*hc.cliques_count) << " " << (*(hc.cliques_offset + (*hc.cliques_count))) << endl;

    if ((*(hd.tasks1_offset + (*hd.tasks1_count))) > mts) {
        mts = (*(hd.tasks1_offset + (*hd.tasks1_count)));
    }
    if ((*(hd.tasks2_offset + (*hd.tasks2_count))) > mts) {
        mts = (*(hd.tasks2_offset + (*hd.tasks2_count)));
    }
    if ((*(hd.buffer_offset + (*hd.buffer_count))) > mbs) {
        mbs = (*(hd.buffer_offset + (*hd.buffer_count)));
    }
    if ((*hd.buffer_count) > mbo) {
        mbo = (*hd.buffer_count);
    }
    if ((*(hc.cliques_offset + (*hc.cliques_count))) > mcs) {
        mcs = (*(hc.cliques_offset + (*hc.cliques_count)));
    }
    if ((*hc.cliques_count) > mco) {
        mco = (*hc.cliques_count);
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

    delete wtasks_count;
    delete wtasks_offset;
    delete wtasks_vertices;
}

void print_WClique_Buffers(GPU_Data& dd)
{
    uint64_t* wcliques_count = new uint64_t[NUMBER_OF_WARPS];
    uint64_t* wcliques_offset = new uint64_t[NUMBER_OF_WARPS * WCLIQUES_OFFSET_SIZE];
    int* wcliques_vertex = new int[NUMBER_OF_WARPS * WCLIQUES_SIZE];

    chkerr(cudaMemcpy(wcliques_count, dd.wcliques_count, sizeof(uint64_t) * NUMBER_OF_WARPS, cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(wcliques_offset, dd.wcliques_offset, sizeof(uint64_t) * (NUMBER_OF_WARPS * WTASKS_OFFSET_SIZE), cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(wcliques_vertex, dd.wcliques_vertex, sizeof(int) * (NUMBER_OF_WARPS * WTASKS_SIZE), cudaMemcpyDeviceToHost));

    cout << endl << " --- Warp Clique Buffers details --- " << endl;
    for (int i = 0; i < NUMBER_OF_WARPS; i++) {
        int wcliques_offset_start = WTASKS_OFFSET_SIZE * i;
        int wcliques_start = WTASKS_SIZE * i;

        cout << endl << "Warp " << i << ": " << "Size : " << wcliques_count[i] << endl;
        cout << "Offsets:" << endl;
        for (int j = 0; j <= wcliques_count[i]; j++) {
            cout << wcliques_offset[wcliques_offset_start + j] << " ";
        }
        cout << endl << "Vertex:" << endl;
        for (int j = 0; j < wcliques_offset[wcliques_offset_start + wcliques_count[i]]; j++) {
            cout << wcliques_vertex[wcliques_start + j] << " ";
        }
    }
    cout << endl << endl;

    delete wcliques_count;
    delete wcliques_offset;
    delete wcliques_vertex;
}

void print_GPU_Cliques(GPU_Data& dd)
{
    uint64_t* cliques_count = new uint64_t;
    uint64_t* cliques_offset = new uint64_t[CLIQUES_OFFSET_SIZE];
    int* cliques_vertex = new int[CLIQUES_SIZE];

    chkerr(cudaMemcpy(cliques_count, dd.cliques_count, sizeof(uint64_t), cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(cliques_offset, dd.cliques_offset, sizeof(uint64_t) * CLIQUES_OFFSET_SIZE, cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(cliques_vertex, dd.cliques_vertex, sizeof(int) * CLIQUES_SIZE, cudaMemcpyDeviceToHost));

    cout << endl << " --- (GPU_Cliques)device_cliques details --- " << endl;
    cout << endl << "Cliques: " << "Size: " << (*cliques_count) << endl;
    cout << endl << "Offsets:" << endl;
    for (uint64_t i = 0; i <= (*cliques_count); i++) {
        cout << cliques_offset[i] << " ";
    }

    cout << endl << "Vertex:" << endl;
    for (uint64_t i = 0; i < (*cliques_count); i++) {
        cout << i << " S: " << cliques_offset[i] << " E: " << cliques_offset[i+1] << " " << flush;
        for (uint64_t j = cliques_offset[i]; j < cliques_offset[i + 1]; j++) {
            cout << cliques_vertex[j] << " " << flush;
        }
        cout << endl;
    }

    delete cliques_count;
    delete cliques_offset;
    delete cliques_vertex;

    return;

    cout << endl << "Vertex:" << endl;
    for (uint64_t i = 0; i < cliques_offset[(*cliques_count)]; i++) {
        cout << cliques_vertex[i] << " ";
    }
    cout << endl;
}

void print_CPU_Cliques(CPU_Cliques& hc)
{
    cout << endl << " --- (CPU_Cliques)host_cliques details --- " << endl;
    cout << endl << "Cliques: " << "Size: " << (*(hc.cliques_count)) << endl;
    cout << endl << "Offsets:" << endl;
    for (uint64_t i = 0; i <= (*(hc.cliques_count)); i++) {
        cout << hc.cliques_offset[i] << " ";
    }
    cout << endl << "Vertex:" << endl;
    for (uint64_t i = 0; i < hc.cliques_offset[(*(hc.cliques_count))]; i++) {
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
    wcs = 0;
    wco = 0;
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
        << "WCLIQUES SIZE: " << wcs << endl
        << "WCLIQUES OFFSET SIZE: " << wco << endl
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

void RemoveNonMax(const char* szset_filename, const char* szoutput_filename)
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


    printf(">:NUMBER OF MAXIMAL CLIQUES: %d\n", gntotal_max_cliques);
}


__global__ void transfer_cliques(GPU_Data dd)
{
    __shared__ uint64_t cliques_write[WARPS_PER_BLOCK];
    __shared__ int cliques_offset_write[WARPS_PER_BLOCK];

    // warp level
    if (LANE_IDX == 0)
    {
        cliques_write[WIB_IDX] = 0;
        cliques_offset_write[WIB_IDX] = 1;

        for (int i = 0; i < WARP_IDX; i++) {
            cliques_offset_write[WIB_IDX] += dd.wcliques_count[i];
            cliques_write[WIB_IDX] += dd.wcliques_offset[(WCLIQUES_OFFSET_SIZE * i) + dd.wcliques_count[i]];
        }
    }
    __syncwarp();
    

    //move to cliques
    for (int i = LANE_IDX + 1; i <= dd.wcliques_count[WARP_IDX]; i += WARP_SIZE) {
        dd.cliques_offset[(*(dd.cliques_offset_start)) + cliques_offset_write[WIB_IDX] + i - 2] = dd.wcliques_offset[(WCLIQUES_OFFSET_SIZE * WARP_IDX) + i] + (*(dd.cliques_start)) + 
            cliques_write[WIB_IDX];
    }
    for (int i = LANE_IDX; i < dd.wcliques_offset[(WCLIQUES_OFFSET_SIZE * WARP_IDX) + dd.wcliques_count[WARP_IDX]]; i += WARP_SIZE) {
        dd.cliques_vertex[(*(dd.cliques_start)) + cliques_write[WIB_IDX] + i] = dd.wcliques_vertex[(WCLIQUES_SIZE * WARP_IDX) + i];
    }

    if (IDX == 0) {
        // handle tasks and buffer counts
        (*(dd.cliques_count)) += (*(dd.total_cliques));

        (*(dd.total_tasks)) = 0;
        (*(dd.total_cliques)) = 0;
    }
}