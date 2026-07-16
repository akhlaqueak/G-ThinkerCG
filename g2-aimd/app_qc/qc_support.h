#ifndef G2_AIMD_APP_QC_SUPPORT_H
#define G2_AIMD_APP_QC_SUPPORT_H

#include <assert.h>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <limits>
#include <locale>
#include <set>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <sys/timeb.h>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <sm_30_intrinsics.h>

using namespace std;

// GPU KERNEL LAUNCH
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 1024
#endif
#ifndef NUM_OF_BLOCKS
#define NUM_OF_BLOCKS 108
#endif
#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

// GPU INFORMATION
#ifndef IDX
#define IDX ((blockIdx.x * blockDim.x) + threadIdx.x)
#endif
#ifndef WARP_IDX
#define WARP_IDX (IDX / WARP_SIZE)
#endif
#ifndef LANE_IDX
#define LANE_IDX (IDX % WARP_SIZE)
#endif
#ifndef WIB_IDX
#define WIB_IDX (threadIdx.x / WARP_SIZE)
#endif
#ifndef WARPS_PER_BLOCK
#define WARPS_PER_BLOCK (BLOCK_SIZE / WARP_SIZE)
#endif
#ifndef NUMBER_OF_WARPS
#define NUMBER_OF_WARPS (NUM_OF_BLOCKS * WARPS_PER_BLOCK)
#endif

// DATA STRUCTURE SIZE
#ifndef BUFFER_SIZE
#define BUFFER_SIZE 900000000
#endif
#ifndef BUFFER_OFFSET_SIZE
#define BUFFER_OFFSET_SIZE 90000000
#endif
#ifndef CLIQUES_SIZE
#define CLIQUES_SIZE 1000000000
#endif
#ifndef CLIQUES_OFFSET_SIZE
#define CLIQUES_OFFSET_SIZE 30000000
#endif
#ifndef WTASKS_SIZE
#define WTASKS_SIZE 150000L
#endif
#ifndef WTASKS_OFFSET_SIZE
#define WTASKS_OFFSET_SIZE 10000
#endif
#ifndef WVERTICES_SIZE
#define WVERTICES_SIZE 32000
#endif
#ifndef VERTICES_SIZE
#define VERTICES_SIZE 70
#endif

struct Vertex
{
    int vertexid;
    int label;
    int indeg;
    int exdeg;
    int lvl2adj;
};

class CPU_Graph
{
public:
    int number_of_vertices;
    uint64_t number_of_edges;
    uint64_t number_of_lvl2adj;

    int *onehop_neighbors;
    uint64_t *onehop_offsets;
    int *twohop_neighbors;
    uint64_t *twohop_offsets;

    explicit CPU_Graph(ifstream &graph_stream)
    {
        uint64_t edge_count_u64 = 0;
        graph_stream.read(reinterpret_cast<char *>(&number_of_vertices), sizeof(number_of_vertices));
        graph_stream.read(reinterpret_cast<char *>(&edge_count_u64), sizeof(edge_count_u64));
        graph_stream.read(reinterpret_cast<char *>(&number_of_lvl2adj), sizeof(number_of_lvl2adj));
        if (!graph_stream.good())
            throw runtime_error("Failed to read graph header");

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
            graph_stream.read(reinterpret_cast<char *>(&uintV_size), sizeof(uintV_size));
            graph_stream.read(reinterpret_cast<char *>(&uintE_size), sizeof(uintE_size));
            graph_stream.read(reinterpret_cast<char *>(&vertex_count), sizeof(vertex_count));
            graph_stream.read(reinterpret_cast<char *>(&edge_count), sizeof(edge_count));

            if (graph_stream.good() && uintV_size == sizeof(int) && uintE_size == sizeof(uint64_t))
                throw runtime_error("Graph file is in the original 1-hop binary format. Run binToSer.cpp to generate the expanded binary graph first.");

            throw runtime_error("Graph file is not in the expected expanded binary format.");
        }

        onehop_neighbors = new int[number_of_edges];
        onehop_offsets = new uint64_t[number_of_vertices + 1];
        twohop_neighbors = new int[number_of_lvl2adj];
        twohop_offsets = new uint64_t[number_of_vertices + 1];

        graph_stream.read(reinterpret_cast<char *>(onehop_neighbors), sizeof(int) * number_of_edges);
        graph_stream.read(reinterpret_cast<char *>(onehop_offsets), sizeof(uint64_t) * (number_of_vertices + 1));
        graph_stream.read(reinterpret_cast<char *>(twohop_neighbors), sizeof(int) * number_of_lvl2adj);
        graph_stream.read(reinterpret_cast<char *>(twohop_offsets), sizeof(uint64_t) * (number_of_vertices + 1));
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

struct CPU_Data
{
    uint64_t *buffer_count;
    uint64_t *buffer_offset;
    Vertex *buffer_vertices;

    uint64_t *current_level;
    bool *maximal_expansion;
    bool *dumping_cliques;

    int *vertex_order_map;
    int *remaining_candidates;
    int *removed_candidates;
    int *remaining_count;
    int *removed_count;
    int *candidate_indegs;

    Vertex *initial_vertices;
    size_t initial_vertices_count;
    int *initial_order_map;
};

struct CPU_Cliques
{
    uint64_t cliques_count = 0;
    uint64_t max_clique_size = 0;
    std::vector<uint64_t> cliques_offset;
    std::vector<int> cliques_vertex;
};

struct GPU_Data
{
    uint64_t *current_level;

    uint64_t *buffer_count;
    uint64_t *buffer_offset;
    Vertex *buffer_vertices;

    uint64_t *wtasks_count;
    uint64_t *wtasks_offset;
    Vertex *wtasks_vertices;

    Vertex *global_vertices;
    Vertex *initial_vertices;
    uint64_t *initial_vertices_count;
    int *initial_order_map;

    int *removed_candidates;
    int *lane_removed_candidates;
    Vertex *remaining_candidates;
    int *lane_remaining_candidates;
    int *candidate_indegs;
    int *lane_candidate_indegs;
    int *adjacencies;

    int *total_tasks;

    double *minimum_degree_ratio;
    int *minimum_degrees;
    int *minimum_clique_size;
    int *scheduling_toggle;
    bool *store_cliques;

    uint64_t *buffer_offset_start;
    uint64_t *buffer_start;

    int *number_of_vertices;
    uint64_t *number_of_edges;
    int *onehop_neighbors;
    uint64_t *onehop_offsets;
    int *twohop_neighbors;
    uint64_t *twohop_offsets;

    uint64_t *cliques_count;
    uint64_t *cliques_vertex_count;
    uint64_t *max_clique_size;
    uint64_t *cliques_offset;
    uint64_t *cliques_size;
    int *cliques_vertex;

    int *current_task;
};

inline double minimum_degree_ratio;
inline int minimum_clique_size;
inline int *minimum_degrees = nullptr;
inline int scheduling_toggle;
inline bool store_cliques = false;

class QCApp;
extern CPU_Data hd;
extern GPU_Data dd;
extern CPU_Cliques hc;
extern CPU_Graph *hg;

inline int h_sort_vert_Q(const void *a, const void *b);

namespace qc_runtime_state
{
static bool runtime_prepared = false;
static bool remove_nonmax = false;
static bool help_requested = false;
static std::string graph_file;
static std::string output_file = "output.txt";
static std::string temp_filename = "t_cliques.txt";
static std::ofstream temp_results;
static Timer search_timer;
static uint64_t total_pre_max_quasi_cliques = 0;
static uint64_t max_clique_size = 0;
}

inline std::vector<std::string> qc_process_args()
{
    std::vector<std::string> args;
    std::ifstream input("/proc/self/cmdline", std::ios::binary);
    if (!input.is_open())
        return args;

    std::string arg;
    while (std::getline(input, arg, '\0'))
        args.push_back(arg);
    return args;
}

inline std::string qc_get_option_value(const std::vector<std::string> &args, const std::string &name, const std::string &default_value)
{
    for (size_t i = 1; i < args.size(); ++i)
    {
        if (args[i] == name)
        {
            if (i + 1 >= args.size())
                return default_value;
            return args[i + 1];
        }
    }
    return default_value;
}

inline int qc_get_option_int(const std::vector<std::string> &args, const std::string &name, int default_value)
{
    std::string value = qc_get_option_value(args, name, "");
    if (value.empty())
        return default_value;
    return std::stoi(value);
}

inline double qc_get_option_double(const std::vector<std::string> &args, const std::string &name, double default_value)
{
    std::string value = qc_get_option_value(args, name, "");
    if (value.empty())
        return default_value;
    return std::stod(value);
}

inline bool qc_has_flag(const std::vector<std::string> &args, const std::string &name)
{
    for (size_t i = 1; i < args.size(); ++i)
    {
        if (args[i] == name)
            return true;
    }
    return false;
}

inline bool qc_has_help_flag(const std::vector<std::string> &args)
{
    return qc_has_flag(args, "--help");
}

inline bool qc_apply_remove_nonmax_from_args(const std::vector<std::string> &args)
{
    for (size_t i = 1; i < args.size(); ++i)
    {
        if (args[i] == "-rmnonmax" || args[i] == "--rmnonmax")
        {
            if (i + 1 >= args.size())
                return true;

            const std::string &value = args[i + 1];
            if (!value.empty() && value[0] == '-')
                return true;

            return !(value == "0" || value == "false" || value == "False" || value == "off");
        }
    }
    return false;
}

inline std::string qc_quote_command_arg(const std::string &arg)
{
    if (arg.empty())
        return "''";

    bool needs_quote = false;
    for (char c : arg)
    {
        if (std::isspace(static_cast<unsigned char>(c)) || c == '\'' || c == '"' || c == '\\' || c == '$' || c == '`')
        {
            needs_quote = true;
            break;
        }
    }

    if (!needs_quote)
        return arg;

    std::string quoted = "'";
    for (char c : arg)
    {
        if (c == '\'')
            quoted += "'\\''";
        else
            quoted += c;
    }
    quoted += "'";
    return quoted;
}

inline std::string qc_supplied_command(const std::vector<std::string> &args)
{
    std::string command;
    for (size_t i = 0; i < args.size(); ++i)
    {
        if (i > 0)
            command += " ";
        command += qc_quote_command_arg(args[i]);
    }
    return command;
}

inline std::string qc_executable_modified_time(const std::string &program)
{
    if (program.empty())
        return "unavailable";

    struct stat st;
    if (stat(program.c_str(), &st) != 0)
        return "unavailable";

    char buffer[64];
    std::tm tm_value;
    if (localtime_r(&st.st_mtime, &tm_value) == nullptr)
        return "unavailable";

    if (std::strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S %Z", &tm_value) == 0)
        return "unavailable";

    return buffer;
}

inline std::string qc_expand_graph_path_from_data_graph(const std::string &data_graph_path)
{
    if (data_graph_path.empty())
        return "";

    size_t slash = data_graph_path.find_last_of("/\\");
    size_t dot = data_graph_path.find_last_of('.');
    if (dot == std::string::npos || (slash != std::string::npos && dot < slash))
        return data_graph_path + ".sbin";

    return data_graph_path.substr(0, dot) + ".sbin";
}

inline void print_help(const char *program)
{
    std::cout << "Usage: " << program << " -dg <graph.bin> [-f <graph.sbin>] [options]" << std::endl;
    std::cout << std::endl;
    std::cout << "Required:" << std::endl;
    std::cout << "  -dg <path>          Input 1-hop binary graph used by the G2-AIMD worker." << std::endl;
    std::cout << std::endl;
    std::cout << "QC graph input:" << std::endl;
    std::cout << "  -f <path>           Expanded binary graph (.sbin). If omitted, derive from -dg." << std::endl;
    std::cout << std::endl;
    std::cout << "QC parameters:" << std::endl;
    std::cout << "  -g <gamma>          Minimum degree ratio in [0.5, 1]. Default: 0.5" << std::endl;
    std::cout << "  -k <size>           Minimum quasi-clique size (> 1). Default: 10" << std::endl;
    std::cout << "  -o <file>           Output file for maximal quasi-cliques. Default: output.txt" << std::endl;
    std::cout << "  -rmnonmax [0|1]     Remove non-maximal results. Default: off" << std::endl;
    std::cout << "  -sched <0|1>        Scheduling mode: 0=dynamic, 1=static. Default: 0" << std::endl;
    std::cout << std::endl;
    std::cout << "Other:" << std::endl;
    std::cout << "  --help              Show this help message and exit." << std::endl;
}

inline void initialize_tasks(CPU_Graph &graph, CPU_Data &host_data)
{
    int pvertexid;
    uint64_t pneighbors_start;
    uint64_t pneighbors_end;
    int phelper1;

    int maximum_degree;
    int maximum_degree_index;

    int total_vertices = graph.number_of_vertices;
    int number_of_candidates = total_vertices;
    Vertex *vertices = new Vertex[total_vertices];

    (*host_data.remaining_count) = 0;
    (*host_data.removed_count) = 0;

    for (int i = 0; i < total_vertices; i++)
    {
        vertices[i].vertexid = i;
        vertices[i].indeg = 0;
        vertices[i].exdeg = graph.onehop_offsets[i + 1] - graph.onehop_offsets[i];
        vertices[i].lvl2adj = graph.twohop_offsets[i + 1] - graph.twohop_offsets[i];
        if (vertices[i].exdeg >= minimum_degrees[minimum_clique_size] && vertices[i].lvl2adj >= minimum_clique_size - 1)
        {
            vertices[i].label = 0;
            host_data.remaining_candidates[(*host_data.remaining_count)++] = i;
        }
        else
        {
            vertices[i].label = -1;
            host_data.removed_candidates[(*host_data.removed_count)++] = i;
        }
    }

    while ((*host_data.remaining_count) < number_of_candidates / 2)
    {
        number_of_candidates = (*host_data.remaining_count);

        for (int i = 0; i < number_of_candidates; i++)
            vertices[host_data.remaining_candidates[i]].exdeg = 0;

        for (int i = 0; i < number_of_candidates; i++)
        {
            pvertexid = host_data.remaining_candidates[i];
            pneighbors_start = graph.onehop_offsets[pvertexid];
            pneighbors_end = graph.onehop_offsets[pvertexid + 1];
            for (uint64_t j = pneighbors_start; j < pneighbors_end; j++)
            {
                phelper1 = graph.onehop_neighbors[j];
                if (vertices[phelper1].label == 0)
                    vertices[phelper1].exdeg++;
            }
        }

        (*host_data.remaining_count) = 0;
        (*host_data.removed_count) = 0;

        for (int i = 0; i < number_of_candidates; i++)
        {
            phelper1 = host_data.remaining_candidates[i];
            if (vertices[phelper1].exdeg >= minimum_degrees[minimum_clique_size])
            {
                host_data.remaining_candidates[(*host_data.remaining_count)++] = phelper1;
            }
            else
            {
                vertices[phelper1].label = -1;
                host_data.removed_candidates[(*host_data.removed_count)++] = phelper1;
            }
        }
    }
    number_of_candidates = (*host_data.remaining_count);

    int removed_start = 0;
    while ((*host_data.removed_count) > removed_start)
    {
        pvertexid = host_data.removed_candidates[removed_start];
        pneighbors_start = graph.onehop_offsets[pvertexid];
        pneighbors_end = graph.onehop_offsets[pvertexid + 1];

        for (uint64_t j = pneighbors_start; j < pneighbors_end; j++)
        {
            phelper1 = graph.onehop_neighbors[j];

            if (vertices[phelper1].label == 0)
            {
                vertices[phelper1].exdeg--;

                if (vertices[phelper1].exdeg < minimum_degrees[minimum_clique_size])
                {
                    vertices[phelper1].label = -1;
                    number_of_candidates--;
                    host_data.removed_candidates[(*host_data.removed_count)++] = phelper1;
                }
            }
        }
        removed_start++;
    }

    maximum_degree = 0;
    maximum_degree_index = 0;
    for (int i = 0; i < total_vertices; i++)
    {
        if (vertices[i].label == 0 && vertices[i].exdeg > maximum_degree)
        {
            maximum_degree = vertices[i].exdeg;
            maximum_degree_index = i;
        }
    }
    vertices[maximum_degree_index].label = 3;

    pneighbors_start = graph.onehop_offsets[maximum_degree_index];
    pneighbors_end = graph.onehop_offsets[maximum_degree_index + 1];
    for (uint64_t i = pneighbors_start; i < pneighbors_end; i++)
    {
        pvertexid = graph.onehop_neighbors[i];
        if (vertices[pvertexid].label == 0)
            vertices[pvertexid].label = 2;
    }

    qsort(vertices, total_vertices, sizeof(Vertex), h_sort_vert_Q);
    total_vertices = number_of_candidates;
    for (int j = 0; j < total_vertices; j++)
        vertices[j].lvl2adj = 0;

    host_data.initial_vertices = vertices;
    host_data.initial_vertices_count = static_cast<size_t>(total_vertices);
    memset(host_data.initial_order_map, -1, sizeof(int) * graph.number_of_vertices);
    for (int i = 0; i < total_vertices; i++)
        host_data.initial_order_map[host_data.initial_vertices[i].vertexid] = i;
}

inline void allocate_runtime_memory(CPU_Data &host_data, GPU_Data &device_data, CPU_Cliques &host_cliques, CPU_Graph &graph)
{
    chkerr(cudaMalloc((void **)&device_data.number_of_vertices, sizeof(int)));
    chkerr(cudaMalloc((void **)&device_data.number_of_edges, sizeof(uint64_t)));
    chkerr(cudaMalloc((void **)&device_data.onehop_neighbors, sizeof(int) * graph.number_of_edges));
    chkerr(cudaMalloc((void **)&device_data.onehop_offsets, sizeof(uint64_t) * (graph.number_of_vertices + 1)));
    chkerr(cudaMalloc((void **)&device_data.twohop_neighbors, sizeof(int) * graph.number_of_lvl2adj));
    chkerr(cudaMalloc((void **)&device_data.twohop_offsets, sizeof(uint64_t) * (graph.number_of_vertices + 1)));

    chkerr(cudaMemcpy(device_data.number_of_vertices, &(graph.number_of_vertices), sizeof(int), cudaMemcpyHostToDevice));
    chkerr(cudaMemcpy(device_data.number_of_edges, &(graph.number_of_edges), sizeof(uint64_t), cudaMemcpyHostToDevice));
    chkerr(cudaMemcpy(device_data.onehop_neighbors, graph.onehop_neighbors, sizeof(int) * graph.number_of_edges, cudaMemcpyHostToDevice));
    chkerr(cudaMemcpy(device_data.onehop_offsets, graph.onehop_offsets, sizeof(uint64_t) * (graph.number_of_vertices + 1), cudaMemcpyHostToDevice));
    chkerr(cudaMemcpy(device_data.twohop_neighbors, graph.twohop_neighbors, sizeof(int) * graph.number_of_lvl2adj, cudaMemcpyHostToDevice));
    chkerr(cudaMemcpy(device_data.twohop_offsets, graph.twohop_offsets, sizeof(uint64_t) * (graph.number_of_vertices + 1), cudaMemcpyHostToDevice));

    host_data.buffer_count = new uint64_t;
    host_data.buffer_offset = new uint64_t[BUFFER_OFFSET_SIZE];
    host_data.buffer_vertices = new Vertex[BUFFER_SIZE];
    host_data.buffer_offset[0] = 0;
    (*(host_data.buffer_count)) = 0;

    host_data.current_level = new uint64_t;
    host_data.maximal_expansion = new bool;
    host_data.dumping_cliques = new bool;
    (*host_data.current_level) = 0;
    (*host_data.maximal_expansion) = false;
    (*host_data.dumping_cliques) = false;

    host_data.vertex_order_map = new int[graph.number_of_vertices];
    host_data.remaining_candidates = new int[graph.number_of_vertices];
    host_data.removed_candidates = new int[graph.number_of_vertices];
    host_data.remaining_count = new int;
    host_data.removed_count = new int;
    host_data.candidate_indegs = new int[graph.number_of_vertices];
    host_data.initial_order_map = new int[graph.number_of_vertices];
    host_data.initial_vertices = nullptr;
    host_data.initial_vertices_count = 0;

    memset(host_data.vertex_order_map, -1, sizeof(int) * graph.number_of_vertices);
    memset(host_data.initial_order_map, -1, sizeof(int) * graph.number_of_vertices);

    host_cliques.cliques_count = 0;
    host_cliques.max_clique_size = 0;
    host_cliques.cliques_vertex.clear();
    if (store_cliques)
        host_cliques.cliques_offset.assign(1, 0);
    else
        host_cliques.cliques_offset.clear();

    chkerr(cudaMalloc((void **)&device_data.current_level, sizeof(uint64_t)));
    chkerr(cudaMalloc((void **)&device_data.wtasks_count, sizeof(uint64_t) * NUMBER_OF_WARPS));
    chkerr(cudaMalloc((void **)&device_data.wtasks_offset, (sizeof(uint64_t) * WTASKS_OFFSET_SIZE) * NUMBER_OF_WARPS));
    chkerr(cudaMalloc((void **)&device_data.wtasks_vertices, (sizeof(Vertex) * WTASKS_SIZE) * NUMBER_OF_WARPS));
    chkerr(cudaMemset(device_data.wtasks_offset, 0, (sizeof(uint64_t) * WTASKS_OFFSET_SIZE) * NUMBER_OF_WARPS));
    chkerr(cudaMemset(device_data.wtasks_count, 0, sizeof(uint64_t) * NUMBER_OF_WARPS));

    chkerr(cudaMalloc((void **)&device_data.global_vertices, (sizeof(Vertex) * WVERTICES_SIZE) * NUMBER_OF_WARPS));

    chkerr(cudaMalloc((void **)&device_data.removed_candidates, (sizeof(int) * WVERTICES_SIZE) * NUMBER_OF_WARPS));
    chkerr(cudaMalloc((void **)&device_data.lane_removed_candidates, (sizeof(int) * WVERTICES_SIZE) * NUMBER_OF_WARPS));
    chkerr(cudaMalloc((void **)&device_data.remaining_candidates, (sizeof(Vertex) * WVERTICES_SIZE) * NUMBER_OF_WARPS));
    chkerr(cudaMalloc((void **)&device_data.lane_remaining_candidates, (sizeof(int) * WVERTICES_SIZE) * NUMBER_OF_WARPS));
    chkerr(cudaMalloc((void **)&device_data.candidate_indegs, (sizeof(int) * WVERTICES_SIZE) * NUMBER_OF_WARPS));
    chkerr(cudaMalloc((void **)&device_data.lane_candidate_indegs, (sizeof(int) * WVERTICES_SIZE) * NUMBER_OF_WARPS));
    chkerr(cudaMalloc((void **)&device_data.adjacencies, (sizeof(int) * WVERTICES_SIZE) * NUMBER_OF_WARPS));

    chkerr(cudaMalloc((void **)&device_data.minimum_degree_ratio, sizeof(double)));
    chkerr(cudaMalloc((void **)&device_data.minimum_degrees, sizeof(int) * (graph.number_of_vertices + 1)));
    chkerr(cudaMalloc((void **)&device_data.minimum_clique_size, sizeof(int)));
    chkerr(cudaMalloc((void **)&device_data.scheduling_toggle, sizeof(int)));
    chkerr(cudaMalloc((void **)&device_data.store_cliques, sizeof(bool)));

    chkerr(cudaMemcpy(device_data.minimum_degree_ratio, &minimum_degree_ratio, sizeof(double), cudaMemcpyHostToDevice));
    chkerr(cudaMemcpy(device_data.minimum_degrees, minimum_degrees, sizeof(int) * (graph.number_of_vertices + 1), cudaMemcpyHostToDevice));
    chkerr(cudaMemcpy(device_data.minimum_clique_size, &minimum_clique_size, sizeof(int), cudaMemcpyHostToDevice));
    chkerr(cudaMemcpy(device_data.scheduling_toggle, &scheduling_toggle, sizeof(int), cudaMemcpyHostToDevice));
    chkerr(cudaMemcpy(device_data.store_cliques, &store_cliques, sizeof(bool), cudaMemcpyHostToDevice));

    chkerr(cudaMalloc((void **)&device_data.total_tasks, sizeof(int)));
    chkerr(cudaMemset(device_data.total_tasks, 0, sizeof(int)));

    chkerr(cudaMalloc((void **)&device_data.cliques_count, sizeof(uint64_t)));
    chkerr(cudaMalloc((void **)&device_data.cliques_vertex_count, sizeof(uint64_t)));
    chkerr(cudaMalloc((void **)&device_data.max_clique_size, sizeof(uint64_t)));
    device_data.cliques_vertex = nullptr;
    device_data.cliques_offset = nullptr;
    device_data.cliques_size = nullptr;
    if (store_cliques)
    {
        chkerr(cudaMalloc((void **)&device_data.cliques_vertex, sizeof(int) * CLIQUES_SIZE));
        chkerr(cudaMalloc((void **)&device_data.cliques_offset, sizeof(uint64_t) * CLIQUES_OFFSET_SIZE));
        chkerr(cudaMalloc((void **)&device_data.cliques_size, sizeof(uint64_t) * CLIQUES_OFFSET_SIZE));
        chkerr(cudaMemset(device_data.cliques_offset, 0, sizeof(uint64_t)));
        chkerr(cudaMemset(device_data.cliques_size, 0, sizeof(uint64_t)));
    }
    chkerr(cudaMemset(device_data.cliques_count, 0, sizeof(uint64_t)));
    chkerr(cudaMemset(device_data.cliques_vertex_count, 0, sizeof(uint64_t)));
    chkerr(cudaMemset(device_data.max_clique_size, 0, sizeof(uint64_t)));

    chkerr(cudaMalloc((void **)&device_data.buffer_offset_start, sizeof(uint64_t)));
    chkerr(cudaMalloc((void **)&device_data.buffer_start, sizeof(uint64_t)));
    chkerr(cudaMalloc((void **)&device_data.current_task, sizeof(int)));
}
inline std::vector<int> output_vertex_id_map;

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

inline void calculate_minimum_degrees(CPU_Graph &hg)
{
    minimum_degrees = new int[hg.number_of_vertices + 1];
    minimum_degrees[0] = 0;
    for (int i = 1; i <= hg.number_of_vertices; i++)
        minimum_degrees[i] = ceil(minimum_degree_ratio * (i - 1));
}

inline int h_sort_vert_Q(const void *a, const void *b)
{
    Vertex *v1 = (Vertex *)a;
    Vertex *v2 = (Vertex *)b;

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

inline void flush_cliques(CPU_Cliques &hc, ofstream &temp_results)
{
    for (size_t i = 0; i + 1 < hc.cliques_offset.size(); i++)
    {
        uint64_t start = hc.cliques_offset[i];
        uint64_t end = hc.cliques_offset[i + 1];
        temp_results << end - start << " ";
        for (uint64_t j = start; j < end; j++)
            temp_results << output_vertex_id(hc.cliques_vertex[j]) << " ";
        temp_results << "\n";
    }
    hc.cliques_vertex.clear();
    if (store_cliques)
        hc.cliques_offset.assign(1, 0);
    else
        hc.cliques_offset.clear();
    hc.cliques_count = 0;
    hc.max_clique_size = 0;
}

inline uint64_t dump_cliques(CPU_Cliques &hc, GPU_Data &dd, ofstream &temp_results, bool read_gpu = true)
{
    uint64_t dumped_cliques_count = hc.cliques_count;

    if (store_cliques)
        flush_cliques(hc, temp_results);
    else
    {
        hc.cliques_count = 0;
        hc.max_clique_size = 0;
        hc.cliques_vertex.clear();
        hc.cliques_offset.clear();
    }

    if (!read_gpu)
        return dumped_cliques_count;

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
        throw std::runtime_error("GPU clique count exceeds CLIQUES_OFFSET_SIZE");
    if (gpu_cliques_size > CLIQUES_SIZE)
        throw std::runtime_error("GPU clique vertex size exceeds CLIQUES_SIZE");

    std::vector<uint64_t> gpu_cliques_offset(gpu_cliques_count);
    std::vector<uint64_t> gpu_cliques_sizes(gpu_cliques_count);
    chkerr(cudaMemcpy(gpu_cliques_offset.data(), dd.cliques_offset, sizeof(uint64_t) * gpu_cliques_count, cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(gpu_cliques_sizes.data(), dd.cliques_size, sizeof(uint64_t) * gpu_cliques_count, cudaMemcpyDeviceToHost));

    for (uint64_t i = 0; i < gpu_cliques_count; i++)
    {
        if (gpu_cliques_offset[i] > gpu_cliques_size || gpu_cliques_sizes[i] > gpu_cliques_size - gpu_cliques_offset[i])
            throw std::runtime_error("GPU clique offsets are corrupted");
    }

    std::vector<int> gpu_cliques_vertex(gpu_cliques_size);
    if (!gpu_cliques_vertex.empty())
        chkerr(cudaMemcpy(gpu_cliques_vertex.data(), dd.cliques_vertex, sizeof(int) * gpu_cliques_vertex.size(), cudaMemcpyDeviceToHost));
    chkerr(cudaDeviceSynchronize());

    for (uint64_t i = 0; i < gpu_cliques_count; i++)
    {
        uint64_t start = gpu_cliques_offset[i];
        uint64_t end = start + gpu_cliques_sizes[i];
        temp_results << end - start << " ";
        for (uint64_t j = start; j < end; j++)
            temp_results << output_vertex_id(gpu_cliques_vertex[j]) << " ";
        temp_results << "\n";
    }

    cudaMemset(dd.cliques_count, 0, sizeof(uint64_t));
    cudaMemset(dd.cliques_vertex_count, 0, sizeof(uint64_t));
    return dumped_cliques_count + gpu_cliques_count;
}

inline void free_memory(CPU_Data &hd, GPU_Data &dd, CPU_Cliques &hc, bool free_gpu = true)
{
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

    hc.cliques_count = 0;
    hc.max_clique_size = 0;
    hc.cliques_vertex.clear();
    hc.cliques_vertex.shrink_to_fit();
    hc.cliques_offset.clear();
    hc.cliques_offset.shrink_to_fit();

    delete[] minimum_degrees;
    minimum_degrees = nullptr;

    if (!free_gpu)
        return;

    chkerr(cudaFree(dd.number_of_vertices));
    chkerr(cudaFree(dd.number_of_edges));
    chkerr(cudaFree(dd.onehop_neighbors));
    chkerr(cudaFree(dd.onehop_offsets));
    chkerr(cudaFree(dd.twohop_neighbors));
    chkerr(cudaFree(dd.twohop_offsets));
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
    chkerr(cudaFree(dd.cliques_count));
    chkerr(cudaFree(dd.cliques_vertex_count));
    chkerr(cudaFree(dd.max_clique_size));
    if (dd.cliques_vertex)
        chkerr(cudaFree(dd.cliques_vertex));
    if (dd.cliques_offset)
        chkerr(cudaFree(dd.cliques_offset));
    if (dd.cliques_size)
        chkerr(cudaFree(dd.cliques_size));
    chkerr(cudaFree(dd.buffer_offset_start));
    chkerr(cudaFree(dd.buffer_start));
    chkerr(cudaFree(dd.current_task));
}

inline int comp_int(const void *e1, const void *e2)
{
    int n1 = *(int *)e1;
    int n2 = *(int *)e2;
    if (n1 > n2)
        return 1;
    else if (n1 < n2)
        return -1;
    else
        return 0;
}

struct TREE_NODE
{
    int nid;
    TREE_NODE *pchild;
    TREE_NODE *pright_sib;
    bool bis_max;
};

#define TNODE_PAGE_SIZE (1 << 10)

struct TNODE_PAGE
{
    TREE_NODE ptree_nodes[TNODE_PAGE_SIZE];
    TNODE_PAGE *pnext;
};

struct TNODE_BUF
{
    TNODE_PAGE *phead;
    TNODE_PAGE *pcur_page;
    int ncur_pos;
    int ntotal_pages;
};

inline int gntotal_max_cliques;
inline TNODE_BUF gotreenode_buf;

inline TREE_NODE *NewTreeNode()
{
    TREE_NODE *ptnode;
    TNODE_PAGE *pnew_page;

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

inline void OutputOneSet(FILE *fp, int *pset, int nlen)
{
    gntotal_max_cliques++;
    fprintf(fp, "%d ", nlen);
    for (int i = 0; i < nlen; i++)
        fprintf(fp, "%d ", pset[i]);
    fprintf(fp, "\n");
}

inline void DelTNodeBuf()
{
    TNODE_PAGE *ppage = gotreenode_buf.phead;
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

inline void InsertOneSet(int *pset, int nlen, TREE_NODE *&proot)
{
    TREE_NODE *pnode, *pparent, *pleftsib, *pnew_node;
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

inline int BuildTree(const char *szset_filename, TREE_NODE *&proot)
{
    FILE *fp;
    int nlen, *pset, nset_size, i, nmax_len;

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
            delete[] pset;
            nset_size *= 2;
            if (nset_size < nlen)
                nset_size = nlen;
            pset = new int[nset_size];
        }
        for (i = 0; i < nlen; i++)
            fscanf(fp, "%d", &pset[i]);
        qsort(pset, nlen, sizeof(int), comp_int);
        InsertOneSet(pset, nlen, proot);
        fscanf(fp, "%d", &nlen);
    }
    fclose(fp);
    delete[] pset;
    return nmax_len;
}

inline void SearchSubset(int *pset, int nset_len, TREE_NODE *proot, TREE_NODE **pstack, int *ppos)
{
    TREE_NODE *pnode;
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

inline void RmNonMax(TREE_NODE *proot, int nmax_len)
{
    TREE_NODE *pnode, **pstack, **psearch_stack;
    int *pset, ntop, i, *ppos;

    if (proot == NULL || nmax_len <= 0)
        return;

    pset = new int[nmax_len];
    pstack = new TREE_NODE *[nmax_len];
    psearch_stack = new TREE_NODE *[nmax_len];
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
            else
            {
                pnode = pnode->pright_sib;
                pstack[ntop - 1] = pnode;
                pset[ntop - 1] = pnode->nid;
            }
        }
    }

    delete[] pset;
    delete[] pstack;
    delete[] psearch_stack;
    delete[] ppos;
}

inline void OutputMaxSet(TREE_NODE *proot, int nmax_len, const char *szoutput_filename)
{
    FILE *fp;
    TREE_NODE **pstack, *pnode;
    int *pset, ntop;

    fp = fopen(szoutput_filename, "wt");
    if (fp == NULL)
    {
        printf("Error: cannot open file %s for write\n", szoutput_filename);
        return;
    }

    pstack = new TREE_NODE *[nmax_len];
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
            else
            {
                pnode = pnode->pright_sib;
                pstack[ntop - 1] = pnode;
                pset[ntop - 1] = pnode->nid;
            }
        }
    }

    delete[] pstack;
    delete[] pset;
    fclose(fp);
}

inline int RemoveNonMax(const char *szset_filename, const char *szoutput_filename)
{
    cout << ">:REMOVING NON-MAXIMAL CLIQUES" << endl;

    TREE_NODE *proot;
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

#endif
