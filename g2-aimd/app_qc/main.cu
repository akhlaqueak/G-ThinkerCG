#include <cctype>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <locale>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <vector>

#include "common/command_line.h"
#include "common/timer.h"
#include "common/gpu_env.h"
#include "system/work_context.h"
#include "qc_support.h"
#include "qc_gpu_context.h"

CPU_Data hd;
GPU_Data dd;
CPU_Cliques hc;
CPU_Graph *hg = nullptr;

template <class Application>
__global__ void qc_generate_subgraphs(Application app, unsigned int base)
{
    app.generateSubgraphs(base);
}

template <class Application>
__global__ void qc_process(Application app)
{
    app.processSubgraphs();
}

template <class Application>
__global__ void qc_expand(Application app)
{
    app.expand();
}

template <class Application>
__global__ void qc_load_from_host(Application app)
{
    app.loadFromHost();
}

bool apply_remove_nonmax(const CommandLine &cmd)
{
    for (int i = 1; i < cmd.argc; i++)
    {
        std::string arg = cmd.argv[i];
        if (arg == "-rmnonmax" || arg == "--rmnonmax")
        {
            if (i + 1 >= cmd.argc)
                return true;

            std::string value = cmd.argv[i + 1];
            if (!value.empty() && value[0] == '-')
                return true;

            return !(value == "0" || value == "false" || value == "False" || value == "off");
        }
    }
    return false;
}

std::string quote_command_arg(const std::string &arg)
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

std::string supplied_command(const CommandLine &cmd)
{
    std::string command;
    for (int i = 0; i < cmd.argc; i++)
    {
        if (i > 0)
            command += " ";
        command += quote_command_arg(cmd.argv[i]);
    }
    return command;
}

std::string executable_modified_time(const CommandLine &cmd)
{
    if (cmd.argc <= 0 || cmd.argv[0] == nullptr)
        return "unavailable";

    struct stat st;
    if (stat(cmd.argv[0], &st) != 0)
        return "unavailable";

    char buffer[64];
    std::tm tm_value;
    if (localtime_r(&st.st_mtime, &tm_value) == nullptr)
        return "unavailable";

    if (std::strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S %Z", &tm_value) == 0)
        return "unavailable";

    return buffer;
}

void print_help(const char *program)
{
    cout << "Usage: " << program << " -f <graph.sbin> [options]" << endl;
    cout << endl;
    cout << "Required:" << endl;
    cout << "  -f <path>           Input expanded binary graph (.sbin)." << endl;
    cout << endl;
    cout << "QC parameters:" << endl;
    cout << "  -g <gamma>          Minimum degree ratio in [0.5, 1]. Default: 0.5" << endl;
    cout << "  -k <size>           Minimum quasi-clique size (> 1). Default: 10" << endl;
    cout << "  -o <file>           Output file for maximal quasi-cliques. Default: output.txt" << endl;
    cout << "  -rmnonmax [0|1]     Remove non-maximal results. Default: off" << endl;
    cout << "  -sched <0|1>        Scheduling mode: 0=dynamic, 1=static. Default: 0" << endl;
    cout << endl;
    cout << "Other:" << endl;
    cout << "  -h, --help          Show this help message and exit." << endl;
}

bool wants_help(const CommandLine &cmd)
{
    for (int i = 1; i < cmd.argc; i++)
    {
        const std::string arg = cmd.argv[i];
        if (arg == "-h" || arg == "--help")
            return true;
    }
    return false;
}

void initialize_tasks(CPU_Graph &graph, CPU_Data &host_data)
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

void allocate_runtime_memory(CPU_Data &host_data, GPU_Data &device_data, CPU_Cliques &host_cliques, CPU_Graph &graph)
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

void setup_work_context(QCApp &app, CPU_Graph &graph)
{
    chkerr(cudaMallocManaged(&(app.ctx), sizeof(WorkContext)));
    app.ctx->context = nullptr;

    chkerr(cudaMalloc(&((app.ctx)->d_row_ptrs), sizeof(uintE) * (graph.number_of_vertices + 1)));
    chkerr(cudaMalloc(&((app.ctx)->d_cols), sizeof(uintV) * graph.number_of_edges));
    chkerr(cudaMemcpy((app.ctx)->d_row_ptrs, graph.onehop_offsets, sizeof(uintE) * (graph.number_of_vertices + 1), cudaMemcpyHostToDevice));
    chkerr(cudaMemcpy((app.ctx)->d_cols, graph.onehop_neighbors, sizeof(uintV) * graph.number_of_edges, cudaMemcpyHostToDevice));

    chkerr(cudaMalloc((void **)&((app.ctx)->sources), hd.initial_vertices_count * sizeof(uintV)));
    std::vector<uintV> roots(hd.initial_vertices_count);
    for (size_t i = 0; i < hd.initial_vertices_count; ++i)
        roots[i] = static_cast<uintV>(i);
    chkerr(cudaMemcpy((app.ctx)->sources, roots.data(), hd.initial_vertices_count * sizeof(uintV), cudaMemcpyHostToDevice));

    chkerr(cudaMallocManaged((void **)&((app.ctx)->sources_num), sizeof(size_t)));
    (app.ctx)->sources_num[0] = hd.initial_vertices_count;

    chkerr(cudaMallocManaged(&((app.ctx)->level), sizeof(ui)));
    (app.ctx)->level[0] = 0;
}

void cleanup_work_context(QCApp &app)
{
    if (app.ctx == nullptr)
        return;
    if (app.ctx->d_row_ptrs)
        chkerr(cudaFree(app.ctx->d_row_ptrs));
    if (app.ctx->d_cols)
        chkerr(cudaFree(app.ctx->d_cols));
    if (app.ctx->sources)
        chkerr(cudaFree(app.ctx->sources));
    if (app.ctx->sources_num)
        chkerr(cudaFree(app.ctx->sources_num));
    if (app.ctx->level)
        chkerr(cudaFree(app.ctx->level));
    chkerr(cudaFree(app.ctx));
    app.ctx = nullptr;
}

void run_qc_gpu(QCApp &app)
{
    app.sg->chunk[0] = MAXCHUNK;

    for (unsigned int i = 0; i < app.ctx->sources_num[0];)
    {
        app.init_chunk();
        app.ctx->level[0] = 1;
        qc_generate_subgraphs<<<NUM_OF_BLOCKS, BLOCK_SIZE>>>(app, i);
        i += app.sg->chunk[0];

        while (true)
        {
            chkerr(cudaDeviceSynchronize());
            app.sg->swapBuffers();

            if (app.sg->isEmpty())
            {
                if (app.sgHost->isEmpty())
                {
                    app.sgHost->swapBuffers();
                    if (app.sgHost->isEmpty())
                        break;
                }

                qc_load_from_host<<<NUM_OF_BLOCKS, BLOCK_SIZE>>>(app);
                chkerr(cudaDeviceSynchronize());
                continue;
            }

            qc_process<<<NUM_OF_BLOCKS, BLOCK_SIZE>>>(app);
            qc_expand<<<NUM_OF_BLOCKS, BLOCK_SIZE>>>(app);
            chkerr(cudaDeviceSynchronize());
            app.init_level();

            if (app.sg->isOverflow())
            {
                app.iterationFailed();
                i -= app.sg->chunk[0];
                break;
            }
            if (app.sgHost->isOverflowToHost())
                throw std::runtime_error("Host overflow occurred");

            app.ctx->level[0]++;
        }

        app.sg->adjustChunk();
        app.iterationSuccess();
    }

    app.completion();
}

int main(int argc, char *argv[])
{
    CommandLine cmd(argc, argv);

    if (wants_help(cmd))
    {
        print_help(argv[0]);
        return 0;
    }

    std::string graph_file = cmd.GetOptionValue("-f", "");
    if (graph_file.empty())
    {
        print_help(argv[0]);
        return 1;
    }

    minimum_degree_ratio = cmd.GetOptionDoubleValue("-g", 0.5);
    minimum_clique_size = cmd.GetOptionIntValue("-k", 10);
    std::string out_file = cmd.GetOptionValue("-o", "output.txt");
    scheduling_toggle = cmd.GetOptionIntValue("-sched", 0);
    bool remove_nonmax = apply_remove_nonmax(cmd);
    store_cliques = remove_nonmax;

    if (minimum_degree_ratio < .5 || minimum_degree_ratio > 1)
    {
        cout << "minimum degree ratio must be between .5 and 1 inclusive" << endl;
        minimum_degree_ratio = 0.5;
    }
    if (minimum_clique_size <= 1)
    {
        cout << "minimum size must be greater than 1" << endl;
        minimum_clique_size = 10;
    }
    if (!(scheduling_toggle == 0 || scheduling_toggle == 1))
    {
        cout << "scheduling toggle must be 0 or 1" << endl;
        scheduling_toggle = 0;
    }

    int gpu_count = 0;
    if (cudaGetDeviceCount(&gpu_count) != cudaSuccess || gpu_count <= 0)
    {
        cerr << "No CUDA device available." << endl;
        return 1;
    }
    chkerr(cudaSetDevice(0));

    std::cout.imbue(std::locale());
    cout << "Command: " << supplied_command(cmd) << endl;
    cout << "Executable compiled time: " << executable_modified_time(cmd) << endl;
    cout << " ======= Parameters ========" << endl;
    cout << "Graph: " << graph_file << endl;
    cout << "Gamma: " << minimum_degree_ratio << endl;
    cout << "Min size: " << minimum_clique_size << endl;
    cout << "Output: " << out_file << endl;
    cout << "Remove non-maximal: " << (remove_nonmax ? "true" : "false") << endl;
    cout << "Scheduling: " << (scheduling_toggle == 0 ? "dynamic" : "static") << endl;
    cout << " ======= ********** ========" << endl;

    QCApp app;
    Timer total_timer;
    Timer search_timer;
    total_timer.StartTimer();
    std::ofstream temp_results;
    std::string temp_filename = "t_cliques.txt";

    try
    {
        if (remove_nonmax)
            temp_results.open(temp_filename);

        ifstream graph_stream(graph_file, ios::in | ios::binary);
        if (!graph_stream.is_open())
            throw std::runtime_error("invalid graph file");

        load_output_vertex_id_map(graph_file);

        cout << ">:PRE-PROCESSING" << endl;
        auto preprocessing_start = chrono::high_resolution_clock::now();
        hg = new CPU_Graph(graph_stream);
        CPU_Graph &graph = *hg;
        cout << "|V| = " << graph.number_of_vertices << endl;
        cout << "|E| = " << graph.number_of_edges << endl;
        cout << "|2-hop| = " << graph.number_of_lvl2adj << endl;
        graph_stream.close();

        calculate_minimum_degrees(graph);
        allocate_runtime_memory(hd, dd, hc, graph);
        initialize_tasks(graph, hd);
        chkerr(cudaMalloc((void **)&dd.initial_vertices, sizeof(Vertex) * hd.initial_vertices_count));
        chkerr(cudaMemcpy(dd.initial_vertices, hd.initial_vertices, sizeof(Vertex) * hd.initial_vertices_count, cudaMemcpyHostToDevice));
        chkerr(cudaMalloc((void **)&dd.initial_vertices_count, sizeof(uint64_t)));
        uint64_t initial_vertices_count_u64 = hd.initial_vertices_count;
        chkerr(cudaMemcpy(dd.initial_vertices_count, &initial_vertices_count_u64, sizeof(uint64_t), cudaMemcpyHostToDevice));
        chkerr(cudaMalloc((void **)&dd.initial_order_map, sizeof(int) * graph.number_of_vertices));
        chkerr(cudaMemcpy(dd.initial_order_map, hd.initial_order_map, sizeof(int) * graph.number_of_vertices, cudaMemcpyHostToDevice));

        auto preprocessing_end = chrono::high_resolution_clock::now();
        auto preprocessing_ms = chrono::duration_cast<chrono::milliseconds>(preprocessing_end - preprocessing_start);
        cout << "No. of candidates: " << hd.initial_vertices_count << endl;
        cout << "--->:LOADING TIME: " << preprocessing_ms.count() << " ms" << endl;

        setup_work_context(app, graph);
        app.allocateMemory();
        app.initialize(0);
        search_timer.StartTimer();
        run_qc_gpu(app);
        chkerr(cudaDeviceSynchronize());

        uint64_t max_clique_size = 0;
        chkerr(cudaMemcpy(&max_clique_size, dd.max_clique_size, sizeof(uint64_t), cudaMemcpyDeviceToHost));

        uint64_t pre_max_quasi_cliques = dump_cliques(hc, dd, temp_results, true);
        if (remove_nonmax)
        {
            temp_results.flush();
            temp_results.close();
        }
        search_timer.EndTimer();

        cout << "Search only time (s): " << search_timer.GetElapsedMicroSeconds() / 1e6 << endl;
        cout << ">:NUMBER OF QUASI-CLIQUES BEFORE MAX CHECK: " << pre_max_quasi_cliques << endl;

        auto start1 = chrono::high_resolution_clock::now();
        uint64_t output_quasi_cliques = 0;
        if (remove_nonmax)
        {
            std::ifstream clique_input(temp_filename);
            if (clique_input.peek() != std::ifstream::traits_type::eof())
            {
                output_quasi_cliques = static_cast<uint64_t>(RemoveNonMax(temp_filename.c_str(), out_file.c_str()));
            }
            else
            {
                std::ofstream empty_output(out_file);
                cout << ">:NUMBER OF FINAL MAXIMAL QUASI-CLIQUES: 0" << endl;
            }
        }
        else
        {
            cout << ">:NUMBER OF FINAL MAXIMAL QUASI-CLIQUES: NA" << endl;
        }
        auto stop1 = chrono::high_resolution_clock::now();
        auto duration1 = chrono::duration_cast<chrono::milliseconds>(stop1 - start1);
        if (remove_nonmax)
            cout << "--->:REMOVE NON-MAX TIME: " << duration1.count() << " ms" << endl;
        else
            cout << "--->:COUNT ONLY POSTPROCESS TIME: " << duration1.count() << " ms" << endl;

        total_timer.EndTimer();
        cout << "Total time (s): " << total_timer.GetElapsedMicroSeconds() / 1e6 << endl;
        cout << "Total count before maximality check: " << pre_max_quasi_cliques << endl;
        cout << "Largest clique size: " << max_clique_size << endl;
        if (remove_nonmax)
            cout << "Total maximal quasi-cliques: " << output_quasi_cliques << endl;
        else
            cout << "Total maximal quasi-cliques: NA" << endl;
    }
    catch (const std::exception &error)
    {
        if (temp_results.is_open())
            temp_results.close();
        if (hg != nullptr)
        {
            delete hg;
            hg = nullptr;
        }
        cleanup_work_context(app);
        free_memory(hd, dd, hc, true);
        cerr << error.what() << endl << endl;
        print_help(argv[0]);
        return 1;
    }

    if (hg != nullptr)
    {
        delete hg;
        hg = nullptr;
    }
    cleanup_work_context(app);
    free_memory(hd, dd, hc, true);
    chkerr(cudaDeviceReset());
    return 0;
}
