#include <algorithm>
#include <chrono>
#include <cstring>
#include <cctype>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <cuda_runtime.h>

#include "utils/CLI11.hpp"
#include "utils/config.h"
#include "utils/globals.h"
#include "utils/cuda_helpers.h"
#include "utils/mem_pool.h"

#include "graph/graph.h"
#include "graph/gmatch_compat.h"
#include "structures/hashed_tries.h"
#include "structures/hashed_trie_manager.h"
#include "processing/plan.h"
#include "execution/execution.h"

static bool is_unsigned_integer(const std::string& value)
{
    return !value.empty() && std::all_of(value.begin(), value.end(), [](unsigned char c) {
        return std::isdigit(c);
    });
}

static std::string format_count(unsigned long long value)
{
    std::string digits = std::to_string(value);
    std::string formatted;
    formatted.reserve(digits.size() + digits.size() / 3);
    const size_t first_group = digits.size() % 3;
    for (size_t i = 0; i < digits.size(); ++i)
    {
        if (i > 0 && (i - first_group) % 3 == 0)
            formatted.push_back(',');
        formatted.push_back(digits[i]);
    }
    return formatted;
}

class NullBuffer : public std::streambuf
{
public:
    int overflow(int c) override { return c; }
};

int main(int argc, char** argv) {
    std::vector<std::string> normalized_args;
    normalized_args.reserve(argc);
    std::vector<char*> normalized_argv;
    normalized_argv.reserve(argc);
    for (int i = 0; i < argc; ++i)
    {
        normalized_args.emplace_back(std::string(argv[i]) == "-dg" ? "-d" : argv[i]);
        normalized_argv.push_back(normalized_args.back().data());
    }
    int normalized_argc = static_cast<int>(normalized_argv.size());

    CLI::App app{"App description"};

    std::string query_path, data_path, method = "BFS-DFS";
    bool filtering_3rd = true, adaptive_ordering = true, load_balancing = true;
    bool verbose = false;
    uint32_t filtering_order_start_v = UINT32_MAX;
    uint32_t gpu_num = 0u;
    app.add_option("-q", query_path, "query graph path or app_gmatch query id")->required();
    app.add_option("-d,--dg", data_path, "data graph path")->required();
    app.add_option("-m", method, "enumeration method");
    app.add_option("--f3", filtering_3rd, "enable the third filtering step or not");
    app.add_option("--f3start", filtering_order_start_v, "start vertex of the third filtering step");
    app.add_option("--ao", adaptive_ordering, "enable adaptive ordering or not");
    app.add_option("--lb", load_balancing, "enable load balancing or not");
    app.add_option("--gpu", gpu_num, "gpu number");
    app.add_flag("--verbose", verbose, "print EGSM stage diagnostics");

    CLI11_PARSE(app, normalized_argc, normalized_argv.data());

    cudaErrorCheck(cudaSetDevice(gpu_num));
    copyConfig(adaptive_ordering, load_balancing);

    const auto total_start = std::chrono::steady_clock::now();
    NullBuffer null_buffer;
    std::streambuf* original_cout = std::cout.rdbuf();
    if (!verbose)
        std::cout.rdbuf(&null_buffer);

    /*************** read graph ***************/
    std::unordered_map<uint32_t, uint32_t> label_map;

    std::unique_ptr<Graph> query_graph_holder;
    uint64_t app_gmatch_automorphisms = 1;
    if (is_unsigned_integer(query_path))
    {
        const uint32_t query_id = static_cast<uint32_t>(std::stoul(query_path));
        app_gmatch_automorphisms = gmatch_compat::automorphism_count_for_query(static_cast<int>(query_id));
        query_graph_holder = std::make_unique<Graph>(query_id, label_map);
    }
    else
    {
        query_graph_holder = std::make_unique<Graph>(query_path, label_map);
    }

    Graph& query_graph = *query_graph_holder;
    Graph data_graph(data_path, label_map);
    GraphGPU query_graph_gpu(query_graph);
    GraphGPU data_graph_gpu(data_graph);
    GraphUtils query_utils;
    query_utils.Set(query_graph);

    copyGraphMeta(query_graph, data_graph, query_utils);

    /*************** filtering ***************/
    HashedTries hashed_tries {};

    TIME_INIT();
    TIME_START();
    HashedTrieManager manager(query_graph, query_graph_gpu, data_graph_gpu, hashed_tries);
    TIME_END();
    PRINT_LOCAL_TIME("Build Cuckoo Tries");

    copyTries(hashed_tries);

    TIME_START();
    manager.Filter(hashed_tries, filtering_3rd, filtering_order_start_v);
    TIME_END();
    PRINT_LOCAL_TIME("Filtering");

    manager.GetCardinalities(hashed_tries);
    manager.Print();

    manager.Deallocate();
    query_graph_gpu.Deallocate();
    data_graph_gpu.Deallocate();

    Plan plan(query_graph, manager.h_compacted_vs_sizes_, method);

    plan.Print(query_graph);

    /*************** memory pool ***************/
    std::cout << '\n';
    MEM_INIT();
    PRINT_MEM_INFO("Before Allocation");

    MemPool pool {};
    pool.Alloc(MAX_RES_MEM_SPACE / sizeof(uint32_t));
    PoolElem res = pool.TryMax();
    unsigned long long int res_size = 0;

    PRINT_MEM_INFO("After allocation");
    std::cout << std::endl;;

    /*************** enumeration ***************/
    matchDFSGroup(manager, plan, pool, res, res_size);

    pool.Free();
    manager.DeallocateTries(hashed_tries);

    if (app_gmatch_automorphisms > 1)
        res_size /= app_gmatch_automorphisms;

    const auto total_end = std::chrono::steady_clock::now();
    const std::chrono::duration<double> total_time = total_end - total_start;
    if (!verbose)
        std::cout.rdbuf(original_cout);

    std::cout << "Total time (s): " << std::fixed << std::setprecision(6) << total_time.count() << std::endl;
    std::cout << "Total count: " << format_count(res_size) << std::endl;
}
