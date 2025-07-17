#ifndef SYSTEM_WORKER_H
#define SYSTEM_WORKER_H

#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <thread>
#include <string>

#include "common/command_line.h"
#include "view/view_bin_manager.h"
#include "view/view_bin_buffer.h"
#include "rigtorp/MPMCQueue.h"
#include "system/pipeline_executor.h"
#include "system/appbase.h"
#include "system/subgraph_container.h"
#include "system/buffer.h"
#include "system/plan.h"

//1158240 = 193040*6 (now 193040)
template<typename Application>
class Worker
{
public:
    void run(int argc, char *argv[], Application app)
    {
        // deviceQuery();
        CommandLine cmd(argc, argv);
        std::string data_filename = cmd.GetOptionValue("-dg", "./data/com-friendster.ungraph.txt.bin");
        int query_type = cmd.GetOptionIntValue("-q", 0);
        double mem = cmd.GetOptionDoubleValue("-m", 10);
        int thread_num = cmd.GetOptionIntValue("-t", 1);
        int queue_size = cmd.GetOptionIntValue("-qs", 1);
        int hop = cmd.GetOptionIntValue("-h", -1); // -1 means worker needs to analyse hop by itself.
        int producers_num = cmd.GetOptionIntValue("-pn", 1);
        int consumers_num = cmd.GetOptionIntValue("-cn", 1);
        int do_reorder = cmd.GetOptionIntValue("-dr", 0);
        int do_split = cmd.GetOptionIntValue("-ds", 0);
        int do_split_times = cmd.GetOptionIntValue("-dst", 1);
        int sort_sources = cmd.GetOptionIntValue("-ss", 0);

        int gpu_count = 0;
        cudaGetDeviceCount(&gpu_count);
        assert(consumers_num <= gpu_count);
        std::cout << "GPU count: " << gpu_count << std::endl;

        assert(queue_size >= consumers_num);
        std::cout << "m: " << mem << " t: " << thread_num << " qs: " << queue_size << " pn: " << producers_num << " cn: " << consumers_num << std::endl;

        Graph *graph = new Graph(data_filename);
        int root_degree = 0;
        std::vector<StoreStrategy> strategy;
        Plan plan;


        Timer timer;
        timer.StartTimer();

        if (query_type != 1000)
        {
            assert(hop == -1);
            Graph query_G("", (PresetPatternType)query_type, GraphType::QUERY);
            query_G.SetConditions(query_G.GetConditions(query_G.GetBlissGraph()));

            auto& order = query_G.order_;
            std::cout << "conditions: " << std::endl;
            for (ui i = 0; i < order.size(); i++) 
            {
                std::cout << i << ": ";
                for (ui j = 0; j < order[i].size(); j++)
                    std::cout << GetCondOperatorString(order[i][j].first) << "(" << order[i][j].second << "), ";
                std::cout << std::endl;
            }

            plan.graph = std::move(query_G);

            // TODO: FIXME: compute hop
            // hop = 2; // test pattern specific for a triangle pattern

            plan.FindRoot();
            plan.GenerateSearchSequence();
            plan.GenerateBackwardNeighbor();
            plan.GeneratePreAfterBackwardNeighbor();
            plan.GenerateUsefulOrder();
            plan.GenerateStoreStrategy();
            hop = plan.GetHop();
            root_degree = plan.root_degree_;
            strategy = plan.strategy;

            std::cout << "Hop = " << hop << std::endl;
            std::cout << "Root degree = " << root_degree << std::endl;
            std::cout << "Match Order: ";
            for (size_t i = 0; i < plan.vertex_count_; ++i)
            {
                std::cout << plan.seq_[i] << " ";
            }
            std::cout << std::endl;
            std::cout << "Moving Level: ";
            for (size_t i = 0; i < plan.vertex_count_; ++i)
            {
                std::cout << plan.moving_lvl[i] << " ";
            }
            std::cout << std::endl;

            for (size_t i = 0; i < plan.vertex_count_; ++i)
            {
                std::cout << i << ": ";
                for (size_t j = 0; j < plan.backNeighborCountHost[plan.reverse_seq_[i]]; ++j)
                {
                    std::cout << plan.backNeighborsHost[plan.reverse_seq_[i] * plan.vertex_count_ + j] << " ";
                }
                std::cout << std::endl;
            }

            for (size_t i = 0; i < plan.vertex_count_; ++i)
            {
                if (plan.share_intersection[i])
                {
                    std::cout << "shared loc = " << i << "       ";
                    std::cout << "pre BN: ";
                    for (size_t j = 0; j < plan.preBackNeighborCountHost[i]; ++j)
                    {
                        std::cout << plan.preBackNeighborsHost[i * plan.vertex_count_ + j] << " ";
                    }
                    std::cout << "after BN: ";
                    for (size_t j = 0; j < plan.afterBackNeighborCountHost[i]; ++j)
                    {
                        std::cout << plan.afterBackNeighborsHost[i * plan.vertex_count_ + j] << " ";
                    }
                    std::cout << std::endl;
                }
            }

            for (size_t i = 0; i < plan.vertex_count_; ++i)
            {
                if (plan.share_intersection[plan.seq_[i]])
                {
                    std::cout << "shared VertexID = " << i << "       ";
                    std::cout << "pre Cond: ";
                    for (size_t j = 0; j < plan.preCondNumHost[i]; ++j)
                    {
                        std::cout << plan.preCondOrderHost[2 * i * plan.vertex_count_ + j] 
                            << " " << plan.preCondOrderHost[2 * i * plan.vertex_count_ + j + 1] << " ";
                    }
                    std::cout << "after Cond: ";
                    for (size_t j = 0; j < plan.afterCondNumHost[i]; ++j)
                    {
                        std::cout << plan.afterCondOrderHost[2 * i * plan.vertex_count_ + j] 
                            << " " << plan.afterCondOrderHost[2 * i * plan.vertex_count_ + j + 1] << " ";
                    }
                    std::cout << std::endl;
                }
            }

            app.context.AddGMContext(plan.seq_, plan.reverse_seq_, plan.backNeighborCountHost,
                                    plan.backNeighborsHost, plan.parentHost, plan.vertex_count_,
                                    plan.condOrderHost, plan.condNumHost, plan.share_intersection,
                                    plan.preBackNeighborCountHost, plan.preBackNeighborsHost, plan.preCondOrderHost,
                                    plan.preCondNumHost, plan.afterBackNeighborCountHost, plan.afterBackNeighborsHost,
                                    plan.afterCondOrderHost, plan.afterCondNumHost, plan.strategy, plan.moving_lvl);
        }
        // PipelineExecutor<Application>* executor = new PipelineExecutor<Application>(c, graph, max_partitioned_sources_num, app, max_view_bin_size, strategy);
        PipelineExecutor<Application>* executor = new PipelineExecutor<Application>(graph, app, plan.strategy);
        executor->Run();

        timer.EndTimer();
        timer.PrintElapsedMicroSeconds("Total time");

    }
};

#endif