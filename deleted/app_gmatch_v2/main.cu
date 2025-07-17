#include "global.h"
#include "master.h"
#include "plan.h"

#include "gm_task.h"
#include "gm_cpu_worker.h"
#include "gm_gpu_context.h"

class GMApp : public Master<GMCPUWorker, GMGPUContext>
{
public:
    GMApp(ui argc, char *argv[])
    {
        CommandLine cmd(argc, argv);

        num_cpu_workers = cmd.GetOptionIntValue("-cpu", 32);
        num_gpu_workers = cmd.GetOptionIntValue("-gpu", 1);

        std::string dg = cmd.GetOptionValue("-dg", "");
        int query_type = cmd.GetOptionIntValue("-q", 1000);
        ui eta_ = cmd.GetOptionIntValue("-eta", 1000);

        std::string gpudg = cmd.GetOptionValue("-gpudg", "");
        gpuG=Graph(gpudg);

        // Extract directory path from 'dg'
        size_t pos = dg.find_last_of("/\\");                                           // Handles both UNIX and Windows paths
        std::string dg_path = (pos == std::string::npos) ? "" : dg.substr(0, pos + 1); // include trailing '/'

        // Load graphs
        data_graph.loadGraphFromFile(dg);
        Graph query_graph_gpu("", (PresetPatternType)query_type, GraphType::QUERY);
        // query_graph_cpu.loadGraphFromFile(cpu_query_file);
        // string path="/home/akhlaque.ak@gmail.com/graphs/queries/query_graph_8_2.graph";
        string path = write_to_file(query_graph_gpu);
        // string path="query_cpu.txt";
        query_graph_cpu.loadGraphFromFile(path);

        gpu_preprocess(query_graph_gpu);
        cpu_preprocess();

        // load first-level candidates in data_array
        ui root_vertex = matching_order[0];
        for (ui i = 0; i < candidates_count[root_vertex]; ++i)
        {
            ui v = candidates[root_vertex][i];
            data_array.push_back(new ui(v));
        }

        // for(ui i=0;i<data_graph.getVerticesCount();i++)
        //     data_array.push_back(new ui(i));
    }

    ui get_results()
    {
        ui res = 0;

        while (workers_list.size())
        {
            WorkerT *w = (WorkerT *)workers_list.dequeue();
            GMCPUWorker *cw = dynamic_cast<GMCPUWorker *>(w);
            GPUWorkerT *gw = dynamic_cast<GPUWorkerT *>(w); //

            if (cw)
                res += cw->counter;
            else if (gw)
            {
                // cout<<"gpu found: "<< gw->getContext()->get_results();
                res += gw->getContext()->get_results();
            }
        }
        return res;
    }

    void generateBN(Graph_CPU &query_graph_cpu, ui *order, ui **&bn, ui *&bn_count)
    {
        ui query_vertices_num = query_graph_cpu.getVerticesCount();
        bn_count = new ui[query_vertices_num];
        std::fill(bn_count, bn_count + query_vertices_num, 0);
        bn = new ui *[query_vertices_num];
        for (ui i = 0; i < query_vertices_num; ++i)
        {
            bn[i] = new ui[query_vertices_num];
        }

        std::vector<bool> visited_vertices(query_vertices_num, false);
        visited_vertices[order[0]] = true;
        for (ui i = 1; i < query_vertices_num; ++i)
        {
            ui vertex = order[i];

            ui nbrs_cnt;
            const ui *nbrs = query_graph_cpu.getVertexNeighbors(vertex, nbrs_cnt);
            for (ui j = 0; j < nbrs_cnt; ++j)
            {
                ui nbr = nbrs[j];

                if (visited_vertices[nbr])
                {
                    bn[i][bn_count[i]++] = nbr;
                }
            }
            visited_vertices[vertex] = true;
        }

        cout << "======= BN ========" << endl;
        for (int i = 1; i < query_vertices_num; ++i)
        {
            for (int j = 0; j < bn_count[i]; ++j)
            {
                cout << bn[i][j] << " ";
            }
            cout << endl;
        }
        cout << "==================" << endl;
    }

    void cpu_preprocess()
    {

        std::cout << "CPU preprocess start..." << std::endl;
        //  ============== Step 1 ==============
        FilterVertices::DPisoFilter(data_graph, query_graph_cpu, candidates, candidates_count,
                                    bfs_order, tree);
        FilterVertices::sortCandidates(candidates, candidates_count, query_graph_cpu.getVerticesCount());

        for (ui i = 0; i < query_graph_cpu.getVerticesCount(); ++i)
        {
            max_candidate_cnt = std::max(max_candidate_cnt, candidates_count[i]);
        }

        std::cout << " MAX CANDS : " << max_candidate_cnt << std::endl;

        // ============== Step 2 ==============
        GenerateQueryPlan::generateGQLQueryPlan(data_graph, query_graph_cpu, candidates_count, matching_order, pivot);

        // matching_order = new ui[query_graph_cpu.getVerticesCount()];
        // pivot = new ui[query_graph_cpu.getVerticesCount()];

        for (ui i = 0; i < query_graph_cpu.getVerticesCount(); i++)
        {
            //  plan.matchOrderHost[i] = matching_order[i];
            matching_order[i] = plan.seq_[i];
        }

        std::cout << "======= print matching order ==========" << std::endl;
        for (ui i = 0; i < query_graph_cpu.getVerticesCount(); i++)
        {
            std::cout << matching_order[i] << " ";
        }
        std::cout << std::endl;

        generateBN(query_graph_cpu, matching_order, bn, bn_count);

        edge_matrix = new Edges **[query_graph_cpu.getVerticesCount()];
        for (ui i = 0; i < query_graph_cpu.getVerticesCount(); ++i)
        {
            edge_matrix[i] = new Edges *[query_graph_cpu.getVerticesCount()];
        }

        BuildTable::buildTable(data_graph, query_graph_cpu, candidates, candidates_count, edge_matrix);
    }

    void gpu_preprocess(Graph& query_graph_gpu )
    {
        std::cout << "GPU preprocess start..." << std::endl;
        query_graph_gpu.SetConditions(query_graph_gpu.GetConditions(query_graph_gpu.GetBlissGraph()));

        auto &order = query_graph_gpu.order_;
        std::cout << "conditions: " << std::endl;
        for (ui i = 0; i < order.size(); i++)
        {
            std::cout << i << ": ";
            for (ui j = 0; j < order[i].size(); j++)
                std::cout << GetCondOperatorString(order[i][j].first) << "(" << order[i][j].second << "), ";
            std::cout << std::endl;
        }

        plan.graph = std::move(query_graph_gpu);
        plan.FindRoot();
        plan.GenerateSearchSequence();
        plan.GenerateBackwardNeighbor();
        plan.GeneratePreAfterBackwardNeighbor();
        plan.GenerateUsefulOrder();
        plan.GenerateStoreStrategy();
        gmp.AddGMContext(plan.seq_, plan.reverse_seq_, plan.backNeighborCountHost,
                                    plan.backNeighborsHost, plan.parentHost, plan.vertex_count_,
                                    plan.condOrderHost, plan.condNumHost, plan.share_intersection,
                                    plan.preBackNeighborCountHost, plan.preBackNeighborsHost, plan.preCondOrderHost,
                                    plan.preCondNumHost, plan.afterBackNeighborCountHost, plan.afterBackNeighborsHost,
                                    plan.afterCondOrderHost, plan.afterCondNumHost, plan.strategy, plan.moving_lvl);
    }

    string write_to_file(auto &query)
    {
        // Convert id to string using std::to_string
        string path="query_cpu.txt";
        std::ofstream outfile(path);

        if (!outfile)
        {
            std::cerr << "Failed to open file for writing.\n";
            return path; // remove 'return 1;' — it's a void function
        }
        ui edges = 0;
        for (ui i = 0; i < query.GetVertexCount(); i++)
        {
            ui j = query.GetRowPtrs()[i];
            ui en = query.GetRowPtrs()[i + 1];
            for (; j < en; j++)
            {
                if (i < query.GetCols()[j])
                    edges++;
            }
        }

        // Header
        outfile << "t " << query.GetVertexCount() << " " << edges << std::endl;

        for (ui i = 0; i < query.GetVertexCount(); i++)
        {
            ui degree = query.GetRowPtrs()[i + 1] - query.GetRowPtrs()[i];
            outfile << "v " << i << " 0 " << degree << std::endl; // use outfile, not cout
        }

        for (ui i = 0; i < query.GetVertexCount(); i++)
        {
            ui j = query.GetRowPtrs()[i];
            ui en = query.GetRowPtrs()[i + 1];
            for (; j < en; j++)
            {
                if (i < query.GetCols()[j])
                    outfile << "e " << i << " " << query.GetCols()[j] << std::endl; // use outfile
            }
        }
        return path;
    }

};

int main(int argc, char *argv[])
{

    GMApp app(argc, argv);
    Timer t;
    app.run();
    cout << "Total time (s): " << t.elapsed() / 1e6 << endl;
    cout << "Total count: " << app.get_results() << endl;
    return 0;
}
