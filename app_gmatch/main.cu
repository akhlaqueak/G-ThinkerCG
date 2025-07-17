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
        tasks_per_fetch_gpu_worker_g = cmd.GetOptionIntValue("-gpuchunk", 500000);
        tasks_per_fetch_g = cmd.GetOptionIntValue("-cpuchunk", 50);
        ui eta_ = cmd.GetOptionIntValue("-eta", 1000);
        std::string dg = cmd.GetOptionValue("-dg", "");
        int query_type = cmd.GetOptionIntValue("-q", 0);
        plan_strategy = cmd.GetOptionValue("-s", "hybrid");

        cout << " ======= Parameters ========" << endl;
        cout << "Data Graph: " << dg << endl;
        cout << "Query Graph: " << query_type << endl;
        cout << "cpu workers: " << num_cpu_workers << endl;
        cout << "gpu workers: " << num_gpu_workers << endl;
        cout << "eta: " << eta_ << endl;
        cout << "cpu chunk: " << tasks_per_fetch_g << endl;
        cout << "gpu chunk: " << tasks_per_fetch_gpu_worker_g << endl;
        cout << "plan strategy: " << plan_strategy << endl;
        cout << " ======= ********** ========" << endl;

        gpu_dg = Graph(dg);
        gpu_qg = Graph("", (PresetPatternType)query_type, GraphType::QUERY);

        cpu_dg.loadGraphFromCSR(gpu_dg.GetRowPtrs(), gpu_dg.GetCols(), gpu_dg.GetVertexCount());
        cpu_qg.loadGraphFromCSR(gpu_qg.GetRowPtrs(), gpu_qg.GetCols(), gpu_qg.GetVertexCount());

        gpu_preprocess();
        cpu_preprocess();
        
        // load first-level candidates in data_array
        ui root_vertex = matching_order[0];
        for (ui i = 0; i < candidates_count[root_vertex]; ++i)
        {
            ui v = candidates[root_vertex][i];
            data_array.push_back(new ui(v));
        }
        // for (ui i = 0; i < gpu_dg.GetVertexCount(); ++i)
        //     data_array.push_back(new ui(i));
    }

    ull get_results()
    {
        ull res = 0;

        while (workers_list.size())
        {
            WorkerT *w = (WorkerT *)workers_list.dequeue();
            GMCPUWorker *cw = dynamic_cast<GMCPUWorker *>(w);
            GPUWorkerT *gw = dynamic_cast<GPUWorkerT *>(w); //

            if (cw)
                res += cw->counter;
            else if (gw)
                // cout<<"gpu found: "<< gw->getContext()->get_results();
                res += gw->getContext()->get_results();
        }
        return res;
    }

    void generateBN(Graph_CPU &cpu_qg, ui *order, ui **&bn, ui *&bn_count)
    {
        ui query_vertices_num = cpu_qg.getVerticesCount();
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
            const ui *nbrs = cpu_qg.getVertexNeighbors(vertex, nbrs_cnt);
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
        FilterVertices::DPisoFilter(cpu_dg, cpu_qg, candidates, candidates_count,
                                    bfs_order, tree);
        FilterVertices::sortCandidates(candidates, candidates_count, cpu_qg.getVerticesCount());

        for (ui i = 0; i < cpu_qg.getVerticesCount(); ++i)
        {
            max_candidate_cnt = std::max(max_candidate_cnt, candidates_count[i]);
        }

        std::cout << " MAX CANDS : " << max_candidate_cnt << std::endl;

        // ============== Step 2 ==============
        GenerateQueryPlan::generateGQLQueryPlan(cpu_dg, cpu_qg, candidates_count, matching_order, pivot);

        for (ui i = 0; i < cpu_qg.getVerticesCount(); i++)
        {
            //  plan.matchOrderHost[i] = matching_order[i];
            matching_order[i] = plan.matchOrderHost[i];
        }
        std::cout << " ROOT CANDS : " << candidates_count[matching_order[0]] << std::endl;

        std::cout << "======= print matching order ==========" << std::endl;
        for (ui i = 0; i < cpu_qg.getVerticesCount(); i++)
        {
            std::cout << matching_order[i] << " ";
        }
        std::cout << std::endl;

        generateBN(cpu_qg, matching_order, bn, bn_count);

        edge_matrix = new Edges **[cpu_qg.getVerticesCount()];
        for (ui i = 0; i < cpu_qg.getVerticesCount(); ++i)
        {
            edge_matrix[i] = new Edges *[cpu_qg.getVerticesCount()];
        }

        BuildTable::buildTable(cpu_dg, cpu_qg, candidates, candidates_count, edge_matrix);
    }

    void gpu_preprocess()
    {
        std::cout << "GPU preprocess start..." << std::endl;
        gpu_qg.SetConditions(gpu_qg.GetConditions(gpu_qg.GetBlissGraph()));

        auto &order = gpu_qg.order_;
        std::cout << "conditions: " << std::endl;
        for (ui i = 0; i < order.size(); i++)
        {
            std::cout << i << ": ";
            for (ui j = 0; j < order[i].size(); j++)
                std::cout << GetCondOperatorString(order[i][j].first) << "(" << order[i][j].second << "), ";
            std::cout << std::endl;
        }

        plan.graph = std::move(gpu_qg);
        plan.FindRoot();
        plan.GenerateSearchSequence();
        plan.GenerateBackwardNeighbor();
        plan.GeneratePreAfterBackwardNeighbor();
        plan.GenerateUsefulOrder();
        plan.GenerateStoreStrategy();
    }

    string write_to_file(auto &query)
    {
        // Convert id to string using std::to_string
        string path = "query_cpu.txt";
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
