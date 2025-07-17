#include "global.h"
#include "master.h"
#include "plan.h"

Graph data_graph;
Plan plan;

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

        std::string fp = cmd.GetOptionValue("-dg", "./data/com-friendster.ungraph.txt.bin");
        int query_type = cmd.GetOptionIntValue("-q", 1000);
        ui eta_ = cmd.GetOptionIntValue("-eta", 1000);

        std::string query_graph_path = cmd.GetOptionValue("-querygraph", "./data/query_graph");
        std::string data_graph_path  = cmd.GetOptionValue("-datagraph", "./data/data_graph");

        data_graph = Graph(fp);

        Graph query_G("", (PresetPatternType)query_type, GraphType::QUERY);
        query_G.SetConditions(query_G.GetConditions(query_G.GetBlissGraph()));

        auto &order = query_G.order_;
        std::cout << "conditions: " << std::endl;
        for (ui i = 0; i < order.size(); i++)
        {
            std::cout << i << ": ";
            for (ui j = 0; j < order[i].size(); j++)
                std::cout << GetCondOperatorString(order[i][j].first) << "(" << order[i][j].second << "), ";
            std::cout << std::endl;
        }

        plan.graph = std::move(query_G);
        plan.FindRoot();
        plan.GenerateSearchSequence();
        plan.GenerateBackwardNeighbor();
        plan.GeneratePreAfterBackwardNeighbor();
        plan.GenerateUsefulOrder();
        plan.GenerateStoreStrategy();


        std::cout << "CPU preprocess start..." << std::endl;

        cpu_preprocess(data_graph_path, query_graph_path);

        // for (int i = 0; i < data_graph.GetVertexCount(); ++i)
        //     data_array.push_back(new ui(i)); // data_array is member of Master

        // load first-level candidates in data_array
        ui root_vertex =  matching_order[0];
        for (ui i = 0; i < candidates_count[root_vertex]; ++i) {
            ui v = candidates[root_vertex][i];
            data_array.push_back(new ui(v));
        }
    }

    ui get_results()
    {
        ui res = 0;

        while (workers_list.size())
        {
            WorkerT *w = (WorkerT *)workers_list.dequeue();
            GMCPUWorker *cw = dynamic_cast<GMCPUWorker *>(w);
            // GPUWorkerT *gw = dynamic_cast<GPUWorkerT *>(w); //

            // if (cw)
            res += cw->counter;
            // else if (gw)
            // {
            //     res += gw->getContext()->get_results();
            // }
        }
        return res;
    }

    void generateBN(Graph_CPU &query_graph, ui *order, ui **&bn, ui *&bn_count) 
    {
        ui query_vertices_num = query_graph.getVerticesCount();
        bn_count = new ui[query_vertices_num];
        std::fill(bn_count, bn_count + query_vertices_num, 0);
        bn = new ui *[query_vertices_num];
        for (ui i = 0; i < query_vertices_num; ++i) {
            bn[i] = new ui[query_vertices_num];
        }

        std::vector<bool> visited_vertices(query_vertices_num, false);
        visited_vertices[order[0]] = true;
        for (ui i = 1; i < query_vertices_num; ++i) {
            ui vertex = order[i];

            ui nbrs_cnt;
            const ui *nbrs = query_graph.getVertexNeighbors(vertex, nbrs_cnt);
            for (ui j = 0; j < nbrs_cnt; ++j) {
                ui nbr = nbrs[j];

                if (visited_vertices[nbr]) {
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


    void cpu_preprocess(std::string data_graph_path, std::string query_graph_path)
    {
        data_graph_cpu.loadGraphFromFile(data_graph_path);
        query_graph.loadGraphFromFile(query_graph_path);

        //  ============== Step 1 ==============
        FilterVertices::DPisoFilter(data_graph_cpu, query_graph, candidates, candidates_count, 
                                        bfs_order, tree);      
        FilterVertices::sortCandidates(candidates, candidates_count, query_graph.getVerticesCount());

        for (ui i = 0; i < query_graph.getVerticesCount(); ++i)
        {
            max_candidate_cnt = std::max(max_candidate_cnt, candidates_count[i]);
        }

        std::cout << " MAX CANDS : " << max_candidate_cnt << std::endl;

        // ============== Step 2 ==============
        GenerateQueryPlan::generateGQLQueryPlan(data_graph_cpu, query_graph, candidates_count, matching_order, pivot);

        // for(ui i = 0; i < query_graph.getVerticesCount(); i++) {
        //     matching_order[i] = plan.arr[i];
        // }

        std::cout<<"======= print matching order =========="<<std::endl;
        for(ui i = 0; i < query_graph.getVerticesCount(); i++) {
            std::cout << matching_order[i] << " ";
        }
        std::cout << std::endl;

        generateBN(query_graph, matching_order, bn, bn_count);

        edge_matrix = new Edges **[query_graph.getVerticesCount()];
        for (ui i = 0; i < query_graph.getVerticesCount(); ++i) {
            edge_matrix[i] = new Edges *[query_graph.getVerticesCount()];
        }
        
        BuildTable::buildTable(data_graph_cpu, query_graph, candidates, candidates_count, edge_matrix);
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



