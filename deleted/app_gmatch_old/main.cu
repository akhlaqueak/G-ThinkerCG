#include "global.h"
#include "master.h"
#include "plan.h"
Graph data_graph;
Plan plan;
#include "gm_task.h"
#include "gm_cpu_worker.h"
#include "gm_gpu_context.h"

void write_to_file(auto &query, ui id)
{
    // Convert id to string using std::to_string
    std::ofstream outfile("query_" + std::to_string(id) + ".txt");

    if (!outfile)
    {
        std::cerr << "Failed to open file for writing.\n";
        return; // remove 'return 1;' — it's a void function
    }

    // Header
    outfile << "t " << query.GetVertexCount() << " " << query.GetEdgeCount() << std::endl;

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
            outfile << "e " << i << " " << query.GetCols()[j] << std::endl; // use outfile
        }
    }
}

class GMApp : public Master<GMCPUWorker, GMGPUContext>
{
public:
    GMApp(ui argc, char *argv[])
    {
        CommandLine cmd(argc, argv);

        std::string fp = cmd.GetOptionValue("-dg", "./data/com-friendster.ungraph.txt.bin");
        int query_type = cmd.GetOptionIntValue("-q", 1000);
        ui eta_ = cmd.GetOptionIntValue("-eta", 1000);

        data_graph = Graph(fp);

        Graph query_G("", (PresetPatternType)query_type, GraphType::QUERY);
        query_G.SetConditions(query_G.GetConditions(query_G.GetBlissGraph()));

        write_to_file(query_G, query_type);
        return;

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

        for (int i = 0; i < data_graph.GetVertexCount(); ++i)
            data_array.push_back(new ui(i)); // data_array is member of Master
    }

    ui get_results()
    {
        ui res = 0;
        // using GPUWorkerT = GPUWorker<GMGPUContext>;
        // while (workers_list.size())
        // {
        //     WorkerT *w = (WorkerT *)workers_list.dequeue();
        //     GMCPUWorker *cw = dynamic_cast<GMCPUWorker *>(w);
        //     GPUWorkerT *gw = dynamic_cast<GPUWorkerT *>(w);

        //     if (cw)
        //         res += cw->total_counts;
        //     else if (gw)
        //     {
        //         res += gw->getContext()->get_results();
        //     }
        // }
        return res;
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
