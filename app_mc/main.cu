#include "global.h"
Graph data_graph;
#include "master.h"
#include "mc_task.h"
#include "mc_gpu_context_no_Q.h"
// #include "mc_gpu_context.h"
#include "mc_cpu_worker.h"
ull spilled_tasks;
class MCApp : public Master<MCCPUWorker, MCGPUContext>
{
public:
    MCApp()
    {
        CommandLine::RuntimeConfig defaults;
        defaults.num_cpu_workers = 28;
        defaults.num_gpu_workers = 1;
        defaults.tasks_per_fetch_gpu_worker = 1000000;
        defaults.tasks_per_fetch_cpu_worker = 50;
        defaults.eta_per_warp = 1000;
        defaults.tau_time_us = 10;
        defaults.ping_pong = true;
        defaults.data_graph = "./data/com-friendster.ungraph.txt.bin";
        cmd.ParseRuntimeConfig(defaults);
        apply_runtime_config(cmd.runtime);
        std::string fp = cmd.runtime.data_graph;
        std::cout.imbue(std::locale());
        cout << " ======= Parameters ========" << endl;
        cout << "Graph: " << fp << endl;
        cout << "cpu workers: " << cmd.runtime.num_cpu_workers << endl;
        cout << "gpu workers: " << cmd.runtime.num_gpu_workers << endl;
        cout << "eta (tasks load per warp): " << eta_per_warp() << endl;
        cout << "cpu chunk: " << cmd.runtime.tasks_per_fetch_cpu_worker << endl;
        cout << "gpu chunk: " << cmd.runtime.tasks_per_fetch_gpu_worker << endl;
        cout << "tau_time: " << cmd.runtime.tau_time_us << endl;
        cout << " ======= ********** ========" << endl;
        
        data_graph = Graph(fp);

        for (int i = 0; i < data_graph.GetVertexCount(); ++i)
            data_array.push_back(i); // data_array is member of Master
    }

    ull get_results()
    {
        ull res = 0;
        using GPUWorkerT = GPUWorker<MCGPUContext>;
        while (workers_list.size())
        {
            WorkerT *w = (WorkerT *)workers_list.dequeue();
            MCCPUWorker *cw = dynamic_cast<MCCPUWorker *>(w);
            GPUWorkerT *gw = dynamic_cast<GPUWorkerT *>(w);

            if (cw)
                res += cw->total_counts;
            else if (gw)
            {
                res += gw->getContext()->get_results();
                spilled_tasks = gw->spilled_tasks;
                gw->getContext()->cleanup();
            }

            delete w;
        }
        return res;
    }
};

int main(int argc, char *argv[])
{
    cmd = CommandLine(argc, argv);

    MCApp app;
    Timer t;
    app.run();
    cout << "Total time (s): " << t.elapsed() / 1e6 << endl;
    cout << "Total count: " << app.get_results() << endl;
    cout << "Total spilled tasks: " << spilled_tasks << endl;

    return 0;
}
