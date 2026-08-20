#include "global.h"
Graph data_graph;
#include "master.h"
#include "mc_task.h"
#include "mc_gpu_context_no_Q.h"
// #include "mc_gpu_context.h"
#include "mc_cpu_worker.h"
ull spilled_tasks;

static void print_help(const char *program)
{
    cout << "Usage: " << program << " [options]" << endl;
    cout << "Options:" << endl;
    cout << "  -dg <path>         Data graph binary file. Default: ./data/com-friendster.ungraph.txt.bin" << endl;
    cout << "  -cpu <n>           CPU workers. Default: 28" << endl;
    cout << "  -gpu <n>           GPU workers. Default: 1" << endl;
    cout << "  -eta <n>           ETA per warp. Default: 2000" << endl;
    cout << "  -cpuchunk <n>      CPU tasks per fetch. Default: 200" << endl;
    cout << "  -gpuchunk <n>      GPU tasks per fetch. Default: 1000000" << endl;
    cout << "  -hg_steal <n>      Host-to-GPU steal chunk. Default: 1000000" << endl;
    cout << "  -min_hg_steal <n>  Minimum queued tasks before GPU stealing. Default: 1000" << endl;
    cout << "  -tau <n>           CPU decomposition threshold (us). Default: 1000" << endl;
    cout << "  -pingpong <0|1|2>  0=no ping-pong, 1=ping-pong with abort, 2=ping-pong without abort. Default: 2" << endl;
}

class MCApp : public Master<MCCPUWorker, MCGPUContext>
{
public:
    bool gpu_enabled_;
    ull max_clique_size_ = 0;

    MCApp()
    {
        CommandLine::RuntimeConfig defaults;
        defaults.num_cpu_workers = 28;
        defaults.num_gpu_workers = 1;
        defaults.tasks_per_fetch_gpu_worker = 1000000;
        defaults.tasks_per_fetch_cpu_worker = 200;
        defaults.eta_per_warp = 2000;
        defaults.tau_time_us = 1000;
        defaults.ping_pong = 2;
        defaults.data_graph = "./data/com-friendster.ungraph.txt.bin";
        cmd.ParseRuntimeConfig(defaults);
        apply_runtime_config(cmd.runtime);
        gpu_enabled_ = cmd.runtime.num_gpu_workers > 0;
        std::string fp = cmd.runtime.data_graph;
        std::cout.imbue(std::locale());
        cout << " ======= Parameters ========" << endl;
        cout << "-dg: " << fp << endl;
        cout << "-cpu: " << cmd.runtime.num_cpu_workers << endl;
        cout << "-gpu: " << cmd.runtime.num_gpu_workers << endl;
        cout << "-eta: " << eta_per_warp() << endl;
        cout << "-cpuchunk: " << cmd.runtime.tasks_per_fetch_cpu_worker << endl;
        cout << "-gpuchunk: " << cmd.runtime.tasks_per_fetch_gpu_worker << endl;
        cout << "-hg_steal: " << cmd.runtime.hg_steal << endl;
        cout << "-min_hg_steal: " << cmd.runtime.min_hg_steal << endl;
        cout << "-tau: " << cmd.runtime.tau_time_us << endl;
        cout << "-pingpong: " << cmd.runtime.ping_pong << endl;
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
            {
                res += cw->total_counts;
                max_clique_size_ = std::max<ull>(max_clique_size_, cw->max_sz);
            }
            else if (gw)
            {
                res += gw->getContext()->get_results();
                max_clique_size_ = std::max(max_clique_size_, gw->getContext()->get_max_clique_size());
                spilled_tasks = gw->spilled_tasks;
                gw->getContext()->cleanup();
            }

            delete w;
        }
        return res;
    }

    ull get_max_clique_size() const
    {
        return max_clique_size_;
    }
};

int main(int argc, char *argv[])
{
    cmd = CommandLine(argc, argv);

    if (cmd.GetOptionValue("-h") != NULL || cmd.GetOptionValue("--help") != NULL)
    {
        print_help(argv[0]);
        return 0;
    }

    MCApp app;
    Timer t;
    app.run();
    ull total_count = app.get_results();
    cout << "Total time (s): " << t.elapsed() / 1e6 << endl;
    cout << "Total count: " << total_count << endl;
    cout << "Largest clique size: " << app.get_max_clique_size() << endl;
    cout << "Total spilled tasks: " << spilled_tasks << endl;

    return 0;
}
