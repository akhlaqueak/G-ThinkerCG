#include "global.h"
Graph data_graph;
#include "master.h"
#include "mc_task.h"
// #include "mc_gpu_context_no_Q.h"
#include "mc_gpu_context.h"
#include "mc_cpu_worker.h"
CommandLine cmd;
class MCApp : public Master<MCCPUWorker, MCGPUContext>
{
public:
    MCApp()
    {
        num_cpu_workers = cmd.GetOptionIntValue("-cpu", 32);
        num_gpu_workers = cmd.GetOptionIntValue("-gpu", 1);
        tasks_per_fetch_gpu_worker_g = cmd.GetOptionIntValue("-gpuchunk", 100000);
        tasks_per_fetch_g = cmd.GetOptionIntValue("-cpuchunk", 10);
        ui eta_ = cmd.GetOptionIntValue("-eta", 1000);
        std::string fp = cmd.GetOptionValue("-dg", "./data/com-friendster.ungraph.txt.bin");
        std::cout.imbue(std::locale());
        cout<<" ======= Parameters ========"<<endl;
        cout<<"Graph: "<<fp<<endl;
        cout<<"cpu workers: "<<num_cpu_workers<<endl;
        cout<<"gpu workers: "<<num_gpu_workers<<endl;
        cout<<"eta: "<<eta_<<endl;
        cout<<"cpu chunk: "<<tasks_per_fetch_g<<endl;
        cout<<"gpu chunk: "<<tasks_per_fetch_gpu_worker_g<<endl;
        cout<<" ======= ********** ========"<<endl;



        data_graph = Graph(fp);

        eta_ *= N_WARPS;
        cudaMemcpyToSymbol(eta, &eta_, sizeof(ui));

        for (int i = 0; i < data_graph.GetVertexCount(); ++i)
            data_array.push_back(new ui(i)); // data_array is member of Master
    }

    ui get_results()
    {
        ui res = 0;
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
            }
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
    return 0;
}
