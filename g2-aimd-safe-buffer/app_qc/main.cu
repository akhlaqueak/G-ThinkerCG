#include "system/worker.h"
#include "qc_support.h"
#include "qc_gpu_context.h"

CPU_Data hd;
GPU_Data dd;
CPU_Cliques hc;
CPU_Graph *hg = nullptr;

int main(int argc, char *argv[])
{
    // Match app_qc's search-time boundary: include application setup,
    // preprocessing, allocation, execution, and result collection.
    qc_runtime_state::search_timer.StartTimer();

    QCApp app;
    Worker<QCApp> worker;

    std::vector<std::string> worker_args(argv, argv + argc);
    if (qc_get_option_value(worker_args, "-dg", "").empty())
    {
        const std::string expanded_graph_file = qc_get_option_value(worker_args, "-f", "");
        if (!expanded_graph_file.empty())
        {
            worker_args.push_back("-dg");
            worker_args.push_back(qc_data_graph_path_from_expanded_graph(expanded_graph_file));
        }
    }

    std::vector<char *> worker_argv;
    worker_argv.reserve(worker_args.size());
    for (std::string &arg : worker_args)
        worker_argv.push_back(&arg[0]);

    worker.run(static_cast<int>(worker_argv.size()), worker_argv.data(), app);

    return 0;
}
