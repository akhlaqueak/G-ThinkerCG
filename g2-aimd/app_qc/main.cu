#include "system/worker.h"
#include "qc_support.h"
#include "qc_gpu_context.h"

CPU_Data hd;
GPU_Data dd;
CPU_Cliques hc;
CPU_Graph *hg = nullptr;

int main(int argc, char *argv[])
{
    QCApp app;
    Worker<QCApp> worker;

    worker.run(argc, argv, app);

    return 0;
}
