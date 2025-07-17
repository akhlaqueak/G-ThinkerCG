#ifndef SYSTEM_PIPELINE_EXECUTOR_H
#define SYSTEM_PIPELINE_EXECUTOR_H

#include <cuda_runtime.h>

#include "common/meta.h"
#include "common/gpu_env.h"
#include "common/timer.h"
#include "common/graph.h"
#include "device/device_memory_info.h"
#include "device/cuda_context.h"
#include "device/device_array.h"
#include "system/work_context.h"

// __global__ functions can't be defined in class
template <class Application>
__global__ void generateSubgraphs(Application app, unsigned int base)
{
    app.generateSubgraphs(base);
}

template <class Application>
__global__ void process(Application app)
{
    app.processSubgraphs();
}

template <class Application>
__global__ void expand(Application app)
{
    app.expand();
}

template <class Application>
__global__ void loadFromHost(Application app)
{
    app.loadFromHost();
}

template <typename Application>
class PipelineExecutor
{
public:

    PipelineExecutor(Graph *graph, Application _app, std::vector<StoreStrategy> &strategy_)
        : device_id_(0), graph_(graph), app(_app), strategy(strategy_)
    {
        tag_ << "device " << device_id_;

        Timer timer;
        timer.StartTimer();

        CUDA_ERROR(cudaSetDevice(0));
        // CUDA_ERROR(cudaDeviceReset()); // erase everything on the device.

        cudaStream_t stream;
        CUDA_ERROR(cudaStreamCreate(&stream));

        DeviceMemoryInfo *device_memory_info = new DeviceMemoryInfo(device_id_, kDeviceMemoryLimits[device_id_], false);
        cuda_context_ = new CudaContext(device_memory_info, stream);
        // std::cout << "available mem: " << cuda_context_->GetDeviceMemoryInfo()->GetAvailableMemorySizeMB() << " MB" << std::endl;

        CUDA_ERROR(cudaMallocManaged(&(app.ctx), sizeof(WorkContext))); // make work_context_ GPU accessible
        app.ctx->context = cuda_context_;

        // app.ctx->d_row_ptrs = new DeviceArray<uintE>(graph->GetVertexCount() + 1, cuda_context_);
        CUDA_ERROR(cudaMalloc(&((app.ctx)->d_row_ptrs), sizeof(uintE) * (graph->GetVertexCount() + 1)));

        CUDA_ERROR(cudaMalloc(&((app.ctx)->d_cols),sizeof(uintV) * graph->GetEdgeCount()));

        // GPU can access array defined like this
        CUDA_ERROR(cudaMalloc((void **)&(app.ctx)->sources, graph->GetVertexCount() * sizeof(uintV)));
        // adding all vertices in sources
        ui* temp=new ui[graph->GetVertexCount()];
        for(ui i=0;i<graph->GetVertexCount();i++)
            temp[i]=i;
        HToD((app.ctx)->sources, temp, graph->GetVertexCount());
        delete [] temp;
        // CUDA_ERROR(cudaMallocManaged(&((app.ctx)->sources), max_partitioned_sources_num * sizeof(uintV)));

        // CUDA_ERROR(cudaMallocHost(&((app.ctx)->sources), max_partitioned_sources_num * sizeof(uintV)));

        CUDA_ERROR(cudaMallocManaged(&((app.ctx)->level), sizeof(ui)));

        CUDA_ERROR(cudaMallocManaged((void **)&(app.ctx)->sources_num, sizeof(size_t)));
        (app.ctx)->sources_num[0] = graph->GetVertexCount();        
        HToD((app.ctx)->d_row_ptrs, graph->GetRowPtrs(), graph->GetVertexCount() + 1);
        

        HToD((app.ctx)->d_cols, graph->GetCols(), graph->GetEdgeCount());

        // std::cout << "available mem after: " << cuda_context_->GetDeviceMemoryInfo()->GetAvailableMemorySizeMB() << " MB" << std::endl;
        timer.EndTimer();
        timer.PrintElapsedMicroSeconds(tag_.str() + " executor construction");

        // TODO: allcate space for rdBuffer and wrBuffer??

        app.allocateMemory();            // UDF


        // TODO: parse .stat file and compute how much reserved_memory is needed.

        app.initialize(0); // allocate space for C1, C2 and host buffer // FIXME:

    }

    virtual ~PipelineExecutor()
    {
        // should call destruction function
        // delete work_context_;
        // work_context_ = nullptr;
        cuda_context_ = nullptr;
        graph_ = nullptr;
    }

    void PrintTotalCounts() const
    {
        // for (int i = 0; i < plans_.size(); i++)
        //     std::cout << "device " << device_id_ << " query " << i << " count: " << total_counts_[i] << std::endl;
        // FIXME:
        assert(false);
    }


    void Transfer(std::unique_ptr<ViewBinHolder> &vbh)
    {
        std::cout << tag_.str() << " transfer view bin " << vbh->GetId() << std::endl;
        // copy view bin sources and csr to gpu memory
        Timer timer;
        timer.StartTimer();

        app.ctx->sources_num[0] = vbh->GetSourcesNum();

        // memcpy(app.ctx->sources, vbh->GetSources().data(), vbh->GetSourcesNum() * sizeof(uintV));
        
        HToD((app.ctx)->sources, vbh->GetSources().data(), vbh->GetSourcesNum());

        HToD((app.ctx)->d_row_ptrs, vbh->GetRowPtrs(), graph_->GetVertexCount() + 1);

        HToD((app.ctx)->d_cols, vbh->GetCols(), vbh->GetTotalSize());

        timer.EndTimer();
        timer.PrintElapsedMicroSeconds(tag_.str() + " memory copy");
    }


    // TODO: change
    void Run()
    {
        app.sg->chunk[0] = MAXCHUNK;

        for (unsigned int i = 0; i < app.ctx->sources_num[0];)
        {
            app.ctx->level[0] = 1;
            generateSubgraphs<<<BLK_NUMS, BLK_DIM>>>(app, i);
            i += app.sg->chunk[0];
            while (true)
            {
                cudaDeviceSynchronize();
                app.sg->swapBuffers();
                if (app.sg->isEmpty())
                {
                    if (app.sgHost->isEmpty())
                    {
                        app.sgHost->swapBuffers();
                        if (app.sgHost->isEmpty())
                        {
                            break;
                        }
                        std::cout << "[" << app.sgHost->rdBuff.otail[0] << ", " << app.sgHost->rdBuff.vtail[0] << "]" << std::endl;
                    }

                    loadFromHost<<<BLK_NUMS, BLK_DIM>>>(app);
                    cudaDeviceSynchronize();
                    continue;
                }
                process<<<BLK_NUMS, BLK_DIM>>>(app);
                expand<<<BLK_NUMS, BLK_DIM>>>(app);
                cudaDeviceSynchronize();
                // if (!app.sgHost->isEmpty())
                // {
                // }
                if (app.sg->isOverflow())
                {
                    app.iterationFailed();
                    i -= app.sg->chunk[0];
                    break;
                }
                if (app.sgHost->isOverflowToHost())
                {
                    std::cout << "Host overflow occured";
                    exit(0);
                }
                app.ctx->level[0]++;
            }
            // std::cout << "i: " << i << " " << std::endl;
            app.sg->adjustChunk();
            app.iterationSuccess();
        }
        app.completion();
    }

public:
    size_t device_id_;
    Graph *graph_;
    CudaContext *cuda_context_;
    WorkContext *work_context_;

    std::stringstream tag_;
    Application app;

    std::vector<StoreStrategy> strategy;
};

#endif