#pragma once

#include "worker.h"
// __global__ functions can't be defined as members
template <class T>
__global__ void generateInitialTasks(T gc)
{
    gc.generateInitialTasks(gc.sources, gc.sources_num, gc.v_proc, gc.Bwr, gc.row_ptrs, gc.cols);
}

template <class T>
__global__ void process(T gc)
{
    gc.process(gc.Brd, gc.row_ptrs, gc.cols);
}

template <class T>
__global__ void extend(T gc)
{
    gc.extend(gc.Brd, gc.Bwr, gc.H, gc.row_ptrs, gc.cols);
}

template <class T>
__global__ void loadFromHost(T gc)
{
    gc.loadFromHost();
}

template <class T>
__global__ void dumpToHost(T gc)
{
    gc.dumpToHost();
}

template <class GPUContext>
class GPUWorker : public Worker<typename GPUContext::TaskType>
{
    using TaskT = typename GPUContext::TaskType;
    using ContextT = typename TaskT::ContextType;

    // using GPUContextype = GPUContext;

public:
    GPUContext gc;
    Timer progress;
    GPUWorker() : Worker<TaskT>(tasks_per_fetch_gpu_worker_g)
    {
        gc.allocateMemory();
        this->Lv.reserve(tasks_per_fetch_gpu_worker_g);
        this->Lt.reserve(tasks_per_fetch_gpu_worker_g);
        this->Lo.reserve(tasks_per_fetch_gpu_worker_g);
    }

    virtual void run()
    {
        if (gc.v_proc[0] >= gc.sources_num[0])
            gc.sources_num[0] = 0;
        if (this->Lv.size())
        {
            cout<<"Lv: "<<this->Lv.size()<<endl;
            gc.move_vertices_to_gpu(this->Lv);
        }
        else
        {
            cout << "Lt: " << this->Lt.size() << endl;
            gc.move_tasks_from_Sc(this->Lt, gc.H);
        }

        this->Lv.clear();
        this->Lt.clear();
        Timer prog_trigger;
        while (true)
        {
            if (not gc.H.empty())
            {
                loadFromHost<<<BLK_NUMS, BLK_DIM>>>(gc);
                deviceSynch();
                move_tasks_to_cpu();
            }
            else if ((!gc.topLevelWorkExist()) && gc.Bwr.empty() && gc.Brd.empty())
                break;
            if (gc.sources_num[0] > 0)
            {
                generateInitialTasks<<<BLK_NUMS, BLK_DIM>>>(gc);
                deviceSynch();
            }

            gc.incrementLevel();
            while (true)
            {
                if (gc.ping_pong_mode)
                {
                    bool next_expansion = ping_pong_style_expansion();
                    if (!next_expansion)
                        break;
                }
                else
                {
                    bool next_expansion = level_style_expansion();
                    if (!next_expansion)
                        break;
                }
            }
        }
    }

    bool level_style_expansion()
    {
        gc.resetLevel();
        // cout<<gc.Brd.size()<<endl;
        process<<<BLK_NUMS, BLK_DIM>>>(gc);
        extend<<<BLK_NUMS, BLK_DIM>>>(gc);

        deviceSynch();
        gc.init_level();

        auto tick = chrono::steady_clock::now();
        deviceSynch();

        // if (prog_trigger.elapsed() / 1e6 > 10)
        // {
        //     prog_trigger.restart();
        // }

        if (!gc.Bwr.empty())
        {
            gc.incrementLevel();
            if (gc.isOverflow())
            {
                dump_to_host();
                move_tasks_to_cpu();
            }
        }
        else if (gc.Brd.empty())
        {
            if (!gc.decrementLevel())
                return false;
        }
        return true;
    }

    bool ping_pong_style_expansion()
    {
        gc.resetLevel();
        gc.init_level();
        // cout<<gc.Brd.size()<<endl;
        process<<<BLK_NUMS, BLK_DIM>>>(gc);
        extend<<<BLK_NUMS, BLK_DIM>>>(gc);

        deviceSynch();

        if (gc.isOverflow())
        {
            dump_to_host();// dumps remaining unxpanded Brd tasks to H
            gc.incrementLevel(); // switch Bwr => Brd
            dump_to_host();// now dump Brd to host... 
            gc.change_expansion_mode();
            move_tasks_to_cpu(); 
            return false;
        }
        gc.incrementLevel();
        if (gc.Brd.empty())
        {
            return false;
        }
        return true;
    }


    void dump_to_host()
    {        
        show_progress(" ** host dump ** ");
        dumpToHost<<<BLK_NUMS, BLK_DIM>>>(gc);
        deviceSynch();
    }

    void show_progress(std::string msg = "Progress Report")
    {
        std::cout << "== " << msg << " ==" << std::endl;
        std::cout << "Elapsed Time (sec): " << progress.elapsed() / 1000000 << std::endl;
        gc.buffers_status();
    }
    ull SC_size()
    {
        stack<TaskT *> *SC = (stack<TaskT *> *)global_SC;

        shared_lock<shared_timed_mutex> lock(SC_mtx);
        // shared_lock lock(SC_mtx);
        // unique_lock lock(SC_mtx);
        return SC->size();
    }
    void move_tasks_to_cpu()
    {
        // return; // disabling spilling...
        if (workers_list.size() > num_cpu_workers / 2 and SC_size() < gpu_to_host_transfer_size_g)
        {
            if (gc.H.otail[0])
                chkerr(cudaMemPrefetchAsync(gc.H.offsets, sizeof(ull) * gc.H.otail[0], cudaCpuDeviceId, 0));
            if (gc.H.vtail[0])
                chkerr(cudaMemPrefetchAsync(gc.H.vertices, sizeof(VertexID) * gc.H.vtail[0], cudaCpuDeviceId, 0));
            deviceSynch();
            gc.move_tasks_to_Sc(this->Lo, gc.H);
            this->spilled_tasks += this->Lo.size();
            this->spill_Lo();
        }
    }

    GPUContext *getContext()
    {
        return &gc;
    }
};
