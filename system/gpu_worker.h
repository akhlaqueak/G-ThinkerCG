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

template <class GPUContext>
class GPUWorker : public Worker<typename GPUContext::TaskType>
{
    using TaskT = typename GPUContext::TaskType;
    using ContextT = typename TaskT::ContextType;

    // using GPUContextype = GPUContext;

public:
    GPUContext gc;
    Timer progress;

    GPUWorker(ui eta_limit) : Worker<TaskT>(tasks_per_fetch_gpu_worker_g)
    {
        gc.set_eta_limit(eta_limit);
        gc.allocateMemory();
        this->Lv.reserve(tasks_per_fetch_gpu_worker_g);
        this->Lt.reserve(tasks_per_fetch_gpu_worker_g);
        this->Lo.reserve(tasks_per_fetch_gpu_worker_g);
    }

    virtual void run()
    {
        // if (gc.v_proc[0] >= gc.sources_num[0])
        //     gc.sources_num[0] = 0;
        // every run invocation, run function sets layered mode, 
        // pingpong mode is enabled only call was for Lv and pingpong flag is on... 
        gc.set_layered_mode(); 
        if (this->Lv.size())
        {
            cout << "Lv: " << this->Lv.size() << endl;
            gc.move_vertices_to_gpu(this->Lv);
            if (g_ping_pong_flag){
                gc.set_ping_pong_mode();
                gc.init_chunk();
            }
        }
        else if (this->Lt.size())
        {
            cout << "Lt: " << this->Lt.size() << endl;
            gc.move_tasks_from_Sc(this->Lt, gc.H);
        }
        else
            return;

        this->Lv.clear();
        this->Lt.clear();
        Timer prog_trigger;
        while (true)
        {
            if (not gc.H.empty())
            {
                gc.load_from_host();
            }
            else if ((!gc.topLevelWorkExist()) && gc.Bwr.empty() && gc.Brd.empty())
                break;
            // if (gc.sources_num[0] > 0)
            if (gc.v_proc[0] < gc.sources_num[0])
            {
                generateInitialTasks<<<BLK_NUMS, BLK_DIM>>>(gc);
                deviceSynch();
            }

            gc.incrementLevel();
            while (true)
            {
                if (gc.ping_pong_mode)
                {
                    bool next_expansion = ping_pong_mode_expansion();
                    if (!next_expansion)
                        break;
                }
                else
                {
                    bool next_expansion = layered_mode_expansion();
                    if (!next_expansion)
                        break;
                }
            }
        }
        cout<<"----"<<endl;
    }

    bool layered_mode_expansion()
    {
        gc.resetLevel();
        // show_progress("layered before ");
        process<<<BLK_NUMS, BLK_DIM>>>(gc);
        extend<<<BLK_NUMS, BLK_DIM>>>(gc);
        deviceSynch();

        gc.init_level();

        auto tick = chrono::steady_clock::now();
        deviceSynch();
        // show_progress("layered after ");

        if (!gc.Bwr.empty())
        {
            bool overflow = gc.isOverflow();
            gc.incrementLevel();
            // if (gc.isOverflow() or gc.Bwr.isApprochingEnd())
            if (overflow)
            {
                gc.dump_to_host();
                move_tasks_to_cpu();
                show_progress(" ** dump done ** ");
                if (!gc.decrementLevel())
                    return false;
            }
        }
        else if (gc.Brd.empty())
        {
            if (!gc.decrementLevel())
                return false;
        }
        return true;
    }

    bool ping_pong_mode_expansion()
    {
        gc.resetLevel();
        // cout<<gc.Brd.size()<<endl;
        // show_progress("pingpong before ");
        process<<<BLK_NUMS, BLK_DIM>>>(gc);
        extend<<<BLK_NUMS, BLK_DIM>>>(gc);
        deviceSynch();

        gc.init_level();

        auto tick = chrono::steady_clock::now();
        deviceSynch();

        if (gc.isOverflow())
        {
            if (g_abort_chunk_flag)
            {
                show_progress("pingpong chunk aborted, moving to layered mode ");
                gc.resetLevel();
                gc.abort_chunk();
                gc.v_proc[0] = 0;
                gc.H.clear();
                g_ping_pong_flag = false;
                gc.set_layered_mode();
                
                show_progress("abort done");
            }
            else
            {
                gc.dump_to_host();   // dumps remaining unxpanded Brd tasks to H
                gc.incrementLevel(); // switch Bwr => Brd
                gc.dump_to_host();   
                gc.set_layered_mode();
                move_tasks_to_cpu();
            }
            return false;
        }
        gc.incrementLevel();
        if (gc.Brd.empty())
        {
            return false;
        }
        return true;
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
        if (workers_list.size() > num_cpu_workers / 2 and SC_size() < gpu_to_host_transfer_size_g)
        {
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
// this is also last line
