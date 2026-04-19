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
            cout << "Lv: " << this->Lv.size() << endl;
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
        if (ping_pong)
            gc.set_ping_pong_mode();
        else
            gc.set_layered_mode();
        while (true)
        {
            if (not gc.H.empty())
            {
                load_from_host();
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
            gc.incrementLevel();
            if (gc.isOverflow() or gc.Bwr.isApprochingEnd())
            {
                dump_to_host();
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

        // show_progress("pingpong after ");

        if (gc.isOverflow())
        {
            dump_to_host();      // dumps remaining unxpanded Brd tasks to H
            gc.incrementLevel(); // switch Bwr => Brd
            dump_to_host();      // now dump Brd to host...
            gc.set_layered_mode();
            // show_progress("pingpong overflow done ");
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
        if (gc.Brd.empty())
            return;

        const ull src_ohead = gc.Brd.ohead[0];
        const ull src_otail = gc.Brd.otail[0];
        const ull offset_count = src_otail - src_ohead;
        cout << "moving D->H: " << offset_count / 3 << endl;
        if (offset_count == 0)
            return;

        ull *offsets = new ull[offset_count];
        chkerr(cudaMemcpy(offsets, gc.Brd.offsets + src_ohead, sizeof(ull) * offset_count, cudaMemcpyDeviceToHost));

        ull min_src_vstart = offsets[0];
        ull max_src_vend = offsets[2];
        for (ull i = 3; i < offset_count; i += 3)
        {
            min_src_vstart = std::min(min_src_vstart, offsets[i]);
            max_src_vend = std::max(max_src_vend, offsets[i + 2]);
        }

        const ull dst_vstart = gc.H.vtail[0];
        const ull total_vertices = max_src_vend - min_src_vstart;

        for (ull i = 0; i < offset_count; i += 3)
        {
            const ull sg_st = offsets[i];
            const ull sg_md = offsets[i + 1];
            const ull sg_en = offsets[i + 2];

            gc.H.offsets[gc.H.otail[0]] = dst_vstart + (sg_st - min_src_vstart);
            gc.H.offsets[gc.H.otail[0] + 1] = (sg_md == 0) ? 0 : dst_vstart + (sg_md - min_src_vstart);
            gc.H.offsets[gc.H.otail[0] + 2] = dst_vstart + (sg_en - min_src_vstart);
            gc.H.otail[0] += 3;
        }

        gc.H.copy_host_range(gc.Brd, dst_vstart, min_src_vstart, total_vertices);
        gc.H.vtail[0] += total_vertices;
        delete[] offsets;
    }

    void load_from_host()
    {
        if (gc.H.empty())
            return;

        const ull src_ohead = gc.H.ohead[0];
        const ull src_otail = gc.H.otail[0];
        const ull available_tasks = (src_otail - src_ohead) / 3;

        if (available_tasks == 0)
            return;

        const ull tasks_to_load = std::min<ull>(ETA*10, available_tasks);
        const ull offset_count = tasks_to_load * 3;
        const ull offsets_start = src_otail - offset_count;

        ull *offsets = new ull[offset_count];
        std::copy(gc.H.offsets + offsets_start, gc.H.offsets + src_otail, offsets);

        ull min_src_vstart = offsets[0];
        ull max_src_vend = offsets[2];
        for (ull i = 3; i < offset_count; i += 3)
        {
            min_src_vstart = std::min(min_src_vstart, offsets[i]);
            max_src_vend = std::max(max_src_vend, offsets[i + 2]);
        }

        const ull total_vertices = max_src_vend - min_src_vstart;
        const ull dst_otail = gc.Bwr.otail[0];
        const ull dst_vstart = gc.Bwr.vtail[0];

        if (dst_otail + offset_count > gc.Bwr.capacity[0] || dst_vstart + total_vertices > gc.Bwr.capacity[0])
            throw std::runtime_error("Device buffer overflow");

        ull *translated_offsets = new ull[offset_count];
        ull write_idx = 0;
        for (ull i = offset_count; i > 0; i -= 3)
        {
            const ull idx = i - 3;
            const ull sg_st = offsets[idx];
            const ull sg_md = offsets[idx + 1];
            const ull sg_en = offsets[idx + 2];

            translated_offsets[write_idx] = dst_vstart + (sg_st - min_src_vstart);
            translated_offsets[write_idx + 1] = (sg_md == 0) ? 0 : dst_vstart + (sg_md - min_src_vstart);
            translated_offsets[write_idx + 2] = dst_vstart + (sg_en - min_src_vstart);
            write_idx += 3;
        }

        chkerr(cudaMemcpy(gc.Bwr.offsets + dst_otail, translated_offsets, sizeof(ull) * offset_count, cudaMemcpyHostToDevice));
        gc.Bwr.copy_host_to_device_range(gc.H, dst_vstart, min_src_vstart, total_vertices);
        gc.Bwr.otail[0] = dst_otail + offset_count;
        gc.Bwr.vtail[0] = dst_vstart + total_vertices;
        gc.H.otail[0] = offsets_start;
        delete[] translated_offsets;
        delete[] offsets;
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
        return; // disabling spilling...
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
