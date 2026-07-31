#ifndef SYSTEM_APPBASE_H
#define SYSTEM_APPBASE_H

template <class BufferT, class TaskT>
class GPUContext
{
    struct BPointers
    {
        ull ohead, otail, vtail;
    };

public:
    struct BufferReservation
    {
        ull vt;
        BufferT *buffer;
    };

    // using ContextT = typename TaskT::ContextType;
    // using TaskType = TaskT;
    typedef typename TaskT::ContextType ContextT;
    typedef TaskT TaskType;
    using TaskContainer = vector<TaskT *>;
    ull *v_proc = nullptr;
    std::stack<BPointers> SL;
    bool ping_pong_mode = true;
    bool abort_chunk_on_device_full = true;

    // memory is allocated only to B and H buffers, Brd, Bwr are just pointers hovering over B
    // host spill buffer
    BufferT B;
    BufferT H;

    BufferT Bwr;
    BufferT Brd;
    ull H_offsets_sorted_until = 0;

    ull *sources_num = nullptr;  // size of Lv
    VertexID *sources = nullptr; // Lv copy on GPU
    ui eta_limit = 1000 * N_WARPS;

    // graph in CSR on GPU
    ull *row_ptrs = nullptr;
    VertexID *cols = nullptr;

    // Device UDFs
    __device__ virtual void generateInitialTasks(VertexID *sources, ull *sources_num, ull *v_proc, BufferT &Bwr, ull *row_ptrs, VertexID *cols) = 0;
    __device__ virtual void process(BufferT &Brd, ull *row_ptrs, VertexID *cols) = 0;
    __device__ virtual void extend(BufferT &Brd, BufferT &Bwr, BufferT &H, ull *row_ptrs, VertexID *cols) = 0;
    // Host UDFs
    virtual void move_tasks_from_Sc(TaskContainer &source, BufferT &H) = 0;
    virtual void move_tasks_to_Sc(TaskContainer &collector, BufferT &H) = 0;
    virtual void load_graph(ull *&row_ptrs, VertexID *&cols) = 0;
    virtual void initialize() = 0;
    virtual void init_level() {};
    virtual void init_chunk() {}
    virtual void abort_chunk() {}

    void cleanup()
    {
        if (H.offsets)
            chkerr(cudaFree(H.offsets));
        if (H.vertices)
            chkerr(cudaFree(H.vertices));
        H.offsets = nullptr;
        H.vertices = nullptr;

        if (B.offsets)
            chkerr(cudaFree(B.offsets));
        if (B.vertices)
            chkerr(cudaFree(B.vertices));
        B.offsets = nullptr;
        B.vertices = nullptr;
        Bwr.offsets = nullptr;
        Bwr.vertices = nullptr;
        Brd.offsets = nullptr;
        Brd.vertices = nullptr;
    }

    bool topLevelWorkExist()
    {
        return v_proc[0] < sources_num[0];
    }

    void set_eta_limit(ui limit)
    {
        eta_limit = limit;
    }

    void allocateMemory(ull reserved_mem = 0)
    {
        initialize(); // call UDF first, so that application specific memory is allocated
        chkerr(cudaMallocManaged((void **)&v_proc, sizeof(ull)));
        chkerr(cudaMalloc(&sources, sizeof(VertexID) * tasks_per_fetch_gpu_worker_g));
        chkerr(cudaMallocManaged((void **)&sources_num, sizeof(ull)));

        size_t total, free;
        cudaMemGetInfo(&free, &total);
        cout << "Available Memory " << free / 1024 / 1024 / 1024 << " GB" << endl;
        // leave some memory for pointers and other variables...
        free -= 500'000'000 + reserved_mem;

        ull sz = free / BufferT::sizeOf();

        B.allocateMemory(sz);
        Bwr = B;
        Brd = B;

        Bwr.allocatePtrs();
        Brd.allocatePtrs();

        // this version allocates pinned host memory
        H.allocateMemory();
    }

    void move_vertices_to_gpu(vector<ui> &data_items)
    {
        HToD(sources, data_items.data(), data_items.size());
        sources_num[0] = data_items.size();
        v_proc[0] = 0;
    }
    void resetLevel()
    {
        Brd.n_tasks_proc[0] = 0;
        Bwr.n_tasks_proc[0] = 0;
        H.n_tasks_proc[0] = 0;
        
        Brd.eta_filled[0] = false;
        Bwr.eta_filled[0] = false;

        Brd.overflow[0] = false;
        Bwr.overflow[0] = false;
    }
    void swap_buffers()
    {
        std::swap(Brd, Bwr);
    }
    void incrementLevel()
    {
        resetLevel();
        if (ping_pong_mode)
        {
            swap_buffers();
            Bwr.reset_pointers();
            return;
        }
        if (!Brd.empty())
            SL.push({Brd.ohead[0], Brd.otail[0], Brd.vtail[0]});

        Brd.ohead[0] = Bwr.ohead[0];
        Brd.otail[0] = Bwr.otail[0];
        Brd.vtail[0] = Bwr.vtail[0];
        Bwr.ohead[0] = Bwr.otail[0];
    }

    bool decrementLevel()
    {
        resetLevel();
        if (SL.empty())
        {
            Brd.clear();
            Bwr.clear();
            return false;
        }
        BPointers bp = SL.top();
        SL.pop();
        Brd.ohead[0] = bp.ohead;
        Brd.otail[0] = bp.otail;
        Brd.vtail[0] = bp.vtail;

        Bwr.otail[0] = Bwr.ohead[0] = Brd.otail[0];
        Bwr.vtail[0] = Brd.vtail[0];
        return true;
    }
    void showStatus(string msg)
    {
        H.print("H: ");
        Bwr.print("Bwr: ");
        Brd.print("Brd: ");
        cout << endl;
    }

    __device__ bool isLevelFilled()
    {
        if (ping_pong_mode)
            return this->isOverflow();
        return (Bwr.n_tasks_proc[0] > eta); // Destination
    }

    __device__ __host__ bool isOverflow()
    {
        return Bwr.isOverflow();
    }

    __device__ BufferReservation append(ull sglen)
    {
        ull vt = Bwr.append(sglen);
        if (vt != INVALID_BUFFER_POS)
            return {vt, &Bwr};

        if (abort_chunk_on_device_full && ping_pong_mode)
            return {INVALID_BUFFER_POS, nullptr};

        vt = H.append(sglen);
        if (vt != INVALID_BUFFER_POS)
            return {vt, &H};

        return {INVALID_BUFFER_POS, nullptr};
    }

    __device__ BufferReservation append_batch(ull sglen, ui num)
    {
        ull vt = Bwr.append_batch(sglen, num);
        if (vt != INVALID_BUFFER_POS)
            return {vt, &Bwr};

        if (abort_chunk_on_device_full && ping_pong_mode)
            return {INVALID_BUFFER_POS, nullptr};

        vt = H.append_batch(sglen, num);
        if (vt != INVALID_BUFFER_POS)
            return {vt, &H};

        return {INVALID_BUFFER_POS, nullptr};
    }

    __device__ void dumpToHost(SubgraphOffsets &so)
    {
        if (H.overflow[0])
            return;

        const ull sglen = so.en - so.st;
        ull vt = H.append(sglen);

        if (H.overflow[0])
            return;

        H.copy_range(Brd, vt, so.st, sglen);
    }

    void set_ping_pong_mode()
    {
        ping_pong_mode = true;  
        abort_chunk_on_device_full = g_abort_chunk_flag;

        Brd.capacity[0] = B.capacity[0] / 2;
        Bwr.capacity[0] = B.capacity[0];

        Brd.second_buffer = false;
        Bwr.second_buffer = true;

        Bwr.reset_pointers(); // Bwr is the second buffer, its starting point is B.capacity[0]/2
        Brd.reset_pointers();
    }
    void set_layered_mode()
    {
        ping_pong_mode = false;
        abort_chunk_on_device_full = false;

        Brd.capacity[0] = B.capacity[0];
        Bwr.capacity[0] = B.capacity[0];

        Brd.second_buffer = false;
        Bwr.second_buffer = false;

        Brd.reset_pointers();
        Bwr.reset_pointers();
        resetLevel();
    }

    void buffers_status()
    {
        cout << endl <<"Device used Memory (%): " << std::fixed << std::setprecision(2) << (double)(Bwr.vtail[0]) / Bwr.capacity[0] * 100 << endl;
        Brd.print("Brd: ");
        Bwr.print("Bwr: ");
        if (not H.empty())
        {
            cout << "Host used Memory (%): " << std::fixed << std::setprecision(2) << (double)H.vtail[0] / HOST_BUFF_SZ * 100 << endl;
            H.print("H: ");
        }
    }

    void dump_to_host()
    {
        if (Brd.empty())
            return;

        if (H.empty())
            H.clear();

        const ull src_ohead = Brd.ohead[0];
        const ull src_otail_raw = std::min<ull>(Brd.otail[0], Brd.capacity[0]);
        if (src_otail_raw <= src_ohead)
        {
            return;
        }
        const ull src_otail = src_ohead + ((src_otail_raw - src_ohead) / 2) * 2;
        const ull offset_count = src_otail - src_ohead;
        if (offset_count == 0)
        {
            return;
        }

        ull *offsets = new ull[offset_count];
        chkerr(cudaMemcpy(offsets, Brd.offsets + src_ohead, sizeof(ull) * offset_count, cudaMemcpyDeviceToHost));

        ull valid_offset_count = 0;
        for (ull i = 0; i < offset_count; i += 2)
        {
            if (offsets[i + 1] > offsets[i] && offsets[i + 1] <= Brd.capacity[0])
            {
                offsets[valid_offset_count++] = offsets[i];
                offsets[valid_offset_count++] = offsets[i + 1];
            }
        }

        if (valid_offset_count == 0)
        {
            Brd.ohead[0] = src_otail;
            delete[] offsets;
            return;
        }

        ull min_src_vstart = offsets[0];
        ull max_src_vend = offsets[1];
        ull total_sg_vertices = offsets[1] - offsets[0];
        for (ull i = 2; i < valid_offset_count; i += 2)
        {
            min_src_vstart = std::min(min_src_vstart, offsets[i]);
            max_src_vend = std::max(max_src_vend, offsets[i + 1]);
            total_sg_vertices += offsets[i + 1] - offsets[i];
        }

        const ull dst_vstart = H.vtail[0];
        const ull total_vertices = max_src_vend - min_src_vstart;
        const bool use_bulk_span = total_vertices <= total_sg_vertices * 2;
        cout<<"dumping to host "<<(valid_offset_count / 2)<<endl;

        if (H.otail[0] + valid_offset_count > HOST_OFFSET_SZ)
        {
            delete[] offsets;
            throw std::runtime_error("Host offset buffer overflow");
        }

        if (use_bulk_span)
        {
            if (dst_vstart + total_vertices > HOST_BUFF_SZ)
            {
                delete[] offsets;
                throw std::runtime_error("Host vertex buffer overflow");
            }

            for (ull i = 0; i < valid_offset_count; i += 2)
            {
                const ull sg_st = offsets[i];
                const ull sg_en = offsets[i + 1];

                H.offsets[H.otail[0]] = dst_vstart + (sg_st - min_src_vstart);
                H.offsets[H.otail[0] + 1] = dst_vstart + (sg_en - min_src_vstart);
                H.otail[0] += 2;
            }

            H.copy_host_range(Brd, dst_vstart, min_src_vstart, total_vertices);
            H.vtail[0] += total_vertices;
        }
        else
        {
            for (ull i = 0; i < valid_offset_count; i += 2)
            {
                const ull sg_st = offsets[i];
                const ull sg_en = offsets[i + 1];
                const ull sglen = sg_en - sg_st;

                if (H.vtail[0] + sglen > HOST_BUFF_SZ)
                {
                    delete[] offsets;
                    throw std::runtime_error("Host vertex buffer overflow");
                }

                ull local_dst = H.append_host(sglen);
                H.copy_host_range(Brd, local_dst, sg_st, sglen);
            }
        }
        Brd.ohead[0] = src_otail;
        delete[] offsets;
    }

    void load_from_host()
    {
        if (H.empty())
        {
            H.clear();
            return;
        }

        const ull src_ohead = H.ohead[0];
        const ull src_otail = H.otail[0];
        const ull available_tasks = (src_otail - src_ohead) / 2;

        if (available_tasks == 0)
        {
            return;
        }

        const ull dst_otail = Bwr.otail[0];
        const ull dst_vstart = Bwr.vtail[0];
        ull tasks_to_load = std::min<ull>(std::max<ull>(eta_limit, host_to_gpu_transfer_size_g), available_tasks);
        const ull max_offset_count = tasks_to_load * 2;
        const ull max_offsets_start = src_otail - max_offset_count;
        ull total_vertices = 0;
        ull min_src_vstart = 0;
        ull offset_count = 0;
        ull offsets_start = 0;
        ull local_offsets_start = 0;
        const ull base = Bwr.second_buffer ? Bwr.capacity[0] / 2 : 0;
        const ull span = Bwr.second_buffer ? Bwr.capacity[0] / 2 : Bwr.capacity[0];

        ull *offsets = new ull[max_offset_count];
        std::copy(H.offsets + max_offsets_start, H.offsets + src_otail, offsets);

        while (tasks_to_load > 0)
        {
            offset_count = tasks_to_load * 2;
            offsets_start = src_otail - offset_count;
            local_offsets_start = max_offset_count - offset_count;

            min_src_vstart = offsets[local_offsets_start];
            ull max_src_vend = offsets[local_offsets_start + 1];
            for (ull i = local_offsets_start + 2; i < max_offset_count; i += 2)
            {
                min_src_vstart = std::min(min_src_vstart, offsets[i]);
                max_src_vend = std::max(max_src_vend, offsets[i + 1]);
            }

            total_vertices = max_src_vend - min_src_vstart;
            const ull output_reserve_factor = ping_pong_mode ? 1 : 2;
            if ((dst_otail - base) + output_reserve_factor * offset_count <= span &&
                (dst_vstart - base) + output_reserve_factor * total_vertices <= span)
                break;

            tasks_to_load /= 2;
        }
        if (tasks_to_load == 0)
        {
            delete[] offsets;
            return;
        }
        cout<<"loading from host "<< tasks_to_load <<endl;
        buffers_status();

        ull *translated_offsets = new ull[offset_count];
        ull write_idx = 0;
        for (ull i = max_offset_count; i > local_offsets_start; i -= 2)
        {
            const ull idx = i - 2;
            const ull sg_st = offsets[idx];
            const ull sg_en = offsets[idx + 1];

            translated_offsets[write_idx] = dst_vstart + (sg_st - min_src_vstart);
            translated_offsets[write_idx + 1] = dst_vstart + (sg_en - min_src_vstart);
            write_idx += 2;
        }

        chkerr(cudaMemcpy(Bwr.offsets + dst_otail, translated_offsets, sizeof(ull) * offset_count, cudaMemcpyHostToDevice));
        Bwr.copy_host_to_device_range(H, dst_vstart, min_src_vstart, total_vertices);
        Bwr.otail[0] = dst_otail + offset_count;
        Bwr.vtail[0] = dst_vstart + total_vertices;
        H.otail[0] = offsets_start;
        if (H.empty())
        {
            H.clear();
        }
        else
        {
            ull remaining_vtail = 0;
            for (ull i = H.ohead[0]; i < H.otail[0]; i += 2)
                remaining_vtail = std::max(remaining_vtail, H.offsets[i + 1]);
            H.vtail[0] = remaining_vtail;
        }
        delete[] translated_offsets;
        delete[] offsets;
    }
};

#endif
