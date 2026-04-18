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
    // using ContextT = typename TaskT::ContextType;
    // using TaskType = TaskT;
    typedef typename TaskT::ContextType ContextT;
    typedef TaskT TaskType;
    using TaskContainer = vector<TaskT *>;
    ull *v_proc;
    std::stack<BPointers> SL;
    bool ping_pong_mode = true;

    // memory is allocated only to B and H buffers, Brd, Bwr are just pointers hovering over B
    // host spill buffer
    BufferT B;
    BufferT H;

    BufferT Bwr;
    BufferT Brd;

    ull *sources_num;  // size of Lv
    VertexID *sources; // Lv copy on GPU

    // graph in CSR on GPU
    ull *row_ptrs;
    VertexID *cols;

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

    bool topLevelWorkExist()
    {
        return v_proc[0] < sources_num[0];
    }

    void allocateMemory(ull reserved_mem = 0)
    {
        initialize(); // call UDF first, so that application specific memory is allocated
        chkerr(cudaMallocManaged((void **)&v_proc, sizeof(ull)));
        chkerr(cudaMalloc(&sources, sizeof(VertexID) * tasks_per_fetch_gpu_worker_g));
        chkerr(cudaMallocManaged((void **)&sources_num, sizeof(ull)));

        size_t total, free;
        cudaMemGetInfo(&free, &total);
        cout << "Available Memory " << free / 1024.0 / 1024 / 1024 << " GB" << endl;
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

    void set_ping_pong_mode()
    {
        ping_pong_mode = true;  

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
        cout << "Device used Memory (%): " << std::fixed << std::setprecision(2) << (double)(Bwr.vtail[0]) / Bwr.capacity[0] * 100 << endl;
        Brd.print("Brd: ");
        Bwr.print("Bwr: ");
        if (not H.empty())
        {
            cout << "Host used Memory (%): " << std::fixed << std::setprecision(2) << (double)H.vtail[0] / HOST_BUFF_SZ * 100 << endl;
            // cout << "Host tasks: " << H.otail[0] - H.ohead[0] << endl;
            H.print("H: ");
        }
    }

    __device__ void loadFromHost()
    {
        while (true)
        {
            if (H.n_tasks_proc[0] > eta)
                return;
            auto so = H.pop();
            if (so.empty())
                return;
            ull vt = Bwr.append(so);
            for (ull i = vt + LANEID, j = so.st + LANEID; j < so.en; i += 32, j += 32)
                Bwr.copy(H, i, j);
        }
    }

    __device__ void dumpToHost(SubgraphOffsets &so)
    {
        ull vt = H.append(so);
        for (ull i = vt + LANEID, j = so.st + LANEID; j < so.en; i += 32, j += 32)
            H.copy(Brd, i, j);
    }

    __device__ void dumpToHost()
    {
        while (true)
        {
            auto so = Brd.next();
            if (so.empty())
                return;
            ull sglen = so.en - so.st;
            ull vt = H.append(sglen);
            for (ull i = vt + LANEID, j = so.st + LANEID; j < so.en; i += 32, j += 32)
                H.copy(Brd, i, j);
        }
    }
};

#endif
