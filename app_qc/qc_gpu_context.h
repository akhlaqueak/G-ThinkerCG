#ifndef MC_GPU_APP
#define MC_GPU_APP

#define TEMPSIZE 200'000
#define QBuff_SIZE 100'000'000

class QCBuffer : public BufferBase
{

public:
    Label *labels;

    static ull sizeOf()
    {
        return BufferBase::sizeOf() + sizeof(Label);
    }
    void allocateMemory(ull sz)
    {
        BufferBase::allocateMemory(sz);
        chkerr(cudaMalloc((void **)&labels, sz * sizeof(Label)));
    }

    void copy(auto &src)
    {
        BufferBase::copy(src);
        labels = src.labels;
    }
    __device__ void copy(auto &src, ull i, ull j)
    {
        BufferBase::copy(src, i, j);
        labels[i] = src.labels[j];
    }
    /**
     * @brief This version is used to allocate memory on host. Call it only for HOST_BUFF_SZ
     *
     */
    void allocateMemory()
    {
        BufferBase::allocateMemory();
        chkerr(cudaMallocManaged((void **)&labels, HOST_BUFF_SZ * sizeof(Label)));
    }
};

class QCGPUContext : public GPUContext<QCBuffer, QCTask>
{



public:


    ull get_results()
    {

    }

    virtual void initialize()
    {

    }

    virtual void load_graph(ull *&row_ptrs, VertexID *&cols)
    {
    }

    __device__ virtual void generateInitialTasks(VertexID *sources, ull *sources_num, ull *v_proc, QCBuffer &Bwr, ull *row_ptrs, VertexID *cols)
    {

    }
    __device__ bool isLevelFilledQ()
    {

    }

public:
    __device__ virtual void process(QCBuffer &Brd, ull *row_ptrs, VertexID *cols)
    {

    }
    virtual void init_level()
    {

    }
    __device__ virtual void extend(QCBuffer &Brd, QCBuffer &Bwr, QCBuffer &H, ull *row_ptrs, VertexID *cols)

    {

    }

 
    virtual void move_tasks_from_Sc(std::vector<QCTask *> &src_tasks, QCBuffer &H)
    {

    }
    virtual void move_tasks_to_Sc(vector<QCTask *> &collector, QCBuffer &H)
    {

    }
};
#endif