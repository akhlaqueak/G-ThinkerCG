#ifndef APP_QC_QC_TASK_H
#define APP_QC_QC_TASK_H

#include <atomic>
#include <utility>

struct QCContext
{
    Vertex *vertices;
    size_t num_vertices;
    bool from_big_root;

    QCContext()
        : vertices(nullptr), num_vertices(0), from_big_root(false)
    {
    }

    QCContext(size_t sz) : QCContext()
    {
        allocate(sz);
    }

    QCContext(const QCContext&) = delete;
    QCContext& operator=(const QCContext&) = delete;

    QCContext(QCContext&& other) noexcept
        : vertices(other.vertices), num_vertices(other.num_vertices), from_big_root(other.from_big_root)
    {
        other.vertices = nullptr;
        other.num_vertices = 0;
        other.from_big_root = false;
    }

    QCContext& operator=(QCContext&& other) noexcept
    {
        if (this != &other)
        {
            release();
            vertices = other.vertices;
            num_vertices = other.num_vertices;
            from_big_root = other.from_big_root;
            other.vertices = nullptr;
            other.num_vertices = 0;
            other.from_big_root = false;
        }
        return *this;
    }

    void allocate(size_t sz)
    {
        release();
        vertices = new Vertex[sz];
        num_vertices = sz;
    }

    QCContext(Vertex *src, size_t sz)
    {
        vertices = src;
        num_vertices = sz;
        from_big_root = false;
    }

    void release()
    {
        delete[] vertices;
        vertices = nullptr;
        num_vertices = 0;
        from_big_root = false;
    }

    ~QCContext()
    {
        release();
    }
};


using QCTask = Task<QCContext>;

extern CPU_Data hd;
extern GPU_Data dd;
extern CPU_Cliques hc;
extern CPU_Graph *hg;
extern std::atomic<uint64_t> qc_big_root_tasks_spawned;
extern std::atomic<uint64_t> qc_big_root_tasks_executed;
extern std::atomic<uint64_t> qc_big_root_cliques_found;
#endif
