#ifndef APP_QC_QC_TASK_H
#define APP_QC_QC_TASK_H

#include <utility>

struct QCContext
{
    Vertex *vertices;
    size_t num_vertices;

    QCContext()
        : vertices(nullptr), num_vertices(0)
    {
    }

    QCContext(size_t sz) : QCContext()
    {
        allocate(sz);
    }

    QCContext(const QCContext&) = delete;
    QCContext& operator=(const QCContext&) = delete;

    QCContext(QCContext&& other) noexcept
        : vertices(other.vertices), num_vertices(other.num_vertices)
    {
        other.vertices = nullptr;
        other.num_vertices = 0;
    }

    QCContext& operator=(QCContext&& other) noexcept
    {
        if (this != &other)
        {
            release();
            vertices = other.vertices;
            num_vertices = other.num_vertices;
            other.vertices = nullptr;
            other.num_vertices = 0;
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
    }

    void release()
    {
        delete[] vertices;
        vertices = nullptr;
        num_vertices = 0;
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
#endif
