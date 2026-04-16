#ifndef APP_QC_QC_TASK_H
#define APP_QC_QC_TASK_H

#include <utility>

struct QCContextSoA
{
    VertexID *vertices;
    Label *label;
    int *indeg;
    int *exdeg;
    int *lvl2adj;
    size_t num_vertices;

    QCContextSoA()
        : vertices(nullptr), label(nullptr), indeg(nullptr), exdeg(nullptr), lvl2adj(nullptr), num_vertices(0)
    {
    }

    QCContextSoA(size_t sz) : QCContextSoA()
    {
        allocate(sz);
    }

    QCContextSoA(const QCContextSoA&) = delete;
    QCContextSoA& operator=(const QCContextSoA&) = delete;

    void allocate(size_t sz)
    {
        release();
        vertices = new VertexID[sz];
        label = new Label[sz];
        indeg = new int[sz];
        exdeg = new int[sz];
        lvl2adj = new int[sz];
        num_vertices = sz;
    }

    QCContextSoA(const Vertex *src, size_t sz)
    {
        allocate(sz);
        for (size_t i = 0; i < sz; i++)
        {
            vertices[i] = src[i].vertexid;
            label[i] = src[i].label;
            indeg[i] = src[i].indeg;
            exdeg[i] = src[i].exdeg;
            lvl2adj[i] = src[i].lvl2adj;
        }
    }

    void release()
    {
        delete[] vertices;
        delete[] label;
        delete[] indeg;
        delete[] exdeg;
        delete[] lvl2adj;
        vertices = nullptr;
        label = nullptr;
        indeg = nullptr;
        exdeg = nullptr;
        lvl2adj = nullptr;
        num_vertices = 0;
    }

    ~QCContextSoA()
    {
        release();
    }
};

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
