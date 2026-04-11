#ifndef APP_QC_QC_TASK_H
#define APP_QC_QC_TASK_H

struct QCContext
{
    Vertex *vertices;
    size_t num_vertices;
    QCContext() : vertices(nullptr) {}
    QCContext(size_t sz) : vertices(new Vertex[sz]) {}

    QCContext(const QCContext&) = delete;
    QCContext& operator=(const QCContext&) = delete;

    ~QCContext()
    {
        delete[] vertices;
        vertices = nullptr;
    }
};


using QCTask = Task<QCContext>;

extern CPU_Data hd;
extern GPU_Data dd;

#endif
