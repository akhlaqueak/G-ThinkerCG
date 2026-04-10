#pragma once

struct Vertex
{
    int vertexid;
    // labels: 0 -> candidate, 1 -> member, 2 -> covered vertex, 3 -> cover vertex, 4 -> critical adjacent vertex
    int label;
    int indeg;
    int exdeg;
    int lvl2adj;
};

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
