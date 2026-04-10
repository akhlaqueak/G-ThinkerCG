#pragma once

struct QCContext
{
    int vertexid;
    // labels: 0 -> candidate, 1 -> member, 2 -> covered vertex, 3 -> cover vertex, 4 -> critical adjacent vertex
    int label;
    int indeg;
    int exdeg;
    int lvl2adj;
};

using QCTask = Task<QCContext>;
