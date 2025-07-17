#pragma once

#include "intersection/computesetintersection.h"

#include <atomic>
#include <bitset>
#include <fstream>
#include <unordered_map>
#include <assert.h>
#include <algorithm>

#include "graph_cpu.h"
#include "FilterVertices.h"
#include "GenerateQueryPlan.h"
#include "BuildTable.h"
#include "leapfrogjoin.h"

struct GMContext
{
    ui query_vertices_num;
    ui cur_depth;

    ui *embedding, *idx_embedding;

    GMContext()
    {
        embedding = NULL;
        idx_embedding = NULL;
    }
    ~GMContext()
    {
        if (embedding != NULL)
            delete[] embedding;
        if (idx_embedding != NULL)
            delete[] idx_embedding;
    }
};
// using GMTask = Task<ContextValue>;
typedef Task<GMContext> GMTask;

