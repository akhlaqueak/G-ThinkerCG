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
//==== global resources held for query ===

Graph_CPU cpu_qg;
Graph_CPU cpu_dg;
Graph gpu_dg;
Graph gpu_qg;
Plan plan;

vector<unsigned long long int> counters;

ui **candidates;
ui *candidates_count;
ui *bfs_order, *matching_order, *pivot;
TreeNode *tree;

Edges ***edge_matrix;

ui **bn;
ui *bn_count;

ui max_candidate_cnt;
ui gm_prefix_batch_size_g = 100;
bool gm_cpu_shared_intersection_g = false;
bool gm_cpu_gpu_style_expand_g = false;
inline constexpr ui GM_QUERY_EMBEDDING_CAP = 8;


int binary_search(ui* a, ui length, ui target) {
    int left = 0;
    int right = length - 1;

    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (a[mid] == target) {
            return mid;
        } else if (a[mid] < target) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    return -1; // not found
}

int binary_search(ui cur_depth, ui data)
{
    ui vertexid = matching_order[cur_depth];
    ui num_candidate = candidates_count[vertexid];
    return binary_search(candidates[vertexid], num_candidate, data); 
}

//==== global resources held for query ===

struct GMContext
{
    ui query_vertices_num;
    ui cur_depth;

    ui embedding_storage[GM_QUERY_EMBEDDING_CAP];
    ui idx_embedding_storage[GM_QUERY_EMBEDDING_CAP];
    ui *embedding, *idx_embedding;
    ui *prefix_candidate_idx;
    ui prefix_candidate_count;

    GMContext()
    {
        embedding = embedding_storage;
        idx_embedding = idx_embedding_storage;
        prefix_candidate_idx = NULL;
        prefix_candidate_count = 0;
    }
    ~GMContext()
    {
        if (prefix_candidate_idx != NULL)
            delete[] prefix_candidate_idx;
    }
};
// using GMTask = Task<ContextValue>;
typedef Task<GMContext> GMTask;
