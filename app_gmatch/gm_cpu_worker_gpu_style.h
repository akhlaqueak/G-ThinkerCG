#ifndef GM_CPU_GPU_STYLE_APP
#define GM_CPU_GPU_STYLE_APP

#include "graph_cpu.h"
#include "FilterVertices.h"
#include "GenerateQueryPlan.h"
#include "BuildTable.h"
#include "leapfrogjoin.h"
#include "intersection/computesetintersection.h"

// Experimental CPU worker that follows the GPU matching logic more directly.
// It keeps CPU-friendly DFS recursion, but candidate generation mirrors the GPU:
// 1. choose the minimum back-neighbor list as the seed,
// 2. probe the remaining back-neighbors with membership checks,
// 3. for prefix tasks with shareIntersection enabled, build a common
//    pre-intersection once and refine it per batched candidate.
class GMCPUWorkerGPUStyle : public CPUWorker<GMTask>
{
public:
    unsigned long long int counter = 0;
    bool *visited_arr = nullptr;
    std::chrono::time_point<std::chrono::steady_clock> st;

    GMCPUWorkerGPUStyle() : CPUWorker<GMTask>()
    {
        visited_arr = new bool[cpu_dg.getVerticesCount()];
        memset(visited_arr, false, sizeof(bool) * cpu_dg.getVerticesCount());
    }

    ~GMCPUWorkerGPUStyle() override
    {
        delete[] visited_arr;
    }

    virtual GMTask *task_spawn(VertexID &data)
    {
        GMTask *t = new GMTask();
        t->context.query_vertices_num = cpu_qg.getVerticesCount();
        t->context.cur_depth = 1;
        t->context.embedding[matching_order[0]] = data;
        t->context.idx_embedding[matching_order[0]] = binary_search(0, data);
        return t;
    }

    virtual void compute(GMContext &context)
    {
        st = start_task_timer();

        if (context.prefix_candidate_idx != nullptr)
            process_prefix_task(context);
        else
            search_expand(context.cur_depth, context.embedding, context.idx_embedding, true);
    }

private:
    bool candidateSatisfiesConditions(ui query_vertex, ui vertex, ui *cond_order, ui cond_count, ui *embedding)
    {
        for (ui k = 0; k < cond_count; ++k)
        {
            ui cond = cond_order[query_vertex * plan.sz * 2 + 2 * k];
            ui cond_vertex = cond_order[query_vertex * plan.sz * 2 + 2 * k + 1];
            ui cond_vertex_M = embedding[cond_vertex];
            if (cond == CondOperator::LESS_THAN)
            {
                if (cond_vertex_M <= vertex)
                    return false;
            }
            else if (cond == CondOperator::LARGER_THAN)
            {
                if (cond_vertex_M >= vertex)
                    return false;
            }
            else if (cond == CondOperator::NON_EQUAL)
            {
                if (cond_vertex_M == vertex)
                    return false;
            }
        }
        return true;
    }

    bool dataGraphHasEdge(ui src, ui dst)
    {
        ui nbr_cnt = 0;
        const ui *nbrs = cpu_dg.getVertexNeighbors(src, nbr_cnt);
        return binary_search(const_cast<ui *>(nbrs), nbr_cnt, dst) != -1;
    }

    void spawn_split_task_gpu_style(ui cur_depth, ui *embedding, ui *idx_embedding, ui *order, ui query_vertices_num)
    {
        GMTask *t = new GMTask();
        t->context.query_vertices_num = query_vertices_num;
        t->context.cur_depth = cur_depth + 1;

        for (ui i = 0; i <= cur_depth; ++i)
        {
            ui u = order[i];
            t->context.embedding[u] = embedding[u];
            t->context.idx_embedding[u] = idx_embedding[u];
        }

        add_task(t);
    }

    void spawn_prefix_task_gpu_style(ui cur_depth, ui *embedding, ui *idx_embedding,
                                     const std::vector<ui> &candidate_indices, size_t start_pos)
    {
        const ui remaining = static_cast<ui>(candidate_indices.size() - start_pos);
        if (remaining == 0)
            return;

        for (size_t batch_begin = start_pos; batch_begin < candidate_indices.size(); batch_begin += gm_prefix_batch_size_g)
        {
            ui batch_size = std::min<ui>(gm_prefix_batch_size_g,
                                         static_cast<ui>(candidate_indices.size() - batch_begin));

            GMTask *t = new GMTask();
            t->context.query_vertices_num = cpu_qg.getVerticesCount();
            t->context.cur_depth = cur_depth + 1;
            t->context.prefix_candidate_idx = new ui[batch_size];
            t->context.prefix_candidate_count = batch_size;

            for (ui i = 0; i < cur_depth; ++i)
            {
                ui u = matching_order[i];
                t->context.embedding[u] = embedding[u];
                t->context.idx_embedding[u] = idx_embedding[u];
            }

            memcpy(t->context.prefix_candidate_idx,
                   candidate_indices.data() + batch_begin,
                   batch_size * sizeof(ui));

            add_task(t);
        }
    }

    void collect_gpu_style_candidate_indices(ui depth, ui *embedding, std::vector<ui> &out_indices)
    {
        out_indices.clear();
        ui u = matching_order[depth];
        ui bnCount = plan.backNeighborCountHost[depth];
        if (bnCount == 0)
            return;

        ui parent_u = plan.backNeighborsHost[depth * plan.sz];
        ui parent_u_M = embedding[parent_u];
        ui parent_deg = 0;
        const ui *parent_neighbors = cpu_dg.getVertexNeighbors(parent_u_M, parent_deg);

        for (ui i = 1; i < bnCount; ++i)
        {
            ui u_prime = plan.backNeighborsHost[depth * plan.sz + i];
            ui u_prime_M = embedding[u_prime];
            ui deg = 0;
            const ui *nbrs = cpu_dg.getVertexNeighbors(u_prime_M, deg);
            if (deg < parent_deg)
            {
                parent_u = u_prime;
                parent_u_M = u_prime_M;
                parent_deg = deg;
                parent_neighbors = nbrs;
            }
        }

        ui condCount = plan.condNumHost[u];
        for (ui i = 0; i < parent_deg; ++i)
        {
            ui vertex = parent_neighbors[i];
            if (visited_arr[vertex])
                continue;
            if (!candidateSatisfiesConditions(u, vertex, plan.condOrderHost, condCount, embedding))
                continue;

            bool pred = true;
            for (ui j = 0; j < bnCount; ++j)
            {
                ui u_prime = plan.backNeighborsHost[depth * plan.sz + j];
                if (u_prime == parent_u)
                    continue;
                ui u_prime_M = embedding[u_prime];
                if (!dataGraphHasEdge(u_prime_M, vertex))
                {
                    pred = false;
                    break;
                }
            }
            if (!pred)
                continue;

            int idx = binary_search(candidates[u], candidates_count[u], vertex);
            if (idx != -1)
                out_indices.push_back(static_cast<ui>(idx));
        }
    }

    void build_shared_prefix_preintersection(ui target_depth, ui *embedding, std::vector<ui> &pre_vertices)
    {
        pre_vertices.clear();
        ui target_u = matching_order[target_depth];
        ui bnCount = plan.preBackNeighborCountHost[target_depth];
        if (bnCount == 0)
            return;

        ui parent_u = plan.preBackNeighborsHost[target_depth * plan.sz];
        ui parent_u_M = embedding[parent_u];
        ui parent_deg = 0;
        const ui *parent_neighbors = cpu_dg.getVertexNeighbors(parent_u_M, parent_deg);

        for (ui i = 1; i < bnCount; ++i)
        {
            ui u_prime = plan.preBackNeighborsHost[target_depth * plan.sz + i];
            ui u_prime_M = embedding[u_prime];
            ui deg = 0;
            const ui *nbrs = cpu_dg.getVertexNeighbors(u_prime_M, deg);
            if (deg < parent_deg)
            {
                parent_u = u_prime;
                parent_u_M = u_prime_M;
                parent_deg = deg;
                parent_neighbors = nbrs;
            }
        }

        ui condCount = plan.preCondNumHost[target_u];
        for (ui i = 0; i < parent_deg; ++i)
        {
            ui vertex = parent_neighbors[i];
            if (visited_arr[vertex])
                continue;
            if (!candidateSatisfiesConditions(target_u, vertex, plan.preCondOrderHost, condCount, embedding))
                continue;

            bool pred = true;
            for (ui j = 0; j < bnCount; ++j)
            {
                ui u_prime = plan.preBackNeighborsHost[target_depth * plan.sz + j];
                if (u_prime == parent_u)
                    continue;
                ui u_prime_M = embedding[u_prime];
                if (!dataGraphHasEdge(u_prime_M, vertex))
                {
                    pred = false;
                    break;
                }
            }
            if (pred)
                pre_vertices.push_back(vertex);
        }
    }

    void collect_after_prefix_candidates(ui target_depth, ui *embedding,
                                         const std::vector<ui> &pre_vertices,
                                         std::vector<ui> &candidate_indices)
    {
        candidate_indices.clear();
        ui target_u = matching_order[target_depth];
        ui condCount = plan.afterCondNumHost[target_u];
        ui bnCount = plan.afterBackNeighborCountHost[target_depth];

        for (ui vertex : pre_vertices)
        {
            if (visited_arr[vertex])
                continue;
            if (!candidateSatisfiesConditions(target_u, vertex, plan.afterCondOrderHost, condCount, embedding))
                continue;

            bool pred = true;
            for (ui j = 0; j < bnCount; ++j)
            {
                ui u_prime = plan.afterBackNeighborsHost[target_depth * plan.sz + j];
                ui u_prime_M = embedding[u_prime];
                if (!dataGraphHasEdge(u_prime_M, vertex))
                {
                    pred = false;
                    break;
                }
            }
            if (!pred)
                continue;

            int idx = binary_search(candidates[target_u], candidates_count[target_u], vertex);
            if (idx != -1)
                candidate_indices.push_back(static_cast<ui>(idx));
        }
    }

    void search_candidate_list(ui depth, ui *embedding, ui *idx_embedding,
                               const std::vector<ui> &candidate_indices, bool allow_prefix_split)
    {
        ui u = matching_order[depth];
        ui max_depth = cpu_qg.getVerticesCount();

        for (size_t pos = 0; pos < candidate_indices.size(); ++pos)
        {
            if (allow_prefix_split &&
                depth + 1 < max_depth &&
                plan.strategyHost[depth + 1] == StoreStrategy::PREFIX &&
                time_over(st))
            {
                spawn_prefix_task_gpu_style(depth, embedding, idx_embedding, candidate_indices, pos);
                return;
            }

            ui valid_idx = candidate_indices[pos];
            ui v = candidates[u][valid_idx];
            if (visited_arr[v])
                continue;

            embedding[u] = v;
            idx_embedding[u] = valid_idx;
            visited_arr[v] = true;

            if (depth == max_depth - 1)
            {
                counter += 1;
                visited_arr[v] = false;
                continue;
            }

            if (!time_over(st))
            {
                search_expand(depth + 1, embedding, idx_embedding, true);
            }
            else
            {
                spawn_split_task_gpu_style(depth, embedding, idx_embedding, matching_order, cpu_qg.getVerticesCount());
            }

            visited_arr[v] = false;
        }
    }

    void search_expand(ui depth, ui *embedding, ui *idx_embedding, bool allow_prefix_split)
    {
        std::vector<ui> candidate_indices;
        collect_gpu_style_candidate_indices(depth, embedding, candidate_indices);
        search_candidate_list(depth, embedding, idx_embedding, candidate_indices, allow_prefix_split);
    }

    void process_prefix_task(GMContext &context)
    {
        ui sglen = context.cur_depth;
        if (sglen == 0)
            return;

        ui pending_depth = sglen - 1;
        for (ui i = 0; i < pending_depth; ++i)
            visited_arr[context.embedding[matching_order[i]]] = true;

        if (context.cur_depth < cpu_qg.getVerticesCount() && plan.shareIntersectionHost[context.cur_depth])
        {
            ui target_depth = context.cur_depth;
            std::vector<ui> pre_vertices;
            build_shared_prefix_preintersection(target_depth, context.embedding, pre_vertices);

            ui current_u = matching_order[pending_depth];
            for (ui i = 0; i < context.prefix_candidate_count; ++i)
            {
                ui candidate_idx = context.prefix_candidate_idx[i];
                ui v = candidates[current_u][candidate_idx];
                if (visited_arr[v])
                    continue;

                context.embedding[current_u] = v;
                context.idx_embedding[current_u] = candidate_idx;
                visited_arr[v] = true;

                std::vector<ui> next_candidate_indices;
                collect_after_prefix_candidates(target_depth, context.embedding, pre_vertices, next_candidate_indices);
                search_candidate_list(target_depth, context.embedding, context.idx_embedding, next_candidate_indices, true);

                visited_arr[v] = false;
            }
        }
        else
        {
            std::vector<ui> candidate_indices(context.prefix_candidate_idx,
                                              context.prefix_candidate_idx + context.prefix_candidate_count);
            search_candidate_list(pending_depth, context.embedding, context.idx_embedding, candidate_indices, false);
        }

        for (ui i = 0; i < pending_depth; ++i)
            visited_arr[context.embedding[matching_order[i]]] = false;
    }
};

#endif
