#ifndef GM_CPU_APP
#define GM_CPU_APP

#include "graph_cpu.h"
#include "FilterVertices.h"
#include "GenerateQueryPlan.h"
#include "BuildTable.h"
#include "leapfrogjoin.h"
#include "intersection/computesetintersection.h"


class GMCPUWorker : public CPUWorker<GMTask>
{
public:
    // ui max_sz = 0;
    // ui total_counts=0;

    // ====== auxiliary arrays in computation =====
    unsigned long long int counter = 0;

    ui *temp_buffer = nullptr;
    bool *visited_arr = nullptr;
    ui *idx = nullptr;
    ui *idx_count = nullptr;
    ui **valid_candidate_idx = nullptr;
    std::vector<ui> *shared_pre_vertices_cache = nullptr;
    bool *shared_pre_vertices_cache_valid = nullptr;
    ui *shared_pre_vertices_cache_owner = nullptr;

    struct timeb thread_local_time;

    std::chrono::time_point<std::chrono::steady_clock> st;

    // =============================================

    GMCPUWorker() : CPUWorker<GMTask>()
    {
        temp_buffer = new ui[max_candidate_cnt];

        visited_arr = new bool[cpu_dg.getVerticesCount()];
        memset(visited_arr, false, sizeof(bool) * cpu_dg.getVerticesCount());

        idx = new ui[cpu_qg.getVerticesCount()];
        idx_count = new ui[cpu_qg.getVerticesCount()];

        valid_candidate_idx = new ui *[cpu_qg.getVerticesCount()];
        for (ui i = 0; i < cpu_qg.getVerticesCount(); ++i)
            valid_candidate_idx[i] = new ui[max_candidate_cnt];
        shared_pre_vertices_cache = new std::vector<ui>[cpu_qg.getVerticesCount()];
        shared_pre_vertices_cache_valid = new bool[cpu_qg.getVerticesCount()];
        std::fill(shared_pre_vertices_cache_valid, shared_pre_vertices_cache_valid + cpu_qg.getVerticesCount(), false);
        shared_pre_vertices_cache_owner = new ui[cpu_qg.getVerticesCount() * GM_QUERY_EMBEDDING_CAP];
        std::fill(shared_pre_vertices_cache_owner,
                  shared_pre_vertices_cache_owner + cpu_qg.getVerticesCount() * GM_QUERY_EMBEDDING_CAP,
                  std::numeric_limits<ui>::max());
        for (ui i = 0; i < cpu_qg.getVerticesCount(); ++i)
            shared_pre_vertices_cache[i].reserve(max_candidate_cnt);
    }

    ~GMCPUWorker() override
    {
        delete[] temp_buffer;
        delete[] visited_arr;
        delete[] idx;
        delete[] idx_count;
        if (valid_candidate_idx != nullptr)
        {
            for (ui i = 0; i < cpu_qg.getVerticesCount(); ++i)
                delete[] valid_candidate_idx[i];
            delete[] valid_candidate_idx;
        }
        delete[] shared_pre_vertices_cache;
        delete[] shared_pre_vertices_cache_valid;
        delete[] shared_pre_vertices_cache_owner;
    }

    virtual GMTask *task_spawn(VertexID &data)
    {
        GMTask *t = new GMTask();
        t->context.query_vertices_num = cpu_qg.getVerticesCount();
        t->context.cur_depth = 1;

        // set data and its index
        t->context.embedding[matching_order[0]] = data;
        t->context.idx_embedding[matching_order[0]] = binary_search(0, data); // [10,22]

        return t;
    }

    virtual void compute(GMContext &context)
    {
        ftime(&thread_local_time);
        st = now();

        if (context.prefix_candidate_idx != nullptr)
            process_prefix_task(context);
        else
            LFTJ(context.cur_depth, cpu_qg, edge_matrix, candidates, candidates_count, matching_order,
                 context.embedding, context.idx_embedding, bn, bn_count);
    }

    double countElaspedTime()
    {
        struct timeb cur_time;
        ftime(&cur_time);
        // return (double)(cur_time.millitm - thread_local_time.millitm);
        return cur_time.time - thread_local_time.time + (double)(cur_time.millitm - thread_local_time.millitm)/1000;
    }

    void spawn_split_task(ui cur_depth, ui *embedding, ui *idx_embedding, ui *order, ui query_vertices_num)
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

    void spawn_prefix_task(ui cur_depth, ui *embedding, ui *idx_embedding, ui *order, ui query_vertices_num)
    {
        ui remaining = idx_count[cur_depth] - idx[cur_depth];
        if (remaining == 0)
            return;

        for (ui batch_begin = idx[cur_depth]; batch_begin < idx_count[cur_depth]; batch_begin += gm_prefix_batch_size_g)
        {
            ui batch_size = std::min<ui>(gm_prefix_batch_size_g, idx_count[cur_depth] - batch_begin);

            GMTask *t = new GMTask();
            t->context.query_vertices_num = query_vertices_num;
            t->context.cur_depth = cur_depth + 1;
            t->context.prefix_candidate_idx = new ui[batch_size];
            t->context.prefix_candidate_count = batch_size;

            for (ui i = 0; i < cur_depth; ++i)
            {
                ui u = order[i];
                t->context.embedding[u] = embedding[u];
                t->context.idx_embedding[u] = idx_embedding[u];
            }

            memcpy(t->context.prefix_candidate_idx,
                   valid_candidate_idx[cur_depth] + batch_begin,
                   batch_size * sizeof(ui));

            add_task(t);
        }
    }

    void generateValidCandidateIndex(ui depth, ui *embedding, ui *idx_embedding, ui *idx_count, ui **valid_candidate_index,
                                    Edges ***edge_matrix, ui **bn, ui *bn_cnt, ui *order, ui *temp_buffer_, ui **candidates)
    {   

        ui u = order[depth];
        ui previous_bn = bn[depth][0];
        ui previous_index_id = idx_embedding[previous_bn];
        ui valid_candidates_count = 0;


        Edges& previous_edge = *edge_matrix[previous_bn][u];

        valid_candidates_count = previous_edge.offset_[previous_index_id + 1] - previous_edge.offset_[previous_index_id];
        ui* previous_candidates = previous_edge.edge_ + previous_edge.offset_[previous_index_id];

        ui *current_buffer = valid_candidate_index[depth];
        ui *next_buffer = temp_buffer_;
        memcpy(current_buffer, previous_candidates, valid_candidates_count * sizeof(ui));

        ui temp_count = 0;
        for (ui i = 1; i < bn_cnt[depth]; ++i) {
            
            VertexID current_bn = bn[depth][i];

            Edges& current_edge = *edge_matrix[current_bn][u];
            ui current_index_id = idx_embedding[current_bn];


            ui current_candidates_count = current_edge.offset_[current_index_id + 1] - current_edge.offset_[current_index_id];

            ui* current_candidates = current_edge.edge_ + current_edge.offset_[current_index_id];

            if (current_candidates_count < valid_candidates_count)
                ComputeSetIntersection::ComputeCandidates(current_candidates, current_candidates_count, current_buffer, valid_candidates_count,
                            next_buffer, temp_count);
            else
                ComputeSetIntersection::ComputeCandidates(current_buffer, valid_candidates_count, current_candidates, current_candidates_count,
                            next_buffer, temp_count);
            valid_candidates_count = temp_count;

            if (valid_candidates_count == 0)
                break;

            std::swap(current_buffer, next_buffer);
        }

        // ====================================================
        ui condCount = plan.condNumHost[u];
        ui tmp_len = 0;
        for (ui i = 0; i < valid_candidates_count; ++i) {
            ui valid_index = current_buffer[i];
            ui vertex = candidates[u][valid_index];
            bool pred = true;

            for (ui k = 0; k < condCount; ++k)
            {
                ui cond = plan.condOrderHost[u * plan.sz * 2 + 2 * k];
                ui cond_vertex = plan.condOrderHost[u * plan.sz * 2 + 2 * k + 1];
                ui cond_vertex_M = embedding[cond_vertex];
                if (cond == CondOperator::LESS_THAN)
                {
                    if (cond_vertex_M <= vertex)
                    {
                        pred = false;
                        break;
                    }
                }
                else if (cond == CondOperator::LARGER_THAN)
                {
                    if (cond_vertex_M >= vertex)
                    {
                        pred = false;
                        break;
                    }
                }
                else if (cond == CondOperator::NON_EQUAL)
                {
                    if (cond_vertex_M == vertex)
                    {
                        pred = false;
                        break;
                    }
                }
            }

            if (pred)
                valid_candidate_index[depth][tmp_len++] = valid_index;
        }

        // idx_count[depth] = valid_candidates_count;
        idx_count[depth] = tmp_len;
    }

    void generateValidCandidateIndexGPUStyle(ui depth, ui *embedding, ui *idx_embedding, ui *idx_count,
                                             ui **valid_candidate_index, ui *order, ui **candidates, ui *candidates_count)
    {
        ui u = order[depth];
        ui bnCount = plan.backNeighborCountHost[depth];
        if (bnCount == 0)
        {
            idx_count[depth] = 0;
            return;
        }

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
                parent_deg = deg;
                parent_neighbors = nbrs;
            }
        }

        ui condCount = plan.condNumHost[u];
        ui out_count = 0;
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
                valid_candidate_index[depth][out_count++] = static_cast<ui>(idx);
        }

        idx_count[depth] = out_count;
    }

    bool sharedPreCacheMatches(ui target_depth, ui *embedding, ui *order)
    {
        if (!shared_pre_vertices_cache_valid[target_depth])
            return false;

        for (ui i = 0; i + 1 < target_depth; ++i)
        {
            ui u = order[i];
            if (shared_pre_vertices_cache_owner[target_depth * GM_QUERY_EMBEDDING_CAP + i] != embedding[u])
                return false;
        }
        return true;
    }

    void saveSharedPreCacheOwner(ui target_depth, ui *embedding, ui *order)
    {
        for (ui i = 0; i < GM_QUERY_EMBEDDING_CAP; ++i)
            shared_pre_vertices_cache_owner[target_depth * GM_QUERY_EMBEDDING_CAP + i] = std::numeric_limits<ui>::max();

        for (ui i = 0; i + 1 < target_depth; ++i)
        {
            ui u = order[i];
            shared_pre_vertices_cache_owner[target_depth * GM_QUERY_EMBEDDING_CAP + i] = embedding[u];
        }
        shared_pre_vertices_cache_valid[target_depth] = true;
    }

    const std::vector<ui> &getSharedPrefixPreintersection(ui target_depth, ui *embedding, ui *order)
    {
        if (!sharedPreCacheMatches(target_depth, embedding, order))
        {
            bool ignore_visited_vertex = target_depth > 0;
            ui ignored_vertex = ignore_visited_vertex ? embedding[order[target_depth - 1]] : std::numeric_limits<ui>::max();
            build_shared_prefix_preintersection(target_depth, embedding, shared_pre_vertices_cache[target_depth],
                                                ignore_visited_vertex, ignored_vertex);
            saveSharedPreCacheOwner(target_depth, embedding, order);
        }
        return shared_pre_vertices_cache[target_depth];
    }

    void collect_after_prefix_candidates_to_buffer(ui target_depth, ui *embedding,
                                                   const std::vector<ui> &pre_vertices,
                                                   ui *out_indices, ui &out_count)
    {
        out_count = 0;
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
                out_indices[out_count++] = static_cast<ui>(idx);
        }
    }

    void generateValidCandidateIndexShared(ui depth, ui *embedding, ui *idx_count,
                                           ui **valid_candidate_index, ui *order)
    {
        const std::vector<ui> &pre_vertices = getSharedPrefixPreintersection(depth, embedding, order);
        collect_after_prefix_candidates_to_buffer(depth, embedding, pre_vertices,
                                                  valid_candidate_index[depth], idx_count[depth]);
    }

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

    void build_shared_prefix_preintersection(ui target_depth, ui *embedding, std::vector<ui> &pre_vertices,
                                             bool ignore_visited_vertex, ui ignored_vertex)
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
            if (visited_arr[vertex] && (!ignore_visited_vertex || vertex != ignored_vertex))
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

    void LFTJ(int enter_depth, Graph_CPU &cpu_qg, Edges ***edge_matrix, ui **candidates,
                ui *candidates_count, ui *order, ui *embedding, ui *idx_embedding,
                ui **bn, ui *bn_count)
    {
        int cur_depth = enter_depth;
        int max_depth = cpu_qg.getVerticesCount();

        if (cur_depth == 0)
        {
            ui start_vertex = order[0];

            idx[cur_depth] = 0;

            idx_count[cur_depth] = candidates_count[start_vertex];

            for (ui i = 0; i < idx_count[cur_depth]; ++i) {
                valid_candidate_idx[cur_depth][i] = i;
            }
        }
        else
        {  
            idx[cur_depth] = 0;

            for (ui i = 0; i < enter_depth; ++i)
            {
                visited_arr[embedding[order[i]]] = true;
            }

            if (gm_cpu_shared_intersection_g && plan.shareIntersectionHost[cur_depth])
                generateValidCandidateIndexShared(cur_depth, embedding, idx_count, valid_candidate_idx, order);
            else if (gm_cpu_gpu_style_expand_g)
                generateValidCandidateIndexGPUStyle(cur_depth, embedding, idx_embedding, idx_count, valid_candidate_idx, order, candidates, candidates_count);
            else
                generateValidCandidateIndex(cur_depth, embedding, idx_embedding, idx_count, valid_candidate_idx, edge_matrix, bn, bn_count, order, temp_buffer, candidates);
        }

        search_from_depth(enter_depth, cur_depth, cpu_qg, edge_matrix, candidates, candidates_count, order,
                          embedding, idx_embedding, bn, bn_count, true);
    }

    void process_prefix_task(GMContext &context)
    {
        ui sglen = context.cur_depth;
        if (sglen == 0)
            return;

        ui pending_depth = sglen - 1;
        for (ui i = 0; i < pending_depth; ++i)
            visited_arr[context.embedding[matching_order[i]]] = true;

        idx[pending_depth] = 0;
        idx_count[pending_depth] = context.prefix_candidate_count;
        memcpy(valid_candidate_idx[pending_depth], context.prefix_candidate_idx,
               context.prefix_candidate_count * sizeof(ui));

        search_from_depth(pending_depth, pending_depth, cpu_qg, edge_matrix, candidates, candidates_count,
                          matching_order, context.embedding, context.idx_embedding, bn, bn_count, false);
    }

    void search_from_depth(int enter_depth, int cur_depth, Graph_CPU &cpu_qg, Edges ***edge_matrix, ui **candidates,
                           ui *candidates_count, ui *order, ui *embedding, ui *idx_embedding, ui **bn, ui *bn_count,
                           bool allow_prefix_split)
    {
        int max_depth = cpu_qg.getVerticesCount();

        while (true) {
            while (idx[cur_depth] < idx_count[cur_depth]) {
                if (allow_prefix_split && plan.strategyHost[cur_depth + 1] == StoreStrategy::PREFIX && time_over(st))
                {
                    spawn_prefix_task(cur_depth, embedding, idx_embedding, order, cpu_qg.getVerticesCount());
                    idx[cur_depth] = idx_count[cur_depth];
                    break;
                }

                ui valid_idx = valid_candidate_idx[cur_depth][idx[cur_depth]];

                ui u = order[cur_depth];
                
                ui v = candidates[u][valid_idx];


                if (visited_arr[v]) {
                    idx[cur_depth] += 1;
                    continue;
                }

                embedding[u] = v;
                idx_embedding[u] = valid_idx;

                visited_arr[v] = true;

                idx[cur_depth] += 1;

                if (cur_depth == max_depth - 1) {

                    counter += 1;

                    // print first 10000 results
                    // if (counter < 10000)
                    // {
                    //     for(ui i = 0; i < max_depth; ++i)
                    //     {
                    //         cout << embedding[i] << " ";
                    //     }
                    //     cout << endl;
                    // }

                    visited_arr[v] = false;
                    
                    // if(counter % 1000000000 == 0) cout<<counter<<endl;
                    
                    continue;
                }

                // if not timeout, continue search 
                // if(countElaspedTime() < TIME_THRESHOLD) 
                if (!time_over(st))
                {
                    cur_depth += 1;
                    idx[cur_depth] = 0;
                    if (gm_cpu_shared_intersection_g && plan.shareIntersectionHost[cur_depth])
                        generateValidCandidateIndexShared(cur_depth, embedding, idx_count, valid_candidate_idx, order);
                    else if (gm_cpu_gpu_style_expand_g)
                        generateValidCandidateIndexGPUStyle(cur_depth, embedding, idx_embedding, idx_count, valid_candidate_idx, order, candidates, candidates_count);
                    else
                        generateValidCandidateIndex(cur_depth, embedding, idx_embedding, idx_count, valid_candidate_idx, edge_matrix, bn, bn_count, order, temp_buffer, candidates);
                }
                else  // if timeout, start task splitting
                {
                    spawn_split_task(cur_depth, embedding, idx_embedding, order, cpu_qg.getVerticesCount());

                    visited_arr[v] = false;
                }

            }
            cur_depth -= 1;
            if (cur_depth < enter_depth)
                break;
            else
            {
                visited_arr[embedding[order[cur_depth]]] = false;
            }
        }

        for (int i = 0; i < enter_depth; ++i)
            visited_arr[embedding[order[i]]] = false;
    }
};

#endif
