#ifndef GM_CPU_OLD_MODULAR_APP
#define GM_CPU_OLD_MODULAR_APP

#define TIME_THRESHOLD 1000
#define TIME_OVER(ST) (chrono::duration_cast<chrono::milliseconds>(TIME_NOW - ST).count() > TIME_THRESHOLD)

#include "graph_cpu.h"
#include "FilterVertices.h"
#include "GenerateQueryPlan.h"
#include "BuildTable.h"
#include "leapfrogjoin.h"
#include "intersection/computesetintersection.h"

class GMCPUWorkerOldModular : public CPUWorker<GMTask>
{
public:
    unsigned long long int counter = 0;

    ui *temp_buffer = nullptr;
    bool *visited_arr = nullptr;
    ui *idx = nullptr;
    ui *idx_count = nullptr;
    ui **valid_candidate_idx = nullptr;

    struct timeb thread_local_time;
    std::chrono::time_point<std::chrono::steady_clock> st;

    virtual GMTask *task_spawn(VertexID &data)
    {
        GMTask *t = new GMTask();
        t->context.query_vertices_num = cpu_qg.getVerticesCount();
        t->context.cur_depth = 1;

        t->context.embedding = new ui[cpu_qg.getVerticesCount()];
        t->context.idx_embedding = new ui[cpu_qg.getVerticesCount()];

        t->context.embedding[matching_order[0]] = data;
        t->context.idx_embedding[matching_order[0]] = binary_search(0, data);

        return t;
    }

    virtual void compute(GMContext &context)
    {
        ensure_auxiliary_buffers();
        start_local_timer();
        LFTJ(context.cur_depth, cpu_qg, edge_matrix, candidates, candidates_count,
             matching_order, context.embedding, context.idx_embedding, bn, bn_count);
    }

    double countElaspedTime()
    {
        struct timeb cur_time;
        ftime(&cur_time);
        return cur_time.time - thread_local_time.time +
               (double)(cur_time.millitm - thread_local_time.millitm) / 1000;
    }

    void generateValidCandidateIndex(ui depth, ui *embedding, ui *idx_embedding, ui *idx_count,
                                     ui **valid_candidate_index, Edges ***edge_matrix, ui **bn,
                                     ui *bn_cnt, ui *order, ui *temp_buffer_, ui **candidates)
    {
        ui u = order[depth];
        ui previous_bn = bn[depth][0];
        ui previous_index_id = idx_embedding[previous_bn];
        ui valid_candidates_count = 0;

        Edges &previous_edge = *edge_matrix[previous_bn][u];
        valid_candidates_count = previous_edge.offset_[previous_index_id + 1] - previous_edge.offset_[previous_index_id];
        ui *previous_candidates = previous_edge.edge_ + previous_edge.offset_[previous_index_id];

        memcpy(valid_candidate_index[depth], previous_candidates, valid_candidates_count * sizeof(ui));

        intersect_backward_neighbors(depth, idx_embedding, valid_candidates_count, idx_count,
                                     valid_candidate_index, edge_matrix, bn, bn_cnt,
                                     temp_buffer_, candidates);
        filter_candidates_by_conditions(depth, embedding, idx_count, valid_candidate_index, candidates);
    }

    void LFTJ(int enter_depth, Graph_CPU &cpu_qg, Edges ***edge_matrix, ui **candidates,
              ui *candidates_count, ui *order, ui *embedding, ui *idx_embedding,
              ui **bn, ui *bn_count)
    {
        int cur_depth = enter_depth;
        int max_depth = cpu_qg.getVerticesCount();

        initialize_search_state(enter_depth, cur_depth, candidates_count, order, embedding,
                                idx_embedding, bn, bn_count, edge_matrix, candidates);

        run_dfs_loop(enter_depth, cur_depth, max_depth, cpu_qg, edge_matrix, candidates,
                     candidates_count, order, embedding, idx_embedding, bn, bn_count);

        clear_entered_vertices(enter_depth, embedding, order);
    }

private:
    void ensure_auxiliary_buffers()
    {
        if (temp_buffer != NULL)
            return;

        temp_buffer = new ui[max_candidate_cnt];

        visited_arr = new bool[cpu_dg.getVerticesCount()];
        memset(visited_arr, false, sizeof(bool) * cpu_dg.getVerticesCount());

        idx = new ui[cpu_qg.getVerticesCount()];
        idx_count = new ui[cpu_qg.getVerticesCount()];

        valid_candidate_idx = new ui *[cpu_qg.getVerticesCount()];
        for (ui i = 0; i < cpu_qg.getVerticesCount(); ++i)
            valid_candidate_idx[i] = new ui[max_candidate_cnt];
    }

    void start_local_timer()
    {
        ftime(&thread_local_time);
        st = TIME_NOW;
    }

    void intersect_backward_neighbors(ui depth, ui *idx_embedding, ui &valid_candidates_count,
                                      ui *idx_count, ui **valid_candidate_index,
                                      Edges ***edge_matrix, ui **bn, ui *bn_cnt,
                                      ui *temp_buffer_, ui **candidates)
    {
        ui u = order_vertex(depth);
        ui temp_count = 0;

        for (ui i = 1; i < bn_cnt[depth]; ++i)
        {
            VertexID current_bn = bn[depth][i];
            Edges &current_edge = *edge_matrix[current_bn][u];
            ui current_index_id = idx_embedding[current_bn];
            ui current_candidates_count = current_edge.offset_[current_index_id + 1] - current_edge.offset_[current_index_id];
            ui *current_candidates = current_edge.edge_ + current_edge.offset_[current_index_id];

            if (current_candidates_count < valid_candidates_count)
                ComputeSetIntersection::ComputeCandidates(current_candidates, current_candidates_count,
                                                          valid_candidate_index[depth], valid_candidates_count,
                                                          temp_buffer_, temp_count);
            else
                ComputeSetIntersection::ComputeCandidates(valid_candidate_index[depth], valid_candidates_count,
                                                          current_candidates, current_candidates_count,
                                                          temp_buffer_, temp_count);

            for (int j = 0; j < temp_count; ++j)
                valid_candidate_index[depth][j] = temp_buffer_[j];

            valid_candidates_count = temp_count;
        }

        idx_count[depth] = valid_candidates_count;
        (void)candidates;
    }

    void filter_candidates_by_conditions(ui depth, ui *embedding, ui *idx_count,
                                         ui **valid_candidate_index, ui **candidates)
    {
        ui u = order_vertex(depth);
        ui valid_candidates_count = idx_count[depth];
        ui condCount = plan.condNumHost[u];
        ui tmp_len = 0;

        for (ui i = 0; i < valid_candidates_count; ++i)
        {
            ui valid_index = valid_candidate_index[depth][i];
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

        idx_count[depth] = tmp_len;
    }

    void initialize_search_state(int enter_depth, int cur_depth, ui *candidates_count,
                                 ui *order, ui *embedding, ui *idx_embedding, ui **bn,
                                 ui *bn_count, Edges ***edge_matrix, ui **candidates)
    {
        if (cur_depth == 0)
            initialize_root_depth(candidates_count, order);
        else
            initialize_non_root_depth(enter_depth, cur_depth, order, embedding, idx_embedding,
                                      bn, bn_count, edge_matrix, candidates);
    }

    void initialize_root_depth(ui *candidates_count, ui *order)
    {
        ui start_vertex = order[0];
        idx[0] = 0;
        idx_count[0] = candidates_count[start_vertex];

        for (ui i = 0; i < idx_count[0]; ++i)
            valid_candidate_idx[0][i] = i;
    }

    void initialize_non_root_depth(int enter_depth, int cur_depth, ui *order, ui *embedding,
                                   ui *idx_embedding, ui **bn, ui *bn_count,
                                   Edges ***edge_matrix, ui **candidates)
    {
        idx[cur_depth] = 0;
        generateValidCandidateIndex(cur_depth, embedding, idx_embedding, idx_count,
                                    valid_candidate_idx, edge_matrix, bn, bn_count,
                                    order, temp_buffer, candidates);

        for (ui i = 0; i < enter_depth; ++i)
            visited_arr[embedding[order[i]]] = true;
    }

    void run_dfs_loop(int enter_depth, int &cur_depth, int max_depth, Graph_CPU &cpu_qg,
                      Edges ***edge_matrix, ui **candidates, ui *candidates_count,
                      ui *order, ui *embedding, ui *idx_embedding, ui **bn, ui *bn_count)
    {
        while (true)
        {
            while (idx[cur_depth] < idx_count[cur_depth])
            {
                ui valid_idx = valid_candidate_idx[cur_depth][idx[cur_depth]];
                ui u = order[cur_depth];
                ui v = candidates[u][valid_idx];

                if (visited_arr[v])
                {
                    idx[cur_depth] += 1;
                    continue;
                }

                embedding[u] = v;
                idx_embedding[u] = valid_idx;
                visited_arr[v] = true;
                idx[cur_depth] += 1;

                if (cur_depth == max_depth - 1)
                {
                    handle_complete_match(v);
                    continue;
                }

                if (should_continue_search())
                    descend_one_level(cur_depth, embedding, idx_embedding, edge_matrix, candidates,
                                      order, bn, bn_count);
                else
                    split_task_and_backtrack(cur_depth, cpu_qg, embedding, idx_embedding, order, v);
            }

            cur_depth -= 1;
            if (cur_depth < enter_depth)
                break;

            visited_arr[embedding[order[cur_depth]]] = false;
        }
    }

    void handle_complete_match(ui v)
    {
        counter += 1;
        visited_arr[v] = false;

        if (counter % 1000000000 == 0)
            cout << counter << endl;
    }

    bool should_continue_search() const
    {
        return not TIME_OVER(st);
    }

    void descend_one_level(int &cur_depth, ui *embedding, ui *idx_embedding,
                           Edges ***edge_matrix, ui **candidates, ui *order,
                           ui **bn, ui *bn_count)
    {
        cur_depth += 1;
        idx[cur_depth] = 0;
        generateValidCandidateIndex(cur_depth, embedding, idx_embedding, idx_count,
                                    valid_candidate_idx, edge_matrix, bn, bn_count,
                                    order, temp_buffer, candidates);
    }

    void split_task_and_backtrack(int cur_depth, Graph_CPU &cpu_qg, ui *embedding,
                                  ui *idx_embedding, ui *order, ui v)
    {
        ui query_vertices_num = cpu_qg.getVerticesCount();
        GMTask *t = new GMTask();

        t->context.query_vertices_num = query_vertices_num;
        t->context.cur_depth = cur_depth + 1;

        t->context.embedding = new ui[query_vertices_num];
        memcpy(t->context.embedding, embedding, sizeof(ui) * query_vertices_num);
        t->context.idx_embedding = new ui[query_vertices_num];
        memcpy(t->context.idx_embedding, idx_embedding, sizeof(ui) * query_vertices_num);

        add_task(t);
        visited_arr[v] = false;
    }

    void clear_entered_vertices(int enter_depth, ui *embedding, ui *order)
    {
        for (ui i = 0; i < enter_depth; ++i)
            visited_arr[embedding[order[i]]] = false;
    }

    ui order_vertex(ui depth) const
    {
        return matching_order[depth];
    }
};

#endif
