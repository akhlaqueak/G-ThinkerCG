#ifndef GM_CPU_APP
#define GM_CPU_APP

#define TIME_THRESHOLD 1000
#define TIME_OVER(ST) (chrono::duration_cast<chrono::milliseconds>(TIME_NOW - ST).count() > TIME_THRESHOLD)


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

    ui *temp_buffer;
    bool *visited_arr;
    ui *idx;
    ui *idx_count;
    ui **valid_candidate_idx;

    struct timeb thread_local_time;

    std::chrono::time_point<std::chrono::steady_clock> st;

    // =============================================

    virtual GMTask *task_spawn(VertexID &data)
    {
        GMTask *t = new GMTask();
        t->context.query_vertices_num = cpu_qg.getVerticesCount();
        t->context.cur_depth = 1;

        t->context.embedding = new ui[cpu_qg.getVerticesCount()];
        t->context.idx_embedding = new ui[cpu_qg.getVerticesCount()];

        // set data and its index
        t->context.embedding[matching_order[0]] = data;
        t->context.idx_embedding[matching_order[0]] = binary_search(0, data); // [10,22]

        return t;
    }

    virtual void compute(GMContext &context)
    {
        if (temp_buffer == NULL)
        {
            // allocate space for temp_buffer array
            temp_buffer = new ui[max_candidate_cnt];

            // allocate space for visited_arr array
            visited_arr = new bool[cpu_dg.getVerticesCount()];
            memset(visited_arr, false, sizeof(bool)*cpu_dg.getVerticesCount());

            // allocate space for idx and idx_count array
            idx = new ui[cpu_qg.getVerticesCount()];
            idx_count = new ui[cpu_qg.getVerticesCount()];

            // allocate space for valid candidate 2-dimensional array
            valid_candidate_idx = new ui*[cpu_qg.getVerticesCount()];
            for (ui i = 0; i < cpu_qg.getVerticesCount(); ++i) {
                valid_candidate_idx[i] = new ui[max_candidate_cnt];
            }
        }

        ftime(&thread_local_time);
        st=TIME_NOW;
    
        LFTJ(context.cur_depth, cpu_qg, edge_matrix, candidates, candidates_count, matching_order, context.embedding, 
                    context.idx_embedding, bn, bn_count);
    }

    double countElaspedTime()
    {
        struct timeb cur_time;
        ftime(&cur_time);
        // return (double)(cur_time.millitm - thread_local_time.millitm);
        return cur_time.time - thread_local_time.time + (double)(cur_time.millitm - thread_local_time.millitm)/1000;
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

        memcpy(valid_candidate_index[depth], previous_candidates, valid_candidates_count * sizeof(ui));

        ui temp_count;
        for (ui i = 1; i < bn_cnt[depth]; ++i) {
            
            VertexID current_bn = bn[depth][i];

            Edges& current_edge = *edge_matrix[current_bn][u];
            ui current_index_id = idx_embedding[current_bn];


            ui current_candidates_count = current_edge.offset_[current_index_id + 1] - current_edge.offset_[current_index_id];

            ui* current_candidates = current_edge.edge_ + current_edge.offset_[current_index_id];


            if (current_candidates_count < valid_candidates_count)
                ComputeSetIntersection::ComputeCandidates(current_candidates, current_candidates_count, valid_candidate_index[depth], valid_candidates_count,
                            temp_buffer_, temp_count);
            else
                ComputeSetIntersection::ComputeCandidates(valid_candidate_index[depth], valid_candidates_count, current_candidates, current_candidates_count,
                            temp_buffer_, temp_count);
          

            // std::swap(temp_buffer, valid_candidate_index[depth]); // all elements are swapped

            for(int i = 0; i < temp_count; ++i)
            {
                valid_candidate_index[depth][i] = temp_buffer_[i];
            }
            valid_candidates_count = temp_count;
        }

        // ====================================================
        ui condCount = plan.condNumHost[u];
        ui tmp_len = 0;
        for (ui i = 0; i < valid_candidates_count; ++i) {
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

        // idx_count[depth] = valid_candidates_count;
        idx_count[depth] = tmp_len;
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
        
            // compute set intersection
            generateValidCandidateIndex(cur_depth, embedding, idx_embedding, idx_count, valid_candidate_idx, edge_matrix, bn, bn_count, order, temp_buffer, candidates);
  
            
            // initialize visited_arr array 
            for (ui i = 0; i < enter_depth; ++i)
            {
                visited_arr[embedding[order[i]]] = true;
            }
        }

        while (true) {
            while (idx[cur_depth] < idx_count[cur_depth]) {
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
                    
                    if(counter % 1000000000 == 0) cout<<counter<<endl;
                    
                    continue;
                }

                // if not timeout, continue search 
                // if(countElaspedTime() < TIME_THRESHOLD) 
                if(not TIME_OVER(st))
                {
                    cur_depth += 1;
                    idx[cur_depth] = 0;
                    generateValidCandidateIndex(cur_depth, embedding, idx_embedding, idx_count, valid_candidate_idx, edge_matrix, bn, bn_count, order, temp_buffer, candidates);
                }
                else  // if timeout, start task splitting
                {
                    ui query_vertices_num = cpu_qg.getVerticesCount();
                    GMTask *t = new GMTask();

                    t->context.query_vertices_num = query_vertices_num;
                    t->context.cur_depth = cur_depth+1;

                    t->context.embedding = new ui[query_vertices_num];
                    memcpy(t->context.embedding, embedding, sizeof(ui)*query_vertices_num);
                    t->context.idx_embedding = new ui[query_vertices_num];
                    memcpy(t->context.idx_embedding, idx_embedding, sizeof(ui)*query_vertices_num);
                      
                    add_task(t);
                    // cout<<"+";

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

        // ####
        for (ui i = 0; i < enter_depth; ++i)
        {
            visited_arr[embedding[order[i]]] = false;
        }
    }
};

#endif

