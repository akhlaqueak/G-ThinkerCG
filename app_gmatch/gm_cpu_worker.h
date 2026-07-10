#ifndef GM_CPU_APP
#define GM_CPU_APP

#include "graph_cpu.h"
#include "FilterVertices.h"
#include "GenerateQueryPlan.h"
#include "BuildTable.h"
#include "leapfrogjoin.h"
#include "intersection/computesetintersection.h"
#include <atomic>
#include <iomanip>

#ifdef GM_CPU_PROFILE
struct GMCPUProfileStats
{
    std::atomic<unsigned long long> root_task_calls{0};
    std::atomic<unsigned long long> regular_task_calls{0};
    std::atomic<unsigned long long> prefix_task_calls{0};
    std::atomic<unsigned long long> split_task_spawns{0};
    std::atomic<unsigned long long> prefix_task_spawns{0};
    std::atomic<unsigned long long> valid_candidate_calls{0};
    std::atomic<unsigned long long> valid_candidate_intersections{0};
    std::atomic<unsigned long long> valid_candidate_seed_sum{0};
    std::atomic<unsigned long long> valid_candidate_pre_filter_sum{0};
    std::atomic<unsigned long long> valid_candidate_post_filter_sum{0};

    std::atomic<unsigned long long> root_batch_ns{0};
    std::atomic<unsigned long long> regular_batch_ns{0};
    std::atomic<unsigned long long> prefix_batch_ns{0};
    std::atomic<unsigned long long> valid_candidate_ns{0};

    static GMCPUProfileStats &instance()
    {
        static GMCPUProfileStats stats;
        return stats;
    }

    static unsigned long long to_ns(const std::chrono::steady_clock::duration &duration)
    {
        return static_cast<unsigned long long>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count());
    }

    static double ns_to_s(unsigned long long ns)
    {
        return static_cast<double>(ns) / 1e9;
    }

    static void add(std::atomic<unsigned long long> &slot, unsigned long long value)
    {
        slot.fetch_add(value, std::memory_order_relaxed);
    }

    static void print_summary()
    {
        auto &s = instance();
        cout << fixed << setprecision(6);
        cout << "===== GM CPU Profile =====" << endl;
        cout << "root task calls: " << s.root_task_calls.load(std::memory_order_relaxed) << endl;
        cout << "regular task calls: " << s.regular_task_calls.load(std::memory_order_relaxed) << endl;
        cout << "prefix task calls: " << s.prefix_task_calls.load(std::memory_order_relaxed) << endl;
        cout << "split task spawns: " << s.split_task_spawns.load(std::memory_order_relaxed) << endl;
        cout << "prefix task spawns: " << s.prefix_task_spawns.load(std::memory_order_relaxed) << endl;
        cout << "valid-candidate calls: " << s.valid_candidate_calls.load(std::memory_order_relaxed) << endl;
        cout << "valid-candidate intersection steps: " << s.valid_candidate_intersections.load(std::memory_order_relaxed) << endl;
        cout << "root batch total (s): " << ns_to_s(s.root_batch_ns.load(std::memory_order_relaxed)) << endl;
        cout << "regular batch total (s): " << ns_to_s(s.regular_batch_ns.load(std::memory_order_relaxed)) << endl;
        cout << "prefix batch total (s): " << ns_to_s(s.prefix_batch_ns.load(std::memory_order_relaxed)) << endl;
        cout << "valid-candidate total (s): " << ns_to_s(s.valid_candidate_ns.load(std::memory_order_relaxed)) << endl;
        const auto vcalls = s.valid_candidate_calls.load(std::memory_order_relaxed);
        if (vcalls != 0)
        {
            cout << "avg seed candidates: "
                 << (static_cast<double>(s.valid_candidate_seed_sum.load(std::memory_order_relaxed)) / vcalls) << endl;
            cout << "avg pre-filter candidates: "
                 << (static_cast<double>(s.valid_candidate_pre_filter_sum.load(std::memory_order_relaxed)) / vcalls) << endl;
            cout << "avg post-filter candidates: "
                 << (static_cast<double>(s.valid_candidate_post_filter_sum.load(std::memory_order_relaxed)) / vcalls) << endl;
            cout << "avg intersection steps/call: "
                 << (static_cast<double>(s.valid_candidate_intersections.load(std::memory_order_relaxed)) / vcalls) << endl;
        }
        cout << "==========================" << endl;
    }
};
#else
struct GMCPUProfileStats
{
    static GMCPUProfileStats &instance()
    {
        static GMCPUProfileStats stats;
        return stats;
    }

    static unsigned long long to_ns(const std::chrono::steady_clock::duration &)
    {
        return 0;
    }

    static void add(...)
    {
    }

    static void print_summary()
    {
    }
};
#endif


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
    }

    void run() override
    {
#ifdef GM_CPU_PROFILE
        auto &stats = GMCPUProfileStats::instance();
#endif
        if (!this->Lv.empty())
        {
#ifdef GM_CPU_PROFILE
            auto root_batch_start = std::chrono::steady_clock::now();
#endif
            for (auto &u : this->Lv)
            {
                GMTask *task = task_spawn(u);
#ifdef GM_CPU_PROFILE
                stats.root_task_calls.fetch_add(1, std::memory_order_relaxed);
#endif
                compute(task->context);
                delete task;
            }
#ifdef GM_CPU_PROFILE
            GMCPUProfileStats::add(stats.root_batch_ns,
                                   GMCPUProfileStats::to_ns(std::chrono::steady_clock::now() - root_batch_start));
#endif
            this->Lv.clear();
        }

        if (!this->Lt.empty())
        {
#ifdef GM_CPU_PROFILE
            auto regular_batch_start = std::chrono::steady_clock::now();
            unsigned long long prefix_tasks = 0;
            unsigned long long regular_tasks = 0;
#endif
            for (auto task : this->Lt)
            {
#ifdef GM_CPU_PROFILE
                if (task->context.prefix_candidate_idx != nullptr)
                    prefix_tasks += 1;
                else
                    regular_tasks += 1;
#endif
                compute(task->context);
                delete task;
            }
#ifdef GM_CPU_PROFILE
            auto batch_ns = GMCPUProfileStats::to_ns(std::chrono::steady_clock::now() - regular_batch_start);
            stats.prefix_task_calls.fetch_add(prefix_tasks, std::memory_order_relaxed);
            stats.regular_task_calls.fetch_add(regular_tasks, std::memory_order_relaxed);
            if (prefix_tasks != 0 && regular_tasks == 0)
                GMCPUProfileStats::add(stats.prefix_batch_ns, batch_ns);
            else
                GMCPUProfileStats::add(stats.regular_batch_ns, batch_ns);
#endif
            this->Lt.clear();
        }
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
#if GM_CPU_PROFILE
        auto &stats = GMCPUProfileStats::instance();
        stats.split_task_spawns.fetch_add(1, std::memory_order_relaxed);
#endif
    }

    void spawn_prefix_task(ui cur_depth, ui *embedding, ui *idx_embedding, ui *order, ui query_vertices_num)
    {
        ui remaining = idx_count[cur_depth] - idx[cur_depth];
        if (remaining == 0)
            return;

        ui spawned = 0;
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
            spawned += 1;
        }
#if GM_CPU_PROFILE
        auto &stats = GMCPUProfileStats::instance();
        stats.prefix_task_spawns.fetch_add(spawned, std::memory_order_relaxed);
#endif
    }

    void generateValidCandidateIndex(ui depth, ui *embedding, ui *idx_embedding, ui *idx_count, ui **valid_candidate_index,
                                    Edges ***edge_matrix, ui **bn, ui *bn_cnt, ui *order, ui *temp_buffer_, ui **candidates)
    {   
#ifdef GM_CPU_PROFILE
        auto &stats = GMCPUProfileStats::instance();
        stats.valid_candidate_calls.fetch_add(1, std::memory_order_relaxed);
        stats.valid_candidate_intersections.fetch_add(bn_cnt[depth] > 0 ? bn_cnt[depth] - 1 : 0, std::memory_order_relaxed);
#endif
        ui u = order[depth];
        ui valid_candidates_count = 0;
        ui seed_bn = bn[depth][0];
        ui seed_index_id = idx_embedding[seed_bn];
        Edges *seed_edge = edge_matrix[seed_bn][u];
        valid_candidates_count = seed_edge->offset_[seed_index_id + 1] - seed_edge->offset_[seed_index_id];

        for (ui i = 1; i < bn_cnt[depth]; ++i)
        {
            ui current_bn = bn[depth][i];
            ui current_index_id = idx_embedding[current_bn];
            Edges *current_edge = edge_matrix[current_bn][u];
            ui current_candidates_count =
                current_edge->offset_[current_index_id + 1] - current_edge->offset_[current_index_id];
            if (current_candidates_count < valid_candidates_count)
            {
                seed_bn = current_bn;
                seed_index_id = current_index_id;
                seed_edge = current_edge;
                valid_candidates_count = current_candidates_count;
            }
        }

        ui *previous_candidates = seed_edge->edge_ + seed_edge->offset_[seed_index_id];
#ifdef GM_CPU_PROFILE
        stats.valid_candidate_seed_sum.fetch_add(valid_candidates_count, std::memory_order_relaxed);
#endif

        ui *current_buffer = valid_candidate_index[depth];
        ui *next_buffer = temp_buffer_;
        memcpy(current_buffer, previous_candidates, valid_candidates_count * sizeof(ui));

        ui temp_count = 0;
        for (ui i = 0; i < bn_cnt[depth]; ++i) {
            
            VertexID current_bn = bn[depth][i];
            if (current_bn == seed_bn)
                continue;

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
#ifdef GM_CPU_PROFILE
        stats.valid_candidate_pre_filter_sum.fetch_add(valid_candidates_count, std::memory_order_relaxed);
#endif
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
#ifdef GM_CPU_PROFILE
        stats.valid_candidate_post_filter_sum.fetch_add(tmp_len, std::memory_order_relaxed);
#endif
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
#ifdef GM_CPU_PROFILE
            auto valid_candidate_start = std::chrono::steady_clock::now();
#endif
            generateValidCandidateIndex(cur_depth, embedding, idx_embedding, idx_count, valid_candidate_idx, edge_matrix, bn, bn_count, order, temp_buffer, candidates);
#ifdef GM_CPU_PROFILE
            GMCPUProfileStats::add(GMCPUProfileStats::instance().valid_candidate_ns,
                                   GMCPUProfileStats::to_ns(std::chrono::steady_clock::now() - valid_candidate_start));
#endif
  
            
            // initialize visited_arr array 
            for (ui i = 0; i < enter_depth; ++i)
            {
                visited_arr[embedding[order[i]]] = true;
            }
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
#ifdef GM_CPU_PROFILE
                    auto valid_candidate_start = std::chrono::steady_clock::now();
#endif
                    generateValidCandidateIndex(cur_depth, embedding, idx_embedding, idx_count, valid_candidate_idx, edge_matrix, bn, bn_count, order, temp_buffer, candidates);
#ifdef GM_CPU_PROFILE
                    GMCPUProfileStats::add(GMCPUProfileStats::instance().valid_candidate_ns,
                                           GMCPUProfileStats::to_ns(std::chrono::steady_clock::now() - valid_candidate_start));
#endif
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
