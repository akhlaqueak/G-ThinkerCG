#ifndef MC_CPU_APP
#define MC_CPU_APP

#define TIME_THRESHOLD 10
#define TIME_OVER(ST) (chrono::duration_cast<chrono::milliseconds>(TIME_NOW - ST).count() > TIME_THRESHOLD)

class QCCPUWorker : public CPUWorker<QCTask>
{
public:
    CPU_Data local_hd;
    CPU_Cliques local_hc;
    CPU_Graph local_hg;

    QCCPUWorker() : CPUWorker<QCTask>()
    {
        local_hd = hd;
        local_hc = hc;
        local_hg = *hg;

        local_hd.vertex_order_map = new int[hg->number_of_vertices];
        local_hd.remaining_candidates = new int[hg->number_of_vertices];
        local_hd.removed_candidates = new int[hg->number_of_vertices];
        local_hd.remaining_count = new int(0);
        local_hd.removed_count = new int(0);
        local_hd.candidate_indegs = new int[hg->number_of_vertices];
        local_hd.maximal_expansion = new bool(false);

        memset(local_hd.vertex_order_map, -1, sizeof(int) * hg->number_of_vertices);

        local_hc.cliques_count = new uint64_t;
        local_hc.cliques_vertex = new int[CLIQUES_SIZE];
        local_hc.cliques_offset = new uint64_t[CLIQUES_OFFSET_SIZE];
        (*local_hc.cliques_count) = 0;
        local_hc.cliques_offset[0] = 0;
    }

    ~QCCPUWorker() override
    {
        delete[] local_hd.vertex_order_map;
        local_hd.vertex_order_map = nullptr;
        delete[] local_hd.remaining_candidates;
        local_hd.remaining_candidates = nullptr;
        delete[] local_hd.removed_candidates;
        local_hd.removed_candidates = nullptr;
        delete local_hd.remaining_count;
        local_hd.remaining_count = nullptr;
        delete local_hd.removed_count;
        local_hd.removed_count = nullptr;
        delete[] local_hd.candidate_indegs;
        local_hd.candidate_indegs = nullptr;
        delete local_hd.maximal_expansion;
        local_hd.maximal_expansion = nullptr;

        delete local_hc.cliques_count;
        local_hc.cliques_count = nullptr;
        delete[] local_hc.cliques_vertex;
        local_hc.cliques_vertex = nullptr;
        delete[] local_hc.cliques_offset;
        local_hc.cliques_offset = nullptr;
    }

    void merge_local_cliques_into(CPU_Cliques &dst)
    {
        if ((*local_hc.cliques_count) == 0)
            return;

        uint64_t global_count = *dst.cliques_count;
        uint64_t local_count = *local_hc.cliques_count;
        uint64_t global_write = dst.cliques_offset[global_count];
        uint64_t local_vertices = local_hc.cliques_offset[local_count];

        if (global_count + local_count >= CLIQUES_OFFSET_SIZE || global_write + local_vertices > CLIQUES_SIZE)
            throw std::runtime_error("CPU clique buffer overflow while merging worker results");

        for (uint64_t i = 0; i < local_vertices; i++)
        {
            dst.cliques_vertex[global_write + i] = local_hc.cliques_vertex[i];
        }

        for (uint64_t i = 1; i <= local_count; i++)
        {
            dst.cliques_offset[global_count + i] = global_write + local_hc.cliques_offset[i];
        }

        *dst.cliques_count = global_count + local_count;
        *local_hc.cliques_count = 0;
        local_hc.cliques_offset[0] = 0;
    }
    void h_expand_level(CPU_Graph &hg, CPU_Data &hd, CPU_Cliques &hc, Vertex *read_vertices, size_t read_vertices_count)
    {
        // initiate the variables containing the location of the read and write task vectors, done in an alternating, odd-even manner like the c-intersection of cuTS

        // old vertices information
        uint64_t start;
        uint64_t end;
        int tot_vert;
        int num_mem;
        int num_cand;
        int expansions;
        int number_of_covered;

        // new vertices information
        Vertex *vertices;
        int number_of_members;
        int number_of_candidates;
        int total_vertices;

        // calculate lower-upper bounds
        int min_ext_deg;
        int lower_bound;
        int upper_bound;

        int method_return;
        int index;

        // set to false later if task is generated indicating non-maximal expansion
        (*hd.maximal_expansion) = true;
        // CURRENT LEVEL
        // for (int i = 0; i < *read_count; i++)
        {

            // get information of vertices being handled within tasks
            start = 0;
            end = read_vertices_count;
            tot_vert = end - start;
            num_mem = 0;
            for (uint64_t j = start; j < end; j++)
            {
                if (read_vertices[j].label != 1)
                {
                    break;
                }
                num_mem++;
            }
            number_of_covered = 0;
            for (uint64_t j = num_mem; j < end; j++)
            {
                if (read_vertices[j].label != 2)
                {
                    break;
                }
                number_of_covered++;
            }
            num_cand = tot_vert - num_mem;
            expansions = num_cand;

            // LOOKAHEAD PRUNING
            method_return = h_lookahead_pruning(hg, hc, hd, read_vertices, tot_vert, num_mem, num_cand, start);
            if (method_return)
            {
                return;
            }

            // NEXT LEVEL
            for (int j = number_of_covered; j < expansions; j++)
            {

                // REMOVE ONE VERTEX
                if (j != number_of_covered)
                {
                    method_return = h_remove_one_vertex(hg, hd, read_vertices, tot_vert, num_cand, num_mem, start);
                    if (method_return)
                    {
                        break;
                    }
                }

                // NEW VERTICES
                vertices = new Vertex[tot_vert];
                number_of_members = num_mem;
                number_of_candidates = num_cand;
                total_vertices = tot_vert;
                for (index = 0; index < number_of_members; index++)
                {
                    vertices[index] = read_vertices[index];
                }
                vertices[number_of_members] = read_vertices[total_vertices - 1];
                for (; index < total_vertices - 1; index++)
                {
                    vertices[index + 1] = read_vertices[index];
                }

                if (number_of_covered > 0)
                {
                    // set all covered vertices from previous level as candidates
                    for (int j = num_mem + 1; j <= num_mem + number_of_covered; j++)
                    {
                        vertices[j].label = 0;
                    }
                }

                // ADD ONE VERTEX
                method_return = h_add_one_vertex(hg, hd, vertices, total_vertices, number_of_candidates, number_of_members, upper_bound, lower_bound, min_ext_deg);

                // if vertex in x found as not extendable, check if current set is clique and continue to next iteration
                if (method_return == 1)
                {
                    if (number_of_members >= minimum_clique_size)
                    {
                        h_check_for_clique(hc, vertices, number_of_members);
                    }

                    delete[] vertices;
                    continue;
                }

                // CRITICAL VERTEX PRUNING
                method_return = h_critical_vertex_pruning(hg, hd, vertices, total_vertices, number_of_candidates, number_of_members, upper_bound, lower_bound, min_ext_deg);

                // if critical fail continue onto next iteration
                if (method_return == 2)
                {
                    delete[] vertices;
                    continue;
                }

                // CHECK FOR CLIQUE
                if (number_of_members >= minimum_clique_size)
                {
                    h_check_for_clique(hc, vertices, number_of_members);
                }

                // if vertex in x found as not extendable, check if current set is clique and continue to next iteration
                if (method_return == 1)
                {
                    delete[] vertices;
                    continue;
                }

                // WRITE TO TASKS
                // sort vertices so that lowest degree vertices are first in enumeration order before writing to tasks
                qsort(vertices, total_vertices, sizeof(Vertex), h_sort_vert_Q);

                if (number_of_candidates > 0)
                {
                    for (int k = 0; k < total_vertices; k++)
                        vertices[k].lvl2adj = 0;
                    h_expand_level(hg, hd, hc, vertices, total_vertices);
                    delete[] vertices;
                }
                else
                {
                    delete[] vertices;
                }
            }
        }
        // (*hd.current_level)++;
    }
    Vertex *h_build_initial_task(CPU_Graph &hg, CPU_Data &hd, CPU_Cliques &hc, size_t root_index, size_t &task_vertices_count)
    {
        task_vertices_count = 0;

        if (root_index >= hd.initial_vertices_count)
        {
            return nullptr;
        }

        Vertex root = hd.initial_vertices[root_index];
        std::vector<int> candidate_source_indices;

        candidate_source_indices.reserve(hg.twohop_offsets[root.vertexid + 1] - hg.twohop_offsets[root.vertexid]);

        // Match the existing DFS enumeration order: a root only expands against
        // earlier vertices in the sorted initial frontier.
        for (uint64_t i = hg.twohop_offsets[root.vertexid]; i < hg.twohop_offsets[root.vertexid + 1]; i++)
        {
            int vertexid = hg.twohop_neighbors[i];
            int source_index = hd.initial_order_map[vertexid];

            if (source_index > -1 && static_cast<size_t>(source_index) < root_index)
            {
                candidate_source_indices.push_back(source_index);
            }
        }

        if (candidate_source_indices.empty())
        {
            return nullptr;
        }

        int total_vertices = 1 + static_cast<int>(candidate_source_indices.size());
        int number_of_members = 1;
        int number_of_candidates = total_vertices - 1;
        int upper_bound;
        int lower_bound;
        int min_ext_deg;
        int method_return;
        auto clear_touched_vertex_map = [&]()
        {
            hd.vertex_order_map[root.vertexid] = -1;
            for (int source_index : candidate_source_indices)
            {
                hd.vertex_order_map[hd.initial_vertices[source_index].vertexid] = -1;
            }
        };

        Vertex *vertices = new Vertex[total_vertices];
        vertices[0] = root;
        vertices[0].label = 1;
        vertices[0].indeg = 0;
        vertices[0].exdeg = 0;
        vertices[0].lvl2adj = 0;
        hd.vertex_order_map[root.vertexid] = 0;

        for (size_t i = 0; i < candidate_source_indices.size(); i++)
        {
            Vertex source_vertex = hd.initial_vertices[candidate_source_indices[i]];
            int child_index = static_cast<int>(i) + 1;

            vertices[child_index] = source_vertex;
            vertices[child_index].label = 0;
            vertices[child_index].indeg = 0;
            vertices[child_index].exdeg = 0;
            vertices[child_index].lvl2adj = 0;

            hd.vertex_order_map[source_vertex.vertexid] = child_index;
        }

        for (int i = 0; i < total_vertices; i++)
        {
            int vertexid = vertices[i].vertexid;
            uint64_t pneighbors_start = hg.onehop_offsets[vertexid];
            uint64_t pneighbors_end = hg.onehop_offsets[vertexid + 1];

            for (uint64_t j = pneighbors_start; j < pneighbors_end; j++)
            {
                int phelper1 = hd.vertex_order_map[hg.onehop_neighbors[j]];

                if (phelper1 > -1)
                {
                    if (i == 0)
                    {
                        if (phelper1 > 0)
                        {
                            vertices[i].exdeg++;
                        }
                    }
                    else if (phelper1 == 0)
                    {
                        vertices[i].indeg++;
                    }
                    else
                    {
                        vertices[i].exdeg++;
                    }
                }
            }
        }

        (*hd.remaining_count) = number_of_candidates;
        for (int i = 0; i < number_of_candidates; i++)
        {
            hd.candidate_indegs[i] = vertices[number_of_members + i].indeg;
        }

        method_return = h_degree_pruning(hg, hd, vertices, total_vertices, number_of_candidates, number_of_members, upper_bound, lower_bound, min_ext_deg);
        clear_touched_vertex_map();

        if (method_return)
        {
            if (number_of_members >= minimum_clique_size)
            {
                h_check_for_clique(hc, vertices, number_of_members);
            }
            delete[] vertices;
            return nullptr;
        }

        method_return = h_critical_vertex_pruning(hg, hd, vertices, total_vertices, number_of_candidates, number_of_members, upper_bound, lower_bound, min_ext_deg);

        if (method_return == 2)
        {
            delete[] vertices;
            return nullptr;
        }

        if (number_of_members >= minimum_clique_size)
        {
            h_check_for_clique(hc, vertices, number_of_members);
        }

        if (method_return == 1 || number_of_candidates <= 0)
        {
            delete[] vertices;
            return nullptr;
        }

        qsort(vertices, total_vertices, sizeof(Vertex), h_sort_vert_Q);
        for (int i = 0; i < total_vertices; i++)
        {
            vertices[i].lvl2adj = 0;
        }

        task_vertices_count = static_cast<size_t>(total_vertices);
        return vertices;
    }

    virtual QCTask *task_spawn(VertexID &index)
    {
        size_t task_vertices_count;
        Vertex *vertices = h_build_initial_task(local_hg, local_hd, local_hc, index, task_vertices_count);
        QCTask *t = new QCTask();
        if (vertices != nullptr)
            t->context = QCContext(vertices, task_vertices_count);
        return t;
    }

    virtual void compute(QCContext &context) override
    {
        if (context.vertices == nullptr || context.num_vertices == 0)
            return;
        h_expand_level(local_hg, local_hd, local_hc, context.vertices, context.num_vertices);
    }
};

#endif
