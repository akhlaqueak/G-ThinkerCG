#ifndef MC_CPU_APP
#define MC_CPU_APP

class QCCPUWorker : public CPUWorker<QCTask>
{
public:
    CPU_Data local_hd;
    CPU_Cliques local_hc;
    bool active_big_root_lineage;

    QCCPUWorker() : CPUWorker<QCTask>()
    {
        local_hd = hd;
        local_hc = hc;

        local_hd.vertex_order_map = new int[hg->number_of_vertices];
        local_hd.remaining_candidates = new int[hg->number_of_vertices];
        local_hd.removed_candidates = new int[hg->number_of_vertices];
        local_hd.remaining_count = new int(0);
        local_hd.removed_count = new int(0);
        local_hd.candidate_indegs = new int[hg->number_of_vertices];
        local_hd.maximal_expansion = new bool(false);

        memset(local_hd.vertex_order_map, -1, sizeof(int) * hg->number_of_vertices);

        local_hc.cliques_count = 0;
        local_hc.cliques_vertex.clear();
        if (store_cliques)
        {
            local_hc.cliques_offset.assign(1, 0);
        }
        else
        {
            local_hc.cliques_offset.clear();
        }

        active_big_root_lineage = false;
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

        local_hc.cliques_vertex.clear();
        local_hc.cliques_vertex.shrink_to_fit();
        local_hc.cliques_offset.clear();
        local_hc.cliques_offset.shrink_to_fit();
    }

    void merge_local_cliques_into(CPU_Cliques &dst)
    {
        dst.cliques_count += local_hc.cliques_count;
        dst.max_clique_size = std::max(dst.max_clique_size, local_hc.max_clique_size);
        local_hc.cliques_count = 0;
        local_hc.max_clique_size = 0;

        if (local_hc.cliques_offset.size() <= 1)
            return;

        uint64_t global_write = dst.cliques_vertex.size();
        dst.cliques_vertex.insert(dst.cliques_vertex.end(), local_hc.cliques_vertex.begin(), local_hc.cliques_vertex.end());

        for (size_t i = 1; i < local_hc.cliques_offset.size(); i++)
        {
            dst.cliques_offset.push_back(global_write + local_hc.cliques_offset[i]);
        }

        local_hc.cliques_vertex.clear();
        if (store_cliques)
        {
            local_hc.cliques_offset.assign(1, 0);
        }
        else
        {
            local_hc.cliques_offset.clear();
        }
    }

private:
    bool is_large_top_level_root(size_t root_index) const
    {
        if (root_index >= local_hd.initial_vertices_count)
            return false;

        const Vertex &root = local_hd.initial_vertices[root_index];
        size_t vertices_count = 1;

        for (uint64_t i = hg->twohop_offsets[root.vertexid]; i < hg->twohop_offsets[root.vertexid + 1]; i++)
        {
            int vertexid = hg->twohop_neighbors[i];
            int source_index = local_hd.initial_order_map[vertexid];

            if (source_index > -1 && static_cast<size_t>(source_index) < root_index)
                vertices_count++;
        }

        return vertices_count > WVERTICES_SIZE;
    }

    void enqueue_task(Vertex *vertices, size_t num_vertices, bool from_big_root)
    {
        QCTask *new_task = new QCTask();
        new_task->context = QCContext(vertices, num_vertices);
        new_task->context.from_big_root = from_big_root;
        if (num_vertices > WVERTICES_SIZE)
            this->add_large_task(new_task);
        else
            this->add_task(new_task);
    }

    void h_write_clique(CPU_Cliques &hc, Vertex *vertices, int clique_size)
    {
        hc.cliques_count++;
        hc.max_clique_size = std::max<uint64_t>(hc.max_clique_size, clique_size);
        if (!store_cliques)
            return;

        uint64_t start_write = hc.cliques_vertex.size();
        for (int i = 0; i < clique_size; i++)
        {
            hc.cliques_vertex.push_back(vertices[i].vertexid);
        }
        hc.cliques_offset.push_back(start_write + clique_size);
    }

    void h_check_for_clique(CPU_Cliques &hc, Vertex *vertices, int number_of_members)
    {
        bool clique = true;

        int degree_requirement = minimum_degrees[number_of_members];
        for (int k = 0; k < number_of_members; k++)
        {
            if (vertices[k].indeg < degree_requirement)
            {
                clique = false;
                break;
            }
        }

        if (clique)
        {
            h_write_clique(hc, vertices, number_of_members);
        }
    }

    bool h_calculate_LU_bounds(CPU_Data &hd, int &upper_bound, int &lower_bound, int &min_ext_deg, Vertex *vertices, int number_of_members, int number_of_candidates)
    {
        bool invalid_bounds = false;
        int index;

        int sum_candidate_indeg = 0;
        int tightened_upper_bound = 0;

        int min_clq_indeg = vertices[0].indeg;
        int min_indeg_exdeg = vertices[0].exdeg;
        int min_clq_totaldeg = vertices[0].indeg + vertices[0].exdeg;
        int sum_clq_indeg = vertices[0].indeg;

        for (index = 1; index < number_of_members; index++)
        {
            sum_clq_indeg += vertices[index].indeg;

            if (vertices[index].indeg < min_clq_indeg)
            {
                min_clq_indeg = vertices[index].indeg;
                min_indeg_exdeg = vertices[index].exdeg;
            }
            else if (vertices[index].indeg == min_clq_indeg)
            {
                if (vertices[index].exdeg < min_indeg_exdeg)
                {
                    min_indeg_exdeg = vertices[index].exdeg;
                }
            }

            if (vertices[index].indeg + vertices[index].exdeg < min_clq_totaldeg)
            {
                min_clq_totaldeg = vertices[index].indeg + vertices[index].exdeg;
            }
        }

        min_ext_deg = h_get_mindeg(number_of_members + 1);

        if (min_clq_indeg < minimum_degrees[number_of_members])
        {
            lower_bound = h_get_mindeg(number_of_members) - min_clq_indeg;

            while (lower_bound <= min_indeg_exdeg && min_clq_indeg + lower_bound < minimum_degrees[number_of_members + lower_bound])
            {
                lower_bound++;
            }

            if (min_clq_indeg + lower_bound < minimum_degrees[number_of_members + lower_bound])
            {
                lower_bound = number_of_candidates + 1;
                invalid_bounds = true;
            }

            upper_bound = floor(min_clq_totaldeg / minimum_degree_ratio) + 1 - number_of_members;

            if (upper_bound > number_of_candidates)
            {
                upper_bound = number_of_candidates;
            }

            if (lower_bound < upper_bound)
            {
                for (index = 0; index < lower_bound; index++)
                {
                    sum_candidate_indeg += hd.candidate_indegs[index];
                }

                while (index < upper_bound && sum_clq_indeg + sum_candidate_indeg < number_of_members * minimum_degrees[number_of_members + index])
                {
                    sum_candidate_indeg += hd.candidate_indegs[index];
                    index++;
                }

                if (sum_clq_indeg + sum_candidate_indeg < number_of_members * minimum_degrees[number_of_members + index])
                {
                    lower_bound = upper_bound + 1;
                    invalid_bounds = true;
                }
                else
                {
                    lower_bound = index;

                    tightened_upper_bound = index;

                    while (index < upper_bound)
                    {
                        sum_candidate_indeg += hd.candidate_indegs[index];
                        index++;

                        if (sum_clq_indeg + sum_candidate_indeg >= number_of_members * minimum_degrees[number_of_members + index])
                        {
                            tightened_upper_bound = index;
                        }
                    }

                    if (upper_bound > tightened_upper_bound)
                    {
                        upper_bound = tightened_upper_bound;
                    }

                    if (lower_bound > 1)
                    {
                        min_ext_deg = h_get_mindeg(number_of_members + lower_bound);
                    }
                }
            }
        }
        else
        {
            upper_bound = number_of_candidates;

            if (number_of_members < minimum_clique_size)
            {
                lower_bound = minimum_clique_size - number_of_members;
            }
            else
            {
                lower_bound = 0;
            }
        }

        if (number_of_members + upper_bound < minimum_clique_size)
        {
            invalid_bounds = true;
        }

        if (upper_bound < 0 || upper_bound < lower_bound)
        {
            invalid_bounds = true;
        }

        return invalid_bounds;
    }

    void h_diameter_pruning(CPU_Graph &hg, CPU_Data &hd, Vertex *vertices, int pvertexid, int &total_vertices, int &number_of_candidates, int number_of_members)
    {
        uint64_t pneighbors_start;
        uint64_t pneighbors_end;
        int phelper1;

        (*hd.remaining_count) = 0;

        for (int i = number_of_members; i < total_vertices; i++)
        {
            vertices[i].label = -1;
        }

        pneighbors_start = hg.twohop_offsets[pvertexid];
        pneighbors_end = hg.twohop_offsets[pvertexid + 1];
        for (uint64_t i = pneighbors_start; i < pneighbors_end; i++)
        {
            phelper1 = hd.vertex_order_map[hg.twohop_neighbors[i]];

            if (phelper1 >= number_of_members)
            {
                vertices[phelper1].label = 0;
                hd.candidate_indegs[(*hd.remaining_count)++] = vertices[phelper1].indeg;
            }
        }
    }

    bool h_degree_pruning(CPU_Graph &hg, CPU_Data &hd, Vertex *vertices, int &total_vertices, int &number_of_candidates, int number_of_members, int &upper_bound, int &lower_bound, int &min_ext_deg)
    {
        int pvertexid;
        uint64_t pneighbors_start;
        uint64_t pneighbors_end;
        int phelper1;
        int num_val_cands;

        qsort(hd.candidate_indegs, (*hd.remaining_count), sizeof(int), h_sort_desc);

        if (h_calculate_LU_bounds(hd, upper_bound, lower_bound, min_ext_deg, vertices, number_of_members, (*hd.remaining_count)))
        {
            for (int i = 0; i < total_vertices; i++)
            {
                hd.vertex_order_map[vertices[i].vertexid] = -1;
            }
            return true;
        }

        for (int k = 0; k < number_of_members; k++)
        {
            if (!h_vert_isextendable_LU(vertices[k], number_of_members, upper_bound, lower_bound, min_ext_deg))
            {
                for (int i = 0; i < total_vertices; i++)
                {
                    hd.vertex_order_map[vertices[i].vertexid] = -1;
                }
                return true;
            }
        }

        (*hd.remaining_count) = 0;
        (*hd.removed_count) = 0;

        for (int i = number_of_members; i < total_vertices; i++)
        {
            if (vertices[i].label == 0 && h_cand_isvalid_LU(vertices[i], number_of_members, upper_bound, lower_bound, min_ext_deg))
            {
                hd.remaining_candidates[(*hd.remaining_count)++] = i;
            }
            else
            {
                hd.removed_candidates[(*hd.removed_count)++] = i;
            }
        }

        while ((*hd.remaining_count) > 0 && (*hd.removed_count) > 0)
        {
            if ((*hd.remaining_count) < (*hd.removed_count))
            {
                for (int i = 0; i < total_vertices; i++)
                {
                    vertices[i].exdeg = 0;
                }

                for (int i = 0; i < (*hd.remaining_count); i++)
                {
                    pvertexid = vertices[hd.remaining_candidates[i]].vertexid;
                    pneighbors_start = hg.onehop_offsets[pvertexid];
                    pneighbors_end = hg.onehop_offsets[pvertexid + 1];
                    for (uint64_t j = pneighbors_start; j < pneighbors_end; j++)
                    {
                        phelper1 = hd.vertex_order_map[hg.onehop_neighbors[j]];

                        if (phelper1 > -1)
                        {
                            vertices[phelper1].exdeg++;
                        }
                    }
                }
            }
            else
            {
                for (int i = 0; i < (*hd.removed_count); i++)
                {
                    pvertexid = vertices[hd.removed_candidates[i]].vertexid;
                    pneighbors_start = hg.onehop_offsets[pvertexid];
                    pneighbors_end = hg.onehop_offsets[pvertexid + 1];
                    for (uint64_t j = pneighbors_start; j < pneighbors_end; j++)
                    {
                        phelper1 = hd.vertex_order_map[hg.onehop_neighbors[j]];

                        if (phelper1 > -1)
                        {
                            vertices[phelper1].exdeg--;
                        }
                    }
                }
            }

            num_val_cands = 0;

            for (int k = 0; k < (*hd.remaining_count); k++)
            {
                if (h_cand_isvalid_LU(vertices[hd.remaining_candidates[k]], number_of_members, upper_bound, lower_bound, min_ext_deg))
                {
                    hd.candidate_indegs[num_val_cands++] = vertices[hd.remaining_candidates[k]].indeg;
                }
            }

            qsort(hd.candidate_indegs, num_val_cands, sizeof(int), h_sort_desc);

            if (h_calculate_LU_bounds(hd, upper_bound, lower_bound, min_ext_deg, vertices, number_of_members, num_val_cands))
            {
                for (int i = 0; i < total_vertices; i++)
                {
                    hd.vertex_order_map[vertices[i].vertexid] = -1;
                }
                return true;
            }

            for (int k = 0; k < number_of_members; k++)
            {
                if (!h_vert_isextendable_LU(vertices[k], number_of_members, upper_bound, lower_bound, min_ext_deg))
                {
                    for (int i = 0; i < total_vertices; i++)
                    {
                        hd.vertex_order_map[vertices[i].vertexid] = -1;
                    }
                    return true;
                }
            }

            num_val_cands = 0;
            (*hd.removed_count) = 0;

            for (int k = 0; k < (*hd.remaining_count); k++)
            {
                if (h_cand_isvalid_LU(vertices[hd.remaining_candidates[k]], number_of_members, upper_bound, lower_bound, min_ext_deg))
                {
                    hd.remaining_candidates[num_val_cands++] = hd.remaining_candidates[k];
                }
                else
                {
                    hd.removed_candidates[(*hd.removed_count)++] = hd.remaining_candidates[k];
                }
            }

            (*hd.remaining_count) = num_val_cands;
        }

        for (int i = 0; i < total_vertices; i++)
        {
            hd.vertex_order_map[vertices[i].vertexid] = -1;
        }

        for (int i = 0; i < (*hd.remaining_count); i++)
        {
            vertices[number_of_members + i] = vertices[hd.remaining_candidates[i]];
        }

        total_vertices = total_vertices - number_of_candidates + (*hd.remaining_count);
        number_of_candidates = (*hd.remaining_count);

        return false;
    }

    int h_lookahead_pruning(CPU_Graph &hg, CPU_Cliques &hc, CPU_Data &hd, Vertex *read_vertices, int tot_vert, int num_mem, int num_cand, uint64_t start)
    {
        int pvertexid;
        uint64_t pneighbors_start;
        uint64_t pneighbors_end;
        int phelper1;

        for (int i = 0; i < num_mem; i++)
        {
            if (read_vertices[start + i].indeg + read_vertices[start + i].exdeg < minimum_degrees[tot_vert])
            {
                return 0;
            }
        }

        for (int i = num_mem; i < tot_vert; i++)
        {
            hd.vertex_order_map[read_vertices[start + i].vertexid] = i;
        }

        for (int i = num_mem; i < tot_vert; i++)
        {
            pvertexid = read_vertices[start + i].vertexid;
            pneighbors_start = hg.twohop_offsets[pvertexid];
            pneighbors_end = hg.twohop_offsets[pvertexid + 1];
            for (uint64_t j = pneighbors_start; j < pneighbors_end; j++)
            {
                phelper1 = hd.vertex_order_map[hg.twohop_neighbors[j]];

                if (phelper1 >= num_mem)
                {
                    read_vertices[start + phelper1].lvl2adj++;
                }
            }
        }

        for (int i = num_mem; i < tot_vert; i++)
        {
            hd.vertex_order_map[read_vertices[start + i].vertexid] = -1;
        }

        for (int j = num_mem; j < tot_vert; j++)
        {
            if (read_vertices[start + j].lvl2adj < num_cand - 1 || read_vertices[start + j].indeg + read_vertices[start + j].exdeg < minimum_degrees[tot_vert])
            {
                return 0;
            }
        }

        h_write_clique(hc, read_vertices + start, tot_vert);
        return 1;
    }

    int h_remove_one_vertex(CPU_Graph &hg, CPU_Data &hd, Vertex *read_vertices, int &tot_vert, int &num_cand, int &num_mem, uint64_t start)
    {
        int pvertexid;
        uint64_t pneighbors_start;
        uint64_t pneighbors_end;
        int phelper1;
        int mindeg;
        bool failed_found;
        int removed_vertexid;

        mindeg = h_get_mindeg(num_mem);

        num_cand--;
        tot_vert--;
        removed_vertexid = read_vertices[start + tot_vert].vertexid;

        for (int i = 0; i < tot_vert; i++)
        {
            hd.vertex_order_map[read_vertices[start + i].vertexid] = i;
        }
        hd.vertex_order_map[removed_vertexid] = -1;

        failed_found = false;

        pvertexid = read_vertices[start + tot_vert].vertexid;
        pneighbors_start = hg.onehop_offsets[pvertexid];
        pneighbors_end = hg.onehop_offsets[pvertexid + 1];
        for (uint64_t i = pneighbors_start; i < pneighbors_end; i++)
        {
            phelper1 = hd.vertex_order_map[hg.onehop_neighbors[i]];

            if (phelper1 > -1)
            {
                read_vertices[start + phelper1].exdeg--;

                if (phelper1 < num_mem && read_vertices[start + phelper1].indeg + read_vertices[start + phelper1].exdeg < mindeg)
                {
                    failed_found = true;
                    break;
                }
            }
        }

        for (int i = 0; i < tot_vert; i++)
        {
            hd.vertex_order_map[read_vertices[start + i].vertexid] = -1;
        }
        hd.vertex_order_map[removed_vertexid] = -1;

        if (failed_found)
        {
            return 1;
        }

        return 0;
    }

    int h_add_one_vertex(CPU_Graph &hg, CPU_Data &hd, Vertex *vertices, int &total_vertices, int &number_of_candidates, int &number_of_members, int &upper_bound, int &lower_bound, int &min_ext_deg)
    {
        bool method_return;
        int pvertexid;
        uint64_t pneighbors_start;
        uint64_t pneighbors_end;
        uint64_t pneighbors_count;
        int phelper1;

        pvertexid = vertices[number_of_members].vertexid;

        vertices[number_of_members].label = 1;
        number_of_members++;
        number_of_candidates--;

        for (int i = 0; i < total_vertices; i++)
        {
            hd.vertex_order_map[vertices[i].vertexid] = i;
        }
        pneighbors_start = hg.onehop_offsets[pvertexid];
        pneighbors_end = hg.onehop_offsets[pvertexid + 1];
        pneighbors_count = pneighbors_end - pneighbors_start;
        for (uint64_t i = 0; i < pneighbors_count; i++)
        {
            phelper1 = hd.vertex_order_map[hg.onehop_neighbors[pneighbors_start + i]];

            if (phelper1 > -1)
            {
                vertices[phelper1].indeg++;
                vertices[phelper1].exdeg--;
            }
        }

        h_diameter_pruning(hg, hd, vertices, pvertexid, total_vertices, number_of_candidates, number_of_members);

        method_return = h_degree_pruning(hg, hd, vertices, total_vertices, number_of_candidates, number_of_members, upper_bound, lower_bound, min_ext_deg);

        if (method_return)
        {
            return 1;
        }

        return 0;
    }

    int h_critical_vertex_pruning(CPU_Graph &hg, CPU_Data &hd, Vertex *vertices, int &total_vertices, int &number_of_candidates, int &number_of_members, int &upper_bound, int &lower_bound, int &min_ext_deg)
    {
        int pvertexid;
        uint64_t pneighbors_start;
        uint64_t pneighbors_end;
        int phelper1;
        bool critical_fail;
        int number_of_crit_adj;
        int *adj_counters;
        bool method_return;

        for (int i = 0; i < total_vertices; i++)
        {
            hd.vertex_order_map[vertices[i].vertexid] = i;
        }

        adj_counters = new int[total_vertices];
        memset(adj_counters, 0, sizeof(int) * total_vertices);

        for (int k = 0; k < number_of_members; k++)
        {
            if (vertices[k].indeg + vertices[k].exdeg == minimum_degrees[number_of_members + lower_bound] && vertices[k].exdeg > 0)
            {
                pvertexid = vertices[k].vertexid;

                pneighbors_start = hg.onehop_offsets[pvertexid];
                pneighbors_end = hg.onehop_offsets[pvertexid + 1];
                for (uint64_t l = pneighbors_start; l < pneighbors_end; l++)
                {
                    phelper1 = hd.vertex_order_map[hg.onehop_neighbors[l]];

                    if (phelper1 >= number_of_members)
                    {
                        vertices[phelper1].label = 4;
                    }
                }
            }
        }

        for (int i = 0; i < total_vertices; i++)
        {
            hd.vertex_order_map[vertices[i].vertexid] = -1;
        }

        qsort(vertices + number_of_members, number_of_candidates, sizeof(Vertex), h_sort_vert_cv);

        number_of_crit_adj = 0;
        for (int i = number_of_members; i < total_vertices; i++)
        {
            if (vertices[i].label == 4)
            {
                number_of_crit_adj++;
            }
            else
            {
                break;
            }
        }

        for (int i = 0; i < total_vertices; i++)
        {
            hd.vertex_order_map[vertices[i].vertexid] = i;
        }

        if (number_of_crit_adj > 0)
        {
            for (int i = number_of_members; i < number_of_members + number_of_crit_adj; i++)
            {
                pvertexid = vertices[i].vertexid;

                pneighbors_start = hg.onehop_offsets[pvertexid];
                pneighbors_end = hg.onehop_offsets[pvertexid + 1];
                for (uint64_t k = pneighbors_start; k < pneighbors_end; k++)
                {
                    phelper1 = hd.vertex_order_map[hg.onehop_neighbors[k]];

                    if (phelper1 > -1)
                    {
                        vertices[phelper1].indeg++;
                        vertices[phelper1].exdeg--;
                    }
                }

                pneighbors_start = hg.twohop_offsets[pvertexid];
                pneighbors_end = hg.twohop_offsets[pvertexid + 1];
                for (uint64_t k = pneighbors_start; k < pneighbors_end; k++)
                {
                    phelper1 = hd.vertex_order_map[hg.twohop_neighbors[k]];

                    if (phelper1 > -1)
                    {
                        adj_counters[phelper1]++;
                    }
                }
            }

            critical_fail = false;

            for (int k = 0; k < number_of_members; k++)
            {
                if (adj_counters[k] != number_of_crit_adj)
                {
                    critical_fail = true;
                }
            }

            if (critical_fail)
            {
                for (int i = 0; i < total_vertices; i++)
                {
                    hd.vertex_order_map[vertices[i].vertexid] = -1;
                }
                delete[] adj_counters;
                return 2;
            }

            for (int k = number_of_members; k < number_of_members + number_of_crit_adj; k++)
            {
                if (adj_counters[k] < number_of_crit_adj - 1)
                {
                    critical_fail = true;
                }
            }

            if (critical_fail)
            {
                for (int i = 0; i < total_vertices; i++)
                {
                    hd.vertex_order_map[vertices[i].vertexid] = -1;
                }
                delete[] adj_counters;
                return 2;
            }

            for (int k = number_of_members; k < number_of_members + number_of_crit_adj; k++)
            {
                vertices[k].label = 1;
            }
            number_of_members += number_of_crit_adj;
            number_of_candidates -= number_of_crit_adj;
        }

        (*hd.remaining_count) = 0;

        for (int k = number_of_members; k < total_vertices; k++)
        {
            if (adj_counters[k] == number_of_crit_adj)
            {
                hd.candidate_indegs[(*hd.remaining_count)++] = vertices[k].indeg;
            }
            else
            {
                vertices[k].label = -1;
            }
        }

        method_return = h_degree_pruning(hg, hd, vertices, total_vertices, number_of_candidates, number_of_members, upper_bound, lower_bound, min_ext_deg);

        delete[] adj_counters;

        if (method_return)
        {
            return 1;
        }

        return 0;
    }

    void h_expand_level(CPU_Graph &hg, CPU_Data &hd, CPU_Cliques &hc, Vertex *read_vertices, size_t read_vertices_count,
                        std::chrono::steady_clock::time_point st)
    {
        uint64_t start;
        uint64_t end;
        int tot_vert;
        int num_mem;
        int num_cand;
        int expansions;
        int number_of_covered;
        Vertex *vertices;
        int number_of_members;
        int number_of_candidates;
        int total_vertices;
        int min_ext_deg;
        int lower_bound;
        int upper_bound;
        int method_return;
        int index;

        (*hd.maximal_expansion) = true;
        {
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

            method_return = h_lookahead_pruning(hg, hc, hd, read_vertices, tot_vert, num_mem, num_cand, start);
            if (method_return)
            {
                return;
            }

            for (int j = number_of_covered; j < expansions; j++)
            {
                if (j != number_of_covered)
                {
                    method_return = h_remove_one_vertex(hg, hd, read_vertices, tot_vert, num_cand, num_mem, start);
                    if (method_return)
                    {
                        break;
                    }
                }

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
                    for (int j = num_mem + 1; j <= num_mem + number_of_covered; j++)
                    {
                        vertices[j].label = 0;
                    }
                }

                method_return = h_add_one_vertex(hg, hd, vertices, total_vertices, number_of_candidates, number_of_members, upper_bound, lower_bound, min_ext_deg);

                if (method_return == 1)
                {
                    if (number_of_members >= minimum_clique_size)
                    {
                        h_check_for_clique(hc, vertices, number_of_members);
                    }

                    delete[] vertices;
                    continue;
                }

                method_return = h_critical_vertex_pruning(hg, hd, vertices, total_vertices, number_of_candidates, number_of_members, upper_bound, lower_bound, min_ext_deg);

                if (method_return == 2)
                {
                    delete[] vertices;
                    continue;
                }

                if (number_of_members >= minimum_clique_size)
                {
                    h_check_for_clique(hc, vertices, number_of_members);
                }

                if (method_return == 1)
                {
                    delete[] vertices;
                    continue;
                }

                qsort(vertices, total_vertices, sizeof(Vertex), h_sort_vert_Q);

                if (number_of_candidates > 0)
                {
                    for (int k = 0; k < total_vertices; k++)
                    {
                        vertices[k].lvl2adj = 0;
                    }

                    if (time_over(st))
                    {
                        enqueue_task(vertices, total_vertices, active_big_root_lineage);
                    }
                    else
                    {
                        h_expand_level(hg, hd, hc, vertices, total_vertices, st);
                        delete[] vertices;
                    }
                }
                else
                {
                    delete[] vertices;
                }
            }
        }
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

public:
    virtual QCTask *task_spawn(VertexID &index)
    {
        const bool from_big_root = is_large_top_level_root(index);
        const uint64_t before_count = local_hc.cliques_count;
        size_t task_vertices_count;
        Vertex *vertices = h_build_initial_task(*hg, local_hd, local_hc, index, task_vertices_count);
        const uint64_t spawned_cliques = local_hc.cliques_count - before_count;
        if (from_big_root)
        {
            qc_big_root_cliques_found.fetch_add(spawned_cliques, std::memory_order_relaxed);
        }
        QCTask *t = new QCTask();
        if (vertices != nullptr)
        {
            t->context = QCContext(vertices, task_vertices_count);
            t->context.from_big_root = from_big_root;
            if (from_big_root)
            {
                qc_big_root_tasks_spawned.fetch_add(1, std::memory_order_relaxed);
            }
        }
        return t;
    }

    virtual void compute(QCContext &context) override
    {
        if (context.vertices == nullptr || context.num_vertices == 0)
            return;
        const uint64_t before_count = local_hc.cliques_count;
        active_big_root_lineage = context.from_big_root;
        h_expand_level(*hg, local_hd, local_hc, context.vertices, context.num_vertices, start_task_timer());
        active_big_root_lineage = false;
        if (context.from_big_root)
        {
            qc_big_root_tasks_executed.fetch_add(1, std::memory_order_relaxed);
            qc_big_root_cliques_found.fetch_add(local_hc.cliques_count - before_count, std::memory_order_relaxed);
        }
    }
};

#endif
