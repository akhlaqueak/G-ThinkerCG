#include "global.h"
#include "master.h"
#include "qc_task.h"
#include "qc_gpu_context.h"
#include "qc_cpu_worker.h"
#include "host_functions.h"
CommandLine cmd;
ull spilled_tasks;

class QCApp : public Master<QCCPUWorker, QCGPUContext>
{
public:
    QCApp()
    {
        num_cpu_workers = cmd.GetOptionIntValue("-cpu", 28);
        num_gpu_workers = cmd.GetOptionIntValue("-gpu", 1);
        tasks_per_fetch_gpu_worker_g = cmd.GetOptionIntValue("-gpuchunk", 500000);
        tasks_per_fetch_g = cmd.GetOptionIntValue("-cpuchunk", 50);
        ui eta_ = cmd.GetOptionIntValue("-eta", 1000);
        std::string fp = cmd.GetOptionValue("-dg", "./data/com-friendster.ungraph.txt.bin");
        std::cout.imbue(std::locale());
        cout << " ======= Parameters ========" << endl;
        cout << "Graph: " << fp << endl;
        cout << "cpu workers: " << num_cpu_workers << endl;
        cout << "gpu workers: " << num_gpu_workers << endl;
        cout << "eta: " << eta_ << endl;
        cout << "cpu chunk: " << tasks_per_fetch_g << endl;
        cout << "gpu chunk: " << tasks_per_fetch_gpu_worker_g << endl;
        cout << " ======= ********** ========" << endl;
        ifstream graph_stream(fp, ios::in);

        CPU_Graph hg = CPU_Graph(graph_stream);
        CPU_Data hd;
        CPU_Cliques hc;
        GPU_Data dd;
        allocate_memory(hd, dd, hc, hg);

        eta_ *= N_WARPS;
        cudaMemcpyToSymbol(eta, &eta_, sizeof(ui));
        QCTask* t= initialize_tasks(hg, hd);
        h_expand_level(hg, hd, hc, t);
    }
    void h_expand_level(CPU_Graph &hg, CPU_Data &hd, CPU_Cliques &hc, QCTask* t)
    {
        // initiate the variables containing the location of the read and write task vectors, done in an alternating, odd-even manner like the c-intersection of cuTS
        uint64_t *read_count;
        uint64_t *read_offsets;
        Vertex *read_vertices;
        uint64_t *write_count;
        uint64_t *write_offsets;
        Vertex *write_vertices;

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

        read_vertices = t->context.vertices;

        // set to false later if task is generated indicating non-maximal expansion
        (*hd.maximal_expansion) = true;

        // CURRENT LEVEL
        // for (int i = 0; i < *read_count; i++)
        {
            // get information of vertices being handled within tasks
            start = 0;
            end = t->context.num_vertices;
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
            for (uint64_t j = start + num_mem; j < end; j++)
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
                    vertices[index] = read_vertices[start + index];
                }
                vertices[number_of_members] = read_vertices[start + total_vertices - 1];
                for (; index < total_vertices - 1; index++)
                {
                    vertices[index + 1] = read_vertices[start + index];
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

                    delete vertices;
                    continue;
                }

                // CRITICAL VERTEX PRUNING
                method_return = h_critical_vertex_pruning(hg, hd, vertices, total_vertices, number_of_candidates, number_of_members, upper_bound, lower_bound, min_ext_deg);

                // if critical fail continue onto next iteration
                if (method_return == 2)
                {
                    delete vertices;
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
                    delete vertices;
                    continue;
                }

                // WRITE TO TASKS
                // sort vertices so that lowest degree vertices are first in enumeration order before writing to tasks
                qsort(vertices, total_vertices, sizeof(Vertex), h_sort_vert_Q);

                if (number_of_candidates > 0)
                {
                    // h_write_to_tasks(hd, vertices, total_vertices, write_vertices, write_offsets, write_count);
                    for (int k = 0; k < total_vertices; k++)
                        vertices[k].lvl2adj = 0;
                    QCTask *t = new QCTask();
                    t->context.vertices = new Vertex[total_vertices];
                    t->context.num_vertices= total_vertices;
                    std::copy(vertices, vertices+total_vertices, t->context.vertices);
                    add_task(t);
                }

                delete[] vertices;
            }
        }
        delete t;
        // (*hd.current_level)++;
    }
    // processes 0th level of expansion
    QCTask* initialize_tasks(CPU_Graph &hg, CPU_Data &hd)
    {
        // intersection
        int pvertexid;
        uint64_t pneighbors_start;
        uint64_t pneighbors_end;
        int phelper1;

        // cover pruning
        int maximum_degree;
        int maximum_degree_index;

        // vertices information
        int total_vertices;
        int number_of_candidates;
        Vertex *vertices;

        (*hd.remaining_count) = 0;
        (*hd.removed_count) = 0;

        // initialize vertices
        total_vertices = hg.number_of_vertices;
        vertices = new Vertex[total_vertices];
        number_of_candidates = total_vertices;
        for (int i = 0; i < total_vertices; i++)
        {
            vertices[i].vertexid = i;
            vertices[i].indeg = 0;
            vertices[i].exdeg = hg.onehop_offsets[i + 1] - hg.onehop_offsets[i];
            vertices[i].lvl2adj = hg.twohop_offsets[i + 1] - hg.twohop_offsets[i];
            if (vertices[i].exdeg >= minimum_degrees[minimum_clique_size] && vertices[i].lvl2adj >= minimum_clique_size - 1)
            {
                vertices[i].label = 0;
                hd.remaining_candidates[(*hd.remaining_count)++] = i;
            }
            else
            {
                vertices[i].label = -1;
                hd.removed_candidates[(*hd.removed_count)++] = i;
            }
        }

        // DEGREE-BASED PRUNING
        // update while half of vertices have been removed
        while ((*hd.remaining_count) < number_of_candidates / 2)
        {
            number_of_candidates = (*hd.remaining_count);

            for (int i = 0; i < number_of_candidates; i++)
            {
                vertices[hd.remaining_candidates[i]].exdeg = 0;
            }

            for (int i = 0; i < number_of_candidates; i++)
            {
                // in 0th level id is same as position in vertices as all vertices are in vertices, see last block
                pvertexid = hd.remaining_candidates[i];
                pneighbors_start = hg.onehop_offsets[pvertexid];
                pneighbors_end = hg.onehop_offsets[pvertexid + 1];
                for (int j = pneighbors_start; j < pneighbors_end; j++)
                {
                    phelper1 = hg.onehop_neighbors[j];
                    if (vertices[phelper1].label == 0)
                    {
                        vertices[phelper1].exdeg++;
                    }
                }
            }

            (*hd.remaining_count) = 0;
            (*hd.removed_count) = 0;

            // remove more vertices based on updated degrees
            for (int i = 0; i < number_of_candidates; i++)
            {
                phelper1 = hd.remaining_candidates[i];
                if (vertices[phelper1].exdeg >= minimum_degrees[minimum_clique_size])
                {
                    hd.remaining_candidates[(*hd.remaining_count)++] = phelper1;
                }
                else
                {
                    vertices[phelper1].label = -1;
                    hd.removed_candidates[(*hd.removed_count)++] = phelper1;
                }
            }
        }
        number_of_candidates = (*hd.remaining_count);

        // update degrees based on last round of removed vertices
        int removed_start = 0;
        while ((*hd.removed_count) > removed_start)
        {
            pvertexid = hd.removed_candidates[removed_start];
            pneighbors_start = hg.onehop_offsets[pvertexid];
            pneighbors_end = hg.onehop_offsets[pvertexid + 1];

            for (int j = pneighbors_start; j < pneighbors_end; j++)
            {
                phelper1 = hg.onehop_neighbors[j];

                if (vertices[phelper1].label == 0)
                {
                    vertices[phelper1].exdeg--;

                    if (vertices[phelper1].exdeg < minimum_degrees[minimum_clique_size])
                    {
                        vertices[phelper1].label = -1;
                        number_of_candidates--;
                        hd.removed_candidates[(*hd.removed_count)++] = phelper1;
                    }
                }
            }
            removed_start++;
        }

        // FIRST ROUND COVER PRUNING
        // find cover vertex
        maximum_degree = 0;
        maximum_degree_index = 0;
        for (int i = 0; i < total_vertices; i++)
        {
            if (vertices[i].label == 0)
            {
                if (vertices[i].exdeg > maximum_degree)
                {
                    maximum_degree = vertices[i].exdeg;
                    maximum_degree_index = i;
                }
            }
        }
        vertices[maximum_degree_index].label = 3;

        // find all covered vertices
        pneighbors_start = hg.onehop_offsets[maximum_degree_index];
        pneighbors_end = hg.onehop_offsets[maximum_degree_index + 1];
        for (int i = pneighbors_start; i < pneighbors_end; i++)
        {
            pvertexid = hg.onehop_neighbors[i];
            if (vertices[pvertexid].label == 0)
            {
                vertices[pvertexid].label = 2;
            }
        }

        // sort enumeration order before writing to tasks
        qsort(vertices, total_vertices, sizeof(Vertex), h_sort_vert_Q);
        total_vertices = number_of_candidates;
        for (int j = 0; j < total_vertices; j++)
            vertices[j].lvl2adj = 0;
        QCTask* t=new QCTask();
        t->context.num_vertices = total_vertices;
        t->context.vertices = vertices;
        return t;
    }

    ui get_results()
    {
        ui res = 0;
        using GPUWorkerT = GPUWorker<QCGPUContext>;
        while (workers_list.size())
        {
            WorkerT *w = (WorkerT *)workers_list.dequeue();
            QCCPUWorker *cw = dynamic_cast<QCCPUWorker *>(w);
            GPUWorkerT *gw = dynamic_cast<GPUWorkerT *>(w);

            // if (cw)
            //     res += cw->total_counts;
            // else if (gw)
            // {
            //     res += gw->getContext()->get_results();
            //     spilled_tasks = gw->spilled_tasks;
            // }
        }
        return res;
    }
};

int main(int argc, char *argv[])
{
    cmd = CommandLine(argc, argv);

    QCApp app;
    Timer t;
    app.run();
    cout << "Total time (s): " << t.elapsed() / 1e6 << endl;
    cout << "Total count: " << app.get_results() << endl;
    cout << "Total spilled vertices: " << spilled_tasks << endl;

    return 0;
}
