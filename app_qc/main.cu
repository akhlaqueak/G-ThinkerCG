#include "global.h"
#include "master.h"
#include "host_functions.h"
#include "qc_task.h"
#include "qc_gpu_context.h"
#include "qc_cpu_worker.h"
CommandLine cmd;
ull spilled_tasks;
CPU_Data hd;
GPU_Data dd;
CPU_Cliques hc;
CPU_Graph *hg = nullptr;

class QCApp : public Master<QCCPUWorker, QCGPUContext>
{
public:
    QCApp()
    {
        std::string graph_file = cmd.GetOptionValue("-g");
        minimum_degree_ratio = cmd.GetOptionDoubleValue("-gamma", 0.5);
        minimum_clique_size = cmd.GetOptionIntValue("-k", 10);
        std::string output_file = cmd.GetOptionValue("-o", "output.txt");
        scheduling_toggle = cmd.GetOptionIntValue("-sched", 0);
        num_cpu_workers = cmd.GetOptionIntValue("-cpu", 28);
        num_gpu_workers = cmd.GetOptionIntValue("-gpu", 1);
        tasks_per_fetch_gpu_worker_g = cmd.GetOptionIntValue("-gpuchunk", 500000);
        tasks_per_fetch_g = cmd.GetOptionIntValue("-cpuchunk", 50);
        ui eta_ = cmd.GetOptionIntValue("-eta", 1000);

        std::cout.imbue(std::locale());
        eta_ *= N_WARPS;
        cudaMemcpyToSymbol(eta, &eta_, sizeof(ui));

        ifstream graph_stream(graph_file, ios::in);
        if (!graph_stream.is_open())
        {
            cout << "invalid graph file" << endl;
        }

        if (minimum_degree_ratio < .5 || minimum_degree_ratio > 1)
        {
            cout << "minimum degree ratio must be between .5 and 1 inclusive" << endl;
            minimum_degree_ratio = 0.5;
        }

        if (minimum_clique_size <= 1)
        {
            cout << "minimum size must be greater than 1" << endl;
            minimum_clique_size = 10;
        }

        if (!(scheduling_toggle == 0 || scheduling_toggle == 1))
        {
            cout << "scheduling toggle must be 0 or 1" << endl;
            scheduling_toggle = 0;
        }

        cout << " ======= Parameters ========" << endl;
        cout << "Graph: " << graph_file << endl;
        cout << "Gamma: " << minimum_degree_ratio << endl;
        cout << "Min size: " << minimum_clique_size << endl;
        cout << "Output: " << output_file << endl;
        cout << "Scheduling: " << (scheduling_toggle == 0 ? "dynamic" : "static") << endl;
        cout << "cpu workers: " << num_cpu_workers << endl;
        cout << "gpu workers: " << num_gpu_workers << endl;
        cout << "eta: " << eta_ << endl;
        cout << "cpu chunk: " << tasks_per_fetch_g << endl;
        cout << "gpu chunk: " << tasks_per_fetch_gpu_worker_g << endl;
        cout << " ======= ********** ========" << endl;

        // TIME
        auto start = chrono::high_resolution_clock::now();

        // GRAPH / MINDEGS
        cout << ">:PRE-PROCESSING" << endl;
        hg = new CPU_Graph(graph_stream);
        CPU_Graph &graph = *hg;
        cout << "|V| = " << graph.number_of_vertices << endl;
        cout << "|E| = " << graph.number_of_edges << endl;
        graph_stream.close();
        calculate_minimum_degrees(graph);

        // TIME
        auto stop = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::milliseconds>(stop - start);

        allocate_memory(hd, dd, hc, graph);
        cudaDeviceSynchronize();

        initialize_tasks(graph, hd);

        for(ui i=0;i<hd.initial_vertices_count; i++){
            data_array.push_back(new ui(i));
        }
        
        chkerr(cudaMalloc((void **)&dd.initial_vertices, sizeof(Vertex) * hd.initial_vertices_count));
        chkerr(cudaMemcpy(dd.initial_vertices, hd.initial_vertices, sizeof(Vertex) * hd.initial_vertices_count, cudaMemcpyHostToDevice));
        chkerr(cudaMalloc((void **)&dd.initial_vertices_count, sizeof(uint64_t)));
        uint64_t initial_vertices_count_u64 = hd.initial_vertices_count;
        chkerr(cudaMemcpy(dd.initial_vertices_count, &initial_vertices_count_u64, sizeof(uint64_t), cudaMemcpyHostToDevice));
        chkerr(cudaMalloc((void **)&dd.initial_order_map, sizeof(int) * graph.number_of_vertices));
        chkerr(cudaMemcpy(dd.initial_order_map, hd.initial_order_map, sizeof(int) * graph.number_of_vertices, cudaMemcpyHostToDevice));

        cout << "--->:LOADING TIME: " << duration.count() << " ms" << endl;
        // h_expand_level(graph, hd, hc, hd.initial_vertices, hd.initial_vertices_count);
    }
    ~QCApp()
    {
        delete hg;
        hg = nullptr;
    }

    void merge_cpu_cliques()
    {
        auto workers = workers_list.queue_;
        while (!workers.empty())
        {
            WorkerT *w = (WorkerT *)workers.front();
            workers.pop();

            QCCPUWorker *cw = dynamic_cast<QCCPUWorker *>(w);
            if (cw)
            {
                cw->merge_local_cliques_into(hc);
            }
        }
    }
      // processes 0th level of expansion
    void initialize_tasks(CPU_Graph &hg, CPU_Data &hd)
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
        hd.initial_vertices = vertices;
        hd.initial_vertices_count = static_cast<size_t>(total_vertices);
        memset(hd.initial_order_map, -1, sizeof(int) * hg.number_of_vertices);
        for (int i = 0; i < total_vertices; i++)
        {
            hd.initial_order_map[hd.initial_vertices[i].vertexid] = i;
        }
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

    // allocates memory for the data structures on the host and device
    void allocate_memory(CPU_Data &hd, GPU_Data &dd, CPU_Cliques &hc, CPU_Graph &hg)
    {
        // GPU GRAPH
        chkerr(cudaMalloc((void **)&dd.number_of_vertices, sizeof(int)));
        chkerr(cudaMalloc((void **)&dd.number_of_edges, sizeof(uint64_t)));
        chkerr(cudaMalloc((void **)&dd.onehop_neighbors, sizeof(int) * hg.number_of_edges));
        chkerr(cudaMalloc((void **)&dd.onehop_offsets, sizeof(uint64_t) * (hg.number_of_vertices + 1)));
        chkerr(cudaMalloc((void **)&dd.twohop_neighbors, sizeof(int) * hg.number_of_lvl2adj));
        chkerr(cudaMalloc((void **)&dd.twohop_offsets, sizeof(uint64_t) * (hg.number_of_vertices + 1)));

        chkerr(cudaMemcpy(dd.number_of_vertices, &(hg.number_of_vertices), sizeof(int), cudaMemcpyHostToDevice));
        chkerr(cudaMemcpy(dd.number_of_edges, &(hg.number_of_edges), sizeof(uint64_t), cudaMemcpyHostToDevice));
        chkerr(cudaMemcpy(dd.onehop_neighbors, hg.onehop_neighbors, sizeof(int) * hg.number_of_edges, cudaMemcpyHostToDevice));
        chkerr(cudaMemcpy(dd.onehop_offsets, hg.onehop_offsets, sizeof(uint64_t) * (hg.number_of_vertices + 1), cudaMemcpyHostToDevice));
        chkerr(cudaMemcpy(dd.twohop_neighbors, hg.twohop_neighbors, sizeof(int) * hg.number_of_lvl2adj, cudaMemcpyHostToDevice));
        chkerr(cudaMemcpy(dd.twohop_offsets, hg.twohop_offsets, sizeof(uint64_t) * (hg.number_of_vertices + 1), cudaMemcpyHostToDevice));

        // CPU DATA
        hd.tasks1_count = new uint64_t;
        hd.tasks1_offset = new uint64_t[EXPAND_THRESHOLD + 1];
        hd.tasks1_vertices = new Vertex[TASKS_SIZE];

        hd.tasks1_offset[0] = 0;
        (*(hd.tasks1_count)) = 0;

        hd.tasks2_count = new uint64_t;
        hd.tasks2_offset = new uint64_t[EXPAND_THRESHOLD + 1];
        hd.tasks2_vertices = new Vertex[TASKS_SIZE];

        hd.tasks2_offset[0] = 0;
        (*(hd.tasks2_count)) = 0;

        hd.buffer_count = new uint64_t;
        hd.buffer_offset = new uint64_t[BUFFER_OFFSET_SIZE];
        hd.buffer_vertices = new Vertex[BUFFER_SIZE];

        hd.buffer_offset[0] = 0;
        (*(hd.buffer_count)) = 0;

        hd.current_level = new uint64_t;
        hd.maximal_expansion = new bool;
        hd.dumping_cliques = new bool;

        (*hd.current_level) = 0;
        (*hd.maximal_expansion) = false;
        (*hd.dumping_cliques) = false;

        hd.vertex_order_map = new int[hg.number_of_vertices];
        hd.remaining_candidates = new int[hg.number_of_vertices];
        hd.removed_candidates = new int[hg.number_of_vertices];
        hd.remaining_count = new int;
        hd.removed_count = new int;
        hd.candidate_indegs = new int[hg.number_of_vertices];
        hd.initial_order_map = new int[hg.number_of_vertices];
        hd.initial_vertices = nullptr;
        hd.initial_vertices_count = 0;

        memset(hd.vertex_order_map, -1, sizeof(int) * hg.number_of_vertices);
        memset(hd.initial_order_map, -1, sizeof(int) * hg.number_of_vertices);

        // GPU DATA
        chkerr(cudaMalloc((void **)&dd.current_level, sizeof(uint64_t)));

        chkerr(cudaMalloc((void **)&dd.wtasks_count, sizeof(uint64_t) * NUMBER_OF_WARPS));
        chkerr(cudaMalloc((void **)&dd.wtasks_offset, (sizeof(uint64_t) * WTASKS_OFFSET_SIZE) * NUMBER_OF_WARPS));
        chkerr(cudaMalloc((void **)&dd.wtasks_vertices, (sizeof(Vertex) * WTASKS_SIZE) * NUMBER_OF_WARPS));

        chkerr(cudaMemset(dd.wtasks_offset, 0, (sizeof(uint64_t) * WTASKS_OFFSET_SIZE) * NUMBER_OF_WARPS));
        chkerr(cudaMemset(dd.wtasks_count, 0, sizeof(uint64_t) * NUMBER_OF_WARPS));

        chkerr(cudaMalloc((void **)&dd.global_vertices, (sizeof(Vertex) * WVERTICES_SIZE) * NUMBER_OF_WARPS));

        chkerr(cudaMalloc((void **)&dd.removed_candidates, (sizeof(int) * WVERTICES_SIZE) * NUMBER_OF_WARPS));
        chkerr(cudaMalloc((void **)&dd.lane_removed_candidates, (sizeof(int) * WVERTICES_SIZE) * NUMBER_OF_WARPS));

        chkerr(cudaMalloc((void **)&dd.remaining_candidates, (sizeof(Vertex) * WVERTICES_SIZE) * NUMBER_OF_WARPS));
        chkerr(cudaMalloc((void **)&dd.lane_remaining_candidates, (sizeof(int) * WVERTICES_SIZE) * NUMBER_OF_WARPS));

        chkerr(cudaMalloc((void **)&dd.candidate_indegs, (sizeof(int) * WVERTICES_SIZE) * NUMBER_OF_WARPS));
        chkerr(cudaMalloc((void **)&dd.lane_candidate_indegs, (sizeof(int) * WVERTICES_SIZE) * NUMBER_OF_WARPS));

        chkerr(cudaMalloc((void **)&dd.adjacencies, (sizeof(int) * WVERTICES_SIZE) * NUMBER_OF_WARPS));

        chkerr(cudaMalloc((void **)&dd.minimum_degree_ratio, sizeof(double)));
        chkerr(cudaMalloc((void **)&dd.minimum_degrees, sizeof(int) * (hg.number_of_vertices + 1)));
        chkerr(cudaMalloc((void **)&dd.minimum_clique_size, sizeof(int)));
        chkerr(cudaMalloc((void **)&dd.scheduling_toggle, sizeof(int)));

        chkerr(cudaMemcpy(dd.minimum_degree_ratio, &minimum_degree_ratio, sizeof(double), cudaMemcpyHostToDevice));
        chkerr(cudaMemcpy(dd.minimum_degrees, minimum_degrees, sizeof(int) * (hg.number_of_vertices + 1), cudaMemcpyHostToDevice));
        chkerr(cudaMemcpy(dd.minimum_clique_size, &minimum_clique_size, sizeof(int), cudaMemcpyHostToDevice));
        chkerr(cudaMemcpy(dd.scheduling_toggle, &scheduling_toggle, sizeof(int), cudaMemcpyHostToDevice));

        chkerr(cudaMalloc((void **)&dd.total_tasks, sizeof(int)));

        chkerr(cudaMemset(dd.total_tasks, 0, sizeof(int)));

        // CPU CLIQUES
        hc.cliques_count = new uint64_t;
        hc.cliques_vertex = new int[CLIQUES_SIZE];
        hc.cliques_offset = new uint64_t[CLIQUES_OFFSET_SIZE];

        hc.cliques_offset[0] = 0;
        (*(hc.cliques_count)) = 0;

        // GPU CLIQUES
        chkerr(cudaMalloc((void **)&dd.cliques_count, sizeof(uint64_t)));
        chkerr(cudaMalloc((void **)&dd.cliques_vertex, sizeof(int) * CLIQUES_SIZE));
        chkerr(cudaMalloc((void **)&dd.cliques_offset, sizeof(uint64_t) * CLIQUES_OFFSET_SIZE));

        chkerr(cudaMemset(dd.cliques_offset, 0, sizeof(uint64_t)));
        chkerr(cudaMemset(dd.cliques_count, 0, sizeof(uint64_t)));

        chkerr(cudaMalloc((void **)&dd.wcliques_count, sizeof(uint64_t) * NUMBER_OF_WARPS));
        chkerr(cudaMalloc((void **)&dd.wcliques_offset, (sizeof(uint64_t) * WCLIQUES_OFFSET_SIZE) * NUMBER_OF_WARPS));
        chkerr(cudaMalloc((void **)&dd.wcliques_vertex, (sizeof(int) * WCLIQUES_SIZE) * NUMBER_OF_WARPS));

        chkerr(cudaMemset(dd.wcliques_offset, 0, (sizeof(uint64_t) * WCLIQUES_OFFSET_SIZE) * NUMBER_OF_WARPS));
        chkerr(cudaMemset(dd.wcliques_count, 0, sizeof(uint64_t) * NUMBER_OF_WARPS));

        chkerr(cudaMalloc((void **)&dd.total_cliques, sizeof(ull)));

        chkerr(cudaMemset(dd.total_cliques, 0, sizeof(ull)));

        chkerr(cudaMalloc((void **)&dd.buffer_offset_start, sizeof(uint64_t)));
        chkerr(cudaMalloc((void **)&dd.buffer_start, sizeof(uint64_t)));
        chkerr(cudaMalloc((void **)&dd.cliques_offset_start, sizeof(uint64_t)));
        chkerr(cudaMalloc((void **)&dd.cliques_start, sizeof(uint64_t)));

        // task scheduling
        chkerr(cudaMalloc((void **)&dd.current_task, sizeof(int)));
    }
};

int main(int argc, char *argv[])
{
    cmd = CommandLine(argc, argv);

    string temp_filename = "t_cliques.txt";
    ofstream temp_results(temp_filename);

    QCApp app;
    Timer t;
    app.run();
    app.merge_cpu_cliques();
    chkerr(cudaDeviceSynchronize());

    dump_cliques(hc, dd, temp_results);

    //     // TIME
    // auto start1 = chrono::high_resolution_clock::now();


    // // RM NON-MAX
    string out_file = cmd.GetOptionValue("-o", "output.txt");
    RemoveNonMax(temp_filename.c_str(), out_file.c_str());

    // TIME
    // auto stop1 = chrono::high_resolution_clock::now();
    // auto duration1 = chrono::duration_cast<chrono::milliseconds>(stop1 - start1);
    // cout << "--->:REMOVE NON-MAX TIME: " << duration1.count() << " ms" << endl;


    ull cliques_count = 0;
    chkerr(cudaMemcpy(&cliques_count, dd.cliques_count, sizeof(ull), cudaMemcpyDeviceToHost));

    cout << "Total time (s): " << t.elapsed() / 1e6 << endl;
    cout << "Total count: " << cliques_count << endl;
    cout << "Total spilled vertices: " << spilled_tasks << endl;

    return 0;
}
