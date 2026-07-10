#ifndef QC_GPU_APP
#define QC_GPU_APP

#include "system/appbase.h"
#include "system/buffer.h"
#include "system/util.h"
#include "qc_support.h"

extern CPU_Data hd;
extern GPU_Data dd;
extern CPU_Cliques hc;
extern CPU_Graph *hg;

struct Warp_Data
{
    uint64_t start[WARPS_PER_BLOCK];
    uint64_t end[WARPS_PER_BLOCK];
    int tot_vert[WARPS_PER_BLOCK];
    int num_mem[WARPS_PER_BLOCK];
    int number_of_covered[WARPS_PER_BLOCK];
    int num_cand[WARPS_PER_BLOCK];
    int expansions[WARPS_PER_BLOCK];

    int number_of_members[WARPS_PER_BLOCK];
    int number_of_candidates[WARPS_PER_BLOCK];
    int total_vertices[WARPS_PER_BLOCK];

    Vertex shared_vertices[VERTICES_SIZE * WARPS_PER_BLOCK];

    int removed_count[WARPS_PER_BLOCK];
    int remaining_count[WARPS_PER_BLOCK];
    int num_val_cands[WARPS_PER_BLOCK];
    int rw_counter[WARPS_PER_BLOCK];

    int min_ext_deg[WARPS_PER_BLOCK];
    int lower_bound[WARPS_PER_BLOCK];
    int upper_bound[WARPS_PER_BLOCK];

    int tightened_upper_bound[WARPS_PER_BLOCK];
    int min_clq_indeg[WARPS_PER_BLOCK];
    int min_indeg_exdeg[WARPS_PER_BLOCK];
    int min_clq_totaldeg[WARPS_PER_BLOCK];
    int sum_clq_indeg[WARPS_PER_BLOCK];
    int sum_candidate_indeg[WARPS_PER_BLOCK];

    bool invalid_bounds[WARPS_PER_BLOCK];
    bool success[WARPS_PER_BLOCK];

    int number_of_crit_adj[WARPS_PER_BLOCK];

    // for dynamic intersection
    int count[WARPS_PER_BLOCK];
};

struct Local_Data
{
    Vertex *vertices;
};

__device__ int d_lookahead_pruning(GPU_Data &dd, Warp_Data &wd, Local_Data &ld);
__device__ int d_remove_one_vertex(GPU_Data &dd, Warp_Data &wd, Local_Data &ld);
__device__ int d_add_one_vertex(GPU_Data &dd, Warp_Data &wd, Local_Data &ld);
__device__ int d_critical_vertex_pruning(GPU_Data &dd, Warp_Data &wd, Local_Data &ld);
__device__ void d_check_for_clique(GPU_Data &dd, Warp_Data &wd, Local_Data &ld);
__device__ void d_write_to_tasks(GPU_Data &dd, Warp_Data &wd, Local_Data &ld);
__device__ void d_diameter_pruning(GPU_Data &dd, Warp_Data &wd, Local_Data &ld, int pvertexid);
__device__ void d_diameter_pruning_cv(GPU_Data &dd, Warp_Data &wd, Local_Data &ld, int number_of_crit_adj);
__device__ void d_calculate_LU_bounds(GPU_Data &dd, Warp_Data &wd, Local_Data &ld, int number_of_candidates);
__device__ bool d_degree_pruning(GPU_Data &dd, Warp_Data &wd, Local_Data &ld);
__device__ void d_sort(Vertex *target, int size, int (*func)(Vertex &, Vertex &));
__device__ void d_sort_i(int *target, int size, int (*func)(int, int));
__device__ int d_sort_vert_Q(Vertex &v1, Vertex &v2);
__device__ int d_sort_vert_cv(Vertex &v1, Vertex &v2);
__device__ int d_sort_degs(int n1, int n2);
__device__ int d_bsearch_array(int *search_array, int array_size, int search_number);
__device__ bool d_cand_isvalid_LU(Vertex &vertex, GPU_Data &dd, Warp_Data &wd, Local_Data &ld);
__device__ bool d_vert_isextendable_LU(Vertex &vertex, GPU_Data &dd, Warp_Data &wd, Local_Data &ld);
__device__ int d_get_mindeg(int number_of_members, GPU_Data &dd);
__device__ void d_print_vertices(Vertex *vertices, int size);
__global__ void transfer_cliques(GPU_Data dd);

template <class IndexType>
class QCBuffer : public BufferBase<IndexType>
{

public:
    Label *label = nullptr;
    int *indeg = nullptr;
    int *exdeg = nullptr;
    int *lvl2adj = nullptr;

    static ull sizeOf()
    {
        return BufferBase<IndexType>::sizeOf() + sizeof(Label) + 3 * sizeof(int);
    }
    void allocateMemory(ull sz)
    {
        cout << "Device buffers size: " << sz << endl;
        BufferBase<IndexType>::allocateMemory(sz);
        chkerr(cudaMalloc((void **)&label, sz * sizeof(Label)));
        chkerr(cudaMalloc((void **)&indeg, sz * sizeof(int)));
        chkerr(cudaMalloc((void **)&exdeg, sz * sizeof(int)));
        chkerr(cudaMalloc((void **)&lvl2adj, sz * sizeof(int)));
    }

    void copy(auto &src)
    {
        BufferBase<IndexType>::copy(src);
        label = src.label;
        indeg = src.indeg;
        exdeg = src.exdeg;
        lvl2adj = src.lvl2adj;
    }
    __device__ void copy(auto &src, ull i, ull j)
    {
        BufferBase<IndexType>::copy(src, i, j);
        label[i] = src.label[j];
        indeg[i] = src.indeg[j];
        exdeg[i] = src.exdeg[j];
        lvl2adj[i] = src.lvl2adj[j];
    }
    void copy_host_range(auto &src, ull dst, ull src_st, ull len)
    {
        BufferBase<IndexType>::copy_host_range(src, dst, src_st, len);
        if (len == 0)
            return;
        chkerr(cudaMemcpy(label + dst, src.label + src_st, sizeof(Label) * len, cudaMemcpyDeviceToHost));
        chkerr(cudaMemcpy(indeg + dst, src.indeg + src_st, sizeof(int) * len, cudaMemcpyDeviceToHost));
        chkerr(cudaMemcpy(exdeg + dst, src.exdeg + src_st, sizeof(int) * len, cudaMemcpyDeviceToHost));
        chkerr(cudaMemcpy(lvl2adj + dst, src.lvl2adj + src_st, sizeof(int) * len, cudaMemcpyDeviceToHost));
    }
    void copy_host_to_device_range(auto &src, ull dst, ull src_st, ull len)
    {
        BufferBase<IndexType>::copy_host_to_device_range(src, dst, src_st, len);
        if (len == 0)
            return;
        chkerr(cudaMemcpy(label + dst, src.label + src_st, sizeof(Label) * len, cudaMemcpyHostToDevice));
        chkerr(cudaMemcpy(indeg + dst, src.indeg + src_st, sizeof(int) * len, cudaMemcpyHostToDevice));
        chkerr(cudaMemcpy(exdeg + dst, src.exdeg + src_st, sizeof(int) * len, cudaMemcpyHostToDevice));
        chkerr(cudaMemcpy(lvl2adj + dst, src.lvl2adj + src_st, sizeof(int) * len, cudaMemcpyHostToDevice));
    }
    /**
     * @brief This version is used to allocate memory on host. Call it only for HOST_BUFF_SZ
     *
     */
    void allocateMemory()
    {
        BufferBase<IndexType>::allocateMemory();
        chkerr(cudaMallocManaged((void **)&label, sizeof(Label) * HOST_BUFF_SZ));
        chkerr(cudaMallocManaged((void **)&indeg, sizeof(int) * HOST_BUFF_SZ));
        chkerr(cudaMallocManaged((void **)&exdeg, sizeof(int) * HOST_BUFF_SZ));
        chkerr(cudaMallocManaged((void **)&lvl2adj, sizeof(int) * HOST_BUFF_SZ));
        chkerr(cudaMemAdvise(label, sizeof(Label) * HOST_BUFF_SZ, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
        chkerr(cudaMemAdvise(indeg, sizeof(int) * HOST_BUFF_SZ, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
        chkerr(cudaMemAdvise(exdeg, sizeof(int) * HOST_BUFF_SZ, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
        chkerr(cudaMemAdvise(lvl2adj, sizeof(int) * HOST_BUFF_SZ, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
    }

};

#define Bwr (this->sg->wrBuff)
#define Brd (this->sg->rdBuff)
#define H   (this->sgHost->wrBuff)

class QCApp : public AppBase<QCBuffer>
{
    GPU_Data dd;
    ull saved_cliques_count_ = 0;
    ull saved_cliques_vertex_count_ = 0;
    ull saved_max_clique_size_ = 0;
    ull saved_cliques_size_ = 0;

public:
    template <class IndexType>
    __device__ Vertex make_vertex(const QCBuffer<IndexType> &buffer, ull pos)
    {
        Vertex v;
        v.vertexid = buffer.vertices[pos];
        v.label = buffer.label[pos];
        v.indeg = buffer.indeg[pos];
        v.exdeg = buffer.exdeg[pos];
        v.lvl2adj = buffer.lvl2adj[pos];
        return v;
    }

    void allocateMemory()
    {
        dd = ::dd;
    }

    void init_chunk()
    {
        chkerr(cudaMemcpy(&saved_cliques_count_, dd.cliques_count, sizeof(ull), cudaMemcpyDeviceToHost));
        chkerr(cudaMemcpy(&saved_cliques_vertex_count_, dd.cliques_vertex_count, sizeof(ull), cudaMemcpyDeviceToHost));
        chkerr(cudaMemcpy(&saved_max_clique_size_, dd.max_clique_size, sizeof(ull), cudaMemcpyDeviceToHost));
        if (store_cliques && dd.cliques_size != nullptr)
            chkerr(cudaMemcpy(&saved_cliques_size_, dd.cliques_size, sizeof(ull), cudaMemcpyDeviceToHost));
    }

    void abort_chunk()
    {
        chkerr(cudaMemcpy(dd.cliques_count, &saved_cliques_count_, sizeof(ull), cudaMemcpyHostToDevice));
        chkerr(cudaMemcpy(dd.cliques_vertex_count, &saved_cliques_vertex_count_, sizeof(ull), cudaMemcpyHostToDevice));
        chkerr(cudaMemcpy(dd.max_clique_size, &saved_max_clique_size_, sizeof(ull), cudaMemcpyHostToDevice));
        if (store_cliques && dd.cliques_size != nullptr)
            chkerr(cudaMemcpy(dd.cliques_size, &saved_cliques_size_, sizeof(ull), cudaMemcpyHostToDevice));
    }

    void iterationFailed()
    {
        abort_chunk();
    }

    void iterationSuccess()
    {
    }

    void completion()
    {
    }

    ull get_results()
    {
        ull result = 0;
        chkerr(cudaMemcpy(&result, dd.cliques_count, sizeof(ull), cudaMemcpyDeviceToHost));
        return result;
    }

    __device__ bool d_build_initial_task(GPU_Data &dd, ull root_index, Warp_Data &wd, Local_Data &ld)
    {
        if (root_index >= (*(dd.initial_vertices_count))) {
            return false;
        }

        Vertex root;
        int candidate_count = 0;

        if (LANE_IDX == 0) {
            root = dd.initial_vertices[root_index];
        }
        root.vertexid = __shfl_sync(0xFFFFFFFF, root.vertexid, 0);
        root.label = __shfl_sync(0xFFFFFFFF, root.label, 0);
        root.indeg = __shfl_sync(0xFFFFFFFF, root.indeg, 0);
        root.exdeg = __shfl_sync(0xFFFFFFFF, root.exdeg, 0);
        root.lvl2adj = __shfl_sync(0xFFFFFFFF, root.lvl2adj, 0);

        for (uint64_t i = dd.twohop_offsets[root.vertexid] + LANE_IDX; i < dd.twohop_offsets[root.vertexid + 1]; i += WARP_SIZE) {
            int vertexid = dd.twohop_neighbors[i];
            int source_index = dd.initial_order_map[vertexid];

            if (source_index > -1 && static_cast<ull>(source_index) < root_index) {
                candidate_count++;
            }
        }
        for (int i = 1; i < WARP_SIZE; i *= 2) {
            candidate_count += __shfl_xor_sync(0xFFFFFFFF, candidate_count, i);
        }

        if (LANE_IDX == 0) {
            wd.number_of_members[WIB_IDX] = 1;
            wd.number_of_candidates[WIB_IDX] = candidate_count;
            wd.total_vertices[WIB_IDX] = candidate_count + 1;
        }
        __syncwarp();

        if (candidate_count == 0) {
            return false;
        }

        if (wd.total_vertices[WIB_IDX] <= VERTICES_SIZE) {
            ld.vertices = wd.shared_vertices + (VERTICES_SIZE * WIB_IDX);
        } else {
            if (wd.total_vertices[WIB_IDX] > WVERTICES_SIZE) {
                if (LANE_IDX == 0) {
                    printf("d_build_initial_task overflow: total_vertices=%d exceeds WVERTICES_SIZE=%d for root_index=%llu\n",
                           wd.total_vertices[WIB_IDX], WVERTICES_SIZE, root_index);
                }
                asm("trap;");
            }
            ld.vertices = dd.global_vertices + (WVERTICES_SIZE * WARP_IDX);
        }

        if (LANE_IDX == 0) {
            ld.vertices[0] = root;
            ld.vertices[0].label = 1;
            ld.vertices[0].indeg = 0;
            ld.vertices[0].exdeg = 0;
            ld.vertices[0].lvl2adj = 0;
        }
        __syncwarp();

        if (LANE_IDX == 0) {
            int write_pos = 1;
            for (uint64_t i = dd.twohop_offsets[root.vertexid]; i < dd.twohop_offsets[root.vertexid + 1]; i++) {
                int vertexid = dd.twohop_neighbors[i];
                int source_index = dd.initial_order_map[vertexid];

                if (source_index > -1 && static_cast<ull>(source_index) < root_index) {
                    ld.vertices[write_pos] = dd.initial_vertices[source_index];
                    ld.vertices[write_pos].label = 0;
                    ld.vertices[write_pos].indeg = 0;
                    ld.vertices[write_pos].exdeg = 0;
                    ld.vertices[write_pos].lvl2adj = 0;
                    write_pos++;
                }
            }
        }
        __syncwarp();

        for (int i = LANE_IDX; i < wd.total_vertices[WIB_IDX]; i += WARP_SIZE) {
            int vertexid = ld.vertices[i].vertexid;
            int indeg = 0;
            int exdeg = 0;
            uint64_t pneighbors_start = dd.onehop_offsets[vertexid];
            uint64_t pneighbors_end = dd.onehop_offsets[vertexid + 1];

            for (int j = 0; j < wd.total_vertices[WIB_IDX]; j++) {
                if (i == j) {
                    continue;
                }
                int phelper = d_bsearch_array(dd.onehop_neighbors + pneighbors_start,
                                              pneighbors_end - pneighbors_start,
                                              ld.vertices[j].vertexid);
                if (phelper > -1) {
                    if (j < wd.number_of_members[WIB_IDX]) {
                        indeg++;
                    } else {
                        exdeg++;
                    }
                }
            }

            ld.vertices[i].indeg = indeg;
            ld.vertices[i].exdeg = exdeg;
            ld.vertices[i].lvl2adj = 0;
        }
        __syncwarp();

        wd.remaining_count[WIB_IDX] = wd.number_of_candidates[WIB_IDX];
        for (int i = LANE_IDX; i < wd.number_of_candidates[WIB_IDX]; i += WARP_SIZE) {
            dd.candidate_indegs[(WVERTICES_SIZE * WARP_IDX) + i] = ld.vertices[wd.number_of_members[WIB_IDX] + i].indeg;
        }
        __syncwarp();

        int method_return = d_degree_pruning(dd, wd, ld) ? 1 : 0;
        if (method_return == 1) {
            if (wd.number_of_members[WIB_IDX] >= (*dd.minimum_clique_size)) {
                d_check_for_clique(dd, wd, ld);
            }
            return false;
        }

        method_return = d_critical_vertex_pruning(dd, wd, ld);
        if (method_return == 2) {
            return false;
        }

        if (wd.number_of_members[WIB_IDX] >= (*dd.minimum_clique_size)) {
            d_check_for_clique(dd, wd, ld);
        }

        if (method_return == 1 || wd.number_of_candidates[WIB_IDX] <= 0) {
            return false;
        }

        d_sort(ld.vertices, wd.total_vertices[WIB_IDX], d_sort_vert_Q);
        for (int i = LANE_IDX; i < wd.total_vertices[WIB_IDX]; i += WARP_SIZE) {
            ld.vertices[i].lvl2adj = 0;
        }
        __syncwarp();

        return true;
    }

    __device__ void generateSubgraphs(unsigned int base)
    {
        __shared__ Warp_Data wd;
        Local_Data ld;

        for (unsigned int i = GLWARPID; i < this->sg->chunk[0]; i += N_WARPS)
        {
            unsigned int vp = base + i;
            if (vp >= ctx->sources_num[0])
                break;

            // QC sources are interpreted as indices into the root frontier built by initialize_tasks().
            ull root_index = ctx->sources[vp];
            bool valid_task = d_build_initial_task(dd, root_index, wd, ld);
            if (!valid_task)
                continue;

            d_write_to_tasks(dd, wd, ld);
        }
    }

public:
    __device__ void processSubgraphs()
    {
    }
    void init_level()
    {
        transfer_cliques<<<NUM_OF_BLOCKS, BLOCK_SIZE>>>(dd);
        chkerr(cudaDeviceSynchronize());
    }
    __device__ void expand()
    {
        // data is stored in data structures to reduce the number of variables that need to be passed to methods
        __shared__ Warp_Data wd;
        Local_Data ld;

        // helper variables, not passed through to any methods
        int num_mem;
        int method_return;
        int index;

        /*
         * The program alternates between reading and writing between to 'tasks' arrays in device global memory. The program will read from one tasks, expand to the next level by generating and pruning, then it will write to the
         * other tasks array. It writes a bounded amount to the tasks array and the rest to the top of the buffer. The buffers acts as a stack containing the excess data not being expanded from tasks. Since the
         * buffer acts as a stack, in a last-in first-out manner, a subsection of the search space will be expanded until completion. This system allows the problem to essentially be divided into smaller problems and thus
         * require less memory to handle.
         */

        // --- CURRENT LEVEL ---

        // scheduling toggle = 0, dynamic intersection
        // if (*dd.scheduling_toggle == 0)
        while (true)
        {
            // initialize i for each warp
            if (sg->isOverflow())
                break;
            SubgraphOffsets so = Brd.next();
            if (Brd.isEnd(so.st))
                break;
            if (sg->isOverflowToHost())
            {
                dumpToHost(&so);
                continue;
            }

            // get information on vertices being handled within tasks
            if (LANE_IDX == 0)
            {
                wd.start[WIB_IDX] = so.st;
                wd.end[WIB_IDX] = so.en;
                wd.tot_vert[WIB_IDX] = so.en - so.st;
            }
            __syncwarp();

            // each warp gets partial number of members
            num_mem = 0;
            for (uint64_t j = wd.start[WIB_IDX] + LANE_IDX; j < wd.end[WIB_IDX]; j += WARP_SIZE)
            {
                if (Brd.label[j] != 1)
                {
                    break;
                }
                num_mem++;
            }
            // sum members across warp
            for (int k = 1; k < 32; k *= 2)
            {
                num_mem += __shfl_xor_sync(0xFFFFFFFF, num_mem, k);
            }

            if (LANE_IDX == 0)
            {
                wd.num_mem[WIB_IDX] = num_mem;
            }
            __syncwarp();

            int number_of_covered = 0;
            for (uint64_t j = wd.start[WIB_IDX] + wd.num_mem[WIB_IDX] + LANE_IDX; j < wd.end[WIB_IDX]; j += WARP_SIZE)
            {
                if (Brd.label[j] != 2)
                {
                    break;
                }
                number_of_covered++;
            }
            for (int k = 1; k < 32; k *= 2)
            {
                number_of_covered += __shfl_xor_sync(0xFFFFFFFF, number_of_covered, k);
            }

            if (LANE_IDX == 0)
            {
                wd.number_of_covered[WIB_IDX] = number_of_covered;
                wd.num_cand[WIB_IDX] = wd.tot_vert[WIB_IDX] - wd.num_mem[WIB_IDX];
                wd.expansions[WIB_IDX] = wd.num_cand[WIB_IDX];
            }
            __syncwarp();

            // LOOKAHEAD PRUNING
            method_return = d_lookahead_pruning(dd, wd, ld);
            if (method_return)
            {
                continue;
            }

            // --- NEXT LEVEL ---
            for (int j = wd.number_of_covered[WIB_IDX]; j < wd.expansions[WIB_IDX]; j++)
            {

                // REMOVE ONE VERTEX
                if (j != wd.number_of_covered[WIB_IDX])
                {
                    method_return = d_remove_one_vertex(dd, wd, ld);
                    if (method_return)
                    {
                        break;
                    }
                }

                // INITIALIZE NEW VERTICES
                if (LANE_IDX == 0)
                {
                    wd.number_of_members[WIB_IDX] = wd.num_mem[WIB_IDX];
                    wd.number_of_candidates[WIB_IDX] = wd.num_cand[WIB_IDX];
                    wd.total_vertices[WIB_IDX] = wd.tot_vert[WIB_IDX];
                }
                __syncwarp();

                // select whether to store vertices in global or shared memory based on size
                if (wd.total_vertices[WIB_IDX] <= VERTICES_SIZE)
                {
                    ld.vertices = wd.shared_vertices + (VERTICES_SIZE * WIB_IDX);
                }
                else
                {
                    ld.vertices = dd.global_vertices + (WVERTICES_SIZE * WARP_IDX);
                }

                for (index = LANE_IDX; index < wd.number_of_members[WIB_IDX]; index += WARP_SIZE)
                {
                    ld.vertices[index] = make_vertex(Brd, wd.start[WIB_IDX] + index);
                    // ld.vertices[index] = dd.read_vertices[wd.start[WIB_IDX] + index];
                }
                for (; index < wd.total_vertices[WIB_IDX] - 1; index += WARP_SIZE)
                {
                    ld.vertices[index + 1] = make_vertex(Brd, wd.start[WIB_IDX] + index);
                    // ld.vertices[index + 1] = dd.read_vertices[wd.start[WIB_IDX] + index];
                }

                if (LANE_IDX == 0)
                {
                    ld.vertices[wd.number_of_members[WIB_IDX]] = make_vertex(Brd, wd.start[WIB_IDX] + wd.total_vertices[WIB_IDX] - 1);
                    // ld.vertices[wd.number_of_members[WIB_IDX]] = dd.read_vertices[wd.start[WIB_IDX] + wd.total_vertices[WIB_IDX] - 1];
                }
                __syncwarp();

                for (int index = wd.num_mem[WIB_IDX] + 1 + LANE_IDX;
                     index <= wd.num_mem[WIB_IDX] + wd.number_of_covered[WIB_IDX];
                     index += WARP_SIZE)
                {
                    ld.vertices[index].label = 0;
                }
                __syncwarp();

                // ADD ONE VERTEX
                method_return = d_add_one_vertex(dd, wd, ld);

                // if failed found check for clique and continue on to the next iteration
                if (method_return == 1)
                {
                    if (wd.number_of_members[WIB_IDX] >= (*dd.minimum_clique_size))
                    {
                        d_check_for_clique(dd, wd, ld);
                    }
                    continue;
                }

                // CRITICAL VERTEX PRUNING
                method_return = d_critical_vertex_pruning(dd, wd, ld);

                // critical fail, cannot be clique continue onto next iteration
                if (method_return == 2)
                {
                    continue;
                }

                // HANDLE CLIQUES
                if (wd.number_of_members[WIB_IDX] >= (*dd.minimum_clique_size))
                {
                    d_check_for_clique(dd, wd, ld);
                }

                // if vertex in x found as not extendable continue to next iteration
                if (method_return == 1)
                {
                    continue;
                }

                // WRITE TASKS TO BUFFERS
                // sort vertices in Quick efficient enumeration order before writing
                d_sort(ld.vertices, wd.total_vertices[WIB_IDX], d_sort_vert_Q);

                if (wd.number_of_candidates[WIB_IDX] > 0)
                {
                    d_write_to_tasks(dd, wd, ld);
                }
            }
        }

        if (LANE_IDX == 0)
        {
            // sum to find tasks count
            atomicAdd(dd.total_tasks, dd.wtasks_count[WARP_IDX]);
        }
    }

    // --- DEVICE EXPANSION KERNELS ---

    __device__ bool d_reserve_global_clique(GPU_Data &dd, int clique_size, uint64_t &start_write)
    {
        unsigned long long clique_id = 0;
        unsigned long long vertex_start = 0;

        if (LANE_IDX == 0)
        {
            clique_id = atomicAdd(reinterpret_cast<unsigned long long *>(dd.cliques_count), 1ULL);
            if (!(*(dd.store_cliques)))
            {
                vertex_start = 0;
            }
            else
            {
                vertex_start = atomicAdd(reinterpret_cast<unsigned long long *>(dd.cliques_vertex_count),
                                         static_cast<unsigned long long>(clique_size));

                if (clique_id < CLIQUES_OFFSET_SIZE && vertex_start + clique_size <= CLIQUES_SIZE)
                {
                    dd.cliques_offset[clique_id] = vertex_start;
                    dd.cliques_size[clique_id] = clique_size;
                }
            }
        }

        clique_id = __shfl_sync(0xFFFFFFFF, clique_id, 0);
        vertex_start = __shfl_sync(0xFFFFFFFF, vertex_start, 0);
        start_write = static_cast<uint64_t>(vertex_start);

        return (*(dd.store_cliques)) && clique_id < CLIQUES_OFFSET_SIZE && vertex_start + clique_size <= CLIQUES_SIZE;
    }

    // returns 1 if lookahead succesful, 0 otherwise
    __device__ int d_lookahead_pruning(GPU_Data &dd, Warp_Data &wd, Local_Data &ld)
    {
        int pvertexid;
        int phelper1;
        int phelper2;

        if (LANE_IDX == 0)
        {
            wd.success[WIB_IDX] = true;
        }
        __syncwarp();

        // check if members meet degree requirement, dont need to check 2hop adj as diameter pruning guarentees all members will be within 2hops of eveything
        for (int i = LANE_IDX; i < wd.num_mem[WIB_IDX] && wd.success[WIB_IDX]; i += WARP_SIZE)
        {
            if (Brd.indeg[wd.start[WIB_IDX] + i] + Brd.exdeg[wd.start[WIB_IDX] + i] < dd.minimum_degrees[wd.tot_vert[WIB_IDX]])
            {
                wd.success[WIB_IDX] = false;
                break;
            }
        }
        __syncwarp();

        if (!wd.success[WIB_IDX])
        {
            return 0;
        }

        // update lvl2adj to candidates for all vertices
        for (int i = wd.num_mem[WIB_IDX] + LANE_IDX; i < wd.tot_vert[WIB_IDX]; i += WARP_SIZE)
        {
            pvertexid = Brd.vertices[wd.start[WIB_IDX] + i];

            for (int j = wd.num_mem[WIB_IDX]; j < wd.tot_vert[WIB_IDX]; j++)
            {
                if (j == i)
                {
                    continue;
                }

                phelper1 = Brd.vertices[wd.start[WIB_IDX] + j];
                phelper2 = d_bsearch_array(dd.twohop_neighbors + dd.twohop_offsets[phelper1], dd.twohop_offsets[phelper1 + 1] - dd.twohop_offsets[phelper1], pvertexid);

                if (phelper2 > -1)
                {
                    Brd.lvl2adj[wd.start[WIB_IDX] + i]++;
                }
            }
        }
        __syncwarp();

        // compares all vertices to the lemmas from Quick
        for (int j = wd.num_mem[WIB_IDX] + LANE_IDX; j < wd.tot_vert[WIB_IDX] && wd.success[WIB_IDX]; j += WARP_SIZE)
        {
            if (Brd.lvl2adj[wd.start[WIB_IDX] + j] < wd.num_cand[WIB_IDX] - 1 || Brd.indeg[wd.start[WIB_IDX] + j] + Brd.exdeg[wd.start[WIB_IDX] + j] < dd.minimum_degrees[wd.tot_vert[WIB_IDX]])
            {
                wd.success[WIB_IDX] = false;
                break;
            }
        }
        __syncwarp();

        if (wd.success[WIB_IDX])
        {
            // write to global clique buffer
            uint64_t start_write = 0;
            if (d_reserve_global_clique(dd, wd.tot_vert[WIB_IDX], start_write))
            {
                for (int j = LANE_IDX; j < wd.tot_vert[WIB_IDX]; j += WARP_SIZE)
                {
                    dd.cliques_vertex[start_write + j] = Brd.vertices[wd.start[WIB_IDX] + j];
                }
            }
            return 1;
        }

        return 0;
    }

    // returns 1 if failed found after removing, 0 otherwise
    __device__ int d_remove_one_vertex(GPU_Data &dd, Warp_Data &wd, Local_Data &ld)
    {
        int pvertexid;
        int phelper1;
        int phelper2;

        int mindeg;

        mindeg = d_get_mindeg(wd.num_mem[WIB_IDX], dd);

        // remove the last candidate in vertices
        if (LANE_IDX == 0)
        {
            wd.num_cand[WIB_IDX]--;
            wd.tot_vert[WIB_IDX]--;
            wd.success[WIB_IDX] = false;
        }
        __syncwarp();

        // update info of vertices connected to removed cand
        pvertexid = Brd.vertices[wd.start[WIB_IDX] + wd.tot_vert[WIB_IDX]];

        for (int i = LANE_IDX; i < wd.tot_vert[WIB_IDX] && !wd.success[WIB_IDX]; i += WARP_SIZE)
        {
            phelper1 = Brd.vertices[wd.start[WIB_IDX] + i];
            phelper2 = d_bsearch_array(dd.onehop_neighbors + dd.onehop_offsets[pvertexid], dd.onehop_offsets[pvertexid + 1] - dd.onehop_offsets[pvertexid], phelper1);

            if (phelper2 > -1)
            {
                Brd.exdeg[wd.start[WIB_IDX] + i]--;

                if (i < wd.num_mem[WIB_IDX] && Brd.indeg[wd.start[WIB_IDX] + i] + Brd.exdeg[wd.start[WIB_IDX] + i] < mindeg)
                {
                    wd.success[WIB_IDX] = true;
                    break;
                }
            }
        }
        __syncwarp();

        if (wd.success[WIB_IDX])
        {
            return 1;
        }

        return 0;
    }

    // returns 1 if failed found or invalid bound, 0 otherwise
    __device__ int d_add_one_vertex(GPU_Data &dd, Warp_Data &wd, Local_Data &ld)
    {
        int pvertexid;
        int phelper1;
        int phelper2;
        bool failed_found;

        // ADD ONE VERTEX
        pvertexid = ld.vertices[wd.number_of_members[WIB_IDX]].vertexid;

        if (LANE_IDX == 0)
        {
            ld.vertices[wd.number_of_members[WIB_IDX]].label = 1;
            wd.number_of_members[WIB_IDX]++;
            wd.number_of_candidates[WIB_IDX]--;
        }
        __syncwarp();

        for (int i = LANE_IDX; i < wd.tot_vert[WIB_IDX]; i += WARP_SIZE)
        {
            phelper1 = ld.vertices[i].vertexid;
            phelper2 = d_bsearch_array(dd.onehop_neighbors + dd.onehop_offsets[pvertexid], dd.onehop_offsets[pvertexid + 1] - dd.onehop_offsets[pvertexid], phelper1);

            if (phelper2 > -1)
            {
                ld.vertices[i].exdeg--;
                ld.vertices[i].indeg++;
            }
        }
        __syncwarp();

        // DIAMETER PRUNING
        d_diameter_pruning(dd, wd, ld, pvertexid);

        // DEGREE BASED PRUNING
        failed_found = d_degree_pruning(dd, wd, ld);

        // if vertex in x found as not extendable continue to next iteration
        if (failed_found)
        {
            return 1;
        }

        return 0;
    }

    // returns 2, if critical fail, 1 if failed found or invalid bound, 0 otherwise
    __device__ int d_critical_vertex_pruning(GPU_Data &dd, Warp_Data &wd, Local_Data &ld)
    {
        // intersection
        int phelper1;

        // pruning
        int number_of_crit_adj;
        bool failed_found;

        // CRITICAL VERTEX PRUNING
        // iterate through all vertices in clique
        for (int k = 0; k < wd.number_of_members[WIB_IDX]; k++)
        {

            // if they are a critical vertex
            if (ld.vertices[k].indeg + ld.vertices[k].exdeg == dd.minimum_degrees[wd.number_of_members[WIB_IDX] + wd.lower_bound[WIB_IDX]] && ld.vertices[k].exdeg > 0)
            {
                phelper1 = ld.vertices[k].vertexid;

                // iterate through all candidates
                for (int i = wd.number_of_members[WIB_IDX] + LANE_IDX; i < wd.total_vertices[WIB_IDX]; i += WARP_SIZE)
                {
                    if (ld.vertices[i].label != 4)
                    {
                        // if candidate is neighbor of critical vertex mark as such
                        if (d_bsearch_array(dd.onehop_neighbors + dd.onehop_offsets[phelper1], dd.onehop_offsets[phelper1 + 1] - dd.onehop_offsets[phelper1], ld.vertices[i].vertexid) > -1)
                        {
                            ld.vertices[i].label = 4;
                        }
                    }
                }
            }
            __syncwarp();
        }

        // sort vertices so that critical vertex adjacent candidates are immediately after vertices within the clique
        d_sort(ld.vertices + wd.number_of_members[WIB_IDX], wd.number_of_candidates[WIB_IDX], d_sort_vert_cv);

        // count number of critical adjacent vertices
        number_of_crit_adj = 0;
        for (int i = wd.number_of_members[WIB_IDX] + LANE_IDX; i < wd.total_vertices[WIB_IDX]; i += WARP_SIZE)
        {
            if (ld.vertices[i].label == 4)
            {
                number_of_crit_adj++;
            }
            else
            {
                break;
            }
        }
        // get sum
        for (int i = 1; i < 32; i *= 2)
        {
            number_of_crit_adj += __shfl_xor_sync(0xFFFFFFFF, number_of_crit_adj, i);
        }

        failed_found = false;

        // reset adjacencies
        for (int i = LANE_IDX; i < wd.total_vertices[WIB_IDX]; i += WARP_SIZE)
        {
            dd.adjacencies[(WVERTICES_SIZE * WARP_IDX) + i] = 0;
        }

        // if there were any neighbors of critical vertices
        if (number_of_crit_adj > 0)
        {
            // iterate through all vertices and update their degrees as if critical adjacencies were added and keep track of how many critical adjacencies they are adjacent to
            for (int k = LANE_IDX; k < wd.total_vertices[WIB_IDX]; k += WARP_SIZE)
            {
                phelper1 = ld.vertices[k].vertexid;

                for (int i = wd.number_of_members[WIB_IDX]; i < wd.number_of_members[WIB_IDX] + number_of_crit_adj; i++)
                {
                    if (d_bsearch_array(dd.onehop_neighbors + dd.onehop_offsets[phelper1], dd.onehop_offsets[phelper1 + 1] - dd.onehop_offsets[phelper1], ld.vertices[i].vertexid) > -1)
                    {
                        ld.vertices[k].indeg++;
                        ld.vertices[k].exdeg--;
                    }

                    if (d_bsearch_array(dd.twohop_neighbors + dd.twohop_offsets[phelper1], dd.twohop_offsets[phelper1 + 1] - dd.twohop_offsets[phelper1], ld.vertices[i].vertexid) > -1)
                    {
                        dd.adjacencies[(WVERTICES_SIZE * WARP_IDX) + k]++;
                    }
                }
            }
            __syncwarp();

            // all vertices within the clique must be within 2hops of the newly added critical vertex adj vertices
            for (int k = LANE_IDX; k < wd.number_of_members[WIB_IDX]; k += WARP_SIZE)
            {
                if (dd.adjacencies[(WVERTICES_SIZE * WARP_IDX) + k] != number_of_crit_adj)
                {
                    failed_found = true;
                    break;
                }
            }
            failed_found = __any_sync(0xFFFFFFFF, failed_found);
            if (failed_found)
            {
                return 2;
            }

            // all critical adj vertices must all be within 2 hops of each other
            for (int k = wd.number_of_members[WIB_IDX] + LANE_IDX; k < wd.number_of_members[WIB_IDX] + number_of_crit_adj; k += WARP_SIZE)
            {
                if (dd.adjacencies[(WVERTICES_SIZE * WARP_IDX) + k] < number_of_crit_adj - 1)
                {
                    failed_found = true;
                    break;
                }
            }
            failed_found = __any_sync(0xFFFFFFFF, failed_found);
            if (failed_found)
            {
                return 2;
            }

            // no failed vertices found so add all critical vertex adj candidates to clique
            for (int k = wd.number_of_members[WIB_IDX] + LANE_IDX; k < wd.number_of_members[WIB_IDX] + number_of_crit_adj; k += WARP_SIZE)
            {
                ld.vertices[k].label = 1;
            }

            if (LANE_IDX == 0)
            {
                wd.number_of_members[WIB_IDX] += number_of_crit_adj;
                wd.number_of_candidates[WIB_IDX] -= number_of_crit_adj;
            }
            __syncwarp();
        }

        // DIAMTER PRUNING
        d_diameter_pruning_cv(dd, wd, ld, number_of_crit_adj);

        // DEGREE BASED PRUNING
        failed_found = d_degree_pruning(dd, wd, ld);

        // if vertex in x found as not extendable continue to next iteration
        if (failed_found)
        {
            return 1;
        }

        return 0;
    }

    // diameter pruning intitializes vertices labels and candidate indegs array for use in iterative degree pruning
    __device__ void d_diameter_pruning(GPU_Data &dd, Warp_Data &wd, Local_Data &ld, int pvertexid)
    {
        // vertices size * warp idx + (vertices size / warp size) * lane idx
        int lane_write = ((WVERTICES_SIZE * WARP_IDX) + ((WVERTICES_SIZE / WARP_SIZE) * LANE_IDX));

        // intersection
        int phelper1;
        int phelper2;

        // vertex iteration
        int lane_remaining_count;

        lane_remaining_count = 0;

        for (int i = wd.number_of_members[WIB_IDX] + LANE_IDX; i < wd.total_vertices[WIB_IDX]; i += WARP_SIZE)
        {
            ld.vertices[i].label = -1;
        }
        __syncwarp();

        for (int i = wd.number_of_members[WIB_IDX] + LANE_IDX; i < wd.total_vertices[WIB_IDX]; i += WARP_SIZE)
        {
            phelper1 = ld.vertices[i].vertexid;
            phelper2 = d_bsearch_array(dd.twohop_neighbors + dd.twohop_offsets[pvertexid], dd.twohop_offsets[pvertexid + 1] - dd.twohop_offsets[pvertexid], phelper1);

            if (phelper2 > -1)
            {
                ld.vertices[i].label = 0;
                dd.lane_candidate_indegs[lane_write + lane_remaining_count++] = ld.vertices[i].indeg;
            }
        }
        __syncwarp();

        // scan to calculate write postion in warp arrays
        phelper2 = lane_remaining_count;
        for (int i = 1; i < WARP_SIZE; i *= 2)
        {
            phelper1 = __shfl_up_sync(0xFFFFFFFF, lane_remaining_count, i, WARP_SIZE);
            if (LANE_IDX >= i)
            {
                lane_remaining_count += phelper1;
            }
            __syncwarp();
        }
        // lane remaining count sum is scan for last lane and its value
        if (LANE_IDX == WARP_SIZE - 1)
        {
            wd.remaining_count[WIB_IDX] = lane_remaining_count;
        }
        // make scan exclusive
        lane_remaining_count -= phelper2;
        __syncwarp();

        // parallel write lane arrays to warp array
        for (int i = 0; i < phelper2; i++)
        {
            dd.candidate_indegs[(WVERTICES_SIZE * WARP_IDX) + lane_remaining_count + i] = dd.lane_candidate_indegs[lane_write + i];
        }
        __syncwarp();
    }

    __device__ void d_diameter_pruning_cv(GPU_Data &dd, Warp_Data &wd, Local_Data &ld, int number_of_crit_adj)
    {
        // (WVERTICES_SIZE * WARP_IDX) /warp write location to adjacencies
        // vertices size * warp idx + (vertices size / warp size) * lane idx
        int lane_write = ((WVERTICES_SIZE * WARP_IDX) + ((WVERTICES_SIZE / WARP_SIZE) * LANE_IDX));

        // vertex iteration
        int lane_remaining_count;

        // intersection
        int phelper1;
        int phelper2;

        lane_remaining_count = 0;

        // remove all cands who are not within 2hops of all newly added cands
        for (int k = wd.number_of_members[WIB_IDX] + LANE_IDX; k < wd.total_vertices[WIB_IDX]; k += WARP_SIZE)
        {
            if (dd.adjacencies[(WVERTICES_SIZE * WARP_IDX) + k] == number_of_crit_adj)
            {
                dd.lane_candidate_indegs[lane_write + lane_remaining_count++] = ld.vertices[k].indeg;
            }
            else
            {
                ld.vertices[k].label = -1;
            }
        }

        // scan to calculate write postion in warp arrays
        phelper2 = lane_remaining_count;
        for (int i = 1; i < WARP_SIZE; i *= 2)
        {
            phelper1 = __shfl_up_sync(0xFFFFFFFF, lane_remaining_count, i, WARP_SIZE);
            if (LANE_IDX >= i)
            {
                lane_remaining_count += phelper1;
            }
            __syncwarp();
        }
        // lane remaining count sum is scan for last lane and its value
        if (LANE_IDX == WARP_SIZE - 1)
        {
            wd.remaining_count[WIB_IDX] = lane_remaining_count;
        }
        // make scan exclusive
        lane_remaining_count -= phelper2;
        __syncwarp();

        // parallel write lane arrays to warp array
        for (int i = 0; i < phelper2; i++)
        {
            dd.candidate_indegs[(WVERTICES_SIZE * WARP_IDX) + lane_remaining_count + i] = dd.lane_candidate_indegs[lane_write + i];
        }
        __syncwarp();
    }

    // returns true if invalid bounds or failed found
    __device__ bool d_degree_pruning(GPU_Data &dd, Warp_Data &wd, Local_Data &ld)
    {
        // vertices size * warp idx + (vertices size / warp size) * lane idx
        int lane_write = ((WVERTICES_SIZE * WARP_IDX) + ((WVERTICES_SIZE / WARP_SIZE) * LANE_IDX));

        // helper variables used throughout method to store various values, names have no meaning
        int pvertexid;
        int phelper1;
        int phelper2;
        Vertex *read;
        Vertex *write;

        // counter for lane intersection results
        int lane_remaining_count;
        int lane_removed_count;

        d_sort_i(dd.candidate_indegs + (WVERTICES_SIZE * WARP_IDX), wd.remaining_count[WIB_IDX], d_sort_degs);

        d_calculate_LU_bounds(dd, wd, ld, wd.remaining_count[WIB_IDX]);
        if (wd.invalid_bounds[WIB_IDX])
        {
            return true;
        }

        // check for failed vertices
        if (LANE_IDX == 0)
        {
            wd.success[WIB_IDX] = false;
        }
        __syncwarp();
        for (int k = LANE_IDX; k < wd.number_of_members[WIB_IDX] && !wd.success[WIB_IDX]; k += WARP_SIZE)
        {
            if (!d_vert_isextendable_LU(ld.vertices[k], dd, wd, ld))
            {
                wd.success[WIB_IDX] = true;
                break;
            }
        }
        __syncwarp();
        if (wd.success[WIB_IDX])
        {
            return true;
        }

        if (LANE_IDX == 0)
        {
            wd.remaining_count[WIB_IDX] = 0;
            wd.removed_count[WIB_IDX] = 0;
            wd.rw_counter[WIB_IDX] = 0;
        }

        lane_remaining_count = 0;
        lane_removed_count = 0;

        for (int i = wd.number_of_members[WIB_IDX] + LANE_IDX; i < wd.total_vertices[WIB_IDX]; i += WARP_SIZE)
        {
            if (ld.vertices[i].label == 0 && d_cand_isvalid_LU(ld.vertices[i], dd, wd, ld))
            {
                dd.lane_remaining_candidates[lane_write + lane_remaining_count++] = i;
            }
            else
            {
                dd.lane_removed_candidates[lane_write + lane_removed_count++] = i;
            }
        }
        __syncwarp();

        // scan to calculate write postion in warp arrays
        phelper2 = lane_remaining_count;
        pvertexid = lane_removed_count;
        for (int i = 1; i < WARP_SIZE; i *= 2)
        {
            phelper1 = __shfl_up_sync(0xFFFFFFFF, lane_remaining_count, i, WARP_SIZE);
            if (LANE_IDX >= i)
            {
                lane_remaining_count += phelper1;
            }
            phelper1 = __shfl_up_sync(0xFFFFFFFF, lane_removed_count, i, WARP_SIZE);
            if (LANE_IDX >= i)
            {
                lane_removed_count += phelper1;
            }
            __syncwarp();
        }
        // lane remaining count sum is scan for last lane and its value
        if (LANE_IDX == WARP_SIZE - 1)
        {
            wd.remaining_count[WIB_IDX] = lane_remaining_count;
            wd.removed_count[WIB_IDX] = lane_removed_count;
        }
        // make scan exclusive
        lane_remaining_count -= phelper2;
        lane_removed_count -= pvertexid;

        // parallel write lane arrays to warp array
        for (int i = 0; i < phelper2; i++)
        {
            dd.remaining_candidates[(WVERTICES_SIZE * WARP_IDX) + lane_remaining_count + i] = ld.vertices[dd.lane_remaining_candidates[lane_write + i]];
        }
        // only need removed if going to be using removed to update degrees
        if (!(wd.remaining_count[WIB_IDX] < wd.removed_count[WIB_IDX]))
        {
            for (int i = 0; i < pvertexid; i++)
            {
                dd.removed_candidates[(WVERTICES_SIZE * WARP_IDX) + lane_removed_count + i] = ld.vertices[dd.lane_removed_candidates[lane_write + i]].vertexid;
            }
        }
        __syncwarp();

        while (wd.remaining_count[WIB_IDX] > 0 && wd.removed_count[WIB_IDX] > 0)
        {
            // different blocks for the read and write locations, vertices and remaining, this is done to avoid using extra variables and only one condition
            if (wd.rw_counter[WIB_IDX] % 2 == 0)
            {
                read = dd.remaining_candidates + (WVERTICES_SIZE * WARP_IDX);
                write = ld.vertices + wd.number_of_members[WIB_IDX];
            }
            else
            {
                read = ld.vertices + wd.number_of_members[WIB_IDX];
                write = dd.remaining_candidates + (WVERTICES_SIZE * WARP_IDX);
            }

            // update degrees
            if (wd.remaining_count[WIB_IDX] < wd.removed_count[WIB_IDX])
            {
                // via remaining, reset exdegs
                for (int i = LANE_IDX; i < wd.number_of_members[WIB_IDX]; i += WARP_SIZE)
                {
                    ld.vertices[i].exdeg = 0;
                }
                for (int i = LANE_IDX; i < wd.remaining_count[WIB_IDX]; i += WARP_SIZE)
                {
                    read[i].exdeg = 0;
                }
                __syncwarp();

                // update exdeg based on remaining candidates, every lane should get the next vertex to intersect dynamically
                for (int i = LANE_IDX; i < wd.number_of_members[WIB_IDX]; i += WARP_SIZE)
                {
                    pvertexid = ld.vertices[i].vertexid;

                    for (int j = 0; j < wd.remaining_count[WIB_IDX]; j++)
                    {
                        phelper1 = read[j].vertexid;
                        phelper2 = d_bsearch_array(dd.onehop_neighbors + dd.onehop_offsets[phelper1], dd.onehop_offsets[phelper1 + 1] - dd.onehop_offsets[phelper1], pvertexid);

                        if (phelper2 > -1)
                        {
                            ld.vertices[i].exdeg++;
                        }
                    }
                }

                for (int i = LANE_IDX; i < wd.remaining_count[WIB_IDX]; i += WARP_SIZE)
                {
                    pvertexid = read[i].vertexid;

                    for (int j = 0; j < wd.remaining_count[WIB_IDX]; j++)
                    {
                        if (j == i)
                        {
                            continue;
                        }

                        phelper1 = read[j].vertexid;
                        phelper2 = d_bsearch_array(dd.onehop_neighbors + dd.onehop_offsets[phelper1], dd.onehop_offsets[phelper1 + 1] - dd.onehop_offsets[phelper1], pvertexid);

                        if (phelper2 > -1)
                        {
                            read[i].exdeg++;
                        }
                    }
                }
            }
            else
            {
                // via removed, update exdeg based on remaining candidates, again lane scheduling should be dynamic
                for (int i = LANE_IDX; i < wd.number_of_members[WIB_IDX]; i += WARP_SIZE)
                {
                    pvertexid = ld.vertices[i].vertexid;

                    for (int j = 0; j < wd.removed_count[WIB_IDX]; j++)
                    {
                        phelper1 = dd.removed_candidates[(WVERTICES_SIZE * WARP_IDX) + j];
                        phelper2 = d_bsearch_array(dd.onehop_neighbors + dd.onehop_offsets[phelper1], dd.onehop_offsets[phelper1 + 1] - dd.onehop_offsets[phelper1], pvertexid);

                        if (phelper2 > -1)
                        {
                            ld.vertices[i].exdeg--;
                        }
                    }
                }

                for (int i = LANE_IDX; i < wd.remaining_count[WIB_IDX]; i += WARP_SIZE)
                {
                    pvertexid = read[i].vertexid;

                    for (int j = 0; j < wd.removed_count[WIB_IDX]; j++)
                    {
                        phelper1 = dd.removed_candidates[(WVERTICES_SIZE * WARP_IDX) + j];
                        phelper2 = d_bsearch_array(dd.onehop_neighbors + dd.onehop_offsets[phelper1], dd.onehop_offsets[phelper1 + 1] - dd.onehop_offsets[phelper1], pvertexid);

                        if (phelper2 > -1)
                        {
                            read[i].exdeg--;
                        }
                    }
                }
            }
            __syncwarp();

            lane_remaining_count = 0;

            for (int i = LANE_IDX; i < wd.remaining_count[WIB_IDX]; i += WARP_SIZE)
            {
                if (d_cand_isvalid_LU(read[i], dd, wd, ld))
                {
                    dd.lane_candidate_indegs[lane_write + lane_remaining_count++] = read[i].indeg;
                }
            }
            __syncwarp();

            // scan to calculate write postion in warp arrays
            phelper2 = lane_remaining_count;
            for (int i = 1; i < WARP_SIZE; i *= 2)
            {
                phelper1 = __shfl_up_sync(0xFFFFFFFF, lane_remaining_count, i, WARP_SIZE);
                if (LANE_IDX >= i)
                {
                    lane_remaining_count += phelper1;
                }
                __syncwarp();
            }
            // lane remaining count sum is scan for last lane and its value
            if (LANE_IDX == WARP_SIZE - 1)
            {
                wd.num_val_cands[WIB_IDX] = lane_remaining_count;
            }
            // make scan exclusive
            lane_remaining_count -= phelper2;

            // parallel write lane arrays to warp array
            for (int i = 0; i < phelper2; i++)
            {
                dd.candidate_indegs[(WVERTICES_SIZE * WARP_IDX) + lane_remaining_count + i] = dd.lane_candidate_indegs[lane_write + i];
            }
            __syncwarp();

            d_sort_i(dd.candidate_indegs + (WVERTICES_SIZE * WARP_IDX), wd.num_val_cands[WIB_IDX], d_sort_degs);

            d_calculate_LU_bounds(dd, wd, ld, wd.num_val_cands[WIB_IDX]);
            if (wd.invalid_bounds[WIB_IDX])
            {
                return true;
            }

            // check for failed vertices
            for (int k = LANE_IDX; k < wd.number_of_members[WIB_IDX] && !wd.success[WIB_IDX]; k += WARP_SIZE)
            {
                if (!d_vert_isextendable_LU(ld.vertices[k], dd, wd, ld))
                {
                    wd.success[WIB_IDX] = true;
                    break;
                }
            }
            __syncwarp();
            if (wd.success[WIB_IDX])
            {
                return true;
            }

            lane_remaining_count = 0;
            lane_removed_count = 0;

            // check for failed candidates
            for (int i = LANE_IDX; i < wd.remaining_count[WIB_IDX]; i += WARP_SIZE)
            {
                if (d_cand_isvalid_LU(read[i], dd, wd, ld))
                {
                    dd.lane_remaining_candidates[lane_write + lane_remaining_count++] = i;
                }
                else
                {
                    dd.lane_removed_candidates[lane_write + lane_removed_count++] = i;
                }
            }
            __syncwarp();

            // scan to calculate write postion in warp arrays
            phelper2 = lane_remaining_count;
            pvertexid = lane_removed_count;
            for (int i = 1; i < WARP_SIZE; i *= 2)
            {
                phelper1 = __shfl_up_sync(0xFFFFFFFF, lane_remaining_count, i, WARP_SIZE);
                if (LANE_IDX >= i)
                {
                    lane_remaining_count += phelper1;
                }
                phelper1 = __shfl_up_sync(0xFFFFFFFF, lane_removed_count, i, WARP_SIZE);
                if (LANE_IDX >= i)
                {
                    lane_removed_count += phelper1;
                }
                __syncwarp();
            }
            // lane remaining count sum is scan for last lane and its value
            if (LANE_IDX == WARP_SIZE - 1)
            {
                wd.num_val_cands[WIB_IDX] = lane_remaining_count;
                wd.removed_count[WIB_IDX] = lane_removed_count;
            }
            // make scan exclusive
            lane_remaining_count -= phelper2;
            lane_removed_count -= pvertexid;

            // parallel write lane arrays to warp array
            for (int i = 0; i < phelper2; i++)
            {
                write[lane_remaining_count + i] = read[dd.lane_remaining_candidates[lane_write + i]];
            }
            // only need removed if going to be using removed to update degrees
            if (!(wd.num_val_cands[WIB_IDX] < wd.removed_count[WIB_IDX]))
            {
                for (int i = 0; i < pvertexid; i++)
                {
                    dd.removed_candidates[(WVERTICES_SIZE * WARP_IDX) + lane_removed_count + i] = read[dd.lane_removed_candidates[lane_write + i]].vertexid;
                }
            }

            if (LANE_IDX == 0)
            {
                wd.remaining_count[WIB_IDX] = wd.num_val_cands[WIB_IDX];
                wd.rw_counter[WIB_IDX]++;
            }
        }

        // condense vertices so remaining are after members, only needs to be done if they were not written into vertices last time
        if (wd.rw_counter[WIB_IDX] % 2 == 0)
        {
            for (int i = LANE_IDX; i < wd.remaining_count[WIB_IDX]; i += WARP_SIZE)
            {
                ld.vertices[wd.number_of_members[WIB_IDX] + i] = dd.remaining_candidates[(WVERTICES_SIZE * WARP_IDX) + i];
            }
        }

        if (LANE_IDX == 0)
        {
            wd.total_vertices[WIB_IDX] = wd.total_vertices[WIB_IDX] - wd.number_of_candidates[WIB_IDX] + wd.remaining_count[WIB_IDX];
            wd.number_of_candidates[WIB_IDX] = wd.remaining_count[WIB_IDX];
        }

        return false;
    }

    __device__ void d_calculate_LU_bounds(GPU_Data &dd, Warp_Data &wd, Local_Data &ld, int number_of_candidates)
    {
        int index;

        int min_clq_indeg;
        int min_indeg_exdeg;
        int min_clq_totaldeg;
        int sum_clq_indeg;

        // initialize the values of the LU calculation variables to the first vertices values so they can be compared to other vertices without error
        min_clq_indeg = ld.vertices[0].indeg;
        min_indeg_exdeg = ld.vertices[0].exdeg;
        min_clq_totaldeg = ld.vertices[0].indeg + ld.vertices[0].exdeg;
        sum_clq_indeg = 0;

        // each warp also has a copy of these variables to allow for intra-warp comparison of these variables.
        if (LANE_IDX == 0)
        {
            wd.invalid_bounds[WIB_IDX] = false;

            wd.sum_candidate_indeg[WIB_IDX] = 0;
            wd.tightened_upper_bound[WIB_IDX] = 0;

            wd.min_clq_indeg[WIB_IDX] = ld.vertices[0].indeg;
            wd.min_indeg_exdeg[WIB_IDX] = ld.vertices[0].exdeg;
            wd.min_clq_totaldeg[WIB_IDX] = ld.vertices[0].indeg + ld.vertices[0].exdeg;
            wd.sum_clq_indeg[WIB_IDX] = ld.vertices[0].indeg;

            wd.min_ext_deg[WIB_IDX] = d_get_mindeg(wd.number_of_members[WIB_IDX] + 1, dd);
        }
        __syncwarp();

        // each warp finds these values on their subsection of vertices
        for (index = 1 + LANE_IDX; index < wd.number_of_members[WIB_IDX]; index += WARP_SIZE)
        {
            sum_clq_indeg += ld.vertices[index].indeg;

            if (ld.vertices[index].indeg < min_clq_indeg)
            {
                min_clq_indeg = ld.vertices[index].indeg;
                min_indeg_exdeg = ld.vertices[index].exdeg;
            }
            else if (ld.vertices[index].indeg == min_clq_indeg)
            {
                if (ld.vertices[index].exdeg < min_indeg_exdeg)
                {
                    min_indeg_exdeg = ld.vertices[index].exdeg;
                }
            }

            if (ld.vertices[index].indeg + ld.vertices[index].exdeg < min_clq_totaldeg)
            {
                min_clq_totaldeg = ld.vertices[index].indeg + ld.vertices[index].exdeg;
            }
        }

        // get sum
        for (int i = 1; i < 32; i *= 2)
        {
            sum_clq_indeg += __shfl_xor_sync(0xFFFFFFFF, sum_clq_indeg, i);
        }
        if (LANE_IDX == 0)
        {
            // add to shared memory sum
            wd.sum_clq_indeg[WIB_IDX] += sum_clq_indeg;
        }
        __syncwarp();

        // CRITICAL SECTION - each lane then compares their values to the next to get a warp level value
        for (int i = 0; i < WARP_SIZE; i++)
        {
            if (LANE_IDX == i)
            {
                if (min_clq_indeg < wd.min_clq_indeg[WIB_IDX])
                {
                    wd.min_clq_indeg[WIB_IDX] = min_clq_indeg;
                    wd.min_indeg_exdeg[WIB_IDX] = min_indeg_exdeg;
                }
                else if (min_clq_indeg == wd.min_clq_indeg[WIB_IDX])
                {
                    if (min_indeg_exdeg < wd.min_indeg_exdeg[WIB_IDX])
                    {
                        wd.min_indeg_exdeg[WIB_IDX] = min_indeg_exdeg;
                    }
                }

                if (min_clq_totaldeg < wd.min_clq_totaldeg[WIB_IDX])
                {
                    wd.min_clq_totaldeg[WIB_IDX] = min_clq_totaldeg;
                }
            }
            __syncwarp();
        }

        if (LANE_IDX == 0)
        {
            if (wd.min_clq_indeg[WIB_IDX] < dd.minimum_degrees[wd.number_of_members[WIB_IDX]])
            {
                // lower
                wd.lower_bound[WIB_IDX] = d_get_mindeg(wd.number_of_members[WIB_IDX], dd) - min_clq_indeg;

                while (wd.lower_bound[WIB_IDX] <= wd.min_indeg_exdeg[WIB_IDX] && wd.min_clq_indeg[WIB_IDX] + wd.lower_bound[WIB_IDX] <
                                                                                     dd.minimum_degrees[wd.number_of_members[WIB_IDX] + wd.lower_bound[WIB_IDX]])
                {
                    wd.lower_bound[WIB_IDX]++;
                }

                if (wd.min_clq_indeg[WIB_IDX] + wd.lower_bound[WIB_IDX] < dd.minimum_degrees[wd.number_of_members[WIB_IDX] + wd.lower_bound[WIB_IDX]])
                {
                    wd.invalid_bounds[WIB_IDX] = true;
                }

                // upper
                wd.upper_bound[WIB_IDX] = floor(wd.min_clq_totaldeg[WIB_IDX] / (*(dd.minimum_degree_ratio))) + 1 - wd.number_of_members[WIB_IDX];

                if (wd.upper_bound[WIB_IDX] > number_of_candidates)
                {
                    wd.upper_bound[WIB_IDX] = number_of_candidates;
                }

                // tighten
                if (wd.lower_bound[WIB_IDX] < wd.upper_bound[WIB_IDX])
                {
                    // tighten lower
                    for (index = 0; index < wd.lower_bound[WIB_IDX]; index++)
                    {
                        wd.sum_candidate_indeg[WIB_IDX] += dd.candidate_indegs[(WVERTICES_SIZE * WARP_IDX) + index];
                    }

                    while (index < wd.upper_bound[WIB_IDX] && wd.sum_clq_indeg[WIB_IDX] + wd.sum_candidate_indeg[WIB_IDX] < wd.number_of_members[WIB_IDX] *
                                                                                                                                dd.minimum_degrees[wd.number_of_members[WIB_IDX] + index])
                    {
                        wd.sum_candidate_indeg[WIB_IDX] += dd.candidate_indegs[(WVERTICES_SIZE * WARP_IDX) + index];
                        index++;
                    }

                    if (wd.sum_clq_indeg[WIB_IDX] + wd.sum_candidate_indeg[WIB_IDX] < wd.number_of_members[WIB_IDX] * dd.minimum_degrees[wd.number_of_members[WIB_IDX] + index])
                    {
                        wd.invalid_bounds[WIB_IDX] = true;
                    }
                    else
                    {
                        wd.lower_bound[WIB_IDX] = index;

                        wd.tightened_upper_bound[WIB_IDX] = index;

                        while (index < wd.upper_bound[WIB_IDX])
                        {
                            wd.sum_candidate_indeg[WIB_IDX] += dd.candidate_indegs[(WVERTICES_SIZE * WARP_IDX) + index];

                            index++;

                            if (wd.sum_clq_indeg[WIB_IDX] + wd.sum_candidate_indeg[WIB_IDX] >= wd.number_of_members[WIB_IDX] *
                                                                                                   dd.minimum_degrees[wd.number_of_members[WIB_IDX] + index])
                            {
                                wd.tightened_upper_bound[WIB_IDX] = index;
                            }
                        }

                        if (wd.upper_bound[WIB_IDX] > wd.tightened_upper_bound[WIB_IDX])
                        {
                            wd.upper_bound[WIB_IDX] = wd.tightened_upper_bound[WIB_IDX];
                        }

                        if (wd.lower_bound[WIB_IDX] > 1)
                        {
                            wd.min_ext_deg[WIB_IDX] = d_get_mindeg(wd.number_of_members[WIB_IDX] + wd.lower_bound[WIB_IDX], dd);
                        }
                    }
                }
            }
            else
            {
                wd.min_ext_deg[WIB_IDX] = d_get_mindeg(wd.number_of_members[WIB_IDX] + 1,
                                                       dd);

                wd.upper_bound[WIB_IDX] = number_of_candidates;

                if (wd.number_of_members[WIB_IDX] < (*(dd.minimum_clique_size)))
                {
                    wd.lower_bound[WIB_IDX] = (*(dd.minimum_clique_size)) - wd.number_of_members[WIB_IDX];
                }
                else
                {
                    wd.lower_bound[WIB_IDX] = 0;
                }
            }

            if (wd.number_of_members[WIB_IDX] + wd.upper_bound[WIB_IDX] < (*(dd.minimum_clique_size)))
            {
                wd.invalid_bounds[WIB_IDX] = true;
            }

            if (wd.upper_bound[WIB_IDX] < 0 || wd.upper_bound[WIB_IDX] < wd.lower_bound[WIB_IDX])
            {
                wd.invalid_bounds[WIB_IDX] = true;
            }
        }
        __syncwarp();
    }

    __device__ void d_check_for_clique(GPU_Data &dd, Warp_Data &wd, Local_Data &ld)
    {
        bool clique = true;

        for (int k = LANE_IDX; k < wd.number_of_members[WIB_IDX]; k += WARP_SIZE)
        {
            if (ld.vertices[k].indeg < dd.minimum_degrees[wd.number_of_members[WIB_IDX]])
            {
                clique = false;
                break;
            }
        }
        // set to false if any threads in warp do not meet degree requirement
        clique = !(__any_sync(0xFFFFFFFF, !clique));

        // if clique, append it to the global clique buffer
        if (clique)
        {
            if (LANE_IDX == 0)
            {
                atomicMax(reinterpret_cast<unsigned long long *>(dd.max_clique_size),
                          static_cast<unsigned long long>(wd.number_of_members[WIB_IDX]));
            }
            uint64_t start_write = 0;
            if (d_reserve_global_clique(dd, wd.number_of_members[WIB_IDX], start_write))
            {
                for (int k = LANE_IDX; k < wd.number_of_members[WIB_IDX]; k += WARP_SIZE)
                {
                    dd.cliques_vertex[start_write + k] = ld.vertices[k].vertexid;
                }
            }
        }
    }

    __device__ void d_write_to_tasks(GPU_Data &dd, Warp_Data &wd, Local_Data &ld)
    {
        uint64_t start_write = sg->append(wd.total_vertices[WIB_IDX]);
        if (sg->isOverflow())
            return;
        for (int k = LANE_IDX; k < wd.total_vertices[WIB_IDX]; k += WARP_SIZE)
        {
            sg->wrBuff.vertices[start_write + k] = ld.vertices[k].vertexid;
            sg->wrBuff.label[start_write + k] = ld.vertices[k].label;
            sg->wrBuff.indeg[start_write + k] = ld.vertices[k].indeg;
            sg->wrBuff.exdeg[start_write + k] = ld.vertices[k].exdeg;
            sg->wrBuff.lvl2adj[start_write + k] = 0;
        }
    }

    // --- HELPER KERNELS ---

    // searches an int array for a certain int, returns the position in the array that item was found, or -1 if not found
    __device__ int d_bsearch_array(int *search_array, int array_size, int search_number)
    {
        // ALGO - binary
        // TYPE - serial
        // SPEED - 0(log(n))

        int low = 0;
        int high = array_size - 1;

        while (low <= high)
        {
            int mid = (low + high) / 2;

            if (search_array[mid] == search_number)
            {
                return mid;
            }
            else if (search_array[mid] > search_number)
            {
                high = mid - 1;
            }
            else
            {
                low = mid + 1;
            }
        }

        return -1;
    }

    // consider using merge
    __device__ void d_sort(Vertex *target, int size, int (*func)(Vertex &, Vertex &))
    {
        // ALGO - ODD/EVEN
        // TYPE - PARALLEL
        // SPEED - O(n^2)

        Vertex vertex1;
        Vertex vertex2;

        for (int i = 0; i < size; i++)
        {
            for (int j = (i % 2) + (LANE_IDX * 2); j < size - 1; j += (WARP_SIZE * 2))
            {
                vertex1 = target[j];
                vertex2 = target[j + 1];

                if (func(vertex1, vertex2) == 1)
                {
                    target[j] = vertex2;
                    target[j + 1] = vertex1;
                }
            }
            __syncwarp();
        }
    }

    __device__ void d_sort_i(int *target, int size, int (*func)(int, int))
    {
        // ALGO - ODD/EVEN
        // TYPE - PARALLEL
        // SPEED - O(n^2)

        int num1;
        int num2;

        for (int i = 0; i < size; i++)
        {
            for (int j = (i % 2) + (LANE_IDX * 2); j < size - 1; j += (WARP_SIZE * 2))
            {
                num1 = target[j];
                num2 = target[j + 1];

                if (func(num1, num2) == 1)
                {
                    target[j] = num2;
                    target[j + 1] = num1;
                }
            }
            __syncwarp();
        }
    }

    // Quick enumeration order sort keys
    static __device__ int d_sort_vert_Q(Vertex &v1, Vertex &v2)
    {
        // order is: member -> covered -> cands -> cover
        // keys are: indeg -> exdeg -> lvl2adj -> vertexid

        if (v1.label == 1 && v2.label != 1)
            return -1;
        else if (v1.label != 1 && v2.label == 1)
            return 1;
        else if (v1.label == 2 && v2.label != 2)
            return -1;
        else if (v1.label != 2 && v2.label == 2)
            return 1;
        else if (v1.label == 0 && v2.label != 0)
            return -1;
        else if (v1.label != 0 && v2.label == 0)
            return 1;
        else if (v1.label == 3 && v2.label != 3)
            return -1;
        else if (v1.label != 3 && v2.label == 3)
            return 1;
        else if (v1.indeg > v2.indeg)
            return -1;
        else if (v1.indeg < v2.indeg)
            return 1;
        else if (v1.exdeg > v2.exdeg)
            return -1;
        else if (v1.exdeg < v2.exdeg)
            return 1;
        else if (v1.lvl2adj > v2.lvl2adj)
            return -1;
        else if (v1.lvl2adj < v2.lvl2adj)
            return 1;
        else if (v1.vertexid > v2.vertexid)
            return -1;
        else if (v1.vertexid < v2.vertexid)
            return 1;
        else
            return 0;
    }

    static __device__ int d_sort_vert_cv(Vertex &v1, Vertex &v2)
    {
        // put crit adj vertices before candidates

        if (v1.label == 4 && v2.label != 4)
            return -1;
        else if (v1.label != 4 && v2.label == 4)
            return 1;
        else
            return 0;
    }

    static __device__ int d_sort_degs(int n1, int n2)
    {
        // descending order

        if (n1 > n2)
        {
            return -1;
        }
        else if (n1 < n2)
        {
            return 1;
        }
        else
        {
            return 0;
        }
    }

    __device__ int d_get_mindeg(int number_of_members, GPU_Data &dd)
    {
        if (number_of_members < (*(dd.minimum_clique_size)))
        {
            return dd.minimum_degrees[(*(dd.minimum_clique_size))];
        }
        else
        {
            return dd.minimum_degrees[number_of_members];
        }
    }

    __device__ bool d_cand_isvalid_LU(Vertex &vertex, GPU_Data &dd, Warp_Data &wd, Local_Data &ld)
    {
        if (vertex.indeg + vertex.exdeg < dd.minimum_degrees[(*(dd.minimum_clique_size))])
        {
            return false;
        }
        else if (vertex.indeg + vertex.exdeg < d_get_mindeg(wd.number_of_members[WIB_IDX] + vertex.exdeg + 1, dd))
        {
            return false;
        }
        else if (vertex.indeg + vertex.exdeg < wd.min_ext_deg[WIB_IDX])
        {
            return false;
        }
        else if (vertex.indeg + wd.upper_bound[WIB_IDX] - 1 < dd.minimum_degrees[wd.number_of_members[WIB_IDX] + wd.lower_bound[WIB_IDX]])
        {
            return false;
        }
        else if (vertex.indeg + vertex.exdeg < d_get_mindeg(wd.number_of_members[WIB_IDX] + wd.lower_bound[WIB_IDX], dd))
        {
            return false;
        }
        else
        {
            return true;
        }
    }

    __device__ bool d_vert_isextendable_LU(Vertex &vertex, GPU_Data &dd, Warp_Data &wd, Local_Data &ld)
    {
        if (vertex.indeg + vertex.exdeg < dd.minimum_degrees[(*(dd.minimum_clique_size))])
        {
            return false;
        }
        else if (vertex.indeg + vertex.exdeg < d_get_mindeg(wd.number_of_members[WIB_IDX] + vertex.exdeg, dd))
        {
            return false;
        }
        else if (vertex.indeg + vertex.exdeg < wd.min_ext_deg[WIB_IDX])
        {
            return false;
        }
        else if (vertex.exdeg == 0 && vertex.indeg < d_get_mindeg(wd.number_of_members[WIB_IDX] + vertex.exdeg, dd))
        {
            return false;
        }
        else if (vertex.indeg + wd.upper_bound[WIB_IDX] < dd.minimum_degrees[wd.number_of_members[WIB_IDX] + wd.upper_bound[WIB_IDX]])
        {
            return false;
        }
        else if (vertex.indeg + vertex.exdeg < d_get_mindeg(wd.number_of_members[WIB_IDX] + wd.lower_bound[WIB_IDX], dd))
        {
            return false;
        }
        else
        {
            return true;
        }
    }

    __device__ void loadFromHost()
    {
        for (ull k = GLWARPID; k < HOSTCHUNK; k += N_WARPS)
        {
            auto so = sgHost->next();
            if (sgHost->isEnd(so))
                break;

            ull sglen = so.en - so.st;
            ull vt = sg->append(sglen);
            if (sg->isOverflow())
                return;

            for (ull i = so.st + LANEID, j = vt + LANEID; i < so.en; i += 32, j += 32)
            {
                sg->wrBuff.vertices[j] = sgHost->rdBuff.vertices[i];
                sg->wrBuff.label[j] = sgHost->rdBuff.label[i];
                sg->wrBuff.indeg[j] = sgHost->rdBuff.indeg[i];
                sg->wrBuff.exdeg[j] = sgHost->rdBuff.exdeg[i];
                sg->wrBuff.lvl2adj[j] = sgHost->rdBuff.lvl2adj[i];
            }
        }
    }

    __device__ void dumpToHost(SubgraphOffsets *so)
    {
        if (sgHost->overflow[0])
            return;

        ull vt = sgHost->append(so->en - so->st);
        for (ull i = vt + LANEID, j = so->st + LANEID; j < so->en; i += 32, j += 32)
        {
            sgHost->wrBuff.vertices[i] = sg->rdBuff.vertices[j];
            sgHost->wrBuff.label[i] = sg->rdBuff.label[j];
            sgHost->wrBuff.indeg[i] = sg->rdBuff.indeg[j];
            sgHost->wrBuff.exdeg[i] = sg->rdBuff.exdeg[j];
            sgHost->wrBuff.lvl2adj[i] = sg->rdBuff.lvl2adj[j];
        }
    }
};

__device__ int d_sort_vert_Q(Vertex &v1, Vertex &v2)
{
    if (v1.label == 1 && v2.label != 1)
        return -1;
    else if (v1.label != 1 && v2.label == 1)
        return 1;
    else if (v1.label == 2 && v2.label != 2)
        return -1;
    else if (v1.label != 2 && v2.label == 2)
        return 1;
    else if (v1.label == 0 && v2.label != 0)
        return -1;
    else if (v1.label != 0 && v2.label == 0)
        return 1;
    else if (v1.label == 3 && v2.label != 3)
        return -1;
    else if (v1.label != 3 && v2.label == 3)
        return 1;
    else if (v1.indeg > v2.indeg)
        return -1;
    else if (v1.indeg < v2.indeg)
        return 1;
    else if (v1.exdeg > v2.exdeg)
        return -1;
    else if (v1.exdeg < v2.exdeg)
        return 1;
    else if (v1.lvl2adj > v2.lvl2adj)
        return -1;
    else if (v1.lvl2adj < v2.lvl2adj)
        return 1;
    else if (v1.vertexid > v2.vertexid)
        return -1;
    else if (v1.vertexid < v2.vertexid)
        return 1;
    else
        return 0;
}

__device__ int d_sort_vert_cv(Vertex &v1, Vertex &v2)
{
    if (v1.label == 4 && v2.label != 4)
        return -1;
    else if (v1.label != 4 && v2.label == 4)
        return 1;
    else
        return 0;
}

__device__ int d_sort_degs(int n1, int n2)
{
    if (n1 > n2)
        return -1;
    else if (n1 < n2)
        return 1;
    else
        return 0;
}

__device__ void d_print_vertices(Vertex *vertices, int size)
{
    printf("\nOffsets:\n0 %i\nVertex:\n", size);
    for (int i = 0; i < size; i++)
        printf("%i ", vertices[i].vertexid);
    printf("\nLabel:\n");
    for (int i = 0; i < size; i++)
        printf("%i ", vertices[i].label);
    printf("\nIndeg:\n");
    for (int i = 0; i < size; i++)
        printf("%i ", vertices[i].indeg);
    printf("\nExdeg:\n");
    for (int i = 0; i < size; i++)
        printf("%i ", vertices[i].exdeg);
    printf("\nLvl2adj:\n");
    for (int i = 0; i < size; i++)
        printf("%i ", vertices[i].lvl2adj);
    printf("\n");
}

__global__ void transfer_cliques(GPU_Data dd)
{
    if (IDX == 0)
    {
        (*(dd.total_tasks)) = 0;
    }
}

#undef Bwr
#undef Brd
#undef H

#endif
