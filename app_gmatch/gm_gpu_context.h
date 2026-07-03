#ifndef APP_GMATCH_GMATCH_H
#define APP_GMATCH_GMATCH_H

#define SHM_CAP 350
#define BATCH_SIZE 100
#define TEMPSIZE 200'000

using GMBuffer = BufferBase;

class GMGPUContext : public GPUContext<GMBuffer, GMTask>
{

    // =============== dedicated for subgraph matching =====================

    ui *matchOrder;
    ui *ID2order;
    ui *backNeighborCount;
    ui *backNeighbors;
    ui *parent;
    bool *shareIntersection;

    ui *querySize;

    ui *condOrder;
    ui *condNum;

    uintV *preBackNeighborCount;
    uintV *preBackNeighbors;
    uintV *preCondOrder;
    uintV *preCondNum;

    uintV *afterBackNeighborCount;
    uintV *afterBackNeighbors;
    uintV *afterCondOrder;
    uintV *afterCondNum;

    StoreStrategy *strategy;
    ui *movingLvl;

    ull *total_counts;
    ull saved_total_count_ = 0;

    virtual void initialize()
    {
        chkerr(cudaMallocManaged((void **)&querySize, sizeof(ui)));
        querySize[0] = plan.sz;

        chkerr(cudaMalloc((void **)&matchOrder, plan.sz * sizeof(ui)));
        chkerr(cudaMemcpy(matchOrder, plan.matchOrderHost.data(), plan.sz * sizeof(ui), cudaMemcpyHostToDevice));

        chkerr(cudaMalloc((void **)&ID2order, plan.sz * sizeof(ui)));
        chkerr(cudaMemcpy(ID2order, plan.ID2orderHost.data(), plan.sz * sizeof(ui), cudaMemcpyHostToDevice));

        chkerr(cudaMalloc((void **)&backNeighborCount, plan.sz * sizeof(ui)));
        chkerr(cudaMemcpy(backNeighborCount, plan.backNeighborCountHost, plan.sz * sizeof(ui), cudaMemcpyHostToDevice));

        chkerr(cudaMalloc((void **)&backNeighbors, plan.sz * plan.sz * sizeof(ui)));
        chkerr(cudaMemcpy(backNeighbors, plan.backNeighborsHost, plan.sz * plan.sz * sizeof(ui), cudaMemcpyHostToDevice));

        chkerr(cudaMalloc((void **)&parent, plan.sz * sizeof(ui)));
        chkerr(cudaMemcpy(parent, plan.parentHost, plan.sz * sizeof(ui), cudaMemcpyHostToDevice));

        chkerr(cudaMalloc((void **)&condOrder, 2 * plan.sz * plan.sz * sizeof(ui)));
        chkerr(cudaMemcpy(condOrder, plan.condOrderHost, 2 * plan.sz * plan.sz * sizeof(ui), cudaMemcpyHostToDevice));

        chkerr(cudaMalloc((void **)&condNum, plan.sz * sizeof(ui)));
        chkerr(cudaMemcpy(condNum, plan.condNumHost, plan.sz * sizeof(ui), cudaMemcpyHostToDevice));

        chkerr(cudaMalloc((void **)&shareIntersection, plan.sz * sizeof(bool)));
        chkerr(cudaMemcpy(shareIntersection, plan.shareIntersectionHost, plan.sz * sizeof(bool), cudaMemcpyHostToDevice));

        chkerr(cudaMalloc((void **)&preBackNeighborCount, plan.sz * sizeof(ui)));
        chkerr(cudaMemcpy(preBackNeighborCount, plan.preBackNeighborCountHost, plan.sz * sizeof(ui), cudaMemcpyHostToDevice));

        chkerr(cudaMalloc((void **)&preBackNeighbors, plan.sz * plan.sz * sizeof(ui)));
        chkerr(cudaMemcpy(preBackNeighbors, plan.preBackNeighborsHost, plan.sz * plan.sz * sizeof(ui), cudaMemcpyHostToDevice));

        chkerr(cudaMalloc((void **)&preCondOrder, 2 * plan.sz * plan.sz * sizeof(ui)));
        chkerr(cudaMemcpy(preCondOrder, plan.preCondOrderHost, 2 * plan.sz * plan.sz * sizeof(ui), cudaMemcpyHostToDevice));

        chkerr(cudaMalloc((void **)&preCondNum, plan.sz * sizeof(ui)));
        chkerr(cudaMemcpy(preCondNum, plan.preCondNumHost, plan.sz * sizeof(ui), cudaMemcpyHostToDevice));

        chkerr(cudaMalloc((void **)&afterBackNeighborCount, plan.sz * sizeof(ui)));
        chkerr(cudaMemcpy(afterBackNeighborCount, plan.afterBackNeighborCountHost, plan.sz * sizeof(ui), cudaMemcpyHostToDevice));

        chkerr(cudaMalloc((void **)&afterBackNeighbors, plan.sz * plan.sz * sizeof(ui)));
        chkerr(cudaMemcpy(afterBackNeighbors, plan.afterBackNeighborsHost, plan.sz * plan.sz * sizeof(ui), cudaMemcpyHostToDevice));

        chkerr(cudaMalloc((void **)&afterCondOrder, 2 * plan.sz * plan.sz * sizeof(ui)));
        chkerr(cudaMemcpy(afterCondOrder, plan.afterCondOrderHost, 2 * plan.sz * plan.sz * sizeof(ui), cudaMemcpyHostToDevice));

        chkerr(cudaMalloc((void **)&afterCondNum, plan.sz * sizeof(ui)));
        chkerr(cudaMemcpy(afterCondNum, plan.afterCondNumHost, plan.sz * sizeof(ui), cudaMemcpyHostToDevice));

        chkerr(cudaMalloc((void **)&strategy, (plan.sz + 1) * sizeof(StoreStrategy)));
        chkerr(cudaMemcpy(strategy, plan.strategyHost.data(), (plan.sz + 1) * sizeof(StoreStrategy), cudaMemcpyHostToDevice));

        chkerr(cudaMalloc((void **)&movingLvl, plan.sz * sizeof(ui)));
        chkerr(cudaMemcpy(movingLvl, plan.movingLvlHost, plan.sz * sizeof(ui), cudaMemcpyHostToDevice));

        chkerr(cudaMalloc((void **)&tempv, TEMPSIZE * N_WARPS * sizeof(ui)));
        // chkerr(cudaMalloc((void **)&templ, TEMPSIZE * N_WARPS * sizeof(bool)));
        chkerr(cudaMalloc((void **)&pre_intersection, TEMPSIZE * N_WARPS * sizeof(ui)));

        chkerr(cudaMallocManaged((void **)&total_counts, sizeof(ull)));
        total_counts[0] = 0;

        chkerr(cudaMalloc(&(row_ptrs), sizeof(ull) * (gpu_dg.GetVertexCount() + 1)));
        chkerr(cudaMalloc(&(cols), sizeof(VertexID) * gpu_dg.GetEdgeCount()));
        cudaMemcpy(row_ptrs, gpu_dg.GetRowPtrs(), sizeof(ull) * (gpu_dg.GetVertexCount() + 1), cudaMemcpyHostToDevice);
        cudaMemcpy(cols, gpu_dg.GetCols(), sizeof(VertexID) * gpu_dg.GetEdgeCount(), cudaMemcpyHostToDevice);
    }

public:
    struct BatchAppendResult
    {
        ull vt;
        bool to_host;
        bool failed;
    };

    // temporary array to store local candidate ?????
    ui *tempv;
    // bool *templ;

    ui *pre_intersection;
    ull get_results()
    {
        return total_counts[0];
    }

    void init_chunk() override
    {
        saved_total_count_ = total_counts[0];
    }

    void abort_chunk() override
    {
        total_counts[0] = saved_total_count_;
    }

    virtual void load_graph(ull *&row_ptrs, VertexID *&cols)
    {
    }

    __device__ BatchAppendResult append_batch(ull sglen, ui num, StoreStrategy mode)
    {
        BatchAppendResult res{0, false, false};
        ull lane_vt = 0;
        unsigned int lane_to_host = 0;
        unsigned int lane_failed = 0;

        ull ot = 0, vt = 0;
        ull host_ot = 0, host_vt = 0;

        if (mode == StoreStrategy::EXPAND)
        {
            if (LANEID == 0)
            {
                ot = atomicAdd(Bwr.otail, 2ULL * num);
                vt = atomicAdd(Bwr.vtail, sglen * num);
                atomicAdd(Bwr.n_tasks_proc, num);

                const ull next_ot = ot + 2ULL * num;
                const ull next_vt = vt + sglen * num;
                const ull vertex_base = Bwr.second_buffer ? Bwr.capacity[0] / 2 : 0;
                const ull vertex_span = Bwr.second_buffer ? Bwr.capacity[0] / 2 : Bwr.capacity[0];

                if (next_ot <= Bwr.capacity[0] && (next_vt - vertex_base) <= vertex_span)
                {
                    lane_vt = vt;
                }
                else
                {
                    Bwr.overflow[0] = true;
                    lane_to_host = 1;
                    host_ot = atomicAdd(H.otail, 2ULL * num);
                    host_vt = atomicAdd(H.vtail, sglen * num);
                    atomicAdd(H.n_tasks_proc, num);

                    if (host_ot + 2ULL * num > HOST_OFFSET_SZ || host_vt + sglen * num > H.capacity[0])
                    {
                        H.overflow[0] = true;
                        lane_failed = 1;
                    }
                    else
                    {
                        lane_vt = host_vt;
                    }
                }
            }

            ot = __shfl_sync(FULL, ot, 0);
            vt = __shfl_sync(FULL, vt, 0);
            host_ot = __shfl_sync(FULL, host_ot, 0);
            host_vt = __shfl_sync(FULL, host_vt, 0);
            lane_vt = __shfl_sync(FULL, lane_vt, 0);
            lane_to_host = __shfl_sync(FULL, lane_to_host, 0);
            lane_failed = __shfl_sync(FULL, lane_failed, 0);

            if (lane_to_host)
            {
                for (ui i = LANEID; i < num; i += 32)
                {
                    Bwr.offsets[ot + i * 2] = vt + sglen * i;
                    Bwr.offsets[ot + i * 2 + 1] = vt + sglen * i;
                }

                if (!lane_failed)
                {
                    for (ui i = LANEID; i < num; i += 32)
                    {
                        H.offsets[host_ot + i * 2] = host_vt + sglen * i;
                        H.offsets[host_ot + i * 2 + 1] = host_vt + sglen * (i + 1);
                    }
                }
            }
            else
            {
                for (ui i = LANEID; i < num; i += 32)
                {
                    Bwr.offsets[ot + i * 2] = vt + sglen * i;
                    Bwr.offsets[ot + i * 2 + 1] = vt + sglen * (i + 1);
                }
            }
        }
        else
        {
            if (LANEID == 0)
            {
                const ull total_len = sglen + num;
                ot = atomicAdd(Bwr.otail, 2ULL);
                vt = atomicAdd(Bwr.vtail, total_len);
                atomicAdd(Bwr.n_tasks_proc, num);

                const ull next_ot = ot + 2;
                const ull next_vt = vt + total_len;
                const ull vertex_base = Bwr.second_buffer ? Bwr.capacity[0] / 2 : 0;
                const ull vertex_span = Bwr.second_buffer ? Bwr.capacity[0] / 2 : Bwr.capacity[0];

                if (next_ot <= Bwr.capacity[0] && (next_vt - vertex_base) <= vertex_span)
                {
                    Bwr.offsets[ot] = vt;
                    Bwr.offsets[ot + 1] = vt + total_len;
                    lane_vt = vt;
                }
                else
                {
                    Bwr.overflow[0] = true;
                    lane_to_host = 1;
                    Bwr.offsets[ot] = vt;
                    Bwr.offsets[ot + 1] = vt;

                    host_ot = atomicAdd(H.otail, 2ULL);
                    host_vt = atomicAdd(H.vtail, total_len);
                    atomicAdd(H.n_tasks_proc, num);

                    if (host_ot + 2 > HOST_OFFSET_SZ || host_vt + total_len > H.capacity[0])
                    {
                        H.overflow[0] = true;
                        lane_failed = 1;
                    }
                    else
                    {
                        H.offsets[host_ot] = host_vt;
                        H.offsets[host_ot + 1] = host_vt + total_len;
                        lane_vt = host_vt;
                    }
                }
            }

            lane_vt = __shfl_sync(FULL, lane_vt, 0);
            lane_to_host = __shfl_sync(FULL, lane_to_host, 0);
            lane_failed = __shfl_sync(FULL, lane_failed, 0);
        }

        res.vt = lane_vt;
        res.to_host = lane_to_host;
        res.failed = lane_failed;
        return res;
    }

    __device__ virtual void process(GMBuffer &Brd, ull *row_ptrs, VertexID *cols) {}

    virtual void move_tasks_from_Sc(std::vector<GMTask *> &src_tasks, GMBuffer &H)
    {
        cout << "H to D: " << src_tasks.size() << endl;
        for (GMTask *task : src_tasks)
        {
            ui sz = task->context.cur_depth;
            ull loc = H.append_host(sz + 1);
            H.vertices[loc] = sz;
            for (ui i = 0; i < sz; ++i)
                H.vertices[loc + 1 + i] = task->context.embedding[matching_order[i]];
            delete task;
        }
        // cout<<"All copied"<<endl;
        src_tasks.clear();
    }

    virtual void move_tasks_to_Sc(vector<GMTask *> &collector, GMBuffer &H)
    {
        cout << "D to H" << endl;
        for (ui i = 0; i < gpu_to_host_transfer_size_g; i++)
        {
            SubgraphOffsets so = H.pop_host();
            if (so.empty())
                break;
            VertexID *data = H.vertices;
            ull sglen64 = data[so.st];
            if (sglen64 == 0 || sglen64 > gpu_qg.GetVertexCount())
                throw std::runtime_error("Invalid GM task header while moving tasks from host buffer");
            ui sglen = static_cast<ui>(sglen64);

            if (plan.strategyHost[sglen] == StoreStrategy::EXPAND)
            {
                GMTask *task = new GMTask();
                task->context.embedding = new ui[gpu_qg.GetVertexCount()];
                task->context.idx_embedding = new ui[gpu_qg.GetVertexCount()];
                task->context.cur_depth = sglen;

                for (ui i = 0; i < sglen; i++)
                {
                    ui v = data[so.st + 1 + i];
                    ui qv = matching_order[i];
                    task->context.embedding[qv] = v;
                    int idx = binary_search(i, v);
                    if (idx == -1)
                    {
                        // it's an invalid task
                        delete task;
                        task = nullptr;
                        break;
                    }
                    task->context.idx_embedding[qv] = static_cast<ui>(idx); // if binary search returned -1, it's invalid
                }
                if (task)
                    collector.push_back(task);
            }
            else
            {
                ui *idx = new ui[sglen - 1]; // common idx_maping
                bool valid_prefix=true;
                for (ui i = 0; i < sglen - 1; i++)
                {
                    int idv = binary_search(i, data[so.st + 1 + i]);
                    if (idv == -1)
                    {
                        valid_prefix=false;
                        break;
                    }
                    idx[i] = static_cast<ui>(idv);
                }

                for (ull j = so.st + sglen; valid_prefix && j < so.en; j++)
                {
                    int idv = binary_search(sglen - 1, data[j]); // -1 because candidates index start from 0
                    if (idv == -1)
                        continue;
                    GMTask *task = new GMTask();
                    task->context.embedding = new ui[gpu_qg.GetVertexCount()];
                    task->context.idx_embedding = new ui[gpu_qg.GetVertexCount()];
                    task->context.cur_depth = sglen;

                    for (ui i = 0; i < sglen - 1; ++i)
                    {
                        ui qv = matching_order[i];
                        task->context.embedding[qv] = data[so.st + 1 + i];
                        task->context.idx_embedding[qv] = idx[i];
                    }
                    ui qv = matching_order[sglen - 1];
                    task->context.embedding[qv] = data[j];
                    task->context.idx_embedding[qv] = static_cast<ui>(idv);
                    collector.push_back(task);
                }
                delete[] idx;
                
            }
        }
    }

    __device__ ui writeToTemp(ui v, ui l, bool pred, unsigned int sglen)
    {
        unsigned int loc = scanIndex(pred) + sglen;
        // popc gives inclusive sum scan, subtract pred to make it exclusive
        // add sglen to find exact location in the temp
        assert(loc < TEMPSIZE);
        if (pred)
        {
            this->tempv[loc + GLWARPID * TEMPSIZE] = v;
            // this->templ[loc + GLWARPID * TEMPSIZE] = l;
        }
        if (LANEID == 31) // last lane's loc+pred is number of items found in this scan
            sglen = loc + pred;
        sglen = __shfl_sync(FULL, sglen, 31);
        return sglen;
    }

    __device__ ui writeToPreIntersection(ui v, bool pred, unsigned int sglen)
    {
        unsigned int loc = scanIndex(pred) + sglen;
        // popc gives inclusive sum scan, subtract pred to make it exclusive
        // add sglen to find exact location in the temp
        assert(loc < TEMPSIZE);
        if (pred)
        {
            this->pre_intersection[loc + GLWARPID * TEMPSIZE] = v;
        }
        if (LANEID == 31) // last lane's loc+pred is number of items found in this scan
            sglen = loc + pred;
        sglen = __shfl_sync(FULL, sglen, 31);
        return sglen;
    }

    __device__ virtual void generateInitialTasks(VertexID *sources, ull *sources_num, ull *v_proc, GMBuffer &Bwr, ull *row_ptrs, VertexID *cols)
    {
        while (true)
        {
            if (isLevelFilled())
                return;
            ull vp;
            if (LANEID == 0)
            {
                vp = atomicAdd(v_proc, 1);
            }
            vp = __shfl_sync(FULL, vp, 0);
            if (vp >= sources_num[0])
                return;

            ull v = sources[vp];

            ull vt = Bwr.append(2); // header + one matched vertex
            if (LANEID == 0)
            {
                Bwr.vertices[vt] = 1;
                Bwr.vertices[vt + 1] = v; // sources[v];
            }
        }
    }
    __device__ virtual void extend(GMBuffer &Brd, GMBuffer &Bwr, GMBuffer &H, ull *row_ptrs, VertexID *cols)
    {
        StoreStrategy CUR_MODE, NEXT_MODE;

        __shared__ ui partial_subgraphs[WARPS_EACH_BLK][8];
        __shared__ ull warp_sums[WARPS_EACH_BLK];

        ull local_thread_count = 0;

        while (true)
        {
            if (isLevelFilled())
                break;
            SubgraphOffsets so = Brd.next();

            if (so.empty())
                break;

            ull sglen64 = Brd.vertices[so.st];
            if (sglen64 == 0 || sglen64 > querySize[0])
            {
                if (LANEID == 0)
                    printf("GM invalid task header: st=%llu en=%llu header=%llu qsz=%u\n",
                           so.st, so.en, sglen64, querySize[0]);
                asm("trap;");
            }
            ui id = static_cast<ui>(sglen64);
            ui sglen = static_cast<ui>(sglen64);
            CUR_MODE = strategy[id];

            NEXT_MODE = strategy[id + 1];
            ui u = matchOrder[id];

            // do pre-intersection here

            if (shareIntersection[id] && CUR_MODE == StoreStrategy::PREFIX)
            {
                if (LANEID < sglen - 1)
                    partial_subgraphs[WARPID][LANEID] = Brd.vertices[so.st + 1 + LANEID];
                __syncwarp();

                // finds least degree vertex
                ui bnCount = preBackNeighborCount[id];
                if (bnCount == 0)
                {
                    if (LANEID == 0)
                        printf("GM invalid prefix bnCount=0: id=%u sglen=%u st=%llu en=%llu mode=%d share=%d\n",
                               id, sglen, so.st, so.en, static_cast<int>(CUR_MODE), static_cast<int>(shareIntersection[id]));
                    asm("trap;");
                }
                ui u_prime = preBackNeighbors[querySize[0] * id];
                if (u_prime >= querySize[0])
                {
                    if (LANEID == 0)
                        printf("GM invalid prefix u_prime: id=%u sglen=%u bnCount=%u u_prime=%u qsz=%u st=%llu en=%llu\n",
                               id, sglen, bnCount, u_prime, querySize[0], so.st, so.en);
                    asm("trap;");
                }
                ui u_prime_M = partial_subgraphs[WARPID][ID2order[u_prime]];
                uintE u_prime_M_st = row_ptrs[u_prime_M];
                uintE u_prime_M_en = row_ptrs[u_prime_M + 1];
                uintE min = u_prime_M_en - u_prime_M_st;
                ui parent_u = u_prime;
                for (ui i = 1; i < bnCount; ++i)
                {
                    ui u_prime = preBackNeighbors[querySize[0] * id + i];
                    ui u_prime_M = partial_subgraphs[WARPID][ID2order[u_prime]];
                    uintE u_prime_M_st = row_ptrs[u_prime_M];
                    uintE u_prime_M_en = row_ptrs[u_prime_M + 1];
                    uintE neigh_len = u_prime_M_en - u_prime_M_st;
                    if (neigh_len < min)
                    {
                        min = neigh_len;
                        parent_u = u_prime;
                    }
                }

                ui parent_u_M = partial_subgraphs[WARPID][ID2order[parent_u]];
                uintE pu_st = row_ptrs[parent_u_M];
                uintE pu_en = row_ptrs[parent_u_M + 1];
                ui len = 0;
                bool pred;
                ui condCount = preCondNum[u];
                ui vertex;
                uintE base_i = pu_st;

                do
                {
                    len = 0;

                    for (; base_i < pu_en; base_i += 32)
                    {
                        uintE il = base_i + LANEID;
                        pred = il < pu_en;

                        if (pred)
                        {
                            vertex = cols[il];

                            for (ui k = 0; k < condCount; ++k)
                            {
                                ui cond = preCondOrder[u * querySize[0] * 2 + 2 * k];
                                ui cond_vertex = preCondOrder[u * querySize[0] * 2 + 2 * k + 1];
                                ui cond_vertex_M = partial_subgraphs[WARPID][ID2order[cond_vertex]];
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
                            {
                                for (ui j = 0; j < bnCount; ++j)
                                {
                                    ui u_prime = preBackNeighbors[querySize[0] * id + j];
                                    if (u_prime == parent_u)
                                        continue;
                                    ui u_prime_M = partial_subgraphs[WARPID][ID2order[u_prime]];
                                    uintE u_prime_M_st = row_ptrs[u_prime_M];
                                    uintE u_prime_M_en = row_ptrs[u_prime_M + 1];
                                    pred = binarySearch(cols, u_prime_M_st, u_prime_M_en, vertex);
                                    if (!pred)
                                        break;
                                }
                            }
                        }
                        ui val = pred ? vertex : 0;
                        len = writeToPreIntersection(val, pred, len);

                        if (len >= TEMPSIZE - 32)
                        {
                            if (GTHID == 0)
                                printf("* ");
                            base_i += 32;
                            break;
                        }
                    }
                    ui pre_len = len;

                    const ui prefix_slot = sglen - 1;
                    for (ull subgraph_id = so.st + sglen; subgraph_id < so.en; ++subgraph_id)
                    {

                        if (LANEID == 0)
                        {
                            partial_subgraphs[WARPID][prefix_slot] = Brd.vertices[subgraph_id];
                        }
                        __syncwarp();

                        ui len = 0;
                        bool pred;
                        ui condCount = afterCondNum[u];
                        ui bnCount = afterBackNeighborCount[id];
                        ui vertex;

                        for (uintE i = 0; i < pre_len; i += 32)
                        {
                            uintE il = i + LANEID;
                            pred = il < pre_len;

                            if (pred)
                            {
                                vertex = pre_intersection[GLWARPID * TEMPSIZE + il];
                                for (ui k = 0; k < condCount; ++k)
                                {
                                    ui cond = afterCondOrder[u * querySize[0] * 2 + 2 * k];
                                    ui cond_vertex = afterCondOrder[u * querySize[0] * 2 + 2 * k + 1];
                                    ui cond_vertex_M = partial_subgraphs[WARPID][ID2order[cond_vertex]];
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
                                {
                                    for (ui j = 0; j < bnCount; ++j)
                                    {
                                        ui u_prime = afterBackNeighbors[querySize[0] * id + j];
                                        ui u_prime_M = partial_subgraphs[WARPID][ID2order[u_prime]];
                                        uintE u_prime_M_st = row_ptrs[u_prime_M];
                                        uintE u_prime_M_en = row_ptrs[u_prime_M + 1];
                                        pred = binarySearch(cols, u_prime_M_st, u_prime_M_en, vertex);
                                        if (!pred)
                                            break;
                                    }
                                    if (pred)
                                    {
                                        if (sglen + 1 == querySize[0])
                                        {
                                            // atomicAdd(&total_counts_[GLWARPID], 1);
                                            local_thread_count += 1;
                                        }
                                    }
                                }
                            }
                            if (sglen + 1 != querySize[0])
                            {
                                ui val = pred ? vertex : 0;
                                len = writeToTemp(val, 1, pred, len);
                            }
                        }

                        if (sglen + 1 == querySize[0])
                        {
                            // total_counts_[GLWARPID] += len;
                            continue;
                        }
                        else
                        {
                            if (NEXT_MODE == StoreStrategy::EXPAND)
                            {

                                for (ui batch_id = 0; batch_id < len; batch_id += BATCH_SIZE)
                                {
                                    ui min = len - batch_id < BATCH_SIZE ? len - batch_id : BATCH_SIZE;
                                    auto alloc = append_batch(sglen + 2, min, StoreStrategy::EXPAND);
                                    if (alloc.failed)
                                        return;
                                    auto &dst = alloc.to_host ? H : Bwr;
                                    auto vt = alloc.vt;
                                    for (ui i = LANEID; i < min; i += 32)
                                    {
                                        dst.vertices[vt + i * (sglen + 2)] = sglen + 1;
                                        for (ui j = 0; j < sglen; ++j)
                                        {
                                            auto k = vt + i * (sglen + 2) + 1 + j;
                                            dst.vertices[k] = partial_subgraphs[WARPID][j];
                                        }
                                        dst.vertices[vt + i * (sglen + 2) + 1 + sglen] = tempv[batch_id + i + GLWARPID * TEMPSIZE]; // add q on the back
                                    }
                                }
                            }
                            else if (NEXT_MODE == StoreStrategy::PREFIX)
                            {
                                for (ui batch_id = 0; batch_id < len; batch_id += BATCH_SIZE)
                                {
                                    ui min = len - batch_id < BATCH_SIZE ? len - batch_id : BATCH_SIZE;
                                    auto alloc = append_batch(sglen + 1, min, StoreStrategy::PREFIX);
                                    if (alloc.failed)
                                        return;
                                    auto &dst = alloc.to_host ? H : Bwr;
                                    auto vt = alloc.vt;
                                    if (LANEID == 0)
                                        dst.vertices[vt] = sglen + 1;
                                    for (ui i = LANEID; i < sglen; i += 32)
                                    {
                                        auto k = vt + 1 + i;
                                        dst.vertices[k] = partial_subgraphs[WARPID][i];
                                    }
                                    for (ui i = LANEID; i < min; i += 32)
                                        dst.vertices[vt + 1 + sglen + i] = tempv[batch_id + i + GLWARPID * TEMPSIZE]; // add q on the back
                                }
                            }
                        }
                    }
                } while (base_i < pu_en);
            }
            else
            {
                if (LANEID < sglen)
                    partial_subgraphs[WARPID][LANEID] = Brd.vertices[so.st + 1 + LANEID];
                __syncwarp();

                // select the pivot with least # of candidates
                ui bnCount = backNeighborCount[id];
                if (bnCount == 0)
                {
                    if (LANEID == 0)
                        printf("GM invalid expand bnCount=0: id=%u sglen=%u st=%llu en=%llu mode=%d share=%d\n",
                               id, sglen, so.st, so.en, static_cast<int>(CUR_MODE), static_cast<int>(shareIntersection[id]));
                    asm("trap;");
                }
                ui u_prime = backNeighbors[querySize[0] * id];
                if (u_prime >= querySize[0])
                {
                    if (LANEID == 0)
                        printf("GM invalid expand u_prime: id=%u sglen=%u bnCount=%u u_prime=%u qsz=%u st=%llu en=%llu\n",
                               id, sglen, bnCount, u_prime, querySize[0], so.st, so.en);
                    asm("trap;");
                }
                ui u_prime_M = partial_subgraphs[WARPID][ID2order[u_prime]];
                uintE u_prime_M_st = row_ptrs[u_prime_M];
                uintE u_prime_M_en = row_ptrs[u_prime_M + 1];
                uintE min = u_prime_M_en - u_prime_M_st;
                ui parent_u = u_prime;
                for (ui i = 1; i < bnCount; ++i)
                {
                    ui u_prime = backNeighbors[querySize[0] * id + i];
                    ui u_prime_M = partial_subgraphs[WARPID][ID2order[u_prime]];
                    uintE u_prime_M_st = row_ptrs[u_prime_M];
                    uintE u_prime_M_en = row_ptrs[u_prime_M + 1];
                    uintE neigh_len = u_prime_M_en - u_prime_M_st;
                    if (neigh_len < min)
                    {
                        min = neigh_len;
                        parent_u = u_prime;
                    }
                }

                ui parent_u_M = partial_subgraphs[WARPID][ID2order[parent_u]];

                uintE pu_st = row_ptrs[parent_u_M];
                uintE pu_en = row_ptrs[parent_u_M + 1];

                ui len = 0;
                bool pred;
                ui condCount = condNum[u];
                ui vertex;
                uintE base_i = pu_st;

                do
                {
                    len = 0;
                    for (; base_i < pu_en; base_i += 32)
                    {
                        uintE il = base_i + LANEID;
                        pred = il < pu_en;

                        if (pred)
                        {
                            vertex = cols[il];
                            for (ui k = 0; k < condCount; ++k)
                            {
                                ui cond = condOrder[u * querySize[0] * 2 + 2 * k];
                                ui cond_vertex = condOrder[u * querySize[0] * 2 + 2 * k + 1];
                                ui cond_vertex_M = partial_subgraphs[WARPID][ID2order[cond_vertex]];
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
                            {
                                for (ui j = 0; j < bnCount; ++j)
                                {
                                    ui u_prime = backNeighbors[querySize[0] * id + j];
                                    if (u_prime == parent_u)
                                        continue;
                                    ui u_prime_M = partial_subgraphs[WARPID][ID2order[u_prime]];
                                    uintE u_prime_M_st = row_ptrs[u_prime_M];
                                    uintE u_prime_M_en = row_ptrs[u_prime_M + 1];
                                    pred = binarySearch(cols, u_prime_M_st, u_prime_M_en, vertex);
                                    if (!pred)
                                        break;
                                }
                                if (pred)
                                {
                                    if (sglen + 1 == querySize[0])
                                    {
                                        // atomicAdd(&total_counts_[GLWARPID], 1); // TODO: no good
                                        local_thread_count += 1;
                                    }
                                }
                            }
                        }
                        if (sglen + 1 != querySize[0])
                        {
                            ui val = pred ? vertex : 0;
                            len = writeToTemp(val, 1, pred, len);

                            if (len >= TEMPSIZE - 32)
                            {
                                if (GTHID == 0)
                                    printf("# ");
                                base_i += 32;
                                break;
                            }
                        }
                    }

                    if (sglen + 1 == querySize[0])
                    {
                        // total_counts_[GLWARPID] += len;
                        continue;
                    }
                    else
                    {
                        if (NEXT_MODE == StoreStrategy::EXPAND)
                        {
                            for (ui batch_id = 0; batch_id < len; batch_id += BATCH_SIZE)
                            {
                                ui min = len - batch_id < BATCH_SIZE ? len - batch_id : BATCH_SIZE;
                                auto alloc = append_batch(sglen + 2, min, StoreStrategy::EXPAND);
                                if (alloc.failed)
                                    return;
                                auto &dst = alloc.to_host ? H : Bwr;
                                auto vt = alloc.vt;
                                for (ui i = LANEID; i < min; i += 32)
                                {
                                    dst.vertices[vt + i * (sglen + 2)] = sglen + 1;
                                    for (ui j = 0; j < sglen; ++j)
                                    {
                                        auto k = vt + i * (sglen + 2) + 1 + j;
                                        dst.vertices[k] = partial_subgraphs[WARPID][j];
                                    }
                                    dst.vertices[vt + i * (sglen + 2) + 1 + sglen] = tempv[batch_id + i + GLWARPID * TEMPSIZE]; // add q on the back
                                }
                            }
                        }
                        else if (NEXT_MODE == StoreStrategy::PREFIX)
                        {
                            for (ui batch_id = 0; batch_id < len; batch_id += BATCH_SIZE)
                            {
                                ui min = len - batch_id < BATCH_SIZE ? len - batch_id : BATCH_SIZE;
                                auto alloc = append_batch(sglen + 1, min, StoreStrategy::PREFIX);
                                if (alloc.failed)
                                    return;
                                auto &dst = alloc.to_host ? H : Bwr;
                                auto vt = alloc.vt;
                                if (LANEID == 0)
                                    dst.vertices[vt] = sglen + 1;
                                for (ui i = LANEID; i < sglen; i += 32)
                                {
                                    auto k = vt + 1 + i;
                                    dst.vertices[k] = partial_subgraphs[WARPID][i];
                                }
                                for (ui i = LANEID; i < min; i += 32)
                                    dst.vertices[vt + 1 + sglen + i] = tempv[batch_id + i + GLWARPID * TEMPSIZE]; // add q on the back
                            }
                        }
                    }
                } while (base_i < pu_en);
            }
        }
        ull warp_count = local_thread_count;
        for (int offset = 16; offset > 0; offset >>= 1)
            warp_count += __shfl_down_sync(FULL, warp_count, offset);

        if (LANEID == 0)
            warp_sums[WARPID] = warp_count;

        __syncthreads();

        if (WARPID == 0)
        {
            ull block_count = (LANEID < WARPS_EACH_BLK) ? warp_sums[LANEID] : 0;
            for (int offset = 16; offset > 0; offset >>= 1)
                block_count += __shfl_down_sync(FULL, block_count, offset);

            if (LANEID == 0)
                atomicAdd(total_counts, block_count);
        }
    }
};

#endif
