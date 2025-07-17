#ifndef APP_GMATCH_GMATCH_H
#define APP_GMATCH_GMATCH_H

#define SHM_CAP 350
#define BATCH_SIZE 10000000
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




    void allocateGMMemory()
    {
        chkerr(cudaMallocManaged((void **)&querySize, sizeof(ui)));
        querySize[0] = gmp.sz;

        chkerr(cudaMalloc((void **)&matchOrder, gmp.sz * sizeof(ui)));
        chkerr(cudaMemcpy(matchOrder, gmp.matchOrderHost.data(), gmp.sz * sizeof(ui), cudaMemcpyHostToDevice));

        chkerr(cudaMalloc((void **)&ID2order, gmp.sz * sizeof(ui)));
        chkerr(cudaMemcpy(ID2order, gmp.ID2orderHost.data(), gmp.sz * sizeof(ui), cudaMemcpyHostToDevice));

        chkerr(cudaMalloc((void **)&backNeighborCount, gmp.sz * sizeof(ui)));
        chkerr(cudaMemcpy(backNeighborCount, gmp.backNeighborCountHost, gmp.sz * sizeof(ui), cudaMemcpyHostToDevice));

        chkerr(cudaMalloc((void **)&backNeighbors, gmp.sz * gmp.sz * sizeof(ui)));
        chkerr(cudaMemcpy(backNeighbors, gmp.backNeighborsHost, gmp.sz * gmp.sz * sizeof(ui), cudaMemcpyHostToDevice));

        chkerr(cudaMalloc((void **)&parent, gmp.sz * sizeof(ui)));
        chkerr(cudaMemcpy(parent, gmp.parentHost, gmp.sz * sizeof(ui), cudaMemcpyHostToDevice));

        chkerr(cudaMalloc((void **)&condOrder, 2 * gmp.sz * gmp.sz * sizeof(ui)));
        chkerr(cudaMemcpy(condOrder, gmp.condOrderHost, 2 * gmp.sz * gmp.sz * sizeof(ui), cudaMemcpyHostToDevice));

        chkerr(cudaMalloc((void **)&condNum, gmp.sz * sizeof(ui)));
        chkerr(cudaMemcpy(condNum, gmp.condNumHost, gmp.sz * sizeof(ui), cudaMemcpyHostToDevice));

        chkerr(cudaMalloc((void **)&shareIntersection, gmp.sz * sizeof(bool)));
        chkerr(cudaMemcpy(shareIntersection, gmp.shareIntersectionHost, gmp.sz * sizeof(bool), cudaMemcpyHostToDevice));

        chkerr(cudaMalloc((void **)&preBackNeighborCount, gmp.sz * sizeof(ui)));
        chkerr(cudaMemcpy(preBackNeighborCount, gmp.preBackNeighborCountHost, gmp.sz * sizeof(ui), cudaMemcpyHostToDevice));

        chkerr(cudaMalloc((void **)&preBackNeighbors, gmp.sz * gmp.sz * sizeof(ui)));
        chkerr(cudaMemcpy(preBackNeighbors, gmp.preBackNeighborsHost, gmp.sz * gmp.sz * sizeof(ui), cudaMemcpyHostToDevice));

        chkerr(cudaMalloc((void **)&preCondOrder, 2 * gmp.sz * gmp.sz * sizeof(ui)));
        chkerr(cudaMemcpy(preCondOrder, gmp.preCondOrderHost, 2 * gmp.sz * gmp.sz * sizeof(ui), cudaMemcpyHostToDevice));

        chkerr(cudaMalloc((void **)&preCondNum, gmp.sz * sizeof(ui)));
        chkerr(cudaMemcpy(preCondNum, gmp.preCondNumHost, gmp.sz * sizeof(ui), cudaMemcpyHostToDevice));

        chkerr(cudaMalloc((void **)&afterBackNeighborCount, gmp.sz * sizeof(ui)));
        chkerr(cudaMemcpy(afterBackNeighborCount, gmp.afterBackNeighborCountHost, gmp.sz * sizeof(ui), cudaMemcpyHostToDevice));

        chkerr(cudaMalloc((void **)&afterBackNeighbors, gmp.sz * gmp.sz * sizeof(ui)));
        chkerr(cudaMemcpy(afterBackNeighbors, gmp.afterBackNeighborsHost, gmp.sz * gmp.sz * sizeof(ui), cudaMemcpyHostToDevice));

        chkerr(cudaMalloc((void **)&afterCondOrder, 2 * gmp.sz * gmp.sz * sizeof(ui)));
        chkerr(cudaMemcpy(afterCondOrder, gmp.afterCondOrderHost, 2 * gmp.sz * gmp.sz * sizeof(ui), cudaMemcpyHostToDevice));

        chkerr(cudaMalloc((void **)&afterCondNum, gmp.sz * sizeof(ui)));
        chkerr(cudaMemcpy(afterCondNum, gmp.afterCondNumHost, gmp.sz * sizeof(ui), cudaMemcpyHostToDevice));

        chkerr(cudaMalloc((void **)&strategy, (gmp.sz + 1) * sizeof(StoreStrategy)));
        chkerr(cudaMemcpy(strategy, gmp.strategyHost.data(), (gmp.sz + 1) * sizeof(StoreStrategy), cudaMemcpyHostToDevice));

        chkerr(cudaMalloc((void **)&movingLvl, gmp.sz * sizeof(ui)));
        chkerr(cudaMemcpy(movingLvl, gmp.movingLvlHost, gmp.sz * sizeof(ui), cudaMemcpyHostToDevice));
    }

    virtual void initialize()
    {
        allocateGMMemory();

        chkerr(cudaMalloc((void **)&tempv, TEMPSIZE * N_WARPS * sizeof(ui)));
        // chkerr(cudaMalloc((void **)&templ, TEMPSIZE * N_WARPS * sizeof(bool)));
        chkerr(cudaMalloc((void **)&pre_intersection, TEMPSIZE * N_WARPS * sizeof(ui)));

        chkerr(cudaMallocManaged((void **)&total_counts, N_WARPS * sizeof(ull)));
        for (ui i = 0; i < N_WARPS; i++)
            total_counts[i] = 0;

        // chkerr(cudaMalloc(&(row_ptrs), sizeof(ull) * (data_graph.getVerticesCount() + 1)));
        // chkerr(cudaMalloc(&(cols), sizeof(VertexID) * data_graph.getEdgesCount()));
        // ui n=data_graph.getVerticesCount() + 1;
        // ui* offs=data_graph.getOffsets();
        // ull *temp=new ull[n];
        // for(ui i=0;i<n;i++) temp[i]=offs[i];
        // cudaMemcpy(row_ptrs, temp, sizeof(ull) * (data_graph.getVerticesCount() + 1), cudaMemcpyHostToDevice);
        // cudaMemcpy(cols, data_graph.getNeighbors(), sizeof(VertexID) * data_graph.getEdgesCount(), cudaMemcpyHostToDevice);
    
        chkerr(cudaMalloc(&(row_ptrs), sizeof(ull) * (gpuG.GetVertexCount() + 1)));
        chkerr(cudaMalloc(&(cols), sizeof(VertexID) * gpuG.GetEdgeCount()));
        cudaMemcpy(row_ptrs, gpuG.GetRowPtrs(), sizeof(ull) * (gpuG.GetVertexCount() + 1), cudaMemcpyHostToDevice);
        cudaMemcpy(cols, gpuG.GetCols(), sizeof(VertexID) * gpuG.GetEdgeCount(), cudaMemcpyHostToDevice);
        
    }

public:
    // temporary array to store local candidate ?????
    ui *tempv;
    // bool *templ;

    ui *pre_intersection;
    ui get_results()
    {
        ull res = 0;
        for (ui i = 0; i < N_WARPS; i++){
            res += total_counts[i];
        }
        return res;
    }
    virtual void load_graph(ull *&row_ptrs, VertexID *&cols)
    {
    }

    __device__ virtual void process(GMBuffer &Brd, ull *row_ptrs, VertexID *cols) {}

    virtual void move_tasks_from_Sc(std::vector<GMTask *> &src_tasks, GMBuffer &Bwr)
    {
        cout << "H to D: " << src_tasks.size() << endl;
        // for (GMTask *task : src_tasks)
        // {
        //     ui rsz = task->plan.R.size(), psz = task->plan.P.size(), xsz = task->plan.X.size();
        //     ull loc = Bwr.append_host(rsz + psz + xsz);
        //     std::copy(task->plan.R.begin(), task->plan.R.end(), Bwr.vertices + loc);
        //     std::fill(Bwr.labels + loc, Bwr.labels + loc + rsz, R);
        //     loc += rsz;

        //     std::copy(task->plan.P.begin(), task->plan.P.end(), Bwr.vertices + loc);
        //     std::fill(Bwr.labels + loc, Bwr.labels + loc + psz, P);
        //     loc += psz;

        //     std::copy(task->plan.X.begin(), task->plan.X.end(), Bwr.vertices + loc);
        //     std::fill(Bwr.labels + loc, Bwr.labels + loc + xsz, X);
        //     delete task;
        // }
        // src_tasks.clear();
    }
    virtual void move_tasks_to_Sc(vector<GMTask *> &collector, GMBuffer &Brd)
    {
        cout << "D to H" << endl;
        // for (ui i = 0; i < gpu_to_host_transfer_size_g; i++)
        // {
        //     SubgraphOffsets so = Brd.pop_host();
        //     if (so.empty())
        //         break;
        //     ui sglen = so.en - so.st;
        //     GMTask *task = new GMTask();
        //     // approximate lengths
        //     task->plan.R.reserve(sglen / 3);
        //     task->plan.P.reserve(sglen / 2);
        //     task->plan.X.reserve(sglen / 2);
        //     for (auto st = so.st; st < so.en; st++)
        //     {
        //         if (Brd.labels[st] == R)
        //             task->plan.R.push_back(Brd.vertices[st]);
        //         else if (Brd.labels[st] == P)
        //             task->plan.P.push_back(Brd.vertices[st]);
        //         else
        //             task->plan.X.push_back(Brd.vertices[st]);
        //     }
        //     collector.push_back(task);
        // }
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
            if (isOverflow())
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

            unsigned int vt = Bwr.append(1); // allocates a subgraph by atomic operations
            if (LANEID == 0)
            {
                Bwr.vertices[vt] = v; // sources[v];
            }
        }
    }
    __device__ virtual void extend(GMBuffer &Brd, GMBuffer &Bwr, GMBuffer &H, ull *row_ptrs, VertexID *cols)
    {
    StoreStrategy CUR_MODE, NEXT_MODE;
    __shared__ ui partial_subgraphs[WARPS_EACH_BLK][8];

    size_t local_thread_count = 0;

    while (true)
    {
        SubgraphOffsets so = Brd.next();

        if (so.empty())
            break;

        if (so.md == 0)
            CUR_MODE = StoreStrategy::EXPAND;
        else
            CUR_MODE = StoreStrategy::PREFIX;
        
        if (isOverflow())
            break;

        ui id, sglen;
        if (CUR_MODE == StoreStrategy::EXPAND)
        {
            id = so.en - so.st;
            sglen = so.en - so.st;
        }
        else if (CUR_MODE == StoreStrategy::PREFIX)
        {
            id = so.md - so.st + 1;
            sglen = so.md - so.st + 1;
        }
        
        NEXT_MODE = strategy[id + 1];
        ui u = matchOrder[id];

        
        /**
        if (CUR_MODE == StoreStrategy::EXPAND && NEXT_MODE != StoreStrategy::PREFIX)
        {
            if (so.st + LANEID < so.en)
                partial_subgraphs[WARPID][LANEID] = Brd.vertices[so.st + LANEID];
            __syncwarp();
            
            // select the pivot with least # of candidates
            ui bnCount = backNeighborCount[id];
            ui u_prime = backNeighbors[querySize[0] * id];
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

            bool pred;
            ui condCount = condNum[u];
            ui vertex;
            uintE base_i = pu_st;

            do {
                ui len = 0;
                for ( ; base_i < pu_en; base_i += 32)
                {
                    uintE il = base_i + LANEID;
                    pred = il < pu_en;

                    if (pred)
                    {
                        vertex = cols[il];
                        for (ui j = 0; j < bnCount; ++j)
                        {
                            ui u_prime = backNeighbors[querySize[0] * id + j];
                            if (u_prime == parent_u)
                                continue;
                            ui u_prime_M = partial_subgraphs[WARPID][ID2order[u_prime]];
                            uintE u_prime_M_st = row_ptrs[u_prime_M];
                            uintE u_prime_M_en = row_ptrs[u_prime_M + 1];

                            pred = binarySearch(cols, u_prime_M_st, u_prime_M_en, vertex);
                            if (!pred) break;
                        }
        
                        if (pred)
                        {
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
                                if (sglen + 1 == querySize[0])
                                {
                                    local_thread_count += 1;
                                }
                            }
                        }
                    }
                    if (sglen + 1 != querySize[0])
                    {
                        // ui val = pred ? vertex : 0;
                        unsigned int loc = scanIndex(pred) + len;
                        assert(loc < SHM_CAP);
                        if (pred)
                        {
                            shm_temp[WARPID][loc] = vertex;
                        }
                        if (LANEID == 31) // last lane's loc+pred is number of items found in this scan
                            len = loc + pred;
                        len = __shfl_sync(FULL, len, 31);
                        
                        if (len >= SHM_CAP - 32)
                        { 
                            // if (GTHID == 0) printf("* "); 
                            base_i += 32;
                            break;
                        }
                    }
                }

                auto vt = Bwr.append_batch(sglen + 1, len, StoreStrategy::EXPAND);
                if (isOverflow())
                    return;
                for (ui i = LANEID; i < len; i += 32)
                {
                    for (ui j = 0; j < sglen; ++j)
                    {
                        auto k = vt + i * (sglen + 1) + j;
                        Bwr.vertices[k] = partial_subgraphs[WARPID][j];
                    }
                    Bwr.vertices[vt + i * (sglen + 1) + sglen] = shm_temp[WARPID][i]; // add q on the back
                }
            } while (base_i < pu_en);
            continue;
        }
        **/
        
    

        // do pre-intersection here

        if (shareIntersection[id])
        {
            if (so.st + LANEID < so.md)
                partial_subgraphs[WARPID][LANEID] = Brd.vertices[so.st + LANEID];
            __syncwarp();

            ui bnCount = preBackNeighborCount[id];
            ui u_prime = preBackNeighbors[querySize[0] * id];
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

            do {
                len = 0;

                for ( ; base_i < pu_en; base_i += 32)
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
                                if (!pred) break;
                            }
                        }
                    }
                    ui val = pred ? vertex : 0;
                    len = writeToPreIntersection(val, pred, len);

                    if (len >= TEMPSIZE - 32)
                    { 
                        if (GTHID == 0) printf("* ");
                        base_i += 32;
                        break;
                    }
                }
                ui pre_len = len;

                for (ui subgraph_id = so.md; subgraph_id < so.en; ++subgraph_id)
                {

                    if (LANEID == 0)
                    {
                        partial_subgraphs[WARPID][so.md - so.st] = Brd.vertices[subgraph_id];
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
                                    if (!pred) break;
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
                                auto vt = Bwr.append_batch(sglen + 1, min, StoreStrategy::EXPAND);
                                // if (isOverflow())
                                //     return;
                                for (ui i = LANEID; i < min; i += 32)
                                {
                                    for (ui j = 0; j < sglen; ++j)
                                    {
                                        auto k = vt + i * (sglen + 1) + j;
                                        Bwr.vertices[k] = partial_subgraphs[WARPID][j];
                                    }
                                    Bwr.vertices[vt + i * (sglen + 1) + sglen] = tempv[batch_id + i + GLWARPID * TEMPSIZE]; // add q on the back
                                } 
                            }
                        }
                        else if (NEXT_MODE == StoreStrategy::PREFIX)
                        {
                            for (ui batch_id = 0; batch_id < len; batch_id += BATCH_SIZE)
                            {
                                ui min = len - batch_id < BATCH_SIZE ? len - batch_id : BATCH_SIZE;
                                auto vt = Bwr.append_batch(sglen, min, StoreStrategy::PREFIX);
                                // if (isOverflow())
                                //     return;
                                for (ui i = LANEID; i < sglen; i += 32)
                                {
                                    auto k = vt + i;
                                    Bwr.vertices[k] = partial_subgraphs[WARPID][i];
                                }
                                for (ui i = LANEID; i < min; i += 32)
                                    Bwr.vertices[vt + sglen + i] = tempv[batch_id + i + GLWARPID * TEMPSIZE]; // add q on the back 
                            }
                        }
                    }
                }
            } while (base_i < pu_en);
        }
        else
        {
            if (so.st + LANEID < so.en)
                partial_subgraphs[WARPID][LANEID] = Brd.vertices[so.st + LANEID];
            __syncwarp();
            
            // select the pivot with least # of candidates
            ui bnCount = backNeighborCount[id];
            ui u_prime = backNeighbors[querySize[0] * id];
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

            do {
                len = 0;
                for ( ; base_i < pu_en; base_i += 32)
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
                                if (!pred) break;
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
                            if (GTHID == 0) printf("# ");
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
                            auto vt = Bwr.append_batch(sglen + 1, min, StoreStrategy::EXPAND);
                            // if (isOverflow())
                            //     return;
                            for (ui i = LANEID; i < min; i += 32)
                            {
                                for (ui j = 0; j < sglen; ++j)
                                {
                                    auto k = vt + i * (sglen + 1) + j;
                                    Bwr.vertices[k] = partial_subgraphs[WARPID][j];
                                }
                                Bwr.vertices[vt + i * (sglen + 1) + sglen] = tempv[batch_id + i + GLWARPID * TEMPSIZE]; // add q on the back
                            }
                        }
                    }
                    else if (NEXT_MODE == StoreStrategy::PREFIX)
                    {
                        for (ui batch_id = 0; batch_id < len; batch_id += BATCH_SIZE)
                        {
                            ui min = len - batch_id < BATCH_SIZE ? len - batch_id : BATCH_SIZE;
                            auto vt = Bwr.append_batch(sglen, min, StoreStrategy::PREFIX);
                            // if (isOverflow())
                            //     return;
                            for (ui i = LANEID; i < sglen; i += 32)
                            {
                                auto k = vt + i;
                                Bwr.vertices[k] = partial_subgraphs[WARPID][i];
                            }
                            for (ui i = LANEID; i < min; i += 32)
                                Bwr.vertices[vt + sglen + i] = tempv[batch_id + i + GLWARPID * TEMPSIZE]; // add q on the back 
                        }
                    }
                }
            } while (base_i < pu_en);
        }
    }
    atomicAdd(&total_counts[GLWARPID], local_thread_count);
}

};

#endif