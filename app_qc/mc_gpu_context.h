#ifndef MC_GPU_APP
#define MC_GPU_APP

#define TEMPSIZE 200'000
#define QBuff_SIZE 100'000'000

class MCBuffer : public BufferBase
{

public:
    Label *labels;

    static ull sizeOf()
    {
        return BufferBase::sizeOf() + sizeof(Label);
    }
    void allocateMemory(ull sz)
    {
        BufferBase::allocateMemory(sz);
        chkerr(cudaMalloc((void **)&labels, sz * sizeof(Label)));
    }

    void copy(auto &src)
    {
        BufferBase::copy(src);
        labels = src.labels;
    }
    __device__ void copy(auto &src, ull i, ull j)
    {
        BufferBase::copy(src, i, j);
        labels[i] = src.labels[j];
    }
    /**
     * @brief This version is used to allocate memory on host. Call it only for HOST_BUFF_SZ
     *
     */
    void allocateMemory()
    {
        BufferBase::allocateMemory();
        chkerr(cudaMallocManaged((void **)&labels, HOST_BUFF_SZ * sizeof(Label)));
    }
};

class MCGPUContext : public GPUContext<MCBuffer, MCTask>
{

    VertexID *tempv;
    Label *templ;

public:
    ull *total_counts; // can be accessed on GPU
    ui *QBuff;
    ui *qtail;
    ui *qhead;

    ull get_results()
    {

        ull res = 0;
        for (ui i = 0; i < N_WARPS; i++)
            res += total_counts[i];
        return res;
    }

    virtual void initialize()
    {
        chkerr(cudaMalloc((void **)&tempv, TEMPSIZE * N_WARPS * sizeof(VertexID)));
        chkerr(cudaMalloc((void **)&templ, TEMPSIZE * N_WARPS * sizeof(Label)));
        chkerr(cudaMallocManaged((void **)&total_counts, N_WARPS * sizeof(ull)));
        chkerr(cudaMallocManaged((void **)&QBuff, QBuff_SIZE * sizeof(ui)));
        chkerr(cudaMallocManaged((void **)&qtail, sizeof(ui)));
        chkerr(cudaMallocManaged((void **)&qhead, sizeof(ui)));
        
        chkerr(cudaMalloc(&(row_ptrs), sizeof(ull) * (data_graph.GetVertexCount() + 1)));
        chkerr(cudaMalloc(&(cols), sizeof(VertexID) * data_graph.GetEdgeCount()));
        cudaMemcpy(row_ptrs, data_graph.GetRowPtrs(), sizeof(ull) * (data_graph.GetVertexCount() + 1), cudaMemcpyHostToDevice);
        cudaMemcpy(cols, data_graph.GetCols(), sizeof(VertexID) * data_graph.GetEdgeCount(), cudaMemcpyHostToDevice);
        qtail[0] = 0;
        qhead[0] = 0;
        for (ui i = 0; i < N_WARPS; i++)
            total_counts[i] = 0;
    }

    virtual void load_graph(ull *&row_ptrs, VertexID *&cols)
    {
    }

    __device__ bool examineClique(SubgraphOffsets *so)
    {
        auto st = so->st;
        auto en = so->en;
        auto data = Brd.labels;
        for (; st < en; st += 32)
        {
            auto k = st + LANEID; // want to include all lanes.
            bool pred = k < en && (data[k] == P || data[k] == X);
            if (__ballot_sync(FULL, pred))
                return false;
        }
        return true;
    }

    __device__ bool crossed(SubgraphOffsets *so)
    {
        auto st = so->st;
        auto en = so->en;
        auto data = Brd.labels;
        for (; st < en; st += 32)
        {
            auto k = st + LANEID; // want to include all lanes.
            bool pred = k < en && (data[k] == P);
            if (__ballot_sync(FULL, pred))
                return false;
        }
        return true;
    }

    __device__ ui writeToTemp(VertexID v, Label label, bool pred, ui sglen)
    {
        ui loc = scanIndex(pred) + sglen;
        // popc gives inclusive sum scan, subtract pred to make it exclusive
        // add sglen to find exact location in the temp

        assert(loc < TEMPSIZE);
        if (pred)
        {

            this->tempv[loc + GLWARPID * TEMPSIZE] = v;
            this->templ[loc + GLWARPID * TEMPSIZE] = label;
        }
        if (LANEID == 31) // last lane's loc+pred is number of items found in this scan
            sglen = loc + pred;
        sglen = __shfl_sync(FULL, sglen, 31);
        return sglen;
    }

    // this overloaded form doesn't write the labels.
    // this version is called when separating P in PivotSelection
    __device__ ui writeToTemp(VertexID v, bool pred, ui sglen)
    {
        ui loc = scanIndex(pred) + sglen;
        // popc gives inclusive sum scan, subtract pred to make it exclusive
        // add sglen to find exact location in the temp
        assert(loc < TEMPSIZE);
        if (pred)
            this->tempv[loc + GLWARPID * TEMPSIZE] = v;
        if (LANEID == 31) // last lane's loc+pred is number of items found in this scan
            sglen = loc + pred;
        sglen = __shfl_sync(FULL, sglen, 31);
        return sglen;
    }

    __device__ ui getSubgraphTemp(SubgraphOffsets *so, VertexID q)
    {
        // ui laneid = LANEID;
        auto st = so->st;
        auto en = so->en;
        // printf("#%u:%u:%u*", s, st, en);
        ull qst = row_ptrs[q];
        ull qen = row_ptrs[q + 1];
        VertexID v;
        ui sglen = 0;
        Label label;
        bool pred;

        // ### Binary search, it'll also need a scan to get the locations.
        for (; st < en; st += 32)
        {
            auto i = st + LANEID;
            pred = false; // if i>=en, then pred will be false...
            if (i < en)
            {
                v = Brd.vertices[i];
                label = Brd.labels[i];
                // no need to intersect R nodes
                pred = (label == R) || binarySearch(cols, qst, qen, v);
            }
            __syncwarp();
            sglen = writeToTemp(v, label, pred, sglen); // appply sum scan and store in temp...
            // sglen is passed by reference to this function, and it gets the length of subgraph
        }
        return sglen;
    }
    __device__ void generateSubgraphDoubleIntersect(SubgraphOffsets *so, VertexID q, MCBuffer& buff)
    {
        auto st = so->st;
        auto en = so->en;
        // printf("#%u:%u:%u*", s, st, en);
        ull qst = row_ptrs[q];
        ull qen = row_ptrs[q + 1];
        VertexID v;
        ui sglen = 0;
        Label label;
        bool pred;

        // Perform intersection to find length of subgraph
        for (; st < en; st += 32)
        {
            auto i = st + LANEID;
            pred = false; // if i>=en, then pred will be false...
            if (i < en)
            {
                v = Brd.vertices[i];
                label = Brd.labels[i];
                // no need to intersect R nodes
                pred = (label == R) || binarySearch(cols, qst, qen, v);
            }
            sglen += __popc(__ballot_sync(FULL, pred)); // appply sum scan and store in temp...
            // sglen is passed by reference to this function, and it gets the length of subgraph
        }

        ull vt = buff.append(sglen + 1);
        if (LANEID == 0)
        {
            buff.vertices[vt] = q;
            buff.labels[vt] = R;
        }
        vt++;

        // Perform the intersection again to find vertices for new subgraph
        for (st = so->st, en = so->en; st < en; st += 32)
        {
            auto i = st + LANEID;
            pred = false;
            if (i < en)
            {
                v = Brd.vertices[i];
                label = Brd.labels[i];
                // no need to intersect R nodes
                pred = label == R || binarySearch(cols, qst, qen, v);
            }

            ull loc = scanIndex(pred) + vt;
            if (pred)
            {
                buff.vertices[loc] = v;
                buff.labels[loc] = label;
                if (label == Q)
                    buff.labels[loc] = v < q ? X : P;
            }

            if (LANEID == 31)
                vt = loc + pred;
            vt = __shfl_sync(FULL, vt, 31);
        }
    }

    // this version is called when subgraphs are spawned from Q nodes
    __device__ void
    generateSubgraphs(SubgraphOffsets *so, unsigned int q, MCBuffer& buff)
    {
        // let's find expected subgraph length...
        ui sglen = min(so->en - so->st, row_ptrs[q + 1] - row_ptrs[q]);

        // this subgraph might not fit into temp area, so go for double intersection option
        if (sglen > TEMPSIZE)
        {
            generateSubgraphDoubleIntersect(so, q, buff);
            return;
        }

        // else the subgraph can be intersected just once and put it to temp for later reading
        sglen = getSubgraphTemp(so, q);
        if (sglen == 0)
            return; // q doesn't have graph to spawn.
        // sglen = |N(q)∩(XUPUR)|
        // adding 1 in sglen, as q itself appears in subgraph as R
        assert(sglen + 1 < TEMPSIZE);
        // allocates a subgraph by atomic operations, and puts q in subgraph as well
        auto vt = buff.append(sglen + 1);

        if (LANEID == 0)
        {
            buff.vertices[vt] = q;
            buff.labels[vt] = R;
        }
        vt++; // as one element is written i.e. q
        VertexID *tempv = this->tempv + GLWARPID * TEMPSIZE;
        Label *templ = this->templ + GLWARPID * TEMPSIZE;

        // subgraph is already stored in temp. q is already written to subgraph
        for (ui i = LANEID; i < sglen; i += 32)
        {
            auto k = vt + i;
            ui v = tempv[i];
            Label label = templ[i];
            buff.vertices[k] = v;
            buff.labels[k] = label;
            if (label == Q)
                buff.labels[k] = v < q ? X : P;
        }
    }

    __device__ VertexID selectPivot(SubgraphOffsets *so)
    {
        auto st = so->st;
        auto en = so->en;
        VertexID max = 0, pivot = 0;
        bool pred;
        ui plen = 0;

        // Let's write P set to temp location
        for (auto i = st; i < en; i += 32)
        {
            auto il = i + LANEID;
            pred = (il < en && Brd.labels[il] == P);          // exploiting short-circuit of &&
            plen = writeToTemp(Brd.vertices[il], pred, plen); // the function returns update value of plen
        }
        for (auto j = st; j < en; j++)
        {
            // entire warp is processing one element in this loop, hence laneid is not added...
            // it's not a divergence, entire warp will continue as result of below condition
            if (Brd.labels[j] == R)
                continue;                 // pivot is selected from P U X
            VertexID v = Brd.vertices[j]; // v ∈ (P U X)
            // (st1, en1) are N(v)
            ull st1 = row_ptrs[v];
            ull en1 = row_ptrs[v + 1];
            ui nmatched = 0;
            for (ui k = 0; k < plen; k += 32)
            {
                ui kl = k + LANEID; // need to run all lanes, so that ballot function works well
                pred = kl < plen && binarySearch(cols, st1, en1, tempv[kl + GLWARPID * TEMPSIZE]);
                nmatched += __popc(__ballot_sync(FULL, pred));
            }
            if (nmatched >= max) // using == just to take care of case when nmatched is zero for all v
            {
                max = nmatched;
                pivot = v;
            }
        }
        return pivot;
    }

    __device__ virtual void generateInitialTasks(VertexID *sources, ull *sources_num, ull *v_proc, MCBuffer &Bwr, ull *row_ptrs, VertexID *cols)
    {
        while (true)
        {
            // if (isLevelFilled() or v_proc[0]>ETA)
            //     return;
            unsigned int vp;
            if (LANEID == 0)
                vp = atomicAdd(v_proc, 1);
            vp = __shfl_sync(FULL, vp, 0);
            if (vp >= sources_num[0])
                return;

            auto v = sources[vp];
            auto st = row_ptrs[v];
            auto en = row_ptrs[v + 1];
            ui sglen = en - st;
            if (sglen == 0)
                continue; // there was no neighbor for this vertex...
            // adding 1 as vertices in new graph are number of neighbors + v itself
            auto vt = Bwr.append(sglen + 1); // allocates a subgraph by atomic operations, and puts v as well
            if (LANEID == 0)
            {
                Bwr.vertices[vt] = v;
                Bwr.labels[vt] = R;
                // printf("%u:%u ", v, sglen);
            }
            vt++; // as one element is written i.e. v
            for (ull j = st + LANEID, k = vt + LANEID; j < en; j += 32, k += 32)
            {
                auto u = cols[j];
                Bwr.vertices[k] = u;
                Bwr.labels[k] = (u < v) ? X : P;
            }
        }
    }
    __device__ bool isLevelFilledQ()
    {
        return (qtail[0] > 3 * eta);
    }

public:
    __device__ virtual void process(MCBuffer &Brd, ull *row_ptrs, VertexID *cols)
    {
        while (true)
        {
#ifdef SRC
            if (isLevelFilled()) // source only
#else
            if (isLevelFilled() or isLevelFilledQ()) // dst or both
#endif
                break;
            SubgraphOffsets so = Brd.next();
            if (so.empty())
                break;
            if (examineClique(&so))
            {
                if (LANEID == 0)
                    atomicAdd(total_counts + GLWARPID, 1);
            }
            else
            {
                unsigned int pivot = selectPivot(&so);
                markQ(&so, pivot);
            }
        }
    }
    virtual void init_level()
    {
        qhead[0] = 0;
        qtail[0] = 0;
    }
    __device__ virtual void extend(MCBuffer &Brd, MCBuffer &Bwr, MCBuffer &H, ull *row_ptrs, VertexID *cols)

    {
        while (true)
        {
            ui qh;
            if (LANEID == 0)
                qh = atomicAdd(qhead, 3);
            qh = __shfl_sync(FULL, qh, 0);
            if (qh >= qtail[0])
                return;
            ui u = QBuff[qh];
            // ui u = Brd.vertices[QBuff[qh]];
            SubgraphOffsets so{QBuff[qh + 1], QBuff[qh + 2]};
            if (isOverflow())
                generateSubgraphs(&so, u, H);
            else
                generateSubgraphs(&so, u, Bwr);
        }
    }

    __device__ void markQ(SubgraphOffsets *so, VertexID pivot)
    {
        auto st = so->st;
        auto en = so->en;
        // if(!LANEID&&load_per_warp[0]<5000) printf("%d ", pivot);
        ull pst = row_ptrs[pivot];
        ull pen = row_ptrs[pivot + 1];

        // subgraph stored in (st, en)
        // N(pivot) are in (pst, pen)
        // find Q=P-N(pivot)
        // for every q ∈ Q, generate a subgraph
        for (; st < en; st += 32)
        {
            ui i = st + LANEID;
            bool pred = i < en && Brd.labels[i] == P && !binarySearch(cols, pst, pen, Brd.vertices[i]);

            if(pred){
                ui loc=atomicAdd(qtail, 3);
                assert(loc + 3 < QBuff_SIZE);
                Brd.labels[i] = Q;
                QBuff[loc] = Brd.vertices[i];
                QBuff[loc + 1] = so->st;
                QBuff[loc + 2] = so->en;
            }
            // ui w_loc = scanIndex(pred);
            // ui g_loc;

            // if (LANEID == 31) // last lane's loc+pred is number of items found in this scan
            // {
            //     ui num_q = w_loc + pred;
            //     g_loc = atomicAdd(qtail, num_q * 3);
            //     // __threadfence();
            // }
            // g_loc = __shfl_sync(FULL, g_loc, 31) + w_loc * 3;

            // if (pred)
            // {
            //     // v belongs to Q, so generate subgraph for it
            //     // simply change their labels to Q, afterwards generate a subgraph for each such node
            //     Brd.labels[i] = Q;
            //     assert(g_loc + 3 < QBuff_SIZE);
            //     QBuff[g_loc] = Brd.vertices[i];
            //     QBuff[g_loc + 1] = so->st;
            //     QBuff[g_loc + 2] = so->en;
            // }
        }
    }

    virtual void move_tasks_from_Sc(std::vector<MCTask *> &src_tasks, MCBuffer &H)
    {
        cout << "H to D: " << src_tasks.size() << endl;
        for (MCTask *task : src_tasks)
        {
            ui rsz = task->context.R.size(), psz = task->context.P.size(), xsz = task->context.X.size();
            ull loc = H.append_host(rsz + psz + xsz);
            std::copy(task->context.R.begin(), task->context.R.end(), H.vertices + loc);
            std::fill(H.labels + loc, H.labels + loc + rsz, R);
            loc += rsz;

            std::copy(task->context.P.begin(), task->context.P.end(), H.vertices + loc);
            std::fill(H.labels + loc, H.labels + loc + psz, P);
            loc += psz;

            std::copy(task->context.X.begin(), task->context.X.end(), H.vertices + loc);
            std::fill(H.labels + loc, H.labels + loc + xsz, X);
            delete task;
        }
        src_tasks.clear();
    }
    virtual void move_tasks_to_Sc(vector<MCTask *> &collector, MCBuffer &H)
    {
        cout << "D to H" << endl;
        for (ui i = 0; i < gpu_to_host_transfer_size_g; i++)
        {
            SubgraphOffsets so = H.pop_host();
            if (so.empty())
                break;
            ui sglen = so.en - so.st;
            MCTask *task = new MCTask();
            // approximate lengths
            task->context.R.reserve(sglen / 3);
            task->context.P.reserve(sglen / 2);
            task->context.X.reserve(sglen / 2);
            for (auto st = so.st; st < so.en; st++)
            {
                if (H.labels[st] == R)
                    task->context.R.push_back(H.vertices[st]);
                else if (H.labels[st] == P)
                    task->context.P.push_back(H.vertices[st]);
                else
                    task->context.X.push_back(H.vertices[st]);
            }
            collector.push_back(task);
        }
    }
};
#endif