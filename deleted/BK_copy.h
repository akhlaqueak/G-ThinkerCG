#ifndef GPU_BK_H
#define GPU_BK_H

#include "system/appbase.h"
#include "system/buffer.h"
#include "common/meta.h"
#include "system/util.h"
#define QBuff_SIZE 100'000'000
template<class IndexType>
class BKBuffer : public BufferBase<IndexType>
{
public:
    Label *labels;

    static size_t sizeOf()
    {
        return BufferBase<IndexType>::sizeOf() + sizeof(Label);
    }
    void allocateMemory(size_t sz)
    {
        BufferBase<IndexType>::allocateMemory(sz);
        chkerr(cudaMalloc((void **)&labels, sz * sizeof(Label)));
    }
    void promoteLevel(IndexType vt){
        BufferBase<IndexType>::promoteLevel(vt);
        labels+=vt;
    }
    void demoteLevel(IndexType vt){
        BufferBase<IndexType>::demoteLevel(vt);
        labels-=vt;
    }
    void copy(auto& src){
        BufferBase<IndexType>::copy(src);
        labels = src.labels;
    }
    __device__ void copy(auto& src, ull i, ull j){
        // assert(i<buffsize[0]);
        // assert(j<buffsize[0]);
        BufferBase<IndexType>::copy(src, i, j);
        labels[i]=src.labels[j];
    }
    /**
     * @brief This version is used to allocate memory on host. Call it only for HOST_BUFF_SZ
     *
     */
    void allocateMemory()
    {
        BufferBase<IndexType>::allocateMemory();
        chkerr(cudaMallocManaged((void **)&labels, HOST_BUFF_SZ * sizeof(Label)));
    }
};

class MC_GPU_App : public AppBase<BKBuffer>
{
    uintV *tempv;
    Label *templ;

public:
    ull *iterCliques;
    ull *totalCliques;
    ui *q_thresh;
    ull *QBuff;
    ui *qtail;
    ui *qhead;
    

    void initLevel(){
        qhead[0] = 0;
        qtail[0] = 0;
    }
    
    uintV firstRoundIterNumHost(){
        // do nothing
        return ctx->sources_num[0];
    }
    void setQThresh(ui q){
        chkerr(cudaMallocManaged((void **)&q_thresh, sizeof(int)));
        q_thresh[0] = q;
    }
    void allocateMemory()
    {
        chkerr(cudaMalloc((void **)&tempv, TEMPSIZE * N_WARPS * sizeof(uintV)));
        chkerr(cudaMalloc((void **)&templ, TEMPSIZE * N_WARPS * sizeof(Label)));
        chkerr(cudaMallocManaged((void **)&QBuff, QBuff_SIZE * sizeof(ull)));

        chkerr(cudaMallocManaged((void **)&iterCliques, sizeof(unsigned long long int)));
        chkerr(cudaMallocManaged((void **)&totalCliques, sizeof(unsigned long long int)));
        chkerr(cudaMallocManaged((void **)&qtail,  sizeof(ui)));
        chkerr(cudaMallocManaged((void **)&qhead,  sizeof(ui)));
        totalCliques[0] = 0;
        iterCliques[0] = 0;
        qtail[0] = 0;
        qhead[0] = 0;
    }

    void iterationFailed()
    {
        std::cout<<"Iteration failed!!"<< std::endl;
        iterCliques[0] = 0;
    }
    void iterationSuccess()
    {
        total_counts_[0] += iterCliques[0];
        iterCliques[0] = 0;
        // std::cout << " cliques " << total_counts_[0] << std::endl;
    }

    void completion()
    {
        std::cout << "Total No. of Cliques: " << totalCliques[0] << std::endl;
        total_counts_[0] = totalCliques[0];
        totalCliques[0] = 0;
    }

    __device__ bool examineClique(SubgraphOffsets *so)
    {
        auto st = so->st;
        auto en = so->en;
        auto data = sg->rdBuff.labels;
        for (; st < en; st += 32)
        {
            auto k = st + LANEID; // want to include all lanes.
            bool pred = k < en && (data[k] == P || data[k] == X);
            if (__ballot_sync(FULL, pred))
                return false;
        }
        return true;
    }
    void move_tasks_to_gpu(std::queue<MCTask *> &tasks)
    {
        auto buff = sg->hostWrBuff;
        cout << "H to D: " << tasks.size() << endl;
        while (not tasks.empty())
        {
            MCTask *task = tasks.front();
            tasks.pop();
            ui rsz = task->context.R.size(), psz = task->context.P.size(), xsz = task->context.X.size();
            ull loc = buff.append_host(rsz + psz + xsz);
            std::copy(task->context.R.begin(), task->context.R.end(), buff.vertices + loc);
            std::fill(buff.labels + loc, buff.labels + loc + rsz, R);
            loc += rsz;

            std::copy(task->context.P.begin(), task->context.P.end(), buff.vertices + loc);
            std::fill(buff.labels + loc, buff.labels + loc + psz, P);
            loc += psz;

            std::copy(task->context.X.begin(), task->context.X.end(), buff.vertices + loc);
            std::fill(buff.labels + loc, buff.labels + loc + xsz, X);
            delete task;
        }
    }
    std::queue<MCTask *> move_tasks_to_cpu()
    {
        // make return type as reference
        queue<MCTask *> tasks;
        auto buff = sg->hostRdBuff; // why not RdBuff?
        cout<<"D to H"<<endl;
        for (ui i = 0; i < gpu_to_host_transfer_size_g; i++)
        {
            SubgraphOffsets so = buff.pop_host();
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
                if (buff.labels[st] == R)
                    task->context.R.push_back(buff.vertices[st]);
                else if (buff.labels[st] == P)
                    task->context.P.push_back(buff.vertices[st]);
                else
                    task->context.X.push_back(buff.vertices[st]);
            }
            tasks.push(task);
        }
        return tasks;
    }
    __device__ bool crossed(SubgraphOffsets *so)
    {
        auto st = so->st;
        auto en = so->en;
        auto data = sg->rdBuff.labels;
        for (; st < en; st += 32)
        {
            auto k = st + LANEID; // want to include all lanes.
            bool pred = k < en && (data[k] == P);
            if (__ballot_sync(FULL, pred))
                return false;
        }
        return true;
    }

    __device__ ui writeToTemp(uintV v, Label label, bool pred, ui sglen)
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
    __device__ ui writeToTemp(uintV v, bool pred, ui sglen)
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

    __device__ ui getSubgraphTemp(SubgraphOffsets *so, auto qloc)
    {
        // ui laneid = LANEID;
        auto st = so->st;
        auto en = so->en;
        ui q=sg->rdBuff.vertices[qloc];
        // printf("#%u:%u:%u*", s, st, en);
        uintE qst = ctx->d_row_ptrs[q];
        uintE qen = ctx->d_row_ptrs[q+1];
        uintV v;
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
                v = sg->rdBuff.vertices[i];
                label = sg->rdBuff.labels[i];
                if(label==Q) label= v<q||i<qloc?X:P;
                // no need to intersect R nodes
                pred = (label == R) || binarySearch(ctx->d_cols, qst, qen, v);
            }
            __syncwarp();
            sglen = writeToTemp(v, label, pred, sglen); // appply sum scan and store in temp...
            // sglen is passed by reference to this function, and it gets the length of subgraph
        }
        return sglen;
    }
    __device__ void generateSubgraphDoubleIntersect(SubgraphOffsets *so, auto qloc, auto buff)
    {
        ui q = sg->rdBuff.vertices[qloc];
        auto st = so->st;
        auto en = so->en;
        // printf("#%u:%u:%u*", s, st, en);
        uintE qst = ctx->d_row_ptrs[q];
        uintE qen = ctx->d_row_ptrs[q + 1];
        uintV v;
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
                v = sg->rdBuff.vertices[i];
                label = sg->rdBuff.labels[i];
                // no need to intersect R nodes
                pred = (label == R) || binarySearch(ctx->d_cols, qst, qen, v);
            }
            sglen += __popc(__ballot_sync(FULL, pred)); // appply sum scan and store in temp...
            // sglen is passed by reference to this function, and it gets the length of subgraph
        }

        uintE vt = buff->append(sglen + 1);
        if (LANEID == 0)
        {
            buff->vertices[vt] = q;
            buff->labels[vt] = R;
        }
        vt++;

        // Perform the intersection again to find vertices for new subgraph
        for (st = so->st, en = so->en; st < en; st += 32)
        {
            auto i = st + LANEID;
            pred = false;
            if (i < en)
            {
                v = sg->rdBuff.vertices[i];
                label = sg->rdBuff.labels[i];
                // no need to intersect R nodes
                pred = label == R || binarySearch(ctx->d_cols, qst, qen, v);
            }

            uintE loc = scanIndex(pred) + vt;
            if (pred)
            {
                buff->vertices[loc] = v;
                buff->labels[loc] = label;
                if (label == Q)
                    buff->labels[loc] = v < q||i<qloc ? X : P;
            }

            if(LANEID==31)
                vt = loc + pred;
            vt = __shfl_sync(FULL, vt, 31);
        }
    }

    // this version is called when subgraphs are spawned from Q nodes
    __device__ void
    generateSubgraphs(SubgraphOffsets *so, auto qloc, auto buff)
    {
        ui q=sg->rdBuff.vertices[qloc];
        // let's find expected subgraph length...
        ui sglen = min((ui) (so->en - so->st), (ui) (ctx->d_row_ptrs[q + 1] - ctx->d_row_ptrs[q]));

        // this subgraph might not fit into temp area, so go for double intersection option
        if (sglen > TEMPSIZE)
        {
            generateSubgraphDoubleIntersect(so, qloc, buff);
            return;
        }

        // else the subgraph can be intersected just once and put it to temp for later reading
        sglen = getSubgraphTemp(so, qloc);
        if (sglen == 0)
            return; // q doesn't have graph to spawn.
        // sglen = |N(q)∩(XUPUR)|
        // adding 1 in sglen, as q itself appears in subgraph as R
        assert(sglen + 1 < TEMPSIZE);
        // allocates a subgraph by atomic operations, and puts q in subgraph as well
        auto vt = buff->append(sglen + 1);

        if (LANEID == 0)
        {
            buff->vertices[vt] = q;
            buff->labels[vt] = R;
        }
        vt++; // as one element is written i.e. q
        uintV *tempv = this->tempv + GLWARPID * TEMPSIZE;
        Label *templ = this->templ + GLWARPID * TEMPSIZE;

        // subgraph is already stored in temp. q is already written to subgraph
        for (ui i = LANEID; i < sglen; i += 32)
        {
            auto k = vt + i;
            ui v = tempv[i];
            Label label = templ[i];
            buff->vertices[k] = v;
            buff->labels[k] = label;
            // if (label == Q)
            //     buff->labels[k] = v < q||i<qloc ? X : P;
        }
    }

    __device__ bool selectPivot(SubgraphOffsets *so, uintV& pivot)
    {
        auto st = so->st;
        auto en = so->en;
        uintV max = 0;
        bool pred;
        ui plen = 0;

        // Let's write P set to temp location
        for (auto i = st; i < en; i += 32)
        {
            auto il = i + LANEID;
            pred = (il < en && sg->rdBuff.labels[il] == P);            // exploiting short-circuit of &&
            __syncwarp();
            if(plen>TEMPSIZE-50) printf("%llu,%llu ", st, en);
            plen = writeToTemp(sg->rdBuff.vertices[il], pred, plen); // the function returns update value of plen
        }
        if(plen==0) return false;
        for (auto j = st; j < en; j++)
        {
            // entire warp is processing one element in this loop, hence laneid is not added...
            // it's not a divergence, entire warp will continue as result of below condition
            if (sg->rdBuff.labels[j] == R)
                continue;                            // pivot is selected from P U X
            uintV v = sg->rdBuff.vertices[j]; // v ∈ (P U X)
            // (st1, en1) are N(v)
            uintE st1 = ctx->d_row_ptrs[v];
            uintE en1 = ctx->d_row_ptrs[v + 1];
            ui nmatched = 0;
            for (ui k = 0; k < plen; k += 32)
            {
                ui kl = k + LANEID; // need to run all lanes, so that ballot function works well
                pred = kl < plen && binarySearch(ctx->d_cols, st1, en1, tempv[kl + GLWARPID * TEMPSIZE]);
                nmatched += __popc(__ballot_sync(FULL, pred));
            }
            if (nmatched >= max) // using == just to take care of case when nmatched is zero for all v
            {
                max = nmatched;
                pivot = v;
            }
        }
        return true;
    }

    __device__ void generateSubgraphs()
    {
        while(true)
        {
            if(sg->isOverflow()) return;
            unsigned int vp;
            if(LANEID==0){
                vp = atomicAdd(vProcessed, 1);
            }
            vp = __shfl_sync(FULL, vp, 0);
            if (vp >= ctx->sources_num[0])
                return;
            // v = dp->peel_sequence[vp];

            auto v = ctx->sources[vp];
            auto st = ctx->d_row_ptrs[v];
            auto en = ctx->d_row_ptrs[v + 1];
            ui sglen = en - st;
            if (sglen == 0)
                continue; // there was no neighbor for this vertex...
            // adding 1 as vertices in new graph are number of neighbors + v itself
            auto vt = sg->wrBuff.append(sglen + 1); // allocates a subgraph by atomic operations, and puts v as well
            if (LANEID == 0)
            {
                sg->wrBuff.vertices[vt] = v;
                sg->wrBuff.labels[vt] = R;
                // printf("%u:%u ", v, sglen);
            }
            vt++; // as one element is written i.e. v
            for (ull j = st + LANEID, k = vt + LANEID; j < en; j += 32, k += 32)
            {
                auto u = ctx->d_cols[j];
                sg->wrBuff.vertices[k] = u;
                sg->wrBuff.labels[k] = (u < v) ? X : P;
            }
        }
    }

public:

    __device__ void processSubgraphs()
    {
        while (true)
        {
            __syncwarp();
            if(qtail[0]>3*load_per_warp[0]*N_WARPS) break;
            if (sg->isOverflow())
                break;
            SubgraphOffsets so = sg->next();
            
            if (sg->isEnd(so))
                break;
         
            if (examineClique(&so))
            {
                if(LANEID==0) atomicAdd(this->iterCliques, 1);
            }
            else 
            // if (!crossed(&so)) // optional, saves the effort of selectPivot for crossed subgraphs, however adds cost for other
            {
                // if (sg->isOverflowToHost())
                // {
                //     dumpToHost(&so);
                //     break;
                // }
                unsigned int pivot;  
                if(selectPivot(&so, pivot))
                    markQ(&so, pivot);
            }
        }

    }

    __device__ void expand(){
        while(true){
            ui qh;
            if(LANEID==0) qh= atomicAdd(qhead, 3);
            qh = __shfl_sync(FULL, qh, 0);
            if(qh>=qtail[0]) return;
            SubgraphOffsets so{QBuff[qh+1], QBuff[qh+2]};
            if(sg->isOverflowToHost())
                generateSubgraphs(&so, QBuff[qh], &(sg->hostWrBuff));
            else
                generateSubgraphs(&so, QBuff[qh], &(sg->wrBuff));
        }
    }

    __device__ void markQ(SubgraphOffsets *so, uintV pivot)
    {
        auto st = so->st;
        auto en = so->en;
        // if(!LANEID&&load_per_warp[0]<5000) printf("%d ", pivot);
        uintE pst = ctx->d_row_ptrs[pivot];
        uintE pen = ctx->d_row_ptrs[pivot + 1];

        // subgraph stored in (st, en)
        // N(pivot) are in (pst, pen)
        // find Q=P-N(pivot)
        // for every q ∈ Q, generate a subgraph
        for (auto i=st+LANEID; i < en; i += 32)
        {
            if (i<en && sg->rdBuff.labels[i] == P && !binarySearch(ctx->d_cols, pst, pen, sg->rdBuff.vertices[i]))
            {
                // v belongs to Q, so generate subgraph for it
                // simply change their labels to Q, afterwards generate a subgraph for each such node
                sg->rdBuff.labels[i] = Q;
                
                ui qt = atomicAdd(qtail, 3);
                assert(qt+3<QBuff_SIZE);
                QBuff[qt] = i;
                QBuff[qt+1] = st;
                QBuff[qt+2] = en;
            }
        // __syncwarp();
        }
        
        // __syncwarp();
        // for (auto i = st; i < en; i++)
        //     if (sg->rdBuff.labels[i] == Q)
        //         generateSubgraphs(so, i, &(sg->wrBuff));
    }
};
#endif