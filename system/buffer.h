#ifndef SYSTEM_BUFFER_H
#define SYSTEM_BUFFER_H

// Prefix
/**************************************
 * ---------------------------
 * | st md en | st' md' en'|
 * ---------------------------
 *   st    md          en
 * --|-----|-----------|-----------
 * | x x x y y y y y y a a a b b b
 *  -------|-----------|-----------
 *  prefix | extended  |
 *         |candidates |
 **************************************/

class SubgraphOffsets
{
public:
    ull st;
    ull md;
    ull en;
    DEVHOST SubgraphOffsets(ull s, ull m, ull e) : st(s), md(m), en(e) {}
    DEVHOST SubgraphOffsets(ull s, ull e) : st(s), md(0), en(e) {}
    DEVHOST SubgraphOffsets() : st(0), md(0), en(0) {}
    DEVHOST bool empty()
    {
        return st == 0 && md == 0 && en ==0;
    }
};

class BufferBase
{
public:
    ull *offsets;
    VertexID *vertices;

    // should be transparent to the users
    ull *otail;
    ull *vtail;
    ull *ohead;
    ull *capacity;
    ui *n_tasks_proc;


    static ull sizeOf()
    {
        return (sizeof(VertexID) + sizeof(ull));
    }
    /**
     * @brief This version is used to allocate memory on host. Call it only for HOST_BUFF_SZ
     *
     */
    void allocateMemory()
    {
        chkerr(cudaMallocManaged((void **)&offsets, sizeof(ull) * HOST_OFFSET_SZ));
        chkerr(cudaMallocManaged((void **)&vertices, sizeof(VertexID) * HOST_BUFF_SZ));
        // todo HOST_BUFF_SZ should be small in final release
        allocatePtrs();
        capacity[0] = HOST_BUFF_SZ;
        n_tasks_proc[0] = 0;
        std::cout << "Host allocated Buffer: " << capacity[0] << std::endl;
    }

    void allocateMemory(ull sz)
    {
        chkerr(cudaMalloc((void **)&offsets, sz * sizeof(ull)));
        chkerr(cudaMalloc((void **)&vertices, sz * sizeof(VertexID)));

        allocatePtrs();
        capacity[0] = sz;
        std::cout << "Device allocated Buffer: " << capacity[0] << std::endl;
    }


    ull append_host(ull sglen, ull md = 0)
    {
        ull ot = otail[0], vt = vtail[0];
        otail[0] += 3;
        vtail[0] += sglen;
        offsets[ot] = vt;
        offsets[ot + 1] = md;
        offsets[ot + 2] = vtail[0];
        return vt;
    }

    __device__ ull append(ull sglen, ull md = 0)
    {
        ull vt;
        if (LANEID == 0)
        {
            ull ot = atomicAdd(otail, 3);
            vt = atomicAdd(vtail, sglen);
            atomicAdd(n_tasks_proc, 1);
            __threadfence();
            // if it's a host buffer
            if (capacity[0] == HOST_BUFF_SZ)
            {
                assert(ot + 3 < HOST_OFFSET_SZ && vt + sglen < capacity[0]);
                // printf("Host level_filled\n");
            }
            else
            {
                // this is device buffer
                assert(vt + sglen < capacity[0]);
                assert(ot + 3 < capacity[0]);
            }
            offsets[ot] = vt;
            offsets[ot + 1] = md;
            offsets[ot + 2] = vt + sglen;
        }
        vt = __shfl_sync(FULL, vt, 0);
        return vt;
    }
    __device__ ull append(SubgraphOffsets &so)
    {
        return append(so.en - so.st, so.md == 0 ? 0 : so.md - so.st);
    }
    __device__ SubgraphOffsets next()
    {
        ull s;
        if (LANEID == 0)
        {
            atomicAdd(n_tasks_proc, 1);
            __threadfence();
            s = atomicAdd(ohead, 3);
        }
        s = __shfl_sync(FULL, s, 0);
        if (s < otail[0])
            return {offsets[s], offsets[s + 1], offsets[s + 2]}; // md is invalid
        else
            return {0, 0, 0};
    }

    __device__ SubgraphOffsets pop()
    {
        ull s;
        if (LANEID == 0)
        {
            atomicAdd(n_tasks_proc, 1);
            s = atomicDecrementNonNegative(otail, 3);
        }
        s = __shfl_sync(FULL, s, 0);
        if (s != 0)
            return {offsets[s - 3], offsets[s - 2], offsets[s - 1]};
        else
            return {0, 0, 0};
    }

    SubgraphOffsets pop_host()
    {
        // removing from the tail
        if (empty())
            return {0, 0, 0};

        otail[0] -= 3;
        ull s = otail[0];

        return {offsets[s], offsets[s + 1], offsets[s + 2]}; // md is invalid
    }

    __device__ SubgraphOffsets next(StoreStrategy mode)
    {
        // A block will try to get a batch of subgraphs
        ull s;
        if (mode == StoreStrategy::EXPAND)
        {
            if (LANEID == 0)
            {
                s = atomicAdd(ohead, 3);
            }
            s = __shfl_sync(FULL, s, 0);
            if (s < otail[0])
                return {offsets[s], 0, offsets[s + 2]}; // md is invalid
            else
                return {0, 0, 0};
        }
        else if (mode == StoreStrategy::PREFIX)
        {
            if (LANEID == 0)
            {
                s = atomicAdd(ohead, 3);
            }
            s = __shfl_sync(FULL, s, 0);
            if (s < otail[0])
                return {offsets[s], offsets[s + 1], offsets[s + 2]};
            else
                return {0, 0, 0};
        }
        else
        {
            assert(false);
            return {0, 0, 0};
        }
    }

    bool empty()
    {
        return (ohead[0] >= otail[0]);
    }

    void clear()
    {
        vtail[0] = 0;
        otail[0] = 0;
        ohead[0] = 0;
    }

    DEVHOST bool isOverflow(){
    //    return vtail[0] >= 0.05 * capacity[0] or otail[0] >= 0.05 * capacity[0];
       return vtail[0] >= 0.9 * capacity[0] or otail[0] >= 0.9 * capacity[0];
    }

    void allocatePtrs()
    {
        chkerr(cudaMallocManaged((void **)&otail, sizeof(ull)));
        chkerr(cudaMallocManaged((void **)&vtail, sizeof(ull)));
        chkerr(cudaMallocManaged((void **)&ohead, sizeof(ull)));
        chkerr(cudaMallocManaged((void **)&capacity, sizeof(ull)));
        chkerr(cudaMallocManaged((void **)&n_tasks_proc, sizeof(ui)));
        otail[0] = 0;
        vtail[0] = 0;
        ohead[0] = 0;
    }
    __device__ void copy(auto &src, ull i, ull j)
    {
        vertices[i] = src.vertices[j];
    }

    void print(string msg)
    {
        cout << msg << ohead[0] << "-" << otail[0] << "-" << vtail[0] << " "<<endl;
    }

    __device__ ull append_batch(ull sglen, ui num, StoreStrategy mode)
    {
        ull vt, ot;
        if (mode == StoreStrategy::EXPAND)
        {
            if (LANEID == 0)
            {
                atomicAdd(n_tasks_proc, num);
                ot = atomicAdd(otail, 3 * num);
                vt = atomicAdd(vtail, sglen * num);
            }
            vt = __shfl_sync(FULL, vt, 0);
            ot = __shfl_sync(FULL, ot, 0);
            for (ui i = LANEID; i < num; i += 32)
            {
                offsets[ot + i * 3] = vt + sglen * i;
                offsets[ot + i * 3 + 1] = 0;
                offsets[ot + i * 3 + 2] = vt + sglen * (i + 1);
            }
            return vt;
        }
        else if (mode == StoreStrategy::PREFIX)
        {
            if (LANEID == 0)
            {
                ot = atomicAdd(otail, 3);
                vt = atomicAdd(vtail, sglen + num); 
                atomicAdd(n_tasks_proc, num);

                offsets[ot] = vt;
                offsets[ot + 1] = vt + sglen;
                offsets[ot + 2] = vt + sglen + num;
            }
            vt = __shfl_sync(FULL, vt, 0);
            return vt;
        }
        else
        {
            assert(false);
            return 0;
        }
    }

};

#endif