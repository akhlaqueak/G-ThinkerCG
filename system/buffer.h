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
        return st == 0 && md == 0 && en == 0;
    }
};

class BufferBase
{
public:
    ull *offsets = nullptr;
    VertexID *vertices = nullptr;

    // should be transparent to the users
    ull *otail = nullptr;
    ull *vtail = nullptr;
    ull *ohead = nullptr;
    ull *capacity = nullptr;
    ui *n_tasks_proc = nullptr;
    bool second_buffer = false; // the second buffer in ping-pong mode.
    volatile bool *overflow = nullptr;
    volatile bool *eta_filled = nullptr;

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
        chkerr(cudaMemAdvise(offsets, sizeof(ull) * HOST_OFFSET_SZ, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
        chkerr(cudaMemAdvise(vertices, sizeof(VertexID) * HOST_BUFF_SZ, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));

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
        if (ot + 3 > capacity[0] || vt + sglen > capacity[0])
        {
            throw std::runtime_error("Host buffer overflow");
        }
        assert(md == 0 || md <= sglen);
        otail[0] += 3;
        vtail[0] += sglen;
        offsets[ot] = vt;
        offsets[ot + 1] = md;
        offsets[ot + 2] = vtail[0];
        return vt;
    }

    SubgraphOffsets append_host_to_device(ull sglen, ull md = 0)
    {
        ull ot = otail[0];
        ull vt = vtail[0];

        if (ot + 3 > capacity[0] || vt + sglen > capacity[0])
        {
            throw std::runtime_error("Device buffer overflow");
        }
        assert(md == 0 || md <= sglen);

        ull host_offsets[3] = {vt, md, vt + sglen};
        chkerr(cudaMemcpy(offsets + ot, host_offsets, sizeof(host_offsets), cudaMemcpyHostToDevice));

        otail[0] = ot + 3;
        vtail[0] = vt + sglen;
        return {vt, md, vt + sglen};
    }

    __device__ ull append(ull sglen, ull md = 0)
    {
        ull vt = 0;
        if (LANEID == 0)
        {
            ull ot = atomicAdd(otail, 3);
            vt = atomicAdd(vtail, sglen);
            ui et = atomicAdd(n_tasks_proc, 1);
            const ull next_ot = ot + 3;
            const ull next_vt = vt + sglen;
            const ull vertex_base = second_buffer ? capacity[0] / 2 : 0;
            const ull vertex_span = second_buffer ? capacity[0] / 2 : capacity[0];

            if ((next_vt - vertex_base) >= static_cast<ull>(0.9 * vertex_span) || next_ot > capacity[0])
                overflow[0] = true;

            if (et + 1 > eta)
                eta_filled[0] = true;

            assert(md == 0 || md <= sglen);

            if (capacity[0] == HOST_BUFF_SZ)
                assert(next_ot <= HOST_OFFSET_SZ && next_vt <= HOST_BUFF_SZ);

            offsets[ot] = vt;
            offsets[ot + 1] = md;
            offsets[ot + 2] = next_vt;
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
            // atomicAdd(n_tasks_proc, 1);
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
            // atomicAdd(n_tasks_proc, 1);
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

    ull size()
    {
        if (empty())
            return 0;
        return (otail[0] - ohead[0]) / 3;
    }

    void clear()
    {
        vtail[0] = 0;
        otail[0] = 0;
        ohead[0] = 0;
    }

    DEVHOST bool isOverflow()
    {
        return overflow[0];
    }
    bool isApprochingEnd()
    {
        const ull vertex_base = second_buffer ? capacity[0] / 2 : 0;
        const ull vertex_span = second_buffer ? capacity[0] / 2 : capacity[0];
        return (vtail[0] - vertex_base) >= static_cast<ull>(0.8 * vertex_span) ||
               otail[0] + 3 > capacity[0];
    }
    void allocatePtrs()
    {
        chkerr(cudaMallocManaged((void **)&otail, sizeof(ull)));
        chkerr(cudaMallocManaged((void **)&vtail, sizeof(ull)));
        chkerr(cudaMallocManaged((void **)&ohead, sizeof(ull)));
        chkerr(cudaMallocManaged((void **)&capacity, sizeof(ull)));
        chkerr(cudaMallocManaged((void **)&n_tasks_proc, sizeof(ui)));
        chkerr(cudaMallocManaged((void **)&overflow, sizeof(bool)));
        chkerr(cudaMallocManaged((void **)&eta_filled, sizeof(bool)));
        overflow[0] = false;
        eta_filled[0] = false;
        otail[0] = 0;
        vtail[0] = 0;
        ohead[0] = 0;
    }
    void reset_pointers()
    {
        if (second_buffer)
        {
            otail[0] = capacity[0] / 2;
            vtail[0] = capacity[0] / 2;
            ohead[0] = capacity[0] / 2;
        }
        else
        {
            otail[0] = 0;
            vtail[0] = 0;
            ohead[0] = 0;
        }
    }
    __device__ void copy(auto &src, ull i, ull j)
    {
        vertices[i] = src.vertices[j];
    }
    __device__ void copy_range(auto &src, ull dst, ull src_st, ull len)
    {
        for (ull i = dst + LANEID, j = src_st + LANEID; j < src_st + len; i += 32, j += 32)
        {
            copy(src, i, j);
        }
    }
    void copy_host_range(auto &src, ull dst, ull src_st, ull len)
    {
        if (len == 0)
            return;
        chkerr(cudaMemcpy(vertices + dst, src.vertices + src_st, sizeof(VertexID) * len, cudaMemcpyDeviceToHost));
    }
    void copy_host_to_device_range(auto &src, ull dst, ull src_st, ull len)
    {
        if (len == 0)
            return;
        chkerr(cudaMemcpy(vertices + dst, src.vertices + src_st, sizeof(VertexID) * len, cudaMemcpyHostToDevice));
    }

    void print(string msg)
    {
        if (empty())
            cout << msg << "- empty -" << endl;
        else
            cout << msg << (otail[0] - ohead[0]) / 3 << " tasks (oh:" << ohead[0] << " ot:" << otail[0] << " vt:" << vtail[0] << ") " << endl;
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
