#ifndef SYSTEM_BUFFER_H
#define SYSTEM_BUFFER_H

#include "common/meta.h"
#include "common/gpu_env.h"

constexpr ull INVALID_BUFFER_POS = std::numeric_limits<ull>::max();

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
    unsigned long long st;
    unsigned long long md; 
    unsigned long long en;
    DEVHOST SubgraphOffsets(unsigned long long s, unsigned long long m, unsigned long long e) : st(s), md(m), en(e) {}
    DEVHOST SubgraphOffsets() : st(0), md(0), en(0) {}
};

template <class IndexType>
class IndexPair
{
public:
    IndexType ot;
    IndexType vt;
};

template <class IndexType>
class BufferBase
{
public:
    /**
     * @brief
     *
     */
    IndexType *offsets;
    VertexID *vertices;

    // should be transparent to the users
    IndexType *otail;
    IndexType *vtail;
    IndexType *ohead;

    size_t *buffsize;

    static size_t sizeOf()
    {
        return (sizeof(VertexID) + sizeof(IndexType));
    }
    /**
     * @brief This version is used to allocate memory on host. Call it only for HOST_BUFF_SZ
     *
     */
    void allocateMemory()
    {
        chkerr(cudaMallocManaged((void **)&offsets, sizeof(IndexType) * HOST_OFFSET_SZ));
        chkerr(cudaMallocManaged((void **)&vertices, sizeof(VertexID) * HOST_BUFF_SZ));
        // todo HOST_BUFF_SZ should be small in final release
        allocatePtrs();
        buffsize[0] = HOST_BUFF_SZ;
        std::cout << "Host allocated Buffer: " << buffsize[0] << std::endl;
    }

    void allocateMemory(size_t sz)
    {
        chkerr(cudaMalloc((void **)&offsets, sz * sizeof(IndexType)));
        chkerr(cudaMalloc((void **)&vertices, sz * sizeof(VertexID)));

        allocatePtrs();
        buffsize[0] = sz;
        std::cout << "Device allocated Buffer: " << buffsize[0] << std::endl;
    }

    __device__ IndexType append_thread(IndexType sglen, volatile bool *overflow)
    {
        IndexType ot = atomicAdd(otail, 3);
        IndexType vt = atomicAdd(vtail, sglen);
        const IndexType next_ot = ot + 3;
        const IndexType next_vt = vt + sglen;
        const IndexType offset_limit = buffsize[0] == HOST_BUFF_SZ ? HOST_OFFSET_SZ : buffsize[0];
        if (next_ot > offset_limit || next_vt > buffsize[0])
        {
            overflow[0] = true;
            if (next_ot <= offset_limit)
                offsets[ot] = offsets[ot + 1] = offsets[ot + 2] = vt;
            return INVALID_BUFFER_POS;
        }
        offsets[ot] = vt;
        offsets[ot + 1] = 0;
        offsets[ot + 2] = next_vt;
    
        return vt;
    }

    __device__ IndexType append(IndexType sglen, IndexType midpos, volatile bool *overflow)
    {
        IndexType vt = INVALID_BUFFER_POS;
        if (LANEID == 0)
        {
            IndexType ot = atomicAdd(otail, 3);
            IndexType write_vt = atomicAdd(vtail, sglen);
            const IndexType next_ot = ot + 3;
            const IndexType next_vt = write_vt + sglen;
            const IndexType offset_limit = buffsize[0] == HOST_BUFF_SZ ? HOST_OFFSET_SZ : buffsize[0];
            if (next_ot > offset_limit || next_vt > buffsize[0])
            {
                overflow[0] = true;
                if (next_ot <= offset_limit)
                    offsets[ot] = offsets[ot + 1] = offsets[ot + 2] = write_vt;
            }
            else
            {
                offsets[ot] = write_vt;
                offsets[ot + 1] = write_vt + midpos;
                offsets[ot + 2] = next_vt;
                vt = write_vt;
            }
        }
        vt = __shfl_sync(FULL, vt, 0);
        return vt;
    }

    __device__ IndexType append(IndexType sglen, volatile bool *overflow)
    {
        IndexType vt = INVALID_BUFFER_POS;
        if (LANEID == 0)
        {
            IndexType ot = atomicAdd(otail, 3);
            IndexType write_vt = atomicAdd(vtail, sglen);
            const IndexType next_ot = ot + 3;
            const IndexType next_vt = write_vt + sglen;
            const IndexType offset_limit = buffsize[0] == HOST_BUFF_SZ ? HOST_OFFSET_SZ : buffsize[0];
            if (next_ot > offset_limit || next_vt > buffsize[0])
            {
                overflow[0] = true;
                if (next_ot <= offset_limit)
                    offsets[ot] = offsets[ot + 1] = offsets[ot + 2] = write_vt;
            }
            else
            {
                offsets[ot] = write_vt;
                offsets[ot + 1] = 0;
                offsets[ot + 2] = next_vt;
                vt = write_vt;
            }
        }
        vt = __shfl_sync(FULL, vt, 0);
        return vt;
    }

    /**
     * if mode == PREFIX, sglen is the prefix length, num is # of subgraphs to be expanded
     * if mode == EXPAND, sglen is length of subgraph, num is # of subgraphs
     */
    __device__ IndexType append_batch(IndexType sglen, ui num, volatile bool *overflow, StoreStrategy mode)
    {
        IndexType vt = INVALID_BUFFER_POS, ot = 0;
        if (mode == StoreStrategy::EXPAND)
        {
            if (LANEID == 0)
            {
                ot = atomicAdd(otail, 3 * num);
                IndexType write_vt = atomicAdd(vtail, sglen * num);
                const IndexType next_ot = ot + 3 * num;
                const IndexType next_vt = write_vt + sglen * num;
                const IndexType offset_limit = buffsize[0] == HOST_BUFF_SZ ? HOST_OFFSET_SZ : buffsize[0];
                if (next_ot > offset_limit || next_vt > buffsize[0])
                {
                    overflow[0] = true;
                    if (next_ot <= offset_limit)
                    {
                        for (ui i = 0; i < num; ++i)
                            offsets[ot + 3 * i] = offsets[ot + 3 * i + 1] = offsets[ot + 3 * i + 2] = write_vt;
                    }
                }
                else
                    vt = write_vt;
            }
            vt = __shfl_sync(FULL, vt, 0);
            if (vt == INVALID_BUFFER_POS)
                return vt;
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
                IndexType write_vt = atomicAdd(vtail, sglen + num);
                const IndexType next_ot = ot + 3;
                const IndexType next_vt = write_vt + sglen + num;
                const IndexType offset_limit = buffsize[0] == HOST_BUFF_SZ ? HOST_OFFSET_SZ : buffsize[0];
                if (next_ot > offset_limit || next_vt > buffsize[0])
                {
                    overflow[0] = true;
                    if (next_ot <= offset_limit)
                        offsets[ot] = offsets[ot + 1] = offsets[ot + 2] = write_vt;
                }
                else
                {
                    offsets[ot] = write_vt;
                    offsets[ot + 1] = write_vt + sglen;
                    offsets[ot + 2] = next_vt;
                    vt = write_vt;
                }
            }
            vt = __shfl_sync(FULL, vt, 0);
            return vt;
        }
        else
        {
            assert(false);
            return INVALID_BUFFER_POS;
        }
    }


    // mode = StoreStrategy::EXPAND by default
    __device__ SubgraphOffsets next()
    {
        const IndexType offset_limit = buffsize[0] == HOST_BUFF_SZ ? HOST_OFFSET_SZ : buffsize[0];
        while (true)
        {
            IndexType s;
            if (LANEID == 0)
                s = atomicAdd(ohead, 3);
            s = __shfl_sync(FULL, s, 0);
            if (s >= otail[0] || s + 3 > offset_limit)
                return {vtail[0], 0, 0};
            SubgraphOffsets so = {offsets[s], offsets[s + 1], offsets[s + 2]};
            if (so.st != so.en)
                return so;
        }
    }

    __device__ SubgraphOffsets next(StoreStrategy mode)
    {
        // A block will try to get a batch of subgraphs
        IndexType s;
        if (mode == StoreStrategy::EXPAND)
        {
            if (LANEID == 0)
            {
                s = atomicAdd(ohead, 2);
            }
            s = __shfl_sync(FULL, s, 0);
            if (s < otail[0])
                return {offsets[s], 0, offsets[s + 1]}; // md is invalid
            else
                return {vtail[0], 0, 0};
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
                return {vtail[0], 0, 0};
        }
        else
        {
            assert(false);
            return {0, 0, 0};
        }
    }

    __device__ bool isEnd(SubgraphOffsets so)
    {
        return so.st == vtail[0];
    }

    // another version of isEnd(so)
    __device__ bool isEnd(unsigned long long st)
    {
        return st == vtail[0];
    }

    bool isEmpty()
    {
        if (ohead[0] >= otail[0])
        {
            return true;
        }
        return false;
    }

    void clear()
    {
        vtail[0] = 0;
        otail[0] = 0;
        ohead[0] = 0;
    }

private:
    void allocatePtrs()
    {
        chkerr(cudaMallocManaged((void **)&otail, sizeof(IndexType)));
        chkerr(cudaMallocManaged((void **)&vtail, sizeof(IndexType)));
        chkerr(cudaMallocManaged((void **)&ohead, sizeof(IndexType)));
        chkerr(cudaMallocManaged((void **)&buffsize, sizeof(size_t)));
        otail[0] = 0;
        vtail[0] = 0;
        ohead[0] = 0;
    }
};

#endif
