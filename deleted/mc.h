#pragma once

#include "../system/timer.h"
#include "../system/worker.h"
#include "../system/task.h"
// #include "graph.h"
#include <vector>
#include "../g2-aimd/system/worker.h"
#include "BK.h"

using namespace std;

#define TIME_THRESHOLD 100
#define TIME_OVER(ST) (chrono::duration_cast<chrono::milliseconds>(TIME_NOW - ST).count() > TIME_THRESHOLD)
//  todo different threshold

using namespace std::chrono;

Graph data_graph;

typedef unsigned long long int ULL;

class bkVector
{
    ui *data;
    unsigned int len, cap;

public:
    bkVector()
    {
        data = nullptr;
        len = cap = 0;
    }
    bkVector(bkVector &src)
    {
        data = new ui[src.len];
        len = src.len;
        cap = src.len;
        copy(src.begin(), src.end(), begin());
    }
    bkVector(bkVector &src, ui sz)
    {
        data = new ui[sz];
        len = src.len;
        cap = sz;
        copy(src.begin(), src.end(), begin());
    }
    bkVector(unsigned int cap)
    {
        data = new ui[cap];
        len = 0;
        this->cap = cap;
    }

    void append(ui v)
    {
        assert(len < cap);
        data[len++] = v;
    }
    void resize(ui sz)
    {
        assert(sz <= cap);
        len = sz;
    }
    ui *begin() const
    {
        return data;
    }
    ui *end() const
    {
        return data + len;
    }
    unsigned int size() const
    {
        return len;
    }
    bool empty()
    {
        return len == 0;
    }
    ui& back(){
        assert(!empty());
        return data[len-1];
    }
    ui& operator[](unsigned int index)
    {
        assert(index < len);
        return data[index];
    }
    void remove(ui v)
    {
        assert(!empty());
        auto it = find(data, data + len, v);
        *it = move(data[len - 1]);
        len--;
    }
    ui pop()
    {
        assert(!empty());
        return data[--len];
    }
    void clear()
    {
        len = 0;
    }

    ~bkVector()
    {
        if (data != nullptr)
        {
            delete[] data;
            data = nullptr;
        }
    }
    void ordered_remove(auto val)
    {
        auto it = std::lower_bound(begin(), end(), val);
        assert(it != end());
        for (; it + 1 < end(); it++)
            *it = *(it + 1);
        *it = val;
        len--;
    }

    void ordered_insert(auto val)
    {
        auto pos = std::lower_bound(begin(), end(), val);
        assert(size() < cap);
        for (auto it = pos; it < end(); it++)
            swap(val, *it);
        append(val);
    }
    friend ifbinstream &operator<<(ifbinstream &m, const bkVector &v)
    {
        m << v.cap;
        m << v.len;
        m.raw_bytes(v.data, v.cap * sizeof(ui));
        return m;
    }

    friend ofbinstream &operator>>(ofbinstream &m, bkVector &v)
    {
        m >> v.cap;
        m >> v.len;
        v.data = new ui[v.cap];
        ui size = v.cap;
        size_t len = STREAM_MEMBUF_SIZE / 2 / sizeof(ui);
        size_t bytes = len * sizeof(ui);
        ui *it = v.data;
        while (size > len)
        {
            ui *data = (ui *)m.raw_bytes(bytes);
            std::copy(data, data + len, it);
            it += len;
            size -= len;
        }
        ui *data = (ui *)m.raw_bytes(bytes);
        std::copy(data, data + len, it);
        return m;
    }
};
void intersect(auto beg1, auto end1, auto beg2, auto end2, auto &res)
{
    while (beg2 < end2)
    {
        ui u = *beg2;
        if (binary_search(beg1, end1, u))
            res.append(u);
        beg2++;
    }
}
ui intersection_length(auto beg1, auto end1, auto beg2, auto end2)
{
    ui res = 0;
    while (beg2 < end2)
    {
        ui u = *beg2;
        if (binary_search(beg1, end1, u))
            res++;
        beg2++;
    }
    return res;
}

struct ContextValue
{
    bkVector *R;
    bkVector *P;
    bkVector *X;
    bkVector *Xp;
};
#define RAPIDS

ofbinstream &operator>>(ofbinstream &m, ContextValue &c)
{
    c.R = new bkVector();
    c.P = new bkVector();
    c.X = new bkVector();
    c.Xp = new bkVector();

    m >> *(c.R);
    m >> *(c.P);
    m >> *(c.X);
    m >> *(c.Xp);
    return m;
}
ifbinstream &operator<<(ifbinstream &m, const ContextValue &c)
{
    m << *(c.R);
    m << *(c.P);
    m << *(c.X);
    m << *(c.Xp);
    return m;
}

typedef Task<ContextValue> MCTask;

class MCComper : public Comper<MCTask, ui>
{
public:
    ULL counter = 0;
    ui max_sz = 0;

    MCComper()
    {
    }

    ui intersection_length(auto beg1, auto end1, auto beg2, auto end2)
    {
        ui count = 0;
        while (beg1 < end1 && beg2 < end2)
            if (*beg1 < *beg2)
                ++beg1;
            else if (*beg1 > *beg2)
                ++beg2;
            else
                ++count, ++beg1, ++beg2;
        return count;
    }

    ui select_pivot(auto &P, auto &X, auto &Xp)
    {
        ui pivot = P[0];
        ui max_count = 0;
        for (ui u : P)
        {
            ui nbr_count;
            const ui *nbrs = data_graph.getVertexNeighbors(u, nbr_count);
            ui count = intersection_length(nbrs, nbrs + nbr_count, P.begin(), P.end());
            if (count > max_count)
                max_count = count, pivot = u;
        }
        for (ui u : X)
        {
            ui nbr_count;
            const ui *nbrs = data_graph.getVertexNeighbors(u, nbr_count);
            ui count = intersection_length(nbrs, nbrs + nbr_count, P.begin(), P.end());
            if (count > max_count)
                max_count = count, pivot = u;
        }
        for (ui u : Xp)
        {
            ui nbr_count;
            const ui *nbrs = data_graph.getVertexNeighbors(u, nbr_count);
            ui count = intersection_length(nbrs, nbrs + nbr_count, P.begin(), P.end());
            if (count > max_count)
                max_count = count, pivot = u;
        }
        return pivot;
    }

    virtual bool task_spawn(ui &data)
    {
        ui i = data;
        ui nbr_count;
        const ui *nbrs = data_graph.getVertexNeighbors(i, nbr_count);
        auto it = std::lower_bound(nbrs, nbrs + nbr_count, i);
        ui psz = (nbrs + nbr_count) - it;

        MCTask *t = new MCTask();
        t->context.R = new bkVector(psz + 1);
        t->context.P = new bkVector(psz);
        t->context.Xp = new bkVector(psz);
        t->context.X = new bkVector(nbr_count);
        t->context.R->append(i);

        // cout << "-----------" << i << ", " << nbr_count << endl;
        for (int j = 0; j < nbr_count; ++j)
        {
            const ui neighbor = nbrs[j];
            if (neighbor < i)
                t->context.X->append(neighbor);
            else
                t->context.P->append(neighbor);
        }
        add_task(t);

        return true;
    }

    void BK(bkVector &R, bkVector &P, bkVector &X, bkVector &Xp, auto st)
    {
        // struct timeb cur_time;

        bkVector newX(X.size());
        bkVector newP(P.size());
        bkVector newXp(Xp.size() + P.size());
        ui psz = P.size();
        if (P.size() == 0)
        {
            if (X.size() == 0 && Xp.size() == 0)
            {
                max_sz = max_sz > R.size() ? max_sz : R.size();
                counter++;
            }
            return;
        }
        // find a pivot
        // todo try without pivoting for mgwikitionary graph
        ui pivot = select_pivot(P, X, Xp);

        ui pivot_nbr_count;
        const ui *pivot_nbrs = data_graph.getVertexNeighbors(pivot, pivot_nbr_count);
        // Q.clear();
        // pivot_nbr_count = 0;
        // todo change it to two pointers merge,
        // todo check which range is smaller and do the intersection with it
        // for (auto u : P){
        //     if(!std::binary_search(pivot_nbrs, pivot_nbrs + pivot_nbr_count, u))
        //         Q.append(u);
        // }
        // auto enq = std::set_difference(P.begin(), P.end(),pivot_nbrs, pivot_nbrs + pivot_nbr_count, Q.begin());
        // Q.resize(distance(Q.begin(), enq));
        for (ui i = 0; i < P.size();)
        {
            ui u = P[i];
            ui nbr_count;
            const ui *nbrs = data_graph.getVertexNeighbors(u, nbr_count);
            if (std::binary_search(pivot_nbrs, pivot_nbrs + pivot_nbr_count, u))
            {
                i++;
                continue;
            }

            R.append(u);

            
            newXp.clear();
            newP.clear();
            newX.clear();
            intersect(nbrs, nbrs + nbr_count, P.begin(), P.end(), newP); // results are placed in P
            intersect(nbrs, nbrs + nbr_count, X.begin(), X.end(), newX); // results are placed in X
            intersect(nbrs, nbrs + nbr_count, Xp.begin(), Xp.end(), newXp);

            if (!TIME_OVER(st))
            {
                BK(R, newP, newX, newXp, st);
            }
            else
            {
                MCTask *t = new MCTask();
                t->context.R = new bkVector(R, R.size() + P.size());
                t->context.P = new bkVector(newP);
                t->context.X = new bkVector(newX);
                t->context.Xp = new bkVector(newXp, newXp.size() + P.size());
                add_task(t);
            }
            Xp.append(u);
            P[i] = P.back();
            P.pop();
            R.pop();
        }
    }

    virtual void compute(ContextT &context)
    {
        // ftime(&data_graph.gtime_start[thread_id]);
        counter = global_counters[thread_id];
        BK(*(context.R), *(context.P), *(context.X), *(context.Xp), TIME_NOW);
        global_counters[thread_id] = counter;
        global_sz[thread_id] = max_sz;
        delete context.R;
        delete context.P;
        delete context.X;
        delete context.Xp;
    }

    virtual bool is_bigTask(MCTask *task)
    {
        if (task->context.P->size() > BIGTASK_THRESHOLD)
        {
            return true;
        }
        return false;
    }
};

class MCWorker : public Worker<MCComper, GPUComper<MCGPU> >
{
public:
    MCWorker(ui num_compers) : Worker(num_compers)
    {
    }

    ~MCWorker()
    {
    }

    void load_data(ui argc, char* argv[])
    {
        CommandLine cmd(argc, argv);
        std::string fp = cmd.GetOptionValue("-dg", "./data/com-friendster.ungraph.txt.bin");
        data_graph = Graph(fp);
        // here read only 50% of data, rest of data should go to gpucomper
        double cpu_share = cmd.GetOptionDoubleValue("-cs", 0.5);

        CPU_PROC = data_graph.GetVertexCount()*cpu_share;
        cout<<"CPU Share: "<<CPU_PROC<<endl;
        for (int i = 0; i < CPU_PROC; ++i)
            data_array.push_back(new ui(i));
        
        MCGPU app;
        gpu_comper=new GPUComper<MCGPU>();
        gpu_comper->start(argc, argv, app);
    }

    virtual bool task_spawn(ui &data)
    {
        ui i = data;
        ui nbr_count;
        const ui *nbrs = data_graph.getVertexNeighbors(i, nbr_count);
        auto it = std::lower_bound(nbrs, nbrs + nbr_count, i);
        ui psz = (nbrs + nbr_count) - it;

        MCTask *t = new MCTask();
        t->context.R = new bkVector(psz + 1);
        t->context.P = new bkVector(psz);
        t->context.Xp = new bkVector(psz);
        t->context.X = new bkVector(nbr_count);
        t->context.R->append(i);

        // cout << "-----------" << i << ", " << nbr_count << endl;
        for (int j = 0; j < nbr_count; ++j)
        {
            const ui neighbor = nbrs[j];
            if (neighbor < i)
                t->context.X->append(neighbor);
            else
                t->context.P->append(neighbor);
        }

        add_task(t);

        return true;
    }

    virtual bool is_bigTask(MCTask *task)
    {
        if (task->context.P->size() > BIGTASK_THRESHOLD)
        {
            return true;
        }
        return false;
    }
};