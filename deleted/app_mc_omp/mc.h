#pragma once

#include "../system/timer.h"
#include "../system/worker.h"
#include "../system/task.h"
#include "graph.h"
#include <vector>
using namespace std;

#define TIME_THRESHOLD 100
#define TIME_OVER(ST) (chrono::duration_cast<chrono::milliseconds>(TIME_NOW - ST).count() > TIME_THRESHOLD)
//  todo different threshold

using namespace std::chrono;

Graph data_graph;

typedef unsigned long long int ULL;
class MCComper;
vector<MCComper*> mcs;
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
    ui operator[](unsigned int index)
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
    void pop()
    {
        assert(!empty());
        len--;
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
};
ui *inplace_set_intersection(auto beg1, auto end1, auto beg2, auto end2)
{
    auto res = beg2;
    while (beg1 < end1 && beg2 < end2)
        if (*beg1 < *beg2)
            ++beg1;
        else if (*beg1 > *beg2)
            ++beg2;
        else
        {
            std::swap(*(res), *beg2);
            ++res, ++beg1, ++beg2;
        }
    return res;
}

struct ContextValue
{
    bkVector *R;
    bkVector *P;
    bkVector *X;
    bkVector *Xp;
    void clear()
    {
        R->clear();
        P->clear();
        X->clear();
        Xp->clear();
    }
};

class MCComper
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
    ui select_pivot_max_degree(auto &P, auto &X)
    {
        ui pivot = P[0];
        ui max_count = 0;
        for (ui u : P)
        {
            ui count = data_graph.getVertexDegree(u);
            if (count > max_count)
                max_count = count, pivot = u;
        }
        for (ui u : X)
        {
            ui count = data_graph.getVertexDegree(u);
            if (count > max_count)
                max_count = count, pivot = u;
        }
        return pivot;
    }

    void BK(bkVector &R, bkVector &P, bkVector &X, bkVector &Xp, auto st)
    {
        struct timeb cur_time;

        // bkVector Q(P.size()+X.size());
        // bkVector newP(P.size());
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

            // newP.clear();
            // vector<ui> test;
            // for(ui u: P) test.push_back(u);
            newXp.clear();
            auto enp = inplace_set_intersection(nbrs, nbrs + nbr_count, P.begin(), P.end()); // results are placed in P
            auto enx = inplace_set_intersection(nbrs, nbrs + nbr_count, X.begin(), X.end()); // results are placed in X
            auto enxp = std::set_intersection(nbrs, nbrs + nbr_count, Xp.begin(), Xp.end(), newXp.begin());
            ui old_p_sz = P.size();
            ui old_x_sz = X.size();
            P.resize(distance(P.begin(), enp));
            X.resize(distance(X.begin(), enx));
            newXp.resize(distance(newXp.begin(), enxp));

            ftime(&cur_time);
            // double drun_time = cur_time.time-data_graph.gtime_start[thread_id].time;
            // double drun_time = cur_time.time-data_graph.gtime_start[thread_id].time+(double)(cur_time.millitm-data_graph.gtime_start[thread_id].millitm)/1000;
            // drun_time = 0;
            if (TIME_OVER(st))
            {
                ContextValue *ctx = new ContextValue();
                ctx->R = new bkVector(R, R.size() + P.size());
                ctx->P = new bkVector(P);
                ctx->X = new bkVector(X);
                ctx->Xp = new bkVector(newXp, newXp.size() + P.size());
#pragma omp task firstprivate(ctx)
                {
                    mcs[omp_get_thread_num()]->BK(*(ctx->R), *(ctx->P), *(ctx->X), *(ctx->Xp), TIME_NOW);
                    delete ctx->R;
                    delete ctx->P;
                    delete ctx->X;
                    delete ctx->Xp;
                }
            }
            else
            {
                BK(R, P, X, newXp, st);
            }

            std::sort(P.end(), P.begin() + old_p_sz);
            std::sort(X.end(), X.begin() + old_x_sz);
            std::inplace_merge(P.begin(), P.end(), P.begin() + old_p_sz);
            std::inplace_merge(X.begin(), X.end(), X.begin() + old_x_sz);
            P.resize(old_p_sz);
            X.resize(old_x_sz);
            P.ordered_remove(u);
            Xp.ordered_insert(u);
            R.pop();
        }
    }
    void createTask(ui &data, ContextValue &c)
    {
        ui i = data;
        ui nbr_count;
        const ui *nbrs = data_graph.getVertexNeighbors(i, nbr_count);

        // cout << "-----------" << i << ", " << nbr_count << endl;
        for (int j = 0; j < nbr_count; ++j)
        {
            const ui neighbor = nbrs[j];
            if (neighbor < i)
                c.X->append(neighbor);
            else
                c.P->append(neighbor);
        }
    }
};
