#ifndef MC_CPU_APP
#define MC_CPU_APP

#define TIME_THRESHOLD 10
#define TIME_OVER(ST) (chrono::duration_cast<chrono::milliseconds>(TIME_NOW - ST).count() > TIME_THRESHOLD)


class MCCPUWorker : public CPUWorker<MCTask>
{
public:
    ui max_sz = 0;
    ui total_counts=0;

    ui intersection_length(auto beg1, auto end1, auto beg2, auto end2)
    {
        auto t = TIME_NOW;
        ui count = 0;
        while (beg2 < end2)
        {
            if (binary_search(beg1, end1, *beg2))
                count++;
            beg2++;
        }
        return count;
    }
    void intersect(auto beg1, auto end1, auto beg2, auto end2, auto &res)
    {
        while (beg2 < end2)
        {
            if (binary_search(beg1, end1, *beg2))
                res.push_back(*beg2);
            beg2++;
        }
    }

    VertexID select_pivot(auto &P, auto &X)
    {
        VertexID pivot = P[0];
        ui max_count = 0;
        for (VertexID u : P)
        {
            VertexID nbr_count;
            const VertexID *nbrs = data_graph.getVertexNeighbors(u, nbr_count);
            VertexID count = intersection_length(nbrs, nbrs + nbr_count, P.begin(), P.end());
            if (count > max_count)
                max_count = count, pivot = u;
        }
        for (VertexID u : X)
        {
            VertexID nbr_count;
            const VertexID *nbrs = data_graph.getVertexNeighbors(u, nbr_count);
            VertexID count = intersection_length(nbrs, nbrs + nbr_count, P.begin(), P.end());
            if (count > max_count)
                max_count = count, pivot = u;
        }
        return pivot;
    }

    virtual MCTask *task_spawn(VertexID &data)
    {
        VertexID i = data;
        vector<VertexID> R{i};
        vector<VertexID> P, X;
        VertexID nbr_count;
        const VertexID *nbrs = data_graph.getVertexNeighbors(i, nbr_count);
        // cout << "-----------" << i << ", " << nbr_count << endl;
        for (int j = 0; j < nbr_count; ++j)
        {
            const VertexID neighbor = nbrs[j];
            if (neighbor < i)
                X.push_back(neighbor);
            else
                P.push_back(neighbor);
        }
        MCTask *t = new MCTask();
        t->context.R = move(R);
        t->context.P = move(P);
        t->context.X = move(X);
        return t;
    }

    void BK(vector<VertexID> &R, vector<VertexID> &P, vector<VertexID> &X, auto st)
    {
        vector<VertexID> Q, newP, newX;
        Q.reserve(P.size());
        newP.reserve(P.size());
        newX.reserve(X.size());

        if (P.size() == 0)
        {
            if (X.size() == 0)
            {
                max_sz = max_sz > R.size() ? max_sz : R.size();
                total_counts++;
            }
            return;
        }
        // find a pivot
        VertexID pivot = select_pivot(P, X);

        VertexID pivot_nbr_count;
        const VertexID *pivot_nbrs = data_graph.getVertexNeighbors(pivot, pivot_nbr_count);
        for (auto i = 0; i < P.size();)
        {
            VertexID u = P[i];
            if (std::binary_search(pivot_nbrs, pivot_nbrs + pivot_nbr_count, u))
            {
                i++;
                continue;
            }
            R.push_back(u);
            VertexID nbr_count;
            const VertexID *nbrs = data_graph.getVertexNeighbors(u, nbr_count);

            newP.clear();
            newX.clear();
            intersect(nbrs, nbrs + nbr_count, P.begin(), P.end(), newP);
            intersect(nbrs, nbrs + nbr_count, X.begin(), X.end(), newX);

            if (TIME_OVER(st))
            {
                MCTask *t = new MCTask();
                t->context.R = R;
                t->context.P = move(newP);
                t->context.X = move(newX);
                add_task(t);
            }
            // if(cond)
            else
            {
                BK(R, newP, newX, st);
            }
            P[i] = P.back();
            P.pop_back();

            X.push_back(u);
            R.pop_back();
        }
    }

    virtual void compute(MCContext &context)
    {
        BK(context.R, context.P, context.X, TIME_NOW);
    }
};

#endif

