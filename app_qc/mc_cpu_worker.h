#ifndef MC_CPU_APP
#define MC_CPU_APP

#define TIME_THRESHOLD 10
#define TIME_OVER(ST) (chrono::duration_cast<chrono::milliseconds>(TIME_NOW - ST).count() > TIME_THRESHOLD)

class CPU_Graph
{
    public:

    int number_of_vertices;
    int number_of_edges;
    uint64_t number_of_lvl2adj;

    // one dimentional arrays of 1hop and 2hop neighbors and the offsets for each vertex
    int* onehop_neighbors;
    uint64_t* onehop_offsets;
    int* twohop_neighbors;
    uint64_t* twohop_offsets;

    CPU_Graph(ifstream& graph_stream)
    {
        graph_stream >> number_of_vertices;
        graph_stream >> number_of_edges;
        graph_stream >> number_of_lvl2adj;

        onehop_neighbors = new int[number_of_edges];
        onehop_offsets = new uint64_t[number_of_vertices + 1];
        twohop_neighbors = new int[number_of_lvl2adj];
        twohop_offsets = new uint64_t[number_of_vertices + 1];

        for (int i = 0; i < number_of_edges; i++) {
            graph_stream >> onehop_neighbors[i];
        }

        for (int i = 0; i < number_of_vertices + 1; i++) {
            graph_stream >> onehop_offsets[i];
        }

        for (int i = 0; i < number_of_lvl2adj; i++) {
            graph_stream >> twohop_neighbors[i];
        }

        for (int i = 0; i < number_of_vertices + 1; i++) {
            graph_stream >> twohop_offsets[i];
        }
    }

    ~CPU_Graph() 
    {
        delete onehop_neighbors;
        delete onehop_offsets;
        delete twohop_neighbors;
        delete twohop_offsets;
    }
};

class QCCPUWorker : public CPUWorker<QCTask>
{
public:

    virtual QCTask *task_spawn(VertexID &data)
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

    virtual void compute(QCContext &context)
    {
        QC(context);
    }
};

#endif

