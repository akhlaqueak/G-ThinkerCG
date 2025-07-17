#include "mc_unordered.h"
// #include "mc_Xp.h"
#include <iostream>
#include <fstream>
#include <ctime>
#include <cstdlib>
#include <string>

// #define GEN_GRAPH

using namespace std;

// #include <chrono>
// using namespace std::chrono;

int main(int argc, char *argv[])
{
    std::string fp = std::string(argv[1]);

    data_graph.loadGraphFromFile(fp);
    Timer t;
    ui degen = 0;
    for (ui i = 0; i < data_graph.getVerticesCount(); i++)
    {
        ui nbr_count;
        const ui *nbrs = data_graph.getVertexNeighbors(i, nbr_count);
        auto it = std::lower_bound(nbrs, nbrs + nbr_count, i);
        ui psz = (nbrs + nbr_count) - it;
        degen = max(degen, psz);
    }

    cout << "Degeneracy : " << degen << endl;
    for(ui i=0; i<omp_get_max_threads(); i++)
        mcs.push_back(new MCComper());
    cout << "Max degree: " << data_graph.getGraphMaxDegree() << endl;
    cout<<"Threads: "<<omp_get_max_threads()<<endl;
    ui cliques = 0;
    ui cliques1 = 0;
#pragma omp parallel
    {

        ContextValue context;
        context.R = new bkVector(degen + 1);
        context.P = new bkVector(degen);
        context.X = new bkVector(data_graph.getGraphMaxDegree());
        context.Xp = new bkVector(degen);
        MCComper* mc = mcs[omp_get_thread_num()];
#pragma omp for schedule(dynamic)
        for (ui i = 0; i < data_graph.getVerticesCount(); i++)
        {
            // cout<<i<<": "<<endl;
            context.clear();
            mc->createTask(i, context);
#pragma omp taskgroup
            {
                mc->BK(*(context.R), *(context.P), *(context.X), *(context.Xp), TIME_NOW);
            }
        }
    }
    for (auto &mc : mcs)
        cliques += mc->counter;
    cout << "time: " << t.elapsed() / 1e6 << "cliques: " << cliques << endl;

    return 0;
}
