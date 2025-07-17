#include "mc_unordered.h"
// #include "mc.h"
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

    // std::string file_path = "/home/lyuheng/TKDE-revise/TthinkerQ/data/data_graphs/amazon/com-amazon.ungraph.graph";
    // char * file_path_c    = "/home/lyuheng/TKDE-revise/TthinkerQ/data/data_graphs/patent/cit-Patents.graph";

    int num_compers = 32;

    MCWorker worker(num_compers);
    worker.load_data(argv[1]);

    auto start_t = std::chrono::steady_clock::now();

    worker.run();

    auto end_t = std::chrono::steady_clock::now();

    cout << "Total time (s): " << (float)std::chrono::duration_cast<std::chrono::milliseconds>(end_t - start_t).count()/1000 << endl;

    ULL total_results = 0;
    for(ui i=0; i<global_counters.size(); i++)
    {
        total_results += global_counters[i];
    }
    ui max_sz = 0;
    for(ui i=0; i<num_compers; i++)
    {
        max_sz = max_sz > worker.compers->max_sz ? max_sz : worker.compers->max_sz;
    }
    cout<<"Total count: "<< total_results << ", Max clique size: " << max_sz << endl;


    return 0;
}

