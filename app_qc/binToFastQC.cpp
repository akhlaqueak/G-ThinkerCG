#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <set>
#include <iostream>
#include <fstream>
#include <string>
#include <cstdio>
#include <sstream>
#include <cmath>
#include <time.h>
#include <chrono>
#include <cstring>
#include <cassert>
#include <limits>
#include <sys/timeb.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <sm_30_intrinsics.h>
#include <device_atomic_functions.h>
using namespace std;



// buffer size for CPU onehop and twohop adjacency array and offsets, ensure these are large enough
#define OFFSETS_SIZE 1000000000
#define LVL1ADJ_SIZE 1000000000
#define LVL2ADJ_SIZE 10000000000



// CPU GRAPH / CONSTRUCTOR
int h_sort_asce(const void* a, const void* b);
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

    CPU_Graph(const char* input_file, const char* output_file)
    {
        FILE* file_in = fopen(input_file, "rb");
        assert(file_in != NULL);

        size_t res = 0;
        size_t uintV_size = 0;
        size_t uintE_size = 0;
        size_t vertex_count = 0;
        size_t edge_count = 0;

        res += fread(&uintV_size, sizeof(size_t), 1, file_in);
        res += fread(&uintE_size, sizeof(size_t), 1, file_in);
        res += fread(&vertex_count, sizeof(size_t), 1, file_in);
        res += fread(&edge_count, sizeof(size_t), 1, file_in);

        assert(uintV_size == sizeof(int));
        assert(uintE_size == sizeof(uint64_t));
        assert(vertex_count <= static_cast<size_t>(numeric_limits<int>::max()));
        assert(edge_count <= static_cast<size_t>(numeric_limits<int>::max()));

        number_of_vertices = static_cast<int>(vertex_count);
        number_of_edges = static_cast<int>(edge_count);
        number_of_lvl2adj = 0;
        std::cout<<"|V|: "<<number_of_vertices<<endl;
        std::cout<<"|E|: "<<number_of_edges<<endl;
        onehop_offsets = new uint64_t[number_of_vertices + 1];
        onehop_neighbors = new int[number_of_edges];

        res += fread(onehop_offsets, sizeof(uint64_t), number_of_vertices + 1, file_in);
        res += fread(onehop_neighbors, sizeof(int), number_of_edges, file_in);
        assert(res == 4 + static_cast<size_t>(number_of_vertices + 1) + static_cast<size_t>(number_of_edges));

        fgetc(file_in);
        assert(feof(file_in));
        fclose(file_in);
        ofstream out(output_file);

        out << number_of_vertices << endl;
        out << number_of_edges << endl;

        // handle lvl2 adj
        for (int i = 0; i < number_of_vertices; i++) {
            out<<i<<" ";
            for (int j = onehop_offsets[i]; j < onehop_offsets[i + 1]; j++) {
                out<<onehop_neighbors[j]<<" ";
            }
            out<<endl;
        }
    }

    void write_serialized(char* output_file)
    {
        ofstream out(output_file);

        out << number_of_vertices << endl;
        out << number_of_edges << endl;
        out << number_of_lvl2adj << endl;

        for (int i = 0; i < number_of_edges; i++) {
            out << onehop_neighbors[i];
            if (i < number_of_edges - 1) {
                out << " ";
            }
        }
        out << endl;

        for (int i = 0; i < number_of_vertices + 1; i++) {
            out << onehop_offsets[i];
            if (i < number_of_vertices) {
                out << " ";
            }
        }
        out << endl;

        for (int i = 0; i < number_of_lvl2adj; i++) {
            out << twohop_neighbors[i];
            if (i < number_of_lvl2adj - 1) {
                out << " ";
            }
        }
        out << endl;

        for (int i = 0; i < number_of_vertices + 1; i++) {
            out << twohop_offsets[i];
            if (i < number_of_vertices) {
                out << " ";
            }
        }
        out << endl;

        out.close();
    }

    ~CPU_Graph()
    {
        delete[] onehop_neighbors;
        delete[] onehop_offsets;
    }
};



// MAIN
int main(int argc, char* argv[])
{
    // ENSURE PROPER USAGE
    if (argc != 3) {
        printf("Usage: ./main <graph_file> <output_file>\n");
        return 1;
    }
    FILE* graph_stream = fopen(argv[1], "rb");
    if (graph_stream == NULL) {
        printf("invalid graph file\n");
        return 1;
    }
    fclose(graph_stream);

    // GRAPH
    CPU_Graph hg(argv[1], argv[2]);
    
    return 0;
}

// sorts degrees in ascending order
int h_sort_asce(const void* a, const void* b)
{
    int n1;
    int n2;

    n1 = *(int*)a;
    n2 = *(int*)b;

    if (n1 < n2) {
        return -1;
    }
    else if (n1 > n2) {
        return 1;
    }
    else {
        return 0;
    }
}
