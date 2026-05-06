#include<fstream>
#include<iostream>
#include<string>
#include<list>

using namespace std;

class Util{
    public:
        int ReadGraph(string dataset_path,int **&Graph, int *&degree, int &bipartite);
        int ReadGraph(string dataset_path,int **&Graph, int *&degree); 
};

static int ReadSerializedGraph(string dataset_path, int **&Graph, int *&degree){
    ifstream read;
    read.open(dataset_path, ios::binary);

    if (!read.is_open()) {
        return 0;
    }

    int graph_size = 0;
    uint64_t number_of_edges = 0;
    uint64_t number_of_lvl2adj = 0;
    read.read(reinterpret_cast<char*>(&graph_size), sizeof(graph_size));
    read.read(reinterpret_cast<char*>(&number_of_edges), sizeof(number_of_edges));
    read.read(reinterpret_cast<char*>(&number_of_lvl2adj), sizeof(number_of_lvl2adj));

    Graph = new int*[graph_size];
    delete []degree;
    degree = new int[graph_size];

    int *onehop_neighbors = new int[static_cast<size_t>(number_of_edges)];
    unsigned long long *onehop_offsets = new unsigned long long[graph_size + 1];

    read.read(reinterpret_cast<char*>(onehop_neighbors), sizeof(int) * static_cast<size_t>(number_of_edges));
    read.read(reinterpret_cast<char*>(onehop_offsets), sizeof(unsigned long long) * (graph_size + 1));

    for (int i = 0; i < graph_size; ++i) {
        degree[i] = static_cast<int>(onehop_offsets[i + 1] - onehop_offsets[i]);
        int *temp_array = new int[degree[i]];
        for (int j = 0; j < degree[i]; ++j) {
            temp_array[j] = onehop_neighbors[onehop_offsets[i] + j];
        }
        Graph[i] = temp_array;
    }

    read.seekg(static_cast<std::streamoff>(sizeof(int) * static_cast<size_t>(number_of_lvl2adj)), ios::cur);
    read.seekg(static_cast<std::streamoff>(sizeof(unsigned long long) * (graph_size + 1)), ios::cur);

    delete []onehop_neighbors;
    delete []onehop_offsets;
    return graph_size;
}

int Util::ReadGraph(string dataset_path,int **&Graph, int *&degree, int &bipartite){
    bipartite = 0;
    return ReadSerializedGraph(dataset_path, Graph, degree);
}

int Util::ReadGraph(string dataset_path,int **&Graph, int *&degree){
    return ReadSerializedGraph(dataset_path, Graph, degree);
}
