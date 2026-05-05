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
    read.open(dataset_path);

    if (!read.is_open()) {
        return 0;
    }

    int graph_size = 0;
    int number_of_edges = 0;
    int number_of_lvl2adj = 0;
    read >> graph_size;
    read >> number_of_edges;
    read >> number_of_lvl2adj;

    Graph = new int*[graph_size];
    delete []degree;
    degree = new int[graph_size];

    int *onehop_neighbors = new int[number_of_edges];
    unsigned long long *onehop_offsets = new unsigned long long[graph_size + 1];

    for (int i = 0; i < number_of_edges; ++i) {
        read >> onehop_neighbors[i];
    }

    for (int i = 0; i < graph_size + 1; ++i) {
        read >> onehop_offsets[i];
    }

    for (int i = 0; i < graph_size; ++i) {
        degree[i] = static_cast<int>(onehop_offsets[i + 1] - onehop_offsets[i]);
        int *temp_array = new int[degree[i]];
        for (int j = 0; j < degree[i]; ++j) {
            temp_array[j] = onehop_neighbors[onehop_offsets[i] + j];
        }
        Graph[i] = temp_array;
    }

    int discard = 0;
    for (int i = 0; i < number_of_lvl2adj; ++i) {
        read >> discard;
    }
    for (int i = 0; i < graph_size + 1; ++i) {
        read >> discard;
    }

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
