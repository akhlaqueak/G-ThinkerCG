#pragma once

#include <cstddef>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <vector>
#include <iostream>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include "config.h"


namespace STMatch {

  inline void stmatch_cuda_check(cudaError_t err, const char* call) {
    if (err != cudaSuccess) {
      std::cerr << "CUDA error after " << call << ": " << cudaGetErrorString(err) << std::endl;
      exit(1);
    }
  }

  typedef struct {

    graph_node_t nnodes = 0;
    graph_edge_t nedges = 0;
    bitarray32* vertex_label;
    graph_edge_t* rowptr;
    graph_node_t* colidx;
  } Graph;

  struct GraphPreprocessor {

    Graph g;

    GraphPreprocessor(std::string filename) {
      readfile(filename);
    }

    Graph* to_gpu() {
      Graph gcopy = g;

      stmatch_cuda_check(cudaMalloc(&gcopy.vertex_label, sizeof(bitarray32) * g.nnodes), "cudaMalloc(graph.vertex_label)");
      stmatch_cuda_check(cudaMalloc(&gcopy.rowptr, sizeof(graph_edge_t) * (g.nnodes + 1)), "cudaMalloc(graph.rowptr)");
      stmatch_cuda_check(cudaMalloc(&gcopy.colidx, sizeof(graph_node_t) * g.nedges), "cudaMalloc(graph.colidx)");
      stmatch_cuda_check(cudaMemcpy(gcopy.vertex_label, g.vertex_label, sizeof(bitarray32) * g.nnodes, cudaMemcpyHostToDevice), "cudaMemcpy(graph.vertex_label)");
      stmatch_cuda_check(cudaMemcpy(gcopy.rowptr, g.rowptr, sizeof(graph_edge_t) * (g.nnodes + 1), cudaMemcpyHostToDevice), "cudaMemcpy(graph.rowptr)");
      stmatch_cuda_check(cudaMemcpy(gcopy.colidx, g.colidx, sizeof(graph_node_t) * g.nedges, cudaMemcpyHostToDevice), "cudaMemcpy(graph.colidx)");

      Graph* gpu_g;
      stmatch_cuda_check(cudaMalloc(&gpu_g, sizeof(Graph)), "cudaMalloc(Graph)");
      stmatch_cuda_check(cudaMemcpy(gpu_g, &gcopy, sizeof(Graph), cudaMemcpyHostToDevice), "cudaMemcpy(Graph)");
      return gpu_g;
    }

    void readfile(std::string& filename) {
      if (is_gthinker_bin_file(filename)) {
        read_gthinker_bin_file(filename);
      }
      else {
        //read_lg_file(filename);
        read_bin_file(filename);
      }
    }

    bool is_gthinker_bin_file(const std::string& filename) const {
      const std::string suffix = ".bin";
      if (filename.size() < suffix.size() ||
          filename.compare(filename.size() - suffix.size(), suffix.size(), suffix) != 0) {
        return false;
      }

      std::ifstream fin(filename.c_str(), std::ios::binary);
      return fin.good();
    }

    void read_lg_file(std::string& filename) {
      std::ifstream fin(filename);
      std::string line;
      while (std::getline(fin, line) && (line[0] == '#'));
      g.nnodes = 0;
      std::vector<int> vertex_labels;
      do {
        std::istringstream sin(line);
        char tmp;
        int v;
        int label;
        sin >> tmp >> v >> label;
        vertex_labels.push_back(label);
        g.nnodes++;
      } while (std::getline(fin, line) && (line[0] == 'v'));
      std::vector<std::vector<graph_node_t>> adj_list(g.nnodes);
      do {
        std::istringstream sin(line);
        char tmp;
        int v1, v2;
        int label;
        sin >> tmp >> v1 >> v2 >> label;
        adj_list[v1].push_back(v2);
        adj_list[v2].push_back(v1);
      } while (getline(fin, line));

      assert(vertex_labels.size() == g.nnodes);

      g.vertex_label = new bitarray32[vertex_labels.size()];
      for (int i = 0; i < g.nnodes; i++) {
        g.vertex_label[i] = (1 << vertex_labels[i]);
      }
      // memcpy(g.vertex_label, vertex_labels.data(), sizeof(int) * vertex_labels.size());

      g.rowptr = new graph_edge_t[g.nnodes + 1];
      g.rowptr[0] = 0;

      std::vector<graph_node_t> colidx;

      for (graph_node_t i = 0; i < g.nnodes; i++) {
        sort(adj_list[i].begin(), adj_list[i].end());
        int pos = 0;
        for (graph_node_t j = 1; j < adj_list[i].size(); j++) {
          if (adj_list[i][j] != adj_list[i][pos]) adj_list[i][++pos] = adj_list[i][j];
        }

        if (adj_list[i].size() > 0)
          colidx.insert(colidx.end(), adj_list[i].data(), adj_list[i].data() + pos + 1);  // adj_list is sorted

        adj_list[i].clear();
        g.rowptr[i + 1] = colidx.size();
      }
      g.nedges = colidx.size();
      g.colidx = new graph_node_t[colidx.size()];

      memcpy(g.colidx, colidx.data(), sizeof(graph_node_t) * colidx.size());

     // std::cout << "Graph read complete. Number of vertex: " << g.nnodes << std::endl;
    }


    template<typename T>
    void read_subfile(std::string fname, T*& pointer, size_t elements) {
      pointer = (T*)malloc(sizeof(T) * elements);
      assert(pointer);
      std::ifstream inf(fname.c_str(), std::ios::binary);
      if (!inf.good()) {
        std::cerr << "Failed to open file: " << fname << "\n";
        exit(1);
      }
      inf.read(reinterpret_cast<char*>(pointer), sizeof(T) * elements);
      inf.close();
    }


    void read_bin_file(std::string& filename) {
      std::ifstream f_meta((filename + ".meta.txt").c_str());
      assert(f_meta);

      graph_node_t n_vertices;
      graph_edge_t n_edges;
      int vid_size;
      graph_node_t max_degree;
      f_meta >> n_vertices >> n_edges >> vid_size >> max_degree;
      assert(sizeof(graph_node_t) == vid_size);
      f_meta.close();

      g.nnodes = n_vertices;
      g.nedges = n_edges;
      read_subfile(filename + ".vertex.bin", g.rowptr, n_vertices + 1);
      read_subfile(filename + ".edge.bin", g.colidx, n_edges);

      int* lb = new int[n_vertices];
      memset(lb, 1, n_vertices * sizeof(int));
      g.vertex_label = new bitarray32[n_vertices];
      if(LABELED) {
        read_subfile(filename + ".label.bin", lb, n_vertices);
      }
      for (int i = 0; i < n_vertices; i++) {
        g.vertex_label[i] = (1 << lb[i]);
      }
      delete[] lb;
    }

    void read_gthinker_bin_file(const std::string& filename) {
      FILE* file_in = fopen(filename.c_str(), "rb");
      if (file_in == nullptr) {
        throw std::runtime_error("failed to open G-Thinker CSR bin file: " + filename);
      }

      size_t uintV_size = 0;
      size_t uintE_size = 0;
      size_t vertex_count = 0;
      size_t edge_count = 0;

      size_t res = 0;
      res += fread(&uintV_size, sizeof(size_t), 1, file_in);
      res += fread(&uintE_size, sizeof(size_t), 1, file_in);
      res += fread(&vertex_count, sizeof(size_t), 1, file_in);
      res += fread(&edge_count, sizeof(size_t), 1, file_in);

      if (res != 4) {
        fclose(file_in);
        throw std::runtime_error("invalid G-Thinker CSR bin header: " + filename);
      }
      if (uintV_size != sizeof(graph_node_t)) {
        fclose(file_in);
        throw std::runtime_error("G-Thinker vertex id size does not match STMatch graph_node_t");
      }
      if (uintE_size != sizeof(graph_edge_t)) {
        fclose(file_in);
        throw std::runtime_error("G-Thinker edge offset size does not match STMatch graph_edge_t");
      }
      if (vertex_count > static_cast<size_t>(std::numeric_limits<graph_node_t>::max())) {
        fclose(file_in);
        throw std::runtime_error("G-Thinker graph has too many vertices for STMatch graph_node_t");
      }
      if (edge_count > static_cast<size_t>(std::numeric_limits<graph_edge_t>::max())) {
        fclose(file_in);
        throw std::runtime_error("G-Thinker graph has too many edges for STMatch graph_edge_t");
      }

      g.nnodes = static_cast<graph_node_t>(vertex_count);
      g.nedges = static_cast<graph_edge_t>(edge_count);
      g.rowptr = new graph_edge_t[g.nnodes + 1];
      g.colidx = new graph_node_t[g.nedges];
      g.vertex_label = new bitarray32[g.nnodes];

      res += fread(g.rowptr, sizeof(graph_edge_t), vertex_count + 1, file_in);
      res += fread(g.colidx, sizeof(graph_node_t), edge_count, file_in);
      fclose(file_in);

      if (res != 4 + vertex_count + 1 + edge_count) {
        throw std::runtime_error("failed to read complete G-Thinker CSR bin file: " + filename);
      }

      for (graph_node_t i = 0; i < g.nnodes; ++i) {
        g.vertex_label[i] = (1 << 1);
      }
    }

  };
}
