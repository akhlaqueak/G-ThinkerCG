#pragma once

namespace STMatch {

  typedef int graph_node_t;
  typedef long graph_edge_t;
  typedef char pattern_node_t;
  typedef char set_op_t;
  typedef unsigned int bitarray32;

  inline constexpr size_t PAT_SIZE = 7;
#ifndef STMATCH_GRAPH_DEGREE
#define STMATCH_GRAPH_DEGREE 16384
#endif

  inline constexpr size_t GRAPH_DEGREE = STMATCH_GRAPH_DEGREE;
  inline constexpr size_t MAX_SLOT_NUM = 15;

#include "config_for_ae/fig_local_global_unroll.h" 

#ifndef STMATCH_GRID_DIM
#define STMATCH_GRID_DIM 16
#endif

#ifndef STMATCH_BLOCK_DIM
#define STMATCH_BLOCK_DIM 128
#endif

  inline constexpr int GRID_DIM = STMATCH_GRID_DIM;
  inline constexpr int BLOCK_DIM = STMATCH_BLOCK_DIM;
  inline constexpr int WARP_SIZE = 32;
  inline constexpr int NWARPS_PER_BLOCK = (BLOCK_DIM / WARP_SIZE);
  inline constexpr int NWARPS_TOTAL = ((GRID_DIM * BLOCK_DIM + WARP_SIZE - 1) / WARP_SIZE);

  inline constexpr graph_node_t JOB_CHUNK_SIZE = 8;
  //static_assert(2 * JOB_CHUNK_SIZE <= GRAPH_DEGREE); 

  // this is the maximum unroll size

  inline constexpr int DETECT_LEVEL = 1;
  inline constexpr int STOP_LEVEL = 2;
}
