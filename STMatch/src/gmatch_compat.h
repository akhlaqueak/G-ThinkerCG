#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <queue>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "config.h"

namespace STMatch {

  enum class GMatchPresetPattern {
    P0,
    P1,
    P2,
    P3,
    P4,
    P5,
    P6,
    P7,
    P8,
    P9,
    P10,
    P11,
    P12,
    P13,
    P14,
    P15,
    P16,
    P17,
    P18,
    P19,
    P20,
    P21,
    P22,
    P23,
    P24,
    P25,
    P26,
    P27
  };

  inline std::vector<std::vector<int>> select_gmatch_preset_pattern(int pattern_id) {
    std::vector<std::vector<int>> conn;
    switch (static_cast<GMatchPresetPattern>(pattern_id)) {
    case GMatchPresetPattern::P0:
      conn = {{1, 2}, {0, 2}, {0, 1}};
      break;
    case GMatchPresetPattern::P1:
      conn = {{1, 3}, {0, 2}, {1, 3}, {0, 2}};
      break;
    case GMatchPresetPattern::P2:
      conn = {{1, 2, 3}, {0, 2}, {0, 1, 3}, {0, 2}};
      break;
    case GMatchPresetPattern::P3:
      conn = {{1, 2}, {0, 2}, {0, 1, 3}, {2, 4}, {3}};
      break;
    case GMatchPresetPattern::P4:
      conn = {{1, 2}, {0, 2, 3}, {0, 1, 4}, {1, 4}, {2, 3}};
      break;
    case GMatchPresetPattern::P5:
      conn = {{1, 2, 3, 4}, {0, 2, 3}, {0, 1, 3}, {0, 1, 2, 4}, {0, 3}};
      break;
    case GMatchPresetPattern::P6:
      conn = {{1, 2, 3}, {0, 2, 3}, {0, 1, 4}, {0, 1, 4}, {2, 3}};
      break;
    case GMatchPresetPattern::P7:
      conn.resize(5);
      for (int i = 1; i <= 4; ++i) {
        conn[0].push_back(i);
        conn[i].push_back(0);
      }
      conn[1].push_back(3);
      conn[3].push_back(1);
      conn[1].push_back(2);
      conn[2].push_back(1);
      conn[2].push_back(4);
      conn[4].push_back(2);
      break;
    case GMatchPresetPattern::P8:
      conn = {{1, 2, 3, 4}, {0, 2, 4}, {0, 1, 3}, {0, 2, 4}, {0, 1, 3}};
      break;
    case GMatchPresetPattern::P9:
      conn = {{1, 2, 3, 4}, {0, 2, 3, 4}, {0, 1, 3}, {0, 1, 2, 4}, {0, 1, 3}};
      break;
    case GMatchPresetPattern::P10:
      conn = {{1, 2, 3, 4, 5}, {0, 2}, {0, 1, 3}, {0, 2, 4}, {0, 3, 5}, {0, 4}};
      break;
    case GMatchPresetPattern::P11:
      conn = {{1, 2, 3, 5}, {0, 2, 3, 4}, {0, 1, 4, 5}, {0, 1}, {1, 2}, {0, 2}};
      break;
    case GMatchPresetPattern::P12:
      conn = {{1, 2}, {0, 2, 3, 4, 5}, {0, 1, 3, 4, 5}, {1, 2, 4}, {1, 2, 3, 5}, {1, 2, 4}};
      break;
    case GMatchPresetPattern::P13:
      conn = {{1, 2}, {0, 3}, {0, 3, 4, 5}, {1, 2, 4, 5}, {2, 3}, {2, 3}};
      break;
    case GMatchPresetPattern::P14:
      conn = {{1, 2, 3, 4, 5}, {0, 2, 3, 5}, {0, 1, 3, 5}, {0, 1, 2, 4, 5, 6}, {0, 3, 5}, {0, 1, 2, 3, 4, 6}, {3, 5}};
      break;
    case GMatchPresetPattern::P15:
      conn = {{1, 2, 3, 4}, {0, 2, 3, 4}, {0, 1, 3, 4}, {0, 1, 2, 4, 5, 6}, {0, 1, 2, 3, 5, 6}, {3, 4}, {3, 4}};
      break;
    case GMatchPresetPattern::P16:
      conn = {{1, 2}, {0, 3}, {0, 4}, {1, 4}, {2, 3}};
      break;
    case GMatchPresetPattern::P17:
      conn = {{1, 2}, {0, 3}, {0, 4}, {1, 5}, {2, 5}, {3, 4}};
      break;
    case GMatchPresetPattern::P18:
      conn = {{1, 2, 4}, {0, 2, 5}, {0, 1, 3}, {2, 4, 5}, {0, 3, 5}, {1, 3, 4}};
      break;
    case GMatchPresetPattern::P23:
      conn = {{1, 2}, {0, 2}, {0, 1}};
      break;
    case GMatchPresetPattern::P24:
      conn = {{1, 2, 3}, {0, 2, 3}, {0, 1, 3}, {0, 1, 2}};
      break;
    case GMatchPresetPattern::P25:
      conn = {{1, 2, 3, 4}, {0, 2, 3, 4}, {0, 1, 3, 4}, {0, 1, 2, 4}, {0, 1, 2, 3}};
      break;
    case GMatchPresetPattern::P26:
      conn = {{1, 2, 3, 4, 5}, {0, 2, 3, 4, 5}, {0, 1, 3, 4, 5}, {0, 1, 2, 4, 5}, {0, 1, 2, 3, 5}, {0, 1, 2, 3, 4}};
      break;
    case GMatchPresetPattern::P27:
      conn = {{1, 2, 3, 4, 5, 6}, {0, 2, 3, 4, 5, 6}, {0, 1, 3, 4, 5, 6}, {0, 1, 2, 4, 5, 6}, {0, 1, 2, 3, 5, 6}, {0, 1, 2, 3, 4, 6}, {0, 1, 2, 3, 4, 5}};
      break;
    default:
      throw std::invalid_argument("unsupported app_gmatch preset query id");
    }
    return conn;
  }

  inline void remap_like_gmatch(std::vector<std::vector<int>>& adj) {
    const int n = static_cast<int>(adj.size());
    if (n == 0) return;

    int root = 0;
    size_t max_degree = 0;
    for (int i = 0; i < n; ++i) {
      if (adj[i].size() > max_degree) {
        max_degree = adj[i].size();
        root = i;
      }
    }

    std::queue<int> queue;
    std::vector<bool> visited(n, false);
    std::vector<int> old_to_new(n, 0);
    std::vector<int> new_to_old(n, 0);
    int new_vid = 0;

    queue.push(root);
    visited[root] = true;

    while (!queue.empty()) {
      const size_t level_size = queue.size();
      std::vector<int> same_level_vertices;
      same_level_vertices.reserve(level_size);

      for (size_t i = 0; i < level_size; ++i) {
        int front = queue.front();
        queue.pop();
        same_level_vertices.push_back(front);

        for (int ne : adj[front]) {
          if (!visited[ne]) {
            visited[ne] = true;
            queue.push(ne);
          }
        }
      }

      std::vector<std::tuple<size_t, size_t, int>> weights;
      weights.reserve(same_level_vertices.size());
      for (int v : same_level_vertices) {
        size_t connections = 0;
        for (int ne : adj[v]) {
          if (visited[ne]) connections++;
        }
        weights.emplace_back(adj[v].size(), connections, v);
      }
      std::sort(weights.begin(), weights.end(), [](const auto& a, const auto& b) {
        if (std::get<0>(a) != std::get<0>(b))
          return std::get<0>(a) > std::get<0>(b);
        if (std::get<1>(a) != std::get<1>(b))
          return std::get<1>(a) > std::get<1>(b);
        return std::get<2>(a) < std::get<2>(b);
      });

      for (const auto& w : weights) {
        old_to_new[std::get<2>(w)] = new_vid;
        new_to_old[new_vid] = std::get<2>(w);
        ++new_vid;
      }
    }

    assert(new_vid == n);
    std::vector<std::vector<int>> remapped(n);
    for (int old_v = 0; old_v < n; ++old_v) {
      int new_v = old_to_new[old_v];
      for (int old_ne : adj[old_v]) {
        remapped[new_v].push_back(old_to_new[old_ne]);
      }
    }

    for (auto& neighbors : remapped) {
      std::sort(neighbors.begin(), neighbors.end());
      neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());
    }
    adj.swap(remapped);
  }

  inline std::vector<std::vector<int>> make_gmatch_query_adjacency(int pattern_id) {
    auto adj = select_gmatch_preset_pattern(pattern_id);
    if (adj.size() > PAT_SIZE) {
      throw std::invalid_argument("app_gmatch preset query exceeds STMatch PAT_SIZE");
    }
    remap_like_gmatch(adj);
    return adj;
  }

}
