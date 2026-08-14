#pragma once

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstddef>
#include <map>
#include <queue>
#include <set>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace gmatch_compat {

inline std::vector<std::vector<uint32_t>> select_preset_pattern(int pattern_id) {
	std::vector<std::vector<uint32_t>> conn;
	switch (pattern_id) {
	case 0:
		conn = {{1, 2}, {0, 2}, {0, 1}};
		break;
	case 1:
		conn = {{1, 3}, {0, 2}, {1, 3}, {0, 2}};
		break;
	case 2:
		conn = {{1, 2, 3}, {0, 2}, {0, 1, 3}, {0, 2}};
		break;
	case 3:
		conn = {{1, 2}, {0, 2}, {0, 1, 3}, {2, 4}, {3}};
		break;
	case 4:
		conn = {{1, 2}, {0, 2, 3}, {0, 1, 4}, {1, 4}, {2, 3}};
		break;
	case 5:
		conn = {{1, 2, 3, 4}, {0, 2, 3}, {0, 1, 3}, {0, 1, 2, 4}, {0, 3}};
		break;
	case 6:
		conn = {{1, 2, 3}, {0, 2, 3}, {0, 1, 4}, {0, 1, 4}, {2, 3}};
		break;
	case 7:
		conn.resize(5);
		for (uint32_t i = 1; i <= 4; ++i) {
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
	case 8:
		conn = {{1, 2, 3, 4}, {0, 2, 4}, {0, 1, 3}, {0, 2, 4}, {0, 1, 3}};
		break;
	case 9:
		conn = {{1, 2, 3, 4}, {0, 2, 3, 4}, {0, 1, 3}, {0, 1, 2, 4}, {0, 1, 3}};
		break;
	case 10:
		conn = {{1, 2, 3, 4, 5}, {0, 2}, {0, 1, 3}, {0, 2, 4}, {0, 3, 5}, {0, 4}};
		break;
	case 11:
		conn = {{1, 2, 3, 5}, {0, 2, 3, 4}, {0, 1, 4, 5}, {0, 1}, {1, 2}, {0, 2}};
		break;
	case 12:
		conn = {{1, 2}, {0, 2, 3, 4, 5}, {0, 1, 3, 4, 5}, {1, 2, 4}, {1, 2, 3, 5}, {1, 2, 4}};
		break;
	case 13:
		conn = {{1, 2}, {0, 3}, {0, 3, 4, 5}, {1, 2, 4, 5}, {2, 3}, {2, 3}};
		break;
	case 14:
		conn = {{1, 2, 3, 4, 5}, {0, 2, 3, 5}, {0, 1, 3, 5}, {0, 1, 2, 4, 5, 6}, {0, 3, 5}, {0, 1, 2, 3, 4, 6}, {3, 5}};
		break;
	case 15:
		conn = {{1, 2, 3, 4}, {0, 2, 3, 4}, {0, 1, 3, 4}, {0, 1, 2, 4, 5, 6}, {0, 1, 2, 3, 5, 6}, {3, 4}, {3, 4}};
		break;
	case 16:
		conn = {{1, 2}, {0, 3}, {0, 4}, {1, 4}, {2, 3}};
		break;
	case 17:
		conn = {{1, 2}, {0, 3}, {0, 4}, {1, 5}, {2, 5}, {3, 4}};
		break;
	case 18:
		conn = {{1, 2, 4}, {0, 2, 5}, {0, 1, 3}, {2, 4, 5}, {0, 3, 5}, {1, 3, 4}};
		break;
	case 23:
		conn = {{1, 2}, {0, 2}, {0, 1}};
		break;
	case 24:
		conn = {{1, 2, 3}, {0, 2, 3}, {0, 1, 3}, {0, 1, 2}};
		break;
	case 25:
		conn = {{1, 2, 3, 4}, {0, 2, 3, 4}, {0, 1, 3, 4}, {0, 1, 2, 4}, {0, 1, 2, 3}};
		break;
	case 26:
		conn = {{1, 2, 3, 4, 5}, {0, 2, 3, 4, 5}, {0, 1, 3, 4, 5}, {0, 1, 2, 4, 5}, {0, 1, 2, 3, 5}, {0, 1, 2, 3, 4}};
		break;
	case 27:
		conn = {{1, 2, 3, 4, 5, 6}, {0, 2, 3, 4, 5, 6}, {0, 1, 3, 4, 5, 6}, {0, 1, 2, 4, 5, 6}, {0, 1, 2, 3, 5, 6}, {0, 1, 2, 3, 4, 6}, {0, 1, 2, 3, 4, 5}};
		break;
	default:
		throw std::invalid_argument("unsupported app_gmatch preset query id");
	}
	return conn;
}

inline void remap_like_gmatch(std::vector<std::vector<uint32_t>>& adj) {
	const uint32_t n = static_cast<uint32_t>(adj.size());
	if (n == 0)
		return;

	uint32_t root = 0;
	size_t max_degree = 0;
	for (uint32_t i = 0; i < n; ++i) {
		if (adj[i].size() > max_degree) {
			max_degree = adj[i].size();
			root = i;
		}
	}

	std::queue<uint32_t> queue;
	std::vector<bool> visited(n, false);
	std::vector<uint32_t> old_to_new(n, 0);
	std::vector<uint32_t> new_to_old(n, 0);
	uint32_t new_vid = 0;

	queue.push(root);
	visited[root] = true;
	while (!queue.empty()) {
		const size_t level_size = queue.size();
		std::vector<uint32_t> same_level_vertices;
		same_level_vertices.reserve(level_size);
		for (size_t i = 0; i < level_size; ++i) {
			uint32_t front = queue.front();
			queue.pop();
			same_level_vertices.push_back(front);
			for (uint32_t ne : adj[front]) {
				if (!visited[ne]) {
					visited[ne] = true;
					queue.push(ne);
				}
			}
		}

		std::vector<std::tuple<size_t, size_t, uint32_t>> weights;
		weights.reserve(same_level_vertices.size());
		for (uint32_t v : same_level_vertices) {
			size_t connections = 0;
			for (uint32_t ne : adj[v]) {
				if (visited[ne])
					++connections;
			}
			weights.emplace_back(adj[v].size(), connections, v);
		}
		typedef std::tuple<size_t, size_t, uint32_t> Weight;
		std::sort(weights.begin(), weights.end(), [](const Weight& a, const Weight& b) {
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
	std::vector<std::vector<uint32_t>> remapped(n);
	for (uint32_t old_v = 0; old_v < n; ++old_v) {
		const uint32_t new_v = old_to_new[old_v];
		for (uint32_t old_ne : adj[old_v])
			remapped[new_v].push_back(old_to_new[old_ne]);
	}
	for (auto& neighbors : remapped) {
		std::sort(neighbors.begin(), neighbors.end());
		neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());
	}
	adj.swap(remapped);
}

inline std::vector<std::vector<uint32_t>> make_query(int pattern_id) {
	auto adj = select_preset_pattern(pattern_id);
	remap_like_gmatch(adj);
	return adj;
}

inline uint64_t automorphism_count(const std::vector<std::vector<uint32_t>>& adj) {
	const uint32_t n = static_cast<uint32_t>(adj.size());
	std::vector<std::vector<bool>> matrix(n, std::vector<bool>(n, false));
	for (uint32_t u = 0; u < n; ++u) {
		for (uint32_t v : adj[u])
			matrix[u][v] = true;
	}

	std::vector<uint32_t> perm(n);
	for (uint32_t i = 0; i < n; ++i)
		perm[i] = i;

	uint64_t count = 0;
	do {
		bool valid = true;
		for (uint32_t u = 0; u < n && valid; ++u) {
			for (uint32_t v = 0; v < n; ++v) {
				if (matrix[u][v] != matrix[perm[u]][perm[v]]) {
					valid = false;
					break;
				}
			}
		}
		if (valid)
			++count;
	} while (std::next_permutation(perm.begin(), perm.end()));
	return count;
}

inline uint64_t automorphism_count_for_query(int pattern_id) {
	return automorphism_count(make_query(pattern_id));
}

inline std::vector<std::vector<uint32_t>> automorphisms(const std::vector<std::vector<uint32_t>>& adj) {
	const uint32_t n = static_cast<uint32_t>(adj.size());
	std::vector<std::vector<bool>> matrix(n, std::vector<bool>(n, false));
	for (uint32_t u = 0; u < n; ++u) {
		for (uint32_t v : adj[u])
			matrix[u][v] = true;
	}

	std::vector<uint32_t> perm(n);
	for (uint32_t i = 0; i < n; ++i)
		perm[i] = i;

	std::vector<std::vector<uint32_t>> result;
	do {
		bool valid = true;
		for (uint32_t u = 0; u < n && valid; ++u) {
			for (uint32_t v = 0; v < n; ++v) {
				if (matrix[u][v] != matrix[perm[u]][perm[v]]) {
					valid = false;
					break;
				}
			}
		}
		if (valid)
			result.push_back(perm);
	} while (std::next_permutation(perm.begin(), perm.end()));
	return result;
}

inline std::map<uint32_t, std::set<uint32_t>> equivalence_classes(const std::vector<std::vector<uint32_t>>& aut,
																   uint32_t vertex_count) {
	std::map<uint32_t, std::set<uint32_t>> classes;
	for (uint32_t i = 0; i < vertex_count; ++i) {
		std::set<uint32_t> eclass;
		for (size_t j = 0; j < aut.size(); ++j)
			eclass.insert(aut[j][i]);
		uint32_t rep = *std::min_element(eclass.begin(), eclass.end());
		classes[rep].insert(eclass.begin(), eclass.end());
	}
	return classes;
}

inline std::vector<std::pair<uint32_t, uint32_t>> symmetry_breaking_conditions(
	const std::vector<std::vector<uint32_t>>& adj) {
	const uint32_t vertex_count = static_cast<uint32_t>(adj.size());
	std::vector<std::vector<uint32_t>> aut = automorphisms(adj);
	std::map<uint32_t, std::set<uint32_t>> classes = equivalence_classes(aut, vertex_count);
	std::vector<std::pair<uint32_t, uint32_t>> conditions;

	while (true) {
		std::map<uint32_t, std::set<uint32_t>>::const_iterator eclass_it = classes.end();
		for (std::map<uint32_t, std::set<uint32_t>>::const_iterator it = classes.begin(); it != classes.end(); ++it) {
			if (it->second.size() > 1) {
				eclass_it = it;
				break;
			}
		}
		if (eclass_it == classes.end())
			break;

		const std::set<uint32_t>& eclass = eclass_it->second;
		uint32_t n0 = *eclass.begin();

		for (size_t p = 0; p < aut.size(); ++p) {
			std::set<uint32_t>::const_iterator it = eclass.begin();
			++it;
			uint32_t min_vertex = *it;
			for (; it != eclass.end(); ++it) {
				if (aut[p][*it] < aut[p][min_vertex])
					min_vertex = *it;
			}
			conditions.push_back(std::make_pair(n0, min_vertex));
		}

		std::vector<std::vector<uint32_t>> filtered;
		for (size_t p = 0; p < aut.size(); ++p) {
			if (aut[p][n0] == n0)
				filtered.push_back(aut[p]);
		}
		aut.swap(filtered);
		classes = equivalence_classes(aut, vertex_count);
	}

	std::sort(conditions.begin(), conditions.end());
	conditions.erase(std::unique(conditions.begin(), conditions.end()), conditions.end());
	return conditions;
}

} // namespace gmatch_compat
