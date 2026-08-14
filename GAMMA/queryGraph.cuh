#include<iostream>
#include<stdlib.h>
#include<vector>
#include "utils.h"
#include<fstream>
#include<string.h>
#include "gmatch_compat.h"
using namespace std;
class queryGraph{
public:
	uint32_t vertex_num = 0;
	vector<uint8_t> vertex_label;
	vector<vector<uint32_t>> adj_list;
	vector<vector<uint32_t>> order_list;
	vector<uint32_t> core;
	vector<uint32_t> satellite;
	void readFromFile(const char filename []) {
		std::ifstream file_read;
		file_read.open(filename, std::ios::in);
		file_read >> vertex_num;
		//cout << vertex_num << endl;
		for (uint32_t i = 0; i < vertex_num; i ++) {
			uint32_t label_now;
			file_read >> label_now;
			//cout << label_now << " ";
			vertex_label.push_back((uint8_t)label_now);
		}
		//cout << endl;
		for (uint32_t i = 0; i < vertex_num; i ++) {
			vector<uint32_t> v;
			uint32_t nbr_size;
			file_read >> nbr_size;
			//cout << nbr_size << " ";
			uint32_t nbr_now;
			for (uint32_t j = 0; j < nbr_size; j ++) {
				file_read >> nbr_now;
				//cout << nbr_now << " ";
				v.push_back(nbr_now);
			}
			adj_list.push_back(v);
			//cout << endl;
		}
		for (uint32_t i = 0; i < vertex_num; i ++) {
			vector<uint32_t> v;
			uint32_t nbr_size;
			file_read >> nbr_size;
			//cout << nbr_size << " ";
			uint32_t order_nbr_now;
			for (uint32_t j = 0; j < nbr_size; j ++) {
				file_read >> order_nbr_now;
				//cout << order_nbr_now << " ";
				v.push_back(order_nbr_now);
			}
			order_list.push_back(v);
			//cout << endl;
		}
		file_read.close();
		for (uint32_t i = 0; i < vertex_num; i ++) {
			if (adj_list[i].size() > 1) {
				core.push_back(i);
			}
			else {
				satellite.push_back(i);
			}
		}
		return ;
	}
	void loadFromGMatchPreset(int query_id) {
		clear();
		auto adj = gmatch_compat::make_query(query_id);
		if (adj.size() > embedding_max_length) {
			fprintf(stderr, "query has %lu vertices, but GAMMA embedding_max_length is %d\n",
					static_cast<unsigned long>(adj.size()), embedding_max_length);
			exit(1);
		}
		vertex_num = static_cast<uint32_t>(adj.size());
		vertex_label.assign(vertex_num, 0);
		adj_list = adj;
		order_list.assign(vertex_num, vector<uint32_t>());
		vector<pair<uint32_t, uint32_t>> conditions = gmatch_compat::symmetry_breaking_conditions(adj);
		for (size_t i = 0; i < conditions.size(); ++i) {
			order_list[conditions[i].second].push_back(conditions[i].first);
		}
		for (uint32_t i = 0; i < vertex_num; ++i) {
			core.push_back(i);
		}
	}
	void clear() {
		for (uint32_t i = 0; i < vertex_num; i ++) {
			adj_list[i].clear();
			order_list[i].clear();
		}
		adj_list.clear();
		order_list.clear();
		vertex_label.clear();
		core.clear();
		satellite.clear();
		vertex_num = 0;
		return ;
	}
};
