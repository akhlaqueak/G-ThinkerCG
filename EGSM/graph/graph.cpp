#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <unordered_map>

#include "graph/graph.h"
#include "graph/gmatch_compat.h"


Graph::Graph(std::string path, std::unordered_map<uint32_t, uint32_t>& label_map)
{
    InitEmpty();
    if (IsGThinkerBinPath(path))
        LoadGThinkerBin(path, label_map);
    else
        LoadText(path, label_map);
}

Graph::Graph(uint32_t app_gmatch_query_id, std::unordered_map<uint32_t, uint32_t>& label_map)
{
    InitEmpty();
    LoadAppGMatchQuery(app_gmatch_query_id, label_map);
}

void Graph::InitEmpty()
{
    vcount_ = 0;
    ecount_ = 0;
    vlabels_ = nullptr;
    vdegs_ = nullptr;
    lcount_ = 0u;
    lfreq_max_ = 0u;
    deg_max_ = 0u;
    offsets_ = nullptr;
    neighbors_ = nullptr;
    vlabel_freq_.clear();
    elabel_freq_.clear();
}

bool Graph::IsGThinkerBinPath(const std::string& path) const
{
    const std::string suffix = ".bin";
    if (path.size() < suffix.size() ||
        path.compare(path.size() - suffix.size(), suffix.size(), suffix) != 0)
        return false;

    std::ifstream fin(path, std::ios::binary);
    return fin.good();
}

void Graph::LoadText(std::string path, std::unordered_map<uint32_t, uint32_t>& label_map)
{
    std::ifstream ifs(path);
    if(ifs.fail())
    {
        std::cout << "File not exist!\n";
        exit(-1);
    }

    // true for the query graph, false for the data graph
    bool set_label_map = label_map.empty();

    char type;
    ifs >> type >> vcount_ >> ecount_;
    if (set_label_map && (vcount_ > 16 || ecount_ > 32))
    {
        std::cout << "The query graph should have at most 16 vertices and 32 edges.\n";
        exit(-1);
    }

    vlabels_ = new uint32_t[vcount_];
    vdegs_ = new uint32_t[vcount_];
    offsets_ = new uint32_t[vcount_ + 1];
    offsets_[0] = 0u;
    neighbors_ = new uint32_t[ecount_ * 2 + 1];

    uint32_t* neighbors_offset = new uint32_t[vcount_]();

    while (ifs >> type)
    {
        if (type == 'v')
        {
            uint32_t vertex_id, degree;
            uint32_t label;
            ifs >> vertex_id >> label >> degree;
            if (set_label_map)
            {
                if (label_map.find(label) == label_map.end())
                {
                    uint32_t label_map_size = label_map.size();
                    label_map[label] = label_map_size;
                }
                label = label_map.at(label);
            }
            else
            {
                if (label_map.find(label) == label_map.end())
                {
                    label = label_map.size();
                }
                else
                {
                    label = label_map.at(label);
                }
            }

            vlabels_[vertex_id] = label;
            if (label < label_map.size())
            {
                if (vlabel_freq_.find(label) == vlabel_freq_.end())
                {
                    vlabel_freq_[label] = 0;
                }
                vlabel_freq_[label] += 1;
            }
            offsets_[vertex_id + 1] = offsets_[vertex_id] + degree;

            vdegs_[vertex_id] = degree;
            if (degree > deg_max_)
            {
                deg_max_ = degree;
            }
        }
        else
        {
            uint32_t from_id, to_id;
            ifs >> from_id >> to_id;
            
            uint32_t offset = offsets_[from_id] + neighbors_offset[from_id];
            neighbors_[offset] = to_id;

            offset = offsets_[to_id] + neighbors_offset[to_id];
            neighbors_[offset] = from_id;

            neighbors_offset[from_id]++;
            neighbors_offset[to_id]++;

            if (elabel_freq_.find({vlabels_[from_id], vlabels_[to_id]}) == elabel_freq_.end())
            {
                elabel_freq_[{vlabels_[from_id], vlabels_[to_id]}] = 0;
            }
            elabel_freq_[{vlabels_[from_id], vlabels_[to_id]}] ++;
            if (elabel_freq_.find({vlabels_[to_id], vlabels_[from_id]}) == elabel_freq_.end())
            {
                elabel_freq_[{vlabels_[to_id], vlabels_[from_id]}] = 0;
            }
            elabel_freq_[{vlabels_[to_id], vlabels_[from_id]}] ++;
        }
    }
    ifs.close();

    lcount_ = std::max_element(vlabel_freq_.begin(), vlabel_freq_.end(), 
        [](const auto& v1, const auto& v2)
        {
            return v1.first < v2.first;
        }
    )->first + 1;
    lfreq_max_ = std::max_element(vlabel_freq_.begin(), vlabel_freq_.end(), 
        [](const auto& v1, const auto& v2)
        {
            return v1.second < v2.second;
        }
    )->second;

    delete[] neighbors_offset;
}

void Graph::LoadGThinkerBin(const std::string& path, std::unordered_map<uint32_t, uint32_t>& label_map)
{
    FILE* file_in = fopen(path.c_str(), "rb");
    if (file_in == nullptr)
        throw std::runtime_error("failed to open G-Thinker CSR bin file: " + path);

    size_t uintV_size = 0;
    size_t uintE_size = 0;
    size_t vertex_count = 0;
    size_t directed_edge_count = 0;

    size_t res = 0;
    res += fread(&uintV_size, sizeof(size_t), 1, file_in);
    res += fread(&uintE_size, sizeof(size_t), 1, file_in);
    res += fread(&vertex_count, sizeof(size_t), 1, file_in);
    res += fread(&directed_edge_count, sizeof(size_t), 1, file_in);

    if (res != 4)
    {
        fclose(file_in);
        throw std::runtime_error("invalid G-Thinker CSR bin header: " + path);
    }
    if (uintV_size != sizeof(uint32_t) || uintE_size != sizeof(uint64_t))
    {
        fclose(file_in);
        throw std::runtime_error("G-Thinker CSR bin type sizes do not match EGSM loader");
    }
    if (vertex_count > std::numeric_limits<uint32_t>::max() ||
        directed_edge_count > std::numeric_limits<uint32_t>::max())
    {
        fclose(file_in);
        throw std::runtime_error("G-Thinker graph is too large for EGSM uint32_t graph storage");
    }

    auto* rowptr64 = new uint64_t[vertex_count + 1];
    auto* cols = new uint32_t[directed_edge_count];
    res += fread(rowptr64, sizeof(uint64_t), vertex_count + 1, file_in);
    res += fread(cols, sizeof(uint32_t), directed_edge_count, file_in);
    fclose(file_in);

    if (res != 4 + vertex_count + 1 + directed_edge_count)
    {
        delete[] rowptr64;
        delete[] cols;
        throw std::runtime_error("failed to read complete G-Thinker CSR bin file: " + path);
    }

    vcount_ = static_cast<uint32_t>(vertex_count);
    ecount_ = static_cast<uint32_t>(directed_edge_count / 2);
    vlabels_ = new uint32_t[vcount_];
    vdegs_ = new uint32_t[vcount_];
    offsets_ = new uint32_t[vcount_ + 1];
    neighbors_ = cols;

    for (uint32_t i = 0; i <= vcount_; ++i)
    {
        if (rowptr64[i] > std::numeric_limits<uint32_t>::max())
        {
            delete[] rowptr64;
            throw std::runtime_error("G-Thinker graph offset exceeds EGSM uint32_t graph storage");
        }
        offsets_[i] = static_cast<uint32_t>(rowptr64[i]);
    }
    delete[] rowptr64;

    for (uint32_t i = 0; i < vcount_; ++i)
    {
        vlabels_[i] = 0;
        vdegs_[i] = offsets_[i + 1] - offsets_[i];
        deg_max_ = std::max(deg_max_, vdegs_[i]);
    }

    if (label_map.empty())
        label_map[0] = 0;
    vlabel_freq_[0] = vcount_;
    lcount_ = 1;
    lfreq_max_ = vcount_;
    elabel_freq_[{0, 0}] = static_cast<uint32_t>(directed_edge_count);
}

void Graph::LoadAppGMatchQuery(uint32_t query_id, std::unordered_map<uint32_t, uint32_t>& label_map)
{
    auto adj = gmatch_compat::make_query(static_cast<int>(query_id));
    vcount_ = static_cast<uint32_t>(adj.size());
    uint64_t directed_edges = 0;
    for (const auto& neighbors : adj)
        directed_edges += neighbors.size();

    if (directed_edges > std::numeric_limits<uint32_t>::max())
        throw std::runtime_error("app_gmatch query is too large for EGSM");

    ecount_ = static_cast<uint32_t>(directed_edges / 2);
    vlabels_ = new uint32_t[vcount_];
    vdegs_ = new uint32_t[vcount_];
    offsets_ = new uint32_t[vcount_ + 1];
    neighbors_ = new uint32_t[directed_edges + 1];

    if (label_map.empty())
        label_map[0] = 0;

    offsets_[0] = 0u;
    for (uint32_t i = 0; i < vcount_; ++i)
    {
        vlabels_[i] = 0;
        vdegs_[i] = static_cast<uint32_t>(adj[i].size());
        deg_max_ = std::max(deg_max_, vdegs_[i]);
        offsets_[i + 1] = offsets_[i] + vdegs_[i];
        std::copy(adj[i].begin(), adj[i].end(), neighbors_ + offsets_[i]);
    }

    vlabel_freq_[0] = vcount_;
    lcount_ = 1;
    lfreq_max_ = vcount_;
    elabel_freq_[{0, 0}] = static_cast<uint32_t>(directed_edges);
}

Graph::~Graph()
{
    delete[] vlabels_;
    delete[] vdegs_;
    delete[] offsets_;
    delete[] neighbors_;
}


void GraphUtils::Set(const Graph& g)
{
    std::fill(eidx_, eidx_ + g.vcount_ * g.vcount_, UINT8_MAX);
    uint8_t edge_pos = 0u;

    for (uint32_t u = 0u; u < g.vcount_; u++)
    {
        for (uint32_t offset = g.offsets_[u]; offset < g.offsets_[u + 1]; offset ++)
        {
            const uint32_t u_other = g.neighbors_[offset];

            uint32_t key = u * (g.vcount_) + u_other;
            eidx_[key] = edge_pos;
            edge_pos++;
        }
    }

    for (uint32_t i = 0; i < g.vcount_; i++)
    {
        nbrbits_[i] = 0u;
        uint8_t *based = eidx_ + i * g.vcount_;
        for (uint32_t j = 0; j < g.vcount_; j++)
        {
            if (based[j] != UINT8_MAX)
            {
                nbrbits_[i] |= (1 << j);
            }
        }
    }
}
