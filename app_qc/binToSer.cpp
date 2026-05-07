#include <algorithm>
#include <cstdint>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace std;

namespace
{
    template <typename T>
    void read_value(ifstream& in, T& value, const string& description)
    {
        in.read(reinterpret_cast<char*>(&value), sizeof(T));
        if (!in) {
            throw runtime_error("Failed to read " + description);
        }
    }

    template <typename T>
    void read_array(ifstream& in, vector<T>& values, const string& description)
    {
        if (values.empty()) {
            return;
        }

        const uint64_t bytes = static_cast<uint64_t>(values.size()) * sizeof(T);
        if (bytes > static_cast<uint64_t>(numeric_limits<streamsize>::max())) {
            throw length_error(description + " is too large to read in one block");
        }

        in.read(reinterpret_cast<char*>(values.data()), static_cast<streamsize>(bytes));
        if (!in) {
            throw runtime_error("Failed to read " + description);
        }
    }

    template <typename T>
    void write_value(ofstream& out, const T& value, const string& description)
    {
        out.write(reinterpret_cast<const char*>(&value), sizeof(T));
        if (!out) {
            throw runtime_error("Failed to write " + description);
        }
    }

    template <typename T>
    void write_array(ofstream& out, const vector<T>& values, const string& description)
    {
        if (values.empty()) {
            return;
        }

        const uint64_t bytes = static_cast<uint64_t>(values.size()) * sizeof(T);
        if (bytes > static_cast<uint64_t>(numeric_limits<streamsize>::max())) {
            throw length_error(description + " is too large to write in one block");
        }

        out.write(reinterpret_cast<const char*>(values.data()), static_cast<streamsize>(bytes));
        if (!out) {
            throw runtime_error("Failed to write " + description);
        }
    }

    size_t checked_size(uint64_t value, const string& description)
    {
        if (value > static_cast<uint64_t>(numeric_limits<size_t>::max())) {
            throw length_error(description + " does not fit in size_t");
        }
        return static_cast<size_t>(value);
    }

    string vertex_error(int vertex, uint64_t position, int neighbor, int vertex_count)
    {
        ostringstream message;
        message << "Invalid neighbor id " << neighbor << " at vertex " << vertex
                << ", adjacency position " << position << ". Expected 0 <= id < "
                << vertex_count << ".";
        return message.str();
    }
}

class CPU_Graph
{
public:
    int number_of_vertices = 0;
    uint64_t number_of_edges = 0;
    uint64_t number_of_lvl2adj = 0;

    vector<int> onehop_neighbors;
    vector<uint64_t> onehop_offsets;
    vector<int> twohop_neighbors;
    vector<uint64_t> twohop_offsets;

    explicit CPU_Graph(const char* input_file)
    {
        read_original_binary(input_file);
        build_twohop_adjacency();
    }

    void write_binary(const char* output_file) const
    {
        ofstream file_out(output_file, ios::binary);
        if (!file_out) {
            throw runtime_error(string("Failed to open output file: ") + output_file);
        }

        write_value(file_out, number_of_vertices, "vertex count");
        write_value(file_out, number_of_edges, "edge count");
        write_value(file_out, number_of_lvl2adj, "two-hop adjacency count");
        write_array(file_out, onehop_neighbors, "one-hop neighbors");
        write_array(file_out, onehop_offsets, "one-hop offsets");
        write_array(file_out, twohop_neighbors, "two-hop neighbors");
        write_array(file_out, twohop_offsets, "two-hop offsets");
    }

private:
    void read_original_binary(const char* input_file)
    {
        ifstream file_in(input_file, ios::binary);
        if (!file_in) {
            throw runtime_error(string("Failed to open input file: ") + input_file);
        }

        size_t uintV_size = 0;
        size_t uintE_size = 0;
        size_t vertex_count = 0;
        size_t edge_count = 0;

        read_value(file_in, uintV_size, "vertex-id type size");
        read_value(file_in, uintE_size, "edge-count type size");
        read_value(file_in, vertex_count, "vertex count");
        read_value(file_in, edge_count, "edge count");

        if (uintV_size != sizeof(int)) {
            throw runtime_error("Unsupported vertex-id type size in input graph");
        }
        if (uintE_size != sizeof(uint64_t)) {
            throw runtime_error("Unsupported edge-count type size in input graph");
        }
        if (vertex_count == 0 || vertex_count > static_cast<size_t>(numeric_limits<int>::max())) {
            throw length_error("Vertex count must fit in a positive int");
        }
        if (edge_count > onehop_neighbors.max_size()) {
            throw length_error("Edge count is too large for this process");
        }

        number_of_vertices = static_cast<int>(vertex_count);
        number_of_edges = static_cast<uint64_t>(edge_count);

        cout << "|V|: " << number_of_vertices << endl;
        cout << "|E|: " << number_of_edges << endl;

        onehop_offsets.resize(static_cast<size_t>(number_of_vertices) + 1);
        onehop_neighbors.resize(edge_count);

        read_array(file_in, onehop_offsets, "one-hop offsets");
        read_array(file_in, onehop_neighbors, "one-hop neighbors");

        if (file_in.peek() != char_traits<char>::eof()) {
            throw runtime_error("Unexpected trailing bytes in input graph");
        }

        validate_onehop_graph();
    }

    void validate_onehop_graph() const
    {
        if (onehop_offsets.front() != 0) {
            throw runtime_error("First one-hop offset must be 0");
        }
        if (onehop_offsets.back() != number_of_edges) {
            throw runtime_error("Last one-hop offset must equal edge count");
        }

        for (int vertex = 0; vertex < number_of_vertices; vertex++) {
            const uint64_t begin = onehop_offsets[static_cast<size_t>(vertex)];
            const uint64_t end = onehop_offsets[static_cast<size_t>(vertex) + 1];
            if (begin > end) {
                ostringstream message;
                message << "One-hop offsets are not monotonic at vertex " << vertex;
                throw runtime_error(message.str());
            }
            if (end > number_of_edges) {
                ostringstream message;
                message << "One-hop offset for vertex " << vertex << " exceeds edge count";
                throw runtime_error(message.str());
            }

            for (uint64_t pos = begin; pos < end; pos++) {
                const int neighbor = onehop_neighbors[checked_size(pos, "one-hop index")];
                if (neighbor < 0 || neighbor >= number_of_vertices) {
                    throw out_of_range(vertex_error(vertex, pos, neighbor, number_of_vertices));
                }
            }
        }
    }

    void build_twohop_adjacency()
    {
        twohop_offsets.assign(static_cast<size_t>(number_of_vertices) + 1, 0);
        vector<int> seen(static_cast<size_t>(number_of_vertices), -1);

        const uint64_t reserve_hint = min<uint64_t>(number_of_edges, 1000000ULL);
        twohop_neighbors.reserve(checked_size(reserve_hint, "two-hop reserve hint"));

        for (int vertex = 0; vertex < number_of_vertices; vertex++) {
            const size_t row_start = twohop_neighbors.size();

            auto add_if_new = [&](int candidate) {
                if (seen[static_cast<size_t>(candidate)] != vertex) {
                    seen[static_cast<size_t>(candidate)] = vertex;
                    twohop_neighbors.push_back(candidate);
                }
            };

            const uint64_t begin = onehop_offsets[static_cast<size_t>(vertex)];
            const uint64_t end = onehop_offsets[static_cast<size_t>(vertex) + 1];

            for (uint64_t pos = begin; pos < end; pos++) {
                const int lvl1adj = onehop_neighbors[checked_size(pos, "one-hop index")];
                add_if_new(lvl1adj);

                const uint64_t lvl2_begin = onehop_offsets[static_cast<size_t>(lvl1adj)];
                const uint64_t lvl2_end = onehop_offsets[static_cast<size_t>(lvl1adj) + 1];
                for (uint64_t lvl2_pos = lvl2_begin; lvl2_pos < lvl2_end; lvl2_pos++) {
                    const int lvl2adj = onehop_neighbors[checked_size(lvl2_pos, "two-hop source index")];
                    if (lvl2adj != vertex) {
                        add_if_new(lvl2adj);
                    }
                }
            }

            sort(twohop_neighbors.begin() + static_cast<vector<int>::difference_type>(row_start),
                 twohop_neighbors.end());

            sort(onehop_neighbors.begin() + static_cast<vector<int>::difference_type>(begin),
                 onehop_neighbors.begin() + static_cast<vector<int>::difference_type>(end));

            twohop_offsets[static_cast<size_t>(vertex) + 1] =
                static_cast<uint64_t>(twohop_neighbors.size());
        }

        number_of_lvl2adj = static_cast<uint64_t>(twohop_neighbors.size());
        cout << "|2-hop|: " << number_of_lvl2adj << endl;
    }
};

int main(int argc, char* argv[])
{
    if (argc != 3) {
        cerr << "Usage: ./binToSer <graph_file> <output_file>" << endl;
        return 1;
    }

    try {
        CPU_Graph hg(argv[1]);
        hg.write_binary(argv[2]);
    }
    catch (const exception& error) {
        cerr << "binToSer: " << error.what() << endl;
        return 1;
    }

    return 0;
}
