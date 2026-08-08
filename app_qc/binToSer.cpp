#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace std;

namespace
{
    namespace fs = filesystem;

    string io_failure(const string& message, int error_number)
    {
        if (error_number == 0) {
            return message;
        }
        return message + ": " + error_code(error_number, generic_category()).message();
    }

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
        errno = 0;
        out.write(reinterpret_cast<const char*>(&value), sizeof(T));
        if (!out) {
            throw runtime_error(io_failure("Failed to write " + description, errno));
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

        errno = 0;
        out.write(reinterpret_cast<const char*>(values.data()), static_cast<streamsize>(bytes));
        if (!out) {
            throw runtime_error(io_failure("Failed to write " + description, errno));
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

    class TemporaryDirectory
    {
    public:
        explicit TemporaryDirectory(const fs::path& output_path)
        {
            const auto nonce = chrono::steady_clock::now().time_since_epoch().count();
            path_ = output_path.string() + ".binToSer-tmp-" + to_string(nonce);
            if (!fs::create_directory(path_)) {
                throw runtime_error("Failed to create temporary directory: " + path_.string());
            }
        }

        ~TemporaryDirectory()
        {
            error_code error;
            fs::remove_all(path_, error);
        }

        fs::path chunk_path(size_t chunk) const
        {
            return path_ / ("chunk-" + to_string(chunk) + ".bin");
        }

        void remove_chunk(size_t chunk) const
        {
            error_code error;
            if (!fs::remove(chunk_path(chunk), error) || error) {
                throw runtime_error("Failed to remove temporary chunk " + to_string(chunk));
            }
        }

    private:
        fs::path path_;
    };

    void append_file(ofstream& output, const fs::path& input_path, vector<char>& copy_buffer)
    {
        ifstream input(input_path, ios::binary);
        if (!input) {
            throw runtime_error("Failed to open temporary chunk: " + input_path.string());
        }

        while (input) {
            input.read(copy_buffer.data(), static_cast<streamsize>(copy_buffer.size()));
            const streamsize bytes_read = input.gcount();
            if (bytes_read > 0) {
                errno = 0;
                output.write(copy_buffer.data(), bytes_read);
                if (!output) {
                    throw runtime_error(io_failure(
                        "Failed to merge temporary chunk: " + input_path.string(), errno));
                }
            }
        }

        if (!input.eof()) {
            throw runtime_error("Failed to read temporary chunk: " + input_path.string());
        }
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
    vector<uint64_t> twohop_offsets;

    explicit CPU_Graph(const char* input_file)
    {
        read_original_binary(input_file);
    }

    void write_binary(const char* output_file, size_t thread_count)
    {
        ofstream file_out(output_file, ios::binary);
        if (!file_out) {
            throw runtime_error(string("Failed to open output file: ") + output_file);
        }

        write_value(file_out, number_of_vertices, "vertex count");
        write_value(file_out, number_of_edges, "edge count");
        const streampos twohop_count_position = file_out.tellp();
        write_value(file_out, number_of_lvl2adj, "two-hop adjacency count placeholder");

        sort_onehop_adjacency();
        write_array(file_out, onehop_neighbors, "one-hop neighbors");
        write_array(file_out, onehop_offsets, "one-hop offsets");

        build_and_write_twohop_adjacency(file_out, output_file, thread_count);
        write_array(file_out, twohop_offsets, "two-hop offsets");

        const streampos output_end = file_out.tellp();
        file_out.seekp(twohop_count_position);
        if (!file_out) {
            throw runtime_error("Failed to seek to two-hop adjacency count");
        }
        write_value(file_out, number_of_lvl2adj, "two-hop adjacency count");
        file_out.seekp(output_end);
        if (!file_out) {
            throw runtime_error("Failed to restore output position");
        }
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

    void build_and_write_twohop_adjacency(ofstream& file_out,
                                           const fs::path& output_path,
                                           size_t thread_count)
    {
        number_of_lvl2adj = 0;
        twohop_offsets.assign(static_cast<size_t>(number_of_vertices) + 1, 0);
        vector<uint64_t> row_sizes(static_cast<size_t>(number_of_vertices), 0);

        const size_t vertex_count = static_cast<size_t>(number_of_vertices);
        thread_count = max<size_t>(1, min(thread_count, vertex_count));
        const size_t target_chunks = min(vertex_count, thread_count * 32);
        const size_t vertices_per_chunk = (vertex_count + target_chunks - 1) / target_chunks;
        const size_t chunk_count = (vertex_count + vertices_per_chunk - 1) / vertices_per_chunk;

        cout << "Two-hop generation threads: " << thread_count << endl;
        cout << "Two-hop generation chunks: " << chunk_count << endl;

        TemporaryDirectory temporary_directory(output_path);
        atomic<size_t> next_chunk{0};
        atomic<uint64_t> generated_entries{0};
        atomic<bool> failed{false};
        exception_ptr worker_error;
        mutex error_mutex;
        vector<thread> workers;
        workers.reserve(thread_count);

        auto generate_chunks = [&]() {
            try {
                vector<int> seen(vertex_count, -1);
                vector<int> twohop_row;

                while (!failed.load(memory_order_relaxed)) {
                    const size_t chunk = next_chunk.fetch_add(1, memory_order_relaxed);
                    if (chunk >= chunk_count) {
                        break;
                    }

                    const fs::path chunk_path = temporary_directory.chunk_path(chunk);
                    errno = 0;
                    ofstream chunk_output(chunk_path, ios::binary);
                    if (!chunk_output) {
                        throw runtime_error(io_failure(
                            "Failed to create temporary chunk " + chunk_path.string(), errno));
                    }
                    const string chunk_description =
                        "two-hop neighbors in temporary chunk " + chunk_path.string();

                    const size_t first_vertex = chunk * vertices_per_chunk;
                    const size_t last_vertex = min(vertex_count, first_vertex + vertices_per_chunk);
                    for (size_t vertex_index = first_vertex; vertex_index < last_vertex; vertex_index++) {
                        const int vertex = static_cast<int>(vertex_index);
                        twohop_row.clear();

                        auto add_if_new = [&](int candidate) {
                            if (seen[static_cast<size_t>(candidate)] != vertex) {
                                seen[static_cast<size_t>(candidate)] = vertex;
                                twohop_row.push_back(candidate);
                            }
                        };

                        const uint64_t begin = onehop_offsets[vertex_index];
                        const uint64_t end = onehop_offsets[vertex_index + 1];
                        for (uint64_t pos = begin; pos < end; pos++) {
                            const int lvl1adj = onehop_neighbors[static_cast<size_t>(pos)];
                            add_if_new(lvl1adj);

                            const uint64_t lvl2_begin = onehop_offsets[static_cast<size_t>(lvl1adj)];
                            const uint64_t lvl2_end = onehop_offsets[static_cast<size_t>(lvl1adj) + 1];
                            for (uint64_t lvl2_pos = lvl2_begin; lvl2_pos < lvl2_end; lvl2_pos++) {
                                const int lvl2adj = onehop_neighbors[static_cast<size_t>(lvl2_pos)];
                                if (lvl2adj != vertex) {
                                    add_if_new(lvl2adj);
                                }
                            }
                        }

                        sort(twohop_row.begin(), twohop_row.end());
                        write_array(chunk_output, twohop_row, chunk_description);
                        row_sizes[vertex_index] = static_cast<uint64_t>(twohop_row.size());
                        generated_entries.fetch_add(
                            static_cast<uint64_t>(twohop_row.size()), memory_order_relaxed);
                    }

                    errno = 0;
                    chunk_output.close();
                    if (!chunk_output) {
                        throw runtime_error(io_failure(
                            "Failed to finish temporary chunk " + chunk_path.string(), errno));
                    }
                }
            }
            catch (...) {
                failed.store(true, memory_order_relaxed);
                lock_guard<mutex> lock(error_mutex);
                if (!worker_error) {
                    worker_error = current_exception();
                }
            }
        };

        try {
            for (size_t worker = 0; worker < thread_count; worker++) {
                workers.emplace_back(generate_chunks);
            }
        }
        catch (...) {
            failed.store(true, memory_order_relaxed);
            for (thread& worker : workers) {
                worker.join();
            }
            throw;
        }

        for (thread& worker : workers) {
            worker.join();
        }
        if (worker_error) {
            try {
                rethrow_exception(worker_error);
            }
            catch (const exception& error) {
                const uint64_t entries = generated_entries.load(memory_order_relaxed);
                const double generated_gib =
                    static_cast<double>(entries) * sizeof(int) / (1024.0 * 1024.0 * 1024.0);
                const fs::path space_path = output_path.has_parent_path()
                    ? output_path.parent_path() : fs::path(".");
                error_code space_error;
                const fs::space_info space = fs::space(space_path, space_error);

                ostringstream message;
                message << error.what() << ". Generated at least " << entries
                        << " two-hop entries (" << generated_gib << " GiB) before failure";
                if (!space_error) {
                    message << "; filesystem space currently available: "
                            << static_cast<double>(space.available) / (1024.0 * 1024.0 * 1024.0)
                            << " GiB";
                }
                message << ". Check filesystem free space and user quota.";
                throw runtime_error(message.str());
            }
        }

        for (size_t vertex = 0; vertex < vertex_count; vertex++) {
            if (row_sizes[vertex] > numeric_limits<uint64_t>::max() - number_of_lvl2adj) {
                throw length_error("Two-hop adjacency count exceeds uint64_t");
            }
            number_of_lvl2adj += row_sizes[vertex];
            twohop_offsets[vertex + 1] = number_of_lvl2adj;
        }

        vector<char> copy_buffer(8 * 1024 * 1024);
        for (size_t chunk = 0; chunk < chunk_count; chunk++) {
            append_file(file_out, temporary_directory.chunk_path(chunk), copy_buffer);
            temporary_directory.remove_chunk(chunk);
        }

        cout << "|2-hop|: " << number_of_lvl2adj << endl;
    }

    void sort_onehop_adjacency()
    {
        for (int vertex = 0; vertex < number_of_vertices; vertex++) {
            const uint64_t begin = onehop_offsets[static_cast<size_t>(vertex)];
            const uint64_t end = onehop_offsets[static_cast<size_t>(vertex) + 1];
            sort(onehop_neighbors.begin() + static_cast<vector<int>::difference_type>(begin),
                 onehop_neighbors.begin() + static_cast<vector<int>::difference_type>(end));
        }
    }
};

int main(int argc, char* argv[])
{
    if (argc != 2 && argc != 3) {
        cerr << "Usage: ./binToSer <graph_file> [threads]" << endl;
        return 1;
    }

    try {
        const fs::path input_path = argv[1];
        fs::path output_path = input_path.filename();
        output_path.replace_extension(".sbin");
        if (input_path.extension() == ".sbin") {
            throw invalid_argument("Input filename must not already end in .sbin");
        }

        const unsigned int hardware_threads = thread::hardware_concurrency();
        size_t thread_count = min<size_t>(hardware_threads == 0 ? 1 : hardware_threads, 16);
        if (argc == 3) {
            const string thread_argument = argv[2];
            if (thread_argument.empty() ||
                any_of(thread_argument.begin(), thread_argument.end(),
                       [](unsigned char character) { return character < '0' || character > '9'; })) {
                throw invalid_argument("Thread count must be a positive integer");
            }
            size_t parsed_characters = 0;
            const unsigned long long parsed_thread_count = stoull(thread_argument, &parsed_characters);
            if (parsed_characters != thread_argument.size() || parsed_thread_count == 0 ||
                parsed_thread_count > numeric_limits<size_t>::max()) {
                throw invalid_argument("Thread count must be a positive integer");
            }
            thread_count = static_cast<size_t>(parsed_thread_count);
        }

        cout << "Output: " << output_path.string() << endl;
        CPU_Graph hg(input_path.c_str());
        hg.write_binary(output_path.c_str(), thread_count);
    }
    catch (const exception& error) {
        cerr << "binToSer: " << error.what() << endl;
        return 1;
    }

    return 0;
}
