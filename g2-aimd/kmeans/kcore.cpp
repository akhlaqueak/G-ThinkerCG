#ifndef KCORE_H
#define KCORE_H

#include "common/graph.h"
#include "common/command_line.h"

#define DEGREESORT 1

#include<numeric>
// #include<filesystem>

class KCore
{
public:
    size_t count, bufTail;
    uintV *degOrder;
    uintV *buffer;
    uintE *degrees;
    uintE level;
    size_t V;
    Graph g;

    void allocateMemory(Graph &g)
    {
        degrees = new uintE[g.GetVertexCount()];
        degOrder = new uintV[g.GetVertexCount()];
        buffer = new uintV[g.GetVertexCount()];
        level = 0;
        count = 0;
        V = g.GetVertexCount();

        for (size_t i = 0; i < g.GetVertexCount(); i++)
        {
            degrees[i] = g.GetRowPtrs()[i + 1] - g.GetRowPtrs()[i];
        }
        this->g = g;
    }

    void scan()
    {
        bufTail = 0;
        for (size_t v = 0; v < V; v++)
        {
            if (degrees[v] == level)
            {
                buffer[bufTail] = v;
                bufTail++;
            }
        }
    }

    void loop()
    {
        for (size_t i = 0; i < bufTail; i++)
        {
            uintV v = buffer[i];
            ull start = g.GetRowPtrs()[v];
            ull end = g.GetRowPtrs()[v + 1];

            for (ull j = start; j < end; j++)
            {

                uintV u = g.GetCols()[j];
                if (degrees[u] > level)
                {

                    degrees[u]--;

                    if (degrees[u] == level)
                    {
                        buffer[bufTail] = u;
                        bufTail++;
                    }
                }
            }
        }
        for (size_t i = 0; i < bufTail; i++)
        {
#if DEGREESORT == 1
            degOrder[count + i] = buffer[i]; // needs to process it again if done this way
#else
            degOrder[buffer[i]] = count + i;
            // printf("%d->%d ", buff->read(i), base+i);
#endif
        }
        count += bufTail;
    }

    // Store degeneracy order...
};

class Degeneracy
{
    Graph g;
    std::vector<size_t> degOrderOffsets;
    KCore kc;

public:
    Degeneracy(Graph &dg) : g(dg)
    {
        degOrderOffsets.push_back(0);
    }
    uintV *degenerate()
    {

        kc.allocateMemory(g);

        std::cout << "K-core Computation Started" << std::endl;

        // auto tick = chrono::steady_clock::now();
        while (kc.count < g.GetVertexCount())
        {
            kc.scan();
            kc.loop();

            std::cout << "*********Completed level: " << kc.level << ", global_count: " << kc.count << " *********" << std::endl;
            kc.level++;
            degOrderOffsets.push_back(kc.count);
        }
        std::cout << "Kcore Computation Done" << std::endl;
        std::cout << "KMax: " << kc.level - 1 << std::endl;

        return kc.degOrder;
    }

    void degreeSort()
    {
        ////////////////// degrees sorting after degenracy...
        // auto tick = chrono::steady_clock::now();

        std::cout << "Degree sorting... " << std::endl;

        // sort each k-shell vertices based on their degrees.
        auto degComp = [&](auto a, auto b)
        {
            return (g.GetRowPtrs()[a + 1] - g.GetRowPtrs()[a]) < (g.GetRowPtrs()[b + 1] - g.GetRowPtrs()[b]);
            // return g.degrees[a] < g.degrees[b];
        };

        for (size_t i = 0; i + 1 < degOrderOffsets.size(); i++)
            std::sort(kc.degOrder + degOrderOffsets[i], kc.degOrder + degOrderOffsets[i + 1], degComp);

        uintV *revOrder = new uintV[g.GetVertexCount()];
        // copy back the sorted vertices to rec array...
        for (size_t i = 0; i < g.GetVertexCount(); i++)
            revOrder[kc.degOrder[i]] = i;
        std::swap(kc.degOrder, revOrder);

        delete[] revOrder;
        // std::cout << "Degree Sorting Time: " << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - tick).count() << std::endl;
    }

    Graph recode()
    {
        uintV *cols = new uintV[g.GetEdgeCount()];
        uintE *rowPtrs = new uintE[g.GetVertexCount() + 1];

        uintE *degrees = new uintE[g.GetVertexCount()];
        // auto tick = chrono::steady_clock::now();
        std::cout << "Degrees copied" << std::endl;
        for (size_t i = 0; i < g.GetVertexCount(); i++)
        {
            degrees[kc.degOrder[i]] = g.GetRowPtrs()[i + 1] - g.GetRowPtrs()[i];
        }

        rowPtrs[0] = 0;
        std::partial_sum(degrees, degrees + g.GetVertexCount(), rowPtrs + 1);
        // delete[] degrees;
        std::cout<<"partial sum complete"<<std::endl;

        for (size_t v = 0; v < g.GetVertexCount(); v++)
        {
            uintV recv = kc.degOrder[v];
            uintE start = rowPtrs[recv];
            uintE end = rowPtrs[recv + 1];
            for (uintE j = g.GetRowPtrs()[v], k = start; j < g.GetRowPtrs()[v + 1]; j++, k++)
            {
                cols[k] = kc.degOrder[g.GetCols()[j]];
            }
            std::sort(cols + start, cols + end);
        }
        // std::cout << "Reordering Time: " << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - tick).count() << std::endl;
        std::cout<<"recoding complete"<<std::endl;
        Graph gRec;
        gRec.SetRowPtrs(rowPtrs);
        gRec.SetCols(cols);
        gRec.SetVertexCount(g.GetVertexCount());
        gRec.SetEdgeCount(g.GetEdgeCount());
        delete[] degrees;

        return gRec;
    }

    void writeBins(ui nb, std::string filename)
    {
        std::ofstream file;
        file.open(filename + ".2.hop.vbmap");
        if (!file)
        {
            std::cout << "Error writing vbmap file : " << filename << std::endl;
            return;
        }
        file << nb << " ";
        for (size_t i = 0; i + 1 < degOrderOffsets.size(); i++)
        {
            size_t window = degOrderOffsets[i + 1] - degOrderOffsets[i];
            // ################### kcore2 ordering...
            // kcore2 is using this format 01230123...
            // ui v = 0;
            // while(window>0){
            //     file<<v%nb<<" ";
            //     v++;
            //     window--;
            // }

            // ################### kcore1 ordering
            // kcore1 is using this format 0000011111222223333300001111...
            size_t binShare = window / nb + 1;
            for (ui j = 0; j < nb && window > 0; j++)
            {
                for (size_t k = 0; k < binShare && window > 0; k++)
                {
                    file << j << " ";
                    window--;
                }
            }
        }
        file.close();
    }

    void writeVertexMap(const std::string &filename)
    {
        std::string map_filename = filename + ".origmap";
        FILE *file = fopen(map_filename.c_str(), "wb");
        if (file == NULL)
        {
            std::cout << "Error writing vertex map file : " << map_filename << std::endl;
            return;
        }

        std::vector<uintV> new_to_old(g.GetVertexCount());
        for (size_t old_id = 0; old_id < g.GetVertexCount(); old_id++)
            new_to_old[kc.degOrder[old_id]] = static_cast<uintV>(old_id);

        size_t uintV_size = sizeof(uintV);
        size_t vertex_count = g.GetVertexCount();
        fwrite(&uintV_size, sizeof(size_t), 1, file);
        fwrite(&vertex_count, sizeof(size_t), 1, file);
        fwrite(new_to_old.data(), sizeof(uintV), new_to_old.size(), file);
        fclose(file);
    }
};

int main(int argc, char *argv[])
{
    CommandLine cmd(argc, argv);
    std::string filename = cmd.GetOptionValue("-f", "../data/com-dblp.ungraph.txt");
    ui nbins = cmd.GetOptionIntValue("-nb", 1);
    std::string binfile = filename.substr(0, filename.rfind(".")) + ".bin";

    std::cout << "Loading Started" << std::endl;
    Graph g(filename);
    std::cout << "Loading Done" << std::endl;

    Degeneracy deg(g);
    deg.degenerate(); // performs KCore Decomposition, and rec is sorted in degeneracy order
#if DEGREESORT == 1
    deg.degreeSort(); // sorts the vertices based on degrees, within the degeneracy order
#endif

    Graph gRec = deg.recode();
    gRec.writeBinFile(binfile);
    deg.writeVertexMap(binfile);
    std::cout << "writing bins... " << std::endl;
    deg.writeBins(nbins, binfile);
}
#endif
