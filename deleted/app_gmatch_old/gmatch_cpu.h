#ifndef APP_GMATCH_CPU_H
#define APP_GMATCH_CPU_H

class GMComper : public Comper<GMTask, ui>
{
public:
    virtual void compute(ContextT &context)
    {
    }
    virtual GMTask *task_spawn(ui &data)
    {
    }
};

class GMWorker : public Worker<GMComper, GM_GPU_App>
{
public:
    GMWorker(ui num_compers) : Worker(num_compers)
    {
    }

    ~GMWorker()
    {
    }

    void load_data(ui argc, char *argv[])
    {
        CommandLine cmd(argc, argv);
        std::string fp = cmd.GetOptionValue("-dg", "./data/com-friendster.ungraph.txt.bin");
        data_graph = Graph(fp);

        for (int i = 0; i < data_graph.GetVertexCount(); ++i)
            data_array.push_back(new ui(i));
    }
};

#endif