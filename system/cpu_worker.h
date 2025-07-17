#pragma once

template <class TaskT>
class CPUWorker : public Worker<TaskT>
{
    using ContextT = typename TaskT::ContextType;

    // UDF1
    virtual TaskT *task_spawn(VertexID &data) = 0;
    // UDF2
    virtual void compute(ContextT &context) = 0;
    // UDF2 wrapper
    void compute(TaskT *task)
    {
        compute(task->context);
    }

public:

    CPUWorker() : Worker<TaskT>(tasks_per_fetch_g) {
        this->Lv.reserve(tasks_per_fetch_g);
        this->Lt.reserve(tasks_per_fetch_g);
        this->Lo.reserve(tasks_per_fetch_g);
    }


    virtual void run()
    {
        for (auto &u : this->Lv)
        {
            TaskT *task = task_spawn(u);
            this->compute(task);
            delete task;
        }
        this->Lv.clear();
        for (auto task : this->Lt)
        {
            this->compute(task);
            delete task;
        }
        this->Lt.clear();
    }
};