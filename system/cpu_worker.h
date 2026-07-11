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

protected:
    using TimePoint = std::chrono::steady_clock::time_point;
    bool timeout_latched_ = false;

    static TimePoint now()
    {
        return TIME_NOW;
    }

    void reset_timeout_latch()
    {
        timeout_latched_ = false;
    }

    TimePoint start_task_timer()
    {
        reset_timeout_latch();
        return now();
    }

    bool time_over(const TimePoint &start)
    {
        if (timeout_latched_)
            return true;

        if (std::chrono::duration_cast<std::chrono::microseconds>(TIME_NOW - start).count() >= tau_time_g)
        {
            timeout_latched_ = true;
            return true;
        }

        return false;
    }

public:

    CPUWorker() : Worker<TaskT>(cmd.runtime.tasks_per_fetch_cpu_worker) {
        this->Lv.reserve(cmd.runtime.tasks_per_fetch_cpu_worker);
        this->Lt.reserve(cmd.runtime.tasks_per_fetch_cpu_worker);
        this->Lo.reserve(cmd.runtime.tasks_per_fetch_cpu_worker);
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
