#ifndef COMPER_H_
#define COMPER_H_
template <class TaskT>
class Worker
{
public:
    // using ContextT = typename TaskT::ContextType;
    typedef typename TaskT::ContextType ContextT;
    typedef vector<TaskT *> TaskContainer;
    thread main_thread;

    TaskContainer Lt;
    TaskContainer Lo;
    TaskContainer big_Lo;

    vector<VertexID> Lv;
    stack<TaskT *> *SC;
    stack<TaskT *> *big_SC;

    ui tasks_per_fetch;
    ull spilled_tasks = 0;
    
    std::mutex ready_mtx;
    std::condition_variable ready_cv;
    bool ready = false; // Whether the consumer should wake up

    Worker(ui n_tasks)
    {
        tasks_per_fetch = n_tasks;
        SC = (stack<TaskT *> *)global_SC;
        big_SC = (stack<TaskT *> *)global_big_SC;
        Lv.reserve(n_tasks);
        Lt.reserve(n_tasks);
        Lo.reserve(n_tasks);
        big_Lo.reserve(n_tasks);
    }

    virtual ~Worker()
    {
        if (main_thread.joinable())
            main_thread.join();
    }

    virtual void run() = 0;

    void start()
    {
        main_thread = thread(&Worker::run_thread, this);
    }

    void spill_Lo()
    {
        if (!Lo.empty())
        {
            unique_lock<shared_timed_mutex> lock(SC_mtx);
            for (TaskT *task : Lo)
                SC->push(task);
            Lo.clear();
        }
        if (!big_Lo.empty())
        {
            unique_lock<shared_timed_mutex> lock(big_SC_mtx);
            for (TaskT *task : big_Lo)
                big_SC->push(task);
            big_Lo.clear();
        }
    }

    void add_task(TaskT *task)
    {
        Lo.push_back(task);
        if (Lo.size() > LO_SPILL_THRESH)
            spill_Lo();
    }

    void add_large_task(TaskT *task)
    {
        big_Lo.push_back(task);
        if (big_Lo.size() > LO_SPILL_THRESH)
            spill_Lo();
    }

    bool is_SC_empty()
    {
        shared_lock<shared_timed_mutex> lock(SC_mtx);
        return SC->empty();
    }

    void notify()
    {
        lock_guard<mutex> lock(ready_mtx);
        ready = true;
        ready_cv.notify_one();
    }

    void run_thread()
    {
        while (true)
        {
            {
                // creating this scope for lock, wait requiers unique_lock
                unique_lock<mutex> lock(ready_mtx);
                ready_cv.wait(lock, [this]
                              { return ready; });
            }
            if (global_end_label)
                break;
            run();
            if (Lo.size() || big_Lo.size())
                spill_Lo();
            {
                // creating this scope for lock
                lock_guard<mutex> lock(ready_mtx);
                workers_list.enqueue(this);
                ready = false;
            }

            {
                // awake master
                lock_guard<mutex> master_lock(mtx_master);
                master_ready = true;
                cv_master.notify_one();
            }
        }
    }
};

#endif
