#ifndef MASTER_H_
#define MASTER_H_

#include "device/util.h"
#include "buffer.h"
#include "worker.h"
#include "cpu_worker.h"
#include "gpu_worker.h"
#include "gpu_context.h"
template <class CPUWorkerT, class GPUContextT>
class Master
{
public:
    using GPUWorkerT = GPUWorker<GPUContextT>;
    using TaskT = typename GPUContextT::TaskType;
    using WorkerT = Worker<TaskT>;

    // Contains all data loaded from file
    deque<VertexID> data_array;

    stack<TaskT *> *SC;

    // Init files seq with 1 for each thread.
    Master()
    {
        std::cout.imbue(std::locale(""));
        global_SC = SC = new stack<TaskT *>();
        global_end_label = false;
    }

    void add_task(TaskT *t)
    {
        unique_lock<shared_timed_mutex> lock(SC_mtx);
        SC->push(t);
    }

    void create_workers()
    {
        for (int i = 0; i < num_gpu_workers; i++)
        {
            WorkerT *worker = new GPUWorkerT();
            worker->start();
            workers_list.enqueue(worker);
        }
        for (int i = 0; i < num_cpu_workers; i++)
        {
            WorkerT *worker = new CPUWorkerT();
            worker->start(); // i is thread id for that worker
            workers_list.enqueue(worker);
        }
        cout << "workers created, cpu: " << num_cpu_workers << ", gpu: " << num_gpu_workers << endl;
    }

    bool is_SC_empty()
    {
        shared_lock<shared_timed_mutex> lock(SC_mtx);
        // shared_lock lock(SC_mtx);
        // unique_lock lock(SC_mtx);
        return SC->empty();
    }
    size_t SC_size()
    {
        shared_lock<shared_timed_mutex> lock(SC_mtx);
        // shared_lock lock(SC_mtx);
        // unique_lock lock(SC_mtx);
        return SC->size();
    }
    void notify_all_workers()
    {
        auto workers = workers_list.queue_; // copy all workers
        while (!workers.empty())
        {
            WorkerT *w = (WorkerT *)workers.front();
            w->notify();
            workers.pop();
        }
    }
    // Program entry point
    void run()
    {
        create_workers();
        do
        {
            if (workers_list.empty())
            {
                unique_lock<mutex> lock(mtx_master);
                master_ready = false;
                cv_master.wait(lock, []
                               { return master_ready; });
            }

            if ((data_array.empty() and is_SC_empty()) or workers_list.empty())
                continue;

            WorkerT *worker = (WorkerT *)workers_list.dequeue();
            bool assigned_work = false;

            if (dynamic_cast<GPUWorkerT *>(worker))
            {
                // cout << "SC size: " << SC->size() << endl;
                if (not data_array.empty())
                {
                    ui chunk = std::min<ull>(data_array.size(), worker->tasks_per_fetch);
                    for (ui i = 0; i < chunk; i++)
                    {
                        VertexID item = data_array.front();
                        worker->Lv.push_back(item);
                        data_array.pop_front();
                    }
                    assigned_work = (chunk > 0);
                }
                else
                {
                    unique_lock<shared_timed_mutex> lock(SC_mtx);
                    const size_t sc_size = SC->size();
                    ui chunk = sc_size >= tasks_per_fetch_gpu_worker_g * gpu_min_thresh_SC ?
                               std::min<ull>(tasks_per_fetch_gpu_worker_g * gpu_min_thresh_SC, sc_size) : 0;
                    for (ui i = 0; i < chunk; i++)
                    {
                        TaskT *task = SC->top();
                        worker->Lt.push_back(task);
                        SC->pop();
                    }
                    assigned_work = (chunk > 0);
                }
            }
            else
            {
                {
                    unique_lock<shared_timed_mutex> lock(SC_mtx);
                    if (!SC->empty())
                    {
                        ui chunk = std::min<ui>(SC->size() / (workers_list.size() + 1), worker->tasks_per_fetch);
                        chunk = std::max(chunk, 1);
                        for (ui i = 0; i < chunk && !SC->empty(); i++)
                        {
                            TaskT *task = SC->top();
                            worker->Lt.push_back(task);
                            SC->pop();
                        }
                        assigned_work = (chunk > 0);
                    }
                }
                if (!assigned_work && not data_array.empty())
                {
                    ui chunk = std::min<ull>(data_array.size() / (workers_list.size() + 1), worker->tasks_per_fetch);
                    chunk = max(chunk, 1);
                    for (ui i = 0; i < chunk; i++)
                    {
                        VertexID item = data_array.back();
                        worker->Lv.push_back(item);
                        data_array.pop_back();
                    }
                    assigned_work = (chunk > 0);
                }
            }

            if (assigned_work)
                worker->notify();
            else
                workers_list.enqueue(worker);
        } while (not(workers_list.size() == num_cpu_workers + num_gpu_workers and data_array.empty() and is_SC_empty()));
        global_end_label = true;
        notify_all_workers();
    }
};

#endif
