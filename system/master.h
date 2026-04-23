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
    using ClockT = std::chrono::high_resolution_clock;

    // Contains all data loaded from file
    deque<VertexID> data_array;

    stack<TaskT *> *SC;
    std::chrono::nanoseconds cpu_root_dispatch_time{0};

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

            if (dynamic_cast<GPUWorkerT *>(worker))
            {
                // cout << "SC size: " << SC->size() << endl;
                if (not data_array.empty())
                    for (ui i = 0; i < worker->tasks_per_fetch && !data_array.empty(); i++)
                    {
                        VertexID item = data_array.front();
                        worker->Lv.push_back(item);
                        data_array.pop_front();
                    }
                else if (SC_size() >= std::max<size_t>(1, tasks_per_fetch_gpu_worker_g * 0.1))
                {
                    // else {
                    unique_lock<shared_timed_mutex> lock(SC_mtx);
                    for (ui i = 0; i < worker->tasks_per_fetch && !SC->empty(); i++)
                    {
                        TaskT *task = SC->top();
                        worker->Lt.push_back(task);
                        SC->pop();
                    }
                }
            }
            else
            {
                if (not is_SC_empty())
                {
                    unique_lock<shared_timed_mutex> lock(SC_mtx);
                    ui chunk = std::min<ui>(SC->size() / (workers_list.size() + 1), worker->tasks_per_fetch);
                    chunk = max(chunk, 1);
                    for (ui i = 0; i < chunk && !SC->empty(); i++)
                    {
                        TaskT *task = SC->top();
                        worker->Lt.push_back(task);
                        SC->pop();
                    }
                }
                else if (not data_array.empty())
                {
                    // cout << "workers: " << workers_list.size() << "data_array: " << data_array.size() << endl;
                    ui chunk = std::min<ull>(data_array.size() / (workers_list.size() + 1), worker->tasks_per_fetch);
                    chunk = max(chunk, 1);
                    auto dispatch_start = ClockT::now();
                    for (ui i = 0; i < chunk; i++)
                    {
                        VertexID item = data_array.back();
                        worker->Lv.push_back(item);
                        data_array.pop_back();
                    }
                    cpu_root_dispatch_time += std::chrono::duration_cast<std::chrono::nanoseconds>(ClockT::now() - dispatch_start);
                }
            }

            worker->notify();
        } while (not(workers_list.size() == num_cpu_workers + num_gpu_workers and data_array.empty() and is_SC_empty()));
        cout << "CPU root dispatch time (s): " << std::fixed << std::setprecision(6)
             << cpu_root_dispatch_time.count() / 1e9 << endl;
        global_end_label = true;
        notify_all_workers();
    }
};

#endif
