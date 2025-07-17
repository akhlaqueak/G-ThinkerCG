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
    deque<VertexID *> data_array;

    stack<TaskT *> *SC;

    // Init files seq with 1 for each thread.
    Master()
    {
        global_SC = SC = new stack<TaskT *>();
        global_end_label = false;
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
                if (not data_array.empty())
                    for (ui i = 0; i < worker->tasks_per_fetch && data_array.size(); i++)
                    {
                        worker->Lv.push_back(*(data_array.front()));
                        data_array.pop_front();
                    }
                // else if (SC_size()>worker->tasks_per_fetch)
                else if (not is_SC_empty())
                {
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
                if (not is_SC_empty()){
                    unique_lock<shared_timed_mutex> lock(SC_mtx);
    
                    for (ui i = 0; i < worker->tasks_per_fetch && !SC->empty(); i++)
                    {
                        TaskT *task = SC->top();
                        worker->Lt.push_back(task);
                        SC->pop();
                    }
                }
                else 
                if (not data_array.empty())
                {
                    for (ui i = 0; i < worker->tasks_per_fetch && data_array.size(); i++)
                    {
                        worker->Lv.push_back(*(data_array.back()));
                        data_array.pop_back();
                    }
                }
            }

            worker->notify();
            // cout<<"workers: " <<workers_list.size()<<endl;
        } while (not(workers_list.size() == num_cpu_workers + num_gpu_workers and data_array.empty() and is_SC_empty()));
        global_end_label = true;
        notify_all_workers();
    }
};

#endif
