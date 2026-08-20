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
    queue<VertexID> big_data_array;

    stack<TaskT *> *SC;
    stack<TaskT *> *big_SC;
    ui eta_per_warp_ = 1000;
    ui eta_total_ = 1000 * N_WARPS;

    // Init files seq with 1 for each thread.
    Master()
    {
        std::cout.imbue(std::locale(""));
        global_SC = SC = new stack<TaskT *>();
        global_big_SC = big_SC = new stack<TaskT *>();
        global_end_label = false;
    }

    void set_eta(ui eta_per_warp)
    {
        eta_per_warp_ = eta_per_warp;
        eta_total_ = eta_per_warp_ * N_WARPS;
        if (num_gpu_workers > 0)
        {
            cudaMemcpyToSymbol(eta, &eta_total_, sizeof(ui));
        }
    }

    ui eta_per_warp() const
    {
        return eta_per_warp_;
    }

    ui eta_total() const
    {
        return eta_total_;
    }

    void apply_runtime_config(const CommandLine::RuntimeConfig &config)
    {
        num_cpu_workers = config.num_cpu_workers;
        num_gpu_workers = config.num_gpu_workers;
        tasks_per_fetch_gpu_worker_g = config.tasks_per_fetch_gpu_worker;
        tasks_per_fetch_g = config.tasks_per_fetch_cpu_worker;
        host_to_gpu_transfer_size_g = config.hg_steal;
        min_transfer_size_g = config.min_hg_steal;
        tau_time_g = config.tau_time_us;
        g_ping_pong_flag = config.ping_pong != 0;
        g_abort_chunk_flag = config.ping_pong == 1;
        set_eta(config.eta_per_warp);
    }

    void add_task(TaskT *t)
    {
        unique_lock<shared_timed_mutex> lock(SC_mtx);
        SC->push(t);
    }

    void create_workers()
    {
        require_gpu_if_requested(num_gpu_workers);

        for (int i = 0; i < num_cpu_workers; i++)
        {
            WorkerT *worker = new CPUWorkerT();
            worker->start(); 
            workers_list.enqueue(worker);
        }
        for (int i = 0; i < num_gpu_workers; i++)
        {
            WorkerT *worker = new GPUWorkerT(eta_total_);
            worker->start();
            workers_list.enqueue(worker);
        }
        cout << "workers created, cpu: " << num_cpu_workers << ", gpu: " << num_gpu_workers << endl;
    }

    bool is_SC_empty()
    {
        shared_lock<shared_timed_mutex> sc_lock(SC_mtx);
        shared_lock<shared_timed_mutex> big_lock(big_SC_mtx);
        return SC->empty() && big_SC->empty();
    }
    size_t SC_size()
    {
        shared_lock<shared_timed_mutex> lock(SC_mtx);
        return SC->size();
    }
    size_t big_SC_size()
    {
        shared_lock<shared_timed_mutex> lock(big_SC_mtx);
        return big_SC->size();
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

            if ((data_array.empty() and big_data_array.empty() and is_SC_empty()) or workers_list.empty())
                continue;

            WorkerT *worker = (WorkerT *)workers_list.dequeue();
            bool assigned_work = false;
            // cout << "SC size: " << SC->size()<<" workers: "<< workers_list.size()  << endl;

            if (dynamic_cast<GPUWorkerT *>(worker))
            {
                // GPU Worker
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
                    ui chunk = sc_size >= min_transfer_size_g ?
                               std::min<ull>(host_to_gpu_transfer_size_g, sc_size) : 0;
                    // ui chunk = std::min<ull>(host_to_gpu_transfer_size_g, sc_size);
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
                // It's a CPU worker
                {
                    unique_lock<shared_timed_mutex> lock(big_SC_mtx);
                    if (!big_SC->empty())
                    {
                        ui chunk = std::min<ui>(big_SC->size() / (workers_list.size() + 1), worker->tasks_per_fetch);
                        chunk = std::max<ui>(chunk, 1);
                        for (ui i = 0; i < chunk && !big_SC->empty(); i++)
                        {
                            TaskT *task = big_SC->top();
                            worker->Lt.push_back(task);
                            big_SC->pop();
                        }
                        assigned_work = (chunk > 0);
                    }
                }
                if (!assigned_work)
                {
                    unique_lock<shared_timed_mutex> lock(SC_mtx);
                    if (!SC->empty())
                    {
                        ui chunk = std::min<ui>(SC->size() / (workers_list.size() + 1), worker->tasks_per_fetch);
                        chunk = std::max<ui>(chunk, 1);
                        for (ui i = 0; i < chunk && !SC->empty(); i++)
                        {
                            TaskT *task = SC->top();
                            worker->Lt.push_back(task);
                            SC->pop();
                        }
                        assigned_work = (chunk > 0);
                    }
                }
                if (!assigned_work && not big_data_array.empty())
                {
                    ui chunk = std::min<ull>(big_data_array.size() / (workers_list.size() + 1), worker->tasks_per_fetch);
                    chunk = max(chunk, 1);
                    for (ui i = 0; i < chunk; i++)
                    {
                        VertexID item = big_data_array.front();
                        worker->Lv.push_back(item);
                        big_data_array.pop();
                    }
                    assigned_work = (chunk > 0);
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
        } while (not(workers_list.size() == num_cpu_workers + num_gpu_workers and data_array.empty() and big_data_array.empty() and is_SC_empty()));
        global_end_label = true;
        notify_all_workers();
    }
};

#endif
