#define TIME_NOW std::chrono::steady_clock::now()

class Timer
{
public:
    Timer() { StartTimer(); }
    ~Timer() {}

    void StartTimer() { start_timestamp_ = TIME_NOW; }
    void EndTimer() { end_timestamp_ = TIME_NOW; }

    double GetElapsedMicroSeconds() const { return std::chrono::duration_cast<std::chrono::microseconds>(end_timestamp_ - start_timestamp_).count(); }
    double elapsed() { return std::chrono::duration_cast<std::chrono::microseconds>(TIME_NOW - start_timestamp_).count(); }
    void restart() { StartTimer(); }
    inline void PrintElapsedMicroSeconds(const std::string &time_tag) const
    {
        std::cout << std::fixed << "finish " << time_tag << ", elapsed_time=" << GetElapsedMicroSeconds() / 1000.0 << "ms" << std::endl;
    }

private:
    std::chrono::steady_clock::time_point start_timestamp_;
    std::chrono::steady_clock::time_point end_timestamp_;
};
template <class T>
class ThreadSafeQueue
{
private:
    mutable std::mutex mutex_;

public:
    std::queue<T> queue_;

    ThreadSafeQueue() = default;
    ~ThreadSafeQueue() = default;

    // Disable copy constructor and assignment
    ThreadSafeQueue(const ThreadSafeQueue &) = delete;
    ThreadSafeQueue &operator=(const ThreadSafeQueue &) = delete;

    void enqueue(T item)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push(item);
    }

    T dequeue()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.empty())
            return nullptr; // or throw an exception
        T item = queue_.front();
        queue_.pop();
        return item;
    }
    T front()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.empty())
            return nullptr; // or throw an exception
        T item = queue_.front();
        return item;
    }
    bool empty() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.empty();
    }

    size_t size() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }

    void clear()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        while (queue_.size())
            queue_.pop();
    }
};
class CommandLine {
    public:
     struct RuntimeConfig {
       size_t num_cpu_workers = 32;
       size_t num_gpu_workers = 1;
       size_t tasks_per_fetch_gpu_worker = 1000000;
       size_t tasks_per_fetch_cpu_worker = 50;
       ui eta_per_warp = 1000;
       ui tau_time_us = 10;
       bool ping_pong = true;
       std::string data_graph = "";
       int query_type = 0;
       std::string plan_strategy = "hybrid";
     };

     int argc;
     char** argv;
     RuntimeConfig runtime;
     CommandLine(): argc(0), argv(nullptr) {}
     CommandLine(int _argc, char** _argv) : argc(_argc), argv(_argv) {}

     void SetArgs(int _argc, char** _argv) {
       argc = _argc;
       argv = _argv;
     }
   
     void BadArgument() {
       std::cout << "usage: " << argv[0] << " bad argument" << std::endl;
       abort();
     }
   
     char* GetOptionValue(const std::string& option) {
       for (int i = 1; i < argc; i++)
         if ((std::string)argv[i] == option)
           return argv[i + 1];
       return NULL;
     }
   
     std::string GetOptionValue(const std::string& option, std::string defaultValue) {
       for (int i = 1; i < argc; i++)
         if ((std::string)argv[i] == option)
           return (std::string)argv[i + 1];
       return defaultValue;
     }
   
     int GetOptionIntValue(const std::string& option, int defaultValue) {
       for (int i = 1; i < argc; i++)
         if ((std::string)argv[i] == option) {
           int r = atoi(argv[i + 1]);
           return r;
         }
       return defaultValue;
     }
   
     long GetOptionLongValue(const std::string& option, long defaultValue) {
       for (int i = 1; i < argc; i++)
         if ((std::string)argv[i] == option) {
           long r = atol(argv[i + 1]);
           return r;
         }
       return defaultValue;
     }
   
     double GetOptionDoubleValue(const std::string& option, double defaultValue) {
       for (int i = 1; i < argc; i++)
         if ((std::string)argv[i] == option) {
           double val;
           if (sscanf(argv[i + 1], "%lf", &val) == EOF) {
             BadArgument();
           }
           return val;
       }
       return defaultValue;
     }

     void ParseRuntimeConfig(const RuntimeConfig& defaults = RuntimeConfig()) {
       runtime = defaults;
       runtime.num_cpu_workers = GetOptionIntValue("-cpu", static_cast<int>(defaults.num_cpu_workers));
       runtime.num_gpu_workers = GetOptionIntValue("-gpu", static_cast<int>(defaults.num_gpu_workers));
       runtime.tasks_per_fetch_gpu_worker = GetOptionIntValue("-gpuchunk", static_cast<int>(defaults.tasks_per_fetch_gpu_worker));
       runtime.tasks_per_fetch_cpu_worker = GetOptionIntValue("-cpuchunk", static_cast<int>(defaults.tasks_per_fetch_cpu_worker));
       runtime.eta_per_warp = GetOptionIntValue("-eta", defaults.eta_per_warp);
       runtime.tau_time_us = GetOptionIntValue("-tau", defaults.tau_time_us);
       runtime.ping_pong = GetOptionIntValue("-pingpong", defaults.ping_pong ? 1 : 0);
       runtime.data_graph = GetOptionValue("-dg", defaults.data_graph);
       runtime.query_type = GetOptionIntValue("-q", defaults.query_type);
       runtime.plan_strategy = GetOptionValue("-s", defaults.plan_strategy);
     }
   };

inline CommandLine cmd;
   
