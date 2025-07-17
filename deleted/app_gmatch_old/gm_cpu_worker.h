#ifndef GM_CPU_APP
#define GM_CPU_APP

#define TIME_THRESHOLD 100
#define TIME_OVER(ST) (chrono::duration_cast<chrono::milliseconds>(TIME_NOW - ST).count() > TIME_THRESHOLD)


class GMCPUWorker : public CPUWorker<GMTask>
{
public:
    ui max_sz = 0;
    ui total_counts=0;


    virtual GMTask *task_spawn(VertexID &data)
    {
        GMTask *t=new GMTask();
        return t;
    }

    virtual void compute(GMContext &context)
    {
        
    }
};

#endif

