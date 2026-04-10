#ifndef MC_CPU_APP
#define MC_CPU_APP

#define TIME_THRESHOLD 10
#define TIME_OVER(ST) (chrono::duration_cast<chrono::milliseconds>(TIME_NOW - ST).count() > TIME_THRESHOLD)

class QCCPUWorker : public CPUWorker<QCTask>
{
public:

    virtual QCTask *task_spawn(VertexID &data)
    {

    }

    void QC(QCContext& ctx, auto st)
    {

    }

    virtual void compute(QCContext &context)
    {
        QC(context, TIME_NOW);
    }
};

#endif

