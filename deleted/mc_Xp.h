#pragma once
#include "../system/timer.h"
#include "../system/worker.h"
#include "../system/task.h"
#include "graph.h"
#define TIME_THRESHOLD 100
#define TIME_OVER(ST) (chrono::duration_cast<chrono::milliseconds>(TIME_NOW - ST).count()>TIME_THRESHOLD)
//  todo different threshold

#include <vector>
using namespace std;

#define TIME_THRESHOLD 0.1

using namespace std::chrono;

Graph data_graph;

typedef unsigned long long int ULL;

struct ContextValue
{   
    vector<ui> R;
    vector<ui> P;
    vector<ui> X;
    vector<ui> Xp;
    ui xsz;
};


ofbinstream & operator>>(ofbinstream & m, ContextValue & c)
{
    m >> c.R;
    m >> c.P;
    m >> c.X;
    m >> c.Xp;
    m >> c.xsz;
    return m;
}
ifbinstream & operator<<(ifbinstream & m, const ContextValue & c) 
{
    m << c.R;
    m << c.P;
    m << c.X;
    m << c.Xp;
    m << c.xsz;
    return m;
}

typedef Task<ContextValue> MCTask;


class MCComper: public Comper<MCTask, ui>
{
public:

    ULL counter=0;
    ui max_sz = 0;

    MCComper() {

    }
    bool is_sorted(auto beg_p, auto end_p){
        while(beg_p+1<end_p)
            {
                if(*beg_p>*(beg_p+1)) return false;
                beg_p++;
            }
        return true;
    }
    void ordered_remove(auto& vec, auto val){
        auto it = std::lower_bound(vec.begin(), vec.end(), val);
        if (it != vec.end()) 
            vec.erase(it);
    }

    void ordered_insert(auto& vec, auto value) {
        auto pos = std::lower_bound(vec.begin(), vec.end(), value);
        vec.insert(pos, value);
    }

    ui intersection_length(auto beg1, auto end1, auto beg2, auto end2) {
        ui count = 0;
        while (beg1 < end1 && beg2<end2)
            if (*beg1 < *beg2) 
                ++beg1;
            else if (*beg1 > *beg2) 
                ++beg2;
            else
                ++count,++beg1,++beg2;
        return count;
    }
    ui intersection_inplace(auto beg1, auto end1, auto beg2, auto end2) {
        ui count = 0;
        auto res=beg2;
        while (beg1 < end1 && beg2<end2)
            if (*beg1 < *beg2) 
                ++beg1;
            else if (*beg1 > *beg2) 
                ++beg2;
            else{
                std::swap(*(res+count), *beg2);
                ++count,++beg1,++beg2;
            }
        return count;
    }
    ui select_pivot(auto& P, auto& X, ui xsz, auto& Xp){
        ui pivot=P[0];
        ui max_count = 0;
        for(ui u:P){
            ui nbr_count;
            const ui *nbrs = data_graph.getVertexNeighbors(u, nbr_count);
            ui count = intersection_length(nbrs,nbrs+nbr_count, P.begin(), P.end());
            if(count>max_count) max_count = count, pivot = u;
        }
        for(ui i=0;i<xsz; i++){
            ui u = X[i];
            ui nbr_count;
            const ui *nbrs = data_graph.getVertexNeighbors(u, nbr_count);
            ui count = intersection_length(nbrs,nbrs+nbr_count, P.begin(), P.end());
            if(count>max_count) max_count = count, pivot = u;
        }
        for(ui u:Xp){
            ui nbr_count;
            const ui *nbrs = data_graph.getVertexNeighbors(u, nbr_count);
            ui count = intersection_length(nbrs,nbrs+nbr_count, P.begin(), P.end());
            if(count>max_count) max_count = count, pivot = u;
        }
        return pivot;
    }

    virtual bool task_spawn(ui& data)
    {
        ui i = data;
        vector<ui> R {i};
        vector<ui> P, X;
        ui nbr_count;
        const ui *nbrs = data_graph.getVertexNeighbors(i, nbr_count);
        // cout << "-----------" << i << ", " << nbr_count << endl;
        for (int j=0; j<nbr_count; ++j)
        {
            const ui neighbor = nbrs[j];
            if (neighbor < i)
                X.push_back(neighbor);
            else
                P.push_back(neighbor);
        }
        MCTask *t = new MCTask();
        t->context.R = move(R);
        t->context.P = move(P);
        t->context.xsz = X.size();
        t->context.X = move(X);
        add_task(t);

        return true;
    }



    void BK(vector<ui> &R, vector<ui> &P, vector<ui> &X, ui xsz, vector<ui> &Xp, auto st)
    {
        struct timeb cur_time;
		double drun_time;
        vector<ui> Q, newP, newXp;
        Q.reserve(P.size());
        newP.reserve(P.size());
        newXp.reserve(P.size());

        if (P.empty()) 
        {
            if(Xp.empty() and xsz==0){
                max_sz = max_sz > R.size() ? max_sz : R.size();
                counter++;
            }
            return;
        }
        // find a pivot
        ui pivot = select_pivot(P, X, xsz, Xp);

        ui pivot_nbr_count;
        const ui *pivot_nbrs = data_graph.getVertexNeighbors(pivot, pivot_nbr_count);
        Q.clear();
        for (auto u : P){
            if(!std::binary_search(pivot_nbrs, pivot_nbrs + pivot_nbr_count, u))
                Q.push_back(u);
        }

        for (auto u: Q){
            R.push_back(u);
            ui nbr_count;
            const ui *nbrs = data_graph.getVertexNeighbors(u, nbr_count);

            newP.clear();
            newXp.clear();
            std::set_intersection(nbrs, nbrs+nbr_count, P.begin(), P.end(), back_inserter(newP));
            std::set_intersection(nbrs, nbrs+nbr_count, Xp.begin(), Xp.end(), back_inserter(newXp));
            ui new_xsz = intersection_inplace(nbrs, nbrs+nbr_count, X.begin(), X.begin()+xsz);


            ftime(&cur_time);
            drun_time = cur_time.time-data_graph.gtime_start[thread_id].time+(double)(cur_time.millitm-data_graph.gtime_start[thread_id].millitm)/1000;
            // drun_time = 0;
            if(!TIME_OVER(st)) {
                BK(R, newP, X, new_xsz, newXp, TIME_NOW);
            } else {
                MCTask *t = new MCTask();
                t->context.R = R;
                t->context.P = newP;
                t->context.X = X;
                t->context.xsz = new_xsz;
                t->context.Xp = newXp;
                add_task(t);
                cout<<"^";
            }
            ordered_insert(Xp, u);
            std::sort(X.begin(), X.begin()+xsz);
            ordered_remove(P, u);
            R.pop_back();
        }
    }

    virtual void compute(ContextT &context)
    {
        ftime(&data_graph.gtime_start[thread_id]);
        counter = global_counters[thread_id];
        BK(context.R, context.P, context.X, context.xsz, context.Xp, TIME_NOW);
        global_counters[thread_id] = counter;
        // global_sz[thread_id] = max_sz;
    }

    virtual bool is_bigTask(MCTask *task)
	{
		if (task->context.P.size() > BIGTASK_THRESHOLD)
		{
			return true;
		}
		return false;
	}
};

class MCWorker : public Worker<MCComper>
{
public:
    MCWorker(ui num_compers) : Worker(num_compers)
    {
    }

    ~MCWorker()
    {
    }

    void load_data(char* file_path)
    {
        std::string fp = std::string(file_path);
        data_graph.loadGraphFromFile(fp);

        for (int i=0; i<data_graph.getVerticesCount(); ++i)
            data_array.push_back(new ui(i));
    }

    virtual bool task_spawn(ui& data)
    {
        ui i = data;
        vector<ui> R;
        vector<ui> P, X;
        R.push_back(i);
        ui nbr_count;
        const ui *nbrs = data_graph.getVertexNeighbors(i, nbr_count);
        P.reserve(nbr_count);
        X.reserve(nbr_count);
        R.reserve(nbr_count);
        // cout << "-----------" << i << ", " << nbr_count << endl;
        for (int j=0; j<nbr_count; ++j)
        {
            const ui neighbor = nbrs[j];
            if (neighbor < i)
                X.push_back(neighbor);
            else
                P.push_back(neighbor);
        }
        MCTask *t = new MCTask();
        t->context.R = move(R);
        t->context.P = move(P);
        t->context.xsz = X.size();
        t->context.X = move(X);
        add_task(t);

        return true;
    }

    virtual bool is_bigTask(MCTask *task)
	{
		if (task->context.P.size() > BIGTASK_THRESHOLD)
		{
			return true;
		}
		return false;
	}
};