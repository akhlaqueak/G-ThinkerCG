#ifndef SYSTEM_PLAN_H
#define SYSTEM_PLAN_H
std::string plan_strategy;

class Plan
{
public:
    Plan() : hop_(0), root_degree_(0)
    {
    }

    int GetMinHopWithBFS(uintV root)
    {
        std::vector<bool> visited(sz, false);
        std::vector<int> level(sz, -1);
        int max_level_v = -1;
        int max_level_e = -1;

        // bfs from root vertex
        std::queue<uintV> queue;
        queue.push(root);
        visited[root] = true;
        level[root] = 0;
        while (!queue.empty())
        {
            uintV front = queue.front();
            queue.pop();
            max_level_v = level[front];
            for (size_t i = graph.GetRowPtrs()[front]; i < graph.GetRowPtrs()[front + 1]; ++i)
            {
                uintV j = graph.GetCols()[i];
                if (!visited[j])
                {
                    visited[j] = true;
                    level[j] = level[front] + 1;
                    queue.push(j);
                }
                else if (level[j] == level[front])
                {
                    max_level_e = level[j];
                }
            }
        }
        return max_level_v + (max_level_v == max_level_e ? 1 : 0);
    }

    void FindRoot()
    {
        sz = graph.GetVertexCount();
        // only one here
        root_ = 0;
        hop_ = 10;
        for (uintV u = 0; u < sz; u++)
        {
            int tmp = GetMinHopWithBFS(u);
            if (tmp < hop_)
            {
                root_ = u;
                hop_ = tmp;
                root_degree_ = graph.GetRowPtrs()[u + 1] - graph.GetRowPtrs()[u];
            }
            else if (tmp == hop_ && root_degree_ < graph.GetRowPtrs()[u + 1] - graph.GetRowPtrs()[u])
            {
                root_ = u;
                hop_ = tmp;
                root_degree_ = graph.GetRowPtrs()[u + 1] - graph.GetRowPtrs()[u];
            }
        }
    }

    void GenerateSearchSequence()
    {
        // (hop, degree, id)
        std::vector<std::tuple<int, int, uintV>> weights(sz);
        std::vector<bool> visited(sz, false);
        visited[root_] = true;
        std::queue<uintV> queue;
        queue.push(root_);
        int hop = 0;
        weights[root_] = std::tuple<int, int, uintV>(hop, graph.GetRowPtrs()[root_ + 1] - graph.GetRowPtrs()[root_], root_);
        while (!queue.empty())
        {
            hop++;
            int size = queue.size();
            for (int i = 0; i < size; i++)
            {
                uintV front = queue.front();
                queue.pop();
                for (size_t i = graph.GetRowPtrs()[front]; i < graph.GetRowPtrs()[front + 1]; ++i)
                {
                    uintV j = graph.GetCols()[i];
                    if (!visited[j])
                    {
                        visited[j] = true;
                        queue.push(j);
                        weights[j] = std::tuple<int, int, uintV>(hop, graph.GetRowPtrs()[j + 1] - graph.GetRowPtrs()[j], j);
                    }
                }
            }
        }
        std::sort(weights.begin(), weights.end(), [](const auto& a, const auto& b) {
            if (std::get<0>(a) != std::get<0>(b))
                return std::get<0>(a) < std::get<0>(b);
            else if (std::get<1>(a) != std::get<1>(b))
                return std::get<1>(a) > std::get<1>(b);
            else if (std::get<2>(a) != std::get<2>(b))
                return std::get<2>(a) < std::get<2>(b);
            return false;
        });

        matchOrderHost.resize(sz);
        ID2orderHost.resize(sz);
        for (size_t i = 0; i < weights.size(); ++i)
        {
            uintV w = std::get<2>(weights[i]);
            matchOrderHost[i] = w;
            ID2orderHost[w] = i;
        }
    }

    void GenerateBackwardNeighbor()
    {
        std::vector<bool> visited(sz, false); 
    
        backNeighborCountHost = new uintV[sz];
        parentHost = new uintV[sz];
        std::fill(backNeighborCountHost, backNeighborCountHost + sz, 0);
        backNeighborsHost = new uintV[sz * sz];

        visited[matchOrderHost[0]] = true;
        for (size_t i = 1; i < sz; ++i)
        {
            uintV vertex = matchOrderHost[i];
            for (uintV j = graph.GetRowPtrs()[vertex]; j < graph.GetRowPtrs()[vertex + 1]; ++j)
            {
                uintV nv = graph.GetCols()[j];
                if (visited[nv])
                {
                    backNeighborsHost[i * sz + backNeighborCountHost[i]] = nv;
                    backNeighborCountHost[i]++;
                    parentHost[i] = nv;
                }
            }
            visited[vertex] = true;
        }

        shareIntersectionHost = new bool[sz];
        shareIntersectionHost[0] = false;
        

        for (size_t i = 1; i < sz; ++i)
        {
            if (backNeighborCountHost[i] > 2)
            {
                shareIntersectionHost[i] = true; // TODO: change this two part will be expanded only
                continue;
            }
            else if (backNeighborCountHost[i] == 2)
            {
                uintV last_vertex = matchOrderHost[i - 1];
                if (backNeighborsHost[i * sz] != last_vertex &&
                    backNeighborsHost[i * sz + 1] != last_vertex)
                {
                    shareIntersectionHost[i] = true; // TODO: 
                    continue;
                }
            }
            shareIntersectionHost[i] = false;
        }

        // // if enabled: prefix-only 
        // shareIntersectionHost[1] = false;
        // for (size_t i = 2; i < sz; ++i)
        // {   
        //     shareIntersectionHost[i] = true;
        // }
    }

    void GeneratePreAfterBackwardNeighbor()
    {
        preBackNeighborCountHost = new uintV[sz];
        afterBackNeighborCountHost = new uintV[sz];
        std::fill(preBackNeighborCountHost, preBackNeighborCountHost + sz, 0);
        std::fill(afterBackNeighborCountHost, afterBackNeighborCountHost + sz, 0);

        preBackNeighborsHost = new uintV[sz * sz];
        afterBackNeighborsHost = new uintV[sz * sz];

        for (size_t i = 2; i < sz; ++i)
        {
            if (shareIntersectionHost[i])
            {
                uintV vertex = matchOrderHost[i];
                for (uintV j = graph.GetRowPtrs()[vertex]; j < graph.GetRowPtrs()[vertex + 1]; ++j) 
                {
                    uintV nv = graph.GetCols()[j];
                    if (ID2orderHost[nv] < ID2orderHost[vertex] - 1)
                    {
                        preBackNeighborsHost[i * sz + preBackNeighborCountHost[i]] = nv;
                        preBackNeighborCountHost[i]++;
                    }
                    else if (ID2orderHost[nv] == ID2orderHost[vertex] - 1)
                    {
                        afterBackNeighborsHost[i * sz + afterBackNeighborCountHost[i]] = nv;
                        afterBackNeighborCountHost[i]++;
                    }
                }
            }
        }
    }

    void GenerateUsefulOrder()
    {
        bool skip;
        condOrderHost = new uintV[sz * sz * 2];
        condNumHost = new uintV[sz];
        std::fill(condNumHost, condNumHost + sz, 0);

        preCondOrderHost = new uintV[sz * sz * 2];
        preCondNumHost = new uintV[sz];
        std::fill(preCondNumHost, preCondNumHost + sz, 0);

        afterCondOrderHost = new uintV[sz * sz * 2];
        afterCondNumHost = new uintV[sz];
        std::fill(afterCondNumHost, afterCondNumHost + sz, 0);

        for (size_t i = 0; i < sz; ++i)
        {
            size_t index = sz * i * 2;
            for (size_t j = 0; j < sz; ++j)
            {
                if (ID2orderHost[i] > ID2orderHost[j])
                {
                    skip = false;
                    // check if there exists larger or less relationship
                    for (size_t k = 0; k < graph.order_[i].size(); ++k)
                    {
                        if (graph.order_[i][k].second == j)
                        {
                            condOrderHost[index] = graph.order_[i][k].first;
                            condOrderHost[index + 1] = j;
                            index += 2;
                            condNumHost[i] ++;
                            skip = true;
                            break;
                        }
                    }
                    if (!skip)
                    {
                        condOrderHost[index] = CondOperator::NON_EQUAL;
                        condOrderHost[index + 1] = j;
                        index += 2;
                        condNumHost[i] ++;
                    }
                }
            }
        }

        for (size_t i = 0; i < sz; ++i)
        {
            size_t index = sz * i * 2;
            for (size_t j = 0; j < sz; ++j)
            {
                if (ID2orderHost[i] - 1 > ID2orderHost[j])
                {
                    skip = false;
                    // check if there exists larger or less relationship
                    for (size_t k = 0; k < graph.order_[i].size(); ++k)
                    {
                        if (graph.order_[i][k].second == j)
                        {
                            preCondOrderHost[index] = graph.order_[i][k].first;
                            preCondOrderHost[index + 1] = j;
                            index += 2;
                            preCondNumHost[i] ++;
                            skip = true;
                            break;
                        }
                    }
                    if (!skip)
                    {
                        preCondOrderHost[index] = CondOperator::NON_EQUAL;
                        preCondOrderHost[index + 1] = j;
                        index += 2;
                        preCondNumHost[i] ++;
                    }
                }
            }
        }
        for (size_t i = 0; i < sz; ++i)
        {
            size_t index = sz * i * 2;
            for (size_t j = 0; j < sz; ++j)
            {
                if (ID2orderHost[i] - 1 == ID2orderHost[j])
                {
                    skip = false;
                    // check if there exists larger or less relationship
                    for (size_t k = 0; k < graph.order_[i].size(); ++k)
                    {
                        if (graph.order_[i][k].second == j)
                        {
                            afterCondOrderHost[index] = graph.order_[i][k].first;
                            afterCondOrderHost[index + 1] = j;
                            index += 2;
                            afterCondNumHost[i] ++;
                            skip = true;
                            break;
                        }
                    }
                    if (!skip)
                    {
                        afterCondOrderHost[index] = CondOperator::NON_EQUAL;
                        afterCondOrderHost[index + 1] = j;
                        index += 2;
                        afterCondNumHost[i] ++;
                    }
                }
            }
        }
    }

    void GenerateStoreStrategy()
    {
        strategyHost.resize(sz + 1);

        // initial plan
        for (size_t i = 0; i < sz; ++i)
        {
            if(plan_strategy=="expand")
                strategyHost[i] = StoreStrategy::EXPAND;
            else{
                if (!shareIntersectionHost[i])
                    strategyHost[i] = StoreStrategy::EXPAND;
                else
                    strategyHost[i] = StoreStrategy::PREFIX;
            }
        }
        strategyHost[sz] = StoreStrategy::COUNT;

        movingLvlHost = new ui[sz];
        std::fill(movingLvlHost, movingLvlHost + sz, 1);

        for (size_t i = 1; i < sz; ++i)
        {
            ui same_BN_cnt = 1; // including itself
            for (size_t j = i + 1; j < sz; ++j)
            {
                bool same_BN = true;
                // two consecutive vertices have the same backward neighbor
                if (backNeighborCountHost[i] == backNeighborCountHost[j])
                {
                    for (size_t k = 0; k < backNeighborCountHost[i]; ++k)
                    {
                        if (backNeighborsHost[i * sz + k] != backNeighborsHost[j * sz + k]) 
                        {
                            same_BN = false;
                            break;
                        }
                    }
                }
                else 
                    same_BN = false;
                if (same_BN)
                    same_BN_cnt += 1;
                else
                    break;
            }
            movingLvlHost[i] = same_BN_cnt;
        }
    }

    int GetHop() { return hop_; }

    std::vector<uintV> GetSequence() { return matchOrderHost; }

    std::vector<uintV> GetReverseSequence() { return ID2orderHost; }


    Graph graph;
    size_t sz;
    size_t root_degree_;
    int hop_;
    uintV root_;
    std::vector<uintV> matchOrderHost; //matchOrderHost
    std::vector<ui> ID2orderHost;//ID2orderHost
    bool *shareIntersectionHost;//shareIntersectionHost

    uintV *backNeighborCountHost;
    uintV *backNeighborsHost;
    uintV *parentHost;
    uintV *condOrderHost;
    uintV *condNumHost;

    uintV *preBackNeighborCountHost;
    uintV *preBackNeighborsHost;
    uintV *preCondOrderHost;
    uintV *preCondNumHost;

    uintV *afterBackNeighborCountHost;
    uintV *afterBackNeighborsHost;
    uintV *afterCondOrderHost;
    uintV *afterCondNumHost;

    std::vector<StoreStrategy> strategyHost;//strategyHost

    ui *movingLvlHost;//movingLvlHost

};



#endif