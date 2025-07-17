#pragma once

struct MCContext
{
    vector<ui> R;
    vector<ui> P;
    vector<ui> X;
};

using MCTask = Task<MCContext>;

enum{R,P,X,Q};