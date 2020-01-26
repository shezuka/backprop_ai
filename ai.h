#ifndef AI_AI_H
#define AI_AI_H

#include "neuron.h"

#include <vector>

using namespace std;

class Ai {
    vector<Layer> _layers;

public:
    explicit Ai(const vector<int> &topology, bool use_bias = true);

    ~Ai();

    Ai *feed_forward(const vector<double> &seed);

    Ai *back_prop(const vector<double> &seed);

    const Neuron &neuron() const;

    double output() const;

    void
    train(const vector<vector<double>> &inputs, const vector<vector<double>> &outputs, unsigned generations = 10000);
};

#endif //AI_AI_H
