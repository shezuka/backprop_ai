#ifndef AI_AI_H
#define AI_AI_H

#include "neuron.h"

#include <vector>

using namespace std;

class Ai {
    vector<Layer> _layers;

public:
    Ai(const vector<int> &topology, const vector<int> &output_values);

    Ai *feed_forward(const vector<double> &seed);

    Ai *back_prop(const vector<double> &seed) {
        for (Neuron &output_neuron: _layers.back()) {
            output_neuron.calc_error(seed[output_neuron.index()]);
        }

        for (size_t i = _layers.size() - 2; i != ((size_t) -1); i--) {
            Layer &next_layer = _layers[i + 1];
            for (Neuron &neuron: _layers[i]) {
                neuron.back_prop(next_layer);
            }
        }

        return this;
    }

    const Neuron &top_neuron() const;

    double top_output() const;

    int top_value() const;

    void train(const vector<vector<double>> &inputs, const vector<vector<double>> &outputs, unsigned generations = 10000);

    unsigned train(const vector<vector<double>> &inputs, const vector<vector<double>> &outputs,
                   const vector<int> &output_values, unsigned MAX_GENERATION = 10000);
};

#endif //AI_AI_H
