#ifndef AI_AI_HPP
#define AI_AI_HPP

#include "neuron.hpp"

#include <vector>

using namespace std;

class Ai {
    vector<Layer> _layers;

public:
    Ai(const vector<int> &topology, const vector<int> &output_values) {
        // Init input and hidden layers
        for (size_t layer_index = 0; layer_index < topology.size() - 1; layer_index++) {
            _layers.emplace_back();
            const size_t neurons_num = topology[layer_index];
            const size_t next_layer_neurons = topology[layer_index + 1];
            for (size_t neuron_index = 0; neuron_index < neurons_num; neuron_index++) {
                _layers.back().emplace_back(neuron_index, next_layer_neurons);
            }
        }

        // Init output layer
        _layers.emplace_back();
        const size_t output_layer_neurons = topology[topology.size() - 1];
        for (size_t neuron_index = 0; neuron_index < output_layer_neurons; neuron_index++) {
            _layers.back().emplace_back(neuron_index, 0);
            _layers.back().back().set_value(output_values[neuron_index]);
        }
    }

    Ai* feed_forward(const vector<double> &seed) {
        // Init input layer
        for (size_t i = 0; i < seed.size(); i++) {
            _layers.front()[i].set_output(seed[i]);
        }

        // Feed forward
        for (size_t i = 1; i < _layers.size(); i++) {
            Layer &prev_layer = _layers[i - 1];
            Layer &layer = _layers[i];
            for (Neuron &neuron: layer) {
                neuron.feed_forward(prev_layer);
            }
        }

        return this;
    }

    Ai* back_prop(const vector<double> &seed) {
        for (Neuron &output_neuron: _layers.back()) {
            output_neuron.calc_error(seed[output_neuron.index()]);
        }

        for (size_t i = _layers.size() - 2; i != ((size_t)-1); i--) {
            Layer &next_layer = _layers[i + 1];
            for (Neuron &neuron: _layers[i]) {
                neuron.back_prop(next_layer);
            }
        }

        return this;
    }

    const Neuron &top_neuron() const {
        const Layer &output = _layers.back();
        const Neuron *neuron = nullptr;
        for (const Neuron &n: output) {
            if (neuron == nullptr || n.output() > neuron->output()) {
                neuron = &n;
            }
        }
        return *neuron;
    }

    double top_output() const {
        return top_neuron().output();
    }

    double top_value() const {
        return top_neuron().value();
    }
};

#endif //AI_AI_HPP
