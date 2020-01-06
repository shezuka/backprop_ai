#include "ai.h"

Ai::Ai(const vector<int> &topology, const vector<int> &output_values) {
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

Ai *Ai::feed_forward(const vector<double> &seed) {
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

const Neuron &Ai::top_neuron() const {
    const Layer &output = _layers.back();
    const Neuron *neuron = nullptr;
    for (const Neuron &n: output) {
        if (neuron == nullptr || n.output() > neuron->output()) {
            neuron = &n;
        }
    }
    return *neuron;
}

double Ai::top_output() const {
    return top_neuron().output();
}

int Ai::top_value() const {
    return top_neuron().value();
}

unsigned
Ai::train(const vector<vector<double>> &inputs, const vector<vector<double>> &outputs, const vector<int> &output_values,
          unsigned int MAX_GENERATION) {
    for (unsigned generation = 0; generation < MAX_GENERATION; generation++) {
        // Train
        for (size_t i = 0; i < inputs.size(); i++) {
            const auto &input_part = inputs[i];
            const auto &output_part = outputs[i];
            this->feed_forward(input_part)->back_prop(output_part);
        }

        // Check
        bool all_success = true;
        for (size_t i = 0; i < inputs.size(); i++) {
            const auto &input_part = inputs[i];
            const auto &output_part = outputs[i];
            if (this->feed_forward(inputs[i])->top_value() != output_values[i]) {
                all_success = false;
                break;
            }
        }

        if (all_success) {
            return generation;
        }
    }

    return MAX_GENERATION;
}
