#include "ai.h"
#include "neuron_bias.h"

Ai::Ai(const vector<int> &topology, bool use_bias) {
    // Init input and hidden layers
    for (size_t layer_index = 0; layer_index < topology.size() - 1; layer_index++) {
        _layers.emplace_back();
        const size_t neurons_num = topology[layer_index];
        const size_t next_layer_neurons = topology[layer_index + 1] + use_bias;
        for (size_t neuron_index = 0; neuron_index < neurons_num; neuron_index++) {
            _layers.back().push_back(new Neuron(neuron_index, next_layer_neurons));
        }

        if (use_bias) {
            _layers.back().push_back(new NeuronBias(_layers.back().size(), next_layer_neurons));
        }
    }

    // Init output layer
    _layers.emplace_back();
    const size_t output_layer_neurons = topology[topology.size() - 1];
    for (size_t neuron_index = 0; neuron_index < output_layer_neurons; neuron_index++) {
        _layers.back().push_back(new Neuron(neuron_index, 0));
    }
}

Ai *Ai::feed_forward(const vector<double> &seed) {
    // Init input layer
    auto &input_layer = _layers.front();
    for (size_t i = 0; i < seed.size(); i++) {
        auto neuron = input_layer[i];
        neuron->set_output(seed[i]);
    }

    // Feed forward
    for (size_t i = 1; i < _layers.size(); i++) {
        Layer &prev_layer = _layers[i - 1];
        Layer &layer = _layers[i];
        for (auto &neuron: layer) {
            neuron->feed_forward(prev_layer);
        }
    }

    return this;
}

const Neuron &Ai::neuron() const {
    const Layer &output = _layers.back();
    const Neuron *neuron = nullptr;
    for (const auto &n: output) {
        if (neuron == nullptr || n->output() > neuron->output()) {
            neuron = n;
        }
    }
    return *neuron;
}

double Ai::output() const {
    return neuron().output();
}

void Ai::train(const vector<vector<double>> &inputs, const vector<vector<double>> &outputs, unsigned generations) {
    for (unsigned generation = 0; generation < generations; generation++) {
        for (size_t i = 0; i < inputs.size(); i++) {
            const auto &input_part = inputs[i];
            const auto &output_part = outputs[i];
            this->feed_forward(input_part)->back_prop(output_part);
        }
    }
}

Ai::~Ai() {
    for (const auto &layer: _layers) {
        for (const auto &neuron: layer) {
            delete neuron;
        }
    }
    _layers.clear();
}

Ai *Ai::back_prop(const vector<double> &seed) {
    for (auto &output_neuron: _layers.back()) {
        output_neuron->calc_error(seed[output_neuron->index()]);
    }

    for (size_t i = _layers.size() - 2; i != ((size_t) -1); i--) {
        Layer &next_layer = _layers[i + 1];
        for (auto &neuron: _layers[i]) {
            neuron->back_prop(next_layer);
        }
    }

    return this;
}
