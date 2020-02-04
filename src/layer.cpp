#include "layer.h"

#include "bias.h"

#include <cassert>

namespace ai {
    Layer::Layer(const uint neurons_num, const uint next_layer_neurons) {
        for (uint i = 0; i < neurons_num; ++i) {
            _neurons.push_back(new Neuron{i, next_layer_neurons});
        }

        _neurons.push_back(new Bias{neurons_num, next_layer_neurons});
    }

    Layer::~Layer() {
        for (auto neuron: _neurons) {
            delete neuron;
        }
    }

    void Layer::next_layer(Layer *layer) {
        _next_layer = layer;
        layer->_prev_layer = this;
    }

    void Layer::feed(const Inputs &inputs) {
        assert(inputs.size() == _neurons.size() - 1);

        for (size_t i = 0; i < _neurons.size() - 1; ++i) {
            _neurons[i]->output(inputs[i]);
        }

        _next_layer->feed();
    }

    void Layer::feed() {
        for (size_t i = 0; i < _neurons.size() - 1; ++i) {
            _neurons[i]->feed(_prev_layer);
        }

        if (_next_layer) _next_layer->feed();
    }

    void Layer::back_prop(const Outputs &outputs) {
        assert(outputs.size() == _neurons.size() - 1);

        for (size_t i = 0; i < _neurons.size() - 1; ++i) {
            auto neuron = _neurons[i];
            neuron->error((outputs[i] - neuron->output()) * Neuron::back_activate(neuron->input()));
        }

        _prev_layer->back_prop();
    }

    void Layer::back_prop() {
        for (const auto layer: _neurons) {
            layer->back_prop(_next_layer);
        }

        if (_prev_layer) _prev_layer->back_prop();
    }

    const Neuron *Layer::neuron(uint index) const {
        return _neurons[index];
    }

    const size_t Layer::neurons_count() const {
        return _neurons.size();
    }
}
