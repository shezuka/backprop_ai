#include "neuron.h"

#include "connection.h"
#include "layer.h"

#include <cmath>

const double LEARN_RATE = 1.0;

namespace ai {
    Neuron::Neuron(const uint index, const uint connections_num) {
        _index = index;
        for (uint i = 0; i < connections_num; ++i)
            _connections.push_back(new Connection{});
    }

    Neuron::~Neuron() {
        for (auto connection: _connections) {
            delete connection;
        }
    }

    Input Neuron::input() const {
        return _input;
    }

    void Neuron::input(Input value) {
        _input = value;
    }

    Output Neuron::output() const {
        return _output;
    }

    void Neuron::output(Output value) {
        _output = value;
    }

    Error Neuron::error() const {
        return _error;
    }

    void Neuron::error(Error value) {
        _error = value;
    }

    void Neuron::feed(Layer *prev_layer) {
        _input = 0.0;
        for (size_t i = 0; i < prev_layer->neurons_count(); ++i) {
            auto neuron = prev_layer->neuron(i);
            _input += neuron->_connections[_index]->weight * neuron->_output;
        }
        _output = activate(_input);
    }

    Output Neuron::activate(Input input) {
        return 1 / (1 + exp(-input));
    }

    Output Neuron::back_activate(Output output) {
        const auto activated = activate(output);
        return activated * (1 - activated);
    }

    void Neuron::back_prop(Layer *next_layer) {
        Error total = 0.0;
        for (size_t i = 0; i < next_layer->neurons_count(); ++i) {
            const auto neuron = next_layer->neuron(i);
            if (neuron->_is_bias) continue;

            total += neuron->_error * _connections[neuron->_index]->weight;
            _connections[neuron->_index]->delta = LEARN_RATE * neuron->_error * _output;
            _connections[neuron->_index]->weight += _connections[neuron->_index]->delta;
        }

        _error = total * back_activate(_input);
    }
}
