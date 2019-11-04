#ifndef AI_NEURON_HPP
#define AI_NEURON_HPP

#include <cstdlib>
#include <cmath>
#include <vector>
#include <iostream>

using namespace std;

class Neuron;
using Layer = vector<Neuron>;

struct Connection {
    double weight = 0.0;
    double delta = 0.0;
    double change = 0.0;
};

class Neuron {
    const size_t _index;
    double _input = 0.0;
    double _output = 0.0;
    double _error = 0.0;
    vector<Connection> _connections;
    int _value = 0;

    double LEARN_RATE = 0.5;

public:
    Neuron(size_t index, size_t connections_num): _index(index) {
        while(_connections.size() != connections_num) {
            _connections.emplace_back();
            _connections.back().weight = random_weight();
        }
    }

    void feed_forward(const Layer &prev_layer) {
        _input = 0.0;
        for (auto &prev_neuron: prev_layer) {
            _input += (prev_neuron._output * prev_neuron._connections[_index].weight);
        }
        _output = activate(_input);
    }

    void back_prop(const Layer &next_layer) {
        double total_error = 0.0;
        for (const Neuron &next_neuron: next_layer) {
            total_error += next_neuron._error * _connections[next_neuron._index].weight;
            _connections[next_neuron._index].delta = LEARN_RATE * next_neuron._error * _output;
            _connections[next_neuron._index].change = LEARN_RATE * next_neuron._error;
        }
        _error = total_error * reverse_activate(_input);

        for (const Neuron &next_neuron: next_layer) {
            _connections[next_neuron._index].weight = _connections[next_neuron._index].weight + _connections[next_neuron._index].delta;
        }
    }

    void calc_error(double expected_output) {
        _error = (expected_output - _output) * reverse_activate(_input);
    }

    size_t index() const {
        return _index;
    }

    double output() const {
        return _output;
    }

    void set_output(double output) {
        _output = output;
    }

    int value() const {
        return _value;
    }

    void set_value(int value) {
        _value = value;
    }

    static double activate_sigmoid(double x) {
        return 1 / (1 + exp(-x));
    }

    static double activate(double x) {
        return activate_sigmoid(x);
    }

    static double reverse_activate_sigmoid(double x) {
        return activate(x) * (1 - activate(x));
    }

    static double reverse_activate(double x) {
        return reverse_activate_sigmoid(x);
    }

    static double random_weight() {
        double f = (double)rand() / RAND_MAX;
        return 0.0 + f * (1.0 - 0.0);
    }
};

#endif //AI_NEURON_HPP
