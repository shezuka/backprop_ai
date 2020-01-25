#ifndef AI_NEURON_H
#define AI_NEURON_H

#include <cstdlib>
#include <cmath>
#include <vector>
#include <memory>

using namespace std;

class Neuron;
using Layer = vector<Neuron*>;

struct Connection {
    double weight = 0.0;
    double delta = 0.0;
    double change = 0.0;
};

class Neuron {
protected:
    const size_t _index;
    double _input = 0.0;
    double _output = 0.0;
    double _error = 0.0;
    vector<Connection> _connections;
    int _value = 0;

    double LEARN_RATE = 0.1;

public:
    Neuron(size_t index, size_t connections_num);

    virtual void feed_forward(const Layer &prev_layer);

    void back_prop(const Layer &next_layer);

    void calc_error(double expected_output);

    size_t index() const;

    double output() const;

    void set_output(double output);

    int value() const;

    void set_value(int value);

    static double activate_sigmoid(double x);

    static double activate(double x);

    static double reverse_activate_sigmoid(double x);

    static double reverse_activate(double x);

    static double random_weight();
};

#endif //AI_NEURON_H
