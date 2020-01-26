#include "neuron_bias.h"

NeuronBias::NeuronBias(size_t index, size_t connections_num) : Neuron(index, connections_num) {
    _input = _output = 1.0;
}

void NeuronBias::feed_forward(const Layer &prev_layer) {
}

void NeuronBias::set_output(double output) {
}
