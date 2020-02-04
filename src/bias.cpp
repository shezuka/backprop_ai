#include "bias.h"

namespace ai {
    Bias::Bias(const uint index, const uint connections_num) : Neuron(index, connections_num) {
        _input = _output = 1.0;
        _is_bias = true;
    }
}
