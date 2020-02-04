#include "ai.h"

#include "layer.h"
#include "neuron.h"

namespace ai {
    Ai::Ai(const Topology &topology) {
        for (uint topology_i = 0; topology_i < topology.size() - 1; ++topology_i) {
            const auto neurons_num = topology[topology_i];
            const auto next_neurons_num = topology[topology_i + 1];

            _layers.push_back(new Layer{neurons_num, next_neurons_num});
        }

        // Create output layer
        _layers.push_back(new Layer{topology.back(), 0});

        // Make connection between layers
        for (uint layer_i = 0; layer_i < _layers.size() - 1; ++layer_i) {
            auto layer = _layers[layer_i];
            auto next_layer = _layers[layer_i + 1];

            layer->next_layer(next_layer);
        }
    }

    Ai::~Ai() {
        for (auto layer: _layers) {
            delete layer;
        }
    }

    Ai *Ai::feed(const Inputs &inputs) {
        _layers.front()->feed(inputs);
        return this;
    }

    Ai *Ai::back_prop(const Outputs &outputs) {
        _layers.back()->back_prop(outputs);
        return this;
    }

    Output Ai::output(uint index) const {
        return _layers.back()->neuron(index)->output();
    }
}
