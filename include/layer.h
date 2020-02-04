#ifndef AI_EXEC_LAYER_H
#define AI_EXEC_LAYER_H

#include "types.h"
#include <vector>
#include <json/json.h>

namespace ai {
    class Layer {
        std::vector<Neuron *> _neurons;
        Layer *_prev_layer = nullptr;
        Layer *_next_layer = nullptr;

        void feed();
        void back_prop();

        explicit Layer() = default;

    public:
        explicit Layer(uint neurons_num, uint next_layer_neurons);

        ~Layer();

        void next_layer(Layer *layer);

        void feed(const Inputs &inputs);

        void back_prop(const Outputs &outputs);

        const Neuron *neuron(uint index) const;

        const size_t neurons_count() const;

        Json::Value toJson() const;

        [[nodiscard]] static Layer* fromJson(const Json::Value &val);
    };
}

#endif //AI_EXEC_LAYER_H
