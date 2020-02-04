#include "ai.h"

#include "layer.h"
#include "neuron.h"

namespace ai {
    void Ai::initLayersConnections() {
        for (uint layer_i = 0; layer_i < _layers.size() - 1; ++layer_i) {
            auto layer = _layers[layer_i];
            auto next_layer = _layers[layer_i + 1];

            layer->next_layer(next_layer);
        }
    }

    Ai::Ai(const Topology &topology) {
        for (uint topology_i = 0; topology_i < topology.size() - 1; ++topology_i) {
            const auto neurons_num = topology[topology_i];
            const auto next_neurons_num = topology[topology_i + 1];

            _layers.push_back(new Layer{neurons_num, next_neurons_num});
        }

        // Create output layer
        _layers.push_back(new Layer{topology.back(), 0});

        initLayersConnections();
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

    Json::Value Ai::toJson() const {
        Json::Value layersJson{Json::ValueType::arrayValue};
        for (const auto layer: _layers) {
            layersJson.append(layer->toJson());
        }

        Json::Value value;
        value["layers"] = layersJson;

        return value;
    }

    Ai *Ai::fromJson(const std::string &json) {
        auto reader = Json::CharReaderBuilder{}.newCharReader();

        Json::Value root;
        std::string errors;
        reader->parse(json.c_str(), json.c_str() + json.length(), &root, &errors);

        auto ai = new Ai{};

        Json::Value layersJson = root["layers"];
        ai->_layers.reserve(layersJson.size());
        for (size_t i = 0; i < layersJson.size(); ++i) {
            auto layer = Layer::fromJson(layersJson.get(i, {}));
            ai->_layers.push_back(layer);
        }

        ai->initLayersConnections();

        return ai;
    }
}
