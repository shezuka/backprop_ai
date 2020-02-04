#ifndef AI_EXEC_AI_H
#define AI_EXEC_AI_H

#include "types.h"
#include <vector>
#include <json/json.h>
#include <string>

namespace ai {
    class Ai {
        Layers _layers;

        explicit Ai() = default;

        void initLayersConnections();

    public:
        explicit Ai(const Topology &topology);

        ~Ai();

        Ai *feed(const Inputs &inputs);

        Ai *back_prop(const Outputs &outputs);

        Output output(uint index) const;

        Json::Value toJson() const;

        [[nodiscard]] static Ai *fromJson(const std::string &json);
    };
}

#endif //AI_EXEC_AI_H
