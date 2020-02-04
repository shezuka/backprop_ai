#ifndef AI_EXEC_AI_H
#define AI_EXEC_AI_H

#include "types.h"
#include <vector>

namespace ai {
    class Ai {
        Layers _layers;

    public:
        explicit Ai(const Topology &topology);

        ~Ai();

        Ai *feed(const Inputs &inputs);

        Ai *back_prop(const Outputs &outputs);

        Output output(uint index) const;
    };
}

#endif //AI_EXEC_AI_H
