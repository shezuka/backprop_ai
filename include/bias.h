#ifndef AI_EXEC_BIAS_H
#define AI_EXEC_BIAS_H

#include "neuron.h"

namespace ai {
    class Bias : public Neuron {
        explicit Bias() = default;

    public:
        explicit Bias(uint index, uint connections_num);

        friend class Layer;
    };
}

#endif //AI_EXEC_BIAS_H
