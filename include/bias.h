#ifndef AI_EXEC_BIAS_H
#define AI_EXEC_BIAS_H

#include "neuron.h"

namespace ai {
    class Bias : public Neuron {
    public:
        explicit Bias(uint index, uint connections_num);
    };
}

#endif //AI_EXEC_BIAS_H
