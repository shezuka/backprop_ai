#ifndef AI_EXEC_TYPES_H
#define AI_EXEC_TYPES_H

#include <vector>

namespace ai {
    class Layer;

    class Neuron;

    class Bias;

    class Connection;

    using uint = unsigned int;
    using Input = double;
    using Output = double;
    using Error = double;

    using Topology = std::vector<uint>;
    using Inputs = std::vector<Input>;
    using Outputs = std::vector<Output>;
    using Layers = std::vector<Layer*>;
    using Connections = std::vector<Connection *>;
}

#endif //AI_EXEC_TYPES_H
