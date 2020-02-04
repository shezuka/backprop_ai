#ifndef AI_EXEC_NEURON_H
#define AI_EXEC_NEURON_H

#include "types.h"

namespace ai {
    class Neuron {
    protected:
        uint _index;
        Connections _connections;
        Input _input = 0.0;
        Output _output = 0.0;
        Error _error = 0.0;
        bool _is_bias = false;

    public:
        explicit Neuron(uint index, uint connections_num);
        virtual ~Neuron();

        Input input() const;
        void input(Input value);
        Output output() const;
        void output(Output value);

        Error error() const;
        void error(Error value);

        void feed(Layer *prev_layer);
        void back_prop(Layer *next_layer);

        static Output activate(Input input);
        static Output back_activate(Output output);
    };
}

#endif //AI_EXEC_NEURON_H
