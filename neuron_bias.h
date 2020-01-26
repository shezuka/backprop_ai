#ifndef AI_NEURONBIAS_H
#define AI_NEURONBIAS_H


#include "neuron.h"

class NeuronBias : public Neuron {
public:
    NeuronBias(size_t index, size_t connections_num);
    void feed_forward(const Layer &prev_layer) override;

    void set_output(double output) override;
};


#endif //AI_NEURONBIAS_H
