#include "connection.h"

#include <random>

namespace ai {
    double random() {
        static std::random_device device;
        static std::mt19937_64 gen(device());
        static std::uniform_real_distribution<double> dist(0.0, 1.0);
        return dist(gen);
    }

    Connection::Connection() {
        this->weight = random();
    }
}
