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

    Json::Value Connection::toJson() const {
        Json::Value value{Json::ValueType::objectValue};
        value["weight"] = weight;
        value["delta"] = delta;

        return value;
    }

    Connection *Connection::fromJson(const Json::Value &val) {
        auto conn = new Connection{};
        conn->weight = val["weight"].asDouble();
        conn->delta = val["delta"].asDouble();
        return conn;
    }
}
