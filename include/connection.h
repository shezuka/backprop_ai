#ifndef AI_EXEC_CONNECTION_H
#define AI_EXEC_CONNECTION_H

#include <json/json.h>

namespace ai {
    struct Connection {
        double weight;
        double delta;

        explicit Connection();

        Json::Value toJson() const;

        [[nodiscard]] static Connection *fromJson(const Json::Value &val);
    };
}

#endif //AI_EXEC_CONNECTION_H
