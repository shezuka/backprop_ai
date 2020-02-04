#ifndef AI_EXEC_CONNECTION_H
#define AI_EXEC_CONNECTION_H

namespace ai {
    struct Connection {
        double weight;
        double delta;

        explicit Connection();
    };
}

#endif //AI_EXEC_CONNECTION_H
