#include "ai.hpp"

#include <iostream>

int main() {
    ::srand(::time(nullptr));

    const size_t INPUT_NUM = 2;
    Ai ai({INPUT_NUM, 4, 2}, {1, 0});

    vector<vector<double>> train_data = {
            {1, 1, 1, 0},
            {0, 0, 1, 0},
            {1, 0, 0, 1},
            {0, 1, 0, 1}
    };

    for (size_t i = 0; i < 1000000; i++) {
        for (const auto &train_set: train_data) {
            vector<double> seed;
            vector<double> back_prop_seed;
            for (size_t i = 0; i < train_set.size(); i++) {
                if (i < INPUT_NUM) {
                    seed.push_back(train_set[i]);
                } else {
                    back_prop_seed.push_back(train_set[i]);
                }
            }
            ai.feed_forward(seed)->back_prop(back_prop_seed);
        }

        bool all_success = true;
        for (const auto &train_set: train_data) {
            vector<double> seed;
            int output = 0;
            for (size_t i = 0; i < train_set.size(); i++) {
                if (i < INPUT_NUM) {
                    seed.push_back(train_set[i]);
                } else {
                    output = (int)train_set[i];
                    break;
                }
            }

            if (ai.feed_forward(seed)->top_value() != output) {
                all_success = false;
                break;
            }
        }

        if (all_success) {
            cout << "Finished early" << endl;
            break;
        }
    }

    cout << ai.feed_forward({1, 1})->top_value() << endl;
    cout << ai.feed_forward({0, 0})->top_value() << endl;
    cout << ai.feed_forward({0, 1})->top_value() << endl;
    cout << ai.feed_forward({1, 0})->top_value() << endl;

    return 0;
}