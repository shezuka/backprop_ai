#include "ai.hpp"

#include <iostream>

int main() {
    ::srand(::time(nullptr));

    const size_t INPUT_NUM = 2;
    Ai ai({INPUT_NUM, 4, 2}, {1, 0});

    vector<vector<double>> train_input_data = {
            {1.0, 1.0},
            {0.0, 0.0},
            {100.0, 100.0},
            {1000.0, 1000.0},

            {1.0, 0.0},
            {0.0, 1.0},
            {100.0, 0.0},
            {0.0, 100.0},
    };

    vector<vector<double>> train_output_data = {
            {1.0, 0.0},
            {1.0, 0.0},
            {1.0, 0.0},
            {1.0, 0.0},

            {0.0, 1.0},
            {0.0, 1.0},
            {0.0, 1.0},
            {0.0, 1.0},
    };

    vector<int> output_values = {
            1, 1, 1, 1, 0, 0, 0, 0
    };

    const auto generations = ai.train(train_input_data, train_output_data, output_values, 100000);
    cout << "Trained for " << generations << " generations" << endl;

    cout << ai.feed_forward({1, 1})->top_value() << endl;
    cout << ai.feed_forward({0, 0})->top_value() << endl;
    cout << ai.feed_forward({0, 1})->top_value() << endl;
    cout << ai.feed_forward({1, 0})->top_value() << endl;
    cout << endl;


    cout << ai.feed_forward({20, 20})->top_value() << endl;
    cout << ai.feed_forward({20, 20})->top_value() << endl;
    cout << ai.feed_forward({99999, 99999})->top_value() << endl;
    cout << ai.feed_forward({99999, 99999})->top_value() << endl;
    cout << endl;


    cout << ai.feed_forward({20, 0})->top_value() << endl;
    cout << ai.feed_forward({0, 20})->top_value() << endl;
    cout << ai.feed_forward({0, 99999})->top_value() << endl;
    cout << ai.feed_forward({99999, 0})->top_value() << endl;
    cout << endl;

    return 0;
}