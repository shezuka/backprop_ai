#include "ai.h"

#include <iostream>
#include <random>
#include <cmath>

using namespace std;

template <typename ElementType>
ostream& operator << (ostream &out, const vector<ElementType> &container) {
    out << "{ ";
    for (const auto &val: container) {
        out << val << "; ";
    }
    out << "}";
    return out;
}

template <typename SubElementType>
ostream& operator << (ostream &out, const vector<vector<SubElementType>> &container) {
    out << "{";
    for (const auto &sub_container: container) {
        out << "\t" << sub_container << ", ";
    }
    out << "}";
    return out;
}

int main() {
    vector<vector<double>> train_input;
    vector<vector<double>> train_output;

    random_device device;
    mt19937_64 mt(device());
    uniform_real_distribution<double> doubler(0.0, 1.0);
    uniform_int_distribution<int> inter(0, 1);

    for (size_t i = 0; i < 20; ++i) {
        if (inter(mt)) {
            const auto val = round(doubler(mt));
            train_input.push_back({val, val});
            train_output.push_back({1.0});
        } else {
            train_input.push_back({round(doubler(mt)), round(doubler(mt))});
            train_output.push_back({0.0});
        }
    }

    const size_t INPUT_NUM = 2;
    Ai ai({INPUT_NUM, 8, 2});
    ai.train(train_input, train_output, 100000);

    for (const auto &input: train_input) {
        cout << input << " = " << ai.feed_forward(input)->output() << endl;
    }

    return 0;
}