#include "ai.h"
#include <cstring>
#include <iostream>

using namespace std;

int main() {
    ai::Ai brain({2, 8, 1});

    for (size_t i = 0; i < 10000; ++i) {
        brain.feed({1.0, 1.0})->back_prop({1.0});
        brain.feed({0.0, 1.0})->back_prop({0.0});
        brain.feed({0.0, 0.0})->back_prop({1.0});
        brain.feed({1.0, 0.0})->back_prop({0.0});
    }

    cout << "Checking single output neural network" << endl;
    cout << "Expects 1: " << brain.feed({1.0, 1.0})->output(0) << endl;
    cout << "Expects 1: " << brain.feed({0.0, 0.0})->output(0) << endl;
    cout << "Expects 0: " << brain.feed({1.0, 0.0})->output(0) << endl;
    cout << "Expects 0: " << brain.feed({0.0, 1.0})->output(0) << endl;

    ai::Ai brain2({2, 8, 2});
    for (size_t i = 0; i < 10000; ++i) {
        brain2.feed({1.0, 1.0})->back_prop({0.0, 1.0});
        brain2.feed({0.0, 1.0})->back_prop({1.0, 0.0});
        brain2.feed({0.0, 0.0})->back_prop({0.0, 1.0});
        brain2.feed({1.0, 0.0})->back_prop({1.0, 0.0});
    }

    cout << "Checking multiple outputs neural network" << endl;
    cout << "Expects 0, 1: "
         << brain2.feed({1.0, 1.0})->output(0) << ", "
         << brain2.output(1)
         << endl;
    cout << "Expects 0, 1: "
         << brain2.feed({0.0, 0.0})->output(0) << ", "
         << brain2.output(1)
         << endl;
    cout << "Expects 1, 0: "
         << brain2.feed({0.0, 1.0})->output(0) << ", "
         << brain2.output(1)
         << endl;
    cout << "Expects 1, 0: "
         << brain2.feed({1.0, 0.0})->output(0) << ", "
         << brain2.output(1)
         << endl;

    auto brain3 = ai::Ai::fromJson(brain2.toJson().toStyledString());
    cout << "After parse from JSON:" << endl;
    cout << "Expects 0, 1: "
         << brain3->feed({1.0, 1.0})->output(0) << ", "
         << brain3->output(1)
         << endl;
    cout << "Expects 0, 1: "
         << brain3->feed({0.0, 0.0})->output(0) << ", "
         << brain3->output(1)
         << endl;
    cout << "Expects 1, 0: "
         << brain3->feed({0.0, 1.0})->output(0) << ", "
         << brain3->output(1)
         << endl;
    cout << "Expects 1, 0: "
         << brain3->feed({1.0, 0.0})->output(0) << ", "
         << brain3->output(1)
         << endl;

    return 0;
}
