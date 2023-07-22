//
// Created by Felix Moeran on 17/07/2023.
//

#pragma once

#include <vector>
#include <string>
#include <fstream>

// a containor for training/test data for a neural network
struct Data {
    std::vector<std::vector<float>> inputs;
    std::vector<std::vector<float>> targets;
    size_t count;
    Data();
    Data(std::vector<std::vector<float>>& inp, std::vector<std::vector<float>>& targ);

    void shuffle();
};



Data getMnistTrainingData();
Data getMnistTestData();