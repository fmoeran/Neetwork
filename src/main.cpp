//
// Created by Felix Moeran on 17/07/2023.
//

#include "network.hpp"
#include <iostream>


int main() {
    std::cout << "Initialising network" << std::endl;
    Network<MSE, Sigmoid> net({784, 10, 10}, 0.1);
    std::cout << "Loading training data" << std::endl;
    Data trainingData = getMnistTrainingData();
    Data testData = getMnistTestData();
    std::cout << "Training" << std::endl;

    net.train(trainingData, 2, 10, testData);

    return 0;
}

