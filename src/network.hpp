//
// Created by Felix Moeran on 17/07/2023.
//
#pragma once

#include "activation.hpp"
#include "cost.hpp"
#include "containers.hpp"
#include "dataHandler.hpp"

#include <vector>
#include <random>

template<CostType cost, ActivationType activation>
struct Network;

template<ActivationType activation>
struct Layer {
public:

    Vector biases;
    Matrix weights;
    Vector activatedValues;
    Vector values;
    Layer(int size, int prevSize);
    size_t size() {return width;}

    void applyActivation();

    template<CostType cost, ActivationType act>
    friend struct Network;
private:
    Vector deltaBiases;
    Matrix deltaWeights;
    int width;

};




template<CostType cost, ActivationType activation>
struct Network {
public:

    std::vector<Layer<activation>> layers{};
    Network(std::vector<int> dimensions, float learningRate);

    void train(Data trainingData, int epochs, int batchSize, bool track = true);
    void train(Data trainingData, int epochs, int batchSize, Data testData, bool track = true);

    // returns the number of tests in the data that the network evaluated correctly
    int test(const Data& data);
    // runs the neural network to update every Layer::Node::value
    void feedForward(const std::vector<float>& inputs);
    // returns the output that the network most favored in the most recent pass
    int getSingleOutput();
private:
    float __learningRate;
    std::vector<float> __costs;
    // sets all of the delta values in layers to 0.0
    void clearDeltas();

    // updates __costs
    void loadCosts(const std::vector<float>& target);
    // updates each of layers[i].delta(Weights/Biases)
    void backPropagate(const std::vector<float>& target);
    // alters the weights and biases of each layer according to the delta values
    // @param batchSize the number of items in the current training batch
    void updatePerams(int batchSize);


};




//template definition
//##################################################################################################################

#include "network.tpp"