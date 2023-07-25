//
// Created by Felix Moeran on 17/07/2023.
//
#pragma once
#include <iostream>

void vecMatMult(std::vector<float>& res, const std::vector<float>& vec, const std::vector<std::vector<float>>&mat);

template<ActivationType activation>
void applyActivation(const std::vector<float>& vec, std::vector<float>& res);


template<ActivationType activation>
Layer<activation>::Layer(int size, int prevSize)
: values(size), activatedValues(size), weights(size, prevSize), biases(size),
deltaBiases(size), deltaWeights(size, prevSize) {
    width = size;
    std::random_device rd {};
    std::default_random_engine generator {rd()};
    std::normal_distribution<float> distribution;

    for (int i=0; i<width; i++) {
        biases[i] = distribution(generator);
        for (int j=0; j<prevSize; j++) weights.at(i, j) = distribution(generator);
    }


}
template<ActivationType activation>
void Layer<activation>::applyActivation() {
    for (int i=0; i<width; i++) {
        activatedValues[i] = activationFunction<activation>(values[i]);
    }
}

template<class T>
int getPreferredItem(const std::vector<T>& items) {
    int max = 0;
    for (int i=0; i<items.size(); i++) {
        if (items[i] > items[max]) max = i;
    }
    return max;
}

template<CostType cost, ActivationType activation>
Network<cost, activation>::Network(std::vector<int> dimensions, float learningRate): __learningRate(learningRate) {
    // create input layer
    // has no previous layer, the actiation function will be ignored, but we give it anyway
    layers.push_back(Layer<activation>(dimensions[0], 0));
    // create the rest of the layers, skipping the first layer


    for (int i=1; i<dimensions.size(); i++) {
        layers.push_back(Layer<activation>(dimensions[i], dimensions[i-1]));
    }
    __costs.resize(dimensions.back());
}

template<CostType cost, ActivationType activation>
void Network<cost, activation>::train(Data trainingData, int epochs, int batchSize, bool track) {
    Data testData;
    train(trainingData, epochs, testData, batchSize, track);
}

template<CostType cost, ActivationType activation>
void Network<cost, activation>::train(Data trainingData, int epochs, int batchSize, Data testData, bool track) {
    assert(trainingData.count % batchSize == 0);
    int miniBatchCount = trainingData.count / batchSize;
    for (int currentEpoch=0; currentEpoch < epochs; currentEpoch++) {

        trainingData.shuffle();
        int correct = 0;
        for (int miniBatch=0; miniBatch<miniBatchCount; miniBatch++) {
            int miniBatchStart = miniBatch * batchSize;
            int miniBatchEnd = (miniBatch+1) * batchSize;
            clearDeltas();
            for (int i=miniBatchStart; i < miniBatchEnd; i++) {

                feedForward(trainingData.inputs[i]);

                backPropagate(trainingData.targets[i]);

                if (track && getSingleOutput() == getPreferredItem(trainingData.targets[i])) correct++;

            }
            updatePerams(batchSize);

        }


        if (track) {
            std::cout << "Epoch " << currentEpoch << ": Training = " << correct << "/" << trainingData.count;
            if (testData.count > 0) {
                std::cout << " Testing = " << test(testData) << "/" << testData.count;
            }
            std::cout << std::endl;
        }
    }
}

template<CostType cost, ActivationType activation>
int Network<cost, activation>::test(const Data &data) {
    int correct = 0;
    for (int i=0; i<data.count; i++) {
        feedForward(data.inputs[i]);
        if (getSingleOutput() == getPreferredItem(data.targets[i])) correct++;
    }
    return correct;
}

template<CostType cost, ActivationType activation>
int Network<cost, activation>::getSingleOutput() {
    int max = 0;
    for (int i=0; i<layers.back().size(); i++) {
        if (layers.back().activatedValues[i] > layers.back().activatedValues[max])
            max = i;
    }
    return max;
}

template<CostType cost, ActivationType activation>
void Network<cost, activation>::clearDeltas() {
    for (int l=1; l<layers.size(); l++) {
        Layer<activation> &layer = layers[l];
        std::memset(layer.deltaBiases.data.get(), 0, layer.deltaBiases.rows * sizeof(float));
        std::memset(layer.deltaWeights.data.get(), 0, layer.deltaWeights.size * sizeof(float));
    }
}

template<CostType cost, ActivationType activation>
void Network<cost, activation>::feedForward(const std::vector<float>& inputs) {
    assert(inputs.size() == layers[0].width);

    layers[0].activatedValues = inputs;

    for (int i=1; i < layers.size(); i++) {
        operators::vecMatMul(layers[i-1].activatedValues, layers[i].weights, layers[i].values);
        layers[i].values += layers[i].biases;
        layers[i].applyActivation();

    }

}

template<CostType cost, ActivationType activation>
void Network<cost, activation>::loadCosts(const std::vector<float>& target) {
    for (int i=0; i<layers.back().size(); i++) {
        __costs[i] = costFunction<cost>(layers.back().values[i], target[i]);
    }
}

template<CostType cost, ActivationType activation>
void Network<cost, activation>::backPropagate(const std::vector<float>& target) {

    Vector deltas(layers.back().size());
    for (int i=0; i<layers.back().size(); i++) {
        deltas[i] = costDerivative<cost>(layers.back().activatedValues[i], target[i]) *
                    activationDerivative<activation>(layers.back().values[i]);
    }

    for (int l=layers.size()-1; l>0; l--) {
        layers[l].deltaBiases += deltas;

        for (int i=0; i<layers[l].weights.rows; i++) {
            for (int j=0; j<layers[l].weights.cols; j++) {
                layers[l].deltaWeights.at(i, j) += deltas[i] * layers[l-1].activatedValues[j];
            }
        }

        if (l==1) break;

        Vector newDeltas(layers[l-1].size());
        for (int i=0; i<layers[l].size(); i++) {
            for (int j=0; j<layers[l-1].size(); j++) {
                newDeltas[j] += deltas[i] * activationDerivative<activation>(layers[l-1].values[j]) * layers[l].weights.at(i, j);
            }
        }

        deltas.replaceWith(newDeltas);

    }
}

template<CostType cost, ActivationType activation>
void Network<cost, activation>::updatePerams(int batchSize) {
    for (int l=1; l<layers.size(); l++) {
        Layer<activation>& layer = layers[l];

        layer.deltaBiases *= (-__learningRate / (float)batchSize);
        layer.biases += layer.deltaBiases;

        layer.deltaWeights *= (-__learningRate / (float)batchSize);
        layer.weights += layer.deltaWeights;
    }
}

void vecMatMult(std::vector<float>& res, const std::vector<float>& vec, const std::vector<std::vector<float>>&mat) {
    assert(res.size() == mat.size() && vec.size() == mat[0].size());
    std::memset((void*)res.begin().base(), 0, res.size() * sizeof(float));
    for (int i=0; i<res.size(); i++) {
        for (int j=0; j<vec.size(); j++) {
            res[i] += vec[j] * mat[i][j];
        }
    }
}

template<ActivationType activation>
void applyActivation(const std::vector<float>& vec, std::vector<float>& res){
    assert(vec.size() == res.size());
    for (int i=0; i<vec.size(); i++) {
        res[i] = activationFunction<activation>(vec[i]);
    }
}

