//
// Created by Felix Moeran on 17/07/2023.
//

#include "activation.hpp"

template<>
float activationFunction<ActivationType::Sigmoid>(float z) {
    return 1.0f / (1 + std::exp(-z));
}

template<>
float activationDerivative<ActivationType::Sigmoid>(float z) {
    float res = activationFunction<Sigmoid>(z);
    return res * (1 - res);
}


