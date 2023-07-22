//
// Created by Felix Moeran on 17/07/2023.
//

#pragma once
#include <cmath>

enum ActivationType {
    Sigmoid,
};

template <ActivationType Type>
float activationFunction(float z);

template <ActivationType Type>
float activationDerivative(float z);
