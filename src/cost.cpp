//
// Created by Felix Moeran on 17/07/2023.
//

#include "cost.hpp"

template<>
float costFunction<CostType::MSE>(float a, float y) {
    return (a - y) * (a - y);
}

template<>
float costDerivative<CostType::MSE>(float a, float y) {
    return 2 * (a - y);
}