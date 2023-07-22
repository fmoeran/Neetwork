//
// Created by Felix Moeran on 17/07/2023.
//

#pragma once

enum CostType {
    MSE
};

template<CostType Type>
float costFunction(float a, float y);



template<CostType Type>
float costDerivative(float a, float y);

