//
// Created by Felix Moeran on 19/07/2023.
//

#pragma once

#include <vector>
#include <memory>

struct Vector;
struct Matrix;

namespace operators
{
    void add(float* a, float* b, float* result, size_t size);
    void add(float* a, float b, float* result, size_t size);
    void mul(float* a, float b, float* result, size_t size);

    void vecMatMul(const Matrix&a, const Vector& b, Vector& result);
    void vecMatMul(const Vector& a, const Matrix& b, Vector& result);
    float dot(const Vector& a, const Vector& b);

}



struct Vector{
public:
    size_t rows;
    std::unique_ptr<float[]> data;
    Vector(int size);

    void operator+=(float scalar);
    void operator+=(const Vector& vector);

    void operator*=(float scalar);

    float& operator[](size_t ind);

    void operator=(std::vector<float> vec);

    // sets data pointer to that of other
    // removes pointer from other
    // NOTE other should not be used again as other.data will equal nullptr
    void replaceWith(Vector& other);

    friend std::ostream &operator<<(std::ostream &os, Vector const &v);
};



struct Matrix {
public:
    size_t rows, cols;
    std::unique_ptr<float[]> data;
    size_t size;
    Matrix(int rows, int cols);

    float& at(int row, int col) const;

    void operator+=(float scalar);
    void operator+=(const Matrix& matrix);

    void operator*=(float scalar);

    friend std::ostream &operator<<(std::ostream &os, Matrix const &m);
};



