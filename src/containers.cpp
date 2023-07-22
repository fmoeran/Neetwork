//
// Created by Felix Moeran on 19/07/2023.
//
#include "containers.hpp"
#include <string>
#include <iostream>
#include <utility>
#include <new>
#include <immintrin.h>

namespace operators {
    void add(float *a, float *b, float *result, size_t size) {

#if defined(SIMD)
        __m256 aVec, bVec, resVec;
        int i = 0;
        std::cout << ((size_t)a % 32)<< std::endl;
        std::cout << ((size_t)b % 32)<< std::endl;
        for (; i + 8 < size; i += 8) {
            std::cout << "x" << std::endl;
            aVec = _mm256_loadu_ps(a + i);

            bVec = _mm256_loadu_ps(b + i);
            std::cout << "y" << std::endl;
            resVec = _mm256_add_ps(aVec, bVec);
            _mm256_storeu_ps(result + i, resVec);


        }


        for (; i < size; i++) {
            result[i] = a[i] + b[i];
        }
#else // SIMD
        while (size--) {
            *result = (*a) + (*b);
            a++;
            b++;
            result++;
        }
#endif
    }


    void add(float *a, float b, float *result, size_t size) {
#if defined(SIMD)
        __m256 aVec, bVec, resVec;
        bVec = _mm256_set1_ps(b);
        int i = 0;
        for (; i + 8 < size; i += 8) {
            aVec = _mm256_loadu_ps(a + i);
            resVec = _mm256_add_ps(aVec, bVec);
            _mm256_storeu_ps(result + i, resVec);
        }
        for (; i < size; i++) {
            result[i] = a[i] + b;
        }
#else // SIMD

        while (size--) {
            *result = (*a) + b;
            a++; result++;
        }
#endif
    }


    void mul(float *a, float b, float *result, size_t size) {
#if defined(SIMD)
        __m256 aVec, bVec, resVec;
        bVec = _mm256_set1_ps(b);
        int i = 0;
        for (; i + 8 < size; i += 8) {
            aVec = _mm256_loadu_ps(a + i);
            resVec = _mm256_mul_ps(aVec, bVec);
            _mm256_storeu_ps(result + i, resVec);
        }
        for (; i < size; i++) {
            result[i] = a[i] * b;
        }
#else // SIMD
        while (size--) {
            *result = (*a) * b;
            a++;
            result++;
        }
#endif
    }

    void mul(const Vector &a, const Matrix &b, Vector &result) {
        mul(b, a, result);
    }

    void mul(const Matrix &a, const Vector &b, Vector &result) {
#if defined(SIMD)

        __m256 aVec, bVec, prodVec;

        for (int row=0; row<a.rows; row++) {
            float* pta = a.data.get() + row * a.cols;
            float* ptb = b.data.get() + row;

            int i = 0;
            result[i] = 0;
            for (; i + 8 < a.cols; i += 8) {

                aVec = _mm256_loadu_ps(pta + i);


                bVec = _mm256_loadu_ps(ptb + i);


                prodVec = _mm256_mul_ps(aVec, bVec);

                prodVec = _mm256_hadd_ps(prodVec, prodVec);
                prodVec = _mm256_hadd_ps(prodVec, prodVec);

                int x = _mm256_extract_epi32(prodVec, 0);
                result[i] += *(float*) &x;
                x = _mm256_extract_epi32(prodVec, 2);
                result[i] += *(float*) &x;
            }
        }
#else // SIMD

        //assert(a.cols == b.rows && a.rows == result.rows);
        std::memset(result.data.get(), 0, result.rows * sizeof(float));

        for (int i = 0; i < a.rows; i++) {
            for (int j = 0; j < b.rows; j++) {
                result.data[i] += a.at(i, j) * b.data[j];
            }
        }
#endif
    }

    float dot(const Vector &a, const Vector &b) {
        //assert(a.rows == b.rows);
        float out = 0.0f;
        for (int i = 0; i < a.rows; i++) {
            out += a.data[i] * b.data[i];
        }
        return out;
    }
}


Vector::Vector(int size): rows(size) {
    data = std::unique_ptr<float[]>(new(std::align_val_t(32)) float[size]);
    std::memset(data.get(), 0, rows*sizeof(float));

}

void Vector::operator+=(float scalar) {
    operators::add(data.get(), scalar, data.get(), rows);
}

void Vector::operator+=(const Vector &vector) {
    assert(rows == vector.rows);
    std::cout << "adding" << std::endl;
    operators::add(data.get(), vector.data.get(), data.get(), rows);
}

void Vector::operator*=(float scalar) {
    operators::mul(data.get(), scalar, data.get(), rows);
}

float &Vector::operator[](size_t ind) {
    return data[ind];
}

void Vector::operator=(std::vector<float> vec) {
    //assert(rows == vec.size());

    std::memcpy(data.get(), vec.begin().base(), rows * sizeof(float));
}

void Vector::replaceWith(Vector &other) {
    data = std::move(other.data);
    rows = other.rows;
}

std::ostream &operator<<(std::ostream &os, const Vector &v) {
    for (int r=0; r<v.rows; r++) {
        os << v.data[r];
        os << std::string(" ");
    }
    return os;
}

Matrix::Matrix(int rows, int cols) : rows(rows), cols(cols), size(rows * cols){
    data = std::unique_ptr<float[]>(new(std::align_val_t(32)) float[size]);
    std::memset(data.get(), 0, rows*cols*sizeof(float));
}

float& Matrix::at(int row, int col) const {
    //assert(row < rows && col < cols);
    return  data[row * cols + col];
}

void Matrix::operator+=(float scalar) {
    operators::add(data.get(), scalar, data.get(), size);
}
void Matrix::operator+=(const Matrix &matrix) {
    operators::add(data.get(), matrix.data.get(), data.get(), size);
}
void Matrix::operator*=(float scalar) {
    operators::mul(data.get(), scalar, data.get(), size);
}

std::ostream &operator<<(std::ostream &os, Matrix const &m) {
    for (int r=0; r<m.rows; r++) {
        for (int c=0; c<m.cols; c++) {
            os << std::to_string(m.at(r, c));
            os << std::string(" ");
        }
        os << std::string("\n");
    }
    return os;
}

/*
int main() {
    Vector a(1000000000);
    Vector b(1000000000);
    b += 400;
    a += 200;
    std::cout << "go" << std::endl;
    a += b;
}*/
