#pragma once

#include <memory>
#include <random>
#include <vector>
#include <complex>
#include <type_traits>

#include "template_utils.hpp"


template<typename T, typename = void>
struct random_engine;

template<typename T>
struct random_engine<T, std::enable_if_t<is_floating_point_value<remove_complex_t<T>>>> {
    typedef std::mt19937 gen;
    typedef std::normal_distribution<remove_complex_t<T>> d;
};

template<typename T>
class generator {
    typename random_engine<T>::d d;
    typename random_engine<T>::gen gen;

public:
    ~generator() {};

    T operator()();
    T operator()(int row, int col);

    void generate_general_matrix(int num_rows, int num_cols, T *mat, int ld);
    void generate_triangular_matrix(char uplo, char diag, int n, T *mat, int ld);
};

template class generator<float>;
template class generator<double>;
template class generator<std::complex<float>>;
template class generator<std::complex<double>>;
