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
struct random_engine<T, std::enable_if_t<std::is_integral_v<T>>> {
    using gen = std::mt19937;
    using d = std::uniform_int_distribution<T>;
};

template<typename T>
struct random_engine<T, std::enable_if_t<is_floating_point_value<remove_complex_t<T>>>> {
    using gen = std::mt19937;
    using d = std::normal_distribution<remove_complex_t<T>>;
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
    void generate_upper_quasitriangular_matrix(int n, T *mat, int ld);
};


template class generator<float>;
template class generator<double>;
template class generator<std::complex<float>>;
template class generator<std::complex<double>>;


template<typename Integer, std::enable_if_t<std::is_integral<Integer>::value, bool> = true>
class index_generator {
    typename random_engine<Integer>::d d;
    typename random_engine<Integer>::gen gen;

public:
    ~index_generator() {};

    Integer operator()(Integer i);

    index_generator(Integer low, Integer high, unsigned seed = 0) {
        gen.seed(seed);
        decltype(d.param()) new_range (low, high);
        d.param(new_range);
    }

    void generate_1d_index_vector(int n, Integer* vec) {
        for (int i = 0; i < n; i++)
            vec[i] = (*this)(i);
    }
};

template class index_generator<int>;
template class index_generator<long>;