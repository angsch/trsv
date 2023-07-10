#include "generator.hpp"

template<typename T>
void generator<T>::generate_general_matrix(int num_rows, int num_cols, T *mat, int ld) {
    for (int j = 0; j < num_cols; j++) {
        for (int i = 0; i < num_rows; i++) {
            mat[(size_t)i + (size_t)j * ld] = (*this)(i, j);
        }
    }
}

template<typename T>
void generator<T>::generate_triangular_matrix(char uplo, char diag, int n, T *mat, int ld) {
    // Decode uplo.
    bool upper = (uplo == 'U') || (uplo == 'u');
    bool lower = (uplo == 'L') || (uplo == 'l');
    if (!upper && !lower)
        return;

    // Decode diag.
    bool unit = (diag == 'U') || (diag == 'u');
    bool nounit = (diag == 'N') || (diag == 'n');
    if (!unit && !nounit)
        return;

    if (upper) {
        for (int j = 0; j < n; j++) {
            // Set lower triangular part to zero for readability.
            for (int i = 0; i < j; i++) {
                mat[(size_t)i + (size_t)j * ld] = T(0.0);
            }

            // Diagonal.
            if (unit) {
                // For readability, typically implicit.
                mat[(size_t)j + (size_t)j * ld] = T(1.0);
            }
            else {
                mat[(size_t)j + (size_t)j * ld] = T(2.0*n) + (*this)(j, j);
            }

            // Upper triangular part.
            for (int i = j + 1; i < n; i++) {
                mat[(size_t)i + (size_t)j * ld] = (*this)(i, j);
            }
        }
    }
    if (lower) {
        for (int j = 0; j < n; j++) {
            // Lower triangular part.
            for (int i = j; i < n; i++) {
                mat[(size_t)i + (size_t)j * ld] = (*this)(j, j);
            }

            // Diagonal.
            if (unit) {
                // For readability, typically implicit.
                mat[(size_t)j + (size_t)j * ld] = T(1.0);
            }
            else {
                mat[(size_t)j + (size_t)j * ld] = T(2.0*n) + (*this)(j, j);
            }

            // Set upper triangular part to zero for readability.
            for (int i = 0; i < j; i++) {
                mat[(size_t)i + (size_t)j * ld] = T(0.0);
            }
        }
    }
    #undef A
}


template<>
float generator<float>::operator()() {
  return d(gen);
}

template<>
float generator<float>::operator()(int row, int col) {
  return d(gen);
}

template<>
double generator<double>::operator()() {
  return d(gen);
}

template<>
double generator<double>::operator()(int row, int col) {
  return d(gen);
}

template<>
std::complex<float> generator<std::complex<float>>::operator()() {
  return std::complex<float>{ d(gen), d(gen) };
}

template<>
std::complex<float> generator<std::complex<float>>::operator()(int row, int col) {
  return std::complex<float>{ d(gen), d(gen) };
}

template<>
std::complex<double> generator<std::complex<double>>::operator()() {
  return std::complex<double>{ d(gen), d(gen) };
}

template<>
std::complex<double> generator<std::complex<double>>::operator()(int row, int col) {
  return std::complex<double>{ d(gen), d(gen) };
}

