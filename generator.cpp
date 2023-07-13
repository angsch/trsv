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
            for (int i = j + 1; i < n; i++) {
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
            for (int i = 0; i < j; i++) {
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

template<typename T>
void generator<T>::generate_upper_quasitriangular_matrix(int n, T *mat, int ld) {
    if constexpr (is_complex_value<T>) {
        generate_triangular_matrix('U', 'N', n, mat, ld);
    }
    else {
        // real
        generate_triangular_matrix('U', 'U', n, mat, ld);
        using Real = remove_complex_t<T>;
        int num_complex = n / 2;
        int num_real = n - num_complex;
        for (int i = 0; i < n; i++) {
            mat[i+i*(size_t)ld] = T(n + i + 1.0);
        }

        index_generator<int> rg(0, num_complex, 1);
        int *indices = (int *) malloc(num_real * sizeof(int));
        rg.generate_1d_index_vector(num_real, indices);
        int *gaps = (int *) malloc((num_complex+1) * sizeof(int));
        for (int i = 0; i < num_real; i++) {
            gaps[indices[i]]++;
        }

        int j = 0;
        for (int i = 0; i < num_complex; i++) {
            j = gaps[i] + j;
            Real lambda_re = mat[j+j*(size_t)ld];
            Real lambda_im = std::abs(lambda_re);
            mat[j+j*(size_t)ld] = lambda_re;   mat[j+(j+1)*(size_t)ld] = -lambda_im;
            mat[j+1+j*(size_t)ld] = lambda_im; mat[(j+1)+(j+1)*(size_t)ld]  =  lambda_re;
            j = j + 2;
        }

        free(indices);
        free(gaps);
    }
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

template<>
int index_generator<int>::operator()(int i) {
  return d(gen);
}

template<>
long index_generator<long>::operator()(long i) {
  return d(gen);
}
