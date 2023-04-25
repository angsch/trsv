#pragma once

#include <complex>
#include <iostream>
#include <vector>
#include <string>
#include <optional>

template<typename T>
struct is_floating_point {
    constexpr static const bool value = false;
};

template<>
struct is_floating_point<float> {
    constexpr static const bool value = true;
};

template<>
struct is_floating_point<double> {
    constexpr static const bool value = true;
};

template<typename T>
inline constexpr bool is_floating_point_value = is_floating_point<T>::value;

////

template <typename T>
struct is_complex_t : public std::false_type {};

template <typename T>
struct is_complex_t<std::complex<T>> : public std::true_type {};

template <typename T>
struct is_complex_t<const std::complex<T>> : public std::true_type {};

template <typename T>
inline constexpr bool is_complex_value = is_complex_t<T>::value;

////

template <typename T>
struct remove_complex {
    typedef T type;
};
template <typename T>
struct remove_complex<std::complex<T>> {
    typedef T type;
};

template <typename T>
using remove_complex_t = typename remove_complex<T>::type;

////

inline float _Complex *to_fcmplx(std::complex<float> *x) {
    return reinterpret_cast<float _Complex *>(x);
}

inline const float _Complex *to_fcmplx(const std::complex<float> *x) {
    return reinterpret_cast<const float _Complex *>(x);
}

inline double _Complex *to_dcmplx(std::complex<double> *x) {
    return reinterpret_cast<double _Complex *>(x);
}

inline const double _Complex *to_dcmplx(const std::complex<double> *x) {
    return reinterpret_cast<const double _Complex *>(x);
}
////

template <typename T>
void print(int m, int n, const std::vector<T> &mat, int ld, std::string label = "") {
    std::cout << label << std::endl;
    std::cout.precision(6);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if constexpr (is_floating_point_value<T>) { // float, double
                std::cout << std::scientific << mat[i + j * ld] << " ";
            } else { // complex<float>, complex<double>
                std::cout << std::scientific 
                          << "(" << mat[i + j * ld].real() << ", "
                          << mat[i + j * ld].imag() << "i) ";
            }
        }
        std::cout << std::endl;
    }
}

