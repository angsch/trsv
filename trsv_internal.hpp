#pragma once


namespace internal {

enum kernel_t {
    USE_BLAS_CALL = 0,
    UNROLL_1      = 1,
    UNROLL_2      = 2
};

inline std::string kernel_type_to_str(kernel_t kernel_type)
{
    if (kernel_type == USE_BLAS_CALL) return "BLAS";
    else if (kernel_type == UNROLL_1) return "unroll_once";
    else if (kernel_type == UNROLL_2) return "unroll_twice";
    else  __builtin_unreachable();
}

template<typename Prec, kernel_t kernel_type>
int trsv_selector(char uplo, char trans, char diag, int n,
    const Prec *__restrict__ A, int ldA, Prec *__restrict__ x, int incx);

template<typename Prec>
inline long trsv_flops(int n)
{
    if constexpr (is_complex_value<Prec>)
        return 4 * long(n) * long(n);
    else
        return long(n) * long(n);
}

} // namespace internal