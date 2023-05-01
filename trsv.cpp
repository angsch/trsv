#include "template_utils.hpp"

#include <algorithm>
#include <cassert>

namespace {
extern "C" {
    void xerbla_(
        const char *srname,
        int *info
#ifdef FORTRAN_STRLEN_END
        , size_t
#endif
        );
}

#ifdef FORTRAN_STRLEN_END
    #define xerbla(...) xerbla_(__VA_ARGS__, 1)
#else
    #define xerbla(...) xerbla_(__VA_ARGS__)
#endif


#define A(i,j) A[(size_t)(i) + (size_t)ldA*(j)]

enum uplo_t {
    UPPER = 0,
    LOWER = 1
};

enum diag_t {
    UNIT   = 0,
    NOUNIT = 1
};

template <typename Prec>
inline __attribute__((always_inline))
int check_param(char uplo, char trans, char diag, int n, int ldA, int incx)
{
    bool upper = (uplo == 'U' || uplo == 'u');
    bool lower = (uplo == 'L' || uplo == 'l');
    bool notrans = (trans == 'N' || trans == 'n');
    bool ttrans = (trans == 'T' || trans == 't');
    bool ctrans = (trans == 'C' || trans == 'c');
    bool unit = (diag == 'U' || diag == 'u');
    bool nounit = (diag == 'N' || diag == 'n');

    int info = 0;
    if (!upper && !lower) {
        info = -1;
    }
    else if (!notrans && !ttrans && !ctrans) {
        info = -2;
    }
    else if (!unit && ! nounit) {
        info = -3;
    }
    else if (n < 0) {
        info = -4;
    }
    else if (ldA < std::max(1, n)) {
        info = -6;
    }
    else if (incx == 0) {
        info = -8;
    }

    if (info != 0) {
        const char *form = "";
        if constexpr (std::is_same_v<Prec, float>) {
            const char *rout = "STRSV";
            xerbla(rout, &info);
        }
        else if constexpr (std::is_same_v<Prec, double>) {
            const char *rout = "DTRSV";
            xerbla(rout, &info);
        }
        else if constexpr (std::is_same_v<Prec, std::complex<float>>) {
            const char *rout = "CTRSV";
            xerbla(rout, &info);
        }
        else if constexpr (std::is_same_v<Prec, std::complex<double>>) {
            const char *rout = "ZTRSV";
            xerbla(rout, &info);
        }
    }

    return info;
}

} // namespace


// upper, notrans
template<typename Prec, diag_t nounit>
void trsv_un(int n, const Prec *__restrict__ A, int ldA, Prec *__restrict__ x)
{
    for (int j = n-1; j >= 0; j--) {
        if constexpr (nounit) {
            x[j] = x[j] / A[j + ldA * j];
        }

        for (int i = j-1; i >= 0; i--) {
            x[i] = x[i] - A[i + j * ldA] * x[j];
        }
    }
}



// lower, notrans
template<typename Prec, diag_t nounit>
void trsv_ln(int n, const Prec *__restrict__ A, int ldA, Prec *__restrict__ x)
{
    for (int j = 0; j < n; j++) {
        if constexpr (nounit) {
            x[j] = x[j] / A[j + ldA * j];
        }

        for (int i = j + 1; i < n; i++) {
            x[i] = x[i] - A[i + j * ldA] * x[j];
        }
    }
}




template<typename Prec>
void trsv(char uplo, char trans, char diag, int n,
    const Prec *__restrict__ A, int ldA, Prec *__restrict__ x, int incx)
{
    int info = check_param<Prec>(uplo, trans, diag, n, ldA, incx);
    // Quick return if possible.
    if (info != 0 || n == 0) {
        return;
    }

    bool upper = (uplo == 'U' || uplo == 'u');
    bool unit = (diag == 'U' || diag == 'u');
    bool notrans = (trans == 'N' || trans == 'n');

    if (incx == 1) {
        if (notrans) {
            // A*x = b
            if (upper) {
                if (unit) {
                    trsv_un<Prec, UNIT>(n, A, ldA, x);
                }
                else {
                    trsv_un<Prec, NOUNIT>(n, A, ldA, x);
                }
            }
            else { // lower
                if (unit) {
                    trsv_ln<Prec, UNIT>(n, A, ldA, x);
                }
                else {
                    trsv_ln<Prec, NOUNIT>(n, A, ldA, x);
                }
            }
        }
        else {
            // A**T*x = b or A**H*x = b

        }
    }
    else { // incx != 1
        // TODO: slow reference
    }

}

template
void trsv(char uplo, char trans, char diag, int n,
          const double *__restrict__ A, int ldA,
          double *__restrict__ x, int incx);

template
void trsv(char uplo, char trans, char diag, int n,
          const float *__restrict__ A, int ldA,
          float *__restrict__ x, int incx);

template
void trsv(char uplo, char trans, char diag, int n,
          const std::complex<float> *__restrict__ A, int ldA,
          std::complex<float> *__restrict__ x, int incx);

template
void trsv(char uplo, char trans, char diag, int n,
          const std::complex<double> *__restrict__ A, int ldA,
          std::complex<double> *__restrict__ x, int incx);