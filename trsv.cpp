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

enum conj_t {
    CONJ   = 0,
    NOCONJ = 1
};

template<typename Prec, conj_t noconj>
constexpr Prec conjg(Prec x) {
    if constexpr (is_floating_point_value<Prec>) {
        return x;
    }
    else {
        if constexpr (noconj) {
            return x;
        }
        else {
            return conj(x);
        }
    }
}


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
            x[j] = x[j] / A(j,j);
        }

        for (int i = j-1; i >= 0; i--) {
            x[i] = x[i] - A(i,j) * x[j];
        }
    }
}

template<typename Prec, diag_t nounit>
void trsv_un_unroll2(int n, const Prec *__restrict__ A, int ldA, Prec *__restrict__ x)
{
    // If necessary, do a single iteration to make n even.
    if (n % 2 == 1) {
        if constexpr (nounit) {
            x[n-1] = x[n-1] / A(n-1,n-1);
        }
        for (int i = n-2; i >= 0; i--) {
            x[i] = x[i] - A(i, n-1) * x[n-1];
        }

        n = n - 1;
    }

    // At this point, n is even.
    for (int j = n-1; j >= 0; j-=2) {
        if constexpr (nounit) {
            x[j] = x[j] / A(j,j);
        }
        x[j-1] = x[j-1] - A(j-1,j) * x[j];
        if constexpr (nounit) {
            x[j-1] = x[j-1] / A(j-1,j-1);
        }

        for (int i = j-2; i >= 0; i--) {
            x[i] = x[i] - A(i,j) * x[j] - A(i,j-1) * x[j-1];
        }
    }
}


// lower, notrans
template<typename Prec, diag_t nounit>
void trsv_ln(int n, const Prec *__restrict__ A, int ldA, Prec *__restrict__ x)
{
    for (int j = 0; j < n; j++) {
        if constexpr (nounit) {
            x[j] = x[j] / A(j,j);
        }

        for (int i = j + 1; i < n; i++) {
            x[i] = x[i] - A(i,j) * x[j];
        }
    }
}

template<typename Prec, diag_t nounit>
void trsv_ln_unroll2(int n, const Prec *__restrict__ A, int ldA, Prec *__restrict__ x)
{
    // If necessary, do a single iteration to make the iteration count even.
    int j = 0;
    if (n % 2 == 1) {
        if constexpr (nounit) {
            x[0] = x[0] / A(0,0);
        }
        for (int i = 1; i < n; i++) {
            x[i] = x[i] - A(i,0) * x[0];
        }

        j++;
    }

    // At this point, an even number of iteration is needed.
    for (; j < n; j+=2) {
        if constexpr (nounit) {
            x[j] = x[j] / A(j,j);
        }
        x[j+1] = x[j+1] - A(j+1,j) * x[j];
        if constexpr (nounit) {
            x[j+1] = x[j+1] / A(j+1,j+1);
        }

        for (int i = j+2; i < n; i++) {
            x[i] = x[i] - A(i,j) * x[j] - A(i,j+1) * x[j+1];
        }
    }
}


// upper, transpose/conjugate transpose
template<typename Prec, diag_t nounit, conj_t noconj>
void trsv_ut(int n, const Prec *__restrict__ A, int ldA, Prec *__restrict__ x)
{
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < j; i++) {
            x[j] = x[j] - conjg<Prec, noconj>(A(i,j)) * x[i];
        }
        if constexpr (nounit) {
            x[j] = x[j] / conjg<Prec, noconj>(A(j,j));
        }
    }
}

template<typename Prec, diag_t nounit, conj_t noconj>
void trsv_ut_unroll_2(int n, const Prec *__restrict__ A, int ldA, Prec *__restrict__ x)
{
    // If necessary, do a single iteration at the end to make the iteration count even.
    int last = (n % 2 == 0) ? n : n-1;

    // Execute an even number of iterations.
    for (int j = 0; j < last; j+=2) {
        for (int i = 0; i < j; i++) {
            x[j] = x[j] - conjg<Prec, noconj>(A(i,j)) * x[i];
            x[j+1] = x[j+1] - conjg<Prec, noconj>(A(i,j+1)) * x[i];
        }
        if constexpr (nounit) {
            x[j] = x[j] / conjg<Prec, noconj>(A(j,j));
        }
        x[j+1] = x[j+1] - conjg<Prec, noconj>(A(j,j+1)) * x[j];
        if constexpr (nounit) {
            x[j+1] = x[j+1] / conjg<Prec, noconj>(A(j+1,j+1));
        }
    }

    // Tail.
    if (last < n) {
        for (int i = 0; i < n-1; i++) {
            x[n-1] = x[n-1] - conjg<Prec, noconj>(A(i,n-1)) * x[i];
        }
        if constexpr (nounit) {
            x[n-1] = x[n-1] / conjg<Prec, noconj>(A(n-1,n-1));
        }
    }
}

// lower, transpose/conjugate transpose
template<typename Prec, diag_t nounit, conj_t noconj>
void trsv_lt(int n, const Prec *__restrict__ A, int ldA, Prec *__restrict__ x)
{
    for (int j = n-1; j >= 0; j--) {
        for (int i = n-1; i > j; i--) {
            x[j] = x[j] - conjg<Prec, noconj>(A(i,j)) * x[i];
        }
        if constexpr (nounit) {
            x[j] = x[j] / conjg<Prec, noconj>(A(j,j));
        }
    }
}

template<typename Prec, diag_t nounit, conj_t noconj>
void trsv_lt_unroll2(int n, const Prec *__restrict__ A, int ldA, Prec *__restrict__ x)
{
    // If necessary, do a single iteration at the end to make the iteration count even.
    int last = (n % 2 == 0) ? 0 : 1;

    for (int j = n-1; j >= last; j-=2) {
        for (int i = n-1; i > j; i--) {
            x[j] = x[j] - conjg<Prec, noconj>(A(i,j)) * x[i];
            x[j-1] = x[j-1] - conjg<Prec, noconj>(A(i,j-1)) * x[i];
        }
        if constexpr (nounit) {
            x[j] = x[j] / conjg<Prec, noconj>(A(j,j));
        }

        x[j-1] = x[j-1] - conjg<Prec, noconj>(A(j,j-1)) * x[j];
        if constexpr (nounit) {
            x[j-1] = x[j-1] / conjg<Prec, noconj>(A(j-1,j-1));
        }
    }

    // Tail.
    if (last != 0) {
        for (int i = n-1; i > 0; i--) {
            x[0] = x[0] - conjg<Prec, noconj>(A(i,0)) * x[i];
        }
        if constexpr (nounit) {
            x[0] = x[0] / conjg<Prec, noconj>(A(0,0));
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
                    //trsv_un<Prec, UNIT>(n, A, ldA, x);
                    trsv_un_unroll2<Prec, UNIT>(n, A, ldA, x);
                }
                else {
                    //trsv_un<Prec, NOUNIT>(n, A, ldA, x);
                    trsv_un_unroll2<Prec, NOUNIT>(n, A, ldA, x);
                }
            }
            else { // lower
                if (unit) {
                    //trsv_ln<Prec, UNIT>(n, A, ldA, x);
                    trsv_ln_unroll2<Prec, UNIT>(n, A, ldA, x);
                }
                else {
                    //trsv_ln<Prec, NOUNIT>(n, A, ldA, x);
                    trsv_ln_unroll2<Prec, NOUNIT>(n, A, ldA, x);
                }
            }
        }
        else {
            // A**T*x = b or A**H*x = b
            if (upper) {
                if constexpr (is_floating_point_value<Prec>) {
                    if (unit) {
                        //trsv_ut<Prec, UNIT, NOCONJ>(n, A, ldA, x);
                        trsv_ut_unroll_2<Prec, UNIT, NOCONJ>(n, A, ldA, x);
                    }
                    else {
                        //trsv_ut<Prec, NOUNIT, NOCONJ>(n, A, ldA, x);
                        trsv_ut_unroll_2<Prec, NOUNIT, NOCONJ>(n, A, ldA, x);
                    }
                }
                else {
                    bool noconj = (trans == 'T' || trans == 't');
                    if (noconj) {
                        if (unit) {
                            //trsv_ut<Prec, UNIT, NOCONJ>(n, A, ldA, x);
                            trsv_ut_unroll_2<Prec, UNIT, NOCONJ>(n, A, ldA, x);
                        }
                        else {
                            //trsv_ut<Prec, NOUNIT, NOCONJ>(n, A, ldA, x);
                            trsv_ut_unroll_2<Prec, NOUNIT, NOCONJ>(n, A, ldA, x);
                        }
                    }
                    else {
                        if (unit) {
                            //trsv_ut<Prec, UNIT, CONJ>(n, A, ldA, x);
                            trsv_ut_unroll_2<Prec, UNIT, CONJ>(n, A, ldA, x);
                        }
                        else {
                            //trsv_ut<Prec, NOUNIT, CONJ>(n, A, ldA, x);
                            trsv_ut_unroll_2<Prec, NOUNIT, CONJ>(n, A, ldA, x);
                        }
                    }
                }
            }
            else { // lower
                if constexpr (is_floating_point_value<Prec>) {
                    if (unit) {
                        trsv_lt<Prec, UNIT, NOCONJ>(n, A, ldA, x);
                    }
                    else {
                        trsv_lt<Prec, NOUNIT, NOCONJ>(n, A, ldA, x);
                    }
                }
                else {
                    bool noconj = (trans == 'T' || trans == 't');
                    if (noconj) {
                        if (unit) {
                            //trsv_lt<Prec, UNIT, NOCONJ>(n, A, ldA, x);
                            trsv_lt_unroll2<Prec, UNIT, NOCONJ>(n, A, ldA, x);
                        }
                        else {
                            //trsv_lt<Prec, NOUNIT, NOCONJ>(n, A, ldA, x);
                            trsv_lt_unroll2<Prec, NOUNIT, NOCONJ>(n, A, ldA, x);
                        }
                    }
                    else {
                        if (unit) {
                            //trsv_lt<Prec, UNIT, CONJ>(n, A, ldA, x);
                            trsv_lt_unroll2<Prec, UNIT, CONJ>(n, A, ldA, x);
                        }
                        else {
                            //trsv_lt<Prec, NOUNIT, CONJ>(n, A, ldA, x);
                            trsv_lt_unroll2<Prec, NOUNIT, CONJ>(n, A, ldA, x);
                        }
                    }
                }
            }
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