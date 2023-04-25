#include <chrono>
#include <complex>
#include <initializer_list>
#include <iostream>
#include <vector>

#include "generator.hpp"
#include "template_utils.hpp"

extern "C" {
    // reference-LAPACK
    void dtrsv_(char *uplo, char *trans, char *diag, int *n, const double *A, int *ldA, double *x, int *incx);
    void strsv_(char *uplo, char *trans, char *diag, int *n, const float *A, int *ldA, float *x, int *incx);
    void ctrsv_(char *uplo, char *trans, char *diag, int *n, const float _Complex *A, int *ldA, float _Complex *x, int *incx);
    void ztrsv_(char *uplo, char *trans, char *diag, int *n, const double _Complex *A, int *ldA, double _Complex *x, int *incx);
}

namespace {

template <typename Prec>
inline __attribute__((always_inline))
remove_complex_t<Prec> deviation(Prec x, Prec y) {
    Prec diff = x - y;
    if constexpr (is_complex_value<Prec>) {
        using Real = remove_complex_t<Prec>;
        Real re = diff.real();
        Real im = diff.imag();
        Real cabs1 = std::fabs(re) + std::fabs(im);
        return cabs1;
    }
    else { // real
        return std::fabs(diff);
    }
}

template<typename Prec>
void compare_with_reference_trsv(char uplo, char trans, char diag, int n,
    const Prec *__restrict__ A, int ldA, Prec *__restrict__ x, int incx)
{
    using Real = remove_complex_t<Prec>;

    // Take a copy of the initial right-hand side.
    Prec *__restrict__ y = (Prec *)malloc(n * sizeof(Prec));
    for (int i = 0; i < n; i++) {
        y[i] = x[i];
    }

    // Compute the solution with reference-LAPACK (Fortran).
    if constexpr (std::is_same_v<Prec, double>) {
        dtrsv_(&uplo, &trans, &diag, &n, A, &ldA, x, &incx);
    }
    else if constexpr (std::is_same_v<Prec, float>) {
        strsv_(&uplo, &trans, &diag, &n, A, &ldA, x, &incx);
    }
    else if constexpr (std::is_same_v<Prec, std::complex<float>>) {
        ctrsv_(&uplo, &trans, &diag, &n,
               reinterpret_cast<const __complex__ float*>(A), &ldA,
               reinterpret_cast<__complex__ float*>(x), &incx);
    }
    else if constexpr (std::is_same_v<Prec, std::complex<double>>) {
        ztrsv_(&uplo, &trans, &diag, &n,
               reinterpret_cast<const __complex__ double*>(A), &ldA,
               reinterpret_cast<__complex__ double*>(x), &incx);
    }
    else {
        __builtin_unreachable();
    }

    // TODO: Compute trsv with y.

    // Componentwise deviation.
    Real max_err = 0.0;
    for (int i = 0; i < n; i++) {
        Real err = deviation(y[i], x[i]);
        max_err = std::max(max_err, err);
    }
    std::cout << "maximum componentwise error " << max_err << std::endl;
    free(y);
}

} // namespace


using namespace std::chrono;

int main(int argc, char **argv)
{
    constexpr int maxn = 10;
    int ldA = maxn;

    typedef std::complex<double> prec_t;
    //typedef double prec_t;

    std::vector<prec_t> x = std::vector<prec_t>(maxn);
    std::vector<prec_t> A = std::vector<prec_t>(maxn * ldA);
    int incx = 1;

    generator<prec_t> rg;

    for (char uplo : {'U'/*, 'L'*/}) {
        for (char diag : {'N'/*, 'U'*/}) {
            for (char trans : {/*'T', */'N'}) {
                for (int n = 0; n < maxn; n++) {
                    std::cout << "uplo = " << uplo << " , diag = " << diag << " , trans = " << trans << " , n = " << n << std::endl;
                    // Generate random right-hand side vector, and a triangular matrix.
                    rg.generate_general_matrix(n, 1, x.data(), n);
                    rg.generate_triangular_matrix(uplo, diag, n, A.data(), ldA);
                    //print(n, n, A, ldA, "A = ");
                    //print(n, 1, x, n, "x = ");

                    compare_with_reference_trsv<prec_t>(uplo, trans, diag, n, A.data(), ldA, x.data(), incx);
                    //print(n, 1, x, n, "sol = ");

                    std::cout << std::endl << std::endl;
                }
            }
        }
    }

    return EXIT_SUCCESS;
}
