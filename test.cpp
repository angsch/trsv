#include <complex>
#include <initializer_list>
#include <iostream>
#include <vector>

#include "generator.hpp"
#include "template_utils.hpp"
#include "trsv.hpp"
#include "trsv_internal.hpp"


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
remove_complex_t<Prec> deviation(Prec x, Prec y)
{
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

template <typename Prec>
remove_complex_t<Prec> max_vector_difference(int n, const std::vector<Prec>& x,
    const std::vector<Prec>& y)
{
    using Real = remove_complex_t<Prec>;
    Real max_err = 0.0;
    for (int i = 0; i < n; i++) {
        Real err = deviation(y[i], x[i]);
        max_err = std::max(max_err, err);
    }
    return max_err;
}

template<typename Prec>
remove_complex_t<Prec> compare_with_reference_trsv(char uplo, char trans, char diag, int n,
    const std::vector<Prec>& A, int ldA, std::vector<Prec>& x, int incx)
{
    using namespace::internal;
    using Real = remove_complex_t<Prec>;

    // Take two copies of the initial right-hand side.
    std::vector<Prec> y(x);
    std::vector<Prec> orig_rhs(x);

    // Compute the solution with reference-LAPACK (Fortran).
    if constexpr (std::is_same_v<Prec, double>) {
        dtrsv_(&uplo, &trans, &diag, &n, A.data(), &ldA, x.data(), &incx);
    }
    else if constexpr (std::is_same_v<Prec, float>) {
        strsv_(&uplo, &trans, &diag, &n, A.data(), &ldA, x.data(), &incx);
    }
    else if constexpr (std::is_same_v<Prec, std::complex<float>>) {
        ctrsv_(&uplo, &trans, &diag, &n, to_fcmplx(A.data()), &ldA, to_fcmplx(x.data()), &incx);
    }
    else if constexpr (std::is_same_v<Prec, std::complex<double>>) {
        ztrsv_(&uplo, &trans, &diag, &n, to_dcmplx(A.data()), &ldA, to_dcmplx(x.data()), &incx);
    }
    else {
        __builtin_unreachable();
    }

    int (*trsv_var[])(char, char, char, int, const Prec *__restrict__ , int,
                      Prec *__restrict__, int) = {
        trsv_selector<Prec, USE_BLAS_CALL>,
        trsv_selector<Prec, UNROLL_1>,
        trsv_selector<Prec, UNROLL_2>
    };

    Real max_err = 0.0;
    // Test all kernel variants
    for (kernel_t kernel_type : {USE_BLAS_CALL, UNROLL_1, UNROLL_2}) {
        trsv_var[kernel_type](uplo, trans, diag, n, A.data(), ldA, y.data(), incx);
        Real err = max_vector_difference<Prec>(n, x, y);
        max_err = std::max(max_err, err);

        // Restore the original right-hand side.
        y.assign(orig_rhs.begin(), orig_rhs.end());
    }

    // BLAS-inspired interface
    trsv<Prec>(uplo, trans, diag, n, A.data(), ldA, y.data(), incx);
    Real err = max_vector_difference<Prec>(n, x, y);
    max_err = std::max(max_err, err);

    return max_err;
}

inline __attribute__((always_inline))
int check(int returned, int expected) {
    if (returned != expected) {
        return 1;
    }
    else {
        return 0;
    }
}

template<typename Prec>
void test_error_exits(bool verbose)
{
    int (*trsv_var)(char, char, char, int, const Prec *__restrict__ , int,
        Prec *__restrict__, int);
    trsv_var = internal::trsv_selector<Prec, internal::UNROLL_1>;

    Prec A[1] = {Prec(42.0)};
    Prec x[1] = {Prec(-42.0)};

    int num_err = 0;

    num_err += check(trsv_var('/', 'N', 'U',  1, A, 1, x, 1), -1);
    num_err += check(trsv_var('U', '/', 'U',  1, A, 1, x, 1), -2);
    num_err += check(trsv_var('U', 'N', '/',  1, A, 1, x, 1), -3);
    num_err += check(trsv_var('U', 'N', 'U', -1, A, 1, x, 1), -4);
    num_err += check(trsv_var('U', 'N', 'U',  1, A, 0, x, 1), -6);
    num_err += check(trsv_var('U', 'N', 'U',  1, A, 1, x, 0), -8);

    if (num_err > 0) {
        std::cout << "There are " << num_err << "invalid exits" << std::endl;
    }
    else {
        std::cout << "All error exist tests passed" << std::endl;
    }
}

template<typename Prec>
void test(bool verbose)
{
    using Real = remove_complex_t<Prec>;

    constexpr int maxn = 10;
    int ldA = maxn;

    Real max_error = 0.0;
    std::vector<Prec> x = std::vector<Prec>(maxn);
    std::vector<Prec> A = std::vector<Prec>(maxn * ldA);
    int incx = 1;

    generator<Prec> rg;

    std::cout << "===================" << std::endl;
    std::cout << "Testing " << prec_to_str<Prec>() << std::endl;
    std::cout << "===================" << std::endl;

    for (char uplo : {'U', 'L'}) {
        for (char diag : {'N', 'U'}) {
            for (char trans : {'C', 'T', 'N'}) {
                for (int n = 0; n < maxn; n++) {
                    if (verbose) {
                        std::cout << "uplo = " << uplo << " , diag = " << diag << " , trans = " 
                                  << trans << " , n = " << n << std::endl;
                    }
                    // Generate random right-hand side vector, and a triangular matrix.
                    rg.generate_general_matrix(n, 1, x.data(), n);
                    rg.generate_triangular_matrix(uplo, diag, n, A.data(), ldA);
                    #ifndef NDEBUG
                    print(n, n, A, ldA, "A = ");
                    print(n, 1, x, n, "x = ");
                    #endif

                    Real err = compare_with_reference_trsv<Prec>(
                        uplo, trans, diag, n, A, ldA, x, incx);
                    #ifndef NDEBUG
                    print(n, 1, x, n, "sol = ");
                    #endif
                    
                    if (verbose) {
                        std::cout << "maximum componentwise error " << err 
                                  << std::endl << std::endl;
                    }

                    max_error = std::max(err, max_error);
                }
            }
        }
    }
    std::cout << "Maximum componentwise error across all tests: " << max_error << std::endl;

    test_error_exits<Prec>(false);
}

} // namespace


int main(int argc, char **argv)
{
    bool verbose = false;
    test<float>(verbose);
    test<double>(verbose);
    test<std::complex<double>>(verbose);
    test<std::complex<float>>(verbose);

    return EXIT_SUCCESS;
}
