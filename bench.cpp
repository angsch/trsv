#include <chrono>
#include <complex>
#include <initializer_list>
#include <iostream>
#include <vector>

#include "generator.hpp"
#include "template_utils.hpp"
#include "trsv.hpp"
#include "trsv_internal.hpp"

using namespace std::chrono;
using namespace internal;

namespace {

template<typename Prec>
microseconds bench(kernel_t variant, char uplo, char trans, char diag, int n,
    const Prec *__restrict__ A, int ldA, Prec *__restrict__ x, int incx)
{
    void (*trsv_var[])(char, char, char, int, const double *__restrict__ , int, double *__restrict__) = {
        trsv_selector<Prec, USE_BLAS_CALL>,
        trsv_selector<Prec, UNROLL_1>,
        trsv_selector<Prec, UNROLL_2>
    };

    auto time_start = high_resolution_clock::now();
    trsv_var[variant](uplo, trans, diag, n, A, ldA, x);
    auto time_end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(time_end - time_start);
    std::cout << "  Time elapsed = " << duration.count() << " µs" << std::endl;
    return duration;
}
} // namespace


int main(int argc, char **argv)
{
    constexpr int maxn = 4000;
    int ldA = maxn;

    //typedef std::complex<double> prec_t;
    typedef double prec_t;

    std::vector<prec_t> x = std::vector<prec_t>(maxn);
    std::vector<prec_t> A = std::vector<prec_t>(maxn * ldA);
    int incx = 1;

    generator<prec_t> rg;

    for (char uplo : {'U', 'L'}) {
        for (char diag : {'N', 'U'}) {
            for (char trans : {'C', 'T', 'N'}) {
                for (int n : {100, 500, 1000, maxn}) {
                    for (kernel_t kernel_type : {USE_BLAS_CALL, UNROLL_1, UNROLL_2}) {
                        std::cout << "Kernel variant = " << kernel_type_to_str(kernel_type) << ", "
                                  << "uplo = " << uplo << " , diag = " << diag << " , trans = " << trans 
                                  << " , n = " << n << std::endl;
                        // Generate random right-hand side vector, and a triangular matrix.
                        rg.generate_general_matrix(n, 1, x.data(), n);
                        rg.generate_triangular_matrix(uplo, diag, n, A.data(), ldA);

                        bench<prec_t>(kernel_type, uplo, trans, diag, n, A.data(), ldA, x.data(), incx);
                        std::cout << std::endl << std::endl;
                    }
                }
            }
        }
    }

    return EXIT_SUCCESS;
}