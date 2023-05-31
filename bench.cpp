#include <chrono>
#include <complex>
#include <fstream>
#include <initializer_list>
#include <iomanip>
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
double bench(kernel_t variant, char uplo, char trans, char diag, int n,
    const Prec *__restrict__ A, int ldA, Prec *__restrict__ x, int incx)
{
    int (*trsv_var[])(char, char, char, int, const Prec *__restrict__ , int,
                      Prec *__restrict__, int) = {
        trsv_selector<Prec, USE_BLAS_CALL>,
        trsv_selector<Prec, UNROLL_1>,
        trsv_selector<Prec, UNROLL_2>
    };

    // Repeat runs until at least 5s have passed.
    double walltime = 0.0;
    long num_iters = 0;
    steady_clock::time_point now;
    steady_clock::time_point start = steady_clock::now();
    do {
        auto time_start = high_resolution_clock::now();
        trsv_var[variant](uplo, trans, diag, n, A, ldA, x, incx);
        auto time_end = high_resolution_clock::now();
        microseconds duration = duration_cast<microseconds>(time_end - time_start);
        walltime += duration.count();
        num_iters++;
        now = steady_clock::now();
    } while(now < start + seconds{5});

    double avg_time = walltime / num_iters;
    #ifndef NDEBUG
    std::cout << "  Average time elapsed = " << avg_time << " µs" << std::endl;
    #endif
    return avg_time;
}

template<typename Prec>
void bench(const std::string &output_filename)
{
    std::ofstream out;
    out.open(output_filename);

    // Write header. (As of C++20, std::format can be used)
    out << std::setw(13) << std::left << "Kernel"
        << std::setw( 6) << std::left << "uplo"
        << std::setw( 6) << std::left << "diag"
        << std::setw( 6) << std::left << "trans"
        << std::setw( 8) << std::left << "n"
        << std::setw(14) << std::left << "duration [µs]"
        << std::endl;

    using Real = remove_complex_t<Prec>;
    constexpr int maxn = 4000;
    int ldA = maxn;

    std::vector<Prec> x = std::vector<Prec>(maxn);
    std::vector<Prec> A = std::vector<Prec>(maxn * ldA);
    int incx = 1;

    generator<Prec> rg;

    for (char uplo : {'U', 'L'}) {
        for (char diag : {'N', 'U'}) {
            for (char trans : {'C', 'T', 'N'}) {
                for (int n : {100, 500, 1000, maxn}) {
                    for (kernel_t kernel_type : {USE_BLAS_CALL, UNROLL_1, UNROLL_2}) {
                        #ifndef NDEBUG
                        std::cout << "Kernel variant = " << kernel_type_to_str(kernel_type) << ", "
                                  << "uplo = " << uplo << " , diag = " << diag << " , trans = " << trans 
                                  << " , n = " << n << std::endl << std::endl;
                        #endif
                        // Generate random right-hand side vector, and a triangular matrix.
                        rg.generate_general_matrix(n, 1, x.data(), n);
                        rg.generate_triangular_matrix(uplo, diag, n, A.data(), ldA);

                        auto duration = bench<Prec>(kernel_type, uplo, trans, diag, n, A.data(), ldA, x.data(), incx);

                        out << std::setw(13) << std::left << kernel_type_to_str(kernel_type)
                            << std::setw( 6) << std::left << uplo
                            << std::setw( 6) << std::left << diag
                            << std::setw( 6) << std::left << trans
                            << std::setw( 8) << std::right << n
                            << std::setw(12) << std::right << duration << std::endl;
                    }
                }
            }
        }
    }

    out.close();
}

} // namespace


int main(int argc, char **argv)
{
    bench<float>("analysis/bench_float.csv");
    bench<double>("analysis/bench_double.csv");
    bench<std::complex<double>>("analysis/bench_double_complex.csv");
    bench<std::complex<float>>("analysis/bench_float_complex.csv");

    return EXIT_SUCCESS;
}