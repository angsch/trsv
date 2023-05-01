#pragma once

template<typename Prec>
void trsv(char uplo, char trans, char diag, int n,
    const Prec *__restrict__ A, int ldA, Prec *__restrict__ x, int incx);