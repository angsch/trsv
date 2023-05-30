#!/bin/bash

set -e

declare -a targets=("test" "bench")

for target in "${targets[@]}"
do
    reportdir=coverage-report-${target}
    rm -rf $reportdir
    mkdir -p $reportdir
    cd ..
    coverage=1 make $target -j4
    ./$target
    gcovr --root . --html --html-details --output analysis/${reportdir}/coverage.html
    make clean
    cd ~-
done
