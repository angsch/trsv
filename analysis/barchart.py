#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import os


def read_file(filename):
    data = []
    with open(filename) as infile:
        # Skip header line/legend.
        infile.readline()

        for row in infile:
            row = row.strip()
            kernel, uplo, diag, trans, n, duration = row.split()
            record = {'n'      : int(n),
                      'config' : (uplo, diag, trans),
                      'perf'   : (kernel, float(duration))}
            data.append(record)
    return data

def generate_bar_chart(data, plotname):
    n_list = sorted(list(set(r['n'] for r in data)))
    configs = list(set(r['config'] for r in data))

    uplo_set = set(map(lambda config: config[0], configs))
    diag_set =  set(map(lambda config: config[1], configs))
    trans_set = sorted(set(map(lambda config: config[2], configs)))

    measurements = list(set(r['perf'] for r in data))
    kernel_list = list(set(map(lambda m: m[0], measurements)))

    coloring = {
        'BLAS'         : 'r',
        'unroll_once'  : 'y',
        'unroll_twice' : 'b'
    }

    fig, axs = plt.subplots(1, len(n_list), figsize=(100, 10))
    for col, n in enumerate(n_list):
        y = []
        x_labels = []
        barcolors = []
        for trans in trans_set:
            for diag in diag_set:
                for uplo in uplo_set:
                    measurements = [r['perf'] for r in data if r['n'] == n and r['config'] == (uplo, diag, trans)]
                    for (kernel, time) in measurements:
                        y.append(time)
                        x_labels.append(f'({trans}, {diag}, {uplo}, {kernel})')
                        barcolors.append(coloring[kernel])

        axs[col].set_ylabel('Time')
        axs[col].set_title(f'n = {n}')
        x = np.arange(len(barcolors))
        axs[col].set_xticks(x, x_labels, rotation=90)
        axs[col].bar(x, y, color = barcolors)
    
    plt.savefig(plotname, format='pdf', dpi=1000, bbox_inches='tight')


for filename in os.listdir(os.getcwd()):
    if filename.endswith('.csv'):
        name, ext = filename.split('.')
        data = read_file(filename)
        plotname = name + '.pdf'
        generate_bar_chart(data, plotname)