# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 14:31:17 2015

@author: Kevin M.

Description:
            This script does CPU and GPU matrix element time complexity
             profiling. It has a function which applies the matrix element
             analysis for a given set of parameters, profiles the code and
             plots the time complexity results (with fit) and plots the matrix
             elements from each case.
"""

import numpy as np
import scipy as sp
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from my_timer import timer
from math import log
from scipy.optimize import curve_fit

def f_MEplaceholder(neval, mode):
    # Placeholder integration instead of ME calc
    result, error = (sp.integrate.quad(lambda x:
                     sp.special.jv(2.5, x), 0, neval) if mode == 'gpu'
                     else   sp.integrate.quadrature(lambda x:
                     sp.special.jv(2.5, x), 0, neval))
    return result, error

def flinear(N, mode):
    """
    O(n) function
    """
    y = np.asarray([i for i in range(N)])
    np.asarray([i for i in range(N)])
    np.asarray([i for i in range(N)])
    return y ,1

def fsquare(N, mode):
    """
    O(n^2) function
    """
    for i in range(N):
        for j in range(N):
            y = i*j
    return y,1

def algoAnalysis(fn, nMin, nMax, mode):
    """
    Run timer and plot time complexity
    """
    n = []
    time_result = []
    y_result = []
    y_err = []

    for i in [j*32 for j in range(nMin,nMax+1)]:
        with timer() as t:
            temp_result, temp_err = fn(i, mode)
        time_result.append(t.msecs)
        y_result.append(temp_result)
        y_err.append(temp_err)
        n.append(i)
    return n, time_result, y_result, y_err


def plotAll(n, time_data, y_data, err_data):
    n = np.asarray(n)
    time_data = np.asarray(time_data)
    y_data = np.asarray(y_data)
    err_data = np.asarray(err_data)
    err_data[0] = err_data[1]*0.5

    # plotting helpers
    nTime = n[2]
    n = map(lambda x: log(x,2), n[0])
    colors = ['lightblue', 'lightgreen']
    edgeColors = ['#1B2ACC','#3F7F4C']
    faceColors = ['#089FFF', '#7EFF99']
    label_entries_for_results = ['GPU Matrix Elements', 'CPU Matrix Elements']
    label_entries_for_time = ['GPU Runtime', 'CPU Runtime']

    plt.figure(figsize=(15,6))
    ###########################################################################

    # The following plots the runtime information for GPU and CPU runs.
    def sqFunc(x, a, b, c):
        return a*x**2 + b*x +c

    def linFunc(x, a, b):
        return a*x + b

    funcList = [linFunc, sqFunc]
    ax = plt.subplot(1,2,1)
    # draw plots for timing data
    for dat_mode in xrange(0,2):
        params = curve_fit(funcList[dat_mode], nTime, time_data[dat_mode])
        x = np.linspace(nTime[0], nTime[-1], 1000)

        if dat_mode == 0:
            [a,b] = params[0]
            y = funcList[dat_mode](x, a, b)
            s = "Fit for GPU: $%.5fx$ + $%.5f$"%(a,b)

        if dat_mode == 1:
            [a,b,c] = params[0]
            y = funcList[dat_mode](x, a, b, c)
            s = "Fit for CPU: $%.5fx^2$ + $%.5fx$ + $%.2f$"%(a,b,c)

        ax.text(0.035, 0.75-dat_mode*0.1, s,
             transform = ax.transAxes,
             fontsize = 16)

        ax.plot(x,y, color='k', linestyle="--", linewidth = 4)
        ax.plot(nTime, time_data[dat_mode], color=colors[dat_mode],
                marker = 'o', label=label_entries_for_time[dat_mode],
                linestyle = 'None')


    # setting axis limits
    plt.xlim([min(nTime)-50, max(nTime)+50])

    plt.ylim([min(min(time_data[0]), min(time_data[1]))*1.3,
              max(max(time_data[0]), max(time_data[1]))*1.3])

   # hiding axis ticks
    plt.tick_params(axis="both", which="both", bottom="off", top="off",
            labelbottom="on", left="off", right="off", labelleft="on")

    # adding horizontal grid lines
    ax.yaxis.grid(True)

    # remove axis spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # labels
    plt.xlabel('Maximum number of phase space points')
    plt.ylabel('Runtime (msec)')
    leg = plt.legend(loc='upper left', fancybox=True, numpoints=1)
    leg.get_frame().set_alpha(0.5)

    ###########################################################################

    # The following plots the Matrix Elements for the GPU and CPU respectively
    # on a subplot, on top of each other with their corresponding errors.

    ax = plt.subplot(1,2,2)
    # draw plots for results
    for dat_mode in xrange(0,2):
        ax.errorbar(x=n, y=y_data[dat_mode], yerr=err_data[dat_mode],
                    fmt='o', color=colors[dat_mode], ecolor='black',
                    alpha = 0.3)
        ax.plot(n, y_data[dat_mode,:], marker='o',
                linestyle = 'None', color=colors[dat_mode],
                label=label_entries_for_results[dat_mode])

        ax.fill_between(n, y_data[dat_mode]-err_data[dat_mode],
                        y_data[dat_mode]+err_data[dat_mode],
                        alpha=0.2, edgecolor=edgeColors[dat_mode],
                        facecolor=faceColors[dat_mode],
                        linewidth=4, linestyle='-.', antialiased=True)

    # setting axis limits
    plt.xlim([min(n)-1*0.2, max(n)+1*0.2])

    plt.ylim([min(min(y_data[0]), min(y_data[1]))*1.3,
              max(max(y_data[0]), max(y_data[1]))*1.3])

   # hiding axis ticks
    plt.tick_params(axis="both", which="both", bottom="off", top="off",
            labelbottom="on", left="off", right="off", labelleft="on")

    # adding horizontal grid lines
    ax.yaxis.grid(True)

    # remove axis spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # labels
    plt.xlabel('$\log_2$(Maximum number of phase space points)')
    plt.ylabel('Matrix Element')
    leg = plt.legend(loc='upper left', fancybox=True, numpoints=1)
    leg.get_frame().set_alpha(0.5)

    plt.tight_layout()

    plt.savefig('plots.pdf')
    plt.show()



# main() function
def main():
    print('\nAnalyzing Algorithms...')
    n_GPU, timeGPU, yResult_GPU, yErr_GPU = algoAnalysis(f_MEplaceholder, 8, 20, 'gpu')
    n_CPU, time_CPU, yResult_CPU, yErr_CPU = algoAnalysis(f_MEplaceholder, 8, 20, 'cpu')
    nLin, timeLin, y1, y2 = algoAnalysis(flinear, 10, 50, 'cpu')
    nSq, timeSq, y1, y2 = algoAnalysis(fsquare, 10, 50, 'cpu')
    nList = [n_GPU, n_CPU, nLin, nSq] ### DELETE NLIN NSQ AFTER

    timeList = [timeLin, timeSq]
    yResultList = [yResult_GPU, yResult_CPU]
    yErrList = [yErr_GPU, yErr_CPU]
    plotAll(nList, timeList, yResultList, yErrList)

# call main
if __name__ == '__main__':
#    matplotlib.rcParams.update({'font.family': 'Zapf Chancery'})
    main()

















