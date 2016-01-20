# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 11:44:38 2015

@author: Kevin
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import scipy

data_path = 'C:\Users\Kevin\Desktop\workStuff\LDA\data' + "\\"

def createDataFrame(vbfMEList, ttbarMEList, path = data_path):
    data_vbfME = np.asarray([0])
    data_ttbarME =  np.asarray([0])
    label_vector = np.asarray([10]) # label is either 0 or 1, 1 = signal, 0 = background
    Qvec_vbfME = np.asarray([10])
    Qvec_ttbarME = np.asanyarray([10])
    evtWeightVector = np.asarray([0])

    for process in vbfMEList:
            tempData = np.genfromtxt(path+process)
            data_vbfME = np.append(data_vbfME, tempData[:,0])
            label_vector = np.append(label_vector, np.ones_like(tempData[:,0])) if 'vbf125data' in process else np.append(label_vector,np.zeros_like(tempData[:,0]))
            Qvec_vbfME = np.append(Qvec_vbfME, tempData[:,4])
            evtWeightVector = np.append(evtWeightVector, tempData[:,5])
            
    for process in ttbarMEList:
            tempData = np.genfromtxt(path+process)
            data_ttbarME = np.append(data_ttbarME, tempData[:,0])
            Qvec_ttbarME = np.append(Qvec_ttbarME, tempData[:,4])

    label_vector = np.delete(label_vector, 0)
    data_vbfME = np.delete(data_vbfME, 0)
    data_ttbarME = np.delete(data_ttbarME, 0)
    Qvec_vbfME = np.delete(Qvec_vbfME, 0)
    Qvec_ttbarME = np.delete(Qvec_ttbarME, 0)
    evtWeightVector = np.delete(evtWeightVector, 0)
    
    dataFrame = np.vstack(([data_vbfME], [data_ttbarME], [label_vector], [Qvec_vbfME], [Qvec_ttbarME], [evtWeightVector]))
    return dataFrame.T


def getFeatureNames(fileNameList):
    vbfList = [process for process in fileNameList if process.split('data_')[1] == 'vbf125mem.txt']
    ttbarList = [process for process in fileNameList if process.split('data_')[1] == 'ttbarmem.txt']
    return sorted(vbfList), sorted(ttbarList)


def getFileNames(path=data_path):
    return os.listdir(path)


def writeDataFrame(dataFrame):
    filename = 'dataFrame.txt'
    np.savetxt(filename, dataFrame, fmt='%.10e')


def getDataFrame():
    currentDirectory = os.listdir('.')
    if 'dataFrame.txt' in currentDirectory:
        dataFrame = np.genfromtxt('dataFrame.txt')
    else:
        fileNameList = getFileNames()
        vbfList, ttbarList = getFeatureNames(fileNameList)
        dataFrame = createDataFrame(vbfList, ttbarList)
        writeDataFrame(dataFrame)
    return dataFrame


def plotHisto(discmin, discmax, signalEvents, backgroundEvents, separation, sigWeights, bgWeights):
    histbins = np.linspace(discmin, discmax, 25)
    plt.figure(figsize=(8,6))
    ax = plt.subplot(111)
    print np.shape(signalEvents)
    print np.shape(sigWeights)
    plt.hist(signalEvents, bins = histbins, label = 'Signal',
             histtype='stepfilled', fc = '#2482FF', linewidth = 2,
             alpha = 0.5, normed = True, weights=sigWeights)
    plt.hist(backgroundEvents, bins = histbins, label = 'Background',
             histtype='stepfilled', fc = 'lightgreen', linewidth = 2, alpha = 0.5,
             hatch = '//', normed = True, weights=bgWeights)
    plt.xlim([discmin, discmax])

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
    plt.xlabel('Discriminant')
    plt.ylabel('Distribution Value')
    plt.legend(loc='upper left', fancybox=True)
    plt.savefig('converged'+"_hist_"+"S"+str(separation)+".pdf")
    plt.show()


def plotROC(signalEffArray, backgroundRejArray, separation):
    plt.figure(figsize=(8,6))
    ax = plt.subplot(111)

    plt.plot([0,1],[1,0], 'k--', alpha=0.5)

    plt.plot(signalEffArray, backgroundRejArray, '-',
             color='#FF8000', alpha = 0.6,
             linewidth = 2, label = 'Separation: {0}'.format(separation) )


    # labels
    plt.xlabel('Signal Efficiency')
    plt.ylabel('Background Rejection')
    plt.legend(loc='best', fancybox=True, fontsize=10)
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
    plt.savefig('converged'+"_roc_"+"S"+str(separation)+".pdf")
    plt.show()


if __name__ == '__main__':

    dataFrame = getDataFrame()
    Q_cut = np.logical_and(dataFrame[:,3]>0.2, dataFrame[:,4]>0.2)
    length_before = len(dataFrame)
    dataFrame = dataFrame[Q_cut]
    length_after = float(len(dataFrame))
    print "\nFraction of events that failed to meet Q requirement: {}".format(1-length_after/length_before)
    print "Out of a total of {} events".format(length_before)

    label = dataFrame[:,2]

    sigdata_sigmem = dataFrame[:,0][label==1]
    bgdata_sigmem = dataFrame[:,0][label==0]
    sigdata_bgmem = dataFrame[:,1][label==1]
    bgdata_bgmem = dataFrame[:,1][label==0]
    
    sig_evtW = dataFrame[:,5][label==1]
    bg_evtW = dataFrame[:,5][label==0]

    xmax = max(np.amax(sigdata_sigmem),np.amax(bgdata_sigmem))
    ymax = max(np.amax(sigdata_bgmem),np.amax(bgdata_bgmem))
    xymax = max(xmax,ymax)

    signalEvents = np.log10(sigdata_sigmem) - np.log10(sigdata_bgmem)
    backgroundEvents = np.log10(bgdata_sigmem) - np.log10(bgdata_bgmem)

    discmin = min(np.amin(signalEvents), np.amin(backgroundEvents))
    discmax = max(np.amax(signalEvents), np.amax(backgroundEvents))

    signalEffArray = []
    backgroundRejArray = []
    discVal = np.linspace(discmin, discmax, 10000)

    sigNorm = sum(sig_evtW)
    bgNorm = sum(bg_evtW)

    for thisVal in discVal:
        signalEff = sum(sig_evtW[np.where(signalEvents >= thisVal)])
        backgroundRej = sum(bg_evtW[np.where(backgroundEvents < thisVal)])
        signalEffArray.append(signalEff/sigNorm)
        backgroundRejArray.append(backgroundRej/bgNorm)

    def roc(x):
        # x: the desired signal efficiency
        opt = (np.abs(np.asarray(signalEffArray)-x)).argmin()
        return 1-sum(bg_evtW[np.where(backgroundEvents > discVal[opt])])/bgNorm


    def getSep(fn = roc):
        separation = 0
#        for i in range(100): separation += scipy.integrate.quad(fn, i*0.01, (i+1)*0.01)[0]
        separation = scipy.integrate.quad(fn, 0, 1)[0]
        return 2*separation-1

    separation = getSep()

    plotHisto(discmin, discmax, signalEvents, backgroundEvents, separation, sig_evtW, bg_evtW)
    plotROC(signalEffArray, backgroundRejArray, separation)





