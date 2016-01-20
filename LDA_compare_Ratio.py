import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
from sklearn import preprocessing
from sklearn.lda import LDA

import os
##############################################

data_path = 'C:\Users\Kevin\Desktop\workStuff\LDA\data' + "\\"

def createDataFrame(vbfMEList, ttbarMEList, path = data_path):
    data_vbfME = np.asarray([0])
    data_ttbarME =  np.asarray([0])
    label_vector = np.asarray([10]) # label is either 0 or 1, 1 = signal, 0 = background

    for process in vbfMEList:
            tempData = np.genfromtxt(path+process)#[:,0]
            data_vbfME = np.append(data_vbfME, tempData)
            label_vector = np.append(label_vector, np.ones_like(tempData)) if 'vbf125data' in process else np.append(label_vector,np.zeros_like(tempData))

    for process in ttbarMEList:
            tempData = np.genfromtxt(path+process)#[:,0]
            data_ttbarME = np.append(data_ttbarME, tempData)

    label_vector = np.delete(label_vector, 0)
    data_vbfME = np.delete(data_vbfME, 0)
    data_ttbarME = np.delete(data_ttbarME, 0)

    dataFrame = np.vstack(([data_vbfME], [data_ttbarME], [label_vector]))
    return dataFrame.T


def getFeatureNames(fileNameList):
    vbfList = [process for process in fileNameList if process.split('data_')[1] == 'vbf125mem.txt']
    ttbarList = [process for process in fileNameList if process.split('data_')[1] == 'ttbarmem.txt']
    return sorted(vbfList), sorted(ttbarList)


def getFileNames(path=data_path):
    return os.listdir(path)
    
    
def makeFeatureHistPlot(X, y, label_dict, feature_dict):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12,6))

    for ax,cnt in zip(axes.ravel(), range(2)):
        # set bin sizes
        min_b = np.min(X[:,cnt])
        max_b = np.max(X[:,cnt])
        bins = np.linspace(min_b, max_b, 50)
        print bins
        # plotting the histograms
        for lab,col in zip(range(0,2), ('blue', 'red')):
            ax.hist(X[y==lab, cnt],
                       color=col,
                       label='class %s' %label_dict[lab],
                       bins=bins,
                       alpha=0.5,
                       log = True,)
        ylims = ax.get_ylim()
        
        # plot annotation
        leg = ax.legend(loc='upper right', fancybox=True, fontsize=8)
        leg.get_frame().set_alpha(0.5)
        ax.set_ylim([0, max(ylims)+2])
        ax.set_xlabel(feature_dict[cnt])
        ax.set_title('Matrix Element Histogram: Feature #%s' %str(cnt+1))
    
        # hide axis ticks
        ax.tick_params(axis="both", which="both", bottom="off", top="off",
                labelbottom="on", left="off", right="off", labelleft="on")
    
        # remove axis spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.yaxis.grid(True)
    axes[0].set_ylabel('count')
    
    fig.tight_layout()
    plt.show()


def plot_lda(X, y, label_dict, title, mirror=1):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize =(12,6))
    # set bin sizes
    min_b = np.min(X[:])
    max_b = np.max(X[:])
    bins = np.linspace(min_b, max_b, 50)

    # plotting the histograms
    for lab,col in zip(range(0,2), ('blue', 'red')):
        ax.hist(X[y==lab],
                   color=col,
                   label='class %s' %label_dict[lab],
                   bins=bins,
                   alpha=0.5,
                   log = True,)
    ylims = ax.get_ylim()
    
    # plot annotation
    leg = ax.legend(loc='upper right', fancybox=True, fontsize=8)
    leg.get_frame().set_alpha(0.5)
    ax.set_ylim([0, max(ylims)+2])
    ax.set_xlabel('LDA1')
    ax.set_title(title)

    # hide axis ticks
    ax.tick_params(axis="both", which="both", bottom="off", top="off",
            labelbottom="on", left="off", right="off", labelleft="on")

    # remove axis spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.yaxis.grid(True)

    plt.grid()
    plt.tight_layout
    plt.show()

def plot_scatter_and_eigendirection(X, y, eigen_dir):
    X = np.log10(X)
    sigdata_sigmem = X[:,0][y==1]
    bgdata_sigmem = X[:,0][y==0]
    sigdata_bgmem = X[:,1][y==1]
    bgdata_bgmem = X[:,1][y==0]
   
    xmax = max(np.amax(sigdata_sigmem), np.amax(bgdata_sigmem))
    xmin = min(np.amin(sigdata_sigmem), np.amin(bgdata_sigmem))

    ymax = max(np.amax(sigdata_bgmem), np.amax(bgdata_bgmem))
    ymin = min(np.amin(sigdata_bgmem), np.amin(bgdata_bgmem))
    
    mu = X.mean(axis=0)
    X = X - mu
#   X = (X - mu)/X.std(axis=0)  # Uncomment this reproduces mlab.PCA results
    
    projected_data = np.dot(X, eigen_dir)
    sigma = projected_data.std(axis=0).mean()

    print(eigen_dir)
    
    def annotate(ax, name, start, end):
        arrow = ax.annotate(name,
                            xy=end, xycoords='data',
                            xytext=start, textcoords='data',
                            arrowprops=dict(facecolor='red', width=2.0))
        return arrow

    fig, ax = plt.subplots(nrows=1, ncols= 1, figsize=(15,15))
    fig.suptitle("VBF MEM Separation")
    ax.set_aspect('equal')
    
    ax.plot(sigdata_sigmem,sigdata_bgmem, linestyle = 'None', marker = 'o',
            c='lightblue', label='Signal Data', alpha = 0.5, 
            markeredgecolor= 'black', markeredgewidth = 1)
    ax.plot(bgdata_sigmem,bgdata_bgmem, linestyle = 'None', marker = 'o',
            c='lightgreen', label='Background Data', alpha = 0.5,
            markeredgecolor = 'black', markeredgewidth = 1.1)
    
    for axis in eigen_dir.T:
        annotate(ax, '', mu, mu + sigma * axis)
    
    ax.plot([xmin,xmax],[ymin,ymax], c='black')


    plt.xlim([xmin,xmax])
    plt.ylim([ymin,ymax])

    plt.xlabel('Signal Matrix Element')
    plt.ylabel('Background Matrix Element')

    ax.legend(loc = 'best', numpoints=1, fancybox=True)
    plt.tight_layout
    ax.yaxis.grid(True)
    plt.show()    
    
if __name__ == '__main__':

    label_dict = {0: 'Background', 1: 'Signal'}
    feature_dict = {0: "Signal Matrix Element", 1: "Background Matrix Element"}

    fileNameList = getFileNames()
    vbfList, ttbarList = getFeatureNames(fileNameList)
    dataFrame = createDataFrame(vbfList, ttbarList)

    X = dataFrame[:, 0:2]
    y = dataFrame[:, 2]
    
    X_copy = createDataFrame(vbfList, ttbarList)[:,0:2]
    preprocessing.scale(X, axis=0, with_mean=True, with_std=True, copy=False)
    makeFeatureHistPlot(X,y,label_dict, feature_dict)
#    sklearn_lda = LDA(n_components=1)
#    X_lda_sklearn = sklearn_lda.fit_transform(X, y)
    
    ## Step 1 LDA: Computing the d-dimensional mean vectors    
    np.set_printoptions(precision = 10)
    
    mean_vectors = []
    for cl in range(0,2):
        mean_vectors.append(np.mean(X[y==cl,:], axis=0))
        print 'Mean Vector class %s: %s\n' %(cl, mean_vectors[cl])
        
    ## Step 2 LDA: Computing the Scatter Matricies

    # 2.1 The Within-class Scatter Matrix, S_w
    S_W = np.zeros((2,2))
    for cl, mv in zip(range(0,2), mean_vectors):
        class_sc_mat = np.zeros((2,2))                  # scatter matrix for every class
        for row in X[y == cl]:
            row, mv = row.reshape(2,1), mv.reshape(2,1) # make column vectors
            class_sc_mat += (row-mv).dot((row-mv).T)
        S_W += class_sc_mat                             # sum class scatter matrices
    print 'within-class Scatter Matrix:\n', S_W       
    
    # 2.2 The Between-class Scatter Matrix, S_b
    overall_mean = np.mean(X, axis=0)
    
    S_B = np.zeros((2,2))
    for i,mean_vec in enumerate(mean_vectors):
        n = X[y==i+1,:].shape[0]
        mean_vec = mean_vec.reshape(2,1) # make column vector
        overall_mean = overall_mean.reshape(2,1) # make column vector
        S_B += n * (mean_vec - overall_mean).dot((mean_vec - overall_mean).T)
    
    print 'between-class Scatter Matrix:\n', S_B
    
    ## Step 3 LDA: Solving the generalized eigenvalue problem
    ##             for the matrix S_w^(-1)S_b
 
    eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
    
    for i in range(len(eig_vals)):
        eigvec_sc = eig_vecs[:,i].reshape(2,1)
        print '\nEigenvector {}: \n{}'.format(i+1, eigvec_sc.real)
        print'Eigenvalue {:}: {:.2e}'.format(i+1, eig_vals[i].real)   
    
    # Check A*eig_v = lambda*eig_v
    for i in range(len(eig_vals)):
        eigv = eig_vecs[:,i].reshape(2,1)
        np.testing.assert_array_almost_equal(np.linalg.inv(S_W).dot(S_B).dot(eigv),
                                         eig_vals[i] * eigv,
                                         decimal=6, err_msg='', verbose=True)
   
    ## Step 4 LDA: Selecing linear discriminants for the new feature (sub)space   
    
    # 4.1 Sorting the eigenvectors and eigenvalues by decreasing eigenvalues
    # Make a list of (eigenvalue, eigenvector) tuples
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
    
    # Sort the (eigenvalue, eigenvector) tuples from high to low
    eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)
    
    # Visually confirm that the list is correctly sorted by decreasing eigenvalues
    
    print'Eigenvalues in decreasing order:\n'
    for i in eig_pairs:
        print(i[0])     
        
    print('\nVariance explained:\n')
    eigv_sum = sum(eig_vals)
    for i,j in enumerate(eig_pairs):
        print('eigenvalue {0:}: {1:.2%}'.format(i+1, (j[0]/eigv_sum).real))
        
    # 4.2 Choosing k-eigenvectors with largest eigenvalues
    W = eig_pairs[0][1].reshape(2,1)
    print '\nMatrix W:\n', W.real
    
    ## Step 5 LDA: Transforming the samples onto the new subspace
    X_lda = X.dot(W)
    
    plot_lda(X_lda, y, label_dict, 'my lda')
#    plot_lda(X_lda_sklearn,y, label_dict, 'LDA - scikit-learn 15.2')
    plot_scatter_and_eigendirection(X_copy, y, eig_vecs)

    
    
    
    
    
    
    
    
    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
