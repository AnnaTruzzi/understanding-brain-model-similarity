import os
import scipy.io
import pickle
import numpy as np
from itertools import compress
from itertools import combinations
from scipy.spatial import distance
from sklearn.manifold import MDS
from statsmodels.stats.anova import AnovaRM
import matplotlib
from matplotlib import pyplot as plt
import re
from scipy.cluster.hierarchy import dendrogram, linkage
from itertools import combinations
import scipy.io
import h5py
import hdf5storage
from scipy import stats
from scipy.spatial.distance import squareform
import collections
import re
from skbio.stats.distance import mantel
import seaborn as sns
import pingouin as pg
import pandas as pd
import glob
from sklearn.linear_model import LinearRegression
import scipy
from scipy.stats import pearsonr, spearmanr, kendalltau
import seaborn as sns
from config import FMRI_PATH, RESULTS_ROOT, load_rdm

def loadmat(matfile):
    try:
        f = h5py.File(matfile)
    except (IOError, OSError):
        return io.loadmat(matfile)
    else:
        return {name: np.transpose(f.get(name)) for name in f.keys()}


def corr_mri_variability(layer, mri_data):
    x = np.mean(np.asarray(layer), axis=0)
    y = mri_data
    corr_list = []
    p_list = []
    for subj in range(0,y.shape[0]):
        y_subj = squareform(squareform(y[subj],checks = False))
        corr_value, p_value, n_value = mantel(x,y_subj, method = 'kendalltau', permutations = 10000)
        corr_list.append(corr_value)
        p_list.append(p_value)
    return corr_list,p_list


def corr_layer_variability(layer, mri_data):
    x = np.array(layer)
    y = squareform(squareform(np.mean(np.asarray(mri_data),axis=0),checks = False))
    corr_list = []
    p_list = []
    for layer in range(0,x.shape[0]):
        x_layer = squareform(squareform(x[layer],checks = False))
        corr_value, p_value, n_value = mantel(x_layer,y, method = 'kendalltau', permutations = 10000)
        corr_list.append(corr_value)
        p_list.append(p_value)
    return corr_list,p_list


def main(list_file):
    for method in corr_methods:
        print(f'Calculating correlations for {net} - {method}')
        all_corr=[]
        all_pvalue=[]
        out_layers_list=[]
        ROI_list=[]
        state_list=[]
        with open(RESULTS_ROOT / f'corr_{net}_{method}.txt', 'w') as f:
            for layer in layers:
                for state in states:
                    all_layer_rdms = []
                    curr_list = [i for i in list_file if state in i and layer in i]
                    for name in curr_list:
                        rdm = load_rdm(name)
                        all_layer_rdms.append(rdm)
                    
                    if 'mri_variability' in method:
                        EVC_corr_list,EVC_pvalue_list = corr_mri_variability(all_layer_rdms, EVC)
                        IT_corr_list,IT_pvalue_list = corr_mri_variability(all_layer_rdms, IT)
                    else:
                        EVC_corr_list,EVC_pvalue_list = corr_layer_variability(all_layer_rdms, EVC)
                        IT_corr_list,IT_pvalue_list = corr_layer_variability(all_layer_rdms, IT)

                    all_corr.extend(EVC_corr_list+IT_corr_list)
                    all_pvalue.extend(EVC_pvalue_list+IT_pvalue_list)
                    out_layers_list.extend(np.repeat(layer,len(EVC_corr_list+IT_corr_list)))
                    ROI_list.extend(np.repeat('EVC',len(EVC_corr_list)))
                    ROI_list.extend(np.repeat('IT',len(IT_corr_list)))
                    state_list.extend(np.repeat(state,len(EVC_corr_list+IT_corr_list)))

                    f.write('******************************** \n')
                    f.write('******************************** \n')
                    f.write(f'Average corr with brain - {method} {layer} - {state} \n')
                    f.write(f'EVC: {np.mean(np.array(EVC_corr_list))} \n')
                    f.write(f'IT: {np.mean(np.array(IT_corr_list))} \n')

        print('Working on the dataset...')
        corr_dict = {'corr': all_corr,
                    'p': all_pvalue,
                    'layer': out_layers_list,
                    'ROI': ROI_list,
                    'state': state_list,
                    'net': np.repeat(net,len(all_corr))}
        corr_df = pd.DataFrame(corr_dict)
        corr_df.to_csv(RESULTS_ROOT / f'corr_{net}_{method}.csv', index=False)



if __name__ == '__main__':
    layers = ['ReLu1', 'ReLu2', 'ReLu3', 'ReLu4', 'ReLu5','ReLu6','ReLu7']
    nets = ['alexnet','dc']
    fmri_mat = loadmat(FMRI_PATH)
    EVC = fmri_mat['EVC_RDMs']
    IT = fmri_mat['IT_RDMs']

    for net in nets:
        list_file = []

        if net == 'dc':
            states = ['randomstate','100epochs']
            instances = 11
            corr_methods = ['layer_variability','mri_variability']
        else:
            states = ['randomstate','pretrained']
            instances = 2
            corr_methods = ['mri_variability']

        for instance in range(1,instances):
            for layer in layers:
                for state in states:
                    list_file.append(f'{net}{instance}_{state}_{layer}')
        
        main(list_file)

