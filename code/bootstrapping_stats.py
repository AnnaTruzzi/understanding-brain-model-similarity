import os
import scipy.io
import pickle
import numpy as np
from itertools import compress
from scipy.spatial import distance
import sklearn
from sklearn.manifold import MDS
from sklearn.metrics import mean_poisson_deviance
from statsmodels.stats.anova import AnovaRM
import matplotlib
from matplotlib import pyplot as plt
import re
from scipy.cluster.hierarchy import dendrogram, linkage
from itertools import combinations
import scipy.io as io
from scipy import stats
from scipy.spatial.distance import squareform
import collections
import re
import skbio
import seaborn as sns
import pandas as pd
import glob
from sklearn.linear_model import LinearRegression
import scipy
from scipy.stats import pearsonr, spearmanr, kendalltau
import statsmodels.formula.api as smf
#import pkg_resources
import pingouin as pg
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneOut
import random
from scipy.stats import ttest_rel
from scipy.stats import ttest_ind
from config import RESULTS_ROOT, load_bootstrap_values


def cohens_d(group1, group2):
    """
    Calculate Cohen's d for two groups.
    
    Parameters:
    group1, group2: array-like
        The two groups to compare
        
    Returns:
    float: Cohen's d effect size
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    # Cohen's d
    d = (np.mean(group1) - np.mean(group2)) / pooled_std
    return d

## load activations dictionary 
def load_dict(path):
    with open(path,'rb') as file:
        dict=pickle.load(file, encoding="latin1")
    return dict


def bootstrapping_confidenceinterval():
    ### model vs baseline
    with open(RESULTS_ROOT / 'CombinedModels_BootstrappingStats.txt', 'w') as f:
        f.write('##### \n')
        f.write('MODELS vs BASELINE\n')
        f.write('##### \n')
        for key in list(corr_values.keys()):
            #b=corr_values[key].mean(axis=1)
            b = corr_values_mean_corrected[key]
            b_sort=np.sort(b)
            b_len=len(b_sort)
            b_lower=b_sort[int(np.round(b_len * alpha/2))]
            b_upper=b_sort[int(np.round(b_len * (1-alpha/2)))]
            #print('** ' + key)
            #print('lower and upper')
            #print(b_lower)
            #print(b_upper)
            #print('mean, sem, and sd')
            mean = np.nanmean(b)
            sem = scipy.stats.sem(b,nan_policy='omit')
            sd = np.nanstd(b)
            f.write('#')
            f.write(f'{key} \n')
            f.write(f'Lower bound={b_lower}, Upper bound={b_upper} \n')
            f.write(f'Mean={mean}, SEM={sem}, SD={sd} \n')
            f.write('\n')
    
    ## model vs model
        f.write('##### \n')
        f.write('MODELS COMPARISON\n')
        f.write('##### \n')
        for comparison in comparisons:
            #d = corr_values[comparison[0]].mean(axis=1) - corr_values[comparison[1]].mean(axis=1)
            d = corr_values_mean_corrected[comparison[0]] - corr_values_mean_corrected[comparison[1]]
            d_sort=np.sort(d)
            d_len=len(d_sort)
            d_lower=d_sort[int(np.round(d_len * alpha/2))]
            d_upper=d_sort[int(np.round(d_len * (1-alpha/2)))]
            #print(f'{comparison[0]} vs {comparison[1]}' )
            #print(d_lower)
            #print(d_upper)
            f.write('#')
            f.write(f'{comparison[0]} VS {comparison[1]} \n')
            f.write(f'Lower bound={d_lower}, Upper bound={d_upper} \n')
            f.write('\n')

def posthoc_ttest():
    with open(RESULTS_ROOT / 'CombinedModels_BootstrappingStats_ttests.txt', 'w') as f:
        f.write('##### \n')
        f.write('T TESTs\n')
        f.write('##### \n')
        f.write('##### \n')
        f.write('MODELS COMPARISON\n')
        f.write('##### \n')
        for comparison in comparisons:
            #d = corr_values[comparison[0]].mean(axis=1) - corr_values[comparison[1]].mean(axis=1)
            t_result = ttest_rel(corr_values_mean_corrected[comparison[0]],corr_values_mean_corrected[comparison[1]])
            t, p = t_result.statistic, t_result.pvalue
            df = t_result.df  # degrees of freedom from the t-test result
            d = cohens_d(corr_values_mean_corrected[comparison[0]], corr_values_mean_corrected[comparison[1]])
            print(f'{comparison[0]} vs {comparison[1]}' )
            print(t)
            print(p)
            print(f"Cohen's d = {d}")
            print(f"df = {df}")
            f.write('#')
            f.write(f'{comparison[0]} VS {comparison[1]} \n')
            f.write(f't={t}, p={p}, df={df}, Cohen\'s d={d:.4f} \n')
            f.write('\n')

if __name__ == '__main__':
    corr_values = load_bootstrap_values()
    alpha=0.05
    comparisons = [ ['all-dctrained','all-alexnettrained'],
                    ['all-dcrandom','all-alexnettrained'],['all-dcrandom','all-dctrained'],
                    ['all-alexnettrained-with-dcrandom2','all-alexnettrained'],
                    ['all-dctrained-with-dcrandom2','all-dctrained']]

    grand_average = np.mean(np.array(list(corr_values.values())))
    across_model_average = np.mean(np.array(list(corr_values.values())), axis=0)
    corr_values_mean_corrected = {}
    for key in corr_values.keys():
        corr_values_mean_corrected[key] = np.mean((corr_values[key] - across_model_average + grand_average),axis=1)
        print(np.mean(corr_values_mean_corrected[key]))

    bootstrapping_confidenceinterval()
    posthoc_ttest()