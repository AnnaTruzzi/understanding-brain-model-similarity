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
from scipy import stats
from scipy.spatial.distance import squareform
import collections
import re
import skbio
import seaborn as sns
import pingouin as pg
import pandas as pd
import glob
from sklearn.linear_model import LinearRegression
import scipy
from scipy.stats import pearsonr, spearmanr, kendalltau,kruskal
import seaborn as sns
from scipy.stats import mannwhitneyu
from scipy.stats import wilcoxon
from scipy.stats import ttest_rel
from config import FIGURES_ROOT, RESULTS_ROOT


DC_RANDOM_COLOR = '#0072B2'
DC_TRAINED_COLOR = '#E69F00'
ALEXNET_TRAINED_COLOR = '#009E73'


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

def corr_with_brain_plot(dataframe,flag):
    color_map = {
        'dcrandomstate': DC_RANDOM_COLOR,
        'dc100epochs': DC_TRAINED_COLOR,
        'alexnetpretrained': ALEXNET_TRAINED_COLOR,
    }
    if flag != 'with_dctrained':
        dataframe = dataframe[dataframe['net_type']!='dc100epochs']
        hue_order = ['dcrandomstate', 'alexnetpretrained']
    else:
        hue_order = ['dcrandomstate', 'dc100epochs', 'alexnetpretrained']
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
    sns.lineplot(x="layer", y="corr", hue="net_type", data=dataframe[dataframe['ROI']=='EVC'], markers=True,
                 dashes=False, ax=ax[0], legend=False, palette=color_map, hue_order=hue_order)
    ax[0].set_ylim((-0.05,0.45))
    ax[0].set_title('Comparison to EVC')
    ax[0].axhline(0.38, color='grey', lw=2, alpha=0.4)
    ax[0].axhline(0.31, color='gray', lw=2, alpha=0.4)
    ax[0].axhspan(0.31, 0.38, facecolor='gray', alpha=0.4)
        
    sns.lineplot(x="layer", y="corr", hue="net_type", data=dataframe[dataframe['ROI']=='IT'], markers=True,
                 dashes=False, ax=ax[1], legend=False, palette=color_map, hue_order=hue_order)
    ax[1].set_title('Comparison to IT')
    ax[1].set_ylim((-0.015,0.45))
    ax[1].axhline(0.28, color='gray', lw=2, alpha=0.4)
    ax[1].axhline(0.42, color='gray', lw=2, alpha=0.4)
    ax[1].axhspan(0.28, 0.42, facecolor='gray', alpha=0.4)

    if flag != 'with_dctrained':
        ax[0].text(0, 0.15, "***", ha='center', va='bottom', fontsize=10)
        ax[0].text(1, 0.20, "***", ha='center', va='bottom', fontsize=10)
        ax[0].text(2, 0.18, "***", ha='center', va='bottom', fontsize=10)
        ax[0].text(3, 0.18, "***", ha='center', va='bottom', fontsize=10)
        ax[0].text(4, 0.18, "***", ha='center', va='bottom', fontsize=10)
        ax[0].text(5, 0.18, "**", ha='center', va='bottom', fontsize=10)
        ax[0].text(6, 0.18, "**", ha='center', va='bottom', fontsize=10)

        ax[1].text(0, 0.15, "***", ha='center', va='bottom', fontsize=10)
        ax[1].text(1, 0.20, "***", ha='center', va='bottom', fontsize=10)
        ax[1].text(2, 0.18, "**", ha='center', va='bottom', fontsize=10)
        ax[1].text(3, 0.18, "*", ha='center', va='bottom', fontsize=10)
        ax[1].text(4, 0.18, "**", ha='center', va='bottom', fontsize=10)

    plt.savefig(FIGURES_ROOT / f'comparison_to_alexnet{flag}.png')
    plt.savefig(FIGURES_ROOT / f'comparison_to_alexnet{flag}.pdf')

def corr_with_brain_check_dist(corr_df,layer,net_type):
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10,10))
    ax = ax.flatten()  # Flatten the 2D array to 1D for easier indexing
    for i,net in enumerate(net_type):
        sns.distplot(corr_df[corr_df['net_type']==net]['corr'],ax=ax[i])
        s,p = scipy.stats.shapiro(corr_df[corr_df['net_type']==net]['corr'])
        ax[i].set_title(f'{net} - s={np.round(s,3)}, p={np.round(p,3)}')
    plt.savefig(FIGURES_ROOT / f'corr_dist_allnets_{layer}.png')

def corr_with_brain_anova(corr_df,flag):
    random_1 = corr_df[(corr_df['ROI']==flag) & (corr_df['net_type']=='dcrandomstate') & (corr_df['layer']=='ReLu1')]['corr']
    random_2 = corr_df[(corr_df['ROI']==flag) & (corr_df['net_type']=='dcrandomstate') & (corr_df['layer']=='ReLu2')]['corr']
    random_3 = corr_df[(corr_df['ROI']==flag) & (corr_df['net_type']=='dcrandomstate') & (corr_df['layer']=='ReLu3')]['corr']
    random_4 = corr_df[(corr_df['ROI']==flag) & (corr_df['net_type']=='dcrandomstate') & (corr_df['layer']=='ReLu4')]['corr']
    random_5 = corr_df[(corr_df['ROI']==flag) & (corr_df['net_type']=='dcrandomstate') & (corr_df['layer']=='ReLu5')]['corr']
    random_6 = corr_df[(corr_df['ROI']==flag) & (corr_df['net_type']=='dcrandomstate') & (corr_df['layer']=='ReLu6')]['corr']
    random_7 = corr_df[(corr_df['ROI']==flag) & (corr_df['net_type']=='dcrandomstate') & (corr_df['layer']=='ReLu7')]['corr']

    trained_1 = corr_df[(corr_df['ROI']==flag) & (corr_df['net_type']=='dc100epochs') & (corr_df['layer']=='ReLu1')]['corr']
    trained_2 = corr_df[(corr_df['ROI']==flag) & (corr_df['net_type']=='dc100epochs') & (corr_df['layer']=='ReLu2')]['corr']
    trained_3 = corr_df[(corr_df['ROI']==flag) & (corr_df['net_type']=='dc100epochs') & (corr_df['layer']=='ReLu3')]['corr']
    trained_4 = corr_df[(corr_df['ROI']==flag) & (corr_df['net_type']=='dc100epochs') & (corr_df['layer']=='ReLu4')]['corr']
    trained_5 = corr_df[(corr_df['ROI']==flag) & (corr_df['net_type']=='dc100epochs') & (corr_df['layer']=='ReLu5')]['corr']
    trained_6 = corr_df[(corr_df['ROI']==flag) & (corr_df['net_type']=='dc100epochs') & (corr_df['layer']=='ReLu6')]['corr']
    trained_7 = corr_df[(corr_df['ROI']==flag) & (corr_df['net_type']=='dc100epochs') & (corr_df['layer']=='ReLu7')]['corr']

    trained_1_alexnet = corr_df[(corr_df['ROI']==flag) & (corr_df['net_type']=='alexnetpretrained') & (corr_df['layer']=='ReLu1')]['corr']
    trained_2_alexnet = corr_df[(corr_df['ROI']==flag) & (corr_df['net_type']=='alexnetpretrained') & (corr_df['layer']=='ReLu2')]['corr']
    trained_3_alexnet = corr_df[(corr_df['ROI']==flag) & (corr_df['net_type']=='alexnetpretrained') & (corr_df['layer']=='ReLu3')]['corr']
    trained_4_alexnet = corr_df[(corr_df['ROI']==flag) & (corr_df['net_type']=='alexnetpretrained') & (corr_df['layer']=='ReLu4')]['corr']
    trained_5_alexnet = corr_df[(corr_df['ROI']==flag) & (corr_df['net_type']=='alexnetpretrained') & (corr_df['layer']=='ReLu5')]['corr']
    trained_6_alexnet = corr_df[(corr_df['ROI']==flag) & (corr_df['net_type']=='alexnetpretrained') & (corr_df['layer']=='ReLu6')]['corr']
    trained_7_alexnet = corr_df[(corr_df['ROI']==flag) & (corr_df['net_type']=='alexnetpretrained') & (corr_df['layer']=='ReLu7')]['corr']

    print(flag)
    all_nets_stats = kruskal(random_1,random_2,random_3,random_4,random_5,random_6,random_7,trained_1,trained_2,trained_3,trained_4,trained_5,trained_6,trained_7,trained_1_alexnet,
                    trained_2_alexnet,trained_3_alexnet,trained_4_alexnet,trained_5_alexnet,trained_6_alexnet,trained_7_alexnet)
    vsalexnet_stats = kruskal(random_1,random_2,random_3,random_4,random_5,random_6,random_7,trained_1_alexnet,
                    trained_2_alexnet,trained_3_alexnet,trained_4_alexnet,trained_5_alexnet,trained_6_alexnet,trained_7_alexnet)
    print(stats)
    print('Testing for normality...')
    print(scipy.stats.shapiro(corr_df[corr_df['ROI']=='EVC']['corr']))
    print(scipy.stats.shapiro(corr_df[corr_df['ROI']=='IT']['corr']))
    print('Testing for homoscedasticity...')
    print(pg.homoscedasticity(data=corr_df[corr_df['ROI']=='EVC']))
    print(pg.homoscedasticity(data=corr_df[corr_df['ROI']=='IT']))
    return all_nets_stats, vsalexnet_stats


def posthoc_tests(corr_df,layer,flag):
        print(f'{flag} - {layer}')
        dist_random = corr_df[(corr_df['ROI']==flag) & (corr_df['net_type']=='dcrandomstate') & (corr_df['layer']==layer)]['corr']
        dist_trained = corr_df[(corr_df['ROI']==flag) & (corr_df['net_type']=='alexnetpretrained') & (corr_df['layer']==layer)]['corr']
        res = ttest_rel(dist_random, dist_trained)
        print(res)

        # Calculate Cohen's d
        d = cohens_d(dist_random, dist_trained)
        print(res)
        print(f"Cohen's d = {d}")
        return res, d

if __name__ == '__main__':
    layers = ['ReLu1','ReLu2','ReLu3','ReLu4','ReLu5','ReLu6','ReLu7']
    corr_dc = pd.read_csv(RESULTS_ROOT / 'corr_dc_mri_variability.csv')
    corr_alexnet = pd.read_csv(RESULTS_ROOT / 'corr_alexnet_mri_variability.csv')
    corr_df = pd.concat([corr_dc,corr_alexnet])
    corr_df['net_type'] = corr_df['net']+corr_df['state']
    net_type_list = ['dcrandomstate','dc100epochs','alexnetpretrained','alexnetrandomstate']
    ROIs = ['EVC','IT']
    plot_flags = ['with_dctrained','']
    for plot_flag in plot_flags:
        corr_with_brain_plot(corr_df[corr_df['net_type']!='alexnetrandomstate'],plot_flag)
    
    with open(RESULTS_ROOT / 'corrvalues_distribution_check_with_alexnet.txt', 'w') as f:
        for ROI in ROIs:
            for layer in layers:
                corr_to_check = corr_df[(corr_df['ROI']==ROI) & (corr_df['layer']==layer)]
                corr_with_brain_check_dist(corr_to_check,layer,net_type_list)
                f.write(f'\n ####### {ROI} \n')
                f.write(f'\n ####### {layer} \n')
                s_dcrandom,p_dcrandom = scipy.stats.shapiro(corr_to_check[corr_to_check['state']=='randomstate']['corr'])
                f.write(f'DC RANDOMSTATE normality - s={s_dcrandom}, p={p_dcrandom} \n')
                s_dctrained,p_dctrained = scipy.stats.shapiro(corr_to_check[corr_to_check['state']=='100epochs']['corr'])
                f.write(f'DC 100 EPOCHS normality - s={s_dctrained}, p={p_dctrained} \n')
                s_alexnetrandom,p_alexnetrandom = scipy.stats.shapiro(corr_to_check[corr_to_check['state']=='randomstate']['corr'])
                f.write(f'ALEXNET RANDOMSTATE normality - s={s_alexnetrandom}, p={p_alexnetrandom} \n')
                s_alexnettrained,p_alexnettrained = scipy.stats.shapiro(corr_to_check[corr_to_check['state']=='100epochs']['corr'])
                f.write(f'ALEXNET 100 EPOCHS normality - s={s_alexnettrained}, p={p_alexnettrained} \n')

    with open(RESULTS_ROOT / 'corr_to_brain_analysis_alexnet_comparison.txt', 'w') as f:
        for ROI in ROIs:
            f.write(f'\n ####### {ROI} \n')
            all_nets_stat_kruskal, vs_alexnet_stats_kruskal = corr_with_brain_anova(corr_df[corr_df['net_type']!='alexnetrandomstate'],flag=ROI)
            f.write('KRUSKAL test - comparison across random dc, trained dc, and trained alexnet \n')
            f.write(f'{all_nets_stat_kruskal} \n')
            f.write('KRUSKAL test - comparison between random DC and trained AlexNet \n')
            f.write(f'{vs_alexnet_stats_kruskal} \n')
            f.write('Student\'s t test \n')
            for layer in layers:
                posthoc = posthoc_tests(corr_df[(corr_df['net_type']!='alexnetrandomstate') & (corr_df['net_type']!='dc100epochs')],layer,flag=ROI)
                f.write(f'{layer}: {posthoc} \n')
