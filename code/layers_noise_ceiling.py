from scipy import fftpack
import numpy as np
import pylab as py
import scipy.io
from PIL import Image
import os
import pandas as pd
import glob
import re
import pickle
from scipy.spatial import distance
import matplotlib
from matplotlib import pyplot as plt
from scipy.spatial.distance import squareform
from itertools import combinations
import pandas as pd
import seaborn as sns
import collections
import h5py
from scipy.stats import pearsonr, spearmanr, kendalltau
import skbio
from scipy import io
from config import RESULTS_ROOT, load_rdm


def mantel(x,y):
    x = squareform(squareform(x,checks = False))
    y = squareform(squareform(y,checks = False))
    corr_value, p_value, n_value = skbio.stats.distance.mantel(x,y, method = 'spearman', permutations = 10000)
    return corr_value


def get_noise_ceiling(all_rdms):
    upper_list = []
    lower_list = []
    upper_list = []
    lower_list = []

    for i in range(0,instances-1):
        ####  upper bound
        interim_upper = mantel(all_rdms[i,:,:], np.mean(all_rdms,axis = 0))
        upper_list.append(interim_upper)

        ####  lower bound
        print(state)
        print(layer)
        print(i)
        selector = [x for x in range(all_rdms.shape[0]) if x !=i]
        interim_lower = mantel(all_rdms[i,:,:], np.mean(all_rdms[selector,:,:],axis = 0))
        lower_list.append(interim_lower)

    upper = np.mean(np.array(upper_list))
    lower = np.mean(np.array(lower_list))

    return upper, lower


if __name__ == '__main__':
    layers = ['ReLu1', 'ReLu2', 'ReLu3', 'ReLu4', 'ReLu5','ReLu6','ReLu7']
    states = ['randomstate','100epochs']
    instances = 11
    list_file = []
    for instance in range(1,instances):
        for layer in layers:
            for state in states:
                list_file.append(f'dc{instance}_{state}_{layer}')

    with open(RESULTS_ROOT / 'layers_noise_ceiling.txt', 'w') as f:
        for state in states:
            for layer in layers:
                all_layer_rdms = []
                curr_list = [i for i in list_file if state in i and layer in i]
                for name in curr_list:
                    rdm = load_rdm(name)
                    all_layer_rdms.append(rdm)

                upper,lower = get_noise_ceiling(np.array(all_layer_rdms))
                print(upper, lower)
                f.write('\n *************************** \n')
                f.write(f'{state} -- layer {layer} \n')
                f.write(f'Upper: {upper} \nLower: {lower} \n')


