import os
import scipy.io
import pickle
import numpy as np
from itertools import compress
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
import skbio
import seaborn as sns
import pingouin as pg
import pandas as pd
import glob
from config import ACTIVATIONS_ROOT, FIGURES_ROOT, IMAGE_NAMES_PATH, RDM_ROOT
from sklearn.linear_model import LinearRegression
import scipy
from scipy.stats import pearsonr, spearmanr, kendalltau


def load_dict(path):
    with open(path,'rb') as file:
        dict=pickle.load(file, encoding="latin1")
    return dict

def reorder_od(dict1,order):
   new_od = collections.OrderedDict([(k,None) for k in order if k in dict1])
   new_od.update(dict1)
   return new_od

def presentation_order(ordered_names_dict):
    orderedNames = []
    for item in sorted(ordered_names_dict.items()):
        orderedNames.append(item[1])
    return orderedNames

def rdm(act_matrix):
    rdm_matrix = distance.squareform(distance.pdist(act_matrix,metric='correlation'))
    return rdm_matrix

def rdm_plot(rdm, vmin, vmax, labels, main, outname):
    fig, ax = plt.subplots(figsize=(20,20))
    sns.heatmap(rdm, ax=ax, cmap='Blues_r', vmin=vmin, vmax=vmax, xticklabels=labels, yticklabels=labels)
    #plt.show()
    output_dir = FIGURES_ROOT / 'rdm_plots'
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / outname)
    plt.close()

def main(net,instance,state):
    act = load_dict(act_pth)
    img_names = load_dict(img_names_pth)
    img_names = reorder_od(img_names, sorted(img_names.keys()))
    orderedNames = presentation_order(img_names)
    orderedKeys = []
    for name in orderedNames:
        number = [i[0] for i in img_names.items() if i[1] == name]
        key = [k for k in act.keys() if '%s.jpg' %number[0] in k]
        orderedKeys.append(key[0])
    
    ############ order activations dictionary and reorganize it in order to get one dictionary per layer and image (instead of one dictionary per image)
    ordered_act = reorder_od(act,orderedKeys)
    layer_dict = collections.OrderedDict()
    for layer in range(0,len(layers)):
        layer_dict[layer] = collections.OrderedDict()
        for item in ordered_act.items():
                if layer < 5:
                    layer_dict[layer][item[0]] = np.mean(np.mean(np.squeeze(item[1][layer]),axis = 2), axis = 1)
                else:
                    layer_dict[layer][item[0]] = np.squeeze(item[1][layer])    


    ####### transform layer activations in matrix and calculate rdms
    layer_matrix_list = []
    net_rdm = []
    for l in layer_dict.keys():
        curr_layer = np.array([layer_dict[l][i] for i in orderedKeys])
        print(l)
        print(curr_layer.shape)
        layer_matrix_list.append(curr_layer)
        net_rdm.append(rdm(curr_layer))

    for i,layer in enumerate(layers):
        main = '%s_%d %s - %s' %(net,instance,state,layer)
        outname = '%s%d_%s_%s' %(net,instance,state,layer)
        print(outname)
        rdm_plot(net_rdm[i], vmin = 0, vmax = 1, labels = orderedNames, main = main, outname = outname+ '.png')
        np.save(RDM_ROOT / f'{outname}.npy', net_rdm[i])


### Use this python when launching the code: /opt/anaconda3/envs/unsupervised_brain/bin/python
if __name__ == '__main__':
    layers = ['ReLu1', 'ReLu2', 'ReLu3', 'ReLu4', 'ReLu5','ReLu6','ReLu7']
    states_dc = ['randomstate','100epochs']
    instances_dc = 11
    instances_alexnet = 2
    states_alexnet = ['randomstate','pretrained']
    nets = ['dc','alexnet']
    img_names_pth = IMAGE_NAMES_PATH
    img_names = load_dict(img_names_pth)
    img_names = reorder_od(img_names, sorted(img_names.keys()))

    for net in nets:
        if net == 'dc':
            for instance in range(1,instances_dc):
                for state in states_dc:
                    net = 'dc'
                    act_pth = ACTIVATIONS_ROOT / ('activations_%s_dc%d.pickle' % (state, instance))
                    main(net,instance, state)
        else:
            for instance in range(0,instances_alexnet):
                for state in states_alexnet:
                    act_pth = ACTIVATIONS_ROOT / ('niko92_activations_%s_torchalexnet.pickle' % state)
                    main(net,instance,state)
   