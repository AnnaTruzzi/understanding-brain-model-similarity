import os
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
import h5py
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
from scipy.stats import pearsonr, spearmanr, kendalltau
import statsmodels.formula.api as smf
from PIL import Image
import random
from config import FMRI_PATH, IMAGE_NAMES_PATH, IMAGE_ROOT, RESULTS_ROOT, load_rdm


## load activations dictionary 
def load_dict(path):
    with open(path,'rb') as file:
        dict=pickle.load(file, encoding="latin1")
    return dict

   
def loadmat(matfile):
    try:
        f = h5py.File(matfile)
    except (IOError, OSError):
        import scipy.io as io
        return io.loadmat(matfile)
    else:
        return {name: np.transpose(f.get(name)) for name in f.keys()}

def reorder_od(dict1,order):
   new_od = collections.OrderedDict([(k,None) for k in order if k in dict1])
   new_od.update(dict1)
   return new_od

def presentation_order(ordered_names_dict):
    orderedNames = []
    for item in sorted(ordered_names_dict.items()):
        orderedNames.append(item[1])
    return orderedNames

def get_visual_features(img_pil,backcol=None):
  img=np.asarray(img_pil).astype('double')
  lum=np.mean(img,axis=2)
  hsv=np.asarray(img_pil.convert('HSV'))
  
  # If background colour not provided, get it from first pixel (may be risky!) number of pixels
  if not backcol:  
    backcol=img[0,0]
    print('Using top left pixel as background colour, value %d %d %d'%(backcol[0],backcol[1],backcol[2]))
  backlum=np.mean(backcol)

  # All pixels background colour
  foreground=np.any(img!=backcol,2)
  
  img_fg=img[foreground]
  hsv_fg=np.asarray(hsv)[foreground]
  
  features={}
  # Size - total of non-background pixels
  features['size']=np.sum(foreground)
  # Contrast - mean of sum of squared difference of each colour channel from background colour
  features['contrast']=np.mean(np.sum(np.power(img_fg-backcol,2.0),1))
  # Hue - mean of H
  features['hue']=np.mean(hsv_fg[:,0])
  # Lurid - mean standard deviation across colour channel
  features['lurid']=np.mean(np.std(img_fg,1))
  # Thinness 
  y,x=np.where(foreground)
  y=y-np.mean(y)
  x=x-np.mean(x)
  a=np.vstack((x,y))
  u,s,v=np.linalg.svd(a,full_matrices=False)
  features['thinness']=s[0]/s[1]
  # Radians away from being horizontal
  features['radiansoffhorizontal']=np.abs(np.arctan(u[0,1]/u[0,0]))
  
  return features


def main():
    dcrandom = {}
    dctrained = {}
    alexnettrained = {}

    for layer in layers:
        dcrandom_layer_rdms = []
        dctrained_layer_rdms = []
        for instance in range(1, dc_instances):
            dcrandom_rdm = load_rdm(f'dc{instance}_randomstate_{layer}')
            dctrained_rdm = load_rdm(f'dc{instance}_100epochs_{layer}')
            dcrandom_layer_rdms.append(dcrandom_rdm)
            dctrained_layer_rdms.append(dctrained_rdm)
        dcrandom_mean = squareform((np.mean(np.array(dcrandom_layer_rdms),axis=0)))
        dcrandom[f'dcrandom_{layer}'] = dcrandom_mean - np.mean(dcrandom_mean)
        
        dctrained_mean = squareform((np.mean(np.array(dctrained_layer_rdms),axis=0)))        
        dctrained[f'dctrained_{layer}'] = dctrained_mean - np.mean(dctrained_mean)
        alexnettrained_mean = squareform(load_rdm(f'alexnet1_pretrained_{layer}'))
        alexnettrained[f'alexnettrained_{layer}'] = alexnettrained_mean - np.mean(alexnettrained_mean)

    all_nets = {'dcrandom': dcrandom,
                'dctrained': dctrained,
                'alexnettrained': alexnettrained}
    
    fmri_mat = loadmat(FMRI_PATH)
    IT = fmri_mat['IT_RDMs']
    IT_mean = squareform(np.mean(IT,axis = 0),checks=False)
    IT_mean_corrected = IT_mean - np.mean(IT_mean)

    EVC = fmri_mat['EVC_RDMs']
    EVC_mean = squareform(np.mean(EVC,axis = 0),checks=False)
    EVC_mean_corrected = EVC_mean - np.mean(EVC_mean)

    all_ROIs = {'IT_mean_corrected': IT_mean_corrected,
                'EVC_mean_corrected':EVC_mean_corrected}

    all_ROIs_allsubj = {'IT': IT,
                'EVC':EVC}

    ############  Semantic features
    semantic_features_vec = np.array([1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,
                3,3,3,3,3,3,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,4,4,5,5,5,5,5,5,5,5,5,5,5,5,
                5,5,5,5,5,5,5,5,5,5,5,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6])

    animacy=np.array([1]*48+[2]*44)

    ############  Perceptual features
    perceptual_features_dic = {'size': collections.OrderedDict(), 
                            'contrast': collections.OrderedDict(), 
                            'hue': collections.OrderedDict(), 
                            'lurid': collections.OrderedDict(), 
                            'thinness': collections.OrderedDict(), 
                            'radiansoffhorizontal': collections.OrderedDict()}

    allimgs=[]
    allimg_names=[]
    backcol=[255,255,255]
    for img in glob.glob(os.path.join(str(IMAGE_ROOT), '*.jpg')):
        print(img)
        img_loaded = Image.open(img)
        features = get_visual_features(img_loaded,backcol)
        img_name = img.split('/')[-1].split('.')[0].split('_')[-1] 
        perceptual_features_dic['size'][img_name] = features['size']
        perceptual_features_dic['contrast'][img_name] = features['contrast']
        perceptual_features_dic['hue'][img_name] = features['hue']
        perceptual_features_dic['lurid'][img_name] = features['lurid']
        perceptual_features_dic['thinness'][img_name] = features['thinness']
        perceptual_features_dic['radiansoffhorizontal'][img_name] = features['radiansoffhorizontal']

        # Prepare pixel overlap measure
        img=np.asarray(img_loaded).astype('double')
        # All pixels background colour
        allimgs.append(np.any(img!=backcol,2))
        allimg_names.append(img_name)

    ind=np.argsort(allimg_names)
    allimgs=np.array(allimgs)
    allimgs=allimgs[ind,:]
    allimgs=np.reshape(allimgs,(allimgs.shape[0],-1))
    silhouetterdm=np.corrcoef(allimgs)    

    perceptual_features_dic['size'] = reorder_od(perceptual_features_dic['size'], sorted(perceptual_features_dic['size'].keys()))
    perceptual_features_dic['contrast'] = reorder_od(perceptual_features_dic['contrast'], sorted(perceptual_features_dic['contrast'].keys()))
    perceptual_features_dic['hue'] = reorder_od(perceptual_features_dic['hue'], sorted(perceptual_features_dic['hue'].keys()))
    perceptual_features_dic['lurid'] = reorder_od(perceptual_features_dic['lurid'], sorted(perceptual_features_dic['lurid'].keys()))
    perceptual_features_dic['thinness'] = reorder_od(perceptual_features_dic['thinness'], sorted(perceptual_features_dic['thinness'].keys()))
    perceptual_features_dic['radiansoffhorizontal'] = reorder_od(perceptual_features_dic['radiansoffhorizontal'], sorted(perceptual_features_dic['radiansoffhorizontal'].keys()))

    # Refactored
    sem_feats=['semantic_category', 'semantic_animacy']
    perc_feats=['size','contrast','hue','lurid','thinness','radiansoffhorizontal','silhouette']
    all_feats=sem_feats+perc_feats

    feat_rdms={}
    a=np.tile(semantic_features_vec,[92,1])
    feat_rdms['semantic_category']=1-(a==a.T)
    a=np.tile(animacy,[92,1])
    feat_rdms['semantic_animacy']=1-(a==a.T)

    # Refactored.
    for perc_feat in perc_feats:
        if perc_feat=='silhouette':
            feat_rdms[perc_feat]=silhouetterdm
        else:
            pf=np.array([perceptual_features_dic[perc_feat][x] for x in perceptual_features_dic[perc_feat]])
            a=np.tile(pf,[92,1])
            feat_rdms[perc_feat]=np.abs(a-a.T) # Euclidian distance between two scalars is just their absolute difference

    # Squareform and zero centre
    for k in feat_rdms:
        feat_rdms[k]=squareform(feat_rdms[k],checks=False)
        feat_rdms[k]=feat_rdms[k]-np.mean(feat_rdms[k])
        # Ensure all features are explicitly float64 
        feat_rdms[k]=feat_rdms[k].astype(np.float64)
        # For semantic features (binary 0/1 RDMs), add extremely small Gaussian noise
        # to prevent sklearn from misclassifying them as categorical variables.
        # Noise: mean=0, std=1e-10 (0.0000000001), which is negligible compared to 
        # typical RDM values (~0-1 range). This preserves the original structure while
        # ensuring the features are treated as continuous. Seed=42 for reproducibility.
        if k in sem_feats:
            np.random.seed(42)
            feat_rdms[k] = feat_rdms[k] + np.random.normal(0, 1e-10, size=feat_rdms[k].shape)


    test_ROIs=['IT','EVC']
    test_nets=['dcrandom','dctrained','alexnettrained']
    test_layers=['ReLu2','ReLu7']
    
    # Bootstrapping parameters
    n_bootstrap = 1000
    random.seed(42)  # For reproducibility

    proportion_list_allsubj=[]
    feature_list_allsubj=[]
    ROI_list_allsubj=[]
    net_list_allsubj=[]
    out_layer_list_allsubj=[]
    indirect_list=[]
    total_list=[]
    bootstrap_iteration_list=[]

    print(f"Starting {n_bootstrap} bootstrap iterations...")

    for test_ROI in test_ROIs:
        for test_net in test_nets:
            for test_layer in test_layers:
                print(f"Processing {test_ROI}, {test_net}, {test_layer}")
                
                # Get the number of subjects available
                n_subjects = all_ROIs_allsubj[test_ROI].shape[0]
                
                for bootstrap_iter in range(n_bootstrap):
                    if bootstrap_iter % 100 == 0:
                        print(f"  Bootstrap iteration {bootstrap_iter}/{n_bootstrap}")
                    
                    # Bootstrap sample: sample subjects with replacement
                    bootstrap_indices = np.random.choice(n_subjects, size=n_subjects, replace=True)
                    
                    # Create bootstrap sample of brain RDMs
                    bootstrap_brain_rdms = all_ROIs_allsubj[test_ROI][bootstrap_indices]
                    
                    # Calculate average brain RDM for this bootstrap sample
                    avg_brain_rdm = squareform(np.mean(bootstrap_brain_rdms, axis=0), checks=False)
                    avg_brain_rdm_corrected = avg_brain_rdm - np.mean(avg_brain_rdm)
                    
                    # Run separate mediation model for each feature
                    for feat in all_feats:
                        # Use the pre-computed average neural network RDM as IV
                        dfcols={f'{test_net}': all_nets[f'{test_net}'][f'{test_net}_{test_layer}'], 
                               f'{test_ROI}': avg_brain_rdm_corrected,
                               feat: feat_rdms[feat]}
                        
                        df=pd.DataFrame(dfcols)
                        
                        # Run mediation with single mediator
                        res = pg.mediation_analysis(data=df, x=test_net, m=[feat], y=test_ROI, 
                                                  alpha=0.05, seed=42+bootstrap_iter, return_dist=False)

                        # Check if expected paths exist in results.
                        # With a single mediator, pingouin labels the indirect path as
                        # "Indirect" (without mediator name). Keep backward compatibility
                        # with formats that include the mediator label.
                        indirect_path = (res['path'] == 'Indirect') | (res['path'] == 'Indirect ' + feat)
                        total_path = res['path'] == 'Total'
                        
                        if indirect_path.sum() == 0 or total_path.sum() == 0:
                            if bootstrap_iter == 0:  # Only print debug info on first iteration
                                print(f"    Debug: Available paths for {feat}: {res['path'].values}")
                            continue
                        
                        # Extract key values (use .values[0] to get the actual value from Series)
                        indirect = float(res['coef'][indirect_path].values[0])
                        tot = float(res['coef'][total_path].values[0])

                        # Skip if total is zero or very close to zero to avoid division issues
                        if abs(tot) < 1e-10:
                            continue

                        proportion_list_allsubj.append(indirect/tot)
                        indirect_list.append(indirect)
                        total_list.append(tot)
                        feature_list_allsubj.append(feat)
                        ROI_list_allsubj.append(test_ROI)
                        net_list_allsubj.append(test_net)
                        out_layer_list_allsubj.append(test_layer)
                        bootstrap_iteration_list.append(bootstrap_iter)                   

    out_dict_allsubj = {'proportion': proportion_list_allsubj,
                'feature': feature_list_allsubj,
                'ROI': ROI_list_allsubj,
                'net': net_list_allsubj,
                'layer': out_layer_list_allsubj,
                'indirect': indirect_list,
                'total': total_list,
                'bootstrap_iter': bootstrap_iteration_list}
    out_df_allsubj = pd.DataFrame(out_dict_allsubj)
    out_df_allsubj.to_csv(RESULTS_ROOT / 'mediation_separate_models_bootstrap.csv', index=False)
    
    print(f"Bootstrapping completed. Results saved with {len(out_df_allsubj)} total observations.")
    print(f"Expected: {len(test_ROIs) * len(test_nets) * len(test_layers) * n_bootstrap * len(all_feats)} observations")
    print(f"Success rate: {len(out_df_allsubj) / (len(test_ROIs) * len(test_nets) * len(test_layers) * n_bootstrap * len(all_feats)) * 100:.1f}%")


if __name__ == '__main__':
    layers = ['ReLu1', 'ReLu2', 'ReLu3', 'ReLu4', 'ReLu5','ReLu6','ReLu7']
    dc_instances = 11

    img_names_pth = IMAGE_NAMES_PATH
    img_names = load_dict(img_names_pth)
    img_names = reorder_od(img_names, sorted(img_names.keys()))
    orderedNames = presentation_order(img_names)

    main()
