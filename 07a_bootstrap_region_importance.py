import warnings

warnings.filterwarnings("ignore")

import sys
import os.path
import ssm
import pickle
import numpy as np
import pandas as pd
import scipy
import copy
from tqdm import tqdm

from runwise_ts_log_data import get_ts_log_data_blocked

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import to_rgba
import seaborn as sns

from sympy.utilities.iterables import multiset_permutations
from ssm.util import *
from scipy.stats import ttest_ind, wilcoxon, ranksums,ttest_1samp,spearmanr
from scipy.signal import find_peaks

from nilearn.image import load_img, new_img_like
from nilearn.plotting import plot_img_on_surf

from scipy.spatial.distance import cdist
import rle
import awkward as ak

with open('pkl/emoprox2_dataset_timeseries+inputs_MAX85.pkl','rb') as f:
    orig_df = pickle.load(f)
subj_list = sorted(orig_df['pid'].unique())
# subj_list = subj_list[:] # remove first 30 subjects
orig_df = orig_df[orig_df['pid'].isin(subj_list)]

K = 6
D = 10
N = 85
num_subjs = 92
M=20
num_resamples = 500

with open(f'pkl/all_dfs_models_K{K}_D{D}_N{N}_{num_subjs}subjs_{num_resamples}resamples.pkl','rb') as f:
    all_dfs,all_models = pickle.load(f)

with open(f'pkl/state_masks_K{K}_D{D}_N{N}_{num_subjs}subjs_{num_resamples}resamples.pkl','rb') as f:
    state_masks = pickle.load(f)

l = int(sys.argv[1])
r = int(sys.argv[2])
keys_to_include = list(range(l,r+1))

all_dfs = {key: all_dfs[key] for key in keys_to_include}
all_models = {key: all_models[key] for key in keys_to_include}
state_masks = {key: state_masks[key] for key in keys_to_include}


print(len(all_dfs),len(all_models),len(state_masks))

for df,model,state_mask in tqdm(zip(all_dfs.values(),all_models.values(),state_masks.values())):
    print(type(df),type(model),type(state_mask))
    As = model.dynamics.As
    bs = model.dynamics.bs
    Vs = model.dynamics.Vs
    C = model.emissions.Cs[0]
    d = model.emissions.ds[0]
    Cinv = np.linalg.pinv(C) 
    Vhat_statewise = np.zeros((K,N,M))
    Ahat_statewise = np.zeros((K,N,N))
    bhat_statewise = np.zeros((K,N))
    for idx_state in range(K):
        # y_{t} = C x_{t} + d
        # x_{t+1} = A x_{t} + b
        # y_{t+1} = C x_{t+1} + d = C (A x_{t} + b) + d = C A Cinv y_{t} + C b + d
        Ahat_statewise[idx_state,:,:] = C@As[idx_state]@Cinv
        Vhat_statewise[idx_state,:,:] = C@Vs[idx_state] 
        bhat_statewise[idx_state,:] = C@bs[idx_state] + (np.eye(N)-C@As[idx_state]@Cinv)@d

    identity = np.eye(N)
    df['roi_importance'] = [None]*df.shape[0]
    for idx_row,row in df.iterrows():
        y = row['timeseries']
        z = row['discrete_states']
        u = np.argmax(row['input'],axis=1)
        roi_importance = np.zeros_like(y)
        for idx_roi in range(N):
            for t in range(y.shape[0]):
                roi_importance[t,idx_roi] = np.linalg.norm(y[t,idx_roi]*Ahat_statewise[z[t],:,idx_roi]+identity[:,idx_roi]*Vhat_statewise[z[t],idx_roi,u[t]]+identity[:,idx_roi]*bhat_statewise[idx_state,idx_roi])
        df.at[idx_row,'roi_importance'] = roi_importance


with open(f'pkl/all_dicts_K{K}_D{D}_N{N}_{num_subjs}subjs_{num_resamples}resamples_l={l}_r={r}_allfactors.pkl','wb') as f:
    pickle.dump([all_dfs,all_models,state_masks],f)
