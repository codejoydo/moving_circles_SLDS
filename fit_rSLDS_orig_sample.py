from genericpath import exists
import warnings
warnings.filterwarnings("ignore")

import os.path
import ssm
import pickle
import numpy as np
import pandas as pd
# from runwise_ts_log_data import get_ts_log_data_blocked
import sys
from os.path import exists

idx_resample = int(sys.argv[1])

with open('pkl/emoprox2_dataset_timeseries+inputs_MAX85.pkl','rb') as f:
    df = pickle.load(f)

subj_list = sorted(df['pid'].unique())
# subj_list = subj_list[:30] # use first 30 subjects
subj_list = subj_list[30:] # remove first 30 subjects

df = df[df['pid'].isin(subj_list)]
print(len(subj_list),len(df['pid'].unique()),"subjects")

resampled_subj_list = subj_list
resampled_df = []
for pid in resampled_subj_list:
    resampled_df.append(df[df['pid']==pid])
resampled_df = pd.concat(resampled_df).reset_index().drop('index',axis=1)

brain_signals = list(resampled_df['timeseries'].values)
inputs = list(resampled_df['input'].values)

M = inputs[0].shape[1]
N = brain_signals[0].shape[1]
K = 6
D = 10
num_iters = 50
print(f"Number of timeseries = {len(brain_signals)}")
print(f"K={K}\tD={D}\tN={N}\tM={M}")

#if exists(f'pkl/rslds_emoprox2_K{K}_D{D}_N{N}_M{M}_{len(subj_list)}subjs_origsample{idx_resample}.pkl'):
#    sys.exit(0)

model = ssm.SLDS(
    N, K, D, M = M,
    transitions="recurrent",
    dynamics="t", 
    emissions="gaussian",
)

elbos, q = model.fit(
    brain_signals,
    inputs=inputs,
    method="laplace_em",
    variational_posterior="structured_meanfield",
    num_iters=num_iters,
    initialize=True,
    num_samples=10,
    num_init_restarts=1,
    num_init_iters=25,
    discrete_state_init_method='kmeans',
    alpha=0.5,
)   

with open(f'pkl/rslds_emoprox2_K{K}_D{D}_N{N}_M{M}_{len(subj_list)}subjs_origsample{idx_resample}.pkl','wb') as f:
    pickle.dump([model,q,elbos,resampled_subj_list],f)
