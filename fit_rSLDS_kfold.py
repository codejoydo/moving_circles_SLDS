import ssm
import pickle
import sys

num_states = int(sys.argv[1])
idx_fold = int(sys.argv[2])
print(f"num states = {num_states}, idx fold = {idx_fold}")

with open('pkl/emoprox2_30subjs_data+inputs.pkl','rb') as f:
    df = pickle.load(f)

num_subjs = 30
subj_list = sorted(df.pid.unique())
del subj_list[(idx_fold-1)*3:idx_fold*3]
print("subjects",subj_list)
df = df[df.pid.isin(subj_list)]

Y = list(df.timeseries.values)
inputs = list(df.input.values)
input_dim = inputs[0].shape[1]
num_timepoints, num_rois = Y[0].shape
latent_dim = 10
num_iters = 50
print(f"Number of timeseries = {len(Y)}")
print(f"K={num_states}\tD={latent_dim}\tN={num_rois}\tT={num_timepoints}\tM={input_dim}")

model = ssm.SLDS(
    num_rois, 
    num_states, 
    latent_dim,
    M = input_dim,
    transitions="recurrent",
    dynamics="t", 
    emissions="gaussian",
)

elbos, q = model.fit(
    Y,
    inputs=inputs,
    # masks=masks,
    method="laplace_em",
    variational_posterior="structured_meanfield",
    num_iters=num_iters,
    initialize=True,
    num_samples=10,
    num_init_restarts=10,
    num_init_iters=100,
    discrete_state_init_method='kmeans',
    alpha=0.5,
    init_zs=None,
)   

with open(f'pkl/kfold/rslds_K{num_states}_D{latent_dim}_N{num_rois}_nsubjs{num_subjs}_emoprox_inputs_kfold{idx_fold}.pkl','wb') as f:
    pickle.dump([model, q, elbos],f)