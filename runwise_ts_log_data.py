# general
import numpy as np
import pandas as pd
from pandas import read_csv
# paths etc
from os.path import join as pjoin, dirname, realpath
import os
# progressbar
from tqdm.notebook import tqdm
# emoprox stimulus utilities
from stimulus_utils import subj_timing, get_base_stimulus
untimed_data = [[get_base_stimulus(run, block) for block in range(2)] for run in range(6)]

# paths
PROJ = '/home/joyneelm/emoprox2_slds_bootstrap_resamples'
DATASET = pjoin(PROJ,'dataset/ROI_timeseries/')
DATASET_ABA36 = pjoin(PROJ,'dataset/ROI_timeseries_ABA36/')
DATASET_MAX85 = pjoin(PROJ,'dataset/ROI_timeseries_MAX85/')

'''
Get list of participant data files
'''

yoked = read_csv(pjoin(dirname(realpath(__file__)),'CON_yoked_table_122PT.csv')).query('use==1')
run_columns = ['run%s' % i for i in range(6)]
control = yoked[['control'] + run_columns]
control.rename(columns={'control':'subj'}, inplace=True)
uncontrol = yoked[['uncontrol'] + run_columns]
uncontrol.rename(columns={'uncontrol':'subj'}, inplace=True)
subj_run_df = pd.concat([control, uncontrol],sort=True)
# display(subj_run_df)

'''
Create dataframe from dataset
'''
num_TRs_per_run = 360

def zscore(X):
    ''' to zscore BOLD timeseries of every run'''
    assert(X.shape[0]==num_TRs_per_run)
    return (X-np.mean(X,axis=0))/np.std(X,axis=0)

def resample_to_TR(df,col,default_val=-1):
    ''' resample a column in behavioral data to TRs'''
    ret = np.zeros(num_TRs_per_run)
    for TR in range(num_TRs_per_run):
        row = np.argmin(np.abs(df.time.values - TR*1.25))
        diff = np.min(np.abs(df.time.values - TR*1.25))
        if diff > 1.25:
            val = default_val
        else:
            val = df.iloc[row][col]
        ret[TR] = val
    return ret

cols = untimed_data[0][0][0].columns
def align_data(run_untimed, run_timing):
    # initialize data for piece caps. end is for the pieces at
    # either end of the block, tween is for those between shocks
    end = pd.Series([0, 0, 0, 0, 0, 0, 0], index=cols)
    # loop over each block assign timing and concatenate
    run_data = []
    for block in range(2):
        block_untimed = run_untimed[block]        
        block_timing = run_timing[block]
        ###########################################
        block_onset_time = block_timing[0][0]
        block_offset_time = block_timing[-1][-1]
        ###########################################
        for i, piece in enumerate(block_untimed):
            piece = piece.copy()
            # evenly divide the piece duration into the correct
            # number of time points to apply to the data
            start, stop = block_timing[i]
            piece['time'] = np.linspace(start, stop, num=piece.shape[0])
            ############################################################
            piece['onset'] = 0
            piece['offset'] = 0
            ############################################################
            piece = piece.set_index('time')
            
            # select the caps for the piece
            if i == 0:
                prerow = end
                postrow = piece.iloc[-1]
                postrow.at['direction'] = 0
                postrow.at['speed'] = 0
                ######################################################################################
                onset_indx = (piece.index >= block_onset_time) & (piece.index <= block_onset_time + 1)
                piece.loc[onset_indx, 'onset'] = 1
                #######################################################################################
            elif i == len(block_untimed) - 1:
                prerow = block_untimed[i-1].iloc[-1]
                prerow.at['direction'] = 0
                prerow.at['speed'] = 0
                postrow = end
                #########################################################################################
                offset_indx = (piece.index >= block_offset_time) & (piece.index <= block_offset_time + 1)
                piece.loc[offset_indx, 'offset'] = 1
                #########################################################################################
            else:
                prerow = block_untimed[i-1].iloc[-1]
                prerow.at['direction'] = 0
                prerow.at['speed'] = 0
                postrow = piece.iloc[-1]
                postrow.at['direction'] = 0
                postrow.at['speed'] = 0
            #######################
            prerow['onset'] = 0
            prerow['offset'] = 0
            postrow['onset'] = 0
            postrow['offset'] = 0
            #######################
            # assign a time offset by one screen refresh duration 
            delta_t = piece.index[1] - piece.index[0]
            prerow = pd.DataFrame([prerow], index=[start - delta_t])
            postrow = pd.DataFrame([postrow], index=[stop + delta_t])
            capped_piece = pd.concat([prerow, piece, postrow])
            
            run_data.append(capped_piece)
        
    run_data = pd.concat(run_data)
    
    # add a row for the end of run
    final_piece = pd.DataFrame([end], index=[run_timing[-1]])
    run_data = run_data.append(final_piece)
    
    return run_data

def GAM(t, p, q):
    return (t/(p*q))**p * np.exp(p - t/q)


untimed_data = [[get_base_stimulus(run, block) for block in range(2)] for run in range(6)]
def get_stim_data(pid, rid, sampling_rate=0.01, to_TR=False):
    ''' get stimulus data of moving circles at every frame '''
    run_timing = subj_timing(pid,rid)
    run_untimed = untimed_data[rid]
    run_data = align_data(run_untimed,run_timing)
    run_data.index = pd.to_datetime(run_data.index, unit='s')
    rule = f'{int(sampling_rate*1000)}L'
    run_data = run_data.resample(rule,closed='right').mean().interpolate('linear')

    # block timing
    bl0_begin, bl0_end = int(run_timing[0][0][0]/sampling_rate), int(run_timing[0][-1][1]/sampling_rate)
    bl1_begin, bl1_end = int(run_timing[1][0][0]/sampling_rate), int(run_timing[1][-1][1]/sampling_rate)
    block = np.zeros(run_data.shape[0])
    block[bl0_begin:bl0_end+1] = 1
    block[bl1_begin:bl1_end+1] = 2
    run_data['block'] = block

    # additional HRF columns
    sampleRate = 1000*sampling_rate
    t = np.arange(0, 20+1/sampleRate, 1/sampleRate)
    hemoResp = GAM(t, 8.6, 0.547)
    hemoResp /= hemoResp.sum()
    for col in ['proximity','direction','speed']:
        run_data[col+'_hrf'] = np.convolve(hemoResp, run_data[col].values, mode='full')[:run_data.shape[0]]


    # run_data[run_data['direction']>0] = 1
    # run_data[run_data['direction']<0] = -1
    tstart = pd.to_datetime('0', unit='s')
    tstop = pd.to_datetime('449.99', unit='s')
    resampled_data = run_data.loc[tstart:tstop].tail(450*100)
    resampled_data.index = resampled_data.index.asi8 / 1e9
    resampled_data = resampled_data.reset_index().rename(columns={'index':'time'})
    if to_TR:
        frac = int(1.25/sampling_rate)
        resampled_data = resampled_data.iloc[::frac, :]
        resampled_data.reset_index(inplace=True)
    return resampled_data



def get_ts_log_data(num_subjects=None):
    
    df_colnames = ['pid','rid','block','timeseries','proximity','direction','speed','time','proximity_hrf','direction_hrf','speed_hrf']
    df_data = {key:[] for key in df_colnames}
    
    counter = 0
    for _,row in tqdm(subj_run_df.iterrows()):
        pid = row.subj.strip('CON')
        run_list = np.arange(6)[row.loc['run0':'run5'].astype(bool)]
        num_runs = len(run_list)
        
        ts = np.loadtxt(pjoin(DATASET,f'CON{pid}/CON{pid}_resids_REML.1D')).T
        
        # splitting into runs
        # each chunk is a (num_TRs_per_run x num_ROIs) 2D array corresponding to a run 
        num_runs = ts.shape[0]//num_TRs_per_run
        ts = np.split(ts, indices_or_sections=num_runs)

        #zscore every run separately
        ts = [zscore(run_ts) for run_ts in ts]

        assert(np.stack(ts).shape[1]==num_TRs_per_run)

        for idx_run, rid in enumerate(run_list):
            df_data['pid'].append(pid)
            df_data['rid'].append(rid)
            df_data['timeseries'].append(ts[idx_run])

            df_stim = get_stim_data(pid, rid,sampling_rate=0.01,to_TR=True)
            df_data['block'].append(df_stim['block'].values)
            df_data['proximity'].append(df_stim['proximity'].values)
            df_data['direction'].append(df_stim['direction'].values)
            df_data['speed'].append(df_stim['speed'].values)
            df_data['proximity_hrf'].append(df_stim['proximity_hrf'].values)
            df_data['direction_hrf'].append(df_stim['direction_hrf'].values)
            df_data['speed_hrf'].append(df_stim['speed_hrf'].values)
            df_data['time'].append(df_stim['time'].values)
        
        counter += 1
        if num_subjects is not None:
            if counter > num_subjects:
                break

    df = pd.DataFrame(df_data)
    # display(df)
    return df

def get_ts_log_data_blocked(num_subjects=None):
    
    df_colnames = ['pid','rid','block','timeseries','proximity','direction','speed','time','proximity_hrf','direction_hrf','speed_hrf','censor','block_mask']
    df_data = {key:[] for key in df_colnames}
    
    counter = 0
    for _,row in tqdm(subj_run_df.iterrows()):
        pid = row.subj.strip('CON')
        run_list = np.arange(6)[row.loc['run0':'run5'].astype(bool)]
        num_runs = len(run_list)

        ts = np.loadtxt(pjoin(DATASET,f'CON{pid}/CON{pid}_resids_REML.1D')).T
        
        censor = np.loadtxt(pjoin(DATASET,f'censorfiles/CON{pid}_censor15.txt')).astype('bool')
        censor = np.split(censor, indices_or_sections=num_runs)
        censor = [np.repeat(np.expand_dims(part,axis=1),ts.shape[1],axis=1) for part in censor]

        # splitting into runs
        # each chunk is a (num_TRs_per_run x num_ROIs) 2D array corresponding to a run 
        num_runs = ts.shape[0]//num_TRs_per_run
        ts = np.split(ts, indices_or_sections=num_runs)
        
        #zscore every run separately
        ts = [zscore(run_ts) for run_ts in ts]

        assert(np.stack(ts).shape[1]==num_TRs_per_run)

        for idx_run, rid in enumerate(run_list):
            for idx_block in [1,2]:
                df_data['pid'].append(pid)
                df_data['rid'].append(rid)
                df_data['block'].append(idx_block)
                
                df_stim = get_stim_data(pid, rid,sampling_rate=0.01,to_TR=True)

                block_mask = (df_stim['block'].values == idx_block)

                df_data['block_mask'].append(block_mask)
                df_data['censor'].append(censor[idx_run][block_mask, :])
                df_data['timeseries'].append(ts[idx_run][block_mask, :])
                df_data['proximity'].append(df_stim['proximity'].values[block_mask])
                df_data['direction'].append(df_stim['direction'].values[block_mask])
                df_data['speed'].append(df_stim['speed'].values[block_mask])
                df_data['proximity_hrf'].append(df_stim['proximity_hrf'].values[block_mask])
                df_data['direction_hrf'].append(df_stim['direction_hrf'].values[block_mask])
                df_data['speed_hrf'].append(df_stim['speed_hrf'].values[block_mask])
                df_data['time'].append(df_stim['time'].values[block_mask])
        
        counter += 1
        if num_subjects is not None:
            if counter >= num_subjects:
                break

    df = pd.DataFrame(df_data)
    # display(df)
    return df

def get_ts_log_data_blocked_ABA36(num_subjects=None):
    
    df_colnames = ['pid','rid','block','timeseries','proximity','direction','speed','time','proximity_hrf','direction_hrf','speed_hrf','censor','block_mask']
    df_data = {key:[] for key in df_colnames}
    
    counter = 0
    for _,row in tqdm(subj_run_df.iterrows()):
        pid = row.subj.strip('CON')
        run_list = np.arange(6)[row.loc['run0':'run5'].astype(bool)]
        num_runs = len(run_list)

        ts = np.loadtxt(pjoin(DATASET_ABA36,f'CON{pid}/CON{pid}_resids_REML.1D')).T
        
        censor = np.loadtxt(pjoin(DATASET,f'censorfiles/CON{pid}_censor15.txt')).astype('bool')
        censor = np.split(censor, indices_or_sections=num_runs)
        censor = [np.repeat(np.expand_dims(part,axis=1),ts.shape[1],axis=1) for part in censor]

        # splitting into runs
        # each chunk is a (num_TRs_per_run x num_ROIs) 2D array corresponding to a run 
        num_runs = ts.shape[0]//num_TRs_per_run
        ts = np.split(ts, indices_or_sections=num_runs)
        
        #zscore every run separately
        ts = [zscore(run_ts) for run_ts in ts]

        assert(np.stack(ts).shape[1]==num_TRs_per_run)

        for idx_run, rid in enumerate(run_list):
            for idx_block in [1,2]:
                df_data['pid'].append(pid)
                df_data['rid'].append(rid)
                df_data['block'].append(idx_block)
                
                df_stim = get_stim_data(pid, rid,sampling_rate=0.01,to_TR=True)

                block_mask = (df_stim['block'].values == idx_block)

                df_data['block_mask'].append(block_mask)
                df_data['censor'].append(censor[idx_run][block_mask, :])
                df_data['timeseries'].append(ts[idx_run][block_mask, :])
                df_data['proximity'].append(df_stim['proximity'].values[block_mask])
                df_data['direction'].append(df_stim['direction'].values[block_mask])
                df_data['speed'].append(df_stim['speed'].values[block_mask])
                df_data['proximity_hrf'].append(df_stim['proximity_hrf'].values[block_mask])
                df_data['direction_hrf'].append(df_stim['direction_hrf'].values[block_mask])
                df_data['speed_hrf'].append(df_stim['speed_hrf'].values[block_mask])
                df_data['time'].append(df_stim['time'].values[block_mask])
        
        counter += 1
        if num_subjects is not None:
            if counter >= num_subjects:
                break

    df = pd.DataFrame(df_data)
    # display(df)
    return df

def get_ts_log_data_blocked_MAX85(num_subjects=None):
    
    df_colnames = ['pid','rid','block','timeseries','proximity','direction','speed','time','proximity_hrf','direction_hrf','speed_hrf','censor','block_mask']
    df_data = {key:[] for key in df_colnames}
    
    counter = 0
    for _,row in tqdm(subj_run_df.iterrows()):
        pid = row.subj.strip('CON')
        run_list = np.arange(6)[row.loc['run0':'run5'].astype(bool)]
        num_runs = len(run_list)

        ts = np.loadtxt(pjoin(DATASET_MAX85,f'CON{pid}/CON{pid}_resids_REML.1D')).T
        
        censor = np.loadtxt(pjoin(DATASET,f'censorfiles/CON{pid}_censor15.txt')).astype('bool')
        censor = np.split(censor, indices_or_sections=num_runs)
        censor = [np.repeat(np.expand_dims(part,axis=1),ts.shape[1],axis=1) for part in censor]

        # splitting into runs
        # each chunk is a (num_TRs_per_run x num_ROIs) 2D array corresponding to a run 
        num_runs = ts.shape[0]//num_TRs_per_run
        ts = np.split(ts, indices_or_sections=num_runs)
        
        #zscore every run separately
        ts = [zscore(run_ts) for run_ts in ts]

        assert(np.stack(ts).shape[1]==num_TRs_per_run)

        for idx_run, rid in enumerate(run_list):
            for idx_block in [1,2]:
                df_data['pid'].append(pid)
                df_data['rid'].append(rid)
                df_data['block'].append(idx_block)
                
                df_stim = get_stim_data(pid, rid,sampling_rate=0.01,to_TR=True)

                block_mask = (df_stim['block'].values == idx_block)

                df_data['block_mask'].append(block_mask)
                df_data['censor'].append(censor[idx_run][block_mask, :])
                df_data['timeseries'].append(ts[idx_run][block_mask, :])
                df_data['proximity'].append(df_stim['proximity'].values[block_mask])
                df_data['direction'].append(df_stim['direction'].values[block_mask])
                df_data['speed'].append(df_stim['speed'].values[block_mask])
                df_data['proximity_hrf'].append(df_stim['proximity_hrf'].values[block_mask])
                df_data['direction_hrf'].append(df_stim['direction_hrf'].values[block_mask])
                df_data['speed_hrf'].append(df_stim['speed_hrf'].values[block_mask])
                df_data['time'].append(df_stim['time'].values[block_mask])
        
        counter += 1
        if num_subjects is not None:
            if counter >= num_subjects:
                break

    df = pd.DataFrame(df_data)
    # display(df)
    return df
