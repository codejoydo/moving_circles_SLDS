import pandas as pd
import numpy as np
from json import load
from os.path import join, isdir, dirname, exists
from os import mkdir
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

def easy_mkdir(dst):
    path = ''
    for direct in dst.split('/'):
        path = join(path, direct)
        if not isdir(path):
            print("Making {}".format(path))
            mkdir(path)
        else:
            print("{} already exists. Skipping...".format(path))

PROJ = "/home/joyneelm/emoprox2_slds_bootstrap_resamples"
STIMHOME = join(PROJ,"dataset/stimtimes_eCON/stimdata") # sshfs this folder from bswift /data/bswift-1/Pessoa_Lab/eCON/documentation
STIMPATH = join(STIMHOME, "run{run}_bl{block}.b")
RAWHOME = join(PROJ,"dataset/stimtimes_eCON/raw_onsets")
DATPATH = join(RAWHOME, "CON{subj}/onsets/subjCON{subj}_run{run}.dat")
# PREPHOME = "../../bswift/dataset/preproc2"
# PREPPATH = join(PREPHOME, "CON{subj}/splitted_regs_fancy/CON{subj}_{reg}.txt")
# REGPATH = join(PREPHOME, "CON{subj}/splitted_regs_fancy/CON{subj}_{reg}.txt")

# Regressor names, processing specification
BASECOLS = ['proximity', 'direction', 'speed', 'c1x', 'c1y', 'c2x', 'c2y']

def get_base_stimulus(run, block):
    # load the data from the .b (json) file into a dictionary
    path = STIMPATH.format(run=run, block=block)
    with open(path, 'r') as f:
        data = load(f)
    
    # extract circle positions from the dictionary
    circle_pos = np.stack([data[cir] for cir in ['x1', 'x2']])
    # compute vector connecting circle centers
    # (squeeze removes length 1 dimensions)
    displacement_vector = np.diff(circle_pos, axis=0).squeeze()
    # convert vector into complex with x component as the real
    # and the y component as the imaginary part
    complex_vector = displacement_vector.view(np.complex128)
    
    ####################
    # Motion Variables
    # distance between circles
    distance = np.abs(complex_vector) # equivalent to euclidean formula
    # we define proximity = 1 - distance (max 0.9 min~0.1)
    proximity = 1 - distance
    # velocity is change in proximity w.r.t. previous time-point
    velocity = np.vstack([0, np.diff(proximity, axis=0)])
    speed = np.abs(velocity)
    # direction is sign of the velocity
    direction = np.sign(velocity)
    direction[0] = direction[direction != 0][0]
    
    # circle positions
    c1x = np.expand_dims(circle_pos[0,:,0], axis=1)
    c1y = np.expand_dims(circle_pos[0,:,1], axis=1)
    c2x = np.expand_dims(circle_pos[1,:,0], axis=1)
    c2y = np.expand_dims(circle_pos[1,:,1], axis=1)
    ########################
    # Create DataFrame
    reg_df = pd.DataFrame(np.hstack([proximity, direction, speed,
                                     c1x, c1y, c2x, c2y]),
                          columns=BASECOLS)
    
    # split the dataframe into segments between shocks
    shock_split = np.split(reg_df, np.array(data['touches']) + 1)
    
    return shock_split

def subj_timing(subj, run):
    # load the data from the .dat (json) file into a dictionary    
    path = DATPATH.format(subj=subj, run=run)
    with open(path, 'r') as f:
        data = load(f)
#     print(data['BlockDur'])
#     print(data['ShockOffset'][0] - data['ShockOnset'][0])
    
    # we can think of the onset of stimulus chunk as being the block onset
    # and when the shock turns off
    onsets = np.sort(data['ShockOffset'] + data['BlockOnset'])
    
    # the stimulus chunk ends at block end and shock onset.
    offsets = np.sort(data['ShockOnset'] + data['BlockOffset'])
    
    # we'll stack them side by side
    timings = np.vstack([onsets, offsets]).T
    
    # and remove the block onset time, as our data starts at index at this time
    timings -= onsets[0]
    
    # split by block
    split_points = np.where(np.diff(timings[:, 1] < data['BlockDur'][0] + 1))[0] + 1
    block_split = np.split(timings, split_points, axis=0)
    
    # also include the final time point to identify the end of the run
    block_split.append(data['RunDur'] - onsets[0])
    return block_split

def subj_specific_near_misses(run_untimed, run_timing):

    near_miss_thresh = 0.75
    peak_times = []
    prox_values = []
    
    def get_prox_values(df, peak_window):
        idxs = np.where(peak_window == True)[0]
        window_idx = np.linspace(idxs[0], idxs[-1], 13).astype(int)
        prox_values = np.array(df.proximity.values[window_idx])
        return prox_values

    for block in range(2):
        block_untimed = run_untimed[block]        
        block_timing = run_timing[block]
#         print("untimed pieces:",len(block_untimed),"timed pieces:",len(block_timing))
        if len(block_untimed) != len(block_timing):
            continue
        for idx_piece, piece in enumerate(block_untimed):
            piece = piece.copy()
            # evenly divide the piece duration into the correct
            # number of time points to apply to the data
            start, stop = block_timing[idx_piece]
            piece['time'] = np.linspace(start, stop, num=piece.shape[0])
            near_miss_idxs, _ = find_peaks(piece.proximity.values)
            # filter near misses greater than near miss threshold
            near_miss_idxs = [peak for peak in near_miss_idxs if piece.proximity.values[peak]>near_miss_thresh]
            
            # create 15 second censor after shock offset
            shock_censor = None
            if idx_piece > 0:
                shock_censor = (piece.time.values >= start) &\
                                (piece.time.values <= start + 15)
            
            for idx in near_miss_idxs:
                peak_time = piece.time.values[idx]
                peak_window = (piece.time.values >= peak_time - 7.5) &\
                               (piece.time.values <= peak_time + 7.5)
                if shock_censor is None:
                    peak_times.append(peak_time)
                    prox_values.append(get_prox_values(piece, peak_window))
                elif np.dot(shock_censor, peak_window) == False: # no intersection
                    peak_times.append(peak_time)
                    prox_values.append(get_prox_values(piece, peak_window))
    
    prox_values = np.stack(prox_values, axis=-1)
#             for visual verification
#             fig = plt.figure()
#             plt.plot(piece.proximity.values)
#             peaks = [peak for peak in near_miss_idxs if piece.time.values[peak] in peak_times]
#             peak_prox = [piece.proximity.values[peak] for peak in peaks]
#             plt.plot(peaks, peak_prox, 'o')
    
    return peak_times, prox_values

if __name__ == '__main__':
    
    untimed_data = [[get_base_stimulus(run, block) for block in range(2)] for run in range(6)]
    subj = '001'
    run = 5

    # get data and timing
    run_untimed = untimed_data[run]    
    run_timing = subj_timing(subj, run)
    print(run_timing)
    # align data to timing
    run_data, prox_values = subj_specific_near_misses(run_untimed, run_timing)
    print(run_data)
    print(prox_values.shape)
    
