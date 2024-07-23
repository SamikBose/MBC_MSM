##
# Description: A standalone code which can calculate the Merging Bias Corrected Counts Matrix from WE simulation data (in h5 format)
#
# Prerequisites: (i) Cluster labels of each snapshot of the WE data, 
#               (ii) The WE simulation data (currently only h5 supported)
#               (iii) WEPY software; latest,stable version.
#               (iv) Input file as the command line argument. (check the example inp file in the repo)
#
# Follow up: Use our package CSNAnalysis to build CSNs and calculate committors and mfpts. (https://github.com/SamikBose/Long_time_lagged_trans_mat/blob/9ba7de15ee465df402f7e45c0aaf765c9c6f38ee/build_correctedMSM_with_iterations_over_clust_adjusted.py#L474) 
#
# Comment: This code implements 'w2'-scheme only, which is more intuitive to capture the non-markovianity in the real systems. 


import numpy as np
import pickle as pkl
import sys
import os
import os.path as osp
from os.path import join
import time
import argparse

from wepy.hdf5 import WepyHDF5
from wepy.resampling.decisions.clone_merge import MultiCloneMergeDecision
from wepy.analysis.parents import (parent_panel,
                                   net_parent_table,
                                   resampling_panel)
from wepy.analysis.parents import sliding_window


# Reads the hdf5 file
def read_hdf5(hdf5_path):
    """
    Read the HDF5 file.

    Parameters:
    ----------
    hdf5_path: str
        Path to the HDF5 file.

    Returns:
    --------
    WepyHDF5: 
        An instance of the WepyHDF5 class for interacting with the HDF5 file, 
        or None if the file cannot be opened.
    """
    
    if not osp.exists(hdf5_path):
        return print('Unable to open: File does not exist...', hdf5_path, flush=True)

    if osp.exists(hdf5_path):
        print("Loading ",hdf5_path,"...", flush=True)

        try:
            wepy_h5 = WepyHDF5(hdf5_path,'r')
        except IOError:
            # Sometimes if the Python script exits prematurely, it does not reset the HDF5
            # file's flag. This manually resets it, allowing the file to be opened.
            os.system(f'h5clear -s {hdf5_path}')
            wepy_h5 = WepyHDF5(hdf5_path,'r')

        return wepy_h5

# Saving a pkl file 
def save_file(data, file_path):
    """
    Save data to a pickle file.

    Parameters:
    -----------
    data: Any kind
        Data to be saved.
    file_path: str
        Path to the file where data will be saved.
    """
    with open(file_path, 'wb') as f:
        pkl.dump(data, f)

# Reads the input file
def read_inputs(input_path):
    """
    Read the input file.

    Parameters:
    ----------
    input_path: str
    input_dict['C_obs_filename']    Path to the input file.

    Returns:
    --------
    input_dict: 
        A dictionary of inputs where,
        keys =  The keywords for the parameter
        values = The value/strings associated to the parameters
    """
 
    if not osp.exists(input_path):
        return print('Input file does not exist...', input_path, flush=True)
    
    else:
        input_dict = {}
        for line in open(input_path, 'r'):
            if line.find('#') >= 0: line = line.split('#')[0]
            #print(line)
            line = line.strip()
            if len(line) > 0:
                segments = line.split('=')
                #print(segments[0])
                input_param = segments[0].strip()
                try:    input_value = segments[1].strip()
                except: input_value = None
                if input_value:
                    
                    if input_param == 'n_clust':                    input_dict['n_cluster'] = int(input_value)
                    if input_param == 'lag_time':                   input_dict['lag'] = int(input_value)
                    if input_param == 'clusterlabels_pkl_path':     input_dict['cluster_path'] = str(input_value)
                    if input_param == 'h5_path':                    input_dict['h5_path'] = str(input_value)
                    if input_param == 'merge_fixed_wts_path':       input_dict['fixed_wts_path'] = str(input_value)
                    if input_param == 'we_wts_path':                input_dict['we_wts'] = str(input_value)
                    if input_param == 'M_path':                     input_dict['M_path'] = str(input_value)
                    if input_param == 'out_path':                   input_dict['out_path'] = str(input_value)
                    if input_param == 'MBC_file':                   input_dict['MBC_C_filename'] = str(input_value)
                    if input_param == 'Obs_file':                   input_dict['C_obs_filename'] = str(input_value)
    return input_dict

# Creating the out folder
def ensure_directory_exists(out_folder):
    """
    Ensure that the specified directory exists. Create it if it does not exist.
    
    Parameters:
    -----------
    out_folder: str 
        The path to the directory to be checked or created.
    """
    if os.path.exists(out_folder):
        print(f'Folder already exists: {out_folder}', flush=True)
    else:
        print(f'Creating folder: {out_folder}', flush=True)
        os.makedirs(out_folder, exist_ok=True)
        

# loading WE weights for a particular run
def load_weights(file_path, hdf5, run, n_walkers, n_cycles, reshape=True):
    """
    Load weighted ensemble weights from a pkl file or initialize them from the HDF5 data.

    Parameters:
    ----------
    file_path: str
        Path to the pkl file where weights are stored.
    hdf5: 
        HDF5 file object containing the weights data.
    run: int
        Run number in the HDF5 file.
    n_walkers: int
        Number of walkers.
    n_cycles: int
        Number of cycles.
    reshape: bool, default=True
        Whether to reshape the weights array. Default is True.

    Returns:
    --------
    weights: List of np.ndarray (n_walker, n_cycle)
        Weights matrix.
    """

    if osp.isfile(file_path):
        print("Loading WE weights...")
        return pkl.load(open(file_path, 'rb'))
    else:
        print("Building WE weights...")
        weights = [np.array(hdf5.h5[f'/runs/{run}/trajectories/{i}/weights']) for i in range(n_walkers)]

        if reshape:
            weights = np.reshape(weights, (n_walkers, n_cycles))

        save_file(weights, file_path)

        return weights

#### IMPORTANT:
# Load (or build) the merge_fixed weights of a WE run from the corresponding WE weights in a cycle specific manner:
# Each key in the returned dictionary is a cycle before which we use WE weights and 
# after which we create the merge_fixed weights
def load_fixed_weights(wt_fixed_path, WE_weights, cl_dict, n_walkers, n_cycles):
    """
    Load or build the merge_fixed weights dictionary.

    Parameters:
    -----------
    wt_fixed_path: str
        Path to the merge_fixed weights output file.
    WE_weights: list of np.ndarray
        Original WE weights of the particular run.
    cl_dict: dict
        Cloning dictionary.
    n_walkers: int
        Number of walkers.
    n_cycles: int
        Number of cycles.

    Returns:
    --------
    wt_dict_merge_fixed: dict
        Merge fixed weights dictionary.

        keys: indexed cycle
        value: (n_walkers, n_cycles) of fixed weights
    """

    if osp.isfile(wt_fixed_path):
        print('Loading the merge fixed weights...')
        wt_dict_merge_fixed = pkl.load(open(wt_fixed_path,'rb'))
    else:
        print('Building the merge fixed weights...')
        wt_dict_merge_fixed = {cyc: np.zeros((n_walkers, n_cycles)) for cyc in range(n_cycles)}

        for cyc in range(n_cycles):
            test_arr = np.zeros((n_walkers, n_cycles))
            # cycles prior to the current cycle: append weights from h5 directly
            test_arr[:, 0:cyc+1] = WE_weights[:,0:cyc+1]
            # move forward in cycle (we correct the wts from here on...)
            for cycle_idx in range(cyc, n_cycles-1):  # refer to the wepy flowchart for the range.
                current_cyc_wt = test_arr[:,cycle_idx]    # store the current cycle wts.
                cloning_record = cl_dict[cycle_idx]   # check the cloning record.
                next_cycle = cycle_idx+1
                test_arr[:, next_cycle] = current_cyc_wt  # pre-load the next cycle weights with current wts
                # check for cloning
                if cloning_record:
                    # check the format of the cloning_record; each key is a parent walker (clonee) and corresponding list
                    # are the walkers into which it got cloned.
                    # these are the walkers where the parent walker have cloned      
                    for cloning_parent, cloned_walker_list in cloning_record.items():
                        # distributed weights (add one to include the parent walkers as well)
                        split_wt = current_cyc_wt[cloning_parent]/(len(cloned_walker_list[0])+1)  # distributed weights
                        # update weight of the parent walker
                        test_arr[cloning_parent,next_cycle] = split_wt
                        # update weight of the squashed walkers
                        test_arr[cloned_walker_list, next_cycle] = split_wt     # update weight of the squashed walkers
            # for a particular 'key' cycles add the corresponding (n_walker, n_cycles) dimension of array
            wt_dict_merge_fixed[cyc] = test_arr
            # print('after loop test arr:', test_arr)
        save_file(wt_dict_merge_fixed, wt_fixed_path)

    return wt_dict_merge_fixed

# Create cloning information dictionary at each cycle 
def cloning_dict(resamp_pan):
    """
    Function to create a dictionary to store cloning information for each cycle.

    Parameters:
    -----------
    resampling_panel: list
        A nested list containing resampling panels.

    Returns
    -------
    cloning_dict: dictionary
        A dictionary where each cycle index maps to another dictionary
        that stores cloning information for each walker.
    """

    #initialization of the big dictionary with cycle idxs as keys
    n_cycs = len(resamp_pan)
    cloning_dict = {cycle_idx: {} for cycle_idx in range(n_cycs)}

    for i in range(n_cycs):
        res_pan = resamp_pan[i][0] #res panel at each cycle
        for walker, item in enumerate(res_pan): # Iterate over each walker
            tmp = []
            if item[1][0] == walker and item[0] == 2:
                tmp.append(item[1][1:])
            if len(tmp) > 0:
                cloning_dict[i][walker] = tmp
    return(cloning_dict)


# Calculate additive correction term to correct the merging bias
def get_deltas(Tmat,M,tau):
    """
    Compute the deltas (sum) matrix using the formula:
    
    sum(T_i * M_i for i in range(1, tau))
    
    Parameters:
    ----------
    Tmat: np.ndarray
        Transition matrix.
    M: dict of np.ndarray
        Dictionary of M matrices, one for each cycle.
    tau: int
        Lag time.

    Returns:
    --------
    deltas: np.ndarray
        Matrix.
    """
    n = Tmat.shape[0]
    deltas = np.zeros((n,n))
    Tpow = np.copy(Tmat)
    for t in range(1,tau+1):
        if t > 1:
            Tpow = np.matmul(Tmat,Tpow)
        deltas += np.matmul(Tpow,M[t])
    return deltas


# Building the counts matrices
def build_matrices(h5_path, wt_fixed_path_name, we_wts_path_name,
                    lag_time, n_clusters, cluster_labels, M_path, Calc_M):



    """
    Function to build the merging bias corrected counts matrices that can later be
    utilized to build CSNs for estimating committors and rates.

    Parameters:
    -----------
    h5_path: str
        Path to the h5 file containing WE simulation data
    
    wt_fixed_path_name: str
        Path to merge fixed weights file, without run and pkl extension.
    
    we_wts_path_name: str
        Path to WE weights file, without run and pkl extension.
    
    lag_time: int
        lag time of the markov model.

    n_clusters: int
        number of clusters in the markov model.
        
    cluster_labels: array/list
        state labels of all the snapshots in the WE simulation.

    M_path (Optional): str
        Path to the M file, if already exists.

    Calc_M: Boolean
        To calculate the M dictionaries or not to.

    Returns
    -------
    C: array
        Counts matrix with merging bias correction

    c_obs: array
        Counts matrix without merging bias correction but with the merging fixed weights.

    c_obs_we: array
        Counts matrix without merging bias correction and with WE wts.
    """

    wepy_h5 = read_hdf5(h5_path)
    window_length = lag_time + 1
    nothing_keep_merging = [1,4] 
    with wepy_h5:
        # determine the 1step counts  (T_i matrix that we have in the equation)
        c_obs_1step = np.zeros((n_clusters,n_clusters))
        c_obs = np.zeros((n_clusters,n_clusters))
        c_obs_we = np.zeros((n_clusters,n_clusters))

        n_runs = wepy_h5.num_runs
        
        if Calc_M == True:
            M = {t: np.zeros((n_clusters, n_clusters)) for t in range(1, window_length)}
        else:
            M = pkl.load(open(M_path, 'rb'))

        for run in range(n_runs):
            #print('...for run:', run)
            
            wt_fixed_path = f'{wt_fixed_path_name}_run{run}.pkl'
            wt_path = f'{we_wts_path_name}_run{run}.pkl'
            
            # normal things
            resampling_rec = wepy_h5.resampling_records([run])
            resamp_panel = resampling_panel(resampling_rec)
            par_panel = parent_panel(MultiCloneMergeDecision, resamp_panel)
            net_par_table = net_parent_table(par_panel)
            n_walkers = wepy_h5.num_run_trajs(run)
            n_cycles = wepy_h5.num_run_cycles(run)

            WE_weights = load_weights(wt_path,
                                        wepy_h5,
                                        run,
                                        n_walkers,
                                        n_cycles,
                                        reshape=True)


            cl_dict = cloning_dict(resamp_panel)
            wt_dict_merge_fixed = load_fixed_weights(wt_fixed_path,
                                        WE_weights,
                                        cl_dict,
                                        n_walkers,
                                        n_cycles)

            # start_build_C = time.time()
            # print('.. the C matrices...') 
            sw_1step = sliding_window(net_par_table , window_length=2)
            sw_old = sliding_window(net_par_table , window_length=window_length)

            # use the one step sliding windows
            for s in sw_1step:

                w1, c1 = s[0]
                w2, c2 = s[-1]
                # no inter run transitions
                x1 = int(cluster_labels[run][w1,c1])
                x2 = int(cluster_labels[run][w2,c2])
                # w2/w1 based-count
                wt2 = wt_dict_merge_fixed[c1][w2,c2]
                wt1 = wt_dict_merge_fixed[c1][w1,c1]

                c_obs_1step[x2, x1] += wt2

            # counts_mat and trans_mat with WE wts
            for s in sw_old:

                # standard procedures
                w1, c1 = s[0]
                w2, c2 = s[-1]
                # Ensure that these are state labels
                x1 = int(cluster_labels[run][w1,c1])
                x2 = int(cluster_labels[run][w2,c2])
                # w2/w1 based-count
                wt2 = wt_dict_merge_fixed[c1][w2,c2]
                wt1 = wt_dict_merge_fixed[c1][w1,c1]

                c_obs[x2, x1] += wt2

                we_wts2 = WE_weights[w2,c2]
                we_wts1 = WE_weights[w1,c1]
                c_obs_we[x2,x1] += we_wts2

            # print(f'...in time: {time.time() - start_build_C}')

            if Calc_M == True:
                start_build_M = time.time()
                # print('.. the M matrices...') 
                for init_cycle in range(n_cycles-lag_time): #check for all cycles that fall within the (final_cycle - lagtime) range
                    for init_walker in range(n_walkers):
                        # keep track of the walkers 
                        successor = {i + init_cycle: [] for i in range(lag_time)}
                        # initial state label at t_0
                        # j index according to the jupyter notebook
                        x1 = int(cluster_labels[run][init_walker, init_cycle])

                        successor[init_cycle-1] = [init_walker]
                        for i in range(lag_time):
                            index = lag_time - i     # i according to the jupyter notebook
                            cycle = i+init_cycle                # moving forward 
                            res_pan = resamp_panel[cycle][0]    # 

                            for j in successor[cycle-1]:
                                if res_pan[j][0] in nothing_keep_merging or res_pan[j][0] == 2:
                                    successor[cycle].append(j)
                                    if res_pan[j][0] == 2:
                                        successor[cycle].extend(res_pan[j][1])
                                if res_pan[j][0] == 3:
                                    # state at which the trajectory is in before being merged/squashed into another walker
                                    x2 = int(cluster_labels[run][j,cycle])
                                    wt2 = wt_dict_merge_fixed[init_cycle][j, cycle]
                                    #assert wt_dict_merge_fixed[init_cycle][init_walker,init_cycle] == we_wts[init_walker][init_cycle]
                                    #wt1 = we_wts[init_walker][init_cycle]
                                    wt1 = wt_dict_merge_fixed[init_cycle][init_walker, init_cycle]

                                    M[index][x2, x1] += wt2

        if Calc_M == True:
            save_file(M, M_path)

    #Calculate T_1step 
    T_1step = c_obs_1step/c_obs_1step.sum(axis=0)
    #Calculate count matrix
    C = c_obs + get_deltas(T_1step,M,lag_time)
    
    return C, c_obs, c_obs_we


# Main function
if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', dest='inpfile', help='Input parameter file', required=True)
    args = parser.parse_args()

    input_dict = read_inputs(args.inpfile)
    
    # Mandatory arguments
    n_clusters = input_dict['n_cluster'] 
    lag_time = input_dict['lag']
    clusterlabels_path = input_dict['cluster_path']
    h5_path = input_dict['h5_path']
    wt_fixed_path_name = input_dict['fixed_wts_path']
    we_wts_path_name = input_dict['we_wts']
    out_folder = input_dict['out_path']
    C_obs_filename = input_dict['C_obs_filename']
    MBC_C_filename = input_dict['MBC_C_filename']

    # Optional arguments
    M_path =  input_dict['M_path']
    ensure_directory_exists(out_folder)
    
    assert osp.exists(clusterlabels_path), f"{clusterlabels_path}, cluster labels file does not exist"
    assert osp.exists(h5_path), f"{h5_path}, h5 file does not exist"
    
    Calc_M = True
    if osp.exists(M_path):
        Calc_M=False

    MBC_C_path = osp.join(out_folder, MBC_C_filename)
    C_obs_path = osp.join(out_folder, C_obs_filename)

    cluster_labels = pkl.load(open(clusterlabels_path,'rb'))

    print("Building the merging corrected counts matrix...")
    start_build = time.time()
    C, c_obs, c_obs_we = build_matrices(h5_path, 
                                        wt_fixed_path_name,
                                        we_wts_path_name, 
                                        lag_time, 
                                        n_clusters, 
                                        cluster_labels,
                                        M_path,
                                        Calc_M)

    print(f'Done calculating matrices with: n_clusters:{n_clusters} and lagtime {lag_time}...')
    print(f'Time taken: {start_build - time.time()} seconds') 
    
    save_file(c_obs, C_obs_path)
    save_file(C, MBC_C_path)


## To Do list (Alex and Samik):
## 1. Avoid saving data in pkls: (i) Use the np.save binary or (ii) use compute observables in the wepy to attach the cluster labels to each h5
## 2. Create the h5 and cloning panel and pass them in the build matrices function instead of passing a bunch of paths.
## 3. wt_fixed_path_name,we_wts_path_name: Use a more rigid nomenclature, which will be coherent with the previous codes (i.e. clustering, calculating weights).
