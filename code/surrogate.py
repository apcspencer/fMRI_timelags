import numpy as np
import nibabel as nb
from multiprocessing import Pool
import copy
import json
import argparse
import os
import sys
from utils import *
from LagObj import CorrBatch
import random
import shutil
from copy import deepcopy

# True if force rerun pre-processing (gaussian smoothing, high-pass filtering, polynomial detrending,...)
force_rerun_preproc = False  
# True if force rerun cross corr, otherwise loads npy from previous run
force_rerun_null    = False 
nthreads            = None

def proc_batch_lag(obj):
    obj.set_lags()      # set lags and normalise signals
    obj.xcorr()         # run object crosscorr function
    return obj.zscore

def get_contrast(param):
    """
    Gets the contrast of the experience
    Args:
        param (dict): contains the parameters needed for the analysis
    Returns:
        contrast (str): "bold", "dfmri200", "dfmri1000", "adc"
    """
    
    if "bold" in param["data_path"]:
        contrast = "BOLD"
    elif "dfmri200" in param["data_path"]:
        contrast = "b200-dfMRI"
    elif "dfmri1000" in param["data_path"]:
        contrast = "b1000-dfMRI"
    elif "adc" in param["data_path"]:
        contrast = "ADC"

    return contrast

def null_distr(surrogate_timeseries, fMRI_flat, TR, param):
    """
    Function that generates a null distribution from surrogate respiratory timeseries 
    vs fMRI timeseries. This is a non-parametric test to find the threshold of significance
    for voxelwise correlations at different timelags. 
    
    The respiratory timeseries is shifted to different timelags and the Pearson's correlation 
    between it and a random voxel from the functional scan is calculated. 
    The maximum correlation (rmax) among all timelags between this voxel and:
        - surrogate respiratory signals -> null distribution. 

    Args:
        surrogate_timeseries (np.ndarray): (nb_surrogates x len(timeseries))
        fMRI_flat            (np.ndarray): (x*y*z, t)
        TR                        (float): repetition time [s]
        param                      (dict): contains the parameters needed for the analysis

    Returns:
        null_distr                 (list): null distribution, (nb_surrogates x 1)  
    """

    lags = get_lags(param)
    fs   = param["tseries_fs"] # in [Hz]

    null_distr = []
    for i in range(param["nb_surr"]):
        # Choose a random voxel to correlation the physiological signal with
        random_vx_idx = random.randint(0, fMRI_flat.shape[0] - 1)
        random_vx     = fMRI_flat[random_vx_idx, :]
        random_vx     = random_vx[np.newaxis, :]

        if nthreads is None:  # run in series for debugging
            # Surrogate physio signal vs fMRI
            output_surr = np.array([proc_batch_lag(copy.deepcopy(CorrBatch(random_vx, TR, surrogate_timeseries[i, :],
                                                                              fs, lag=lag,
                                                                              tseries_padding=param['tseries_padding'],
                                                                              use_fft=param['FFT'])))
                                       for lag in lags])
        else:
            with Pool(nthreads) as p:  # run in parallel
                # Surrogate physio signal vs fMRI
                output_surr = np.array(p.map(proc_batch_lag, [copy.deepcopy(CorrBatch(random_vx, TR, surrogate_timeseries[i, :],
                                                                                         fs, lag=lag,
                                                                                         tseries_padding=param['tseries_padding'],
                                                                                         use_fft=param['FFT']))
                                                                 for lag in lags]))

        # Store the max correlation rmax among the time lags
        null_distr.append(max(output_surr))

    return null_distr

def run_surrogate(timelag_folder, param, surr_ref_subject, force_rerun_null):
    """
    Function that runs the generation of the null distribution for all subjects.

    Args:
        timelag_folder    (str): folder where to find the preprocessed respiratory time series
        param            (dict): contains the parameters needed for the analysis
        surr_ref_subject  (str): subject whose respiratory timeseries is used as reference for surrogate signals computation
        force_rerun_null (bool): if True, force rerun the null distribution computation

    Returns:
        null_distr_r_all (list): (nb_subjects x nb_surrogates) contains the null distribution generated from the surrogate signals 
                                 from each subject 
    """

    folder_exp = '/'.join(timelag_folder.split('/')[6:8])

    physio_tseries_surr = np.loadtxt(timelag_folder + "processed_tseries.txt")  
    # (nb_surrogates x len(respiratory_timeseries))
    surrogate_signals   = create_surrogate_signal(physio_tseries_surr, nb_surrogates=param["nb_surr"])

    subjects = deepcopy(param["subjects"])
    subjects.remove(surr_ref_subject)
    
    print("Reference subject for surrogate ", surr_ref_subject)
    # print("Test on ", subjects)
    null_distr_z_all = []
    for subject in subjects:
        subject_folder  = param['data_folder'] + "/" + subject + "/"
        if os.path.exists(subject_folder + folder_exp):
            output_folder   = subject_folder + folder_exp + param["output_folder"]
            path_null_distr = output_folder + f"null_petco2.npy"

            # Calculate the null distribution
            if not os.path.exists(path_null_distr) or force_rerun_null:
                fMRI           = nb.load(output_folder + "filtered_data.nii.gz")  # load functional timeseries
                mask           = nb.load(param['data_folder'] + "/" + subject + "/" +  param['mask_path'])  # load brain mask
                TR             = fMRI.header['pixdim'][4]  # get TR from header

                fMRI = fMRI.get_fdata()
                # Remove timeseries filled with 0
                fMRI[np.all(fMRI == 0, axis=-1)] = np.nan

                # convert image timeseries into a vector timeseries (nvox,nT) containing only voxels within the mask
                batch = batch_image(fMRI, mask.get_fdata())
                null_distr_z = null_distr(surrogate_signals, batch, TR, param)

                np.save(path_null_distr, null_distr_z)

                null_distr_z_all.append(null_distr_z)
            else:
                null_distr_z_all.append(np.load(path_null_distr))

    null_distr_z_all = np.squeeze(np.array(null_distr_z_all).reshape((-1, 1))) 

    return null_distr_z_all

def preprocess(timelag_folder, subject_folder, param, param_path):
    """
    Preprocess the fMRI and physio time series
    Args:
        timelag_folder (str): folder where to find the preprocessed respiratory time series
        subject_folder (str): folder where subject data are stored
        param         (dict): contains the parameters needed for the analysis
        param_path     (str): path of the parameter file
    """

    if not os.path.exists(timelag_folder):
        os.mkdir(timelag_folder)
    
    print('Preprocessing the input into: ' + timelag_folder)
    shutil.copy(param_path, timelag_folder + 'params.json') 

    # preprocessing fMRI
    fMRI = nb.load(subject_folder + param['data_path'])
    TR   = fMRI.header['pixdim'][4]  # get TR from header
    filtered_fMRI = filter_mri_data(fMRI.get_fdata(), 
                                    TR, 
                                    fMRI.header['pixdim'][1:4], 
                                    t_hp_cutoff=param['temporal_filter_hp'], 
                                    spatial_fwhm=param['spatial_filter_fwhm'])
    nb.save(nb.Nifti1Pair(filtered_fMRI, fMRI.affine, header=fMRI.header), timelag_folder + 'filtered_data.nii.gz')

    # preprocessing physiological time series
    physio_tseries = np.loadtxt(subject_folder + param['tseries_path'])
    physio_tseries = preproc_tseries(physio_tseries, 
                                     convolve=param['convolve'], 
                                     detrend=param['detrend_polynomial'], 
                                     fs=param['tseries_fs'])
    np.savetxt(timelag_folder + 'processed_tseries.txt', physio_tseries)

def plot_distr(distr_null, ax, thresh, label, xlabel, contrast):
    """
    Plots the null distribution as histograms
    Args:
        distr_null (np.ndarray): null distribution of r o z scores
        ax    (matplotlib.axes): axis of the figure
        thresh          (float): significance threshold
        xlabel            (str): either "z score" or "rmax"
    """
    ax.hist(distr_null, bins=50, color='k', alpha=0.3)
    ax.axvline(x = thresh, color='r', label = label, linewidth=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.set_title(contrast)
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax.legend(frameon=True, framealpha=1)

def plot_and_save_distribution(distr, threshold, exp, contrast, outdir):
    """Generate and save the two plot variants."""
    legend = f"z > {threshold:.2f}"
    xlabel = "z score"

    os.makedirs(outdir, exist_ok=True)
    
    # two-panel figure
    _, ax = plt.subplots(1, 2, figsize=(11, 5))
    plot_distr(distr, ax[0], threshold, legend, xlabel, contrast)
    plt.savefig(f"{outdir}/surr_{exp}_{contrast}_both.png")
    plt.savefig(f"{outdir}/surr_{exp}_{contrast}_both.pdf")
    plt.close()

    # single-panel figure
    _, ax = plt.subplots(1, 1, figsize=(6, 5))
    plot_distr(distr, ax, threshold, legend, xlabel, contrast)
    plt.savefig(f"{outdir}/surr_{exp}_{contrast}.png")
    plt.savefig(f"{outdir}/surr_{exp}_{contrast}.pdf")
    plt.close()

def get_experiment_name(data_path):
    """
    Function that returns the experiment name
    Args:
        data_path (str): path to the input data
    """
    
    if "bh_run1" in data_path:
        return "BH_run1"
    elif "bh_run2" in data_path:
        return "BH_run2"
    elif "rs" in data_path:
        return "rs"
    return "unknown"

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('param_file', type=str)
    args   = parser.parse_args()

    # load params.json file at path passed to argparse
    f      = open(args.param_file)
    params = json.load(f)
    f.close()
    
    resp_folder = params[0]["data_folder"]

    # Each param contains the info about the data from which to calculate the null distr
    # One param corresponds to one contrast (BOLD, b200, b1000, ADC)
    for param in params:
        surrogate_folder = f"{resp_folder}/surrogate_fwhm_4"
        exp = get_experiment_name(param['data_path'])

        # Load any functional data to get the timeseries number of timepoints
        subject_1_path = param['data_folder'] + "/" + param['subjects'][0] + "/" + param['data_path']
        data           = nb.load(subject_1_path)  # load functional timeseries
        nb_timepoints  = data.header['dim'][4]

        
        # 1. Pre-processing the fMRI volumes & the physiological time series
        for subject in param["subjects"]:
            subject_folder = param['data_folder'] + "/" + subject + "/"
            folder_exp     = '/'.join(param['data_path'].split('/')[:-1])
            output_folder  = subject_folder + folder_exp + param["output_folder"]

            if not os.path.exists(output_folder) or force_rerun_preproc:
                preprocess(output_folder, subject_folder, param, args.param_file)

        # 2. Run the surrogate analysis for every params
        surr_subjects = []
        # If specified, calculate the surrogate signals only on the reference subject
        if len(param['surr_ref']) != 0:
            surr_subjects.append(param['surr_ref'])
        # Else, calculate them on all subjects
        else:
            surr_subjects = param["subjects"]
        
        surr_distr_null_pooled = []
        for surr_ref_subject in surr_subjects:
            subject_folder = param['data_folder'] + "/" + surr_ref_subject + "/"
            folder_exp     = '/'.join(param['data_path'].split('/')[:-1])
            output_folder  = subject_folder + folder_exp + param["output_folder"]

            null_distr_z_all = run_surrogate(output_folder, param, surr_ref_subject, force_rerun_null)
            surr_distr_null_pooled.append(null_distr_z_all)
            null_distr_z_all = np.sort(null_distr_z_all)

            alpha = 0.05
            upper_percentile_index = int((1 - alpha) * null_distr_z_all.shape[0])

            # Get the z-score thresholds
            z_upper_threshold = null_distr_z_all[upper_percentile_index]
            contrast = f"{param['data_path'].split('/')[1]}"

            if not os.path.exists(f"{surrogate_folder}/{exp.replace('BH', 'bh/BH')}/{contrast}/"):
                os.system(f"mkdir -p {surrogate_folder}/{exp.replace('BH', 'bh/BH')}/{contrast}/")

            fig, ax = plt.subplots(1, 1, figsize=(6, 5))
            xlabel = "z score"
            legend = f"z > {z_upper_threshold:.2f}"
            plot_distr(null_distr_z_all, ax, z_upper_threshold, legend, xlabel, contrast)
            plt.savefig(f"{surrogate_folder}/{exp.replace('BH', 'bh/BH')}/{contrast}/{surr_ref_subject}_{exp}_{contrast}_distr.png")
            plt.close()

        surr_distr_null_pooled = [item for row in surr_distr_null_pooled for item in row]

        distr_null_pooled_z = np.sort(surr_distr_null_pooled)

        alpha = 0.05
        upper_percentile_index = int((1 - alpha) * distr_null_pooled_z.shape[0])

        # Get the z-score thresholds
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        z_upper_threshold = distr_null_pooled_z[upper_percentile_index]
        contrast = f"{param['data_path'].split('/')[1]}"
        xlabel = "z score"
        legend = f"z > {z_upper_threshold:.2f}"
        plot_distr(distr_null_pooled_z, ax, z_upper_threshold, legend, xlabel, contrast)
        plt.savefig(f"{surrogate_folder}/{exp.replace('BH', 'bh/BH')}/{contrast}/surr_{exp}_{contrast}.png")
        plt.savefig(f"{surrogate_folder}/{exp.replace('BH', 'bh/BH')}/{contrast}/surr_{exp}_{contrast}.pdf")
        plt.close()

    # 3. If BH, pools the 2 BH null distributions together
    for param in params:
        if "bh_run1" in param['data_path']:
            contrast = get_contrast(param)
            
            # Load any functional data to get the timeseries number of timepoints
            subject_1_path = param['data_folder'] + "/" + param['subjects'][0] + "/" + param['data_path']
            data           = nb.load(subject_1_path)  # load functional timeseries
            nb_timepoints  = data.header['dim'][4]

            surr_distr_null_pooled = []
            # 1. Pre-processing the fMRI volumes & the physiological time series
            for subject in param["subjects"]:   
                surr_subjects = []
            # If specified, calculate the surrogate signals only on the reference subject
            if len(param['surr_ref']) != 0:
                surr_subjects.append(param['surr_ref'])
            # Else, calculate them on all subjects
            else:
                surr_subjects = param["subjects"]
            
            for surr_ref_subject in surr_subjects:
                subject_folder = param['data_folder'] + "/" + surr_ref_subject + "/"
                folder_exp     = '/'.join(param['data_path'].split('/')[:-1])

                for k in range(2):
                    output_folder  = subject_folder + folder_exp + param["output_folder"]
                    if os.path.exists(subject_folder + folder_exp):
                        null_distr_z_all = run_surrogate(output_folder, param, surr_ref_subject, False)
                        surr_distr_null_pooled.append(null_distr_z_all)

                    if "run1" in folder_exp:
                        folder_exp = folder_exp.replace("run1", "run2")
                    else:
                        folder_exp = folder_exp.replace("run2", "run1")
        
            surr_distr_null_pooled = [item for row in surr_distr_null_pooled for item in row]

            distr_null_pooled_z = np.sort(surr_distr_null_pooled)

            alpha = 0.05
            upper_percentile_index = int((1 - alpha) * distr_null_pooled_z.shape[0])

            # Get the z-score thresholds
            z_upper_threshold = distr_null_pooled_z[upper_percentile_index]
            contrast = f"{param['data_path'].split('/')[1]}"
            exp = "bh"

            fig, ax = plt.subplots(1, 1, figsize=(6, 5))
            xlabel = "z score"
            legend = f"z > {z_upper_threshold:.2f}"
            plot_distr(distr_null_pooled_z, ax, z_upper_threshold, legend, xlabel, contrast)
            plt.savefig(f"{surrogate_folder}/{exp}/surr_{exp}_{contrast}.png")
            plt.savefig(f"{surrogate_folder}/{exp}/surr_{exp}_{contrast}.pdf")
            plt.close()
