import matplotlib.pyplot as plt
import numpy as np
import nibabel as nb
from multiprocessing import Pool
from utils import *
from LagObj import CorrBatch
import copy
import json
import argparse
import os
import shutil

nthreads = 22  # number of threads for multiproc. set to None to run in series (for debugging)
force_rerun = True  # Set to true to force rerun cross corr, otherwise will load npy from previous run


def proc_batch_lag(obj):

    obj.set_lags()      # set lags and normalise signals
    obj.xcorr()         # run object crosscorr function

    return obj.corrs, obj.zscore


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('param_file', type=str)
    args = parser.parse_args()

    # load params.json file at path passed to argparse
    f = open(args.param_file)
    params = json.load(f)
    f.close()

    data = nb.load(params['data_path'])  # load functional timeseries
    mask = nb.load(params['mask_path'])  # load brain mask
    TR = data.header['pixdim'][4]  # get TR from header
    tseries = np.loadtxt(params['tseries_path'])  # load regressor timeseries

    # get timelag parameters to test [min,max,step] s
    minlag = params['lags'][0]
    maxlag = params['lags'][1]
    lagstep = params['lags'][2]
    lag_range = np.arange(minlag, maxlag + lagstep, lagstep)

    output_folder = '/'.join(params['data_path'].split('/')[:-1]) + '/timelag_outputs/'

    if os.path.exists(output_folder) and not force_rerun:

        print('Using existing ouput: ' + output_folder)
        print('Loading existing results (.npy) (set force_rerun=True to overwrite)')

        corr_lags_flat = np.load(output_folder + 'corr_lags_flat.npy')
        zscore_lags_flat = np.load(output_folder + 'zscore_lags_flat.npy')

    else:

        if os.path.exists(output_folder):
            print('Overwriting output ' + output_folder)
            shutil.rmtree(output_folder)

        print('Output path: ' + output_folder)

        os.mkdir(output_folder)
        shutil.copy(args.param_file, output_folder + 'params.json')

        # preprocessing
        filtered_data = filter_mri_data(data.get_fdata(), TR, data.header['pixdim'][1:4], t_hp_cutoff=params['temporal_filter_hp'], spatial_fwhm=params['spatial_filter_fwhm'])
        nb.save(nb.Nifti1Pair(filtered_data, data.affine, header=data.header), output_folder + 'filtered_data.nii.gz')
        tseries = preproc_tseries(tseries, convolve=params['convolve'], detrend=params['detrend_polynomial'], fs=params['tseries_fs'])
        np.savetxt(output_folder + 'processed_tseries.txt', tseries)

        # convert image timeseries into a vector timeseries (nvox,nT) containing only voxels within the mask
        batch = batch_image(filtered_data, mask.get_fdata())

        if params['FFT']:
            print('Running cross correlation with FFT')
        else:
            print('Running full cross correlation')

        if nthreads is None:  # run in series for debugging
            print('Running in series (set nthreads for multiproc)')
            output = np.array([proc_batch_lag(copy.deepcopy(CorrBatch(batch, TR, tseries,
                                                                              params['tseries_fs'], lag=lag,
                                                                              tseries_padding=params['tseries_padding'],
                                                                              use_fft=params['FFT'])))
                                       for lag in lag_range])
        else:
            print('Running in parallel with %i threads' % nthreads)
            with Pool(nthreads) as p:  # run in parallel
                output = np.array(p.map(proc_batch_lag, [copy.deepcopy(CorrBatch(batch, TR, tseries,
                                                                                         params['tseries_fs'], lag=lag,
                                                                                         tseries_padding=params[
                                                                                             'tseries_padding'],
                                                                                         use_fft=params['FFT']))
                                                                 for lag in lag_range]))

        corr_lags_flat = output[:,0,:]
        zscore_lags_flat = output[:,1,:]

        np.save(output_folder + 'corr_lags_flat.npy', corr_lags_flat)       # save timelag correlations
        np.save(output_folder + 'zscore_lags_flat.npy', zscore_lags_flat)   # save timelag zscores

    indmax_flat = np.argmax(zscore_lags_flat, axis=0)   # get index of rmax
    rmax_flat = get_max(corr_lags_flat, indmax_flat)    # get rmax value
    zmax_flat = get_max(zscore_lags_flat, indmax_flat)  # get zmax value
    peaklag_flat = minlag + lagstep * indmax_flat       # get lag for rmax

    corr_lags = rebuild_image(corr_lags_flat, mask.get_fdata())  # rebuild image volume from batch vector
    zscore_lags = rebuild_image(zscore_lags_flat, mask.get_fdata())

    indmax = rebuild_image(indmax_flat, mask.get_fdata())
    rmax = rebuild_image(rmax_flat, mask.get_fdata())
    zmax = rebuild_image(zmax_flat, mask.get_fdata())
    peaklag = rebuild_image(peaklag_flat, mask.get_fdata())

    inrange = (peaklag != minlag) * (peaklag != maxlag)
    zmax[~inrange] = np.nan
    rmax[~inrange] = np.nan
    peaklag[~inrange] = np.nan

    # save results
    hdr = data.header
    hdr['pixdim'][4] = lagstep
    hdr['dim'][4] = len(lag_range)
    nb.save(nb.Nifti1Pair(corr_lags.transpose(1, 2, 3, 0), data.affine, header=hdr), output_folder + 'full_r.nii.gz')
    nb.save(nb.Nifti1Pair(zscore_lags.transpose(1, 2, 3, 0), data.affine, header=hdr), output_folder + 'full_z.nii.gz')
    hdr2 = data.header
    hdr2['dim'][0] = 3
    hdr2['dim'][4] = 1
    nb.save(nb.Nifti1Pair(rmax, mask.affine, header=hdr2), output_folder + 'rmax.nii.gz')
    nb.save(nb.Nifti1Pair(zmax, mask.affine, header=hdr2), output_folder + 'zmax.nii.gz')
    nb.save(nb.Nifti1Pair(peaklag, mask.affine, header=hdr2), output_folder + 'lags.nii.gz')

    # plots
    plot_lightbox(rmax, np.arange(1, 16, 2), title='Max correlation', cmap_label='$r_{max}$', vrange=[0, 0.5])
    plt.savefig(output_folder + 'rmax.png')
    plot_lightbox(zmax, np.arange(1, 16, 2), title='Max z-score', cmap_label='$z_{max}$')
    plt.savefig(output_folder + 'zmax.png')
    plot_lightbox(peaklag, np.arange(1, 16, 2), title='Time lag', cmap_label='lag /s')
    plt.savefig(output_folder + 'lag.png')

    #plt.show()
