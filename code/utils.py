import numpy as np
import matplotlib.pyplot as plt
from scipy import special, ndimage, signal


def get_max(arr, inds):
    """
    Get values which correspond to the peak correlations identified by inds.
    Args:
        arr : 2D array [N,n]; Batched data for N time lags (e.g. r, pvals)
        inds : 1D list [n]; Each element contains the index (of axis 0 in arr) which gives the peak
                                        correlation
    Returns:
        1D list [n]; Time-shifted image data, where each voxel's time lag corresponds to the peak
                                    correlation for that voxel
    """
    amax = np.zeros_like(inds).astype(float)
    for i in range(len(inds)):
        amax[i] = arr[inds[i], i]

    return amax


def batch_image(data, mask):
    """
    Convert 4D image timeseries to 2D vector timeseries containing only voxels within the mask
    TODO support selection of number of batches
    Args:
        data : 4D array [xdim,ydim,zdim,nT]; Functional timeseries as a 4D image volume
        mask : 3D array [xdim,ydim,zdim]; Brain mask as a 3D volume
    Returns:
        2D array [nvox,nT]; Vector timeseries containing only brain mask voxels
    """
    inds = np.where(mask != 0)
    batch = []
    for i in range(data.shape[3]):
        im = data[:, :, :, i]
        batch.append(im[inds])

    return np.array(batch).T


def rebuild_image(flat_im, mask):
    """
    Reverse of batch_image. Uses the brain mask volume to repopulate the image volume from batch vector data
    Args:
        flat_im : 1D list or 2D array [(N),nvox]; Batch data (optionally for N time lags)
        mask : 3D array [xdim,ydim,zdim]; Brain mask as a 3D volume
    Returns:
        3D or 4D array [(N),xdim,ydim,zdim]; Image volumes (optionally for N time lags)
    """
    inds = np.where(mask != 0)

    if len(flat_im.shape) == 2:
        im = np.nan*np.ones([len(flat_im), mask.shape[0], mask.shape[1], mask.shape[2]])
        for ii, ind in enumerate(np.array(inds).T):
            im[:, ind[0], ind[1], ind[2]] = flat_im[:, ii]
    elif len(flat_im.shape) == 1:
        im = np.nan*np.ones(mask.shape)
        for ii, ind in enumerate(np.array(inds).T):
            im[ind[0], ind[1], ind[2]] = flat_im[ii]
    else:
        raise('dimensionality not supported')

    return im


def normalise(x):
    """
    Normalise vector, or each timeseries in an array. For a 1D vector, subtract mean and divide by 2-norm. For a 2D
    array, do this for each row separately.
    Args:
        x : 1D [N] or 2D array [nvox,nT]; vector or timeseries array to be normalised.
    Returns:
        1D [N] or 2D array [nvox,nT]; normalised.
    """
    if len(x.shape) > 1:
        x = x - np.mean(x, axis=-1)[:, None]
        x = x / np.linalg.norm(x, axis=-1)[:, None]
    else:
        x = x - np.mean(x)
        x = x / np.linalg.norm(x)
    return x


def preproc_tseries(tseries, convolve=None, detrend=None, fs=None):
    """
    Preprocessing for timeseries. Detrending up to nth order polynomial followed by convolution with response function.
    Args:
        tseries : 1D [N]; timeseries array
        convolve : str; response to convolve tseries with, or None for no convolution
        detrend : int; polynomial order to detrend up to and including, or None for no detrending
        fs : float; sampling frequency of tseries
    Returns:
        1D [N]; detrended/convolved timeseries.
    """
    if detrend is not None:
        t = np.arange(0, len(tseries)/fs, 1/fs)
        model = np.polyfit(t, tseries, detrend)
        tseries = tseries - np.polyval(model, t)

    if convolve is not None:
        tseries = convolve_tseries(tseries, convolve, fs)

    return tseries


def preproc_lags(data, tseries, TR, fs, lag, tseries_padding):
    """
    Set lags for data and timeseries to run correlation at a given lag.
    Args:
        data : 4D array [xdim,ydim,zdim,nT]; Functional timeseries as a 4D image volume
        tseries : 1D [N]; timeseries array
        TR : float; timestep of data
        fs : float; sampling frequency of tseries
        lag : float; time lag in seconds
        tseries_padding : str; padding for lag. If none, will not pad but crop if needed
    Returns:
        4D array [xdim,ydim,zdim,nt]; Lagged image volume time series
        1D [nt]; lagged timeseries array
    """
    data_dur = data.shape[1] * TR       # duration of data /s
    tseries_dur = len(tseries) / fs     # duration of tseries /s
    if lag == 0:
        # interpolate tseries to TR
        lagged_tseries = np.interp(np.arange(0, (data.shape[1] * TR), TR),
                                   np.arange(0, len(tseries) / fs, 1 / fs), tseries)
        cropped_data = data
    else:
        if lag < 0:
            # negative lag; shift tseries back in time
            if tseries_dur > data_dur + lag:
                # tseries is long enough to avoid padding, interpolate directly to TR
                lagged_tseries = np.interp(np.arange(np.abs(lag), np.abs(lag) + (data.shape[1] * TR), TR),
                                           np.arange(0, len(tseries) / fs, 1 / fs), tseries)
                cropped_data = data
            else:
                # tseries must be cropped and padded if necessary
                lagged_tseries = np.interp(np.arange(np.abs(lag), tseries_dur - np.abs(lag), TR),
                                           np.arange(0, len(tseries) / fs, 1 / fs), tseries)
                if tseries_padding == 'zeros':
                    # zero padding on the end of lagged tseries
                    lagged_tseries = np.concatenate([lagged_tseries, np.zeros(int(np.round(np.abs(lag)/TR)))], axis=0)
                    cropped_data = data
                elif tseries_padding == 'none':
                    # no padding; crop data to match lagged tseries
                    cropped_data = data[:, :len(lagged_tseries)]
                else:
                    raise('unsupported padding')

        else:
            # positive lag; interpolate to TR with lag subtracted tseries always starts synced with image volume, so
            # positive lag will always require cropping or padding
            lagged_tseries = np.interp(np.arange(0, data_dur - lag, TR),
                                       np.arange(0, len(tseries) / fs, 1 / fs), tseries)
            if tseries_padding == 'zeros':
                # zero padding at the start of lagged tseries
                lagged_tseries = np.concatenate([np.zeros(int(np.round(lag/TR))), lagged_tseries], axis=0)
                cropped_data = data
            elif tseries_padding == 'none':
                # no padding; crop data to match lagged tseries
                cropped_data = data[:, -len(lagged_tseries):]
            else:
                raise('unsupported padding')

    # normalise vectors so that dot product = pearson correlation
    lagged_tseries = normalise(lagged_tseries)
    cropped_data = normalise(cropped_data)

    return cropped_data, lagged_tseries


def filter_mri_data(data, TR, pixdims, t_hp_cutoff=None, spatial_fwhm=None):
    """
    Preprocessing for MRI data, including spatial filtering with Gaussian kernel followed by highpass temporal
    filtering with 4th oreder butterworth filter.
    Args:
        data : 4D array [xdim,ydim,zdim,nT]; Functional timeseries as a 4D image volume
        TR : int; timestep of data
        pixdims : list [3]; voxel dimensions in mm
        t_hp_cutoff : float; cutoff frequency for highpass filtering
        spatial_fwhm : float; FWHM of Gaussian kernel for spatial filtering
    Returns:
        4D array [xdim,ydim,zdim,nT]; Filtered image volume time series
    """
    if spatial_fwhm is not None:
        # spatial filtering
        sigma = spatial_fwhm/np.sqrt(8*np.log(2))
        filtered = ndimage.gaussian_filter(data, sigma/pixdims, axes=[0,1,2])
    else:
        filtered = data
    if t_hp_cutoff is not None:
        # temporal filtering
        sos = signal.butter(4, t_hp_cutoff, fs=1/TR, output='sos', btype='highpass')
        filtered = signal.sosfiltfilt(sos, filtered)

    return filtered


def gamma_diff(t, a1, a2, l1, l2, A):

    f1 = ((l1 ** a1) / special.gamma(a1)) * t ** (a1 - 1) * np.exp(-l1 * t)
    f2 = ((l2 ** a2) / special.gamma(a2)) * t ** (a2 - 1) * np.exp(-l2 * t)

    return f1 - f2 / A


def convolve_tseries(tseries, conv_func, fs):
    
    nt = len(tseries)
    t = np.arange(0,30, 1 / fs)
    if conv_func == 'hrf':
        response = gamma_diff(t, 6., 16., 1., 1., 6.)
    else:
        raise('unsupposed convolution')

    convolved_tseries = np.convolve(tseries, response)[:nt]

    return convolved_tseries


def plot_lightbox(volume, slices, title=None, cmap_label=None, vrange=[None, None]):
    """
    Plot axial lightbox of image volume for selected slices.
    Args:
        volume : 3D array [xdim,ydim,zdim]; Image volume
        slices : 1D list [Nsl]; List of slices to plot
        title : str; String for figure title
        cmap_label : str; String for colour bar label
        vrange : 1D list [2,]; vmin and vmax values for colour bar
    """
    Nsl = len(slices)
    nrows = np.ceil(np.sqrt(Nsl)).astype(int)
    ncols = np.ceil(Nsl / nrows).astype(int)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 2))

    for i, ax in enumerate(axes.reshape(-1)):
        if i < Nsl:
            im = ax.imshow(volume[:, :, slices[i]].T, cmap='hot', vmin=vrange[0], vmax=vrange[1], origin='lower')
            ax.set_title('z = %i' % slices[i])
            ax.axes.get_xaxis().set_ticks([])
            ax.axes.get_yaxis().set_ticks([])
        else:
            ax.axis('off')

    fig.colorbar(im, ax=axes[np.unravel_index(Nsl - 1, [nrows, ncols])], label=cmap_label)
    plt.suptitle(title)
    fig.tight_layout()
