# Cross-correlations

Set parameters in the params.json file, as follows:
- data_path: path to fMRI data
- mask_path: path to brain mask
- tseries_path: path to timeseries for cross-correlation
- tseries_fs: sampling frequency of the timeseries (Hz)
- convolve: "hrf" or null. Function to convolve the timeseries before correlation.
- tseries_padding: "none", "ones", or "zeros". Padding for cross correlation.
- lags: [min,max,step]. Sets the range of timelags to test in the cross-correlation.
- FFT: 1 or 0. Whether to use FFT in the cross-correlation calculation.
- detrend_polynomial: null or an integer, to detrend nth order polynomials from the timeseries.
- temporal_filter_hp: null or float. Cutoff frequency for high-pass filtering (Hz).
- spatial_filter_fwhm: null or float. FWHM of gaussian spatial filtering for fMRI data.

`run python main.py params.json`
