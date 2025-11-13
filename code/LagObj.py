import numpy as np
from utils import *
from scipy import signal


class CorrBatch:
    """
    Correlation Batch class for processing correlations between a batch of voxels and a timeseries at a specific lag.
    init args:
        data : 2D array [nvox,nT]; voxel timeseries data
        TR : float; repetition time of voxel timeseries /s
        tseries : 1D list [nt] regressor timeseries (original without lag)
        fs : float; sampling frequency of regressor timeseries /Hz
        lag : float; lag to apply to regressor timeseries /s (positive lag delays regressor wrt voxels)
        tseries_padding : str 'none' or 'zeros'; padding for lagged regressor timeseries
    """
    def __init__(self, data, TR, tseries, fs, lag=0, tseries_padding='none', use_fft=False):
        self.TR = TR
        self.nvox = data.shape[0]
        self.nT = data.shape[1]
        self.fs = fs
        self.lag = lag
        self.tseries_padding = tseries_padding
        self.use_fft = use_fft

        self.data = data
        self.tseries = tseries

        self.lagged_tseries = None
        self.corrs = None
        self.zscore = None
        self.indmax = None
        self.peaklag = None
        self.nTlag = None


    def set_lags(self):
        self.data, self.lagged_tseries = preproc_lags(self.data, self.tseries, self.TR, self.fs, self.lag, self.tseries_padding)
        self.nTlag = len(self.lagged_tseries)

    def xcorr(self):
        """
        Get correlation between voxel timeseries and regressor at the single lag specified for this instance of the
        object.
        """

        if self.use_fft:
            corr_method = 'fft'
        else:
            corr_method = 'direct'

        self.corrs = np.array([signal.correlate(self.data[i, :], self.lagged_tseries,
                               mode='valid', method=corr_method)[0] for i in range(self.nvox)])

        self.zscore = np.arctanh(self.corrs) * np.sqrt(self.nTlag - 3)