import math
import numpy as np
from scipy import ndimage
from tqdm import tqdm
import sys
from bciloop_utilities.TimeFilters import ButterFilter
from bciloop_utilities.Hilbert import Hilbert
from bciloop_utilities.Entropy import ShannonEntropy
from bciloop_utilities.SpatialFilters import CommonSpatialPatterns, car_filter

# ProcEEGentropy class
class ProcEEGentropy:
    def __init__(self, WinLength, WinStep, NumBins, low_f, high_f, forder, srate):
        self.WinLength = math.floor(WinLength*srate)
        self.WinStep = math.floor(WinStep*srate)
        self.btfilt = [ButterFilter(forder[i], low_f=low_f[i], high_f=high_f[i], filter_type='bandpass', fs=srate) for i in range(len(low_f))]
        self.hilb = Hilbert()
        self.entropy = ShannonEntropy(NumBins)

    def apply(self, signal):
        NumSamples = np.shape(signal)[0]
        NumChans = np.shape(signal)[1]
        WinStart = np.arange(0, NumSamples-self.WinLength+1, self.WinStep)
        WinStop = WinStart + self.WinLength - 1
        NumWins = len(WinStart)
        numBands = len(self.btfilt)
        signal_entropy = np.empty([NumWins, NumChans, numBands])
        #print(signal[0:4,:])

        for i in range(numBands):
            for wid, wstart in enumerate(tqdm(WinStart, file=sys.stdout)):
                wsignal = signal[wstart:WinStop[wid], :]
                np.clip(wsignal, -400, 400, out=wsignal)
                # CAR filter
                wcar = car_filter(wsignal, axis=1)
                # Bandpass filter
                wfilt = self.btfilt[i].apply_filt(wcar)
                # Hilbert envelope
                self.hilb.apply(wfilt)
                wenv = self.hilb.get_envelope()
                # Shannon Entropy
                signal_entropy[wid, :, i] = self.entropy.apply(wenv)
        return signal_entropy

    # apply_offline(self, signal, mode, lap_path)
    def apply_offline(self, signal, *args):
        NumSamples = np.shape(signal)[0]
        NumChans = np.shape(signal)[1]
        WinStart = np.arange(0, NumSamples-self.WinLength+1, self.WinStep)
        WinStop = WinStart + self.WinLength - 1
        NumWins = len(WinStart)
        numBands = len(self.btfilt)
        signal_entropy = np.empty([NumWins, NumChans, numBands])
        car_flag = True
        laplacian = None

        if(len(args) == 1):
            #then we apply the lap filter
            car_flag = False
            lap_path = args[0]
            laplacian = np.load(lap_path)

        for i in range(numBands):
            for wid, wstart in enumerate(tqdm(WinStart, file=sys.stdout)):
                wsignal = signal[wstart:WinStop[wid], :]
                np.clip(wsignal, -400, 400, out=wsignal)

                if(car_flag):
                    # CAR filter
                    wfil = car_filter(wsignal, axis=1)
                else:
                    #Laplacian filter
                    wfil = ndimage.convolve(wsignal, laplacian, mode='constant', cval=0.0)

                # Bandpass filter
                wfilt = self.btfilt[i].apply_filt(wfil)
                # Hilbert envelope
                self.hilb.apply(wfilt)
                wenv = self.hilb.get_envelope()
                # Shannon Entropy
                signal_entropy[wid, :, i] = self.entropy.apply(wenv)
        return signal_entropy
