import numpy as np
from scipy.fft import fft
import pywt

def basic_features(signal):
    rms = np.sqrt(np.mean(signal**2))
    peak = np.max(np.abs(signal))
    crest = peak / rms

    spectrum = np.abs(fft(signal))
    thd = np.sqrt(np.sum(spectrum[2:]**2)) / spectrum[1]

    return [rms, peak, crest, thd]

def wavelet_features(signal, wavelet='db4', level=4):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    features = []
    for c in coeffs:
        features.extend([
            np.mean(c),
            np.std(c),
            np.max(c),
            np.min(c)
        ])
    return features

def extract_features(signal):
    return basic_features(signal) + wavelet_features(signal)

def build_feature_matrix(X):
    return np.array([extract_features(x) for x in X])