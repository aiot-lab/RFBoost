import os
import csv
import yaml
import numpy as np
import matplotlib.pyplot as plt
import time
# from torch.autograd.profiler import profile, record_function, ProfilerActivity
from sklearn.cluster import KMeans

# for UniTS only, copy from it
def read_data(args, config, part="train"):
    if part not in ["train", "test", "all", "train_valid", "valid"]:
        raise ValueError("part must be either train or test")

    path = os.path.join('../dataset', args.dataset)
    if part == "train":
        x_train = np.load(os.path.join(path, 'x_train.npy'))
        y_train = np.load(os.path.join(path, 'y_train.npy')).astype('int64').tolist()
    elif part == "test":
        x_test = np.load(os.path.join(path, 'x_test.npy'))
        y_test = np.load(os.path.join(path, 'y_test.npy')).astype('int64').tolist()
    elif part == "all":
        x_all = np.load(os.path.join(path, 'x_all.npy'))
        y_all = np.load(os.path.join(path, 'y_all.npy')).astype('int64').tolist()
    np.random.seed(args.seed)

    if args.exp == 'noise': # Robustness test (noise)
        if part == "train":
            for i in range(len(x_train)):
                for j in range(x_train.shape[2]):
                    noise = np.random.normal(1,1 , size= x_train[i][:, j].shape)
                    x_train[i][:, j] = x_train[i][:, j] + noise * args.ratio * np.mean(np.absolute(x_train[i][:, j] ))
        if part == "test":
            for i in range(len(x_test)):
                for j in range(x_test.shape[2]):
                    noise = np.random.normal(1, 1, size= x_test[i][:, j].shape)
                    x_test[i][:, j] = x_test[i][:, j] + noise * args.ratio * np.mean(np.absolute(x_test[i][:, j] ))

    elif args.exp == 'missing_data': # Robustness test (missing value)
        if part == "train":
            for i in range(len(x_train)):
                for j in range(x_train.shape[2]):
                    mask = np.random.random(x_train[i][:, j].shape) >= args.ratio
                    x_train[i][:, j] = x_train[i][:, j] * mask
        if part == "test":
            for i in range(len(x_test)):
                for j in range(x_test.shape[2]):
                    mask = np.random.random(x_test[i][:, j].shape) >= args.ratio
                    x_test[i][:, j] = x_test[i][:, j] * mask
    if part == "train":
        args.num_labels = max(y_train) + 1
        summary = [0 for i in range(args.num_labels)]
        for i in y_train:
            summary[i] += 1
        args.log("Label num cnt: "+ str(summary))
        args.log("Training size: " + str(len(y_train)))
        return list(x_train), y_train

    if part == "test":
        args.log("Testing size: " + str(len(y_test)))
        return list(x_test), y_test

    if part == "all":
        args.log("All size: " + str(len(y_all)))
        return list(x_all), y_all

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def read_config(path):
    return AttrDict(yaml.load(open(path, 'r'), Loader=yaml.FullLoader))

def logging(file):
    def write_log(s):
        print(s)
        with open(file, 'a') as f:
            f.write(s+'\n')
    return write_log

def set_up_logging(args, config):
    log = logging(os.path.join(args.log_path, args.model+'.txt'))
    for k, v in config.items():
        log("%s:\t%s\n" % (str(k), str(v)))
    return log

import scipy.stats as st

def compute_mean_and_conf_interval(accuracies, confidence=.95):
    accuracies = np.array(accuracies)
    n = len(accuracies)
    if n <= 1:
        return accuracies[0], -1
    # st.sem() computes the standard error of the mean
    m, se = np.mean(accuracies), st.sem(accuracies)
    # ppf = Percent point function of student's t distribution
    h = se * st.t.ppf((1 + confidence) / 2., n-1)
    return m, h
# retry with exception
def read_npz_data(path):
    for trial_i in range(10):
        try:
            data = np.load(path)
            return data["data"], data["ms"]
        except:
            print("attempt {}".format(trial_i))
            # sleep 1 ms
            time.sleep(0.001)
            continue
    print("problem reading {}".format(path))
    raise Exception("Failed to read data")

def read_any_data(path):
    if path.endswith(".npz"):
        return read_npz_data(path)
    elif path.endswith(".npy"):
        return np.load(path, allow_pickle=True)
    elif path.endswith(".mat"):
        try:
            import scipy.io as sio
            return sio.loadmat(path)
        except:
            import mat73
            return mat73.loadmat(path)
    else:
        raise Exception("Unsupported data format")
# generate doppler freq spectrum for each subcarrier
import numpy as np
import scipy.signal as signal
from sklearn.decomposition import PCA
import hashlib
import json

def stringfy_data(csi_data, window_size, window_step, agg_type):
    mean = np.mean(np.reshape(csi_data,(-1, 1)))
    var = np.var(np.reshape(csi_data,(-1, 1)))
    data_str = "{},{},{},{},{}".format(str(mean), str(var), str(window_size), str(window_step), agg_type)
    # if log == False:
    data_str += ",no_log"
    return data_str

# MATLAB code
# nFFT = 2^(nextpow2(length(y))+1);
# F = fft(y,nFFT);
# F = F.*conj(F);
# acf = ifft(F);
# acf = acf(1:(numLags+1)); % Retain nonnegative lags
# acf = real(acf);
# acf = acf/acf(1); % Normalize
# write python code below:
def nextpow2(i):
    return np.ceil(np.log2(i))
def autocorr(x):
    nFFT = int(2**(nextpow2(len(x))+1))
    F = np.fft.fft(x,nFFT)
    F = F*np.conj(F)
    acf = np.fft.ifft(F)
    acf = acf[0:len(x)] # Retain nonnegative lags
    acf = np.real(acf)
    acf = acf/acf[0] # Normalize
    return acf

def get_acf_(csi_data, channel_gain, samp_rate = 1000, window_size = 128, nfft=1000, window_step = 10, agg_type = 'pca', cache_folder=None):
    if agg_type == "pca":
        pca = PCA(n_components=1)
        pca_coef = pca.fit_transform(np.absolute(np.transpose(csi_data, [1,0])))
        # [T,1]
        csi_data_agg = np.dot(csi_data, pca_coef[:, 0])
    else:
        csi_data_agg = csi_data
    spectrogram = []
    # calculate autocorrelation function
    for i in range(0, csi_data_agg.shape[0] - window_size, window_step):
        # remove mean
        window = csi_data_agg[i:i+window_size] - np.mean(csi_data_agg[i:i+window_size])
        csi_agg_acf = autocorr(window)
        spectrogram.append(csi_agg_acf)
    # spectrogram = spectrogram / spectrogram.max()
    return None, None, spectrogram

def get_dfs_(csi_data, channel_gain, samp_rate = 1000, window_size = 256, nfft=1000, window_step = 10, agg_type = 'ms', n_pca=1, log=False, cache_folder=None):
    """
    :param csi_data: csi data in the form of a list of numpy arrays
    ms: channel gain of each subcarrier
    """
    # start_time = time.time()   
    data_str = stringfy_data(csi_data, window_size, window_step, agg_type)
    # print("dump_string: ", time.time() - start_time)

    # start_time = time.time()
    hash_data = hashlib.md5(data_str.encode("utf-8")).hexdigest()
    # print("hash: ", time.time() - start_time)
    
    try:
        # disable
        # assert cache_folder is None
        assert type(cache_folder) == str
        # if the file exists, load it
        data = np.load(cache_folder + hash_data + '.npz', allow_pickle=True)
        freq_bin = data['freq_bin']
        ticks = data['ticks']
        doppler_spectrum = data['doppler_spectrum']
    except:
        # if the file does not exist, compute it 
        # with record_function("compute_DFS"):
        half_rate = samp_rate / 2
        uppe_stop = 60
        freq_bins_unwrap = np.concatenate((np.arange(0, half_rate, 1) / samp_rate, np.arange(-half_rate, 0, 1) / samp_rate))
        freq_lpf_sele = np.logical_and(np.less_equal(freq_bins_unwrap,(uppe_stop / samp_rate)),np.greater_equal(freq_bins_unwrap,(-uppe_stop / samp_rate)))
        freq_lpf_positive_max = 60

        if agg_type == 'pca' and csi_data.shape[1] >= 1:
            pca = PCA(n_components=n_pca)
            pca_coef = pca.fit_transform(np.absolute(np.transpose(csi_data, [1,0])))
            # [T,1]
            csi_data_agg = np.dot(csi_data, pca_coef)
            # always report the last pca component
            csi_data_agg = csi_data_agg[:,-1]
        elif agg_type == 'ms':
            # L1-normalize ms
            csi_data_agg = csi_data
        
        # DC removal
        csi_data_agg = csi_data_agg - np.mean(csi_data_agg, axis=0)
        noverlap = window_size - window_step
        freq, ticks, freq_time_prof_allfreq = signal.stft(csi_data_agg, fs=samp_rate, nfft=samp_rate,
                                window=('gaussian', window_size), nperseg=window_size, noverlap=noverlap, 
                                return_onesided=False,
                                padded=True)
        
        freq_time_prof_allfreq = np.array(freq_time_prof_allfreq)
        freq_time_prof = freq_time_prof_allfreq[freq_lpf_sele, :]

        if log:
            doppler_spectrum = np.log10(np.square(np.abs(freq_time_prof)) + 1e-20) + 20
        else:
            # DO NOT USE widar3 version, will introduce interference in the frequency axis. making empty timeslots too large
            # doppler_spectrum = np.divide(abs(freq_time_prof), np.sum(abs(freq_time_prof), axis=0), out=np.zeros(freq_time_prof.shape), where=abs(freq_time_prof) != 0)
            # cal signal’s energy
            doppler_spectrum = np.square(np.abs(freq_time_prof))
        # doppler_spectrum = np.divide(abs(doppler_spectrum), np.sum(abs(doppler_spectrum), axis=0), out=np.zeros(doppler_spectrum.shape), where=abs(doppler_spectrum) != 0)
        # doppler_spectrum = freq_time_prof
        # freq_bin = 0:freq_lpf_positive_max - 1 * freq_lpf_negative_min:-1]
        freq_bin = np.array(freq)[freq_lpf_sele]
        
        # shift the doppler spectrum to the center of the frequency bins
        # freq_time_prof_allfreq = [0, 1, 2 ... -2, -1]
        doppler_spectrum = np.roll(doppler_spectrum, freq_lpf_positive_max, axis=0)
        freq_bin = np.roll(freq_bin, freq_lpf_positive_max)

        if cache_folder is not None and os.path.exists(cache_folder):
            try:
                np.savez(cache_folder + hash_data + '.npz', freq_bin=freq_bin, ticks=ticks, doppler_spectrum=doppler_spectrum)
            except:
                pass

    return freq_bin, ticks, doppler_spectrum

def get_dfs(csi_data, channel_gain, samp_rate = 1000, window_size = 256, nfft=1000, window_step = 10, agg_type="ms", n_pca=1, spec_type = 'dfs', log=False, cache_folder=None):
    # dfs: [F, W]
    # acf: [F, W]
    if spec_type == "dfs":
        spec = get_dfs_(csi_data, channel_gain, samp_rate, window_size, nfft, window_step, agg_type, n_pca, log, cache_folder)[2]
    elif spec_type == "acf":
        spec = get_acf_(csi_data, channel_gain, samp_rate, 128, nfft, window_step, agg_type, cache_folder)[2]
        spec = np.array(spec).transpose()
    elif spec_type == "acf+dfs":
        dfs = get_dfs_(csi_data, channel_gain, samp_rate, window_size, nfft, window_step, agg_type, n_pca, log, cache_folder)[2]
        acf = get_acf_(csi_data, channel_gain, samp_rate, 128, nfft, window_step, agg_type, cache_folder)[2]
    
        dfs = np.array(dfs)
        acf = np.array(acf).transpose()
        # padding acf dim 1
        if dfs.shape[1] > acf.shape[1]:
            acf = np.pad(acf, ((0,0),(0, dfs.shape[1] - acf.shape[1])), 'constant')
        # padding acf dim 0 to 132
        acf = np.pad(acf, ((0, 132 - acf.shape[0]),(0,0)), 'constant')
        spec = np.concatenate((dfs, acf), axis=0)
    return None, None, spec
    
    
def show_dfs(f, t, Zxx, ax):
    ax.pcolormesh(t, f, np.abs(Zxx), vmin = 0, vmax = 0.1, cmap = 'jet')
    ax.set_ylabel('Frequency [Hz]')
    ax.set_xlabel('Time [sec]')
    ax.grid(False)

def pad_data(data, pad_to_length):
    data_padded = np.zeros((np.array(data).shape[0], pad_to_length))
    data_padded[:, :np.array(data).shape[1]] = data
    return data_padded

def downsample_data(data, rate, target_length):
    l = len(data)
    data = data[:, ::rate]
    data_downsampled = data[:, :target_length]
    return data_downsampled

def pad_and_downsample(data, target_length, t=None, axis=1):
    # default axis = 1
    if axis == 0:
        data = np.transpose(data, [1,0])
    y_len = np.array(data).shape[1]
    # more left
    down_rate = y_len // target_length
    if down_rate == 0:
        down_rate = 1
    # fewer left
    up_rate = down_rate + 1
    
    down_cropped_loss =  int(np.ceil(y_len / down_rate)) - target_length
    up_pad_loss =  target_length - int(np.ceil(y_len / up_rate))

    if y_len > target_length:
        if down_cropped_loss <= up_pad_loss:
            out = downsample_data(data, down_rate, target_length)
            if t is not None:
                out_t = t[::down_rate]
                out_t = out_t[:target_length]
        else:
            out = downsample_data(data, up_rate, target_length)
            out = pad_data(out, target_length)
            if t is not None:
                out_t = t[::up_rate]
                avg_interval = (out_t[-1] - out_t[0]) / (len(out_t) - 1)
                # pad t to target_length
                pad_t = list(out_t[-1]+(np.arange(0, target_length - len(out_t), 1) + 1) * avg_interval)
                out_t = np.concatenate((out_t, pad_t))

    elif y_len < target_length:
        out = pad_data(data, target_length)
        if t is not None:
            avg_interval = (t[-1] - t[0]) / (len(t) - 1)
            # pad t to target_length
            pad_t = list(t[-1]+(np.arange(0, target_length - len(t), 1) + 1) * avg_interval)
            out_t = np.concatenate((t, pad_t))
    else:
        out = data
        if t is not None:
            out_t = t
    
    # transpose back
    if axis == 0:
        out = np.transpose(out, [1,0])

    if t is not None:
        if axis == 0:
            out_t = np.transpose(out_t, [1,0])
        return out, out_t
    else:
        return out

# SLNet
def complex_array_to_2_channel_float_array(data_complex):
    # data_complex(complex128/float64)=>data_float: [R,6,121,T_MAX]=>[2,R,6,121,T_MAX]
    data_complex = data_complex.astype('complex64')
    data_real = data_complex.real
    data_imag = data_complex.imag
    data_2_channel_float = np.stack((data_real, data_imag), axis=0)
    return data_2_channel_float