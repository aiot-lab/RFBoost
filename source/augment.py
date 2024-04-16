from ast import AugStore
from cProfile import label
from select import select
import numpy as np
from sklearn.cluster import KMeans
from utils import *
import random
import sys, time
import torch
class Augmentation(object):
    """
    Augmentation: output augmented spectrograms using registered RF data

    usage:
        # prepare processed RF data
        augmentation = Augmentation()

    """
    
    def __init__(self, default_stft_window=256, window_step=10):
        
        self.default_stft_window = default_stft_window
        self.kmeans_grp = dict()
        self.random_grp = dict()
        self.kmeans_top_idx = dict()
        self.window_step = window_step
        """
        time:
            window_size: 256(default)
            ratio: 2, 1, 0.5, 0.25, 0.125
        frequency:
            
        """
    def time_augment(self, data, ms, ratio, agg_type="ms", args = None):
        window_size = int(self.default_stft_window * ratio)
        f,t,augmented = get_dfs(data, ms, samp_rate=args.sample_rate, window_size=window_size, agg_type=agg_type, window_step=args.window_step, spec_type=args.loader, cache_folder=args.cache_folder)
        return f,t,augmented

    def frequency_augment(self, data, ms, method, param = None, args=None, fda_th=None, file_th=None, rx_th=None):
        # TODO set policies using parameters instead of "hard-coding"
        """
        data: dfs data (ts, sub_num)
        ms: motion statistics (sub_num, win_num)
        param: parameters
        args: global args
        fda_th: fda_th policy
        file_th: file_th file
        rx_th: rx_th RX
        """
        data = np.array(data)
        ms = np.array(ms)

        def index_map(index, shape):
            # shape: [N, A, Rx]
            # return N_i, A_i, Rx_i of index
            N, A, Rx = shape
            N_i = index // (A*Rx)
            A_i = (index % (A*Rx)) // Rx
            Rx_i = (index % (A*Rx)) % Rx

            return N_i, A_i, Rx_i

        f_list = []
        t_list = []
        augmented = []

        if method == "pca":
            f,t,spec = get_dfs(data, ms, samp_rate=args.sample_rate, window_size=args.default_stft_window, agg_type="pca", window_step=args.window_step, spec_type=args.loader, cache_folder=args.cache_folder)
            return f,t,spec
        elif method == "all":
            k_list = np.arange(data.shape[1])
            # k_list = np.argsort(-np.mean(ms,axis=1))
            k_idx = k_list[fda_th]

            f, t, spec = get_dfs(data[:, k_idx], ms[k_idx], samp_rate=args.sample_rate,
                                 window_size=args.default_stft_window, window_step=args.window_step, agg_type="ms",
                                 spec_type=args.loader, cache_folder=args.cache_folder)
            augmented = spec
            return f, t, augmented
        elif method == "subband-ms-top1":
            n_groups = param
            n_per_group = data.shape[1] // n_groups
            k_list = np.arange(0, data.shape[1], n_per_group)
            k_sel = k_list[fda_th]
            k_range = np.arange(k_sel, k_sel + n_per_group)

            k_idx = k_range[np.argsort(-np.mean(ms, axis=1)[k_range])[0]]

            f, t, spec = get_dfs(data[:, k_idx], ms[k_idx], samp_rate=args.sample_rate,
                                 window_size=self.default_stft_window, window_step=args.window_step, agg_type="ms",
                                 spec_type=args.loader, cache_folder=args.cache_folder)
            augmented = spec

            return f, t, augmented
        elif method == "ISS6TDAx3_FDA+kmeans6top2insubband3+motion-aware-random-shift-50-and-mask-guassion-75":
            n_groups = 6
            win_list = [0.5, 1, 2]
            n_win = len(win_list)
            if fda_th < 18:
                # TDAx3
                grp_th, win_th, fda_th = index_map(fda_th, (n_groups, n_win, 1))
                
                ratio = win_list[win_th]
                window_size = int(self.default_stft_window * ratio)
                
                n_per_group = data.shape[1] // n_groups
                k_list = np.arange(0, data.shape[1], n_per_group)
                k_sel = k_list[grp_th]
                
                k_range = np.arange(k_sel, k_sel+n_per_group)
                k_idx = k_range[np.argsort(-np.mean(ms,axis=1)[k_range])[0]]
            
                f,t,spec = get_dfs(data[:,k_idx], ms[k_idx], samp_rate=args.sample_rate, window_size=window_size, window_step=args.window_step ,agg_type="ms", spec_type=args.loader, cache_folder=args.cache_folder)
                augmented = spec
                return f,t,augmented
            elif fda_th < 36:
                # TDAx3 x MDA
                n_top = 2
                n_groups = 6
                fda_th -= 18
                grp_th, win_th, _ = index_map(fda_th, (n_groups, n_win, 1))
                        
                ratio = win_list[win_th]
                window_size = int(self.default_stft_window * ratio)
                
                n_per_group = data.shape[1] // n_groups
                k_list = np.arange(0, data.shape[1], n_per_group)
                k_sel = k_list[grp_th]
                
                k_range = np.arange(k_sel, k_sel+n_per_group)
                k_idx = k_range[np.argsort(-np.mean(ms,axis=1)[k_range])[0]]
            
                f,t,spec = get_dfs(data[:,k_idx], ms[k_idx], samp_rate=args.sample_rate, window_size=window_size, window_step=args.window_step ,agg_type="ms", spec_type=args.loader, cache_folder=args.cache_folder)

                ms_this = ms[k_idx,:]
                # if ms is high(above 50% time frame), use the mean value of the whole spectrogram to mask this time frame
                ms_quartile = np.quantile(ms_this, 0.5)
                ms_high_idx = ms_this > ms_quartile
                
                time_len = ms_this.shape[0]
                # find left and right boundary of 1 in ms_high_idx
                left_idx = np.where(ms_high_idx)[0][0]
                right_idx = np.where(ms_high_idx)[0][-1]
                
                random.seed(int(time.time()))
                # shift to maintain [left_idx, right_idx+1] in the spectrogram(no cross boundary)
                shift = random.randint(-left_idx, time_len - right_idx - 1)
                spec = np.roll(spec, shift, axis=1)
                return f,t,spec    
            elif fda_th < 54:
                fda_th -= 36
                grp_th, win_th, _ = index_map(fda_th, (n_groups, n_win, 1))
                            
                ratio = win_list[win_th]
                window_size = int(self.default_stft_window * ratio)
                
                n_per_group = data.shape[1] // n_groups
                k_list = np.arange(0, data.shape[1], n_per_group)
                k_sel = k_list[grp_th]
                
                k_range = np.arange(k_sel, k_sel+n_per_group)
                k_idx = k_range[np.argsort(-np.mean(ms,axis=1)[k_range])[0]]
            
                f,t,spec = get_dfs(data[:,k_idx], ms[k_idx], samp_rate=args.sample_rate, window_size=window_size, window_step=args.window_step ,agg_type="ms", spec_type=args.loader, cache_folder=args.cache_folder)
                
                ms_this = ms[k_idx,:]
                # if ms is high(above 75% time frame), use the noise to mask this time frame
                ms_quartile = np.quantile(ms_this, 0.75)
                ms_high_idx = ms_this > ms_quartile
                
                # set random seed using current time 
                np.random.seed(int(time.time()))
                # generate a random mask in time axis, with a length of 0.25-0.5 of the whole time axis
                random_start = np.random.randint(0, ms_this.shape[0] - ms_this.shape[0]//4)
                random_len = np.random.randint(0, ms_this.shape[0]//2)
                random_end = (random_start + random_len) if (random_start + random_len) < ms_this.shape[0] else ms_this.shape[0]
                random_range = np.zeros(ms_this.shape[0])
                random_range[random_start:random_end] = 1
                # intersection
                random_range = random_range * ms_high_idx
                noise_range = 1 - ms_high_idx
                # calculate noise level of noise range
                noise_mean = np.mean(spec[:,noise_range])
                noise_std = np.std(spec[:,noise_range])
                # generate noise
                noise = np.random.normal(noise_mean, noise_std, spec.shape)
                
                # mask the spectrogram with noise in random range
                for i in range(random_range.shape[0]):
                    if random_range[i] == 1:
                        spec[:,i] = noise[:,i]
                return f,t,spec

            elif fda_th < 60:
                # GSM
                n_ant = 3
                n_kernel = 6
                n_top = 2
      
                fda_th -= 54

                ms_list = []
                spectrograms = []
                _, ant_th, grp_th = index_map(fda_th, (1, n_ant, n_top))

                ratio = 1
                window_size = int(self.default_stft_window * ratio)
                
                # 0:[0-30] 1:[30-60] 2:[60-90]
                ant_range = np.arange(ant_th*30, (ant_th+1)*30)
                data_tmp = data[:,ant_range]
                ms_tmp = ms[ant_range,:]
                
                file_key = (file_th, rx_th, ant_th)
                
                if file_key in self.kmeans_grp:
                    labels = self.kmeans_grp[file_key]
                    top_idx = self.kmeans_top_idx[file_key]
                else:
                    kmeans = KMeans(n_clusters=n_kernel, random_state=0, n_init="auto").fit(np.transpose(np.abs(data_tmp), (1,0)))
                    labels = kmeans.labels_
                    ms_by_grp = []
                    for i in range(n_kernel):
                        ms_i = np.mean(ms_tmp[labels==i, :])
                        ms_by_grp.append(ms_i)
                    ms_by_grp = np.array(ms_by_grp)
                    # find n_top labels with highest ms
                    top_idx = np.argsort(-ms_by_grp)[:n_top]
                    self.kmeans_top_idx[file_key] = top_idx
                    self.kmeans_grp[file_key] = labels

                # data with labels==fda_th
                data_grp = data_tmp[:,labels==top_idx[grp_th]]
                cache_folder = args.cache_folder
                data_str = stringfy_data(data_grp, window_size, args.window_step, method+"v3" + str(file_key))
                hash_data = hashlib.md5(data_str.encode("utf-8")).hexdigest()
                try:
                    # if the file exists, load it
                    dfs_cached = np.load(cache_folder + hash_data + '.npz', allow_pickle=True)
                    freq_bin = dfs_cached['freq_bin']
                    ticks = dfs_cached['ticks']
                    doppler_spectrum = dfs_cached['doppler_spectrum']
            
                    f_list = freq_bin
                    t_list = ticks
                    augmented = doppler_spectrum
                except:
                    # calculate spec for subcarriers in i-th kernel 
                    for sub_idx in range(data_tmp.shape[1]):
                        if labels[sub_idx] == top_idx[grp_th]:
                            f,t,spec = get_dfs(data_tmp[:,sub_idx], ms_tmp[sub_idx], samp_rate=args.sample_rate, window_size=window_size, window_step=args.window_step ,agg_type="ms", spec_type=args.loader, cache_folder=args.cache_folder)
                            f_list.append(f)
                            t_list.append(t)
                            ms_this = ms_tmp[sub_idx,:]    
                            # give it a small number to avoid zeros/negative values in the spec
                            ms_this[ms_this<=0] = 1e-6
                            try:
                                ms_list.append(ms_this)
                            except:
                                pass
                            spectrograms.append(spec)
                    spectrograms = np.array(spectrograms)
                    ms_list = np.array(ms_list)
                    augmented = np.zeros(spectrograms.shape[1:])
                    # ms_list /= np.sum(ms_list, axis=1, keepdims=True)
                    ms_weights = None
                    for i in range(spectrograms.shape[2]):
                        # aggregated spec by weighted sum along time axis
                        if i < ms_list.shape[1]:
                            ms_weights = ms_list[:,i] / np.sum(ms_list[:,i]) 
                            # else: use last weight
                        
                        augmented[:,i] = np.dot(ms_weights, spectrograms[:,:,i])
                        
                    # np.savez(cache_folder + hash_data + '.npz', freq_bin=f, ticks=t, doppler_spectrum=augmented)
                return f_list,t_list,augmented
        elif method.startswith("all+"):
            method = method.replace("all+", "")
            k_list = np.arange(data.shape[1])
            k_idx = fda_th % 90

            f, t, spec = get_dfs(data[:, k_idx], ms[k_idx], samp_rate=args.sample_rate,
                                 window_size=args.default_stft_window, window_step=args.window_step, agg_type="ms",
                                 spec_type=args.loader, cache_folder=args.cache_folder)

            if fda_th < 90:
                augmented = spec
                return f, t, augmented
            else:
                fda_th -= 90
                if method == "time-wrap5-mask-time-20-mask-freq-20":
                    # #Export
                    def time_warp(spec, W=50):
                        from SparseImageWarp import sparse_image_warp
                        # (freq, time)
                        num_rows = spec.shape[1]
                        spec_len = spec.shape[2]
                        device = spec.device

                        # adapted from https://github.com/DemisEom/SpecAugment/
                        pt = (num_rows - 2 * W) * torch.rand([1], dtype=torch.float) + W  # random point along the time axis
                        src_ctr_pt_freq = torch.arange(0, spec_len // 2)  # control points on freq-axis
                        src_ctr_pt_time = torch.ones_like(src_ctr_pt_freq) * pt  # control points on time-axis
                        src_ctr_pts = torch.stack((src_ctr_pt_freq, src_ctr_pt_time), dim=-1)
                        src_ctr_pts = src_ctr_pts.float().to(device)

                        # Destination
                        w = 2 * W * torch.rand([1], dtype=torch.float) - W  # distance
                        dest_ctr_pt_freq = src_ctr_pt_freq
                        dest_ctr_pt_time = src_ctr_pt_time + w
                        dest_ctr_pts = torch.stack((dest_ctr_pt_freq, dest_ctr_pt_time), dim=-1)
                        dest_ctr_pts = dest_ctr_pts.float().to(device)

                        # warp
                        source_control_point_locations = torch.unsqueeze(src_ctr_pts, 0)  # (1, v//2, 2)
                        dest_control_point_locations = torch.unsqueeze(dest_ctr_pts, 0)  # (1, v//2, 2)
                        warped_spectro, dense_flows = sparse_image_warp(spec, source_control_point_locations,
                                                                        dest_control_point_locations)
                        return warped_spectro.squeeze(3)

                    def freq_mask(spec, F=20, num_masks=1, replace_with_zero=False):
                        cloned = spec.clone()
                        num_mel_channels = cloned.shape[1]

                        for i in range(0, num_masks):
                            f = random.randrange(0, F)
                            f_zero = random.randrange(0, num_mel_channels - f)

                            # avoids randrange error if values are equal and range is empty
                            if (f_zero == f_zero + f): return cloned

                            mask_end = random.randrange(f_zero, f_zero + f)
                            if (replace_with_zero):
                                cloned[0][f_zero:mask_end] = 0
                            else:
                                cloned[0][f_zero:mask_end] = cloned.mean()

                        return cloned

                    def time_mask(spec, T=10, num_masks=1, replace_with_zero=False):
                        cloned = spec.clone()
                        len_spectro = cloned.shape[2]

                        for i in range(0, num_masks):
                            t = random.randrange(0, T)
                            t_zero = random.randrange(0, len_spectro - t)

                            # avoids randrange error if values are equal and range is empty
                            if (t_zero == t_zero + t): return cloned

                            mask_end = random.randrange(t_zero, t_zero + t)
                            if (replace_with_zero):
                                cloned[0][:, t_zero:mask_end] = 0
                            else:
                                cloned[0][:, t_zero:mask_end] = cloned.mean()
                        return cloned

                    try:
                        spec_warped = time_mask(freq_mask(time_warp(torch.tensor(spec).unsqueeze(0), W=5), F=20, num_masks=2),
                                                T=20, num_masks=2)
                        spec_warped = spec_warped.squeeze(0)
                    except:
                        spec_warped = spec

                    return f, t, spec_warped
        return f_list,t_list,augmented

    def space_augment(self, data, ms, method, args=None):
        """
        data: (ts, sub_num)
        args: (sub_num)
        """
        return None, None, None
