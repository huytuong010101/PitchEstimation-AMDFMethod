from scipy.io.wavfile import read
from tqdm import tqdm
import numpy as np
from scipy.ndimage import gaussian_filter1d
from matplotlib import pyplot as plt
from utils import get_label_of_time
from time import time


class PitchEstimation:
    def __init__(self, frame_time: float = 0.05, fre_range: tuple = (70, 450), overlap_frame=0.3):
        self.overlap_frame = overlap_frame
        self.frame_time = frame_time
        self.fre_range = fre_range
        self.fs, self.signal = None, None
        self.fs_predicts = None
        self.fs_predicts_median = None
        self.avg = {
            "sil": [],
            "uv": [],
            "v": []
        }

    def amf(self, median_size=None, threshold: float = 0.5, label=None):
        """
        This function use to predict pitch of an audio using AMDF algorithm
        params:
            median_size: size of median filter
            threshold: amdf less than threshold will be ignore
            label (use to debug): time of silent, unvoice, voice and mean, std of f0
        return:
            fs_predicts: pitchs which estimate by AMFD (by sample)
            unvoice_amdf: an amdf of unvoice frame
            unvoice_frame: an audio signal of unvoice
            voice_amdf: an amdf of voice frame
            voice_frame: an audio signal of voice frame
        """
        '''
            self.avg = {
                "sil": [],
                "uv": [],
                "v": []
            }
        '''
        unvoice_frame = None
        unvoice_amdf = None
        voice_frame = None
        voice_amdf = None
        # Calculate frame size
        frame_size = int(self.frame_time * self.fs)  # time of frame / T
        # Limit Frequency from min frequency -> max frequency
        max_sample_period = int((1 / self.fre_range[0])/(1 / self.fs))
        min_sample_period = int((1 / self.fre_range[-1])/(1 / self.fs))
        # Init array to save result
        frames_amf = np.full((len(self.signal), max_sample_period), np.inf)
        self.fs_predicts = np.zeros(self.signal.shape)
        self.fs_predicts_median = np.zeros(self.signal.shape)
        # Loop all center of frame
        start = frame_size // 2
        stop = len(self.signal) - frame_size // 2
        step = int((1 - self.overlap_frame) * frame_size)
        for center_index in tqdm(range(start, stop, step)):
            # center_index is center of frame
            # Get frame
            frame = self.signal[center_index - frame_size // 2:center_index + frame_size // 2]
            for n in range(min_sample_period, max_sample_period):
                # Init value for amdf of n
                frames_amf[center_index][n] = 0
                # Calculate AMDF
                amdf_value = frame[:-n] - frame[n:]
                frames_amf[center_index][n] = np.sum(np.abs(amdf_value))
                frames_amf[center_index][n] /= frame_size - n
            amdf_frame = frames_amf[center_index][min_sample_period: max_sample_period]
            # nomalize by divided to maximum
            max_amdf = np.max(amdf_frame)
            if max_amdf != 0:
                amdf_frame /= max_amdf
            minimum_index, count_peak = self.get_minimum(amdf_frame)
            n_min = minimum_index + min_sample_period
            fs_predict = 1 / ((1 / self.fs) * n_min)
            # Filter if non-voice or silent
            if n_min == min_sample_period \
                    or n_min == max_sample_period \
                    or amdf_frame[n_min - min_sample_period] > threshold\
                    or count_peak > 35:
                fs_predict = 0

            # return frame and amfd once time
            if fs_predict == 0 and unvoice_amdf is None and get_label_of_time(label, center_index / self.fs - 0.01) == "uv":
                unvoice_frame = frame.copy()
                unvoice_amdf = amdf_frame.copy()
            if fs_predict != 0 and voice_amdf is None and get_label_of_time(label, center_index / self.fs - 0.01) == "v":
                voice_frame = frame.copy()
                voice_amdf = amdf_frame.copy()
                # Calc avg
            '''
            l = get_label_of_time(label, center_index / self.fs)
            if l is not None:
                self.avg[l].append(amdf_frame[n_min - min_sample_period])
            '''
            # Assign fs for current frame
            self.fs_predicts[center_index] = fs_predict
            self.fs_predicts_median[center_index] = fs_predict
            # Median filter
            if median_size is not None:
                frame = self.fs_predicts[center_index - 2 * step * (median_size // 2): center_index + 1: step]
                if len(frame) != 0:
                    self.fs_predicts_median[center_index - step * (median_size // 2)] = int(self.median_of_frame(frame))
        '''
        print("===========Avarage==========")
        print("sil", np.mean(self.avg["sil"]), np.std(self.avg["sil"]))
        print("v", np.mean(self.avg["v"]), np.std(self.avg["v"]))
        print("uv", np.mean(self.avg["uv"]), np.std(self.avg["uv"]))
        print("===========================")
        '''
        if median_size is not None:
            return self.fs_predicts_median, unvoice_amdf, unvoice_frame, voice_amdf, voice_frame
        return self.fs_predicts, unvoice_amdf, unvoice_frame, voice_amdf, voice_frame

    def get_minimum(self, frame: np.array):
        """
        This function use to find the minimum and number of minimum of array
        :param
            frame: array of number
        :return:
            min: minimum of array
            count_peak: number of minimum
        """
        count_peak = 0
        for i in range(1, len(frame) - 1):
            if frame[i] < frame[i + 1] and frame[i] < frame[i - 1]:
                count_peak += 1
        return np.argmin(frame), count_peak

    def median_of_frame(self, frame):
        """
        Calculate the median value of an frame
        :param
            frame: array of number
        :return:
            median: the median value of frame
        """
        n = len(frame)
        new_frame = sorted(frame)
        if n % 2 == 1:
            return new_frame[n // 2]
        return (new_frame[n // 2] + new_frame[n // 2 - 1]) / 2

    def read_audio(self, wav_path: str) -> np.array:
        """
        Read and assign audio and sample frequency to class
        :param
            wav_path: path to audio
        :return:
            None
        """
        self.fs, self.signal = read(wav_path)

    def get_audio(self):
        """
        Return the audio signal and frequency
        :return:
            fs: sample frequency of audio
            signal: the audio signal
        """
        return self.fs, self.signal



