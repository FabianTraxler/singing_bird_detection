"""
This file converts Labels in XLSX Format into a CSV Format and the other way around
"""
import pandas as pd
import os
from scipy.io import wavfile
from typing import Tuple
import numpy as np
from scipy.signal import spectrogram, butter, lfilter
import matplotlib.pyplot as plt
import noisereduce as nr


class BirdData:

    def __init__(self, data_path: str = "./data"):
        self.data_path = data_path
        self.files = set(os.listdir(os.path.join(data_path, "audio")))

        self.dataset = pd.read_excel(os.path.join(data_path, "labels", "weak_labels.xls")).drop_duplicates()
        labeled_files = self.dataset["filename.new"].unique()
        self.labels = pd.DataFrame()
        for i in range(1, self.dataset["filename.new"].max()):
            if i in labeled_files:
                vogelarten = self.dataset[self.dataset["filename.new"] == i]["species"].tolist()
                site = self.dataset[self.dataset["filename.new"] == i]["studysite"].tolist()[0]
                filename = "{}.wav".format(i)
                if filename in self.files:
                    audio_available = True
                else:
                    audio_available = False
                row = {"filename": filename, "species": vogelarten, "num_species": len(vogelarten),
                       "studysite": site, "audio_available": audio_available}
                self.labels = self.labels.append(row, ignore_index=True)

    def _read_audio(self, file: str) -> Tuple:
        if self.labels[self.labels["filename"] == file]["audio_available"].all():
            file_name = "{}/{}".format(os.path.join(self.data_path, "audio"), file)
            fs, audio_buffer = wavfile.read(file_name)
            return fs, audio_buffer
        else:
            print("File not available")
            return None, np.array([])

    def _calc_overlap(self, sample_rate: int, frame_rate:int, frame_size:int) -> int:
        overlap = 0
        overlap = frame_size - sample_rate / frame_rate
        return int(overlap)

    def _get_spectogram(self, audio_buffer: np.ndarray, fs: int,  frame_size: int, frame_rate: int = 100, window: str = "hann") -> np.ndarray:
        overlap = self._calc_overlap(fs, frame_rate, frame_size)
        freqencies, times, plain_specgram = spectrogram(audio_buffer, fs, window, frame_size, overlap)

        return freqencies, times, plain_specgram

    def plot_spectrogram(self, audio_buffer: np.ndarray, fs: int,  frame_size: int, log_spectrogram: bool = True, frame_rate: int = 100, window: str = "hann", axs = None):
        freqencies, times, specgram = self._get_spectogram(audio_buffer, fs, frame_size, frame_rate=frame_rate, window=window)
        if log_spectrogram:
            specgram = np.log10(specgram + 1)
        if not axs:
            fig, axs = plt.subplots(1, figsize=(15, 5))
        a = axs.imshow(specgram, origin='lower', aspect='auto')
        a = axs.set_ylabel("Frequency [Hz]")
        a = axs.set_yticks(np.arange(0, 800, 100))
        a = axs.set_yticklabels(np.arange(0, 16000, 2000))

        a = axs.set_xticks(np.arange(0, specgram.shape[1], 100))
        a = axs.set_xticklabels(np.arange(0, specgram.shape[1] / 100, 1))

        return specgram, axs

    def reduce_noise(self, audio_buffer: np.ndarray, fs: int) -> np.ndarray:
        audio_buffer = nr.reduce_noise(y=audio_buffer, sr=fs)
        audio_buffer = self._bandpass_filter(audio_buffer, 500, 8000, fs)
        return audio_buffer

    def _bandpass_filter(self, audio_buffer: np.ndarray, lowcut: int, highcut: int, fs: int, order: int = 5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        y = lfilter(b, a, audio_buffer)
        return y


if __name__ == "__main__":
    a = BirdData()
    a._read_audio("1.wav")