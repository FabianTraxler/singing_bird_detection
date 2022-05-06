"""
This file converts Labels in XLSX Format into a CSV Format and the other way around
"""
import json
import pydub
import pandas as pd
import os
from scipy.io import wavfile
from typing import Tuple
import numpy as np
from scipy.signal import spectrogram, butter, lfilter
import matplotlib.pyplot as plt
import noisereduce as nr
from tqdm import tqdm
import torch
from typing import Dict


class BokuBirdData:

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

    def _calc_overlap(self, sample_rate: int, frame_rate: int, frame_size: int) -> int:
        overlap = 0
        overlap = frame_size - sample_rate / frame_rate
        return int(overlap)

    def _get_spectogram(self, audio_buffer: np.ndarray, fs: int, frame_size: int, frame_rate: int = 100,
                        window: str = "hann") -> np.ndarray:
        overlap = self._calc_overlap(fs, frame_rate, frame_size)
        freqencies, times, plain_specgram = spectrogram(audio_buffer, fs, window, frame_size, overlap)

        return freqencies, times, plain_specgram

    def plot_spectrogram(self, audio_buffer: np.ndarray, fs: int, frame_size: int, log_spectrogram: bool = True,
                         frame_rate: int = 100, window: str = "hann", axs=None):
        freqencies, times, specgram = self._get_spectogram(audio_buffer, fs, frame_size, frame_rate=frame_rate,
                                                           window=window)
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


class XenoCantoData(torch.utils.data.Dataset):
    def __init__(self, cfg, dataset: pd.DataFrame = None, frame_duration: int = 50,
                 freq_min: int = 200, freq_max: int = 14000):
        self.data_path = cfg.data_dir

        self.audio_time = 5  # s

        self.frame_duration = frame_duration
        self.freq_min = freq_min
        self.freq_max = freq_max

        with open(os.path.join(cfg.data_dir, "species_mapping.json")) as f:
            self.species_mapping = json.load(f)

        self.species = cfg.birds
        self.species2int = {species: i for i, species in enumerate(self.species)}
        self.int2species = {i: species for i, species in enumerate(self.species)}

        if not os.path.exists(os.path.join(cfg.data_dir, "dataset.csv")) and dataset is None:
            self._generate_dataset()

        if dataset is None:
            self.dataset = pd.read_csv(os.path.join(self.data_path, "dataset.csv"))
        else:
            self.dataset = dataset

        self.dataset.reset_index(inplace=True)

    def _generate_dataset(self, frame_size: int = 5):
        dataset = pd.DataFrame()
        print("Generate Dataset..")

        for species in tqdm(self.species):
            species_mapped = self.species_mapping[species]
            if os.path.exists(os.path.join(self.data_path, "audio", species_mapped)):
                all_files = os.listdir(os.path.join(self.data_path, "audio", species_mapped))
                for file in all_files:
                    file_path = os.path.join(self.data_path, "audio", species_mapped, file)
                    fs, audio_buffer = self._read_audio(file_path)
                    if fs is None:
                        continue
                    audio_length = len(audio_buffer) / fs
                    for i in range(int(audio_length / frame_size) - 1):
                        new_row = {
                            "id": file.split(".")[0] + "_" + str(i),
                            "file_path": file_path,
                            "species": species,
                            "total_file_length": audio_length,
                            "start": frame_size * i * fs,
                            "end": frame_size * (i + 1) * fs
                        }
                        dataset = dataset.append(new_row, ignore_index=True)
            else:
                print("No Audio files found for species {}".format(species))

        dataset.to_csv(os.path.join(self.data_path, "dataset.csv"), index=False)

    def _read_audio(self, file: str, normalized: bool = False) -> Tuple:
        if os.path.exists(file):
            try:
                a = pydub.AudioSegment.from_mp3(file)
                y = np.array(a.get_array_of_samples())
                if a.channels == 2:
                    y = y.reshape((-1, 2))
                    y = y[:, 0]
                if normalized:
                    return a.frame_rate, np.float32(y) / 2 ** 15
                else:
                    return a.frame_rate, y
            except pydub.exceptions.CouldntDecodeError:
                pass

        print("File not available")
        return None, np.array([])

    def _calc_overlap(self, sample_rate: int, frame_rate: int, frame_size: int) -> int:
        overlap = 0
        overlap = frame_size - sample_rate / frame_rate
        return int(overlap)

    def _get_spectogram(self, audio_buffer: np.ndarray, fs: int, frame_size: int, frame_rate: int = 100,
                        window: str = "hann") -> np.ndarray:
        overlap = self._calc_overlap(fs, frame_rate, frame_size)
        freqencies, times, plain_specgram = spectrogram(audio_buffer, fs, window, frame_size, overlap)

        return freqencies, times, plain_specgram

    def plot_spectrogram(self, audio_buffer: np.ndarray, fs: int, frame_size: int, log_spectrogram: bool = True,
                         frame_rate: int = 100, window: str = "hann", axs=None):
        freqencies, times, specgram = self._get_spectogram(audio_buffer, fs, frame_size, frame_rate=frame_rate,
                                                           window=window)
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

    def get_spectrogram(self, data: Dict):
        fs = data["fs"]
        audio_buffer = data["audio"]

        frame_size = int(self.frame_duration / 1000 * fs)

        frequencies, _, spectrogram = self._get_spectogram(audio_buffer, fs, frame_size)

        freq_slice = np.where((frequencies >= self.freq_min) & (frequencies <= self.freq_max))[0]
        spectrogram = np.log10(spectrogram + 1)[freq_slice, :]

        if frequencies[-1] < self.freq_max:
            diff = int((self.freq_max - frequencies[-1]) / 20)
            spectrogram = np.pad(spectrogram, ((0, diff), (0, 0)), "constant")
        elif frequencies[1] > self.freq_min:
            diff = int((frequencies[0] - self.freq_min) / 20)
            spectrogram = np.pad(spectrogram, ((diff, 0), (0, 0)), "constant")

        return spectrogram  # , metadata

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        data = {}
        file_path = self.dataset.loc[idx, "file_path"]
        species = self.dataset.loc[idx, "species"]
        start = int(self.dataset.loc[idx, "start"])
        end = int(self.dataset.loc[idx, "end"])

        fs, audio_buffer = self._read_audio(file_path)
        audio_buffer = audio_buffer[start: end]
        if len(audio_buffer) < self.audio_time * fs:
            diff = self.audio_time * fs - len(audio_buffer)
            audio_buffer = np.pad(audio_buffer, ((0, diff)), "constant")

        data["audio"] = audio_buffer
        data["fs"] = fs
        spect = self.get_spectrogram(data)
        #data["spectrogram"] = self.get_spectrogram(data)

        label = np.zeros((len(self.species,)))
        label[self.species2int[species]] = 1

        return spect, label



if __name__ == "__main__":
    from torch.utils.data import DataLoader
    import torch

    print(torch.cuda.is_available())
    data = XenoCantoData()

    dataloader = DataLoader(data, num_workers=0, batch_size=5, shuffle=False)

    for label, images in tqdm(dataloader):
        a = 0
