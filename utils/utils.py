import re
import nltk
from nltk.corpus import cmudict
import torchaudio
from textgrid import TextGrid
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import numpy as np

try:
    import nltk
    nltk.data.find("../data/LibriTTS/librispeech-lexicon.txt")
except:
    nltk.download("cmudict")

cmu_dict = cmudict.dict()


class PhonemeVocab:
    def __init__(self, name: str="vocab"):
        self.name = name
        self._phoneme2idx = {"<PAD>": 0, "<UNK>": 1, "<SOS>": 2, "<EOS>": 3}
        self._phoneme2count = {"<PAD>": 1, "<UNK>": 1, "<SOS>": 1, "<EOS>": 1}
        self._idx2phoneme = {0: "<PAD>", 1: "<UNK>", 2: "<SOS>", 3: "<EOS>"}
        self._n_phonemes = 4
        self.puncuation2prosody = {
            " ": "<SHORT_PAUSE>",
            ",": "<SHORT_PAUSE>",
            ".": "<LONG_PAUSE>",
            "?": "<RAISE_PITCH>",
            "!": "<EMPHASIS>",
        }

    def get_phoneme(self) -> list[str]:
        return list(self._phoneme2count.keys())

    def num_phonemes(self) -> int:
        return self._n_phonemes

    def phoneme2index(self, word: str) -> int:
        return self._phoneme2idx.get(word, self._phoneme2idx["<UNK>"])

    def idx2phoneme(self, idx: int) -> str:
        return self._idx2phoneme[idx]

    def phoneme2count(self, word: str) -> int:
        return self._phoneme2count[word]

    def add_sentence(self, sentence: str) -> None:
        for word in self.tokenize(sentence):
            self.add_phoneme(word)

    def add_phoneme(self, phoneme: str) -> None:
        if phoneme not in self._phoneme2idx:
            self._phoneme2idx[phoneme] = self._n_phonemes
            self._phoneme2count[phoneme] = 1
            self._idx2phoneme[self._n_phonemes] = phoneme
            self._n_phonemes += 1
        else:
            self._phoneme2count[phoneme] += 1

    def normalize_text(self, text: str) -> str:
        text = text.lower()

        text = text.replace("-", " ")
        text = re.sub("[^a-zA-Z0-9,.!? ]+", "", text)

        return text

    def text2phonemes(self, text: str) -> list[str]:
        words = text.split()
        phonemes = []
        for word in words:
            if word in self.puncuation2prosody:
                phonemes.append(self.puncuation2prosody[word])
            elif word in cmu_dict:
                phonemes.extend(cmu_dict[word][0])
            else:
                phonemes.append("<UNK>")

        return phonemes

    def tokenize(self, text: str) -> list[str]:
        normalized_text = self.normalize_text(text)

        phonemes = self.text2phonemes(normalized_text)

        return ["<SOS>"] + phonemes + ["<EOS>"]


def compute_mel_spectrogram(
    dataset, sample_rate=16000, n_mels=80, n_fft=1024, hop_length=256
):
    """
    Compute the mel-spectrogram for FastSpeech 2.

    Args:
        dataset (list): List of dictionaries that contain the wav file path.
        sample_rate (int): Target sample rate (default: 16kHz).
        n_mels (int): Number of mel bands (default: 80).
        n_fft (int): FFT size (default: 1024).
        hop_length (int): Hop length between frames (default: 256).

    Returns:
        Tensor: Mel-spectrogram of shape (n_mels, time_frames).
    """
    # Load the audio
    mel_data = []

    # Create the mel-spectrogram transformation
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
    )

    for data in dataset:
        tmp = {}
        tmp["id"] = data["id"]
        wav_path = data["wav_path"]
        waveform, sr = torchaudio.load(wav_path)

        # Compute the mel-spectrogram
        mel = mel_spectrogram(waveform)

        # Convert to log scale (log-magnitude spectrogram)
        mel = torchaudio.functional.amplitude_to_DB(
            mel, multiplier=10.0, amin=1e-10, db_multiplier=0.0
        )
        tmp["mel"] = mel.squeeze(0)
        mel_data.append(tmp)

    return mel_data  # Shape: (n_mels, time_frames)


def extract_energy(wav_path, start_time, end_time, sample_rate=16000):
    """
    Extract mean energy (RMS) for a phoneme's duration.

    Args:
        wav_path (str): Path to the WAV file.
        start_time (float): Start time of the phoneme (in seconds).
        end_time (float): End time of the phoneme (in seconds).
        sample_rate (int): Sample rate of the audio (default: 16kHz).

    Returns:
        float: RMS energy for the phoneme duration.
    """
    # Load the audio
    waveform, sr = torchaudio.load(wav_path)

    # Resample if necessary
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
        waveform = resampler(waveform)

    # Convert time to sample indices
    start_sample = int(start_time * sample_rate)
    end_sample = int(end_time * sample_rate)

    # Extract the relevant segment
    segment = waveform[:, start_sample:end_sample].numpy()

    # Compute RMS energy
    rms_energy = np.sqrt(np.mean(segment**2))

    return rms_energy


def extract_mean_pitch(pitch, start_time, end_time):
    """
    Extract the mean pitch within a specified time range from a WAV file.

    Args:
        pitch (object): paselmouth object.
        start_time (float): Start time of the phoneme (in seconds).
        end_time (float): End time of the phoneme (in seconds).

    Returns:
        float: Mean pitch (in Hz) within the time range, or None if no pitch detected.
    """
    pitch = pitch
    selected_pitch = pitch.selected_array["frequency"]
    timestamps = pitch.xs()

    # Filter pitch values within the start and end time
    pitch_values = [
        freq
        for t, freq in zip(timestamps, selected_pitch)
        if start_time <= t <= end_time and freq > 0  # Exclude unvoiced segments
    ]

    # Return mean pitch, or None if no pitch values are found
    return sum(pitch_values) / len(pitch_values) if pitch_values else None


def extract_phoneme_durations(textgrid_path):
    """
    Extract phoneme durations from a TextGrid file for FastSpeech 2 training.

    Args:
        textgrid_path (str): Path to the TextGrid file.

    Returns:
        list: List of dictionaries containing phoneme, start_time, end_time, and duration.
    """
    # Load the TextGrid file
    try:
        tg = TextGrid.fromFile(textgrid_path)
    except:
        return None
    # Find the "phones" tier
    phoneme_tier = tg.getFirst("phones")
    phoneme_durations = []

    # Extract intervals from the "phones" tier
    for interval in phoneme_tier:
        if interval.mark.strip():  # Skip empty intervals
            phoneme_durations.append(
                {
                    "phoneme": interval.mark.strip(),
                    "start_time": interval.minTime,
                    "end_time": interval.maxTime,
                    "duration": interval.maxTime - interval.minTime,
                }
            )

    return phoneme_durations


class VoiceDataset(Dataset):
    def __init__(self, dataset: list[dict], mel_data: torch.Tensor, pitch_spectrograms, speaker_embedding, vocab: PhonemeVocab, max_length: int=400, max_mel_length: int=2080, max_spectrogram_length: int=5868):
        self.dataset = dataset
        self.mel_data = mel_data
        self.vocab = vocab
        self.max_length = max_length
        self.max_mel_length = max_mel_length
        self.max_spectrogram_length = max_spectrogram_length
        self.merged_data = list(zip(self.dataset, mel_data, pitch_spectrograms, speaker_embedding))

    def _phoneme_pad_handling(self, data):
        phonemes = []
        pitch = []
        energy = []
        duration = []
        for phoneme in data["phoneme_durations"]:
            phoneme_idx = self.vocab.phoneme2index(phoneme["phoneme"])
            phonemes.append(phoneme_idx)
            pitch.append(phoneme["pitch"])
            energy.append(phoneme["energy"])
            duration.append(phoneme["duration"])

        phonemes = phonemes + [self.vocab.phoneme2index("<PAD>")] * (
            self.max_length - len(phonemes)
        )
        pitch = pitch + [0.0] * (self.max_length - len(pitch))
        energy = energy + [0.0] * (self.max_length - len(energy))
        duration = duration + [0] * (self.max_length - len(duration))

        return phonemes, pitch, energy, duration

    def __len__(self) -> int:
        return len(self.merged_data)

    def __getitem__(self, idx: int):
        data, mels, spectrograms, speaker_embedding = self.merged_data[idx]
        spectrogram = torch.from_numpy(spectrograms)
        speaker_embedding = torch.tensor(speaker_embedding)
        phonemes, pitch, energy, duration = self._phoneme_pad_handling(data)

        mels = F.pad(
            mels.clone(), (0, self.max_mel_length - mels.shape[1]), mode="constant", value=0.0
        )

        spectrogram = F.pad(
            spectrogram.clone(), (0, self.max_spectrogram_length - spectrogram.shape[1]), mode="constant", value=0.0
        )
        return (
            torch.tensor(phonemes, dtype=torch.long),
            torch.tensor(pitch, dtype=torch.float32),
            torch.tensor(energy, dtype=torch.float32),
            torch.tensor(duration, dtype=torch.float32),
            mels,
            spectrogram,
            speaker_embedding
        )
