import librosa
import librosa.filters
import math
import numpy as np
from scipy import signal
from hyperparams import Hyperparams as hp


def load_wav(path):
  return librosa.core.load(path, sr=hp.sr)[0]


def save_wav(wav, path):
  wav *= 32767 / max(0.01, np.max(np.abs(wav)))
  librosa.output.write_wav(path, wav.astype(np.int16), hp.sr)


def preemphasis(x):
  return signal.lfilter([1, -hp.preemphasis], [1], x)


def inv_preemphasis(x):
  return signal.lfilter([1], [1, -hp.preemphasis], x)


def spectrogram(y):
  D = _stft(preemphasis(y))
  S = _amp_to_db(np.abs(D)) - hp.ref_db
  return _normalize(S)

def save_spec(spectrogram,path):
	wav = inv_spectrogram(spectrogram)
	save_wav(wav,path)

def inv_spectrogram(spectrogram):
  '''Converts spectrogram to waveform using librosa'''
  S = _db_to_amp(_denormalize(spectrogram) + hp.ref_db)  # Convert back to linear
  return inv_preemphasis(_griffin_lim(S ** hp.power))          # Reconstruct phase



def _griffin_lim(S):
  '''librosa implementation of Griffin-Lim
  Based on https://github.com/librosa/librosa/issues/434
  '''
  angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
  S_complex = np.abs(S).astype(np.complex)
  y = _istft(S_complex * angles)
  for i in range(hp.griffin_lim_iters):
    angles = np.exp(1j * np.angle(_stft(y)))
    y = _istft(S_complex * angles)
  return y


def _stft(y):
  n_fft, hop_length, win_length = _stft_parameters()
  return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)


def _istft(y):
  _, hop_length, win_length = _stft_parameters()
  return librosa.istft(y, hop_length=hop_length, win_length=win_length)



def _stft_parameters():
  n_fft = hp.n_fft
  hop_length = int(hp.frame_shift * hp.sr)
  win_length = int(hp.frame_length  * hp.sr)
  hop_length = hp.hop_length
  win_length = hp.win_length
  return n_fft, hop_length, win_length


# Conversions:



def _amp_to_db(x):
  return 20 * np.log10(np.maximum(1e-5, x))

def _db_to_amp(x):
  return np.power(10.0, x * 0.05)


def _normalize(S):
  return np.clip((S - hp.min_db) / -hp.min_db, 0, 1)

def _denormalize(S):
  return (np.clip(S, 0, 1) * -hp.min_db) + hp.min_db

