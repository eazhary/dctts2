import librosa
import librosa.filters
import math
import numpy as np
from scipy import signal
from hyperparams import Hyperparams as hp
import tensorflow as tf


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

def inv_spectrogram_tensorflow(spectrogram):
    '''Builds computational graph to convert spectrogram to waveform using TensorFlow.
    Unlike inv_spectrogram, this does NOT invert the preemphasis. The caller should call
    inv_preemphasis on the output after running the graph.
    '''
    S = _db_to_amp_tensorflow(_denormalize_tensorflow(spectrogram) + hp.ref_db)
    return _griffin_lim_tensorflow(tf.pow(S, hp.power))

def _griffin_lim_tensorflow(S):
    '''TensorFlow implementation of Griffin-Lim
    Based on https://github.com/Kyubyong/tensorflow-exercises/blob/master/Audio_Processing.ipynb
    '''
    with tf.variable_scope('griffinlim'):
    # TensorFlow's stft and istft operate on a batch of spectrograms; create batch of size 1
        S = tf.expand_dims(S, 0)
        S_complex = tf.identity(tf.cast(S, dtype=tf.complex64))
        y = _istft_tensorflow(S_complex)
        for i in range(hp.griffin_lim_iters):
            est = _stft_tensorflow(y)
            angles = est / tf.cast(tf.maximum(1e-8, tf.abs(est)), tf.complex64)
            y = _istft_tensorflow(S_complex * angles)
    return tf.squeeze(y, 0)

def _denormalize_tensorflow(S):
  return (tf.clip_by_value(S, 0, 1) * -hp.min_db) + hp.min_db

def _db_to_amp_tensorflow(x):
  return tf.pow(tf.ones(tf.shape(x)) * 10.0, x * 0.05)
  
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

def _istft_tensorflow(stfts):
  n_fft, hop_length, win_length = _stft_parameters()
  return tf.contrib.signal.inverse_stft(stfts, win_length, hop_length, n_fft)

def _stft_tensorflow(signals):
  n_fft, hop_length, win_length = _stft_parameters()
  return tf.contrib.signal.stft(signals, win_length, hop_length, n_fft, pad_end=False)

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

