# coding: utf-8

import numpy as np
import math
import torch
import librosa
from scipy.interpolate import interp1d
from parselmouth import Sound

def normalize_int16(wav):
	if isinstance(wav, np.ndarray):
		wav = wav.astype(np.float32)
	else: # torch.Tensor
		wav = wav.float()
	wav /= 2**15
	return wav

class MuLawCompund(object):
	def __init__(self, mu=255):
		self.mu = mu

	def __call__(self, wav):
		if isinstance(wav, np.ndarray):
			return np.sign(wav) * np.log1p(self.mu*np.abs(wav)) / np.log1p(self.mu)
		else: # torch.Tensor
			return wav.sign() * (self.mu*wav.abs()).log1p() / math.log1p(self.mu)

	def inverse(self, signed_magnitude):
		if isinstance(signed_magnitude, np.ndarray):
			return np.sign(signed_magnitude) * ((self.mu+1.0)**np.abs(signed_magnitude) - 1.0) / self.mu
		else: # torch.Tensor
			return signed_magnitude.sign() * ((self.mu+1.0)**signed_magnitude.abs() - 1.0) / self.mu

class Quantize(object):
	def __init__(self, levels=256):
		self.levels = levels

	def __call__(self, x):
		"""
		-1 <= x < 1
		"""
		q = ((x+1) * 0.5 * (self.levels-1)).round()
		if isinstance(q, np.ndarray):
			return q.astype(int)
		else: # torch.Tensor
			return q.long()

	def dequantize(self, c):
		"""
		c: integer indexing quantized value.
		"""
		if isinstance(c, np.ndarray):
			c = c.astype(np.float32)
		else: # torch.Tensor
			c = c.float()
		return (c / (self.levels-1.0)) * 2 - 1.0


class MFCC(object):
	def __init__(
			self,
			fs,
			win_length,
			hop_length,
			n_mfcc=13,
			max_delta_order=2,
			):
		self.fs = fs
		self.win_length = win_length
		self.hop_length = hop_length
		self.n_mfcc = n_mfcc
		self.max_delta_order = max_delta_order
		try:
			from torchaudio.transforms import MelSpectrogram
			self.torch_mfcc = MelSpectrogram(sample_rate=self.fs, win_length=self.win_length, hop_length=self.hop_length, n_mfcc=self.n_mfcc)
		except:
			self.torch_mfcc = None

	def __call__(self, wav):
		"""
		Outout form are (optional dim for torch.Tensor) x n_mfcc x seq_length.
		"""
		if isinstance(wav, np.ndarray):
			wav = np.pad(wav, (self.win_length-1)//2, mode='reflect')
			mfcc = librosa.feature.mfcc(
						wav.astype(np.float32),
						sr=self.fs,
						n_fft=self.win_length,
						hop_length=self.hop_length,
						n_mfcc=self.n_mfcc,
						center=False,
						)
			out = np.concatenate(
					[mfcc]
					+
					[librosa.feature.delta(mfcc, order=order)
						for order in range(1,self.max_delta_order+1)]
					,
					axis=0
					)
			return np.transpose(out)
		# elif self.torch_mfcc is None:
			# raise TypeError('You need to install torchaudio to get MFCC from torch.Tensor data.')
		else:
			TypeError('torch.Tensor is not currently supported for MFCC derivative estimation.')
			# return self.torch_mfcc(wav)


class STFT(object):
	def __init__(
			self,
			fs,
			win_length,
			hop_length,
			):
		self.fs = fs
		self.win_length = win_length
		self.hop_length = hop_length

	def __call__(self, wav):
		"""
		Outout form are (optional dim for torch.Tensor) x n_mfcc x seq_length.
		"""
		if isinstance(wav, np.ndarray):
			spectra = librosa.core.stft(
						wav.astype(np.float32),
						n_fft=self.win_length,
						hop_length=self.hop_length,
						center=True,
						window='hann'
						)
			spectral_amplitude = np.absolute(spectra)
			spectral_amplitude = np.transpose(spectral_amplitude)
		else:
			spectra = torch.stft(
						wav,
						self.win_length,
						hop_length=self.hop_length,
						center=True,
						window='hann_window'
						)
			spectral_amplitude = spectra.pow(2).sum(-1).sqrt()
			spectral_amplitude = spectral_amplitude.transpose(0,1).contiguous()
		return spectral_amplitude

class Rescale(object):
	def __init__(self, scalar):
		self.scalar = float(scalar)

	def __call__(self, x):
		return x / self.scalar

def shift(x):
	"""
	[0,1] -> [-1,1]
	"""
	return 2*x - 1.0

class Normalize(object):
	def __init__(self, mean, std):
		self.mean = mean
		self.std = std

	def __call__(self, x):
		return (x-self.mean) / self.std

class F0(object):
	"""
	Pitch detection.
	"""
	def __init__(self, fs, freq_low=75.0, freq_high=600.0, eps=10e-5): # Default freq_low and freq_high follow Praat
		self.fs = fs
		self.freq_low = freq_low
		self.freq_high = freq_high
		self.eps = eps

	def __call__(self, wav):
		snd = Sound(wav, self.fs)
		pitch = snd.to_pitch(pitch_floor=self.freq_low, pitch_ceiling=self.freq_high)
		freq = np.array([pitch.get_value_in_frame(ix) for ix in range(1,pitch.get_number_of_frames()+1)])
		freq = np.clip(np.nan_to_num(freq,nan=self.eps), self.eps, None)
		freq = torch.from_numpy(freq)
		freq = torch.nn.functional.interpolate(freq[None,None,:], size=wav.size).view(-1)
		return freq