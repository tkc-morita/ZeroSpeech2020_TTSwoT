# coding: utf-8

import torch
import math
import numpy as np
import scipy.signal as spsig
from .rnn_utils import pad_flatten_sequence


class Articulation(torch.nn.Module):
	def __init__(
			self,
			channels,
			num_harmonics=7,
			src_noise_std=0.003,
			src_amplitude=0.1,
			filter_blocks=5,
			filter_layers_per_block=10,
			filter_length=3,
			out_freq=16000,
			low_passband_voiced=[0,5000],
			high_passband_voiced=[7000,8000],
			low_passband_voiceless=[0,1000],
			high_passband_voiceless=[3000,8000],
			fir_order=10
			):
		super(Articulation, self).__init__()
		self.source = Source(out_freq=out_freq,num_harmonics=num_harmonics, noise_std=src_noise_std, amplitude=src_amplitude)
		self.filter_harmonic = Filter(channels, num_blocks=filter_blocks, num_layers_per_block=filter_layers_per_block, filter_length=filter_length)
		self.filter_noisy = Filter(channels, num_blocks=1, num_layers_per_block=filter_layers_per_block, filter_length=filter_length)
		self.fir_voiced_low_pass = FIRFilter(low_passband_voiced,high_passband_voiced,fs=out_freq,order=fir_order)
		self.fir_voiced_high_pass = FIRFilter(high_passband_voiced,low_passband_voiced,fs=out_freq,order=fir_order)
		self.fir_voiceless_low_pass = FIRFilter(low_passband_voiceless,high_passband_voiceless,fs=out_freq,order=fir_order)
		self.fir_voiceless_high_pass = FIRFilter(high_passband_voiceless,low_passband_voiceless,fs=out_freq,order=fir_order)
		self.out_freq = out_freq

	def forward(self, feature_padded, out_lengths):
		voicedness = (feature_padded[:,0,:]*5.0).sigmoid()
		feature_padded = feature_padded.transpose(1,2).contiguous()
		feature = torch.nn.utils.rnn.pack_padded_sequence(feature_padded, out_lengths, batch_first=True)

		harmonic,noisy = self.source(feature.data[:,0], feature.batch_sizes)

		harmonic = self.filter_harmonic(harmonic, feature.data, feature.batch_sizes)
		noisy = self.filter_noisy(noisy, feature.data, feature.batch_sizes)

		harmonic,_ = pad_flatten_sequence(harmonic, feature.batch_sizes, batch_first=True)
		noisy,_ = pad_flatten_sequence(noisy, feature.batch_sizes, batch_first=True)

		voiced = self.fir_voiced_low_pass(harmonic) + self.fir_voiced_high_pass(noisy)
		voiceless = self.fir_voiceless_low_pass(harmonic) + self.fir_voiceless_high_pass(noisy)

		wave = voicedness * voiced + (1-voicedness) * voiceless
		return wave, feature.data[:,0]

	def pack_init_args(self):
		args = {
			"num_harmonics":self.source.harmonic_ixs.size(0)-1,
			"src_noise_std":self.source.noise_std,
			"src_amplitude":self.source.amplitude,
			"channels":self.filter_harmonic.channels,
			"filter_blocks":self.filter_harmonic.num_blocks,
			"filter_layers_per_block":self.filter_harmonic.num_layers_per_block,
			"filter_length":self.filter_harmonic.filter_length,
			"out_freq":self.out_freq,
			"low_passband_voiced":self.fir_voiced_low_pass.passband,
			"high_passband_voiced":self.fir_voiced_high_pass.passband,
			"low_passband_voiceless":self.fir_voiceless_low_pass.passband,
			"high_passband_voiceless":self.fir_voiceless_high_pass.passband,
			"fir_order":self.fir_voiced_low_pass.order
		}
		return args

class Source(torch.nn.Module):
	def __init__(self, out_freq=16000, num_harmonics=7, noise_std=0.003, amplitude=0.1, f0_max=600.0):
		super(Source, self).__init__()
		self.noise_std = noise_std
		self.amplitude = amplitude
		self.combine_harmonics = torch.nn.Linear(num_harmonics+1, 1)
		self.register_buffer(
			"harmonic_ixs",
			torch.arange(1,num_harmonics+2).float()
			)
		self.out_freq = out_freq
		self.log_f0_max = math.log(f0_max)

	def forward(self, log_f0, batch_sizes):
		"""
		log_f0: 1D array flattening batched time series. Of size sum(seq_len)
					  Intended to represent log(F0).
		"""
		f0 = log_f0.clamp(max=self.log_f0_max).exp()
		standard_noise = torch.randn_like(f0)

		voiceless = (self.amplitude / 3.0) * standard_noise

		freqs = 2 * math.pi * f0[:,None] * self.harmonic_ixs[None,:] / self.out_freq
		freqs,lengths = pad_flatten_sequence(freqs, batch_sizes, batch_first=True)
		init_phase = (2 * torch.rand_like(freqs[:,0,0]) - 1.0) * math.pi
		angle = freqs.cumsum(1) + init_phase[:,None,None]
		angle = torch.nn.utils.rnn.pack_padded_sequence(angle, lengths, batch_first=True).data
		harmonics = self.amplitude * angle.sin() + self.noise_std * standard_noise[:,None]
		voiced = self.combine_harmonics(harmonics).squeeze(-1)

		voiced = voiced.tanh()
		return voiced, voiceless

class Filter(torch.nn.Module):
	def __init__(self, channels, num_blocks=5, num_layers_per_block=10, filter_length=3):
		super(Filter, self).__init__()
		self.filter_blocks = torch.nn.Sequential(*[
			FilterBlock(channels, filter_length=filter_length, num_layers=num_layers_per_block)
			for b in range(num_blocks)
		])
		self.num_blocks = num_blocks
		self.num_layers_per_block = num_layers_per_block
		self.filter_length = filter_length
		self.channels = channels

	def forward(self, src, condition, batch_sizes):
		args = {'src':src[:,None], 'condition':condition, 'batch_sizes':batch_sizes}
		filtered = self.filter_blocks(args)['src'].squeeze(-1)
		return filtered


class FilterBlock(torch.nn.Module):
	def __init__(self, channels, filter_length=3, num_layers=10):
		super(FilterBlock, self).__init__()
		self.preprocess = torch.nn.Linear(1, channels)
		self.dilated_convs = torch.nn.Sequential(*[
								DilatedConv(channels, filter_length, 2**k)
								for k in range(num_layers)
								])
		self.postprocess = torch.nn.Linear(channels, 1)

	def forward(self, args):
		src_orig = args['src']
		args['src'] = self.preprocess(src_orig)
		filtered = self.dilated_convs(args)['src']
		args['src'] = self.postprocess(filtered) + src_orig
		return args


class DilatedConv(torch.nn.Module):
	"""
	Note: The neural source-filter model is not auto-regressive,
		  and thus, the dillated convolution needs not be causal.
		  However, Wang et al. (2020) adopt the causal convolution,
		  so we follow them here.
	"""
	def __init__(self, channels, filter_length, dilation):
		super(DilatedConv, self).__init__()
		self.conv = torch.nn.Conv1d(
						channels,
						channels,
						kernel_size=filter_length,
						dilation=dilation)
		self.left_padding = dilation * (filter_length-1)

	def forward(self, args):
		src_padded, lengths = pad_flatten_sequence(args['src'], args['batch_sizes'], batch_first=True)
		conved = self.conv(src_padded.transpose(1,2).contiguous())
		conved = torch.nn.functional.pad(conved, (self.left_padding,0)).transpose(1,2).contiguous()
		conved = torch.nn.utils.rnn.pack_padded_sequence(conved, lengths, batch_first=True).data
		conved = conved.tanh() + args['src'] + args['condition']
		args['src'] = conved
		return args

class FIRFilter(torch.nn.Module):
	def __init__(
			self,
			passband,
			stopband,
			fs=16000,
			order=10
			):
		super(FIRFilter, self).__init__()
		self.passband = passband
		self.stopband = stopband
		if passband[0] < stopband[0]: # low pass
			bands = list(passband) + list(stopband)
			desired = [1,0]
		else: # high pass
			bands = list(stopband) + list(passband)
			desired = [0,1]
		taps = spsig.remez(order+1, bands, desired, fs=fs).astype(np.float32)

		self.register_parameter(
			"filter_coefs",
			torch.nn.Parameter(torch.from_numpy(taps)[None,None,:], requires_grad=False)
		)
		self.order = order

	def forward(self, wav):
		wav = torch.nn.functional.pad(wav, (self.order,0), mode='constant', value=0.0)
		if wav.ndim<3:
			wav = wav.unsqueeze(1)
		wav = torch.nn.functional.conv1d(wav, self.filter_coefs)
		wav = wav.squeeze(1)
		return wav