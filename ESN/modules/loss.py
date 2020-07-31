# coding: utf-8

import torch

class SpectralLoss(torch.nn.Module):
	def __init__(
			self,
			n_ffts=[128, 512, 2048],
			frame_lengths=[80, 400, 1920],
			step_sizes=[40, 100, 640],
			eps=1e-5
			):
		super(SpectralLoss, self).__init__()
		self.n_ffts = n_ffts
		self.frame_lengths = frame_lengths
		self.step_sizes = step_sizes
		self.eps = eps

	def forward(self, wave1_padded, wave2_padded, wave_lengths):
		loss = 0.0
		for n_fft,frame_length,step_size in zip(self.n_ffts,self.frame_lengths,self.step_sizes):
			spectra_lengths = (wave_lengths.float() / step_size).floor().long() + 1
			spectra1 = wave1_padded.stft(n_fft,hop_length=step_size,win_length=frame_length)
			spectra1 = spectra1.transpose(1,2).contiguous()
			spectra1 = torch.nn.utils.rnn.pack_padded_sequence(spectra1, spectra_lengths, batch_first=True).data
			amplitude1 = spectra1.pow(2).sum(-1) + self.eps

			spectra2 = wave2_padded.stft(n_fft,hop_length=step_size,win_length=frame_length)
			spectra2 = spectra2.transpose(1,2).contiguous()
			spectra2 = torch.nn.utils.rnn.pack_padded_sequence(spectra2, spectra_lengths, batch_first=True).data
			amplitude2 = spectra2.pow(2).sum(-1) + self.eps

			loss = loss + (amplitude1.log() - amplitude2.log()).pow(2).mean() * 0.5
		loss = loss / len(self.n_ffts)
		return loss

	def pack_init_args(self):
		args = {
			"n_ffts":self.n_ffts,
			"frame_lengths":self.frame_lengths,
			"step_sizes":self.step_sizes,
			"eps":self.eps
		}
		return args