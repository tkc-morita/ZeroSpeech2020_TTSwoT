# coding: utf-8

import torch
import torch.utils.data
import pandas as pd
import numpy as np
import scipy.io.wavfile as spw
import os.path


class Dataset(torch.utils.data.Dataset):
	def __init__(
			self,
			data_root,
			director_path,
			path_col='path',
			speaker_col='speaker',
			duration_col='duration',
			max_duration=float('inf'),
			channel=0,
			common_trans = None,
			in_trans = None,
			out_trans = None,
			f0_trans=None,
			speaker2ix = None,
		):
		self.data_root = data_root
		self.df_director = pd.read_csv(director_path)
		self.path_col = path_col
		self.speaker_col = speaker_col
		self.duration_col = duration_col
		self.channel = channel
		self.common_trans = common_trans
		self.in_trans = in_trans
		self.out_trans = out_trans
		self.f0_trans = f0_trans

		self.fs = self.get_fs()
		if max_duration<float('inf'):
			self.max_length = int(max_duration*self.fs)
		else:
			self.max_length = max_duration
		
		self.speaker2ix = speaker2ix
		if self.speaker2ix is None:
			self.speaker2ix = {speaker:ix for ix,speaker in enumerate(sorted(self.df_director[self.speaker_col].unique()))}
		self.ix2speaker = {ix:speaker for speaker,ix in self.speaker2ix.items()}

	def get_fs(self, data_ix=0):
		fs, _ = self.read_wav(data_ix)
		return fs

	
	def get_sample_freq(self):
		return self.fs

	def get_num_samples(self):
		return self.df_director[self.duration_col].values*self.fs

	def get_num_speakers(self):
		return len(self.speaker2ix)

	def set_transforms(self, common_trans=None, in_trans=None, out_trans=None, f0_trans=None):
		if not common_trans is None:
			self.common_trans = common_trans
		if not in_trans is None:
			self.in_trans = in_trans
		if not out_trans is None:
			self.out_trans = out_trans
		if not f0_trans is None:
			self.f0_trans = f0_trans

	def read_wav(self, data_ix):
		path = os.path.join(self.data_root, self.df_director.loc[data_ix,self.path_col])
		fs, wav = spw.read(path)
		if len(wav.shape)>1:
			wav = wav[:,self.channel]
		return fs, wav

	def get_speaker(self, data_ix):
		speaker = self.df_director.loc[data_ix,self.speaker_col]
		if speaker in self.speaker2ix:
			speaker_ix = self.speaker2ix[speaker]
		else:
			speaker_ix = float('nan')
		return speaker, speaker_ix

	def reset_director(self, new_df):
		self.df_director = new_df.reset_index(drop=True)

	def __len__(self):
		"""Return # of data strings."""
		return self.df_director.shape[0]

	def __getitem__(self, data_ix):
		_,wav = self.read_wav(data_ix)
		_,speaker_ix = self.get_speaker(data_ix)

		if wav.shape[0]>self.max_length:
			onset = torch.randint(wav.shape[0]-self.max_length+1,(1,)).item()
			wav = wav[onset:onset+self.max_length]

		if self.in_trans:
			input_data = self.in_trans(wav)
		if self.out_trans:
			target_data = self.out_trans(wav)
		if self.f0_trans:
			f0 = self.f0_trans(wav)
		return input_data, target_data, f0, speaker_ix, data_ix


	def get_wave_stats(self, transform=None):
		if transform is None:
			samples = [self.read_wav(data_ix)[1] for data_ix in self.df_director.index.tolist()]
		else:
			samples = [transform(self.read_wav(data_ix)[1]) for data_ix in self.df_director.index.tolist()]
		samples = np.concatenate(samples, axis=0)
		return np.mean(samples, axis=0), np.std(samples, axis=0, ddof=1)

class Compose(object):
	def __init__(self, transforms):
		self.transforms = transforms

	def __call__(self, data):
		for trans in self.transforms:
			data = trans(data)
		return data

class IterationBasedBatchSampler(torch.utils.data.BatchSampler):
	"""
	Wraps a BatchSampler, resampling from it until
	a specified number of iterations have been sampled.
	Partially Copied from maskedrcnn-benchmark.
	https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/data/samplers/iteration_based_batch_sampler.py
	"""

	def __init__(self, batch_sampler, num_iterations, start_iter=0):
		self.batch_sampler = batch_sampler
		self.num_iterations = num_iterations
		self.start_iter = start_iter
		if hasattr(self.batch_sampler.sampler, 'set_start_ix'):
			start_ix = (self.start_iter % len(self.batch_sampler)) * self.batch_sampler.batch_size
			self.batch_sampler.sampler.set_start_ix(start_ix)

	def __iter__(self):
		iteration = self.start_iter
		epoch = iteration // len(self.batch_sampler)
		while iteration <= self.num_iterations:
			if hasattr(self.batch_sampler.sampler, 'set_epoch'):
				self.batch_sampler.sampler.set_epoch(epoch)
			for batch in self.batch_sampler:
				iteration += 1
				if iteration > self.num_iterations:
					break
				yield batch
			epoch += 1

	def __len__(self):
		return self.num_iterations

class RandomSampler(torch.utils.data.RandomSampler):
	"""
	Custom random sampler for iteration-based learning.
	"""
	def __init__(self, *args, seed=111, **kwargs):
		super(RandomSampler, self).__init__(*args, **kwargs)
		self.epoch = 0
		self.start_ix = 0
		self.seed = seed

	def set_epoch(self, epoch):
		self.epoch = epoch

	def set_start_ix(self, start_ix):
		self.start_ix = start_ix

	def __iter__(self):
		g = torch.Generator()
		g.manual_seed(self.epoch+self.seed)
		start_ix = self.start_ix
		self.start_ix = 0
		n = len(self.data_source)
		if self.replacement:
			return iter(torch.randint(high=n, size=(self.num_samples,), dtype=torch.int64, generator=g).tolist()[start_ix:])
		return iter(torch.randperm(n, generator=g).tolist()[start_ix:])


def get_data_loader(dataset, batch_size=1, shuffle=False, num_iterations=None, start_iter=0, num_workers=1, random_seed=111):
	if shuffle:
		sampler = RandomSampler(dataset, replacement=False, seed=random_seed)
	else:
		sampler = torch.utils.data.SequentialSampler(dataset)
	batch_sampler = torch.utils.data.BatchSampler(sampler, batch_size, drop_last=False)
	if not num_iterations is None:
		batch_sampler = IterationBasedBatchSampler(batch_sampler, num_iterations, start_iter=start_iter)

	data_loader = torch.utils.data.DataLoader(dataset, num_workers=num_workers, batch_sampler=batch_sampler, collate_fn=collator)
	return data_loader

def collator(batch):
	batch = sorted(batch, key=lambda data: data[1].size(0), reverse=True) # Reordering
	input_data, target_data, f0, speaker_ix, data_ix = list(zip(*batch))
	input_data = torch.nn.utils.rnn.pack_sequence(input_data)
	target_data = torch.nn.utils.rnn.pack_sequence(target_data)
	f0 = torch.nn.utils.rnn.pack_sequence(f0)
	speaker_ix = torch.tensor(speaker_ix)
	return input_data, target_data, f0, speaker_ix, data_ix