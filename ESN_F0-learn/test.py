# coding: utf-8

import torch
from modules.data_utils import Compose
import numpy as np
import pandas as pd
import scipy.io.wavfile as spw
from modules import data_utils, rnn_utils, sound_transforms
import learning
import os, argparse, itertools, json


class Tester(learning.Learner):
	def __init__(self, model_config_path, device = 'cpu', seed=111):
		torch.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)
		self.device = torch.device(device)
		self.retrieve_model(checkpoint_path = model_config_path, device=device)
		for param in self.parameters():
			param.requires_grad = False
		self.audition.eval() # Turn off dropout
		self.cognition.eval()
		self.articulation.eval()


	def encode_and_decode(self, input_frames, target_wav, speaker, is_packed = False, to_numpy = True):
		if not is_packed:
			if not isinstance(input_frames, list):
				input_frames = [input_frames]
			if not isinstance(target_wav, list):
				target_wav = [target_wav]
			input_frames = torch.nn.utils.rnn.pack_sequence(input_frames)
			target_wav = torch.nn.utils.rnn.pack_sequence(target_wav)
		with torch.no_grad():
			_,out_lengths = torch.nn.utils.rnn.pad_packed_sequence(target_wav, batch_first=True)
			input_frames = input_frames.to(self.device)

			code = self.audition(input_frames)
			logits,downsampled_batch_sizes = self.cognition.encode(code)

			probs = logits.softmax(-1)
			one_hot,features = self.cognition.vae.get_map_inference(logits)
			one_hot_padded,code_lengths = rnn_utils.pad_flatten_sequence(one_hot, downsampled_batch_sizes, batch_first=True)
			features_padded,_ = rnn_utils.pad_flatten_sequence(features, downsampled_batch_sizes, batch_first=True)
			probs_padded,_ = rnn_utils.pad_flatten_sequence(probs, downsampled_batch_sizes, batch_first=True)
			if speaker is None:
				wav_padded = None
				out_lengths = None
			else:
				speaker = speaker.expand(input_frames.batch_sizes[0]).to(self.device)
				features = self.cognition.decode(features, speaker, downsampled_batch_sizes, out_lengths)
				wav_padded,_ = self.articulation(features, out_lengths)
		if to_numpy:
			one_hot_padded = one_hot_padded.data.cpu().numpy()
			features_padded = features_padded.data.cpu().numpy()
			probs_padded = probs_padded.data.cpu().numpy()
			code_lengths = code_lengths.data.cpu().numpy()
			if not speaker is None:
				wav_padded = wav_padded.data.cpu().numpy()
				out_lengths = out_lengths.data.cpu().numpy()
		return one_hot_padded,probs_padded,features_padded,code_lengths,wav_padded,out_lengths


	def main(
			self,
			dataset,
			save_dir,
			synthesis_speaker_col='synthesis_speaker',
			batch_size=1,
			num_workers=1,
			omit_repetition=False
			):
		feature_dir = os.path.join(save_dir, 'as_features/')
		prob_dir = os.path.join(save_dir, 'as_probs/')
		onehot_dir = os.path.join(save_dir, 'as_one-hot/')
		synthesis_dir = os.path.join(save_dir, 'synthesis/')
		if not os.path.isdir(feature_dir):
			os.makedirs(feature_dir)
		if not os.path.isdir(prob_dir):
			os.makedirs(prob_dir)
		if not os.path.isdir(onehot_dir):
			os.makedirs(onehot_dir)
		if not os.path.isdir(synthesis_dir):
			os.makedirs(synthesis_dir)
		full_df = dataset.df_director.fillna(value={synthesis_speaker_col:'__NA__'})
		for synthesis_speaker,sub_df in full_df.groupby(synthesis_speaker_col):
			dataset.reset_director(sub_df)
			if synthesis_speaker=='__NA__':
				synthesis_speaker = None
			else:
				synthesis_speaker_name = synthesis_speaker
				synthesis_speaker = torch.tensor(dataset.speaker2ix[synthesis_speaker])
			dataloader = data_utils.get_data_loader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
			for input_frames, target_wav, _, speaker, data_ixs in dataloader:

				one_hot_padded,probs_padded,features_padded,code_lengths,wav_padded,out_lengths = self.encode_and_decode(input_frames, target_wav, synthesis_speaker, is_packed=True, to_numpy=True)

				data_paths = dataset.df_director.loc[data_ixs,dataset.path_col]
				for ix_in_batch,p in enumerate(data_paths):
					filename_wo_ext = os.path.splitext(os.path.basename(p))[0]

					code_l = code_lengths[ix_in_batch]
					one_hot = one_hot_padded[ix_in_batch,:code_l,:]
					features = features_padded[ix_in_batch,:code_l,:]
					probs = probs_padded[ix_in_batch,:code_l,:]
					if omit_repetition:
						one_hot, features, probs = self.omit_consecutive_repetition(one_hot, features, probs)

					df_category = pd.DataFrame(one_hot)
					df_category.to_csv(os.path.join(onehot_dir, filename_wo_ext)+'.txt', index=False, header=None, sep=' ')
					df_feat = pd.DataFrame(features)
					df_feat.to_csv(os.path.join(feature_dir, filename_wo_ext)+'.txt', index=False, header=None, sep=' ')
					df_prob = pd.DataFrame(probs)
					df_prob.to_csv(os.path.join(prob_dir, filename_wo_ext)+'.txt', index=False, header=None, sep=' ')
					if not synthesis_speaker is None:
						out_l = out_lengths[ix_in_batch]
						wav = wav_padded[ix_in_batch,:out_l]
						filename_wo_ext = '_'.join([synthesis_speaker_name]+filename_wo_ext.split('_')[1:])
						basename = os.path.join(synthesis_dir, filename_wo_ext)
						spw.write(basename+'.wav', dataset.fs, wav)

	def omit_consecutive_repetition(self, vecs, *other_vecs):
		"""
		Omit consecutive repetitions in vecs.
		vecs: ndarray[time,dim]
		"""
		changes = [True]+(vecs[:-1]!=vecs[1:]).any(axis=-1).tolist()
		out = vecs[changes,...]
		if other_vecs:
			out = [out]
			for v in other_vecs:
				out.append(v[changes,...])
		return out


def get_args():
	parser = argparse.ArgumentParser()

	parser.add_argument('model_path', type=str, help='Path to the model checkpoint.')
	parser.add_argument('input_root', type=str, help='Path to the root directory under which inputs are located.')
	parser.add_argument('director_file', type=str, help='Path to the csv file containing relative paths to the wav files and speaker IDs.')
	parser.add_argument('--path_col', type=str, default='filename', help='Name of the column containing relative paths to the wav files.')
	parser.add_argument('--speaker_col', type=str, default='speaker', help='Name of the column containing speaker IDs.')
	parser.add_argument('--duration_col', type=str, default='duration', help='Name of the column containing the file duration (in sec).')
	parser.add_argument('--synthesis_speaker_col', type=str, default='synthesis_speaker', help='Name of the column containing the synthesis speaker.')
	parser.add_argument('-S', '--save_dir', type=str, default=None, help='Path to the directory where results are saved.')
	parser.add_argument('-s', '--seed', type=int, default=1111, help='random seed (used in the wave production).')
	parser.add_argument('-d', '--device', type=str, default='cpu', help='Computing device.')
	parser.add_argument('-b', '--batch_size', type=int, default=64, help='Batch size for training.')
	parser.add_argument('--fft_frame_length', type=float, default=0.025, help='FFT frame length in sec.')
	parser.add_argument('--fft_step_size', type=float, default=0.01, help='FFT step size in sec.')
	parser.add_argument('--n_mfcc', type=int, default=13, help='# of MFCCs to use.')
	parser.add_argument('--mfcc_max_delta', type=int, default=2, help='Max order of derivatives of MFCCs to use.')
	parser.add_argument('--channel', type=int, default=0, help='Channel ID # (starting from 0) of multichannel recordings to use.')
	parser.add_argument('--num_workers', type=int, default=1, help='# of workers for dataloading (>=1).')
	parser.add_argument('--omit_repetition', action='store_true', help='Omit consecutive repetition in latent representation sequences.')

	return parser.parse_args()

if __name__ == '__main__':
	args = get_args()


	model_dir = os.path.dirname(args.model_path)
	speaker_coding_path = os.path.join(model_dir, 'speaker_coding.json')
	with open(speaker_coding_path, 'r') as f:
		speaker2ix = json.load(f)

	dataset = data_utils.Dataset(
					args.input_root,
					args.director_file,
					path_col=args.path_col,
					speaker_col=args.speaker_col,
					duration_col=args.duration_col,
					channel=args.channel,
					speaker2ix=speaker2ix
					)
	fs = dataset.get_sample_freq() # Assuming all the wav files have the same fs, get the 1st file's.

	fft_frame_length = int(np.floor(args.fft_frame_length * fs))
	fft_step_size = int(np.floor(args.fft_step_size * fs))
	mfcc = sound_transforms.MFCC(
			fs,
			fft_frame_length,
			fft_step_size,
			n_mfcc=args.n_mfcc,
			max_delta_order=args.mfcc_max_delta,
			)
	mfcc_mean_path = os.path.join(model_dir, 'mfcc_mean.npy')
	mfcc_std_path = os.path.join(model_dir, 'mfcc_std.npy')
	mfcc_mean = np.load(mfcc_mean_path)
	mfcc_std = np.load(mfcc_std_path)

	in_trans = data_utils.Compose([
		mfcc,
		sound_transforms.Normalize(mfcc_mean, mfcc_std),
		torch.from_numpy,
		])
	out_trans = data_utils.Compose([
		sound_transforms.normalize_int16,
		torch.from_numpy,
		])
	f0_trans = data_utils.Compose([
		sound_transforms.normalize_int16,
		sound_transforms.F0(fs),
		# torch.from_numpy,
		])
	dataset.set_transforms(in_trans=in_trans, out_trans=out_trans, f0_trans=f0_trans)

	tester = Tester(args.model_path, device=args.device, seed=args.seed)

	tester.main(
		dataset,
		args.save_dir,
		synthesis_speaker_col=args.synthesis_speaker_col,
		batch_size=args.batch_size,
		num_workers=args.num_workers,
		omit_repetition=args.omit_repetition
		)