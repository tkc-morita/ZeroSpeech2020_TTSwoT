# coding: utf-8

import torch
from modules.data_utils import Compose
import numpy as np
import pandas as pd
import scipy.io.wavfile as spw
from modules import data_utils, rnn_utils, sound_transforms
import learning
from parselmouth import Sound
import os, argparse, json
import matplotlib.pyplot as plt


class Tester(learning.Learner):
	def __init__(self, model_config_path, device = 'cpu'):
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

			one_hot,features = self.cognition.vae.get_map_inference(logits)
			speaker = speaker.expand(input_frames.batch_sizes[0]).to(self.device)
			features = self.cognition.decode(features, speaker, downsampled_batch_sizes, out_lengths)
			wav_padded,_ = self.articulation(features, out_lengths)
		if to_numpy:
			f0 = features[:,0,:].exp().data.cpu().numpy()
			wav_padded = wav_padded.data.cpu().numpy()
		return f0,wav_padded


	def main(
			self,
			wav,
			input_frames,
			f0_target,
			speaker,
			fs,
			save_path,
			to_f0,
			):
		wav = torch.from_numpy(wav)
		speaker = torch.tensor(speaker)
		f0_source,wav_synthesis = self.encode_and_decode(input_frames, wav, speaker, is_packed=False, to_numpy=True)
		f0_source = f0_source[0]
		wav_synthesis = wav_synthesis[0]
		f0_synthesis = to_f0(wav_synthesis)

		time = np.arange(f0_target.shape[0]) * 1000 / fs
		plt.subplots(1,1,figsize=(6.4,2))
		plt.plot(time, f0_source, color='C2', label='Source', linewidth=3)
		plt.plot(time, f0_synthesis, color='C0', label='Synthesis', linewidth=3)
		plt.plot(time, f0_target, color='C1', label='Target', linewidth=3)
		plt.xlabel('Time (ms)', fontsize='xx-large')
		plt.ylabel('F0 (Hz)', fontsize='xx-large')
		plt.yscale('log')
		plt.xticks(fontsize='x-large')
		plt.yticks(fontsize='x-large')
		plt.legend(fontsize='xx-large', loc=(1.01,0.3))
		if save_path is None:
			plt.tight_layout()
			plt.show()
		else:
			save_dir = os.path.dirname(save_path)
			if not os.path.isdir(save_dir):
				os.makedirs(save_dir)
			plt.savefig(save_path, bbox_inches='tight')


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
		return freq.numpy()

def get_args():
	parser = argparse.ArgumentParser()

	parser.add_argument('model_path', type=str, help='Path to the model checkpoint.')
	parser.add_argument('data_path', type=str, help='Path to the wav file.')
	parser.add_argument('-S', '--save_path', type=str, default=None, help='Path to the image file where results are saved.')
	parser.add_argument('-d', '--device', type=str, default='cpu', help='Computing device.')
	parser.add_argument('--fft_frame_length', type=float, default=0.025, help='FFT frame length in sec.')
	parser.add_argument('--fft_step_size', type=float, default=0.01, help='FFT step size in sec.')
	parser.add_argument('--n_mfcc', type=int, default=13, help='# of MFCCs to use.')
	parser.add_argument('--mfcc_max_delta', type=int, default=2, help='Max order of derivatives of MFCCs to use.')
	parser.add_argument('--channel', type=int, default=0, help='Channel ID # (starting from 0) of multichannel recordings to use.')

	return parser.parse_args()

if __name__ == '__main__':
	args = get_args()

	model_dir = os.path.dirname(args.model_path)
	speaker_coding_path = os.path.join(model_dir, 'speaker_coding.json')
	with open(speaker_coding_path, 'r') as f:
		speaker2ix = json.load(f)
	filename = os.path.basename(args.data_path)
	speaker = filename.split('_')[0]
	speaker = speaker2ix[speaker]

	fs,wav = spw.read(args.data_path)
	if len(wav.shape)>1:
		wav = wav[:,args.channel]

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
	mfcc = sound_transforms.MFCC(
			fs,
			fft_frame_length,
			fft_step_size,
			n_mfcc=args.n_mfcc,
			max_delta_order=args.mfcc_max_delta,
			)
	in_trans = data_utils.Compose([
			mfcc,
			sound_transforms.Normalize(mfcc_mean, mfcc_std),
			torch.from_numpy,
		])
	to_f0 = F0(fs)
	f0_trans = data_utils.Compose([
			sound_transforms.normalize_int16,
			to_f0,
		])
	input_frames = in_trans(wav)
	f0_target = f0_trans(wav)
	
	tester = Tester(args.model_path, device=args.device)

	tester.main(wav,input_frames,f0_target,speaker,fs,args.save_path,to_f0)