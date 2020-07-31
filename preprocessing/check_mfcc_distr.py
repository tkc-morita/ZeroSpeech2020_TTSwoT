# coding: utf-8

import numpy as np
import pandas as pd
import librosa
import scipy.io.wavfile as spw
import os.path, argparse, glob

def get_mfccs(dir_path, win_length_in_sec, hop_length_in_sec, n_mfcc=13):
	spectra = []
	mfccs = []
	first_ders = []
	second_ders = []
	for path in glob.glob(os.path.join(dir_path, '*.wav')):
		fs, wav = spw.read(path)
		# wav = normalize_int16(wav)
		wav = wav.astype(np.float32)
		win_length = int(win_length_in_sec*fs)
		hop_length = int(hop_length_in_sec*fs)
		stft = librosa.core.stft(wav, n_fft=win_length, hop_length=hop_length)
		spectra.append(np.absolute(stft))
		mfcc = librosa.feature.mfcc(wav, sr=fs, n_mfcc=n_mfcc, n_fft=win_length, hop_length=hop_length)
		mfccs.append(mfcc)
		first_ders.append(librosa.feature.delta(mfcc, order=1))
		second_ders.append(librosa.feature.delta(mfcc, order=2))
	spectra = np.concatenate(spectra, axis=-1)
	mfccs = np.concatenate(mfccs, axis=-1)
	first_ders = np.concatenate(first_ders, axis=-1)
	second_ders = np.concatenate(second_ders, axis=-1)
	return spectra, mfccs, first_ders, second_ders

def normalize_int16(wav):
	if isinstance(wav, np.ndarray):
		wav = wav.astype(np.float32)
	else: # torch.Tensor
		wav = wav.float()
	wav /= 2**15
	return wav

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('input_root', type=str, help='Path to the root directory under which inputs are located.')
	parser.add_argument('--fft_frame_length', type=float, default=0.025, help='FFT frame length in sec.')
	parser.add_argument('--fft_step_size', type=float, default=0.01, help='FFT step size in sec.')
	parser.add_argument('--n_mfcc', type=int, default=13, help='# of MFCCs to use.')
	args = parser.parse_args()

	spectra, mfccs, first_ders, second_ders = get_mfccs(args.input_root, args.fft_frame_length, args.fft_step_size, args.n_mfcc)

	mean_and_stds = []
	for name,array in zip(['spectra', '0th_der', '1st_der', '2nd_der'],[spectra, mfccs, first_ders, second_ders]):
		print(name)
		df = pd.DataFrame(np.transpose(array), columns=['dim{:02}'.format(dim) for dim in range(array.shape[0])])
		df.loc[:,'data_ix'] = df.index
		df = df.melt(id_vars=['data_ix'], var_name='dim', value_name='value')
		gp = df.groupby('dim')
		print(gp.value.describe())