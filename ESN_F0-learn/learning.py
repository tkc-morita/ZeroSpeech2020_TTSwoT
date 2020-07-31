# coding: utf-8

import torch
from modules.data_utils import Compose
import numpy as np
from modules import data_utils, sound_transforms
from modules import audition, cognition, articulation, loss
from logging import getLogger,FileHandler,DEBUG,Formatter
import os, argparse, itertools, json

logger = getLogger(__name__)

def update_log_handler(file_dir):
	current_handlers=logger.handlers[:]
	for h in current_handlers:
		logger.removeHandler(h)
	log_file_path = os.path.join(file_dir,'history.log')
	if os.path.isfile(log_file_path):
		retrieval = True
	else:
		retrieval = False
	handler = FileHandler(filename=log_file_path)	#Define the handler.
	handler.setLevel(DEBUG)
	formatter = Formatter('{asctime} - {levelname} - {message}', style='{')	#Define the log format.
	handler.setFormatter(formatter)
	logger.setLevel(DEBUG)
	logger.addHandler(handler)	#Register the handler for the logger.
	if retrieval:
		logger.info("LEARNING RETRIEVED.")
	else:
		logger.info("Logger set up.")
		logger.info("PyTorch ver.: {ver}".format(ver=torch.__version__))
	return retrieval,log_file_path



class Learner(object):
	def __init__(self,
			save_dir,
			input_size,
			num_latent_categories,
			feature_dim,
			esn_hidden_size,
			articulatory_channels,
			other_hidden_size,
			num_speakers,
			total_input_frames,
			esn_leak=0.5,
			jitter_prob=0.12,
			prior_concentration=1.0,
			min_temperature=0.5,
			temperature_update_freq=1000,
			temperature_anneal_rate=1e-5,
			loss_n_ffts=[128, 512, 2048],
			loss_frame_lengths=[80, 400, 1920],
			loss_step_sizes=[40, 100, 640],
			device='cpu',
			seed=1111,
			):
		self.retrieval,self.log_file_path = update_log_handler(save_dir)
		if torch.cuda.is_available():
			if device.startswith('cuda'):
				logger.info('CUDA Version: {version}'.format(version=torch.version.cuda))
				if torch.backends.cudnn.enabled:
					logger.info('cuDNN Version: {version}'.format(version=torch.backends.cudnn.version()))
			else:
				print('CUDA is available. Restart with option -C or --cuda to activate it.')


		self.save_dir = save_dir

		self.device = torch.device(device)
		logger.info('Device: {device}'.format(device=device))

		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False

		self.seed = seed
		self.f0_loss = loss.F0Loss()

		if self.retrieval:
			self.last_iter = self.retrieve_model(device=device)
			logger.info('Model retrieved.')
		else:
			torch.manual_seed(seed)
			torch.cuda.manual_seed_all(seed) # According to the docs, "It’s safe to call this function if CUDA is not available; in that case, it is silently ignored."
			self.audition = audition.Audition(input_size, esn_hidden_size, esn_leak=esn_leak)
			self.cognition = cognition.Cognition(
				esn_hidden_size,
				other_hidden_size,
				articulatory_channels,
				num_latent_categories,
				feature_dim,
				num_speakers,
				total_input_frames=total_input_frames,
				prior_concentration=prior_concentration,
				min_temperature=min_temperature,
				temperature_update_freq=temperature_update_freq,
				temperature_anneal_rate=temperature_anneal_rate,
				jitter_prob=jitter_prob,
			)
			self.articulation = articulation.Articulation(articulatory_channels)
			self.spectral_loss = loss.SpectralLoss(
									n_ffts=loss_n_ffts,
									frame_lengths=loss_frame_lengths,
									step_sizes=loss_step_sizes,
									)
			logger.info("Data are coded by {num_latent_categories} feature vectors of {feature_dim} dimensions.".format(num_latent_categories=num_latent_categories, feature_dim=feature_dim))
			logger.info("The dimensionality of ESN's hidden states: {}".format(esn_hidden_size))
			logger.info("The dimensionality of articulatory convolution channel: {}".format(articulatory_channels))
			logger.info('The dimensionality of other hidden states: {}'.format(other_hidden_size))
			logger.info('Leak of ESN: {}'.format(esn_leak))
			logger.info('Prior concentration of Dirichlet: {}'.format(prior_concentration))
			logger.info('The temperature of Gumbel-Softmax is at least {min_temperature}, and multiplied by {temperature_anneal_rate} for every {temperature_update_freq} iterations.'.format(min_temperature=min_temperature, temperature_anneal_rate=temperature_anneal_rate, temperature_update_freq=temperature_update_freq))
			logger.info('# of speakers: {}'.format(num_speakers))
			logger.info('Trained with the specral loss with n_ffts={n_ffts}, frame_lengths={frame_lengths}, step_sizes={step_sizes}'.format(n_ffts=loss_n_ffts, frame_lengths=loss_frame_lengths, step_sizes=loss_step_sizes))
			self.parameters = lambda:itertools.chain(self.audition.parameters(), self.cognition.parameters(), self.articulation.parameters())

			self.audition.to(self.device)
			self.cognition.to(self.device)
			self.articulation.to(self.device)




	def train(self, dataloader, saving_interval, start_iter=0):
		"""
		Training phase. Updates weights.
		"""
		self.audition.train() # Turn on training mode which enables dropout.
		self.cognition.train()
		self.articulation.train()

		emission_loss = 0
		f0_loss = 0
		kl_loss = 0
		elbo = 0
		clustering_entropy = 0
		clustering_counts = 0
		num_frames = 0

		num_iterations = len(dataloader)

		for iteration,(input_, target, f0, speaker, _) in enumerate(dataloader, start_iter):
			iteration += 1 # make indexation start with 1.
			input_ = input_.to(self.device)
			target = target.to(self.device)
			f0 = f0.to(self.device)
			speaker = speaker.to(self.device)

			self.optimizer.zero_grad()
			torch.manual_seed(iteration+self.seed)
			torch.cuda.manual_seed_all(iteration+self.seed)

			audio_processed = self.audition(input_)

			target_padded,out_lengths = torch.nn.utils.rnn.pad_packed_sequence(target, batch_first=True)
			code, category_logits, kl_loss_total = self.cognition(audio_processed, speaker, out_lengths)

			output_padded,log_f0_out = self.articulation(code, out_lengths)

			emission_loss_per_sample = self.spectral_loss(output_padded, target_padded, out_lengths)
			
			f0_loss_per_sample = self.f0_loss(log_f0_out, f0.data.log())

			sample_size = out_lengths.sum()
			kl_loss_per_sample = kl_loss_total / sample_size
			loss = (emission_loss_per_sample+f0_loss_per_sample)*0.5 + kl_loss_per_sample
			elbo += -loss.item()
			loss.backward()

			# `clip_grad_norm` helps prevent the exploding gradient problem in RNNs.
			torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)

			self.optimizer.step()
			self.lr_scheduler.step()

			self.cognition.vae.increment_iter_counts()

			emission_loss += emission_loss_per_sample.item()
			f0_loss += f0_loss_per_sample.item()
			kl_loss += kl_loss_per_sample.item()
			# num_samples += sample_size
			num_frames += category_logits.size(0)
			with torch.no_grad():
				clustering_probs = torch.nn.functional.softmax(category_logits,-1)
				log_clustering_probs = torch.nn.functional.log_softmax(category_logits, -1)
				clustering_entropy += (-clustering_probs*log_clustering_probs).sum()
				clustering_counts += clustering_probs.sum(0)

			if iteration % saving_interval == 0:
				logger.info('{iteration}/{num_iterations} iterations complete.'.format(
							iteration=iteration,
							num_iterations=num_iterations,
							))
				emission_loss /= saving_interval
				f0_loss /= saving_interval
				kl_loss /= saving_interval
				elbo /= saving_interval
				clustering_perplex = (clustering_entropy / num_frames).exp().item()
				category_proportion = clustering_counts / clustering_counts.sum()
				log_category_proportion = category_proportion.where(category_proportion==0.0, category_proportion.log())
				proportion_perplex = (-category_proportion*log_category_proportion).sum().exp().item()
				pi_shape = torch.nn.functional.softmax(self.cognition.vae.posterior_shape_logits,-1)
				pi_shape_perplex = (-pi_shape*pi_shape.log()).sum().exp().item()
				logger.info('mean emission loss (per sound sample): {:5.4f}'.format(emission_loss))
				logger.info('mean F0 loss (per sound sample): {:5.4f}'.format(f0_loss))
				logger.info('mean KL (per sound sample): {:5.4f}'.format(kl_loss))
				logger.info('mean ELBO (per sound sample): {:5.4f}'.format(elbo))
				logger.info('perplexity of q(z | x): {:5.4f}'.format(clustering_perplex))
				logger.info('perplexity of sum(q(z | x)): {:5.4f}'.format(proportion_perplex))
				logger.info('perplexity of q(pi): {:5.4f}'.format(pi_shape_perplex))
				emission_loss = 0
				f0_loss = 0
				kl_loss = 0
				elbo = 0
				clustering_entropy = 0
				clustering_counts = 0
				num_frames = 0
				# num_samples = 0
				self.save_model(iteration-1)
		self.save_model(iteration-1)
		logger.info('END OF TRAINING')



	def learn(self, dataset, num_iters, batch_size, milestones=list(), pretrain_iters=0, learning_rate=4*(10**-4), gamma=0.5, num_workers=1, saving_interval=1):
		if self.retrieval:
			start_iter = self.last_iter + 1
			logger.info('To be restarted from the beginning of iter #: {iter}'.format(iter=start_iter))
			self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
			self.optimizer.load_state_dict(self.checkpoint['optimizer'])

			self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones, gamma=gamma)
			self.lr_scheduler.load_state_dict(self.checkpoint['lr_scheduler'])
		else:
			self.cognition.vae.set_pretrain_iters(pretrain_iters)
			self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
			self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones, gamma=gamma)
			logger.info("START LEARNING.")
			logger.info("# of iters: {ep}".format(ep=num_iters))
			logger.info("first {} iters are for pretraining w/o gumbel-softmax sampling.".format(pretrain_iters))
			logger.info("batch size for training data: {size}".format(size=batch_size))
			logger.info("initial learning rate: {lr}".format(lr=learning_rate))
			logger.info("Learning rate is multiplied by {gamma} at {milestones} iterations.".format(gamma=gamma, milestones=milestones))
			start_iter = 0
		dataloader = data_utils.get_data_loader(dataset, batch_size=batch_size, shuffle=True, num_iterations=num_iters, start_iter=start_iter, num_workers=num_workers, random_seed=self.seed)
		self.train(dataloader, saving_interval, start_iter=start_iter)


	def save_model(self, iteration):
		"""
		Save model config.
		Allow multiple tries to prevent immediate I/O errors.
		"""
		checkpoint = {
			'iteration':iteration,
			'audition':self.audition.state_dict(),
			'audition_init_args':self.audition.pack_init_args(),
			'cognition':self.cognition.state_dict(),
			'cognition_init_args':self.cognition.pack_init_args(),
			'articulation':self.articulation.state_dict(),
			'articulation_init_args':self.articulation.pack_init_args(),
			'spectral_loss_init_args':self.spectral_loss.pack_init_args(),
			'optimizer':self.optimizer.state_dict(),
			'lr_scheduler':self.lr_scheduler.state_dict(),
			'random_state':torch.get_rng_state(),
		}
		if torch.cuda.is_available():
			checkpoint['random_state_cuda'] = torch.cuda.get_rng_state_all()
		torch.save(checkpoint, os.path.join(self.save_dir, 'checkpoint_after-{iteration}-iters.pt'.format(iteration=iteration+1)))
		torch.save(checkpoint, os.path.join(self.save_dir, 'checkpoint.pt'))
		logger.info('Config successfully saved.')


	def retrieve_model(self, checkpoint_path = None, device='cpu'):
		if checkpoint_path is None:
			checkpoint_path = os.path.join(self.save_dir, 'checkpoint.pt')
		self.checkpoint = torch.load(checkpoint_path, map_location='cpu') # Random state needs to be loaded to CPU first even when cuda is available.

		self.audition = audition.Audition(**self.checkpoint['audition_init_args'])
		self.cognition = cognition.Cognition(**self.checkpoint['cognition_init_args'])
		self.articulation = articulation.Articulation(**self.checkpoint['articulation_init_args'])
		self.audition.load_state_dict(self.checkpoint['audition'], strict=False)
		self.cognition.load_state_dict(self.checkpoint['cognition'])
		self.articulation.load_state_dict(self.checkpoint['articulation'])
		self.audition.to(self.device)
		self.cognition.to(self.device)
		self.articulation.to(self.device)


		self.parameters = lambda:itertools.chain(self.audition.parameters(), self.cognition.parameters(), self.articulation.parameters())

		self.spectral_loss = loss.SpectralLoss(**self.checkpoint['spectral_loss_init_args'])
		
		try:
			torch.set_rng_state(self.checkpoint['random_state'])
		except RuntimeError:
			msg = 'Failed to retrieve random_state.'
			try:
				logger.warning(msg)
			except NameError:
				print(msg)
		if device=='cuda':
			torch.cuda.set_rng_state_all(self.checkpoint['random_state_cuda'])
		return self.checkpoint['iteration']



def get_args():
	parser = argparse.ArgumentParser()

	parser.add_argument('input_root', type=str, help='Path to the root directory under which inputs are located.')
	parser.add_argument('director_file', type=str, help='Path to the csv file containing relative paths to the wav files and speaker IDs.')
	parser.add_argument('--path_col', type=str, default='filename', help='Name of the column containing relative paths to the wav files.')
	parser.add_argument('--speaker_col', type=str, default='speaker', help='Name of the column containing speaker IDs.')
	parser.add_argument('--duration_col', type=str, default='duration', help='Name of the column containing the file duration (in sec).')
	parser.add_argument('-S', '--save_root', type=str, default=None, help='Path to the directory where results are saved.')
	parser.add_argument('-j', '--job_id', type=str, default='NO_JOB_ID', help='Job ID. For users of computing clusters.')
	parser.add_argument('-s', '--seed', type=int, default=1111, help='random seed')
	parser.add_argument('-d', '--device', type=str, default='cpu', help='Computing device.')
	parser.add_argument('-i', '--iterations', type=int, default=900000, help='# of iterations to train the model.')
	parser.add_argument('--pretrain_iters', type=int, default=1000, help='# of initial iterations to pretrain the model w/o gumbel-softmax sampling.')
	parser.add_argument('-b', '--batch_size', type=int, default=64, help='Batch size for training.')
	parser.add_argument('-l', '--learning_rate', type=float, default=4*(10**-4), help='Initial learning rate.')
	parser.add_argument('--milestones', type=int, nargs='*', default=[], help='Iterations at which the learning rate is reduced.')
	parser.add_argument('--gamma', type=float, default=0.5, help='Multiplier to the learning rate.')
	parser.add_argument('-K', '--num_feature_categories', type=int, default=128, help='# of possible discrete values token on by latent features into which data are encoded.')
	parser.add_argument('-f', '--feature_dim', type=int, default=128, help='# of dimensions of features into which the discrete feature are linear-transformed.')
	# parser.add_argument('--output_levels', type=int, default=256, help='# of levels of output values.')
	parser.add_argument('--esn_hidden_size', type=int, default=2048, help='Dimensionality of the hidden states of ESN.')
	parser.add_argument('--articulatory_channels', type=int, default=64, help='Dimensionality of the convolutional layers.')
	parser.add_argument('--other_hidden_size', type=int, default=128, help='Dimensionality of the other hidden states (e.g. MLP).')
	parser.add_argument('--jitter_prob', type=float, default=.12, help='Probability of jitter in the decoder.')
	parser.add_argument('--esn_leak', type=float, default=0.5, help='Leak rate of ESN.')
	parser.add_argument('--fft_frame_length', type=float, default=0.025, help='FFT frame length in sec.')
	parser.add_argument('--fft_step_size', type=float, default=0.01, help='FFT step size in sec.')
	parser.add_argument('--n_mfcc', type=int, default=13, help='# of MFCCs to use.')
	parser.add_argument('--mfcc_max_delta', type=int, default=2, help='Max order of derivatives of MFCCs to use.')
	parser.add_argument('--channel', type=int, default=0, help='Channel ID # (starting from 0) of multichannel recordings to use.')
	parser.add_argument('--max_duration', type=float, default=3.0, help='Max duration of input wave (in sec).')
	parser.add_argument('--prior_concentration', type=float, default=1.0, help='Concentration of the Dirichlet prior on the probability of the discrete feature.')
	parser.add_argument('--min_temperature', type=float, default=0.5, help='Minimum temperature of gumbel-softmax sampling.')
	parser.add_argument('--temperature_update_freq', type=int, default=1000, help='Frequency of temperature annealing.')
	parser.add_argument('--temperature_anneal_rate', type=float, default=1e-5, help='Rate of temperature annealing.')
	parser.add_argument('--num_workers', type=int, default=1, help='# of workers for dataloading (>=1).')
	parser.add_argument('--saving_interval', type=int, default=200, help='# of iterations in which model parameters are saved once.')

	return parser.parse_args()


def get_save_dir(save_root, job_id_str):
	save_dir = os.path.join(
					save_root,
					job_id_str # + '_START-AT-' + datetime.datetime.now().strftime('%y-%m-%d-%H-%M-%S-%f')
				)
	if not os.path.isdir(save_dir):
		os.makedirs(save_dir)
	return save_dir

if __name__ == '__main__':
	args = get_args()

	save_root = args.save_root
	if save_root is None:
		save_root = args.input_root
	save_dir = get_save_dir(save_root, args.job_id)

	speaker_coding_path = os.path.join(save_dir, 'speaker_coding.json')
	if os.path.isfile(speaker_coding_path):
		with open(speaker_coding_path, 'r') as f:
			speaker2ix = json.load(f)
	else:
		speaker2ix = None

	dataset = data_utils.Dataset(
					args.input_root,
					args.director_file,
					path_col=args.path_col,
					speaker_col=args.speaker_col,
					duration_col=args.duration_col,
					channel=args.channel,
					max_duration=args.max_duration,
					speaker2ix=speaker2ix
					)
	fs = dataset.get_sample_freq() # Assuming all the wav files have the same fs, get the 1st file's.
	num_speakers = dataset.get_num_speakers()
	if num_speakers>0 and speaker2ix is None:
		with open(speaker_coding_path, 'w') as f:
			json.dump(dataset.speaker2ix, f)

	fft_frame_length = int(np.floor(args.fft_frame_length * fs))
	fft_step_size = int(np.floor(args.fft_step_size * fs))
	mfcc = sound_transforms.MFCC(
			fs,
			fft_frame_length,
			fft_step_size,
			n_mfcc=args.n_mfcc,
			max_delta_order=args.mfcc_max_delta,
			)
	mfcc_mean_path = os.path.join(save_dir, 'mfcc_mean.npy')
	mfcc_std_path = os.path.join(save_dir, 'mfcc_std.npy')
	if not (os.path.isfile(mfcc_mean_path) and os.path.isfile(mfcc_std_path)):
		mfcc_mean,mfcc_std = dataset.get_wave_stats(transform=mfcc)
		np.save(mfcc_mean_path, mfcc_mean)
		np.save(mfcc_std_path, mfcc_std)
	else:
		mfcc_mean = np.load(mfcc_mean_path)
		mfcc_std = np.load(mfcc_std_path)
	total_input_frames = np.floor(dataset.get_num_samples() / fft_step_size + 1).astype(np.float32).sum()

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

	# Get a model.
	learner = Learner(
			save_dir,
			3*args.n_mfcc,
			args.num_feature_categories,
			args.feature_dim,
			args.esn_hidden_size,
			args.articulatory_channels,
			args.other_hidden_size,
			num_speakers,
			total_input_frames,
			esn_leak=args.esn_leak,
			jitter_prob=args.jitter_prob,
			prior_concentration=args.prior_concentration,
			min_temperature=args.min_temperature,
			temperature_update_freq=args.temperature_update_freq,
			temperature_anneal_rate=args.temperature_anneal_rate,
			device = args.device,
			seed = args.seed,
		)

	logger.info("Sampling frequency of data: {fs}".format(fs=fs))
	logger.info("{mfcc} MFCCs and their 1st- to {max_delta}-th derivatives are used as the input.".format(mfcc=args.n_mfcc, max_delta=args.mfcc_max_delta))
	logger.info("Use the default FFT window type (hann).")
	logger.info("Input FFT frame lengths: {fft_frame_length_in_sec} sec".format(fft_frame_length_in_sec=args.fft_frame_length))
	logger.info("Input FFT step size: {fft_step_size_in_sec} sec".format(fft_step_size_in_sec=args.fft_step_size))




	# Train the model.
	learner.learn(
			dataset,
			args.iterations,
			args.batch_size,
			milestones=args.milestones,
			pretrain_iters=args.pretrain_iters,
			learning_rate=args.learning_rate,
			gamma=args.gamma,
			num_workers=args.num_workers,
			saving_interval = args.saving_interval,
			)