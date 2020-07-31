# coding: utf-8

import torch
import math
from .rnn_utils import pad_flatten_sequence

class Cognition(torch.nn.Module):
	"""
	Downsample -> Discrete VAE -> Time Jitter -> Speaker Coloring -> Upsample
	"""
	def __init__(
		self,
		input_size,
		hidden_size,
		output_size,
		num_categories,
		feature_dim,
		num_speakers,
		*args,
		total_input_frames=None,
		data_size=None,
		prior_concentration=1.0,
		min_temperature=0.5,
		temperature_update_freq=1000,
		temperature_anneal_rate=1e-5,
		iter_counts=0,
		pretrain_iters=0,
		jitter_prob=0.12,
		**kwargs
		):
		super(Cognition, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.downsample = Downsample()
		if data_size is None:
			data_size = total_input_frames//self.downsample.downsample_rate
		self.vae = ABCD_VAE(
					input_size,
					num_categories,
					feature_dim,
					hidden_size,
					data_size,
					prior_concentration=prior_concentration,
					min_temperature=min_temperature,
					temperature_update_freq=temperature_update_freq,
					temperature_anneal_rate=temperature_anneal_rate,
					iter_counts=iter_counts,
					pretrain_iters=pretrain_iters,
					)
		self.time_jitter = TimeJitter(p=jitter_prob)
		self.speaker_coloring = SpeakerColoring(num_speakers, feature_dim, hidden_size)
		self.upsample = Upsample(2*hidden_size,hidden_size,output_size)

	def forward(self, packed_audio, speaker, output_lengths):
		flatten_code,category_logits,downsampled_batch_sizes,kl_loss_total = self.encode(packed_audio)

		flatten_code = self.time_jitter(flatten_code, downsampled_batch_sizes)

		padded_code = self.decode(flatten_code, speaker, downsampled_batch_sizes, output_lengths)
		return padded_code, category_logits, kl_loss_total

	def encode(self, packed_audio):
		packed_audio = self.downsample(packed_audio)
		category_logits = self.vae(packed_audio.data)
		if self.training:
			flatten_code = self.vae.sample(category_logits, no_sample=self.vae.is_pretrain())
			counts = self.count_consecutive_repetitions(flatten_code, packed_audio.batch_sizes)
			kl_loss_total = self.vae.kl_divergence(category_logits, counts.float().pow(-1))
			return flatten_code,category_logits,packed_audio.batch_sizes,kl_loss_total
		return category_logits,packed_audio.batch_sizes

	def decode(self, flatten_code, speaker, downsampled_batch_sizes, output_lengths):
		packed_code = self.speaker_coloring(flatten_code, speaker, downsampled_batch_sizes)
		padded_code,_ = torch.nn.utils.rnn.pad_packed_sequence(packed_code, batch_first=True)
		padded_code = padded_code.transpose(1,2).contiguous()

		padded_code = self.upsample(padded_code, output_lengths[0].item())
		return padded_code

	def count_consecutive_repetitions(self, flatten_code, batch_sizes):
		map_categories = flatten_code.argmax(dim=-1)
		padded_cat,lengths = pad_flatten_sequence(map_categories, batch_sizes, batch_first=True)
		out = []
		for seq in padded_cat:
			_,inverse_indices,counts = seq.unique_consecutive(return_inverse=True, return_counts=True)
			out.append(counts[inverse_indices])
		out = torch.stack(out, dim=0)
		out = torch.nn.utils.rnn.pack_padded_sequence(out, lengths, batch_first=True).data
		return out


	def pack_init_args(self):
		args = self.vae.pack_init_args()
		args['input_size'] = self.input_size
		args['hidden_size'] = self.hidden_size
		args['output_size'] = self.output_size
		args['num_speakers'] = self.speaker_coloring.num_speakers
		args['jitter_prob'] = self.time_jitter.p
		return args

class TimeJitter(torch.nn.Module):
	"""
	Randomly replace adjacent frames.
	"""
	def __init__(self, p=0.12):
		super(TimeJitter, self).__init__()
		self.p = p

	def forward(self, x, batch_sizes):
		if self.training:
			samples = torch.rand_like(x)
			t_minus_1,t_plus_1 = self._get_shifted_idxs(batch_sizes)
			x = x[t_minus_1].where(samples<self.p*.5, x[t_plus_1].where(samples>=1-self.p*.5, x))
		return x

	def _get_shifted_idxs(self, batch_sizes):
		idxs = [list(range(batch_sizes[0]))] + [list(range(start,start+bs)) for bs,start in zip(batch_sizes[1:],batch_sizes.cumsum(0)[:-1])]
		t_minus_1 = []
		t_plus_1 = []
		for t in range(batch_sizes.size(0)):
			tm1_or_0 = max(t-1,0)
			t_minus_1 += idxs[tm1_or_0][:batch_sizes[t]]
			tp1_or_T = min(t+1,batch_sizes.size(0)-1)
			t_plus_1 += idxs[tp1_or_T] + idxs[t][batch_sizes[tp1_or_T]:]
		return t_minus_1,t_plus_1

class SpeakerColoring(torch.nn.Module):
	def __init__(self, num_speakers, input_size, hidden_size, num_layers=3):
		super(SpeakerColoring, self).__init__()
		self.num_speakers = num_speakers
		self.embed_speaker_for_input = torch.nn.Embedding(num_speakers, input_size)
		self.embed_speaker_for_init = torch.nn.Embedding(num_speakers, 2*hidden_size*num_layers)
		self.rnn = torch.nn.LSTM(input_size*2, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)
		
	def forward(self, flatten_code, speaker, code_batch_sizes):
		speaker_for_input = self.embed_speaker_for_input(speaker)
		speaker_for_input = torch.cat([speaker_for_input[:bs] for bs in code_batch_sizes], dim=0)
		flatten_code = torch.cat([flatten_code, speaker_for_input], dim=-1)
		padded_code,code_lengths = pad_flatten_sequence(flatten_code, code_batch_sizes)
		packed_code = torch.nn.utils.rnn.pack_padded_sequence(padded_code, code_lengths)

		init_hidden = self.embed_speaker_for_init(speaker)
		init_hidden = init_hidden.view(speaker.size(0), 2*self.rnn.num_layers, -1).transpose(0,1).contiguous()
		init_hidden = (init_hidden,torch.zeros_like(init_hidden))

		packed_code,_ = self.rnn(packed_code,init_hidden)
		return packed_code

class Downsample(torch.nn.Module):
	def __init__(self, downsample_rate=2):
		super(Downsample, self).__init__()
		self.downsample_rate = downsample_rate

	def forward(self, packed):
		padded,lengths = torch.nn.utils.rnn.pad_packed_sequence(packed, batch_first=True)
		padded = padded[:,self.downsample_rate-1::self.downsample_rate,:].contiguous()
		lengths = lengths // self.downsample_rate
		packed = torch.nn.utils.rnn.pack_padded_sequence(padded, lengths, batch_first=True)
		return packed

class Upsample(torch.nn.Module):
	def __init__(self, in_channels, hidden_channels, out_channels,strides=[5,4,4,4], kernels=None):
		super(Upsample, self).__init__()
		if kernels is None:
			kernels = [s**2 for s in strides]
		paddings = [(k-s)//2 for s,k in zip(strides,kernels)]
		transforms = []
		ic = in_channels
		oc = hidden_channels
		for k,s,p in zip(kernels[:-1],strides[:-1],paddings[:-1]):
			transforms.append(
				torch.nn.ConvTranspose1d(
					ic,oc,
					kernel_size=k,stride=s,
					padding=p)
				)
			transforms.append(torch.nn.LeakyReLU())
			ic = oc
		transforms.append(
			torch.nn.ConvTranspose1d(
				ic,out_channels,
				kernel_size=kernels[-1],stride=strides[-1],
				padding=paddings[-1])
			)
		self.transforms = torch.nn.Sequential(*transforms)

	def forward(self, x, output_size=None):
		x = self.transforms(x)
		x = torch.nn.functional.interpolate(x, size=output_size, mode='nearest') # Currently (Pytorch ver. 1.4 stable), this interpolation is somehow necessary to prevent "RuntimeError: grad_columns needs to be contiguous" even when the length of x is already as desired.
		return x


class ABCD_VAE(torch.nn.Module):
	"""
	"A"ttention-"B"ased "C"ategorical sampling with the "D"irichlet prior.
	"""
	def __init__(self,
			input_size,
			num_categories,
			feature_dim,
			mlp_hidden_size,
			data_size,
			prior_concentration=1.0,
			min_temperature=0.5,
			temperature_update_freq=1000,
			temperature_anneal_rate=1e-5,
			iter_counts=0,
			pretrain_iters=0
			):
		super(ABCD_VAE, self).__init__()
		self.num_categories = num_categories
		self.to_code_like = MLP(input_size, mlp_hidden_size, feature_dim)
		self.min_temperature = min_temperature
		self.iter_counts = iter_counts
		self.pretrain_iters = pretrain_iters
		self.temperature_update_freq = temperature_update_freq
		self.temperature_anneal_rate = temperature_anneal_rate
		self.data_size = data_size
		self.update_temperature((self.iter_counts//self.temperature_update_freq)*self.temperature_update_freq)
		self.register_buffer(
			"prior_concentration",
			torch.tensor(prior_concentration)
		)
		self.register_parameter(
			"posterior_shape_logits",
			torch.nn.Parameter(
				torch.randn(num_categories),
				requires_grad=True
				)
			)
		self.register_parameter(
			"codebook",
			torch.nn.Parameter(
				torch.randn((feature_dim,num_categories)),
				requires_grad=True
				)
			)

	def forward(self, x):
		"""
		Convert vectors into the unnormalized log posterior distribution of the categorical variable, z.
		That is, q(z | x), where x is the data.
		The logits are given by the unnormalized cosine similarity b/w the transformed x and the codebook.
		(Or Attention weights to the codebook.)
		"""
		x = self.to_code_like(x)
		logits = torch.mm(x, self.codebook) / math.sqrt(x.size(-1))
		return logits

	def sample(self, logits, no_sample=False):
		"""
		Discrete sample z is approximated by a prob vector y from the Gumble-Softmax distribution.
		y is then matrix-multiplied with the codebook;
		That is, the output is the weighted sum of the codebook (cf. VQ-VAE).

		If no_sample=True, the probability of z is used instead of y, w/o sampling.
		This might be useful for pretraining.
		"""
		if no_sample:
			sample = torch.nn.functional.softmax(logits, -1)
		else:
			sample = torch.nn.functional.gumbel_softmax(logits,tau=self.temperature,dim=-1)
		features = torch.mm(sample, self.codebook.t())
		return features

	def get_map_inference(self, logits):
		map_categories = logits.argmax(-1)
		one_hot = torch.nn.functional.one_hot(map_categories, num_classes=self.num_categories)
		features = torch.mm(one_hot.float(), self.codebook.t())
		return one_hot, features

	def kl_divergence(self, logits, data_weights=1.0):
		"""
		KL(q(pi)*q(z|x) || p(pi)*p(z|pi))
		"""
		if isinstance(data_weights, torch.Tensor) and data_weights.ndim==1:
			data_weights = data_weights[:,None]
		# Eq[log q(pi)]
		# Optimal pars of the posterior Dirichlet is
		# a distribution of the total data size + the prior pseudo counts.
		posterior_shape = torch.nn.functional.softmax(self.posterior_shape_logits, -1)
		posterior_concentration = posterior_shape*self.data_size + self.prior_concentration
		sum_posterior_concentration = posterior_concentration.sum()
		expected_log_pi = posterior_concentration.digamma() - sum_posterior_concentration.digamma()
		Eq_log_q_pi = sum_posterior_concentration.lgamma() \
						- posterior_concentration.lgamma().sum() \
						+ ((posterior_concentration-1.0)*expected_log_pi).sum()

		# Eq[log p(pi)]
		Eq_log_p_pi = (self.prior_concentration*self.num_categories).lgamma() \
						- self.prior_concentration.lgamma() * self.num_categories \
						+ ((self.prior_concentration-1.0)*expected_log_pi).sum()
		
		# Eq[log q(z|x)]
		q_z = torch.nn.functional.softmax(logits, -1)
		log_q_z = torch.nn.functional.log_softmax(logits, -1) # More stable than q_z.log()
		Eq_log_q_z = (q_z * log_q_z * data_weights).sum()

		# Eq[log p(z|pi)]
		Eq_log_p_z = (q_z * expected_log_pi[None,:] * data_weights).sum()

		batch_size = logits.size(0)

		kl = (Eq_log_q_pi - Eq_log_p_pi) * (batch_size / self.data_size) + Eq_log_q_z - Eq_log_p_z
		return kl

	def log_pmf(self, targets, logits):
		return torch.nn.functional.cross_entropy(logits, targets, reduction='sum')

	def increment_iter_counts(self):
		self.iter_counts += 1
		if max(1, self.iter_counts-self.pretrain_iters) % self.temperature_update_freq == 0:
			self.update_temperature()

	def update_temperature(self, steps=None):
		if steps is None:
			steps = self.iter_counts
		steps -= self.pretrain_iters
		self.temperature = min(
							self.min_temperature,
							math.exp(-self.temperature_anneal_rate*steps)
							)

	def is_pretrain(self):
		return self.iter_counts<self.pretrain_iters

	def set_pretrain_iters(self, pretrain_iters):
		self.pretrain_iters = pretrain_iters

	def pack_init_args(self):
		parameters = {
			"input_size": self.to_code_like.input_size,
			"num_categories": self.num_categories,
			"mlp_hidden_size": self.to_code_like.hidden_size,
			"feature_dim": self.to_code_like.output_size,
			"data_size":self.data_size,
			"prior_concentration": self.prior_concentration.item(),
			"min_temperature": self.min_temperature,
			"temperature_update_freq": self.temperature_update_freq,
			"temperature_anneal_rate": self.temperature_anneal_rate,
			"iter_counts":self.iter_counts,
			"pretrain_iters":self.pretrain_iters
		}
		return parameters


class MLP(torch.jit.ScriptModule):
	"""
	Multi-Layer Perceptron.
	"""
	def __init__(self, input_size, hidden_size, output_size):
		super(MLP, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.whole_network = torch.nn.Sequential(
			torch.nn.Linear(input_size, hidden_size),
			torch.nn.LeakyReLU(),
			torch.nn.Linear(hidden_size, output_size)
			)

	@torch.jit.script_method
	def forward(self, batched_input):
		return self.whole_network(batched_input)