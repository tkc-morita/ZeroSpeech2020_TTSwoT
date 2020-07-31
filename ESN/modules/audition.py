# coding: utf-8

import torch
from .rnn_utils import pad_flatten_sequence

class Audition(torch.nn.Module):
	"""
	Echo-State Network
	"""
	def __init__(self, input_size, hidden_size, esn_leak=0.5):
		super(Audition, self).__init__()
		self.rnn = ESN(input_size, hidden_size, 1, leak=esn_leak)

	def forward(self, packed_seq):
		packed_out,_ = self.rnn(packed_seq)
		return packed_out

	def pack_init_args(self):
		args = {
			"input_size":self.rnn.input_size,
			"hidden_size":self.rnn.hidden_size,
			"esn_leak":self.rnn.leak,
		}
		return args

class ESN(torch.jit.ScriptModule):
	__constants__ = ['leak','hidden_size']
# class ESN(torch.nn.Module):
	def __init__(self, input_size, hidden_size, num_layers, dropout=0.0, bidirectional=False, batch_first=True, bias=False, leak=1.0, q=0.95, sparsity=0.1):
		super(ESN, self).__init__()
		self.input_size = input_size
		self.mode = 'ESN'
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.bidirectional = bidirectional
		self.dropout = dropout
		self.bias = bias
		self.batch_first = batch_first
		self.leak = leak

		internal_input_size = hidden_size
		if bidirectional:
			internal_input_size *= 2
			self._init_parameters(q, internal_input_size, sparsity, '_reverse')
		# 0th layer
		self._init_parameters(q, internal_input_size, sparsity)

		self.drop = torch.nn.Dropout(p=dropout)
		self.activation = torch.nn.Tanh()

	def _init_parameters(self, q, internal_input_size, sparsity, parname_suffix=''):
		# input2hidden matrix.
		# Either either -3.0/input_size or 3.0/input_size. 
		# tanh(x) almost ceils and floors at x=3 and -3, 
		# so the sum should stay in the range most of the time.
		# 0th layer
		import scipy.stats as spstats
		input_quantile = spstats.binom.ppf(q, self.input_size, 0.5).astype('float32')
		self.register_parameter(
				'weight_ih_l0'+parname_suffix,
				torch.nn.Parameter(
					torch.randint(2, (self.hidden_size, self.input_size), dtype=torch.float32)
					* (6.0 / input_quantile)
					- (3.0 / input_quantile),
					requires_grad=False)
				)

		# Intermediate layers.
		internal_quantile = spstats.binom.ppf(q, internal_input_size, 0.5).astype('float32')
		[self.register_parameter(
				'weight_ih_l{l}'.format(l=l)+parname_suffix,
				torch.nn.Parameter(
					torch.randint(2, (self.hidden_size, internal_input_size), dtype=torch.float32)
					* (6.0 / internal_quantile)
					- (3.0 / internal_quantile),
					requires_grad=False)
				) for l in range(1,self.num_layers)]

		# hidden2hidden matrix.
		for l in range(self.num_layers):
			self.register_parameter(
					'weight_hh_l{l}'.format(l=l)+parname_suffix,
					torch.nn.Parameter(
						torch.randn(self.hidden_size, self.hidden_size),
						requires_grad=False)
					)
			weight_hh = getattr(self, 'weight_hh_l{l}'.format(l=l)+parname_suffix)
			weight_hh.data = torch.nn.Dropout(p=1.0-sparsity)(weight_hh.data)
			eig_val,_ = torch.eig(weight_hh)
			weight_hh /= (eig_val.pow(2).sum(-1)).max().sqrt() / 0.99 # Adjust by the spectral radius.


	def forward(self, packed_input, h_0=None):
		flatten_input = packed_input.data
		last_hidden = []
		if h_0 is None:
			h_0 = self.init_hidden(packed_input.batch_sizes[0])
			if self.bidirectional:
				h_0 = torch.cat([
						h_0,
						self.init_hidden(packed_input.batch_sizes[0])
				], dim=0)
			h_0 = h_0.to(packed_input.data.device)
		h_0 = h_0.view(-1, self.num_layers, h_0.size(1), h_0.size(2))
		for l in range(self.num_layers):
			flatten_hidden, last_hidden_l = self._forward_per_layer(flatten_input, packed_input.batch_sizes, getattr(self, 'weight_ih_l{l}'.format(l=l)), getattr(self, 'weight_hh_l{l}'.format(l=l)), h_0[0,l])
			last_hidden += [last_hidden_l]
			if self.bidirectional:
				flatten_hidden_back, last_hidden_l_back = self._forward_per_layer_backward(flatten_input, packed_input.batch_sizes, getattr(self, 'weight_ih_l{l}_reverse'.format(l=l)), getattr(self, 'weight_hh_l{l}_reverse'.format(l=l)), h_0[1,l])
				flatten_hidden = torch.cat([flatten_hidden, flatten_hidden_back], dim=-1)
				last_hidden += [last_hidden_l_back]
			flatten_input = self.drop(flatten_hidden)
		last_hidden = torch.cat(last_hidden, dim=0)
		padded_out,lengths = pad_flatten_sequence(flatten_hidden, packed_input.batch_sizes)
		packed_out = torch.nn.utils.rnn.pack_padded_sequence(padded_out, lengths)
		return packed_out, last_hidden

	@torch.jit.script_method
	def _forward_per_layer(self, flatten_input, batch_sizes, weight_ih, weight_hh, h_0):
		# input2hidden
		input2hidden_transposed = weight_ih.mm(flatten_input.t())
		# hidden2hidden
		hidden_transposed = h_0.t()
		# flatten_hidden_transposed = torch.tensor([]).to(input2hidden_transposed.device)
		# last_hidden_transposed = torch.tensor([]).to(input2hidden_transposed.device)
		flatten_hidden_transposed = []
		last_hidden_transposed = []
		next_batch_sizes = torch.cat([batch_sizes[1:], torch.tensor([0])]).to(input2hidden_transposed.device)
		for t in range(len(batch_sizes)):
			bs,next_bs = batch_sizes[t], next_batch_sizes[t]
		# for bs,next_bs in zip(batch_sizes, next_batch_sizes):
			hidden_transposed = hidden_transposed[...,:bs]
			input2hidden_at_t_transposed = input2hidden_transposed[...,:bs]
			hidden2hidden_at_t_transposed = weight_hh.to_sparse().mm(hidden_transposed)
			hidden_transposed = (1.0 - self.leak) * hidden_transposed + self.leak * self.activation(input2hidden_at_t_transposed + hidden2hidden_at_t_transposed)
			# flatten_hidden_transposed = torch.cat([flatten_hidden_transposed,hidden_transposed], dim=-1)
			flatten_hidden_transposed += [hidden_transposed]
			input2hidden_transposed = input2hidden_transposed[...,bs:]
			# last_hidden_transposed = torch.cat([hidden_transposed[...,next_bs:], last_hidden_transposed], dim=-1)
			last_hidden_transposed = [hidden_transposed[...,next_bs:]] + last_hidden_transposed
		flatten_hidden_transposed = torch.cat(flatten_hidden_transposed, dim=-1)
		last_hidden_transposed = torch.cat(last_hidden_transposed, dim=-1)
		return flatten_hidden_transposed.t(), last_hidden_transposed.t().view(1,last_hidden_transposed.size(1),last_hidden_transposed.size(0))

	def _forward_per_layer_backward(self, flatten_input, batch_sizes, weight_ih, weight_hh, h_0):
		# input2hidden
		input2hidden_transposed = weight_ih.mm(flatten_input.t())
		# hidden2hidden
		init_hidden_fullsize_transposed = h_0.t()
		hidden_transposed = init_hidden_fullsize_transposed[:,:batch_sizes[-1]]
		# flatten_hidden_transposed = torch.tensor([]).to(input2hidden_transposed.device)
		flatten_hidden_transposed = []
		next_batch_sizes = torch.cat([batch_sizes[:-1].flip(0), torch.tensor([0])]).to(input2hidden_transposed.device)
		reversed_batch_sizes = batch_sizes.flip(0)
		for t in range(len(batch_sizes)):
			bs,next_bs = reversed_batch_sizes[t], next_batch_sizes[t]
		# for bs,next_bs in zip(, next_batch_sizes):
			input2hidden_at_t_transposed = input2hidden_transposed[...,-bs:]
			hidden2hidden_at_t_transposed = weight_hh.to_sparse().mm(hidden_transposed)
			hidden_transposed = (1.0 - self.leak) * hidden_transposed + self.leak * self.activation(input2hidden_at_t_transposed + hidden2hidden_at_t_transposed)
			# flatten_hidden_transposed = torch.cat([hidden_transposed,flatten_hidden_transposed], dim=-1)
			flatten_hidden_transposed = [hidden_transposed] + flatten_hidden_transposed
			input2hidden_transposed = input2hidden_transposed[...,:-bs]
			hidden_transposed = torch.cat([hidden_transposed,init_hidden_fullsize_transposed[:,bs:next_bs]], dim=-1)
		flatten_hidden_transposed = torch.cat(flatten_hidden_transposed, dim=-1)
		return flatten_hidden_transposed.t(), hidden_transposed.t().view(1,hidden_transposed.size(1),hidden_transposed.size(0))

	def init_hidden(self, batch_size):
		return torch.zeros((self.num_layers, batch_size, self.hidden_size), requires_grad=False)