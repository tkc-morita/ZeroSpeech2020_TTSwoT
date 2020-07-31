# coding: utf-8

import torch
from collections import OrderedDict
from .rnn_utils import pad_flatten_sequence

class Audition(torch.nn.Module):
	"""
	Convolutional encoder proposed by Chorowski et al. (2019).
	"""
	def __init__(self, in_channel, out_channel):
		super(Audition, self).__init__()
		self.in_channel = in_channel
		self.out_channel = out_channel
		self.network = torch.nn.Sequential(OrderedDict([
			('conv1',ConvSubLayer(in_channel, out_channel)),
			('conv2',ConvSubLayer(out_channel, out_channel)),
			('strided_conv',ConvSubLayer(out_channel, out_channel, filter_length=4, stride=2)),
			('conv3',ConvSubLayer(out_channel, out_channel)),
			('conv4',ConvSubLayer(out_channel, out_channel)),
			('ff1',ConvSubLayer(out_channel, out_channel, filter_length=1)),
			('ff2',ConvSubLayer(out_channel, out_channel, filter_length=1)),
			('ff3',ConvSubLayer(out_channel, out_channel, filter_length=1)),
			('ff4',ConvSubLayer(out_channel, out_channel, filter_length=1)),
		]))

	def forward(self, packed):
		padded,lengths = torch.nn.utils.rnn.pad_packed_sequence(packed, batch_first=True)
		padded = padded.transpose(1,2).contiguous()
		padded = self.network(padded)
		padded = padded.transpose(1,2).contiguous()
		lengths = lengths // 2
		packed = torch.nn.utils.rnn.pack_padded_sequence(padded, lengths, batch_first=True)
		return packed

	def pack_init_args(self):
		args = {
			"in_channel":self.in_channel,
			"out_channel":self.out_channel,
		}
		return args


class ConvSubLayer(torch.nn.Module):
	def __init__(self, in_channel, out_channel, filter_length=3, stride=1):
		super(ConvSubLayer, self).__init__()
		self.conv = torch.nn.Conv1d(in_channel, out_channel, kernel_size=filter_length, stride=stride, padding=(filter_length-stride)//2, padding_mode='zeros')
		self.relu = torch.nn.ReLU()
		self.residual_connection = (in_channel==out_channel) and (stride==1)

	def forward(self, input_):
		output = self.conv(input_)
		output = self.relu(output)
		# output = torch.nn.functional.pad(output, (self.left_padding, self.right_padding), mode='constant', value=0.0)
		if self.residual_connection:
			output = output + input_
		return output