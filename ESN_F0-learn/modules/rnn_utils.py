# coding: utf-8

import torch

def pad_flatten_sequence(flatten, batch_sizes, padding_value=0.0, batch_first=False):
	return torch.nn._VF._pad_packed_sequence(
				flatten,
				batch_sizes.cpu(),
				batch_first,
				padding_value,
				batch_sizes.size(0)
				)