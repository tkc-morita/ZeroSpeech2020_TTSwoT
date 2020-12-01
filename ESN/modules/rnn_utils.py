# coding: utf-8

try:
	# for pytorch<=1.4
	from torch.nn._VF import _pad_packed_sequence
except:
	# for pytorch>=1.5
	from torch._VF import _pad_packed_sequence

def pad_flatten_sequence(flatten, batch_sizes, padding_value=0.0, batch_first=False):
	return torch.nn._VF._pad_packed_sequence(
				flatten,
				batch_sizes.cpu(),
				batch_first,
				padding_value,
				batch_sizes.size(0)
				)