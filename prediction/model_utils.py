import torch
import torch.nn as nn
import torch.nn.functional as F


def get_activation_fn(activation):
	"""Get activation function from string
	Args:
		activation (str): activation type
	Returns:
		activation function as nn.Module
	"""
	if activation == 'relu':
		return nn.ReLU()
	elif activation == 'leaky_relu':
		return nn.LeakyReLU()
	elif activation == 'tanh':
		return nn.Tanh()
	elif activation == 'sigmoid':
		return nn.Sigmoid()
	elif activation == 'gelu':
		return nn.GELU()
	elif activation == 'none':
		return nn.Identity()
	else:
		raise ValueError('Invalid activation function')


def count_params(model, trainable_only=True):
	"""Count number of parameters in a model
	Args:
		model (nn.Module): model to count parameters
		trainable_only (bool): count only trainable parameters
	Returns:
		number of parameters
	"""
	if trainable_only:
		return sum(p.numel() for p in model.parameters() if p.requires_grad)
	else:
		return sum(p.numel() for p in model.parameters())