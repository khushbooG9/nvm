import torch
import numpy

def totensor(arr):
	if isinstance(arr, torch.Tensor):
		return arr
	if not isinstance(arr, numpy.ndarray):
		arr = numpy.array(arr)
	return torch.from_numpy(arr)

def fromtensor(t):
	if isinstance(t, numpy.ndarray ):
		return t
	return t.cpu().detach().numpy()