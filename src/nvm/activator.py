import torch as tr 
import numpy as np
from tensor import *

class Activator:
	def __init__(self, f, g, e, make_pattern, hash_pattern, on, off, label):
		self.f = f
		self.g = g
		self.e = e
		self.make_pattern = make_pattern
		self.hash_pattern = hash_pattern
		self.on = on
		self.off = off
		self.label = label

	def gain(self):
		w = (self.g(self.on) - self.g(self.off))/(self.on - self.off)
		b = (self.g(self.off)*self.on - self.g(self.on)*self.off)/(self.on - self.off)
		return w, b

	def corrosion(self, pattern):
		return tr.min( tr.abs(pattern - self.on), tr.abs(pattern - self.off)).max()

def tanh_activator(pad, layer_size): 
	return Activator(
		f = tr.tanh,
		g = lambda v : totensor(np.arctanh(np.clip(v, pad - 1., 1 - pad))),
		e = lambda a, b: ((a > 0) == (b > 0)),
		make_pattern = lambda : (1.-pad)*tr.sign(tr.randn(layer_size,1)) ,
		hash_pattern = lambda p: (p > 0).numpy().tobytes() if isinstance(p, torch.Tensor) else (p>0).tobytes(),
		on = 1. - pad,
		off = -(1. - pad),
		label = "tanh" )

def logistic_activator(pad, layer_size):
	def make_pattern():
		r = tr.randn(layer_size, 1) > 0
		r = (1. - pad)*r + (0. + pad)*(~r)
		r = r.float()
		return r

	return Activator(
		f = lambda v: .5*(tr.tanh(v)+1),
		g = lambda v: totensor(np.arctanh(2*np.clip(v, pad, 1. - pad) - 1)), 
		e = lambda a, b: ((a > .5) == (b > .5)),
		make_pattern = make_pattern,
		hash_pattern = lambda p: (p > .5).numpy().tobytes() if isinstance(p, torch.Tensor) else (p>0).tobytes() ,
		on = 1. - pad,
		off = 0. + pad,
		label = "logistic")

def heaviside_activator(layer_size):
	return Activator(
		f = lambda v: (v > .5).float(),
		g = lambda v: (-1.)**(v < .5),
		e = lambda a, b: ((a > .5) == (b > .5)),
		make_pattern = lambda : (tr.randn(layer_size, 1) > 0).float(),
		hash_pattern = lambda p: (p > .5).numpy().tobytes() if isinstance(p, torch.Tensor) else (p>0).tobytes(),
		on = 1.,
		off = 0.,
		label = "heaviside")

def gate_activator(pad, layer_size):
	#arctanh = totensor(np.arctanh(np.clip(v, 0., 1. - pad)))
	return Activator(
		f = tr.tanh,
		g = lambda v : totensor(np.arctanh(np.clip(v, 0., 1. - pad))),
		e = lambda a, b: ((a > .5) == (b > .5)),
		make_pattern = lambda : (1.-pad)*(tr.randn(layer_size,1) > 0.),
		hash_pattern = lambda p: (p > .5).numpy().tobytes() if isinstance(p, torch.Tensor) else (p>0).tobytes(),
		on = 1. - pad,
		off = 0.,
		label = "gate")
