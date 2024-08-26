##############################################
#    Author : Yucheng Xing
#    Description : Traffic Dataset
##############################################


import h5py, os, torch
import numpy as np
from torch.utils.data import Dataset


class Traffic(Dataset):
	def __init__(self, data_path, split, noise_type='gaussian', noise_ratio=0.0, poisson_lambda=20, rayleigh_scale=10, gamma_shape=10, gamma_scale=1):
		super(Traffic, self).__init__()
		f = h5py.File(data_path, 'r')
		self.adj = f['adj']
		self.data = f['data'][()]
		self.mean = np.expand_dims(f['mean'][()], axis = 2)
		self.std = np.expand_dims(f['std'][()], axis = 2)
		self.data_idx = f['data_idx']
		self.item_idx = f['{}_idx'.format(split)]
		self.size = self.item_idx.shape[0]
		self.noise_type = noise_type
		self.noise_ratio = noise_ratio
		self.poisson_lambda = poisson_lambda
		self.rayleigh_scale = rayleigh_scale
		self.gamma_shape, self.gamma_scale = gamma_shape, gamma_scale
		return

	def __len__(self):
		return self.size

	def __getitem__(self, idx):
		adj = self.adj[()]
		item_idx = self.item_idx[idx]
		data_idx = self.data_idx[item_idx]
		X_ori = (self.data[data_idx] - self.mean) / self.std
		if self.noise_ratio != 0:
			if self.noise_type == 'ratio':
				rms_X = np.sqrt((X_ori * X_ori).sum((0)) / X_ori.shape[0])
				X_noise = np.random.randn(X_ori.shape[0], X_ori.shape[1], X_ori.shape[2]) * self.noise_ratio * rms_X
			elif self.noise_type == 'gaussian':
				X_noise = np.random.normal(0, self.noise_ratio, size=X_ori.shape)
			elif self.noise_type == 'poisson':
				X_noise = np.random.poisson(lam=self.poisson_lambda, size=X_ori.shape) - self.poisson_lambda
			elif self.noise_type == 'rayleigh':
				X_noise = (np.random.rayleigh(scale=self.rayleigh_scale, size=X_ori.shape) - self.rayleigh_scale)# * X_ori
			elif self.noise_type == 'gamma':
				X_noise = np.random.gamma(shape=self.gamma_shape, scale=self.gamma_scale, size=X_ori.shape) - self.gamma_shape / self.gamma_scale
			elif self.noise_type == 'mixture':
				X_noise1 = np.random.normal(0, self.noise_ratio, size=X_ori.shape)
				X_noise2 = np.random.poisson(lam=self.poisson_lambda, size=X_ori.shape) - self.poisson_lambda
				X_noise3 = (np.random.rayleigh(scale=self.rayleigh_scale, size=X_ori.shape) - self.rayleigh_scale)# * X_ori
				X_noise4 = np.random.gamma(shape=self.gamma_shape, scale=self.gamma_scale, size=X_ori.shape) - self.gamma_shape / self.gamma_scale
				X_noise = 0.25 * (X_noise1 + X_noise2 + X_noise3 + X_noise4)
			X = X_ori + X_noise
		else:
			X = X_ori
		return {'A': adj, 'X': X, 'Y': X_ori}