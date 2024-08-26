##############################################
#    Author : Yucheng Xing
#    Description : Power Dataset
##############################################


import h5py, os
import numpy as np
import torch
from torch.utils.data import Dataset


class IEEE(Dataset):
	def __init__(self, data_path, split, noise_type='gaussian', noise_ratio=0.0, poisson_lambda=20, rayleigh_scale=10, gamma_shape=10, gamma_scale=1):
		super(IEEE, self).__init__()
		self.data = h5py.File(data_path, 'r')
		self.mean = self.data['current_mean_v'][()]
		self.std = self.data['current_std_v'][()]
		self.size = self.data['num_seg'][()]
		self.noise_type = noise_type
		self.noise_ratio = noise_ratio
		self.poisson_lambda = poisson_lambda
		self.rayleigh_scale = rayleigh_scale
		self.gamma_shape, self.gamma_scale = gamma_shape, gamma_scale
		return

	def __len__(self):
		return self.size

	def __getitem__(self, idx):
		X_ori = (self.data['node_currents_{}'.format(idx)][()] - self.mean) / self.std
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
		edges = self.data['topologies_node_{}'.format(self.data['topologies_{}'.format(idx)][()])][()]
		i = torch.LongTensor(edges)
		v = torch.FloatTensor(torch.ones(i.size(0)))
		adjacent_nodes = torch.sparse.FloatTensor(i.t(), v, torch.Size([33, 33])).to_dense()
		adjacent_nodes += adjacent_nodes.clone().t()
		return {'A': adjacent_nodes, 'X': X.astype(np.float32), 'Y': X_ori.astype(np.float32)}
