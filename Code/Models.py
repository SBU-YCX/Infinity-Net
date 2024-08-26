##############################################
#    Author : Yucheng Xing
#    Description : Models
##############################################
import torch
import numpy as np
import torch.nn as nn
import networkx as nx
import torch.nn.functional as F
from Modules import GBSN
from Losses import InfLoss


class InfinityNet(nn.Module):
	def __init__(self, dim_in, dim_out, dim_hidden, num_hidden, num_node):
		super(InfinityNet, self).__init__()
		self.block = GBSN(dim_in, dim_out, dim_hidden, num_hidden, num_node)
		self.dim_in, self.dim_out, self.dim_hidden, self.num_hidden = dim_in, dim_out, dim_hidden, num_hidden
		self.criteria = InfLoss()
		return

	def forward(self, x_c, A, y_p=None):
		if y_p is None:
			y_p = torch.zeros_like(x_c)
		y_c = self.block(x_c, y_p, A)
		return y_c

	def forward_seq(self, xs, A, is_train=False):
		batch_size, seq_len, _, _ = xs.size()
		ys, zs = [], []
		x_p, x_c, y_p = None, None, None
		for n_frame in range(seq_len):
			x_c = xs[:, n_frame, :, :]
			y_c = self.forward(x_c, A, y_p)
			ys.append(y_c)
			if n_frame != 0:
				z_p = self.forward(x_p, A, y_c)
				zs.append(z_p)
			x_p, y_p = x_c, y_c
		ys = torch.stack(ys, 1)
		zs = torch.stack(zs, 1)
		return ys, zs

	def cal_loss(self, xs, A):
		ys, zs = self.forward_seq(xs, A, is_train=True)
		l = self.criteria(xs, ys, zs)
		return l


class MA(nn.Module):
	def __init__(self, dim_in, dim_out, num_node, window_size, on_off):
		super(MA, self).__init__()
		self.dim_in, self.dim_out, self.num_node, self.window_size, self.on_off = dim_in, dim_out, num_node, window_size, on_off
		return

	def forward_seq(self, xs, A, is_train=False):
		batch_size, seq_len, _, _ = xs.size()
		ys, zs = [], []
		for n_frame in range(seq_len):
			if self.on_off == 'off':
				if self.window_size % 2:
					t_i = max(0, n_frame - self.window_size / 2)
					t_j = min(seq_len - 1, n_frame + self.window_size / 2)
				else:
					t_i = max(0, n_frame - self.window_size / 2)
					t_j = min(seq_len - 1, n_frame + self.window_size / 2 - 1)
			else:
				t_i = max(0, n_frame - self.window_size + 1)
				t_j = min(seq_len - 1, n_frame)
			y_c = xs[:, int(t_i): int(t_j) + 1, :, :].mean(dim=1)
			ys.append(y_c)
		ys = torch.stack(ys, 1)
		return ys, zs


class Noise2Noise(nn.Module):
	def __init__(self, dim_in, dim_out, dim_hidden, num_hidden, num_node):
		super(Noise2Noise, self).__init__()
		self.dim_in, self.dim_out, self.dim_hidden, self.num_hidden, self.num_node = dim_in, dim_out, dim_hidden, num_hidden, num_node
		self.net = self.built_net()
		self.criteria = nn.MSELoss()
		return

	def built_net(self):
		modules = []
		modules.extend([nn.Linear(self.dim_in, self.dim_hidden), 
						nn.BatchNorm1d(self.num_node), 
						nn.LeakyReLU()])
		for _ in range(self.num_hidden - 2):
			modules.extend([nn.Linear(self.dim_hidden, self.dim_hidden), 
						nn.BatchNorm1d(self.num_node), 
						nn.LeakyReLU()])
		modules.extend([nn.Linear(self.dim_hidden, self.dim_out)])
		net = nn.Sequential(*modules)
		return net

	def forward_seq(self, xs, A, is_train=False):
		batch_size, seq_len, _, _ = xs.size()
		ys = []
		for n_frame in range(seq_len):
			x_c = xs[:, n_frame, :, :]
			if (is_train):
				x_c = x_c + torch.normal(torch.zeros_like(x_c), 0.5 * torch.ones_like(x_c))
			y_c = self.net(x_c)
			ys.append(y_c)
		ys = torch.stack(ys, 1)
		return ys, None


	def cal_loss(self, xs, A):
		ys, _ = self.forward_seq(xs, A, is_train=True)
		l = self.criteria(xs, ys)
		return l


class MedianFilter(nn.Module):
	def __init__(self, dim_in, dim_out, num_node, on_off):
		super(MedianFilter, self).__init__()
		self.dim_in, self.dim_out, self.num_node, self.on_off = dim_in, dim_out, num_node, on_off
		self.criteria = nn.MSELoss()

	def forward_seq(self, xs, A, is_train=False):
		A_f = A + torch.eye(A.size(-1)).cuda()
		batch_size, seq_len, _, _ = xs.size()
		ys = []
		for n_frame in range(seq_len):
			X = []
			x_c = xs[:, n_frame, :, :]
			for n_node in range(self.num_node):
				X_c = []
				if n_frame > 0:
					X_c.append(xs[:, n_frame - 1, n_node, :].unsqueeze(1))
				if self.on_off == 'off' and n_frame <= seq_len - 2:
					X_c.append(xs[:, n_frame + 1, n_node, :].unsqueeze(1))
				X_c.append(x_c[:, torch.argwhere(A_f[:, n_node, :])[:, 1], :])
				X_c = torch.cat(X_c, 1)
				x_m, _ = torch.median(X_c, 1)
				X.append(x_m)
			X = torch.stack(X, 1)
			ys.append(X)
		ys = torch.stack(ys, 1)
		return ys, None


	def cal_loss(self, xs, A):
		ys, _ = self.forward_seq(xs, A, is_train=True)
		l = self.criteria(xs, ys)
		return l
		

class TGSR(nn.Module):
	def __init__(self, dim_in, dim_out, num_node, on_off):
		super(TGSR, self).__init__()
		self.dim_in, self.dim_out, self.num_node, self.on_off = dim_in, dim_out, num_node, on_off
		self.criteria = nn.MSELoss()

	def closedform_forward(self, x_c, L, J, y_p=None):
		if y_p is None:
			y_p = torch.zeros_like(x_c)
		y_c = torch.matmul(torch.linalg.inv(L + J), torch.matmul(L, y_p) + x_c)
		return y_c

	def iterative_forward(self, x_c, L, J, y_p=None):
		if y_p is None:
			y_p = torch.zeros_like(x_c)
		y_c = torch.zeros_like(x_c)
		g_p = (torch.bmm(J, y_c) - x_c) + 0.5 * torch.bmm(L, y_c - y_p)
		dy = -g_p
		for _ in range(20000):
			tmp = (torch.bmm(J, dy) - x_c) + 0.5 * torch.bmm(L, dy - y_p) + x_c + 0.5 * torch.bmm(L, y_p)
			s = -torch.bmm(torch.transpose(g_p, 1, 2), dy) / torch.bmm(torch.transpose(tmp, 1, 2), dy)
			y_c = y_c + torch.bmm(dy, s)
			g_c = (torch.bmm(J, y_c) - x_c) + 0.5 * torch.bmm(L, y_c - y_p)
			bk = -torch.bmm(torch.transpose(g_c, 1, 2), g_c) / (torch.bmm(torch.transpose(g_p, 1, 2), g_p) + 1e-12)
			g_p = g_c
			dy = -g_c + torch.bmm(dy, bk)
			if (torch.linalg.norm(dy) < 1e-6):
				break
		return y_c

	def forward_seq(self, xs, A, is_train=False):
		D = A.sum(-1)
		L = torch.diag_embed(D) - A
		J = torch.eye(A.size(-1)).unsqueeze(0).cuda()
		batch_size, seq_len, _, _ = xs.size()
		ys = []
		y_p = None
		for n_frame in range(seq_len):
			x_c = xs[:, n_frame, :, :]
			y_c = self.closedform_forward(x_c, L, J, y_p)
			ys.append(y_c)
			y_p = y_c
		ys = torch.stack(ys, 1)
		return ys, None

	def cal_loss(self, xs, A):
		ys, _ = self.forward_seq(xs, A, is_train=True)
		l = self.criteria(xs, ys)
		return l


class GraphTRSS(nn.Module):
	def __init__(self, dim_in, dim_out, num_node, on_off):
		super(GraphTRSS, self).__init__()
		self.dim_in, self.dim_out, self.num_node, self.on_off = dim_in, dim_out, num_node, on_off
		self.criteria = nn.MSELoss()

	def closedform_forward(self, x_c, L, J, I, y_p=None):
		if y_p is None:
			y_p = torch.zeros_like(x_c)
		y_c = torch.matmul(torch.linalg.inv(L + I + J), torch.matmul(L + I, y_p) + x_c)
		return y_c

	def iterative_forward(self, x_c, L, J, I, y_p=None):
		if y_p is None:
			y_p = torch.zeros_like(x_c)
		y_c = torch.zeros_like(x_c)
		g_p = (torch.bmm(J, y_c) - x_c) + 0.5 * torch.bmm(L + I, y_c - y_p)
		dy = -g_p
		for _ in range(100):
			tmp = (torch.bmm(J, dy) - x_c) + 0.5 * torch.bmm(L + I, dy - y_p) + x_c + 0.5 * torch.bmm(L + I, y_p)
			s = -torch.bmm(torch.transpose(g_p, 1, 2), dy) / torch.bmm(torch.transpose(tmp, 1, 2), dy)
			y_c = y_c + torch.bmm(dy, s)
			g_c = (torch.bmm(J, y_c) - x_c) + 0.5 * torch.bmm(L + I, y_c - y_p)
			bk = -torch.bmm(torch.transpose(g_c, 1, 2), g_c) / (torch.bmm(torch.transpose(g_p, 1, 2), g_p) + 1e-12)
			g_p = g_c
			dy = -g_c + torch.bmm(dy, bk)
			if (torch.linalg.norm(dy) < 1e-6):
				break
		return y_c

	def forward_seq(self, xs, A, is_train=False):
		D = A.sum(-1)
		L = torch.diag_embed(D) - A
		J = torch.eye(A.size(-1)).unsqueeze(0).cuda()
		I = 0.1 * torch.eye(A.size(-1)).unsqueeze(0).cuda()
		batch_size, seq_len, _, _ = xs.size()
		ys = []
		y_p = None
		for n_frame in range(seq_len):
			x_c = xs[:, n_frame, :, :]
			y_c = self.closedform_forward(x_c, L, J, I, y_p)
			ys.append(y_c)
			y_p = y_c
		ys = torch.stack(ys, 1)
		return ys, None

	def cal_loss(self, xs, A):
		ys, _ = self.forward_seq(xs, A, is_train=True)
		l = self.criteria(xs, ys)
		return l
