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
		self.criteria = InfLoss()#alpha=1.0) nn.MSELoss()#
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
		#l = self.criteria(ys, xs)
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
			y_c = xs[:, int(t_i): int(t_j) + 1, :, :].mean(dim=1)#.sum(dim=1) / self.window_size
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
		#A = A + torch.eye(A.size(-1)).cuda()
		#D = torch.sqrt(A.sum(-1))
		#L = A / (D.unsqueeze(-1) * D.unsqueeze(-2) + 1e-12)
		D = A.sum(-1)
		L = torch.diag_embed(D) - A
		#L = D - A
		J = torch.eye(A.size(-1)).unsqueeze(0).cuda()
		batch_size, seq_len, _, _ = xs.size()
		ys = []
		y_p = None
		for n_frame in range(seq_len):
			x_c = xs[:, n_frame, :, :]
			y_c = self.closedform_forward(x_c, L, J, y_p)
			#y_c = self.iterative_forward(x_c, L, J, y_p)
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
		#A = A + torch.eye(A.size(-1)).cuda()
		#D = torch.sqrt(A.sum(-1))
		#L = A / (D.unsqueeze(-1) * D.unsqueeze(-2) + 1e-12)
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
			#y_c = self.iterative_forward(x_c, L, J, I, y_p)
			ys.append(y_c)
			y_p = y_c
		ys = torch.stack(ys, 1)
		return ys, None

	def cal_loss(self, xs, A):
		ys, _ = self.forward_seq(xs, A, is_train=True)
		l = self.criteria(xs, ys)
		return l


class GUSC(nn.Module):
	def __init__(self, dim_in, dim_out, dim_hidden, num_hidden, num_node):
		super(GUSC, self).__init__()
		self.dim_in, self.dim_out, self.dim_hidden, self.num_hidden, self.num_node = dim_in, dim_out, dim_hidden, num_hidden, num_node
		self.net_A = self.create_net()
		self.net_B = self.create_net()
		self.net_D = self.create_net()
		self.net_E = self.create_net()
		self.net_H = self.create_net()
		self.alpha = nn.Parameter(torch.Tensor([0.5]))
		self.criteria = nn.MSELoss()
		return

	def soft_thresholding(self, s):
		s_new = torch.where(s > self.alpha, s - self.alpha, torch.where(s < -self.alpha, s + self.alpha, 0))
		return s_new

	def create_net(self):
		MLPs = nn.ModuleList()
		for _ in range(2):
			MLP = nn.Sequential(
						nn.Linear(self.dim_in, self.dim_hidden),
						nn.ReLU(), 
						nn.Linear(self.dim_hidden, 1) 
						)
			MLPs.append(MLP)
		return MLPs

	def create_A_pows(self, A):
		A_pows = []
		idxs_pows = []
		for l in range(2):#self.num_hidden):
			if l == 0:
				A_pows.append(torch.eye(A.size(-1)).repeat(A.size(0), 1, 1).cuda())
			else:
				A_pows.append(A_pows[l - 1]@A)
			idxs = torch.where(A_pows[l] != 0)
			idxs_pows.append(list(zip(idxs[0], idxs[1])))
		return A_pows, idxs_pows

	def create_conv_mat(self, P, MLPs, A_pows, idxs_pows):
		conv_mat = torch.zeros_like(A_pows[0]).cuda()
		for l, A_l in enumerate(A_pows):
			for idx in idxs_pows[l]:
				input = P[:, idx[0], :] - P[:, idx[1], :]
				k = A_l[:, idx[0], idx[1]] * MLPs[l](input)[:, 0]
				conv_mat[:, idx[0], idx[1]] += k
		return conv_mat

	def forward(self, x_c, conv_A, conv_B, conv_D, conv_E, conv_H):
		s = torch.zeros_like(x_c).cuda()
		z = torch.zeros_like(x_c).cuda()
		for _ in range(self.num_hidden):
			#y_c = self.net_A(torch.matmul(A, s)) + self.net_B(torch.matmul(A, x_c))#conv_A@s + conv_B@x_c#
			#s = self.net_D(torch.matmul(A, y_c)) + self.net_E(torch.matmul(A, z))#conv_D@y_c + conv_E@z#
			y_c = conv_A@s + conv_B@x_c#self.net_A(s) + self.net_B(x_c)#
			s = conv_D@y_c + conv_E@z#self.net_D(y_c) + self.net_E(z)#
			z = self.soft_thresholding(s)
		y_c = conv_H@s
		return y_c

	def forward_seq(self, xs, A, is_train=False):
		A = A + torch.eye(A.size(-1)).cuda()
		lambdas, V = torch.linalg.eigh(A)
		A_norm = A / torch.abs(lambdas[0])
		A_pows, idxs_pows = self.create_A_pows(A_norm)
		P = V[:, :, :self.dim_in]
		conv_A = self.create_conv_mat(P, self.net_A, A_pows, idxs_pows)
		conv_B = self.create_conv_mat(P, self.net_B, A_pows, idxs_pows)
		conv_D = self.create_conv_mat(P, self.net_D, A_pows, idxs_pows)
		conv_E = self.create_conv_mat(P, self.net_E, A_pows, idxs_pows)
		conv_H = self.create_conv_mat(P, self.net_H, A_pows, idxs_pows)
		batch_size, seq_len, _, _ = xs.size()
		ys = []
		for n_frame in range(seq_len):
			x_c = xs[:, n_frame, :, :]
			y_c = self.forward(x_c, conv_A, conv_B, conv_D, conv_E, conv_H)
			ys.append(y_c)
		ys = torch.stack(ys, 1)
		return ys, None

	def cal_loss(self, xs, A):
		ys, _ = self.forward_seq(xs, A, is_train=True)
		l = self.criteria(xs, ys)
		return l



class GUSC1(nn.Module):
	def __init__(self, dim_in, dim_out, dim_hidden, num_hidden, num_node):
		super(GUSC1, self).__init__()
		self.dim_in, self.dim_out, self.dim_hidden, self.num_hidden, self.num_node = dim_in, dim_out, dim_hidden, num_hidden, num_node
		self.net_A = self.create_net()
		self.net_B = self.create_net()
		self.net_D = self.create_net()
		self.net_E = self.create_net()
		self.net_H = self.create_net()
		self.alpha = nn.Parameter(torch.Tensor([0.1]))
		self.criteria = nn.MSELoss()
		return

	def soft_thresholding(self, s):
		s_new = torch.where(s > self.alpha, s - self.alpha, torch.where(s < -self.alpha, s + self.alpha, 0))
		return s_new

	def create_net(self):
		MLP = nn.Sequential(
						nn.Linear(self.dim_in, self.dim_hidden),
						nn.ReLU(), 
						nn.Linear(self.dim_hidden, self.dim_out) 
						)
		return MLP

	def forward(self, x_c, A):
		s = torch.zeros_like(x_c).cuda()
		z = torch.zeros_like(x_c).cuda()
		for _ in range(self.num_hidden):
			y_c = self.net_A(A@s) + self.net_B(A@x_c)
			s = self.net_D(A@y_c) + self.net_E(A@z)
			z = self.soft_thresholding(s)
		y_c = self.net_D(s)
		return y_c, s

	def forward_seq(self, xs, A, is_train=False):
		A = A + torch.eye(A.size(-1)).cuda()
		A = A / (A.sum(-1, keepdims=True) + 1e-12)
		batch_size, seq_len, _, _ = xs.size()
		ys = []
		l = 0.0
		for n_frame in range(seq_len):
			x_c = xs[:, n_frame, :, :]
			y_c, s = self.forward(x_c, A)
			ys.append(y_c)
			l += torch.abs(s).mean()
		ys = torch.stack(ys, 1)
		return ys, l / seq_len

	def cal_loss(self, xs, A):
		ys, dl = self.forward_seq(xs, A, is_train=True)
		l = self.criteria(xs, ys) + 0 * dl
		return l



class GUTF(nn.Module):
	def __init__(self, dim_in, dim_out, dim_hidden, num_hidden, num_node):
		super(GUTF, self).__init__()
		self.dim_in, self.dim_out, self.dim_hidden, self.num_hidden, self.num_node = dim_in, dim_out, dim_hidden, num_hidden, num_node
		self.net_B = self.create_net()
		self.net_C = self.create_net()
		self.alpha = nn.Parameter(torch.Tensor([0.5]))
		self.criteria = nn.MSELoss()
		return

	def soft_thresholding(self, s):
		s_new = torch.where(s > self.alpha, s - self.alpha, torch.where(s < -self.alpha, s + self.alpha, 0))
		return s_new

	def create_net(self):
		MLPs = nn.ModuleList()
		for _ in range(2):
			MLP = nn.Sequential(
						nn.Linear(self.dim_in, self.dim_hidden),
						nn.ReLU(), 
						nn.Linear(self.dim_hidden, 1) 
						)
			MLPs.append(MLP)
		return MLPs

	def create_A_pows(self, A):
		A_pows = []
		idxs_pows = []
		for l in range(2):#self.num_hidden):
			if l == 0:
				A_pows.append(torch.eye(A.size(-1)).repeat(A.size(0), 1, 1).cuda())
			else:
				A_pows.append(A_pows[l - 1]@A)
			idxs = torch.where(A_pows[l] != 0)
			idxs_pows.append(list(zip(idxs[0], idxs[1])))
		return A_pows, idxs_pows

	def create_conv_mat(self, P, MLPs, A_pows, idxs_pows):
		conv_mat = torch.zeros_like(A_pows[0]).cuda()
		for l, A_l in enumerate(A_pows):
			for idx in idxs_pows[l]:
				input = P[:, idx[0], :] - P[:, idx[1], :]
				k = A_l[:, idx[0], idx[1]] * MLPs[l](input)[:, 0]
				conv_mat[:, idx[0], idx[1]] += k
		return conv_mat

	def forward(self, x_c, L, conv_B, conv_C):
		y_c = torch.zeros_like(x_c).cuda()
		for _ in range(self.num_hidden):
			z = self.soft_thresholding(L.T@y_c)
			y_c = conv_C@(L@z) + conv_B@x_c
		return y_c

	def forward_seq(self, xs, A, is_train=False):
		A = A + torch.eye(A.size(-1)).cuda()
		#D = torch.sqrt(A.sum(-1))
		#A = A / (torch.matmul(D.unsqueeze(-1), D.unsqueeze(-2)) + 1e-12)
		#D = A.sum(-1)
		#L = torch.diag_embed(D) - A
		G = nx.from_numpy_array(np.sqrt(A[0].cpu().numpy()))
		L = torch.Tensor(nx.linalg.graphmatrix.incidence_matrix(G).todense()).cuda()
		lambdas, V = torch.linalg.eigh(A)
		A_norm = A / torch.abs(lambdas[0])
		A_pows, idxs_pows = self.create_A_pows(A_norm)
		P = V[:, :, :self.dim_in]
		conv_B = self.create_conv_mat(P, self.net_B, A_pows, idxs_pows)
		conv_C = self.create_conv_mat(P, self.net_C, A_pows, idxs_pows)
		batch_size, seq_len, _, _ = xs.size()
		ys = []
		for n_frame in range(seq_len):
			x_c = xs[:, n_frame, :, :]
			y_c = self.forward(x_c, L, conv_B, conv_C)
			ys.append(y_c)
		ys = torch.stack(ys, 1)
		return ys, None

	def cal_loss(self, xs, A):
		ys, _ = self.forward_seq(xs, A, is_train=True)
		l = self.criteria(xs, ys)
		return l



class GUTF1(nn.Module):
	def __init__(self, dim_in, dim_out, dim_hidden, num_hidden, num_node):
		super(GUTF1, self).__init__()
		self.dim_in, self.dim_out, self.dim_hidden, self.num_hidden, self.num_node = dim_in, dim_out, dim_hidden, num_hidden, num_node
		self.net_B = self.create_net()
		self.net_C = self.create_net()
		self.alpha = nn.Parameter(torch.Tensor([0.5]))
		self.criteria = nn.MSELoss()
		return

	def soft_thresholding(self, s):
		s_new = torch.where(s > self.alpha, s - self.alpha, torch.where(s < -self.alpha, s + self.alpha, 0))
		return s_new

	def create_net(self):
		MLP = nn.Sequential(
						nn.Linear(self.dim_in, self.dim_hidden),
						nn.ReLU(), 
						nn.Linear(self.dim_hidden, self.dim_out) 
						)
		return MLP

	def forward(self, x_c, A, L):
		y_c = torch.zeros_like(x_c).cuda()
		for _ in range(self.num_hidden):
			#z = self.soft_thresholding(torch.matmul(L, y_c))
			#y_c = self.net_C(torch.matmul(A, torch.matmul(L, z))) + self.net_B(torch.matmul(A, x_c))
			z = self.soft_thresholding(L.T@y_c)
			y_c = self.net_C(L@z) + self.net_B(x_c)
		return y_c

	def forward_seq(self, xs, A, is_train=False):
		#A = A + torch.eye(A.size(-1)).cuda()
		#D = torch.sqrt(A.sum(-1))
		#A = A / (torch.matmul(D.unsqueeze(-1), D.unsqueeze(-2)) + 1e-12)
		#D = A.sum(-1)
		#L = torch.diag_embed(D) - A
		l = 0.0
		G = nx.from_numpy_array(np.sqrt(A[0].cpu().numpy()))
		L = torch.Tensor(nx.linalg.graphmatrix.incidence_matrix(G).todense()).cuda()
		batch_size, seq_len, _, _ = xs.size()
		ys = []
		for n_frame in range(seq_len):
			x_c = xs[:, n_frame, :, :]
			y_c = self.forward(x_c, A, L)
			l += torch.abs(L.T@y_c).mean()
			ys.append(y_c)
		ys = torch.stack(ys, 1)
		return ys, l / seq_len

	def cal_loss(self, xs, A):
		ys, dl = self.forward_seq(xs, A, is_train=True)
		l = self.criteria(xs, ys) + 0 * dl
		return l
