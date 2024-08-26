##############################################
#    Author : Yucheng Xing
#    Description : Modules
##############################################
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class GBSL(nn.Module):
	def __init__(self, dim_in, dim_out, num_node, activation=nn.ReLU):
		super(GBSL, self).__init__()
		self.net = nn.Sequential()
		self.net.add_module('conv', nn.Linear(dim_in * 2, dim_out))
		if activation is not None:
			self.net.add_module('act', activation(inplace=True))
		self.dim_in, self.dim_out = dim_in, dim_out
		self.W_c = nn.Parameter(0.01 * torch.randn(size=(1, num_node, num_node)))
		self.W_p = nn.Parameter(0.01 * torch.randn(size=(1, num_node, num_node)))
		return

	def forward(self, x_p, x_c, A_c):
		A_p = A_c + torch.eye(A_c.size(-1)).cuda()
		A_p = A_p / (A_p.sum(-1, keepdims=True) + 1e-12)
		WA_p = torch.softmax(torch.matmul(self.W_p, A_p), dim=-1)
		y_p = torch.matmul(WA_p, x_p)
		A_c = A_c / (A_c.sum(-1, keepdims=True) + 1e-12)
		WA_c = torch.softmax(torch.matmul(self.W_c, A_c), dim=-1)
		y_c = torch.matmul(WA_c, x_c)
		y = torch.cat([y_p, y_c], -1)
		y = self.net(y)
		return y


class GCL(nn.Module):
	def __init__(self, dim_in, dim_out, num_node, activation=nn.ReLU):
		super(GCL, self).__init__()
		self.net = nn.Sequential()
		self.net.add_module('conv', nn.Linear(dim_in, dim_out))
		if activation is not None:
			self.net.add_module('act', activation(inplace=True))
		self.dim_in, self.dim_out = dim_in, dim_out
		self.W = nn.Parameter(0.01 * torch.randn(size=(1, num_node, num_node)))
		return

	def forward(self, x, A):
		A = A + torch.eye(A.size(-1)).cuda()
		A = A / (A.sum(-1, keepdims=True) + 1e-12)
		WA = torch.softmax(torch.matmul(self.W, A), dim=-1)
		y = torch.matmul(WA, x)
		y = self.net(y)
		return y


class GBSN(nn.Module):
	def __init__(self, dim_in, dim_out, dim_hidden, num_hidden, num_node, activation=nn.ReLU):
		super(GBSN, self).__init__()
		self.net1 = nn.ModuleList([])
		self.net2 = nn.ModuleList([])
		for _ in range(0, num_hidden - 1):
			self.net1.append(GBSL(dim_in, dim_in, num_node))
			self.net2.append(nn.Linear(dim_hidden, dim_hidden))
			self.net2.append(nn.ReLU())
		self.net1.append(GBSL(dim_in, dim_hidden, num_node))
		self.net2.append(nn.Linear(dim_hidden, dim_out))
		self.dim_in, self.dim_hidden, self.dim_out, self.num_hidden = dim_in, dim_hidden, dim_out, num_hidden
		self.W_c = nn.Parameter(0.01 * torch.randn(size=(1, num_node, num_node)))
		self.W_p = nn.Parameter(0.01 * torch.randn(size=(1, num_node, num_node)))
		return


	def forward(self, x_p, x_c, A_c, choice='S'):
		y = x_p
		for l in range(0, self.num_hidden):
			y = self.net1[l](y, x_c, A_c)
		for l in range(0, 2 * self.num_hidden - 1):
			y = self.net2[l](y)
		return y