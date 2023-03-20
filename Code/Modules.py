##############################################
#    Author : Yucheng Xing
#    Description : Modules
##############################################
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class GBSN(nn.Module):
	def __init__(self, dim_in, dim_out, activation=nn.ReLU):
		super(GBSN, self).__init__()
		self.net = nn.Sequential()
		self.net.add_module('conv', nn.Linear(dim_in * 2, dim_out))
		if activation is not None:
			self.net.add_module('act', activation(inplace=True))
		return

	def forward(self, x_p, x_c, A_c):
		A_p = A_c + torch.eye(A_c.size(-1)).cuda()
		D_p = torch.sqrt(A_p.sum(-1))
		L_p = A_p / (D_p.unsqueeze(-1) * D_p.unsqueeze(-2) + 1e-12)
		if len(L_p.size()) != len(x_p.size()):
			L_p = L_p.unsqueeze(1)
		y_p = torch.matmul(L_p, x_p)
		D_c = torch.sqrt(A_c.sum(-1))
		L_c = A_c / (D_c.unsqueeze(-1) * D_c.unsqueeze(-2) + 1e-12)
		if len(L_c.size()) != len(x_c.size()):
			L_c = L_c.unsqueeze(1)
		y_c = torch.matmul(L_c, x_c)
		y = torch.cat([y_p, y_c], -1)
		y = self.net(y)
		return y