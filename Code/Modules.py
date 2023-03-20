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
		self.net_p = nn.Linear(dim_in, dim_out)
		self.net_c = nn.Linear(dim_in, dim_out)
		self.act = activation(inplace=True)
		return

	def forward(self, x_p, x_c, A_c):
		A_p = A_c + torch.eye(A_c.size(-1)).cuda()
		D_p = torch.sqrt(A_p.sum(-1))
		L_p = A_p / (D_p.unsqueeze(-1) * D_p.unsqueeze(-2) + 1e-12)
		if len(L_p.size()) != len(x_p.size()):
			L_p = L_p.unsqueeze(1)
		x_p = self.net_p(x_p)
		y_p = torch.matmul(L_p, x_p)
		D_c = torch.sqrt(A_c.sum(-1))
		L_c = A_c / (D_c.unsqueeze(-1) * D_c.unsqueeze(-2) + 1e-12)
		if len(L_c.size()) != len(x_c.size()):
			L_c = L_c.unsqueeze(1)
		x_c = self.net_c(x_c)
		y_c = torch.matmul(L_c, x_c)
		y = self.act(y_p + y_c)
		return y