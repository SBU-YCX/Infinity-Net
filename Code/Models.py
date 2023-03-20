##############################################
#    Author : Yucheng Xing
#    Description : Models
##############################################
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from Modules import GSBN


class InfinityNet(nn.Module):
	def __init__(self, 
				 dim_in, 
				 dim_hidden, 
				 dim_out, 
				 ):
		super(InfinityNet, self).__init__()
		self.net = nn.Sequential(
						GSBN(dim_in, dim_hidden),
						GSBN(dim_hidden, dim_hidden),  
						GSBN(dim_hidden, dim_out)
					)
		self.loss_f = nn.MSELoss()
		self.loss_b = nn.MSELoss()
		return

	def forward(self, x_c, A, y_p=None):
		if y_p is None:
			y_p = torch.zeros_like(x)
		y_c = self.net(y_p, x_c, A)
		return y_c

	def forward_seq(self, xs, A):
		ys_f, ys_b = [], []
		y_p = None
		for i in range(xs.shape[1]):
			x = xs[:, i, :, :]
			y_f = self.forward(x, A, y_p)
			ys_f.append(y_f)
			y_p = y_f
			if i == 0:
				continue
			x_p = xs[:, i - 1, :, :]
			y_b = self.forward(x_p, A, y_p)
			ys_b.append(y_b)
		ys_f = torch.stack(ys_f, 1)
		ys_g = torch.stack(ys_b, 1)
		return ys_f, ys_b

	def get_loss(self, xs, A):
		ys_f, ys_b = self.forward_seq(xs, A)
		l_f = self.loss_f(ys_f, xs)
		l_b = self.loss_b(ys_b, ys_f[:, 1:-1, :, :])
		l = l_f + l_b
		return l