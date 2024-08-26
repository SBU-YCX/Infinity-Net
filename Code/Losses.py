##############################################
#    Author : Yucheng Xing
#    Description : Loss Functions
##############################################


import torch
import torch.nn as nn


class InfLoss(nn.Module):
	def __init__(self, alpha=0.5):
		super(InfLoss, self).__init__()
		self.alpha = alpha
		return

	def forward(self, xs, ys, zs):
		forward_loss = ((ys - xs) ** 2).mean()
		backward_loss = ((ys[:, :-1, :, :] - zs) ** 2).mean()
		loss = self.alpha * forward_loss + (1 - self.alpha) * backward_loss
		return loss

	