##############################################
#    Author : Yucheng Xing
#    Description : Train & Test
##############################################


import os, time, logging, argparse, torch
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.nn import functional as F
from Processing import IEEE, Traffic
from Models import InfinityNet, MA, Noise2Noise, MedianFilter, TGSR, GraphTRSS, GUSC, GUTF, GUSC1, GUTF1
from Losses import InfLoss


parser = argparse.ArgumentParser(description='Experiment Configuration.')
parser.add_argument('--log_path', required=False, default='../Log', type=str, help='')
parser.add_argument('--data_name', required=False, default='ieee', type=str, help='')
parser.add_argument('--data_path', required=False, default='../Data/IEEE/processed/ieee.hdf5', type=str, help='')
parser.add_argument('--noise_type', required=False, default='gaussian', type=str, help='')
parser.add_argument('--noise_ratio', required=False, default=0.1, type=float, help='')
parser.add_argument('--poisson_lambda', required=False, default=20, type=float, help='')
parser.add_argument('--rayleigh_scale', required=False, default=10, type=float, help='')
parser.add_argument('--gamma_shape', required=False, default=10, type=float, help='')
parser.add_argument('--gamma_scale', required=False, default=1, type=float, help='')
parser.add_argument('--train_noise_type', required=False, default='gaussian', type=str, help='')
parser.add_argument('--train_noise_ratio', required=False, default=0.1, type=float, help='')
parser.add_argument('--train_poisson_lambda', required=False, default=20, type=float, help='')
parser.add_argument('--train_rayleigh_scale', required=False, default=10, type=float, help='')
parser.add_argument('--train_gamma_shape', required=False, default=10, type=float, help='')
parser.add_argument('--train_gamma_scale', required=False, default=1, type=float, help='')
parser.add_argument('--model_name', required=False, default='infinity_net', type=str, help='')
parser.add_argument('--model_path', required=False, default='../Model', type=str, help='')
parser.add_argument('--batch_size', required=False, default=1, type=int, help='')
parser.add_argument('--lr', required=False, default=1e-2, type=float, help='')
parser.add_argument('--max_epoch', required=False, default=200, type=int, help='')
parser.add_argument('--early_stop', required=False, default=20, type=int, help='')
parser.add_argument('--pretrained', required=False, default=False, type=bool, help='')
parser.add_argument('--type_n', required=False, default='gaussian', type=str, help='')
parser.add_argument('--p_n', required=False, default=0.5, type=float, help='')
parser.add_argument('--dim_in', required=False, default=2, type=int, help='')
parser.add_argument('--dim_hidden', required=False, default=32, type=int, help='')
parser.add_argument('--dim_out', required=False, default=2, type=int, help='')
parser.add_argument('--num_hidden', required=False, default=2, type=int, help='')
parser.add_argument('--num_node', required=False, default=33, type=int, help='')
parser.add_argument('--num_head', required=False, default=3, type=int, help='')
parser.add_argument('--window_size', required=False, default=2, type=int, help='')
parser.add_argument('--on_off', required=False, default='on', type=str, help='')


def get_logger(args):
	## Log File
	if args.train_noise_type.lower() == 'gaussian':
		noise_name = '{}_{}'.format(args.train_noise_type, args.train_noise_ratio)
	elif args.train_noise_type.lower() == 'poisson':
		noise_name = '{}_{}'.format(args.train_noise_type, args.train_poisson_lambda)
	elif args.train_noise_type.lower() == 'rayleigh':
		noise_name = '{}_{}'.format(args.train_noise_type, args.train_rayleigh_scale)
	elif args.train_noise_type.lower() == 'gamma':
		noise_name = '{}_{}_{}'.format(args.train_noise_type, args.train_gamma_shape, args.train_gamma_scale)
	else:
		noise_name = '{}'.format(args.train_noise_type)
	log_folder = os.path.join(os.path.join(args.log_path, args.data_name), noise_name)
	if not os.path.exists(log_folder):
		os.makedirs(log_folder)
	log_path = os.path.join(log_folder, args.model_name + '.log')
	logging.basicConfig(level=logging.DEBUG)
	logger = logging.getLogger()
	logger.setLevel(logging.DEBUG)
	if log_path:
		file_handler = logging.FileHandler(log_path)
		logger.addHandler(file_handler)
	return logger


def get_model(args):
	## Model
	if args.model_name.lower() == 'infinity_net':
		return InfinityNet(args.dim_in,
						   args.dim_out, 
						   args.dim_hidden, 
						   args.num_hidden, 
						   args.num_node).cuda()
	elif args.model_name.lower() == 'noise2noise':
		return Noise2Noise(args.dim_in, 
						   args.dim_out, 
						   args.dim_hidden, 
						   6, 
						   args.num_node).cuda()
	elif args.model_name.lower() == 'ma':
		return MA(args.dim_in, 
				  args.dim_out, 
				  args.num_node, 
				  args.window_size, 
				  args.on_off).cuda()
	elif args.model_name.lower() == 'median_filter':
		return MedianFilter(args.dim_in, 
				  			args.dim_out, 
				  			args.num_node, 
				  			args.on_off).cuda()
	elif args.model_name.lower() == 'tgsr':
		return TGSR(args.dim_in, 
				  	args.dim_out, 
				  	args.num_node, 
				  	args.on_off).cuda()
	elif args.model_name.lower() == 'graphtrss':
		return GraphTRSS(args.dim_in, 
				  		 args.dim_out, 
				  		 args.num_node, 
				  		 args.on_off).cuda()
	elif args.model_name.lower() == 'gusc':
		return GUSC(args.dim_in, 
					args.dim_out, 
					args.dim_hidden, 
					args.num_hidden, 
					args.num_node).cuda()
	elif args.model_name.lower() == 'gusc1':
		return GUSC1(args.dim_in, 
					args.dim_out, 
					args.dim_hidden, 
					args.num_hidden, 
					args.num_node).cuda()
	elif args.model_name.lower() == 'gutf':
		return GUTF(args.dim_in, 
					args.dim_out, 
					args.dim_hidden, 
					args.num_hidden, 
					args.num_node).cuda()
	elif args.model_name.lower() == 'gutf1':
		return GUTF1(args.dim_in, 
					args.dim_out, 
					args.dim_hidden, 
					args.num_hidden, 
					args.num_node).cuda()
	else:
		return None


def get_data(args):
	## Dataset
	if args.data_name.lower() == 'ieee':
		data_train = IEEE(args.data_path, 'train', args.noise_type, args.noise_ratio, args.poisson_lambda, args.rayleigh_scale, args.gamma_shape, args.gamma_scale)
		data_val = IEEE(args.data_path, 'valid', args.noise_type, args.noise_ratio, args.poisson_lambda, args.rayleigh_scale, args.gamma_shape, args.gamma_scale)
		data_test = IEEE(args.data_path, 'test', args.noise_type, args.noise_ratio, args.poisson_lambda, args.rayleigh_scale, args.gamma_shape, args.gamma_scale)
	elif args.data_name.lower() == 'metr':
		data_train = Traffic(args.data_path, 'train', args.noise_type, args.noise_ratio, args.poisson_lambda, args.rayleigh_scale, args.gamma_shape, args.gamma_scale)
		data_val = Traffic(args.data_path, 'valid', args.noise_type, args.noise_ratio, args.poisson_lambda, args.rayleigh_scale, args.gamma_shape, args.gamma_scale)
		data_test = Traffic(args.data_path, 'test', args.noise_type, args.noise_ratio, args.poisson_lambda, args.rayleigh_scale, args.gamma_shape, args.gamma_scale)
	elif args.data_name.lower() == 'pems':
		data_train = Traffic(args.data_path, 'train', args.noise_type, args.noise_ratio, args.poisson_lambda, args.rayleigh_scale, args.gamma_shape, args.gamma_scale)
		data_val = Traffic(args.data_path, 'valid', args.noise_type, args.noise_ratio, args.poisson_lambda, args.rayleigh_scale, args.gamma_shape, args.gamma_scale)
		data_test = Traffic(args.data_path, 'test', args.noise_type, args.noise_ratio, args.poisson_lambda, args.rayleigh_scale, args.gamma_shape, args.gamma_scale)
	else:
		data_train = None
		data_val = None
		data_test = None
	train = DataLoader(dataset=data_train, 
					   shuffle=True,
					   batch_size=args.batch_size, 
					   pin_memory=True, 
					   drop_last=True)
	valid = DataLoader(dataset=data_val, 
					   shuffle=True, 
					   batch_size=args.batch_size, 
					   pin_memory=True, 
					   drop_last=True)
	test = DataLoader(dataset=data_test, 
					  shuffle=False, 
					  batch_size=1, 
					  pin_memory=True, 
					  drop_last=True)
	return train, valid, test


def get_optim(args, model):
	## Optimizer
	optimizer = optim.Adam(model.parameters(), lr=args.lr)
	return optimizer


def get_metrics(X, Y_p, Y=None):
	## Evaluation Metrics
	if Y is None:
		PS = (Y_p ** 2).mean(1)
		PN = ((X - Y_p) ** 2).mean(1)
		snr = 10 * torch.log10(PS / (PN + 1e-12))
		#print(snr.mean())
		return snr, snr, snr, snr
	else:
		# SNR
		Y_mean = Y.mean(1, keepdim=True)
		PS = (Y ** 2).mean(1)#((Y - Y_mean) ** 2).sum()
		PN = ((Y_p - Y) ** 2).mean(1)#.sum()
		snr = 10 * torch.log10(PS / PN)
		PN_ori = ((X - Y) ** 2).mean(1)#.sum()
		snr_ori = 10 * torch.log10(PS / PN_ori)
		# MAE
		mae = torch.abs(Y_p - Y)
		mae_ori = torch.abs(X - Y)
		#print(mae.shape, snr.shape)
	return snr, snr_ori, mae, mae_ori


def train(args):
	## Train
	logger = get_logger(args)
	model = get_model(args)
	if args.train_noise_type.lower() == 'gaussian':
		noise_name = '{}_{}'.format(args.train_noise_type, args.train_noise_ratio)
	elif args.train_noise_type.lower() == 'poisson':
		noise_name = '{}_{}'.format(args.train_noise_type, args.train_poisson_lambda)
	elif args.train_noise_type.lower() == 'rayleigh':
		noise_name = '{}_{}'.format(args.train_noise_type, args.train_rayleigh_scale)
	elif args.train_noise_type.lower() == 'gamma':
		noise_name = '{}_{}_{}'.format(args.train_noise_type, args.train_gamma_shape, args.train_gamma_scale)
	else:
		noise_name = '{}'.format(args.train_noise_type)
	if args.pretrained:
		check_name = os.path.join(args.model_path, os.path.join(args.data_name, os.path.join(noise_name, args.model_name + '.pth')))
		model.load_state_dict(torch.load(check_name))
	optimizer = get_optim(args, model)
	train, valid, _ = get_data(args)
	best_score, epoch_counter = float('inf'), 0
	for epoch in range(1, args.max_epoch + 1):
		model.train()
		pbar = tqdm(train)
		pbar.write('\x1b[1;35mTraining Epoch\t{:03d}:\x1b[0m'.format(epoch))
		for n, data_batch in enumerate(pbar):
			X = data_batch['X'].cuda().float()
			#print(X.shape)
			A = data_batch['A'].cuda().float()
			#Y = None
			Y = data_batch['Y'].cuda().float()
			optimizer.zero_grad()
			loss = model.cal_loss(X, A)
			loss.backward()
			optimizer.step()
			pbar.write('Epoch\t{:03d}, Iteration\t{:04d}: loss\t{:6.4f}'.format(epoch, n, loss))
		model.eval()
		pbar = tqdm(valid)
		pbar.write('\x1b[1;35mValidation Epoch\t{:03d}:\x1b[0m'.format(epoch))
		snr, snr_ori, mae, mae_ori, k = 0., 0., 0., 0., 0.
		for n, data_batch in enumerate(pbar):
			X = data_batch['X'].cuda().float()
			#Y = None
			Y = data_batch['Y'].cuda().float()
			A = data_batch['A'].cuda().float()
			#print(A.shape)
			with torch.no_grad():
				Y_p, _ = model.forward_seq(X, A)
				snr_n, snr_ori_n, mae_n, mae_ori_n = get_metrics(X, Y_p, Y)
				snr += snr_n.mean()
				snr_ori += snr_ori_n.mean()
				mae += mae_n.mean()
				mae_ori += mae_ori_n.mean()
				k += 1#args.batch_size
		snr = (snr / k).detach().cpu().numpy()
		snr_ori = (snr_ori / k).detach().cpu().numpy()
		mae = (mae / k).detach().cpu().numpy()
		mae_ori = (mae_ori / k).detach().cpu().numpy()
		pbar.write('\x1b[1;35mValidation Epoch\t{:03d}: SNR-{:6.4f}dB\tSNR_ORI-{:6.4f}dB\tMAE-{:6.4f}\tMAE_ORI-{:6.4f}.\x1b[0m.\x1b[0m'.format(epoch, snr, snr_ori, mae, mae_ori))
		logger.info('Epoch\t{:03d}: SNR-{:6.4f}dB\tSNR_ORI-{:6.4f}dB\tMAE-{:6.4f}\tMAE_ORI-{:6.4f}'.format(epoch, snr, snr_ori, mae, mae_ori))
		score = (-snr * 0.1 + mae) / 2
		if score < best_score:
			best_score, epoch_counter = score, 0
			save_to = os.path.join(os.path.join(args.model_path, args.data_name), noise_name)
			if not os.path.exists(save_to):
				os.makedirs(save_to)
			save_to = os.path.join(save_to, args.model_name + '.pth')
			torch.save(model.state_dict(), save_to)
		else: 
			epoch_counter += 1
			if epoch_counter >= args.early_stop:
				break
	for file_handler in logger.handlers:
		logger.removeHandler(file_handler)
		file_handler.close()
	return		


def test(args):
	## Test
	logger = get_logger(args)
	model = get_model(args)
	if args.train_noise_type.lower() == 'gaussian':
		noise_name = '{}_{}'.format(args.train_noise_type, args.train_noise_ratio)
	elif args.train_noise_type.lower() == 'poisson':
		noise_name = '{}_{}'.format(args.train_noise_type, args.train_poisson_lambda)
	elif args.train_noise_type.lower() == 'rayleigh':
		noise_name = '{}_{}'.format(args.train_noise_type, args.train_rayleigh_scale)
	elif args.train_noise_type.lower() == 'gamma':
		noise_name = '{}_{}_{}'.format(args.train_noise_type, args.train_gamma_shape, args.train_gamma_scale)
	else:
		noise_name = '{}'.format(args.train_noise_type)
	if args.model_name != 'ma' and args.model_name != 'median_filter' and args.model_name != 'tgsr' and args.model_name != 'graphtrss':
		check_name = os.path.join(args.model_path, os.path.join(args.data_name, os.path.join(noise_name, args.model_name + '.pth')))
		model.load_state_dict(torch.load(check_name))
	model.eval()
	_, _, test = get_data(args)
	pbar = tqdm(test)
	pbar.write('\x1b[1;35mTesting Phase:\x1b[0m')
	snr, snr_ori, mae, mae_ori, k = 0., 0., 0., 0., 0.
	for n, data_batch in enumerate(pbar):
		X = data_batch['X'].cuda().float()
		#Y = None
		Y = data_batch['Y'].cuda().float()
		A = data_batch['A'].cuda().float()
		with torch.no_grad():
			Y_p, _ = model.forward_seq(X, A)
			snr_n, snr_ori_n, mae_n, mae_ori_n = get_metrics(X, Y_p, Y)
			snr += snr_n.mean()
			snr_ori += snr_ori_n.mean()
			mae += mae_n.mean()
			mae_ori += mae_ori_n.mean()
			k += 1#args.batch_size
			'''
			if (n == 0):
				ts = np.arange(0, X.size()[1])
				plt.plot(np.array(ts), X[0, :, 15, 0].cpu().numpy(), 'r*', np.array(ts), Y[0, :, 15, 0].cpu().numpy(), 'g-', np.array(ts), Y_p[0, :, 15, 0].cpu().numpy())
				#plt.savefig('{}_ori.png'.format(args.model_name))
				#plt.plot(np.array(ts), Y_p[0].cpu().numpy(), 'r*', np.array(ts), Y[0].cpu().numpy())
				if (args.model_name == 'ma'):
					plt.savefig('{}_{}_{}_{}_denoise.png'.format(args.model_name, args.window_size, args.noise_type, args.noise_ratio))
				else:
					plt.savefig('{}_{}_{}_denoise.png'.format(args.model_name, args.noise_type, args.noise_ratio))
				return 
			'''
	snr = (snr / k).detach().cpu().numpy()
	snr_ori = (snr_ori / k).detach().cpu().numpy()
	mae = (mae / k).detach().cpu().numpy()
	mae_ori = (mae_ori / k).detach().cpu().numpy()
	pbar.write('\x1b[1;35mTesting Phase: SNR-{:6.4f}dB\tSNR_ORI-{:6.4f}dB\tMAE-{:6.4f}\tMAE_ORI-{:6.4f}'.format(snr, snr_ori, mae, mae_ori))
	logger.info('Testing: SNR-{:6.4f}dB\tSNR_ORI-{:6.4f}dB\tMAE-{:6.4f}\tMAE_ORI-{:6.4f}'.format(snr, snr_ori, mae, mae_ori))
	for file_handler in logger.handlers:
		logger.removeHandler(file_handler)
		file_handler.close()
	return		
