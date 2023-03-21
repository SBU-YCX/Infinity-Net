##############################################
#    Author : Yucheng Xing
#    Description : Train
##############################################
import argparse, os, logging, torch
import numpy as np
from torch import optim
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from Processing import RTDS, BCM
from Models import InfinityNet


parser = argparse.ArgumentParser(description='Experiment Configuration.')
parser.add_argument('--data_name', required=False, default='bcm', type=str, help='')
parser.add_argument('--data_path', required=False, default='', type=str, help='')
parser.add_argument('--model_name', required=False, default='infinitynet', type=str, help='')
parser.add_argument('--model_path', required=False, default='', type=str, help='')
parser.add_argument('--pre_trained', required=False, default=False, type=bool, help='')
parser.add_argument('--optim', required=False, default='adam', type=str, help='')
parser.add_argument('--log_path', required=False, default='../Log', type=str, help='')
parser.add_argument('--max_epoch', required=False, default=100, type=int, help='')
parser.add_argument('--early_stop', required=False, default=5, type=int, help='')
parser.add_argument('--lr', required=False, default=1e-2, type=float, help='')
parser.add_argument('--batch_size', required=False, default=30, type=int, help='')
parser.add_argument('--dim_in', required=False, default=6, type=int, help='')
parser.add_argument('--dim_out', required=False, default=6, type=int, help='')
parser.add_argument('--dim_hidden', required=False, default=12, type=int, help='')


def get_logger(log_file):
	logging.basicConfig(level=logging.DEBUG)
	logger = logging.getLogger()
	logger.setLevel(logging.DEBUG)
	if log_file:
		file_handler = logging.FileHandler(log_file)
		logger.addHandler(file_handler)
	return logger


def get_dataset(args):
	if args.data_name.lower() == 'rtds':
		train_set = RTDS('train', args.data_path)
		valid_set = RTDS('valid', args.data_path)
		test_set = RTDS('test', args.data_path)
	elif args.data_name.lower() == 'bcm':
		train_set = BCM('train', args.data_path)
		valid_set = BCM('valid', args.data_path)
		test_set = BCM('test', args.data_path)
	train = DataLoader(dataset=train_set, shuffle=True, batch_size=args.batch_size, pin_memory=True)
	valid = DataLoader(dataset=valid_set, shuffle=False, batch_size=args.batch_size, pin_memory=True)
	test = DataLoader(dataset=test_set, shuffle=False, batch_size=1, pin_memory=True)
	return train, valid, test


def get_model(args):
	if args.model_name.lower() == 'infinitynet':
		model = InfinityNet(dim_in=args.dim_in, 
							dim_hidden=args.dim_hidden, 
							dim_out=args.dim_out)
	else:
		model = None
	if torch.cuda.is_available():
		return model.cuda()
	return model


def get_optim(args, model):
	if args.optim == 'adam':
		optimizer = optim.Adam(model.parameters(), lr=args.lr)
	elif args.optim == 'sgd':
		optimizer = optim.SGD(model.parameters(), lr=args.lr)
	return optimizer


def get_metric():
	evaluator = nn.MSELoss()


def train(args):
	## Get Logger
	log_folder = os.path.join(args.log_path, args.data_name)
	if not os.path.exists(log_folder):
		os.makedirs(log_folder)
	log_file = os.path.join(log_folder, args.model_name.lower() + '_{}.log'.format(args.batch_size))
	logger = get_logger(log_file)
	## Get Model
	model = get_model(args)
	if args.pre_trained:
		if torch.cuda.is_available():
			model.load_state_dict(torch.load(args.model_path))
		else:
			model.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu')))
	## Get Data
	train_set, valid_set, _ = get_dataset(args)
	## Get Optimizer
	optimizer = get_optim(args, model)
	## Start Training
	best_score, bad_epoch = float('inf'), 0
	for epoch in range(1, args.max_epoch + 1):
		# Train
		model.train()
		pbar = tqdm(train_set)
		pbar.write('\x1b[1;35mTraining Epoch\t%03d:\x1b[0m' % epoch)
		for n, data_batch in enumerate(pbar):
			x = data_batch['data']
			A = data_batch['adj_mat']
			if torch.cuda.is_available():
				x = x.cuda()
				A = A.cuda()
			optimizer.zero_grad()
			loss = model.get_loss(x, A)
			loss.backward()
			optimizer.step()
			pbar.write('Epoch\t{:04d}, Iteration\t{:04d}: Loss\t{:6.4f}'.format(epoch, n, loss))
		# Valid
		model.eval()
		pbar = tqdm(valid_set)
		pbar.write('\x1b[1;35mValidation Epoch\t%03d:\x1b[0m' % epoch)
		accumulate_loss, k = 0., 0.
		for n, data_batch in enumerate(pbar):
			x = data_batch['data']
			A = data_batch['adj_mat']
			if torch.cuda.is_available():
				x = x.cuda()
				A = A.cuda()
			with torch.no_grad():
				loss = model.get_loss(x, A)
				accumulate_loss += loss
				k += 1
		avg_loss = (accumulate_loss / k).detach().cpu().numpy()
		pbar.write('Valid-Epoch\t{:04d}: Loss\t{:6.4f}'.format(epoch, avg_loss))
		logger.info('Valid-Epoch\t{:04d}: Loss\t{:6.4f}'.format(epoch, avg_loss))
		if avg_loss <= best_score:
			best_score = avg_loss
			if args.model_path:
				torch.save(model.state_dict(), args.model_path)
			bad_epoch = 0
		else:
			bad_epoch += 1
		if bad_epoch >= args.early_stop:
			break
	for handler in logger.handlers[:]:
		logger.removeHandler(handler)
		handler.close()
	return


def test(args):
	## Get Logger
	log_folder = os.path.join(args.log_path, args.data_name)
	log_file = os.path.join(log_folder, args.model_name.lower() + '_{}.log'.format(args.batch_size))
	logger = get_logger(log_file)
	## Get Model
	model = get_model(args)
	if torch.cuda.is_available():
		model.load_state_dict(torch.load(args.model_path))
	else:
		model.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu')))
	model.eval()
	## Get Data
	_, _, test_set = get_dataset(args)
	## Get Evaluation Metric
	metric = get_metric()
	## Start Testing
	accumulate_error, k = 0., 0., 0.
	pbar = tqdm(test_set)
	pbar.write('\x1b[1;35mTesting:\x1b[0m')
	for n, data_batch in enumerate(pbar):
		x = data_batch['data']
		y = data_batch['ori_data']
		A = data_batch['adj_mat']
		if torch.cuda.is_available():
			x = x.cuda()
			y = y.cuda()
			A = A.cuda()
		with torch.no_grad():
			y_p = model.forward_seq(x, A)
			accumulate_error += metric(y_p, y).mean().item()
			k += 1
	avg_error = (accumulate_error / k)
	pbar.write('Testing:\tError\t{:6.4f}'.format(avg_error))
	logger.info('Testing:\tError\t{:6.4f}'.format(avg_error))
	for handler in logger.handlers[:]:
		logger.removeHandler(handler)
		handler.close()
	return