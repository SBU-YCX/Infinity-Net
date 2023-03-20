##############################################
#    Author : Yucheng Xing
#    Description : Train
##############################################
import argparse, os, logging, torch
import numpy as np
from torch import optim
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from Processing import RTDS
from Models import InfinityNet





def get_logger(log_file):
	logging.basicConfig(level=logging.DEBUG)
	logger = logging.getLogger()
	logger.setLevel(logging.DEBUG)
	if log_file:
		file_handler = logging.FileHandler(log_file)
		logger.addHandler(file_handler)
	return logger


def get_dataset():
	if args.data_name.lower() == 'rtds':
		train_set = RTDS('train', args.data_path)
		valid_set = RTDS('valid', args.data_path)
		test_set = RTDS('test', args.data_path)
	train = DataLoader(dataset=train_set, shuffle=True, batch_size=args.batch_size, pin_memory=True)

def get_model():


def get_metric():


def train():
