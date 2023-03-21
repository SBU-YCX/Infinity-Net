import os
from Train import parser, train, test


args = parser.parse_args()

args.batch_size = 30
args.max_epoch = 100
args.early_stop = 5
args.optim = 'adam'

args.data_name = 'bcm'
args.data_path = '../Data/BCM/'

args.model_name = 'infinitynet'
args.model_path = '../Model/{}/'.format(args.data_name)
if not os.path.exists(args.model_path):
	os.mkdir(args.model_path)
args.model_path = os.path.join(args.model_path, '{}_{}.pth'.format(args.model_name, args.batch_size))

args.dim_in = 6
args.dim_out = 6
args.dim_hidden = 12

args.pre_trained = False
args.lr = 1e-2
train(args)
args.pre_trained = True
test(args)
args.lr = 1e-3
train(args)
test(args)
args.lr = 1e-4
train(args)
test(args)