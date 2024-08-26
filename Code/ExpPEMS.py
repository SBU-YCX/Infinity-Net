import os
from Execution import parser, train, test

args = parser.parse_args()
args.noise_ratio = 0.5#0.3#0.1#
args.poisson_lambda = 5
args.rayleigh_scale = 1.0
args.gamma_shape = 2
args.gamma_scale = 1
args.noise_type = 'mixture'#'gamma'#'rayleigh'#'poisson'#'gaussian'#'ratio'#
args.train_poisson_lambda = 5
args.train_rayleigh_scale = 1.0
args.train_gamma_shape = 2
args.train_gamma_scale = 1
args.train_noise_ratio = 0.5#0.3#0.1#
args.train_noise_type = 'mixture'#'poisson'#'gamma'#'rayleigh'#'gaussian'#'ratio'#
args.data_path = "/home/yucxing/exp/C004_ICONIP23/EvolveSDE/datasets/Traffic/pems.hdf5"
args.data_name = 'pems'
args.num_node = 325
args.dim_in = 1
args.dim_out = 1
args.window_size = 2#5#10#
args.num_hidden = 3
args.on_off = 'on'#'off'#
args.model_name = 'infinity_net'#'gutf'#'gutf1'#'noise2noise'#'median_filter'#'graphtrss'#'tgsr'#'ma'#'gusc1'#'gusc'#
args.batch_size = 32
args.early_stop = 500
args.max_epoch = 500#20#
args.lr = 5e-3
train(args)
test(args)
args.pretrained = True
args.lr = 1e-3
train(args)
test(args)
args.pretrained = True
args.lr = 5e-4
train(args)
test(args)
args.pretrained = True
args.lr = 1e-4
train(args)
test(args)
