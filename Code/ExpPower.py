import os
from Execution import parser, train, test

args = parser.parse_args()
args.noise_ratio = 0.5#0.3#0.1#
args.poisson_lambda = 5
args.rayleigh_scale = 1.0
args.gamma_shape = 2
args.gamma_scale = 1
args.noise_type = 'poisson'#'gaussian'#'ratio'#'mixture'#'gamma'#'rayleigh'#
args.train_poisson_lambda = 5
args.train_rayleigh_scale = 1.0
args.train_gamma_shape = 2
args.train_gamma_scale = 1
args.train_noise_ratio = 0.1#0.5#0.3#
args.train_noise_type = 'gaussian'#'ratio'#'mixture'#'poisson'#'gamma'#'rayleigh'#
args.data_path = "/home/yucxing/exp/C004_ICONIP23/EvolveSDE/datasets/RTDS/rtds_noisy_len100.hdf5"#"/home/yucxing/exp/C004_ICONIP23/EvolveSDE/datasets/RTDS/rtds_len100.hdf5"#
args.data_name = 'ieee'
args.num_node = 33
args.window_size = 2#5#10#
args.num_hidden = 3
args.on_off = 'on'#'off'#
args.model_name = 'infinity_net'#'gutf'#'gusc1'#'gutf1'#'gusc'#'graphtrss'#'tgsr'#'median_filter'#'noise2noise'#'ma'#
args.batch_size = 32
args.early_stop = 500
args.max_epoch = 500#20#
args.lr = 5e-3
#train(args)
#test(args)
args.pretrained = True
args.lr = 1e-3
#train(args)
#test(args)
args.pretrained = True
args.lr = 5e-4
#train(args)
#test(args)
args.pretrained = True
args.lr = 1e-4
#train(args)
test(args)
