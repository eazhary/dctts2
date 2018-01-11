# -*- coding: utf-8 -*-
class Hyperparams:
	'''Hyperparameters'''
	data_dir = 'LJSpeech-1.0/'
	metafile = 'LJSpeech-1.0/metadata.csv'
	#metafile = 'LJSpeech-1.0/m.csv'
	batch_size = 2 # alias = N
	warmup_steps = 200
	logdir = 'logdir' # log directory
	sr = 22050
	n_fft = 2048 # fft points (samples)
	frame_shift = 0.0125 # seconds
	frame_length = 0.05 # seconds
	hop_length = 256 # samples	This is dependent on the frame_shift.
	win_length = 1024 # samples This is dependent on the frame_length.
	n_mels = 80 # Number of Mel banks to generate
	sharpening_factor = 1.4 # Exponent for amplifying the predicted magnitude
	n_iter = 50 # Number of inversion iterations
	preemphasis = .97 # or None
	max_db = 100
	ref_db = 20
	max_grad_norm = 100.
	max_grad_val = 5.
	
	# model
	maxlen = 180 # Maximum number of letters in a sentance = T.
	Ty = 868 # Max number of timesteps 
	Tyr = 217 # Max number of timesteps 
	e = 128
	d = 256
	c = 512
	lr = 2e-4
	g=0.2
	b1 = 0.5
	b2 = 0.9
	eps = 1e-6
	
	dropout_rate = 0.1
	
	
	
	
