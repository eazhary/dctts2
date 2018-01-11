# -*- coding: utf-8 -*-

import os

import tensorflow as tf
import numpy as np

from hyperparams import Hyperparams as hp
from train import Graph, load_vocab, clean, showmels, show

def eval(): 
	# Load graph
	g = Graph(is_training=False)
	print("Graph loaded")
	
	# Load data
	#X, Sources, Targets = load_test_data()
	char2idx, idx2char = load_vocab()
	dest = 'LJ017-0105'
	inp = "Cook's death was horrible"
	mel = np.load(os.path.join(hp.data_dir, "mels", dest + ".npy"))
	mel = mel[::4,:]
	mels = np.zeros(shape=(hp.Tyr,hp.n_mels))
	mels[:mel.shape[0],:mel.shape[1]]=mel
		
	#inp = "Printing, in the only sense with which we are at present concerned, differs from most if not from all the arts and crafts represented in the Exhibition"
	inp = clean(inp)
	x = [char2idx[c] for c in inp+'E']
	x += [0]*(hp.maxlen-len(x))
	x = np.array(x)
	x = x.reshape(1,-1)
#	x = x.repeat(hp.batch_size,axis=0)
#	  X, Sources, Targets = X[:33], Sources[:33], Targets[:33]
	 
	# Start session			
	with g.graph.as_default():
		with tf.Session() as sess:
			saver = tf.train.Saver()
			saver.restore(sess, tf.train.latest_checkpoint(hp.logdir));
			print("Restored")
			preds = np.zeros((1, hp.Tyr, hp.n_mels), np.float32)
			for j in range(200):
				print("Processing %d"%j)
				_preds,a = sess.run([g.mel_output, g.A], {g.text: x, g.mel: preds})
				preds[:,j,:] = _preds[:,j,:]
				#show(preds[0],mels,"pred%d.png"%j)
			#	showmels(a[0],"Attention","att%d.png"%j)
				#preds = _preds
			show(preds[0],mels,"predicted.png")
							  
if __name__ == '__main__':
	eval()
	print("Done")
	
	
