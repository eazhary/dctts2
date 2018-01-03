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
	mels = np.zeros(shape=(hp.Ty,hp.n_mels))
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
		sv = tf.train.Supervisor(logdir=hp.logdir)
		with sv.managed_session() as sess:
			preds = np.zeros((1, hp.Ty, hp.n_mels), np.float32)
			for j in range(200):
				print("Processing %d"%j)
				_preds,a = sess.run([g.mel_output, g.A], {g.text: x, g.mel: preds})
				preds[:,j,:] = _preds[:,j,:]
				#show(preds[0],mels,"pred%d.png"%j)
				#showmels(a[0],"Attention","att%d.png"%j)
				#preds = _preds
			show(preds[0],mels,"predicted.png")
			## Inference
			# while (1):
					# testVar = input("input:")
					# x = [char2idx[c] for c in testVar+'E']
					# x+=[0]*(hp.maxlen-len(x))
					# x = np.array(x)
					# x = x.reshape(1,-1)
					# #preds = np.zeros((hp.batch_size, hp.maxlen), np.int32)
					# preds = np.zeros((1, hp.maxlen), np.int32)
					# for j in range(hp.maxlen):
							# _preds = sess.run(g.preds, {g.x: x, g.y: preds})
							# #print(j,"->","".join(idx2char[idx] for idx in _preds[0]).split("E")[0].strip())
							# preds[:, j] = _preds[:, j]
					# got = "".join(idx2char[idx] for idx in preds[0]).split("E")[0].strip()
					# print("Source: ",testVar)
					# print("got : ", got)
							  
if __name__ == '__main__':
	eval()
	print("Done")
	
	
