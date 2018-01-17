# -*- coding: utf-8 -*-

import os

import tensorflow as tf
import numpy as np
import sys
from hyperparams import Hyperparams as hp
from train import Graph, load_vocab, clean, showmels, show

def eval(): 
	# Load graph
	g = Graph(is_training=False)
	print("Graph loaded")
	
	# Load data
	#X, Sources, Targets = load_test_data()
	char2idx, idx2char = load_vocab()
#	inp = "For although the Chinese took impressions from wood blocks engraved in relief for centuries before the woodcutters of the Netherlands, by a similar process"
#	dest = 'LJ001-0003'
#	dest = 'LJ017-0105'
#	inp = "Cook's death was horrible"
	inp = "Printing, in the only sense with which we are at present concerned, differs from most if not from all the arts and crafts represented in the Exhibition"
	dest = 'LJ001-0001'
#	inp = "one two three one two three one two three one two three one two three one two three one two three one two three"
	mel = np.load(os.path.join(hp.data_dir, "mels", dest + ".npy"))
	mel = mel[::4,:]
	mels = np.zeros(shape=(hp.Tyr,hp.n_mels))
	mels[:mel.shape[0],:mel.shape[1]]=mel
	mels=mels.reshape(1,-1,80)
		
	#inp = "Printing, in the only sense with which we are at present concerned, differs from most if not from all the arts and crafts represented in the Exhibition"
	inp = clean(inp)
	print(inp)
	x = [char2idx[c] for c in inp+'E']
	x += [0]*(hp.maxlen-len(x))
	print(x)
	x = np.array(x)
	x = x.reshape(1,-1)
	print(x.shape)
	#quit()
#	x = x.repeat(hp.batch_size,axis=0)
#	  X, Sources, Targets = X[:33], Sources[:33], Targets[:33]
	 
	# Start session			
	with g.graph.as_default():
		with tf.Session() as sess:
			saver = tf.train.Saver()
			saver.restore(sess, tf.train.latest_checkpoint(hp.logdir));
			print("Restored")
			#preds = np.concatenate((np.zeros((1, 1, hp.n_mels), np.float32),mels[:,:100,:]),axis=1)
			preds = np.zeros((1, 1, hp.n_mels), np.float32)
			cnt = hp.Tyr
			for j in range(hp.Tyr):
				sys.stdout.write('\rProcessing %d' % j)
				sys.stdout.flush()
				#print("Input Shape is ",preds.shape)
				_preds,a = sess.run([g.mel_output, g.A], {g.text: x, g.mel: preds})
				#print("Output shape is ", _preds.shape)
				preds = np.concatenate((np.zeros((1,1,hp.n_mels)),_preds),axis=1)  
				cnt -=1
				if np.argmax(a[0,:,-1]) >= len(inp)-3:
					cnt = min(cnt,10)
				if cnt<=0:
					break
					
				#show(preds[0],mel,"pred%d.png"%j)
				#showmels(a[0],"Attention","att%d.png"%j)
				#preds = _preds
			show(preds[0],mels[0],"predicted.png")
			showmels(a[0],"Attention","attfinal.png")
							  
if __name__ == '__main__':
	eval()
	print("Done")
	
	
