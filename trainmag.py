# -*- coding: utf-8 -*-
'''
June 2017 by kyubyong park. 
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer
'''
from __future__ import print_function
import tensorflow as tf
from matplotlib import pyplot as plt
from hyperparams import Hyperparams as hp
from modules import *
import os
import time
import sys

import numpy as np
import re
import audio

	
def get_data():
	def mypyfunc(text):
		text = text.decode("utf-8")
		items = text.split("|")
		dest = items[0]
		mels = np.load(os.path.join(hp.data_dir, "mels", dest + ".npy"))
		mels = mels[::4,:]
		mags = np.load(os.path.join(hp.data_dir, "mags", dest + ".npy"))
		return mels,mags
	def _pad(mel,mag):
		mel = tf.pad(mel, ((0, hp.Tyr), (0, 0)))[:hp.Tyr] # (Tyr, n_mels)
		mag = tf.pad(mag, ((0, hp.Ty), (0, 0)))[:hp.Ty] # (Ty, 1+n_fft/2)
		return mel,mag
	dataset = tf.data.TextLineDataset(tf.convert_to_tensor(hp.metafile))
	dataset = dataset.map(lambda text: tuple(tf.py_func(mypyfunc, [text], [tf.float32, tf.float32])))
	dataset = dataset.map(_pad)
	dataset = dataset.repeat()
	dataset = dataset.batch(hp.batch_size)
	iterator = dataset.make_one_shot_iterator()
	next_element = iterator.get_next()
	return(next_element)


class Graph():
	def __init__(self, is_training=True):
		self.graph = tf.Graph()
		with self.graph.as_default():
			if is_training:
				self.mel, self.mag=get_data() # (N,Tyr,nmels), (N,Ty,1+n_ffts//2)
				self.mel = tf.reshape(self.mel,shape=[-1,hp.Tyr,hp.n_mels])
			else: # inference
				self.mel = tf.placeholder(tf.float32, shape=(None,None,hp.n_mels))
			with tf.variable_scope("SSRN"):
				self.ssrn = Conv1D(self.mel,hp.c,1,1,causal=False,is_training=is_training,scope='c1d-1')
				self.ssrn = HConv1D(self.ssrn,hp.c,3,1,causal=False,is_training=is_training,scope='hc1d-1')
				self.ssrn = HConv1D(self.ssrn,hp.c,3,3,causal=False,is_training=is_training,scope='hc1d-2')
				for i in range(2):
					self.ssrn = Deconv1D(self.ssrn,hp.c,2,1,scope='deconv-%d'%i)
					self.ssrn = HConv1D(self.ssrn,hp.c,3,1,causal=False,is_training=is_training,scope='hc1d-31-%d'%i)
					self.ssrn = HConv1D(self.ssrn,hp.c,3,3,causal=False,is_training=is_training,scope='hc1d-32-%d'%i)
				self.ssrn = Conv1D(self.ssrn,hp.c*2,1,1,causal=False,is_training=is_training,scope='c1d-2')
				for i in range(2):
					self.ssrn=HConv1D(self.ssrn,hp.c*2,3,1,causal=False,is_training=is_training,scope='hc1d-4-%d'%i)
				self.ssrn = Conv1D(self.ssrn,hp.fd,1,1,causal=False,is_training=is_training,scope='c1d-3')
				for i in range(2):
					self.ssrn=Conv1D(self.ssrn,hp.fd,1,1,causal=False,is_training=is_training,activation=tf.nn.relu,scope='c1d-4-%d'%i)
				self.mag_logits = Conv1D(self.ssrn,hp.fd,1,1,causal=False,is_training=is_training,scope='c1d-5')
				self.mag_output = tf.nn.sigmoid(self.mag_logits)
			if is_training:	 
				# Loss
				self.global_step = tf.Variable(0, name='global_step', trainable=False)
				self.learning_rate = _learning_rate_decay(self.global_step)
				
				
				self.mag_l1_loss = tf.reduce_mean(tf.abs(self.mag-self.mag_output))
				self.mag_bin_div = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.mag_logits,labels=self.mag))
				
				
				self.loss_mags = self.mag_l1_loss + self.mag_bin_div
				self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=hp.b1, beta2=hp.b2, epsilon=hp.eps)
#				self.gvs = self.optimizer.compute_gradients(self.loss_mels) 
#				self.clipped = []
#				for grad, var in self.gvs:
#					if grad is not None:
#						grad = tf.clip_by_norm(grad, hp.max_grad_norm)
#						
#					self.clipped.append((grad, var))
#				self.train_op = self.optimizer.apply_gradients(self.clipped, global_step=self.global_step)
				self.train_mag = self.optimizer.minimize(self.loss_mags,global_step=self.global_step)
				tf.summary.scalar('loss_mags', self.loss_mags)
				tf.summary.scalar('loss_mag_binary', self.mag_bin_div)
				tf.summary.scalar('loss_mag_l1', self.mag_l1_loss)
				tf.summary.scalar('learning_rate', self.learning_rate)
			self.merged = tf.summary.merge_all()

def show(mel1,mel2,name):
	plt.figure(figsize=(8,4))
	plt.subplot(2,1,1)
	plt.imshow(np.transpose(mel1),interpolation='nearest',  cmap=plt.cm.afmhot, origin='lower')
	plt.title("Generated")
	plt.colorbar()
	plt.subplot(2,1,2)
	plt.imshow(np.transpose(mel2),interpolation='nearest',  cmap=plt.cm.afmhot, origin='lower')
	plt.title("Original")
	plt.colorbar()
	plt.savefig(name)
	plt.cla()
	plt.close('all')

			
def showmels(mel,msg,file):
	fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(8,4))
	cax = ax.matshow(mel, interpolation='nearest',  cmap=plt.cm.afmhot, origin='lower')
	fig.colorbar(cax)
	plt.title(msg+str(len(msg)))
	plt.savefig(file,format='png')
	plt.cla()
	plt.close('all')

def _learning_rate_decay(global_step):
  # Noam scheme from tensor2tensor:
  step = tf.cast(global_step + 1, dtype=tf.float32)
  return hp.init_lr * hp.warmup_steps**0.5 * tf.minimum(step * hp.warmup_steps**-1.5, step**-0.5)


				
if __name__ == '__main__':	
	g = Graph(); print("Training Graph loaded")
	sv = tf.train.Supervisor(graph=g.graph, 
							 logdir=hp.logdirmag,)
							 #save_model_secs=0)
	with sv.managed_session() as sess:
		while not sv.should_stop():
			gs,l_M,l_M_l1,l_M_b,ops = sess.run([g.global_step,
				g.loss_mags,g.mag_l1_loss,g.mag_bin_div,g.train_mag])
			message = "Step %d : l=%.05f (Ml1=%.05f,Mb=%.05f)" % (gs,l_M,l_M_l1,l_M_b)
			sys.stdout.write('\r'+message)
			sys.stdout.flush()
			#print(message)
			if (gs+1) % hp.logevery == 0:
				gs,l_M,l_M_l1,l_M_b,M_o,M_i,ops = sess.run([g.global_step,
					g.loss_mags,g.mag_l1_loss,g.mag_bin_div,
					g.mag_output, g.mag,g.train_mag])
				message = "Step %d : l=%.05f (Ml1=%.05f,Mb=%.05f)" % (gs,l_M,l_M_l1,l_M_b)
				sys.stdout.write('\r'+message)
				sys.stdout.flush()
#				audio.save_spec(M_o[0].T,"out0.wav")
#				audio.save_spec(M_o[1].T,"out1.wav")
				show(M_o[0],M_i[0],"mag0.png")		
				show(M_o[1],M_i[1],"mag1.png")		
			pass
				

	print("Done")	 
	

