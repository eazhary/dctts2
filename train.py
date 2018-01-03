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
import numpy as np
import re

def load_vocab():
	#characters = "PSEاإأآبتثجحخدذرزسشصضطظعغفقكلمنهويىؤءةئ ًٌٍَُِّْ،." # Arabic character set
	characters = "PE abcdefghijklmnopqrstuvwxyz'.,?"  # P: Padding E: End of Sentence
	
	char2idx = {char: idx for idx, char in enumerate(characters)}
	idx2char = {idx: char for idx, char in enumerate(characters)}
	return char2idx, idx2char
	
def clean(text):
	text=text.lower()
	re_list = r"[^ abcdefghijklmnopqrstuvwxyz'.,?]" # E: Empty. ignore G
	_text = re.sub(re_list, "", text)
	return(_text)
		
	
def get_data():
	def mypyfunc(text):
		text = text.decode("utf-8")
		items = text.split("|")
		char2idx,_=load_vocab()
		text = items[2].lower()
		text = clean(text)
		source = [char2idx[c] for c in text+'E']
		dest = items[0]
		mels = np.load(os.path.join(hp.data_dir, "mels", dest + ".npy"))
		mags = np.load(os.path.join(hp.data_dir, "mags", dest + ".npy"))
		return np.array(source, dtype=np.int32),mels,mags
	def _pad(text,mel,mag):
		text = tf.pad(text, ((0, hp.maxlen),))[:hp.maxlen] # (Tx,)
		mel = tf.pad(mel, ((0, hp.Ty), (0, 0)))[:hp.Ty] # (Ty, n_mels)
		mag = tf.pad(mag, ((0, hp.Ty), (0, 0)))[:hp.Ty] # (Ty, 1+n_fft/2)
		return text,mel,mag
	#filenames = tf.gfile.Glob("data/*.txt")
	dataset = tf.data.TextLineDataset(tf.convert_to_tensor(hp.metafile))
	dataset = dataset.map(lambda text: tuple(tf.py_func(mypyfunc, [text], [tf.int32, tf.float32, tf.float32])))
	dataset = dataset.map(_pad)
	#dataset = dataset.filter(lambda x,y: tf.less_equal(tf.size(y),hp.maxlen))
	dataset = dataset.batch(hp.batch_size)
	dataset = dataset.repeat()
	iterator = dataset.make_one_shot_iterator()
	next_element = iterator.get_next()
	return(next_element)


def w2(t,n):
	return np.exp(-((n/(180-1) - t/(870-1))**2) / (2 * 0.02**2))
	
def w_fun(t, n):
	return 1 - np.exp(-((n/(hp.maxlen-1) - t/(hp.Ty-1))**2) / (2 * hp.g**2))
    
class Graph():
	def __init__(self, is_training=True):
		self.graph = tf.Graph()
		with self.graph.as_default():
			if is_training:
				self.text, self.mel, self.mag = get_data() # (N, T), (N,Ty,nmels), (N,Ty,nffts)
				self.text = tf.reshape(self.text,shape=[-1,hp.maxlen])
				self.mel = tf.reshape(self.mel,shape=[-1,hp.Ty,hp.n_mels])
				self.mag = tf.reshape(self.mag,shape=[-1,hp.Ty,1+hp.n_fft//2])
				w = np.fromfunction(w_fun, (hp.Ty, hp.maxlen), dtype='f')
				w = np.expand_dims(w,0)
				w = np.repeat(w,hp.batch_size,0)
				self.A_guide = tf.convert_to_tensor(w)
			
#				self.y = tf.reshape(self.y,shape=[-1,hp.Ty,])
			else: # inference
				self.text = tf.placeholder(tf.int32, shape=(None, hp.maxlen))
				self.mel = tf.placeholder(tf.float32, shape=(None,hp.Ty,hp.n_mels))
				w = np.fromfunction(w2, (hp.Ty, hp.maxlen), dtype='f')
				w = np.expand_dims(w,0)
				#w = np.repeat(w,2,0)
				self.A_guide = tf.convert_to_tensor(w)
			
				#self.y = tf.placeholder(tf.int32, shape=(None, hp.maxlen))

			# define decoder inputs
			self.decoder_inputs = tf.concat((tf.zeros_like(self.mel[:, :1,:]), self.mel[:, :-1,:]), 1) # shift mels to right
			char2idx, idx2char = load_vocab()
			with tf.variable_scope("Text2Mel"):
				with tf.variable_scope("TextEnc"):
					self.emb=embedding(self.text,
										vocab_size=len(char2idx), 
										num_units=hp.e,
										scale = False,
										scope="embedding") #in (N,T) out (N,T,e) (32,180,128)
					self.t1=Conv1D(self.emb,hp.d*2,1,1,causal=False,is_training=is_training,activation=tf.nn.relu,scope='c1d-1')
					self.t2=Conv1D(self.t1,hp.d*2,1,1,causal=False,is_training=is_training,scope='c1d-2')
					for i in range(2):
						self.t2=HConv1D(self.t2,hp.d*2,3,1,causal=False,is_training=is_training,scope='hc1d-1-%d'%i)
						self.t2=HConv1D(self.t2,hp.d*2,3,3,causal=False,is_training=is_training,scope='hc1d-2-%d'%i)
						self.t2=HConv1D(self.t2,hp.d*2,3,9,causal=False,is_training=is_training,scope='hc1d-3-%d'%i)
						self.t2=HConv1D(self.t2,hp.d*2,3,27,causal=False,is_training=is_training,scope='hc1d-4-%d'%i)
					self.textenc=self.t2	
					for i in range(2):
						self.textenc=HConv1D(self.textenc,hp.d*2,3,1,causal=False,is_training=is_training,scope='hc1d-11-%d'%i)
					for i in range(2):
						self.textenc=HConv1D(self.textenc,hp.d*2,1,1,causal=False,is_training=is_training,scope='hc1d-12-%d'%i) #(N,T,2*d) (32,180,512)

					self.t3 =self.textenc	

					self.K,self.V = tf.split(self.textenc,num_or_size_splits=2,axis=2)	#k=(B,N,d) v=(B,N,d)
				with tf.variable_scope("AudioEnc"):
					self.audioenc = Conv1D(self.decoder_inputs,hp.d,1,1,is_training=is_training,activation=tf.nn.relu,scope='c1d-1') # from (B,Ty,80) -> (B,Ty,d)
					self.audioenc = Conv1D(self.audioenc,hp.d,1,1,is_training=is_training,activation=tf.nn.relu,scope='c1d-2')
					self.audioenc = Conv1D(self.audioenc,hp.d,1,1,is_training=is_training,scope='c1d-3')
					for i in range(2):
						self.audioenc=HConv1D(self.audioenc,hp.d,3,1,is_training=is_training,scope='hc1d-1-%d'%i)
						self.audioenc=HConv1D(self.audioenc,hp.d,3,3,is_training=is_training,scope='hc1d-2-%d'%i)
						self.audioenc=HConv1D(self.audioenc,hp.d,3,9,is_training=is_training,scope='hc1d-3-%d'%i)
						self.audioenc=HConv1D(self.audioenc,hp.d,3,27,is_training=is_training,scope='hc1d-4-%d'%i)
					for i in range(2):
						self.audioenc=HConv1D(self.audioenc,hp.d,3,3,is_training=is_training,scope='hc1d-11-%d'%i)
					self.Q = self.audioenc										# (B,Ty,d)
				self.A = tf.matmul(self.Q,self.K,transpose_b=True)	  # (B,Ty,d) * (B,N,d).T  -> (B,T,N) ---- d,N Ty,d
#				self.A *= tf.sqrt(1/tf.to_float(hp.d))
				self.A = tf.nn.softmax(self.A/tf.sqrt(tf.to_float(hp.d)))				# (B,Ty,N) (32,870,180)
				if not is_training:
					self.A = self.A_guide			
				self.R = tf.matmul(self.A,self.V)			# (B,Ty,N) * (B,N,d) = (B,Ty,d)
				self.Rhat = tf.concat((self.R,self.Q),2)		# (B,Ty,d),(B,Ty,d) --> (B,Ty,2d)
				with tf.variable_scope("AudioDec"):
					self.audiodec = Conv1D(self.Rhat,hp.d,1,1,is_training=is_training,scope='c1d-1')
					self.audiodec=HConv1D(self.audiodec,hp.d,3,1,is_training=is_training,scope='hc1d-1')
					self.audiodec=HConv1D(self.audiodec,hp.d,3,3,is_training=is_training,scope='hc1d-2')
					self.audiodec=HConv1D(self.audiodec,hp.d,3,9,is_training=is_training,scope='hc1d-3')
					self.audiodec=HConv1D(self.audiodec,hp.d,3,27,is_training=is_training,scope='hc1d-4')
					for i in range(2):
						self.audiodec=HConv1D(self.audiodec,hp.d,3,1,is_training=is_training,scope='hc1d-5-%d'%i)
					for i in range(3):
						self.audiodec=Conv1D(self.audiodec,hp.d,1,1,dropout=0,is_training=is_training,scope='c1d-2-%d'%i,activation=tf.nn.relu)
					self.mel_logits = Conv1D(self.audiodec,hp.n_mels,1,1,dropout=0,is_training=is_training,scope='c1d-3') # (B,Ty,nmels)
					self.mel_output = tf.nn.sigmoid(self.mel_logits)
							
			
			if is_training:	 
				# Loss
				self.global_step = tf.Variable(0, name='global_step', trainable=False)
				self.learning_rate = _learning_rate_decay(self.global_step)
				self.l1_loss = tf.reduce_mean(tf.abs(self.mel_output - self.mel))
				self.bin_div = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.mel_logits,labels=self.mel))
				self.A_loss = tf.reduce_mean(tf.abs(self.A_guide*self.A))
				self.loss_mels = self.l1_loss+self.bin_div+self.A_loss
				self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=hp.b1, beta2=hp.b2, epsilon=hp.eps)
				self.gvs = self.optimizer.compute_gradients(self.loss_mels) 
				self.clipped = []
				for grad, var in self.gvs:
					if grad is not None:
					#grad = tf.clip_by_value(grad, -1. * hp.max_grad_val, hp.max_grad_val)
						grad = tf.clip_by_norm(grad, hp.max_grad_norm)
						
					self.clipped.append((grad, var))
				self.train_op = self.optimizer.apply_gradients(self.clipped, global_step=self.global_step)
				#self.train_op = self.optimizer.apply_gradients(self.clipped, global_step=self.global_step)
				
				#self.train_op = self.optimizer.minimize(self.loss_mels, global_step=self.global_step)
				tf.summary.scalar('loss_mels', self.loss_mels)
				tf.summary.scalar('loss_l1', self.l1_loss)
				tf.summary.scalar('learning_rate', self.learning_rate)
				tf.summary.scalar('loss_binary', self.bin_div)
				tf.summary.scalar('loss_Attention', self.A_loss)
			self.merged = tf.summary.merge_all()

def show(mel1,mel2,name):
	plt.figure(figsize=(20,4))
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
	fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(20,4))
	cax = ax.matshow(np.transpose(mel), interpolation='nearest',  cmap=plt.cm.afmhot, origin='lower')
	fig.colorbar(cax)
	plt.title(msg)
	plt.savefig(file,format='png')
	plt.cla()
	plt.close('all')

def _learning_rate_decay(global_step):
  # Noam scheme from tensor2tensor:
  step = tf.cast(global_step + 1, dtype=tf.float32)
  return 0.0002
  #return hp.c**-0.5 * tf.minimum(step * hp.warmup_steps**-1.5, step**-0.5)


def tdecode(text):
	char2idx,idx2char=load_vocab()
	return("".join(idx2char[i] for i in text).split('P')[0])

				
if __name__ == '__main__':	
	g = Graph(); print("Training Graph loaded")
	sv = tf.train.Supervisor(graph=g.graph, 
							 logdir=hp.logdir,)
							 #save_model_secs=0)
	
	with sv.managed_session() as sess:
		while not sv.should_stop(): 
			gs,loss,l1,bin,A_loss,mels,inp,A,guide,text,ops=sess.run([g.global_step,g.loss_mels,g.l1_loss,g.bin_div,g.A_loss,g.mel_output,g.mel,g.A,g.A_guide,g.text,g.train_op])
#			gs, text = sess.run([g.global_step,g.text])
			message = "Step %-7d : loss=%.05f,l1=%.05f,bin=%.05f,A_loss=%.05f" % (gs,loss,l1,bin,A_loss)
			print(message)
			if gs % 10 == 0:
				show(mels[0],inp[0],"0.png")
				show(mels[1],inp[1],"1.png")
				showmels(A[0],tdecode(text[0]),"a0.png")
				showmels(A[1],tdecode(text[1]),"a1.png")
				showmels(guide[0],tdecode(text[0]),"m0.png")
				

	# next_element = get_data()
	# with tf.Session() as sess:
		# for i in range(500):
			# s = time.time()
			# text,mel,mag = sess.run(next_element)
			# e = time.time()
			# print("took:",e-s,text.shape,mel.shape,mag.shape)
	print("Done")	 
	

