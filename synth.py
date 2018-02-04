# -*- coding: utf-8 -*-

import os
import argparse
import tensorflow as tf
import numpy as np
import io
import sys
from hyperparams import Hyperparams as hp
from trainmel import load_vocab, clean, showmels, show
from trainmel import Graph as melmodel
from trainmag import Graph as magmodel
import audio


class Synth:
	def __init__(self):
		self.melmodel = melmodel(is_training=False)
		self.magmodel = magmodel(is_training=False)
		print("Graphs Loaded")
		self.c2i,_ = load_vocab()
		self.melsession = tf.Session(graph=self.melmodel.graph)		
		self.magsession = tf.Session(graph=self.magmodel.graph)
		with self.melsession.as_default():
			with self.melmodel.graph.as_default():
				saver = tf.train.Saver()
				saver.restore(self.melsession,tf.train.latest_checkpoint(hp.logdirmel))
		print("Restored Mels")
		with self.magsession.as_default():
			with self.magmodel.graph.as_default():
				saver = tf.train.Saver()
				saver.restore(self.magsession,tf.train.latest_checkpoint(hp.logdirmag))
		print("Restored Mags")
	def synth(self,text,save=None):
		inp = clean(text)
		print(inp)
		x = [self.c2i[c] for c in inp+'E']
		x += [0]*(hp.maxlen-len(x))	
		x = np.array(x)
		x = x.reshape(1,-1)
		with self.melsession.as_default():
			preds = np.zeros((1, 1, hp.n_mels), np.float32)
			cnt = hp.Tyr
			for j in range(hp.Tyr):
				sys.stdout.write('\rProcessing %d' % j)
				sys.stdout.flush()
				_preds,a = self.melsession.run([self.melmodel.mel_output, self.melmodel.A], {self.melmodel.text: x, self.melmodel.mel: preds})
				preds = np.concatenate((np.zeros((1,1,hp.n_mels)),_preds),axis=1)  
				cnt -=1
				if np.argmax(a[0,:,-1]) >= len(inp)-3:
					cnt = min(cnt,10)
				if cnt<=0:
					break
		with self.magsession.as_default():
			wav = self.magsession.run(self.magmodel.wav_output,{self.magmodel.mel: preds})
			wav = audio.inv_preemphasis(wav)
			if save is not None:
				audio.save_wav(wav[0],save)
			else:
				out = io.BytesIO()
				audio.save_wav(wav[0], out)
				return out.getvalue()
			#audio.save_spec(mags[0].T,"out.wav")				
		#showmels(preds[0].T,"Mel Prediction","prediction.png")
		#showmels(a[0],"Attention","attfinal.png")

#s = Synth()
#s.load()
#s.synth("Hillary Clinton made a surprise appearance on Sunday night in a Grammy Awards comedy bit that took a jab at President Trump.")


if __name__ == '__main__':
	s = Synth()
	parser = argparse.ArgumentParser()
	parser.add_argument('--text',help='Text to Synthesize',default='The Big Brown Fox Jumped Over The Lazy Dog')
	parser.add_argument('--file',help='File to save to ',default='output.wav')
	args = parser.parse_args()
	s.synth(args.text, args.file)