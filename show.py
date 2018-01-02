#from matplotlib.pyplot import plt
import numpy as np
from matplotlib import pyplot as plt


filename = "LJSpeech-1.0/mels/LJ001-0001.npy"
filename2 = "LJSpeech-1.0/mels/LJ001-0002.npy"

mels = np.load(filename)
mels2 = np.load(filename2)
print(mels.shape)

def showmels(mel1,mel2,msg):
	fig, ax = plt.subplots(2,1)
	ax[0][:matshow](np.transpose(mel1))
	ax[1][:matshow](np.transpose(mel2))
	#cax = ax.matshow(np.transpose(mel1), interpolation='nearest',  cmap=plt.cm.afmhot, origin='lower')
	#fig.colorbar(cax)
	plt.title(msg)
	plt.show()
	#plt.savefig(msg,format='png')

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
	#plt.show()
	
	
show(mels,mels2)#,"Original")
#showmels(mels2,"Second")	
