# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
Originally by
June 2017 by kyubyong park. 
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer
'''

from __future__ import print_function
import tensorflow as tf

def embedding(inputs, 
			  vocab_size, 
			  num_units, 
			  zero_pad=False, 
			  scale=True,
			  scope="embedding", 
			  reuse=None):
	'''Embeds a given tensor.

	Args:
	  inputs: A `Tensor` with type `int32` or `int64` containing the ids
		 to be looked up in `lookup table`.
	  vocab_size: An int. Vocabulary size.
	  num_units: An int. Number of embedding hidden units.
	  zero_pad: A boolean. If True, all the values of the fist row (id 0)
		should be constant zeros.
	  scale: A boolean. If True. the outputs is multiplied by sqrt num_units.
	  scope: Optional scope for `variable_scope`.
	  reuse: Boolean, whether to reuse the weights of a previous layer
		by the same name.

	Returns:
	  A `Tensor` with one more rank than inputs's. The last dimensionality
		should be `num_units`.
		
	For example,
	
	```
	import tensorflow as tf
	
	inputs = tf.to_int32(tf.reshape(tf.range(2*3), (2, 3)))
	outputs = embedding(inputs, 6, 2, zero_pad=True)
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		print sess.run(outputs)
	>>
	[[[ 0.			0.		  ]
	  [ 0.09754146	0.67385566]
	  [ 0.37864095 -0.35689294]]

	 [[-1.01329422 -1.09939694]
	  [ 0.7521342	0.38203377]
	  [-0.04973143 -0.06210355]]]
	```
	
	```
	import tensorflow as tf
	
	inputs = tf.to_int32(tf.reshape(tf.range(2*3), (2, 3)))
	outputs = embedding(inputs, 6, 2, zero_pad=False)
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		print sess.run(outputs)
	>>
	[[[-0.19172323 -0.39159766]
	  [-0.43212751 -0.66207761]
	  [ 1.03452027 -0.26704335]]

	 [[-0.11634696 -0.35983452]
	  [ 0.50208133	0.53509563]
	  [ 1.22204471 -0.96587461]]]	 
	```	   
	'''
	with tf.variable_scope(scope, reuse=reuse):
		lookup_table = tf.get_variable('lookup_table',
									   dtype=tf.float32,
									   shape=[vocab_size, num_units],
									   initializer=tf.contrib.layers.xavier_initializer())
		if zero_pad:
			lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
									  lookup_table[1:, :]), 0)
		outputs = tf.nn.embedding_lookup(lookup_table, inputs)
		
		if scale:
			outputs = outputs * (num_units ** 0.5) 
			
	return outputs

def conv1d(inputs, 
           filters, 
           size=1, 
           rate=1, 
           padding="SAME", 
           causal=False,
           use_bias=False,
           scope="conv1d"):
    '''
    Args:
      inputs: A 3-D tensor of [batch, time, depth].
      filters: An int. Number of outputs (=activation maps)
      size: An int. Filter size.
      rate: An int. Dilation rate.
      padding: Either `SAME` or `VALID`.
      causal: A boolean. If True, zeros of (kernel size - 1) * rate are padded on the left
        for causality.
      use_bias: A boolean.
    
    Returns:
      A masked tensor of the sampe shape as `tensor`.
    '''
    
    with tf.variable_scope(scope):
        if causal:
            # pre-padding for causality
            pad_len = (size - 1) * rate  # padding size
            inputs = tf.pad(inputs, [[0, 0], [pad_len, 0], [0, 0]])
            padding = "VALID"
            
        params = {"inputs":inputs, "filters":filters, "kernel_size":size,
                "dilation_rate":rate, "padding":padding, "activation":None, 
                "use_bias":use_bias}
        
        out = tf.layers.conv1d(**params)
    
    return out
	
def conv1d_transpose(x,filters,kernel_size,strides):
	x = tf.expand_dims(x,1)
	outputs=tf.layers.conv2d_transpose(x,filters,kernel_size,strides=(1,strides),padding='same')
	outputs = tf.squeeze(outputs,1)
	return outputs

def Deconv1D(inputs, channels, kernel_size,dilation,scope="deconv1d"):
	with tf.variable_scope(scope, reuse=False):
		outputs = conv1d_transpose(inputs,channels,kernel_size,2)
		return outputs
	
def Conv1D(inputs, channels, kernel_size, dilation,causal=True,is_training=True,dropout=0.1, activation=None, scope = "Conv1D", reuse=None):
	with tf.variable_scope(scope, reuse=reuse):
		outputs = conv1d(inputs, channels, size=kernel_size, scope=scope, rate=dilation, causal=causal,)
		if activation is not None:
			outputs=activation(outputs)
		return tf.layers.dropout(outputs, rate=dropout,training=is_training)

def HConv1D(inputs, channels, kernel_size, dilation, causal=True,is_training=True, activation=None, scope = "HConv1D", reuse=None):
	with tf.variable_scope(scope, reuse=reuse):
		H = Conv1D(inputs, 2*channels, kernel_size, dilation=dilation, causal=causal,is_training=is_training,activation=activation,scope='c1d-H')
		H1,H2 = tf.split(H,num_or_size_splits=2,axis=2)
		H1 = tf.nn.sigmoid(H1)
		return H1 * H2 + inputs * (1.0 - H1)
