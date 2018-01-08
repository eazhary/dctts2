# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
June 2017 by kyubyong park. 
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer
'''

from __future__ import print_function
import tensorflow as tf

def normalize(inputs, 
			  epsilon = 1e-8,
			  scope="ln",
			  reuse=None):
	'''Applies layer normalization.
	
	Args:
	  inputs: A tensor with 2 or more dimensions, where the first dimension has
		`batch_size`.
	  epsilon: A floating number. A very small number for preventing ZeroDivision Error.
	  scope: Optional scope for `variable_scope`.
	  reuse: Boolean, whether to reuse the weights of a previous layer
		by the same name.
	  
	Returns:
	  A tensor with the same shape and data dtype as `inputs`.
	'''
	with tf.variable_scope(scope, reuse=reuse):
		inputs_shape = inputs.get_shape()
		params_shape = inputs_shape[-1:]
	
		mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
		beta= tf.Variable(tf.zeros(params_shape))
		gamma = tf.Variable(tf.ones(params_shape))
		normalized = (inputs - mean) / ( (variance + epsilon) ** (.5) )
		outputs = gamma * normalized + beta
		
	return outputs

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
	
	
	
def embeddings(inputs, 
			  vocab_size, 
			  num_units, 
			  zero_pad=True, 
			  scale=True,
			  lookup=None,
			  scope="embedding", 
			  reuse=None):
	with tf.variable_scope(scope, reuse=reuse):
		if zero_pad:
			lookup = tf.concat((tf.zeros(shape=[1, num_units]),
									  lookup[1:, :]), 0)
		outputs = tf.nn.embedding_lookup(lookup, inputs)
		
		if scale:
			outputs = outputs * (num_units ** 0.5) 
			
	return outputs
	

def positional_encoding(inputs,
						num_units,
						zero_pad=True,
						scale=True,
						scope="positional_encoding",
						reuse=None):
	'''Sinusoidal Positional_Encoding.

	Args:
	  inputs: A 2d Tensor with shape of (N, T).
	  num_units: Output dimensionality
	  zero_pad: Boolean. If True, all the values of the first row (id = 0) should be constant zero
	  scale: Boolean. If True, the output will be multiplied by sqrt num_units(check details from paper)
	  scope: Optional scope for `variable_scope`.
	  reuse: Boolean, whether to reuse the weights of a previous layer
		by the same name.

	Returns:
		A 'Tensor' with one more rank than inputs's, with the dimensionality should be 'num_units'
	'''

	N, T = inputs.get_shape().as_list()
	with tf.variable_scope(scope, reuse=reuse):
		position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1])

		# First part of the PE function: sin and cos argument
		position_enc = np.array([
			[pos / np.power(10000, 2.*i/num_units) for i in range(num_units)]
			for pos in range(T)])

		# Second part, apply the cosine to even columns and sin to odds.
		position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
		position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1

		# Convert to a tensor
		lookup_table = tf.convert_to_tensor(position_enc)

		if zero_pad:
			lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
									  lookup_table[1:, :]), 0)
		outputs = tf.nn.embedding_lookup(lookup_table, position_ind)

		if scale:
			outputs = outputs * num_units**0.5

		return outputs



def multihead_attention(queries, 
						keys, 
						num_units=None, 
						num_heads=8, 
						dropout_rate=0,
						is_training=True,
						causality=False,
						scope="multihead_attention", 
						reuse=None):
	'''Applies multihead attention.
	
	Args:
	  queries: A 3d tensor with shape of [N, T_q, C_q].
	  keys: A 3d tensor with shape of [N, T_k, C_k].
	  num_units: A scalar. Attention size.
	  dropout_rate: A floating point number.
	  is_training: Boolean. Controller of mechanism for dropout.
	  causality: Boolean. If true, units that reference the future are masked. 
	  num_heads: An int. Number of heads.
	  scope: Optional scope for `variable_scope`.
	  reuse: Boolean, whether to reuse the weights of a previous layer
		by the same name.
		
	Returns
	  A 3d tensor with shape of (N, T_q, C)	 
	'''
	with tf.variable_scope(scope, reuse=reuse):
		# Set the fall back option for num_units
		if num_units is None:
			num_units = queries.get_shape().as_list[-1]
		
		# Linear projections
		Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu) # (N, T_q, C)
		K = tf.layers.dense(keys, num_units, activation=tf.nn.relu) # (N, T_k, C)
		V = tf.layers.dense(keys, num_units, activation=tf.nn.relu) # (N, T_k, C)
		
		# Split and concat
		Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0) # (h*N, T_q, C/h) 
		K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0) # (h*N, T_k, C/h) 
		V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0) # (h*N, T_k, C/h) 

		# Multiplication
		outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1])) # (h*N, T_q, T_k)
		
		# Scale
		outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)
		
		# Key Masking
		key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1))) # (N, T_k)
		key_masks = tf.tile(key_masks, [num_heads, 1]) # (h*N, T_k)
		key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1]) # (h*N, T_q, T_k)
		
		paddings = tf.ones_like(outputs)*(-2**32+1)
		outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs) # (h*N, T_q, T_k)
  
		# Causality = Future blinding
		if causality:
			diag_vals = tf.ones_like(outputs[0, :, :]) # (T_q, T_k)
			tril = tf.contrib.linalg.LinearOperatorTriL(diag_vals).to_dense() # (T_q, T_k)
			masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1]) # (h*N, T_q, T_k)
   
			paddings = tf.ones_like(masks)*(-2**32+1)
			outputs = tf.where(tf.equal(masks, 0), paddings, outputs) # (h*N, T_q, T_k)
  
		# Activation
		outputs = tf.nn.softmax(outputs) # (h*N, T_q, T_k)
		 
		# Query Masking
		query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1))) # (N, T_q)
		query_masks = tf.tile(query_masks, [num_heads, 1]) # (h*N, T_q)
		query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]]) # (h*N, T_q, T_k)
		outputs *= query_masks # broadcasting. (N, T_q, C)
		  
		# Dropouts
		outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
			   
		# Weighted sum
		outputs = tf.matmul(outputs, V_) # ( h*N, T_q, C/h)
		
		# Restore shape
		outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2 ) # (N, T_q, C)
			  
		# Residual connection
		outputs += queries
			  
		# Normalize
		outputs = normalize(outputs) # (N, T_q, C)
 
	return outputs

def feedforward(inputs, 
				num_units=[2048, 512],
				scope="multihead_attention", 
				reuse=None):
	'''Point-wise feed forward net.
	
	Args:
	  inputs: A 3d tensor with shape of [N, T, C].
	  num_units: A list of two integers.
	  scope: Optional scope for `variable_scope`.
	  reuse: Boolean, whether to reuse the weights of a previous layer
		by the same name.
		
	Returns:
	  A 3d tensor with the same shape and dtype as inputs
	'''
	with tf.variable_scope(scope, reuse=reuse):
		# Inner layer
		params = {"inputs": inputs, "filters": num_units[0], "kernel_size": 1,
				  "activation": tf.nn.relu, "use_bias": True}
		outputs = tf.layers.conv1d(**params)
		
		# Readout layer
		params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1,
				  "activation": None, "use_bias": True}
		outputs = tf.layers.conv1d(**params)
		
		# Residual connection
		outputs += inputs
		
		# Normalize
		outputs = normalize(outputs)
	
	return outputs
	
def conv_block(inputs,
			   num_units=None,
			   size=5,
			   rate=1,
			   padding="SAME",
			   dropout_rate=0,
			   training=False,
			   activation=None,
			   scope="conv_block",
			   reuse=None):
	'''Convolution block.
	Args:
	  inputs: A 3-D tensor with shape of [batch, time, depth].
	  size: An int. Filter size.
	  padding: Either `same` or `valid` or `causal` (case-insensitive).
	  norm_type: A string. See `normalize`.
	  activation_fn: A string. Activation function.
	  training: A boolean. Whether or not the layer is in training mode.
	  scope: Optional scope for `variable_scope`.
	  reuse: Boolean, whether to reuse the weights of a previous layer
		by the same name.
	Returns:
	  A tensor of the same shape and dtype as inputs.
	'''
	in_dim = inputs.get_shape().as_list()[-1]
	if num_units is None: num_units = in_dim

	with tf.variable_scope(scope, reuse=reuse):
		inputs = tf.layers.dropout(inputs, rate=dropout_rate, training=training)

		if padding.lower() == "causal":
			# pre-padding for causality
			pad_len = (size - 1) * rate	 # padding size
			inputs = tf.pad(inputs, [[0, 0], [pad_len, 0], [0, 0]])
			padding = "VALID"

		V = tf.get_variable('V',
							shape=[size, in_dim, num_units],
							dtype=tf.float32) # (width, in_dim, out_dim)
		g = tf.get_variable('g',
							shape=(num_units,),
							dtype=tf.float32,
							initializer=tf.contrib.layers.variance_scaling_initializer(factor=(4.*(1.-dropout_rate))/size))
		b = tf.get_variable('b',
							shape=(num_units,),
							dtype=tf.float32,
							initializer=tf.zeros_initializer)

		V_norm = tf.nn.l2_normalize(V, [0, 1])	# (width, in_dim, out_dim)
		W = V_norm * tf.reshape(g, [1, 1, num_units])

		outputs = tf.nn.convolution(inputs, W, padding, dilation_rate=[rate]) + b
		if activation is not None:
			outputs=activation(outputs)
		#outputs = glu(outputs)

	return outputs	

	
def conv_block2(inputs,
			   num_units=None,
			   size=5,
			   rate=1,
			   padding="SAME",
			   dropout_rate=0,
			   training=False,
			   activation=None,
			   scope="conv_block",
			   reuse=None):
	with tf.variable_scope(scope, reuse=reuse):
		inputs = tf.layers.dropout(inputs, rate=dropout_rate, training=training)
		w = tf.get_variable('w', [1, size, inputs.get_shape()[-1], num_units ],
			initializer=tf.truncated_normal_initializer(stddev=0.02))
		b = tf.get_variable('b', [num_units ],
			initializer=tf.constant_initializer(0.0))

		if padding.lower() == "causal":
			# pre-padding for causality
			pad_len = (size - 1) * rate	 # padding size
			inputs = tf.pad(inputs, [[0, 0], [pad_len, 0], [0, 0]])
			inputs_expanded = tf.expand_dims(inputs,dim=1)
			padding = "VALID"
			out = tf.nn.atrous_conv2d(inputs_expanded,w,rate=rate,padding='VALID')+b
		else:
			inputs_expanded = tf.expand_dims(inputs,dim=1)
			out = tf.nn.atrous_conv2d(inputs_expanded,w,rate=rate,padding='SAME')+b
		out = tf.squeeze(out,[1])
		if activation is not None:
			out = activation(out)
	return out	
	
	
def Conv1D(inputs, channels, kernel_size, dilation,causal=True,is_training=True,dropout=0.1, activation=None, scope = "Conv1D", reuse=None):
	with tf.variable_scope(scope, reuse=reuse):
		inputs = tf.layers.dropout(inputs, rate=dropout,training=is_training)
		outputs = conv1d(inputs, channels, kernel_size, scope, dilation, causal,)
		if activation is not None:
			outputs=activation(outputs)
	return outputs

def ddConv1D(inputs, channels, kernel_size, dilation,causal=True,is_training=True, activation=None, dropout=0.1, scope = "Conv1D", reuse=None):
	with tf.variable_scope(scope, reuse=reuse):
		outputs = conv_block2(inputs,num_units=channels, size=kernel_size,rate=dilation,training=is_training,activation=activation,dropout_rate=dropout)
	return outputs

def HConv1D(inputs, channels, kernel_size, dilation, causal=True,is_training=True, activation=None, scope = "HConv1D", reuse=None):
	with tf.variable_scope(scope, reuse=reuse):
		H2 = Conv1D(inputs, channels, kernel_size, dilation=dilation, causal=causal,is_training=is_training,activation=activation,scope='c1d-H2')
		H1 = Conv1D(inputs, channels, kernel_size, dilation=dilation, causal=causal,is_training=is_training,activation=tf.nn.sigmoid,scope='c1d-H1')
	return H1 * H2 + inputs * (1.0 - H1)

def dilated_causal_conv1d(x, filter,kernel, dialation):
	padding = (tf.shape(filter)[0] - 1) * dialation
	x = tf.pad(x, ((0, 0), (padding, 0), (0, 0)))
	filter = tf.expand_dims(filter, 0)
	x = tf.expand_dims(x, 0)
	x = tf.nn.atrous_conv2d(x, filter, dialation, 'VALID')
	x = tf.squeeze(x, (0,))
	return x[:, padding:]	
def label_smoothing(inputs, epsilon=0.1):
	'''Applies label smoothing. See https://arxiv.org/abs/1512.00567.
	
	Args:
	  inputs: A 3d tensor with shape of [N, T, V], where V is the number of vocabulary.
	  epsilon: Smoothing rate.
	
	For example,
	
	```
	import tensorflow as tf
	inputs = tf.convert_to_tensor([[[0, 0, 1], 
	   [0, 1, 0],
	   [1, 0, 0]],

	  [[1, 0, 0],
	   [1, 0, 0],
	   [0, 1, 0]]], tf.float32)
	   
	outputs = label_smoothing(inputs)
	
	with tf.Session() as sess:
		print(sess.run([outputs]))
	
	>>
	[array([[[ 0.03333334,	0.03333334,	 0.93333334],
		[ 0.03333334,  0.93333334,	0.03333334],
		[ 0.93333334,  0.03333334,	0.03333334]],

	   [[ 0.93333334,  0.03333334,	0.03333334],
		[ 0.93333334,  0.03333334,	0.03333334],
		[ 0.03333334,  0.93333334,	0.03333334]]], dtype=float32)]	 
	```	   
	'''
	K = inputs.get_shape().as_list()[-1] # number of channels
	return ((1-epsilon) * inputs) + (epsilon / K)
	
	
def mul_or_none(a, b):
	"""Return the element wise multiplicative of the inputs.
	If either input is None, we return None.
  Args:
	a: A tensor input.
	b: Another tensor input with the same type as a.
  Returns:
	None if either input is None. Otherwise returns a * b.
	"""
	if a is None or b is None:
		return None
	return a * b

	
def time_to_batch(value, dilation, name=None):
	with tf.name_scope('time_to_batch'):
		shape = tf.shape(value)
		pad_elements = dilation - 1 - (shape[1] + dilation - 1) % dilation
		padded = tf.pad(value, [[0, 0], [0, pad_elements], [0, 0]])
		reshaped = tf.reshape(padded, [-1, dilation, shape[2]])
		transposed = tf.transpose(reshaped, perm=[1, 0, 2])
		return tf.reshape(transposed, [shape[0] * dilation, -1, shape[2]])


def batch_to_time(value, dilation, name=None):
	with tf.name_scope('batch_to_time'):
		shape = tf.shape(value)
		prepared = tf.reshape(value, [dilation, -1, shape[2]])
		transposed = tf.transpose(prepared, perm=[1, 0, 2])
		return tf.reshape(transposed,
						  [tf.div(shape[0], dilation), -1, shape[2]])
	

def otime_to_batch(x, block_size):
	"""Splits time dimension (i.e. dimension 1) of `x` into batches.
  Within each batch element, the `k*block_size` time steps are transposed,
  so that the `k` time steps in each output batch element are offset by
  `block_size` from each other.
  The number of input time steps must be a multiple of `block_size`.
  Args:
	x: Tensor of shape [nb, k*block_size, n] for some natural number k.
	block_size: number of time steps (i.e. size of dimension 1) in the output
	  tensor.
  Returns:
	Tensor of shape [nb*block_size, k, n]
	"""
	#shape = x.get_shape().as_list()
	shape = tf.shape(x)
	y = tf.reshape(x, shape=[
		shape[0], shape[1] // block_size, block_size, shape[2]
		])
	y = tf.transpose(y, perm=[0, 2, 1, 3])
	y = tf.reshape(y, shape=[
		shape[0] * block_size, shape[1] // block_size, shape[2]
		])
	#y.set_shape([
	#	mul_or_none(shape[0], block_size), mul_or_none(shape[1], 1. / block_size),
	#	shape[2]
	#	])
	return y


def obatch_to_time(x, block_size):
	"""Inverse of `time_to_batch(x, block_size)`.
  Args:
	x: Tensor of shape [nb*block_size, k, n] for some natural number k.
	block_size: number of time steps (i.e. size of dimension 1) in the output
	  tensor.
  Returns:
	Tensor of shape [nb, k*block_size, n].
	"""
	#shape = x.get_shape().as_list()
	shape = tf.shape(x)
	y = tf.reshape(x, shape=[shape[0] // block_size, block_size, shape[1], shape[2]])
	y = tf.transpose(y, perm=[0, 2, 1, 3])
	y = tf.reshape(y, shape=[shape[0] // block_size, shape[1] * block_size, shape[2]])
	#y.set_shape([mul_or_none(shape[0], 1. / block_size),
	#	mul_or_none(shape[1], block_size),
	#	shape[2]])
	return y
def conv1d(x,
		   num_filters,
		   filter_length,
		   name,
		   dilation=1,
		   causal=True):
	if not causal:
		return tf.layers.conv1d(x,filters=num_filters,kernel_size=filter_length,padding='same',dilation_rate=dilation)
	shapes=tf.shape(x)
	b, in_time,in_channels = x.get_shape().as_list()
	#filt = (filter_length,shapes[1],num_filters)
	##print("in ",name,b,in_time,in_channels)
	init = tf.contrib.layers.xavier_initializer()
	##print(filter_length,in_channels,num_filters)
	filters_ = tf.Variable(init([filter_length,in_channels,num_filters]), name = 'filter')
	padding = [[0, 0], [(filter_length - 1) * dilation, 0], [0, 0]] #this may need to be adjusted for padding on both sides 
	padded = tf.pad(x, padding)
	if dilation>1:
		transformed = time_to_batch(padded,dilation)
		b,t,c = transformed.get_shape().as_list()
		##print("trans ",name,b,t,c)
		conv = tf.nn.conv1d(transformed,filters_,stride=1,padding='SAME')
		restored = batch_to_time(conv,dilation)
	else:
		restored = tf.nn.conv1d(padded,filters_,stride=1,padding='SAME')
	b,t,c = restored.get_shape().as_list()
	##print("restored ",name,b,t,c)
	#out_width = in_time-(filter_length-1)*dilation
	result = tf.slice(restored,[0,0,0],[-1,in_time,-1])
	b,t,c = result.get_shape().as_list()
	##print("out ",name,b,t,c)
	return result
		

			
