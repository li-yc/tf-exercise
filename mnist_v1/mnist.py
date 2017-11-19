
# coding: utf-8

# In[1]:


import tensorflow as tf
sess = tf.InteractiveSession()


# In[12]:


from tensorflow.examples.tutorials.mnist import input_data

with tf.device('/device:GPU:2'):
	mnist = input_data.read_data_sets('MINST_data', one_hot=True)


	# In[2]:


	x = tf.placeholder(tf.float32, shape=[None, 784])
	y_ = tf.placeholder(tf.float32, shape=[None, 10])


	# In[4]:


	W = tf.Variable(tf.zeros([784, 10]))
	b = tf.Variable(tf.zeros([10]))


	# In[5]:


	sess.run(tf.global_variables_initializer())


	# In[6]:


	y = tf.matmul(x, W) + b


	# In[9]:


	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))


	# In[10]:


	train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


	# In[13]:


	for _ in range(2000):
	    batch = mnist.train.next_batch(100)
	    train_step.run(feed_dict={x: batch[0], y_: batch[1]})


	# In[14]:


	correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

