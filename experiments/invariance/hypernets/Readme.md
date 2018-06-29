# Experiments with second order models

Here are three different experiments with "second order" neural networks. Second order means that there are some control weights which are applied to weights between layers. So there are weights for weights. In our experiments, we have tried to see if such type of model could generalize some transformations on MNIST digits like rotation and random affine transform. Each folder contains results and some explanation on experiments. Mainly, we have used models of following types:

## basic model
Basic model is just a simple model with input, output and control between them. Code is below

	W = tf.constant(np.ones((OUTPUT_DIM, OUTPUT_DIM)), dtype=tf.float32, name="W")
	#W = tf.get_variable("W",(OUTPUT_DIM, OUTPUT_DIM)) #it could be trainable or constant control weights
    W = tf.reshape(W, (1, OUTPUT_DIM, OUTPUT_DIM))
    W = tf.tile(W, [tf.shape(real_data)[0], 1, 1])
    W_control = tf.layers.dense(params, 64, activation=tf.nn.relu) #params are different for every experiment. Sin and cos for rotation experiment, affine transform's parameters for 
	# affine experiment etc
    W_control = tf.layers.dense(W_control, OUTPUT_DIM * OUTPUT_DIM, activation=None)
    W_control = tf.reshape(W_control, [-1, OUTPUT_DIM, OUTPUT_DIM])
    W_control = tf.nn.softmax(W_control, dim=1)
    W_rez = W * W_control
    x = tf.contrib.layers.flatten(real_data)
    x = tf.einsum('bi,bij->bj', x, W_rez)
    rec_data = tf.reshape(x, [-1, *IMG_DIM])
	rec_data = tf.identity(rec_data, "output_img")

## convolutional model

Basic model sometimes showed not so good results so it was decided to slightly deepen it with convolutional layers. Code is below

	W = tf.constant(np.ones((OUTPUT_DIM, OUTPUT_DIM)), dtype=tf.float32, name="W")
	#W = tf.get_variable("W",(OUTPUT_DIM, OUTPUT_DIM))
    W = tf.reshape(W, (1, OUTPUT_DIM, OUTPUT_DIM))
    W = tf.tile(W, [tf.shape(real_data)[0], 1, 1])
    W_control = tf.layers.dense(parameters_tf, 64, activation=tf.nn.relu)
    W_control = tf.layers.dense(W_control, OUTPUT_DIM * OUTPUT_DIM, activation=None)
    W_control = tf.reshape(W_control, [-1, OUTPUT_DIM, OUTPUT_DIM])
    W_control = tf.nn.softmax(W_control, dim=1)
    W_rez = W * W_control
    x = tf.contrib.layers.flatten(real_data)
    x = tf.reshape(x, shape=[-1, 28, 28, 1])
    x = tf.layers.conv2d(x, 10, 5, 2, padding='same', activation=tf.nn.relu)
    x = tf.layers.conv2d(x, 20, 5, 2, padding='same', activation=tf.nn.relu)
    x = tf.contrib.layers.flatten(x)
    x = tf.layers.dense(x, 200, activation=None)
    x = tf.einsum('bi,bij->bj', x, W_rez)
    x = tf.layers.dense(x, INNER_LAYER_DIM, activation=None)
    x = tf.reshape(x, [-1, 7, 7, 20])
    y = tf.layers.conv2d_transpose(x, 10, 5, 2, padding='same', activation=tf.nn.relu)
    y = tf.layers.conv2d_transpose(y, 1, 5, 2, padding='same', activation=tf.nn.tanh)
    rec_data = tf.identity(y, "output_img")

## Address model

This model is learning via slightly different way than pther 2nd order models. If previously we have just used our control weights to manipulate with weights between layers and looked at what happened, here we have tried to see if we can manually say to our model to learn transformation of pixels from i,j (input) to x,y position (output). So our control matrix in this scenario had to be "address matrix" transforming pixels from one place to another. Code is below:

	W_control = tf.layers.dense(acosi, N * N * 2, activation=None)
    W_control = tf.reshape(W_control, (-1, N, N, 1, 1, 2))
    # [-1,1] - > [0,N-1] (to index space)                                                                                    
    W_control = (W_control + 1) * (N - 1) / 2.0
    # fixed address on the images (in index space)
    NI = fixed_address(N)
    NI = tf.constant(NI, dtype=W_control.dtype, name="NI")
    # NI -> (1,1,1,N,N,2)  
    NI = tf.reshape(NI, (1,1,1,N,N,2))
    x = tf.nn.relu(1 - tf.abs(NI - W_control))
    x = tf.reduce_prod(x, axis=-1)       # (1-dx) * (1-dy)
    layer1 = tf.layers.conv2d(real_data, 16, 5, 2, padding='same', activation=tf.nn.tanh)
    layer2 = tf.layers.conv2d(layer1, 32, 5, 2, padding='same', activation=tf.nn.relu)
    layerControl = tf.einsum('bijc,bijxy->bxyc', layer2, x)
    layer3 = tf.layers.conv2d_transpose(layerControl, 16, 5, 2, padding='same', activation=tf.nn.relu)
    layer4 = tf.layers.conv2d_transpose(layer3, 1, 5, 2, padding='same', activation=tf.nn.tanh)
    rec_data = tf.identity(layer4, "output_img")

## baseline_autoencoder

This model was used for comparison with our HyperNets. It's a regular autoencoder in which angles or affine parameters are applied to latent code using concatenation:

    x = tf.concat((x, angles), axis=1)

see * [baseline autoencoders](./../baseline/autoencoders/) for more info
