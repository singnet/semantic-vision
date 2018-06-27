import tensorflow as tf
import numpy as np

Nnets=10
Nsum = 20
input_size = 2048
learning_rate = 1e-1
Nit = 1024
Nbatch = 128
# we create N networks and N mask
# the target for the network will be to sum inputs at position defined by masks
def create_network():
    net  =  []
    masks = []  
    for i in range(Nnets):
        x = tf.placeholder(tf.float32, shape=(None, input_size), name = "input_%i"%i)
        input = x
        x = tf.layers.dense(x, 64, activation = tf.nn.relu, name = "dense_1_%i"%i)
        x = tf.layers.dense(x, 32, activation = tf.nn.relu, name = "dense_2_%i"%i)
        x = tf.layers.dense(x, 1,  activation = None, name = "dense_3_%i"%i)
        output = x
        sel = np.random.choice(input_size, size=Nsum, replace = False)
#        sel = np.arange(Nsum, dtype=np.int)
        net.append((input, output, sel))
    return net

def create_batch(Nbatch, sel):
    x = np.random.rand(Nbatch,input_size)
    y = np.sum(x[:,sel], axis=1)
    return (x,y)


net = create_network()

ys        = []
for i in range(Nnets):
    y = tf.placeholder(tf.float32, shape=(None,))
    ys.append(y)
    
sess = tf.Session()
sess.run(tf.global_variables_initializer())
    
for it in range(Nit):
    
    sum_loss = 0
    for i in range(Nnets):
        y = ys[i]
        output = net[i][1]
        loss   = tf.reduce_mean(tf.square((y - output) / Nsum))
        #    train_op = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)
        train_op = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

        input,output,sel = net[i]
        (batch_x, batch_y) = create_batch(Nbatch, sel)
        rez_loss, _ = sess.run( [loss, train_op], feed_dict={input: batch_x, ys[i]: batch_y})
        sum_loss += rez_loss
    print (it, sum_loss/Nnets)
            
