import tensorflow as tf
import numpy as np
import time

Nnets=10
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
        output = x[:,0]
        index = i % input_size                
        net.append((input, output, index))
    return net

def create_batch(Nbatch, index):
    y = np.random.randint(2, size=Nbatch)
    x = np.zeros((Nbatch,input_size))
    
    # put true to true
    x[:,index] += y
    
    # add contamination
    x += np.random.binomial(1, np.random.rand() * 0.02, size=(Nbatch,input_size))
    
    # put false to false
    x[:,index] *= y
                                    
    
    x[x>0.5] = 1
    return x,y
    

net = create_network()
ys        = []

for i in range(Nnets):
    y = tf.placeholder(tf.float32, shape=(None,))
    ys.append(y)
    
sess = tf.Session()
sess.run(tf.global_variables_initializer())
    
for it in range(Nit):
    
    all_batch = []
        
    # we exclude batch generation from time
    for i in range(Nnets):
        _,_,sel = net[i]
        all_batch.append(create_batch(Nbatch, sel))
                                    
    sum_loss = 0
    sum_score = 0
    
    start = time.time()
    
    for i in range(Nnets):
        y = ys[i]
        output = net[i][1]
        loss   = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=output))
        
        rez_y    = tf.round((tf.nn.sigmoid(output)))
        score_tf = tf.reduce_mean(tf.cast(tf.equal(rez_y, y),dtype=tf.float32))
               
        #    train_op = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)
        train_op = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

        input,output,sel = net[i]
        (batch_x, batch_y) = all_batch[i]
        rez_loss,score, _ = sess.run( [loss, score_tf, train_op], feed_dict={input: batch_x, ys[i]: batch_y})
                
        
        sum_loss += rez_loss
        sum_score += score
        
    t = time.time() - start
    print (it, sum_loss/Nnets, sum_score/Nnets, t)
            
