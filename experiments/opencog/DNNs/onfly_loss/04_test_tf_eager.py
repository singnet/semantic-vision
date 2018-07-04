import tensorflow as tf
import numpy as np
import time

tf.enable_eager_execution()
tfe = tf.contrib.eager

Nnets=10
input_size = 2048
learning_rate = 1e-1
Nit = 1024
Nbatch = 128


# we create N networks and N mask
# the target for the network will be to sum inputs at position defined by masks
def create_network():
    nets  =  []
    for i in range(Nnets):
        model = tf.keras.Sequential([
        tf.keras.layers.Dense(64,  activation=tf.keras.activations.relu, input_shape=(input_size,)), 
        tf.keras.layers.Dense(32,  activation=tf.keras.activations.relu), 
        tf.keras.layers.Dense(1,   activation=None), 
        tf.keras.layers.Reshape([]), 
        ])            
        index = i % input_size                
        nets.append((model, index))
    return nets

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
    

nets = create_network()
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    
for it in range(Nit):
    
    sum_loss = 0
    sum_score = 0
    
    start = time.time()
    
    for i in range(Nnets):
        
        model, sel = nets[i]
        (batch_x, batch_y) = create_batch(Nbatch, sel)
        batch_y_tf = tf.constant(batch_y, dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            output = model(batch_x)
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=batch_y_tf, logits=output))
            
        grads     = tape.gradient(loss, model.variables)
            
        optimizer.apply_gradients(zip(grads, model.variables), global_step=tf.train.get_or_create_global_step())

        rez_y    = tf.round((tf.nn.sigmoid(output)))
        score    = tf.reduce_mean(tf.cast(tf.equal(rez_y, batch_y),dtype=tf.float32))
        
        sum_loss  += loss.numpy()
        sum_score += score.numpy()

        
    t = time.time() - start
    print (it, sum_loss/Nnets, sum_score/Nnets, t)
            
