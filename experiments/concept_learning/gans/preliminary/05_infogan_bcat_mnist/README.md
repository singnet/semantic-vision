### Description 

The target of this work was to study possibility of extracting some
information from latent variables of InfoGAN.

Here I show several preliminary experiments, mainly with binary categorical units on MNIST.
It should be noted that in theory NBC binary categorical variables are equivalent to
one 2^NBC categorical variables. However implementation in InfoGAN is
slightly different. I model binary categorical variables as a binary
unit (0 or 1), and N-categorical variables as one N-one-hot unit. 

Here we have the following simulations

* 11_only_cat10 . one 10-categorical variable. It is similar to classical
InfoGAN MNIST test but without 2 uniform variables.

* 17_only_cat8 . one 8-categorical variables. This simulations should
be equivalent to 14_bcat3 with 3 binary variables


* 12_bcat1 . 1 binary variable
* 13_bcat2 
* 14_bcat3 
* 15_bcat4 
* 16_bcat5

All model where trained for 100000 steps with constant, and relatively
hight learning rate (1e-3).  We can assume that already at 10000 steps, training
more or less converged. At least images do not become better. So,
presumably, after 10000 steps models only wandering around "equilibrium".

For each model we have several checkpoints between 9999 and 100000.

For each model in directory test_plots3 we have plots. On these plots
each column correspond to unique combination of latent variables. For
example, for 4 binary variables we will have 16 columns. For each raw,
all noise variables are fixed.

Also for each model and each checkpoint I did the following. For
training set of MNIST I calculated number of examples for each of unique combinations of latent variables. 
Results are in count_test/count.txt. The first column is the
checkpoint index.


### Important points

1. It seems that model with only one 10-categorical variables is not as good to
semantically separate digits as classical InfoGAN model with
one 10-categorical and two uniform latent variables (11_only_cat10 vs
[10_originfo_sepQ_v2_lr1e-3](https://github.com/elggem/semantic-vision-internal/tree/master/experiments/sergey/info-wgan-gp))

2. One can expect the following, In the perfect situation, distribution of latent variables computed
for the training set should be equivalent to distribution of latent
variables which we feed to the Generator. At least InfoGAN will try to
do it in such a way. In these models it is not exactly like this, but
rather close to it. See "count_test/count.txt" for each model.
As you can see distribution is not completely uniform (for true random
distribution, each combination should have the same probability). More
important that counts are fluctuates significantly for different
checkpoints.


