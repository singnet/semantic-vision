## Experiments with multiple DNNs for Visual Question Answering (VQA)

The goal of our experiment was to evaluate ability of DNNs batch to be trained to solve VQA task and experiment with different loss functions. 

We trained models in supervised manner on the [VQA v2 dataset](http://www.visualqa.org/download.html) 

### Training on "yes/no" questions

For the beginning we trained models on yes/no questions only, where each question is described only by the following grounded formula: _predadj(A, B).
It means that a question asks: "does an object A have a property B". Thus we've got 12117 training questions and 5796 validation questions.

All key words (A and B) from all questions are putted into vocabulary. The acquired vocabulary contains 1353 words. So we create a DNN for each word in the vocabulary, and try to train it to predict is there an object or property described by the word. We used pretrained [bottom-up-attention features](https://github.com/peteanderson80/bottom-up-attention) for bounding boxes as input of DNNs. 

Training iteration flow is the following:

* Get a pair of question key words and ground truth answer from the training set.

* Find indices of the key words in vocabulary.

* Run forward a pair of correspondent DNNs, feeding a batch of boundig boxes features. 

* Get a joint probability of A and B for every bounding box by multiplying outputs of DNN pair.

* Compute a loss function value (binary cross entropy) depending on the acquired joint probability and ground truth answer.

* Do backpropagation.

There are two options of the loss function at the moment:

1. In [train_00_pytorch.py](./train_00_pytorch.py) the loss function gets only a maximum value of the predicted joint probability, if the ground truth answer is "yes", or the whole batch of the predicted joint probabilities, if the answer is "no".  
Thus, in the first case we want to do backpropagation considering the corresponded bounding box with maximum response only, while others are wanted to be ignored. In the second case we want to minimse response for the whole batch, since the answer should be "no".

Acquired validation score for this type of loss is 62.8%.


2. In [train_01_pytorch.py](./train_01_pytorch.py) the loss function always considers all predictions: 

        output = nets.feed_forward(nBBox, inputs, words)
        sum = torch.sum(output)
        sum_sq = torch.sum(torch.mul(output, output))
        output = sum_sq / sum
        loss = F.binary_cross_entropy(output, ans)


Acquired validation score for this type of loss is 67.65%.
