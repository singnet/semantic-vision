## Description

The general setup is following. We have N networks which are created
in advance, and we want dynamically combine these networks by creating
loss (and calculate gradients via this loss) which depend on two or more these networks.

Our simple test is following. We have N networks that receive input of
size (-1,2048). We note that 2048 is the size of faster RCNN features
which are used in our VQA project. Each network should solve simple
binary classification problem: check that at given position in the
input vector we have 0 or 1. 

We have the following tests:

* 01_test.py baseline tesorflow where all computation graphs
(including losses) are created in advance
* 02_test.py classical tensorflow (with statical computation graph)
where we attempt to create losses on fly. It is only a test. It is
obviously very slow, but if we need to create only few losses and
reuse all them we can do it.
* 03_test_pytorch.py pytorch version
* 04_test_tf_eager.py tensroflow with eager execution version

### Results

We've found that in this particular test pytroch is ~2 times faster
than tensorflow with eager execution on both CPU and GPU. 

