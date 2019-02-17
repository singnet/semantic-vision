Baseline bVAE was taken here https://github.com/1Konny/Beta-VAE

In this experiment we have tried to see if different neural network models could possibly model class distribution of mnist digits. We have trained the model on digit "3" and have tried to distinguish "3"(test) from "5" by obtaining reconstruction loss of each digit using the trained model to see if trained only on "3" model could see "5" as the most probable pattern of distribution it learns.

PCA

Here we have tried to learn PCA model on each MNIST digit. Run all_z_all_digit_models.sh, then you will have evaluation results (as .png pics).

cgan

We trained conditional GAN (https://arxiv.org/abs/1611.07004) for the same task. To run this experiment start run_gpnd.sh (nets configuration is borrowed from https://arxiv.org/abs/1807.02588).

AAE

We also trained baseline AAE for comparison. Run train.py to train the model and novelty_detector.py for evaluating.

