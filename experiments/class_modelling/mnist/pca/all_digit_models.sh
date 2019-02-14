#!/usr/bin/env bash
[ -e "metrics_z{$1}.txt" ] && rm "metrics_z{$1}.txt"
for i in `seq 0 9`; # `seq 2 2 100`
do
    #echo $i
    #python train.py -zs $i
    python pca_n_mnist.py --seed 9412 --zsize $1 --ntrain $i
done
