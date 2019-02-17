#!/usr/bin/env bash
rm pca_report/*
ZSIZES=( 16 32 64 )
for N in `seq 0 9`;
do
    for Z in "${ZSIZES[@]}"; # `seq 10 10 110`
    do
        python train_AAE.py $Z $N
        #echo $N $Z
    done
done