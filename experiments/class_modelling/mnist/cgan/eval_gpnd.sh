#!/usr/bin/env bash
rm *.txt
ZSIZES=( 16 32 64 )
for N in `seq 0 9`;
do
    for Z in "${ZSIZES[@]}"; # `seq 10 10 110`
    do
        echo $N $Z
        python novelty_detector.py $Z $N > "report $N $Z.txt"
        
        #python train_AAE.py $Z $N
        #echo $N $Z
    done
done