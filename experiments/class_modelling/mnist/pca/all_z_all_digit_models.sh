#!/usr/bin/env bash
for i in `seq 10 10 110`;
do
    sh all_digit_models.sh $i
done

