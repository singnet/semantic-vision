#!/usr/bin/env bash

python3 pattern_matcher_vqa.py \
    --questions ~/projects/data/questions.934.txt \
    --model-kind SPLITMULTIDNN \
    --atomspace ~/projects/data/train_tv_atomspace.scm \
    --multidnn-model ~/projects/data/visual_genome \
    --features-extractor-kind PRECALCULATED \
    --precalculated-features ~/projects/data/val2014_parsed_features.zip \
    --precalculated-features-prefix val2014_parsed_features/COCO_val2014_ \
    --python-log-level INFO \
    --opencog-log-level INFO 

