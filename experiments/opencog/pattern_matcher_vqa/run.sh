#!/usr/bin/env bash

python3 pattern_matcher_vqa.py \
    --questions /mnt/fileserver/shared/models/vqa_split_multidnn/questions.txt \
    --model-kind SPLITMULTIDNN \
    --atomspace /mnt/fileserver/shared/models/vqa_split_multidnn/vqa_dataset/atomspace_val.scm \
    --multidnn-model /mnt/fileserver/shared/models/vqa_split_multidnn/visual_genome \
    --features-extractor-kind PRECALCULATED \
    --precalculated-features /mnt/fileserver/shared/datasets/at-on-at-data/val2014_parsed_features.zip \
    --precalculated-features-prefix val2014_parsed_features/COCO_val2014_ \
    --python-log-level INFO \
    --opencog-log-level INFO 

