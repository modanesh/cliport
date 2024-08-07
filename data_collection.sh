#!/bin/bash

python cliport/demos.py n=100 task=align-rope mode=train
python cliport/demos.py n=100 task=assembling-kits-seq-unseen-colors mode=train
python cliport/demos.py n=100 task=packing-boxes-pairs-unseen-colors mode=train
python cliport/demos.py n=100 task=packing-shapes mode=train
python cliport/demos.py n=100 task=packing-unseen-google-objects-seq mode=train
python cliport/demos.py n=100 task=packing-unseen-google-objects-group mode=train
python cliport/demos.py n=100 task=put-block-in-bowl-unseen-colors mode=train
python cliport/demos.py n=100 task=stack-block-pyramid-seq-unseen-colors mode=train
python cliport/demos.py n=100 task=separating-piles-unseen-colors mode=train
python cliport/demos.py n=100 task=towers-of-hanoi-seq-unseen-colors mode=train
python cliport/demos.py n=100 task=align-rope mode=val
python cliport/demos.py n=100 task=assembling-kits-seq-unseen-colors mode=val
python cliport/demos.py n=100 task=packing-boxes-pairs-unseen-colors mode=val
python cliport/demos.py n=100 task=packing-shapes mode=val
python cliport/demos.py n=100 task=packing-unseen-google-objects-seq mode=val
python cliport/demos.py n=100 task=packing-unseen-google-objects-group mode=val
python cliport/demos.py n=100 task=put-block-in-bowl-unseen-colors mode=val
python cliport/demos.py n=100 task=stack-block-pyramid-seq-unseen-colors mode=val
python cliport/demos.py n=100 task=separating-piles-unseen-colors mode=val
python cliport/demos.py n=100 task=towers-of-hanoi-seq-unseen-colors mode=val
python cliport/demos.py n=100 task=align-rope mode=test
python cliport/demos.py n=100 task=assembling-kits-seq-unseen-colors mode=test
python cliport/demos.py n=100 task=packing-boxes-pairs-unseen-colors mode=test
python cliport/demos.py n=100 task=packing-shapes mode=test
python cliport/demos.py n=100 task=packing-unseen-google-objects-seq mode=test
python cliport/demos.py n=100 task=packing-unseen-google-objects-group mode=test
python cliport/demos.py n=100 task=put-block-in-bowl-unseen-colors mode=test
python cliport/demos.py n=100 task=stack-block-pyramid-seq-unseen-colors mode=test
python cliport/demos.py n=100 task=separating-piles-unseen-colors mode=test
python cliport/demos.py n=100 task=towers-of-hanoi-seq-unseen-colors mode=test
#
python cliport/demos.py n=100 task=assembling-kits-seq-seen-colors mode=train
python cliport/demos.py n=100 task=packing-boxes-pairs-seen-colors mode=train
python cliport/demos.py n=100 task=packing-seen-google-objects-seq mode=train
python cliport/demos.py n=100 task=packing-seen-google-objects-group mode=train
python cliport/demos.py n=100 task=put-block-in-bowl-seen-colors mode=train
python cliport/demos.py n=100 task=stack-block-pyramid-seq-seen-colors mode=train
python cliport/demos.py n=100 task=separating-piles-seen-colors mode=train
python cliport/demos.py n=100 task=towers-of-hanoi-seq-seen-colors mode=train
python cliport/demos.py n=100 task=assembling-kits-seq-seen-colors mode=val
python cliport/demos.py n=100 task=packing-boxes-pairs-seen-colors mode=val
python cliport/demos.py n=100 task=packing-seen-google-objects-seq mode=val
python cliport/demos.py n=100 task=packing-seen-google-objects-group mode=val
python cliport/demos.py n=100 task=put-block-in-bowl-seen-colors mode=val
python cliport/demos.py n=100 task=stack-block-pyramid-seq-seen-colors mode=val
python cliport/demos.py n=100 task=separating-piles-seen-colors mode=val
python cliport/demos.py n=100 task=towers-of-hanoi-seq-seen-colors mode=val
python cliport/demos.py n=100 task=assembling-kits-seq-seen-colors mode=test
python cliport/demos.py n=100 task=packing-boxes-pairs-seen-colors mode=test
python cliport/demos.py n=100 task=packing-seen-google-objects-seq mode=test
python cliport/demos.py n=100 task=packing-seen-google-objects-group mode=test
python cliport/demos.py n=100 task=put-block-in-bowl-seen-colors mode=test
python cliport/demos.py n=100 task=stack-block-pyramid-seq-seen-colors mode=test
python cliport/demos.py n=100 task=separating-piles-seen-colors mode=test
python cliport/demos.py n=100 task=towers-of-hanoi-seq-seen-colors mode=test
#
#