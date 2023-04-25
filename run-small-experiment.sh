#!/bin/bash
# Use this script to run a small experiment to test the functionality of the system.
# -g specifies the number of GPUs

python dpt.py -n 1 -g 1 -nr 0 --description 'small-exp' --once-or-interval interval --total-epochs 10 --warmup-epochs 3 --ignore-epochs 1 --interval-epochs 3 --seed 0 --heuristic roundrobin --measure variance --shuffle yes  --batch-size 128 --dataset cifar10 --model resnet18;
