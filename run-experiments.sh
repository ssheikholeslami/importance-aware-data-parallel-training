#!/bin/bash
BATCH_SIZE=256

# These are the 10 different seeds obtained by running generate_seeds.py --master-seed 1 --number-of-seeds 10
# These seeds are used for the experiments in the paper
SEEDS=( 17611 74606 8271 33432 15455 64937 99740 58915 61898 85405 )

# number of warmup epochs (E_warmup)
WARMUPS=( 10 15 20 30 40 60 )

# number of interval epochs (E_interval)
INTERVALS=( 1 5 8 10 15 30 )

# "roundrobin" corresponds to the "Stripes" heuristic, and "steps" corresponds to the "Blocks" heuristic

#----------------------------------------------------------------------
#Baseline 1: ResNet-18 on CIFAR-10 (Baseline for Tables 1-3)
DATASET='cifar10'
MODEL='resnet18'
HEURISTIC='none'

for SEED in "${SEEDS[@]}"
do
    python dpt.py -n 1 -g 4 -nr 0 --description 'baseline1' --once-or-interval once --total-epochs 100 --warmup-epochs 2 --ignore-epochs 1 --interval-epochs 0 --seed $SEED --heuristic $HEURISTIC --measure variance --shuffle yes  --batch-size $BATCH_SIZE --dataset $DATASET --model $MODEL;
done

#----------------------------------------------------------------------
#Baseline 2: ResNet-34 on CIFAR-10 (Baseline for Table 4)
DATASET='cifar10'
MODEL='resnet34'
HEURISTIC='none'

for SEED in "${SEEDS[@]}"
do
    python dpt.py -n 1 -g 4 -nr 0 --description 'baseline2' --once-or-interval once --total-epochs 100 --warmup-epochs 2 --ignore-epochs 1 --interval-epochs 0 --seed $SEED --heuristic $HEURISTIC --measure variance --shuffle yes  --batch-size $BATCH_SIZE --dataset $DATASET --model $MODEL;
done

#----------------------------------------------------------------------
#Baseline 2: ResNet-34 on CIFAR-100 (Baseline for Table 5)
DATASET='cifar100'
MODEL='resnet34'
HEURISTIC='none'

for SEED in "${SEEDS[@]}"
do
    python dpt.py -n 1 -g 4 -nr 0 --description 'baseline3' --once-or-interval once --total-epochs 100 --warmup-epochs 2 --ignore-epochs 1 --interval-epochs 0 --seed $SEED --heuristic $HEURISTIC --measure variance --shuffle yes  --batch-size $BATCH_SIZE --dataset $DATASET --model $MODEL;
done

#----------------------------------------------------------------------
# CIFAR-10, ResNet-18, Stripes, Variance (Table 1)
DATASET='cifar10'
MODEL='resnet18'
HEURISTIC='roundrobin'
MEASURE='variance'

for WARMUP in "${WARMUPS[@]}"
do
    for INTERVAL in "${INTERVALS[@]}"
    do
        for SEED in "${SEEDS[@]}"
        do
            python dpt.py -n 1 -g 4 -nr 0 --description 'experiment' --once-or-interval interval --total-epochs 100 --warmup-epochs $WARMUP --ignore-epochs 5 --interval-epochs $INTERVAL --seed $SEED --heuristic $HEURISTIC --measure $MEASURE --shuffle yes  --batch-size $BATCH_SIZE --dataset $DATASET --model $MODEL;
        done
    done
done

#----------------------------------------------------------------------
# CIFAR-10, ResNet-18, Stripes, Average (Table 2)
DATASET='cifar10'
MODEL='resnet18'
HEURISTIC='roundrobin'
MEASURE='average'

for WARMUP in "${WARMUPS[@]}"
do
    for INTERVAL in "${INTERVALS[@]}"
    do
        for SEED in "${SEEDS[@]}"
        do
            python dpt.py -n 1 -g 4 -nr 0 --description 'experiment' --once-or-interval interval --total-epochs 100 --warmup-epochs $WARMUP --ignore-epochs 5 --interval-epochs $INTERVAL --seed $SEED --heuristic $HEURISTIC --measure $MEASURE --shuffle yes  --batch-size $BATCH_SIZE --dataset $DATASET --model $MODEL;
        done
    done
done

#----------------------------------------------------------------------
# CIFAR-10, ResNet-18, Blocks, Variance (Table 3)
DATASET='cifar10'
MODEL='resnet18'
HEURISTIC='steps'
MEASURE='variance'

for WARMUP in "${WARMUPS[@]}"
do
    for INTERVAL in "${INTERVALS[@]}"
    do
        for SEED in "${SEEDS[@]}"
        do
            python dpt.py -n 1 -g 4 -nr 0 --description 'experiment' --once-or-interval interval --total-epochs 100 --warmup-epochs $WARMUP --ignore-epochs 5 --interval-epochs $INTERVAL --seed $SEED --heuristic $HEURISTIC --measure $MEASURE --shuffle yes  --batch-size $BATCH_SIZE --dataset $DATASET --model $MODEL;
        done
    done
done

#----------------------------------------------------------------------
# CIFAR-10, ResNet-34, Stripes, Variance (Table 4)
DATASET='cifar10'
MODEL='resnet34'
HEURISTIC='roundrobin'
MEASURE='variance'

for WARMUP in "${WARMUPS[@]}"
do
    for INTERVAL in "${INTERVALS[@]}"
    do
        for SEED in "${SEEDS[@]}"
        do
            python dpt.py -n 1 -g 4 -nr 0 --description 'experiment' --once-or-interval interval --total-epochs 100 --warmup-epochs $WARMUP --ignore-epochs 5 --interval-epochs $INTERVAL --seed $SEED --heuristic $HEURISTIC --measure $MEASURE --shuffle yes  --batch-size $BATCH_SIZE --dataset $DATASET --model $MODEL;
        done
    done
done

#----------------------------------------------------------------------
# CIFAR-100, ResNet-34, Stripes, Variance (Table 5)
DATASET='cifar100'
MODEL='resnet34'
HEURISTIC='roundrobin'
MEASURE='variance'

for WARMUP in "${WARMUPS[@]}"
do
    for INTERVAL in "${INTERVALS[@]}"
    do
        for SEED in "${SEEDS[@]}"
        do
            python dpt.py -n 1 -g 4 -nr 0 --description 'experiment' --once-or-interval interval --total-epochs 100 --warmup-epochs $WARMUP --ignore-epochs 5 --interval-epochs $INTERVAL --seed $SEED --heuristic $HEURISTIC --measure $MEASURE --shuffle yes  --batch-size $BATCH_SIZE --dataset $DATASET --model $MODEL;
        done
    done
done

#----------------------------------------------------------------------