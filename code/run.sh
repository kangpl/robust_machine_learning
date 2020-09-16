#!/bin/bash
FILE_NAME="$1"
MODEL="$2"
USERNAME="pekang"
PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
bsub -n 8 -W 04:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
module load $PCOMMAND
python train.py --tensorboard ./tensorboard/$FILE_NAME --model $MODEL --checkpoint ./checkpoint/$FILE_NAME &> output/$FILE_NAME
ENDBSUB