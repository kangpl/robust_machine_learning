#!/bin/bash
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python main.py --model PreActResNet18 --attack_during_train none --exp_name cifar10_standard_preActResNet18_exp1
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python main.py --model PreActResNet18 --attack_during_train fgsm --fgsm_alpha 7 --exp_name cifar10_fgsm_7_preActResNet18_exp1
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python main.py --model PreActResNet18 --attack_during_train fgsm --fgsm_alpha 10 --exp_name cifar10_fgsm_10_preActResNet18_exp1
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 24:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python main.py --model PreActResNet18 --attack_during_train pgd --exp_name cifar10_pgd_preActResNet18_exp1
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
##bsub -n 8 -W 08:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
##module load $PCOMMAND
##python main.py --model PreActResNet18 --attack_during_train fgsm --fgsm_alpha 16 --exp_name cifar10_fgsm_16_preActResNet18_exp1
##ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python main.py --model PreActResNet18 --lr_schedule constant --num_epochs 20 --attack_during_train fgsm --fgsm_alpha 7 --finetune --resumed_model_name cifar10_standard_preActResNet18_exp1_final.pth --exp_name cifar10_finetune_fgsm_7_lr_0.1_preActResNet18_exp1
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 03:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python main.py --model PreActResNet18 --lr 0.01 --lr_schedule constant --num_epochs 20 --attack_during_train fgsm --fgsm_alpha 7 --finetune --resumed_model_name cifar10_standard_preActResNet18_exp1_final.pth --exp_name cifar10_finetune_fgsm_7_lr_0.01_preActResNet18_exp1
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 03:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python main.py --model PreActResNet18 --lr 0.1 --lr_schedule multistep --lr_change_epoch 10 15 --num_epochs 20 --attack_during_train fgsm --fgsm_alpha 7 --finetune --resumed_model_name cifar10_standard_preActResNet18_exp1_final.pth --exp_name cifar10_finetune_fgsm_7_lr_multi_preActResNet18_exp1
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python main.py --model PreActResNet18 --lr 0.1 --lr_schedule multistep --lr_change_epoch 30 45 --num_epochs 60 --attack_during_train fgsm --fgsm_alpha 7 --finetune --resumed_model_name cifar10_standard_preActResNet18_exp1_final.pth --exp_name cifar10_finetune_fgsm_7_lr_multi_epoch_60_preActResNet18_exp1
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python main.py --model PreActResNet18 --lr 0.1 --lr_schedule multistep --lr_change_epoch 50 75 --num_epochs 100 --attack_during_train fgsm --fgsm_alpha 7 --finetune --resumed_model_name cifar10_standard_preActResNet18_exp1_final.pth --exp_name cifar10_finetune_fgsm_7_lr_multi_epoch_100_preActResNet18_exp1
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python main.py --model PreActResNet18 --lr 0.1 --lr_schedule multistep --lr_change_epoch 100 150 --num_epochs 200 --attack_during_train fgsm --fgsm_alpha 7 --finetune --resumed_model_name cifar10_standard_preActResNet18_exp1_final.pth --exp_name cifar10_finetune_fgsm_7_lr_multi_epoch_200_preActResNet18_exp1
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python main.py --model PreActResNet18 --lr 0.1 --lr_schedule multistep --lr_change_epoch 100 150 --num_epochs 200 --attack_during_train fgsm --fgsm_alpha 16 --finetune --resumed_model_name cifar10_standard_preActResNet18_exp1_final.pth --exp_name cifar10_finetune_fgsm_16_lr_multi_epoch_200_preActResNet18_exp1
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
##bsub -n 8 -W 04:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
##module load $PCOMMAND
##python main.py --model PreActResNet18 --attack_during_train none --exp_name cifar10_standard_preActResNet18_exp2
##ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python main.py --model PreActResNet18 --attack_during_train none --exp_name cifar10_standard_preActResNet18_add_norm_exp3
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python main.py --model PreActResNet18 --attack_during_train fgsm --train_fgsm_alpha 7 --exp_name cifar10_fgsm_7_preActResNet18_add_norm_exp2
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python main.py --model PreActResNet18 --attack_during_train fgsm --train_fgsm_alpha 10 --exp_name cifar10_fgsm_10_preActResNet18_add_norm_exp2
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python main.py --model PreActResNet18 --attack_during_train fgsm --train_fgsm_alpha 16 --exp_name cifar10_fgsm_16_preActResNet18_add_norm_exp2
#ENDBSUB


#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python main.py --model PreActResNet18 --attack_during_train none --exp_name cifar10_standard_preActResNet18_add_norm_v2_exp1
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python main.py --model PreActResNet18 --attack_during_train fgsm --train_fgsm_alpha 7 --exp_name cifar10_fgsm_7_preActResNet18_add_norm_v2_exp1
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python main.py --model PreActResNet18 --attack_during_train fgsm --train_fgsm_alpha 10 --exp_name cifar10_fgsm_10_preActResNet18_add_norm_v2_exp1
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python main.py --model PreActResNet18 --attack_during_train fgsm --train_fgsm_alpha 16 --exp_name cifar10_fgsm_16_preActResNet18_add_norm_v2_exp1
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python main.py --model PreActResNet18 --lr 0.01 --lr_schedule multistep --lr_change_epoch 100 150 --num_epochs 200 --attack_during_train fgsm --train_fgsm_alpha 7 --finetune --resumed_model_name cifar10_standard_preActResNet18_add_norm_exp3_final.pth --exp_name cifar10_finetune_fgsm_7_lr_0.01_multi_epoch_200_preActResNet18_exp1
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python main.py --model PreActResNet18 --lr 0.01 --lr_schedule multistep --lr_change_epoch 100 150 --num_epochs 200 --attack_during_train fgsm --train_fgsm_alpha 16 --finetune --resumed_model_name cifar10_standard_preActResNet18_add_norm_exp3_final.pth --exp_name cifar10_finetune_fgsm_16_lr_0.01_multi_epoch_200_preActResNet18_exp1
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python main.py --model PreActResNet18 --lr 0.001 --lr_schedule multistep --lr_change_epoch 100 150 --num_epochs 200 --attack_during_train fgsm --train_fgsm_alpha 7 --finetune --resumed_model_name cifar10_standard_preActResNet18_add_norm_exp3_final.pth --exp_name cifar10_finetune_fgsm_7_lr_0.001_multi_epoch_200_preActResNet18_exp1
#ENDBSUB
#
PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
bsub -n 8 -W 08:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
module load $PCOMMAND
python main.py --model PreActResNet18 --lr 0.001 --lr_schedule multistep --lr_change_epoch 100 150 --num_epochs 200 --attack_during_train fgsm --train_fgsm_alpha 16 --finetune --resumed_model_name cifar10_standard_preActResNet18_add_norm_exp3_final.pth --exp_name cifar10_finetune_fgsm_16_lr_0.001_multi_epoch_200_preActResNet18_exp1
ENDBSUB