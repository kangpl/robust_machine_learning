#!/bin/bash
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --attack_during_train none --exp_name cifar10_standard_preActResNet18_exp1
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --attack_during_train fgsm --fgsm_alpha 7 --exp_name cifar10_fgsm_7_preActResNet18_exp1
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --attack_during_train fgsm --fgsm_alpha 10 --exp_name cifar10_fgsm_10_preActResNet18_exp1
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 24:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --attack_during_train pgd --exp_name cifar10_pgd_preActResNet18_exp1
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
##bsub -n 8 -W 08:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
##module load $PCOMMAND
##python train_fgsm.py --model PreActResNet18 --attack_during_train fgsm --fgsm_alpha 16 --exp_name cifar10_fgsm_16_preActResNet18_exp1
##ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --lr_schedule constant --num_epochs 20 --attack_during_train fgsm --fgsm_alpha 7 --finetune --resumed_model_name cifar10_standard_preActResNet18_exp1_final.pth --exp_name cifar10_finetune_fgsm_7_lr_0.1_preActResNet18_exp1
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 03:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --lr 0.01 --lr_schedule constant --num_epochs 20 --attack_during_train fgsm --fgsm_alpha 7 --finetune --resumed_model_name cifar10_standard_preActResNet18_exp1_final.pth --exp_name cifar10_finetune_fgsm_7_lr_0.01_preActResNet18_exp1
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 03:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --lr 0.1 --lr_schedule multistep --lr_change_epoch 10 15 --num_epochs 20 --attack_during_train fgsm --fgsm_alpha 7 --finetune --resumed_model_name cifar10_standard_preActResNet18_exp1_final.pth --exp_name cifar10_finetune_fgsm_7_lr_multi_preActResNet18_exp1
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --lr 0.1 --lr_schedule multistep --lr_change_epoch 30 45 --num_epochs 60 --attack_during_train fgsm --fgsm_alpha 7 --finetune --resumed_model_name cifar10_standard_preActResNet18_exp1_final.pth --exp_name cifar10_finetune_fgsm_7_lr_multi_epoch_60_preActResNet18_exp1
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --lr 0.1 --lr_schedule multistep --lr_change_epoch 50 75 --num_epochs 100 --attack_during_train fgsm --fgsm_alpha 7 --finetune --resumed_model_name cifar10_standard_preActResNet18_exp1_final.pth --exp_name cifar10_finetune_fgsm_7_lr_multi_epoch_100_preActResNet18_exp1
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --lr 0.1 --lr_schedule multistep --lr_change_epoch 100 150 --num_epochs 200 --attack_during_train fgsm --fgsm_alpha 7 --finetune --resumed_model_name cifar10_standard_preActResNet18_exp1_final.pth --exp_name cifar10_finetune_fgsm_7_lr_multi_epoch_200_preActResNet18_exp1
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --lr 0.1 --lr_schedule multistep --lr_change_epoch 100 150 --num_epochs 200 --attack_during_train fgsm --fgsm_alpha 16 --finetune --resumed_model_name cifar10_standard_preActResNet18_exp1_final.pth --exp_name cifar10_finetune_fgsm_16_lr_multi_epoch_200_preActResNet18_exp1
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
##bsub -n 8 -W 04:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
##module load $PCOMMAND
##python train_fgsm.py --model PreActResNet18 --attack_during_train none --exp_name cifar10_standard_preActResNet18_exp2
##ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --attack_during_train none --exp_name cifar10_standard_preActResNet18_add_norm_exp3
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --attack_during_train fgsm --train_fgsm_alpha 7 --exp_name cifar10_fgsm_7_preActResNet18_add_norm_exp2
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --attack_during_train fgsm --train_fgsm_alpha 10 --exp_name cifar10_fgsm_10_preActResNet18_add_norm_exp2
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --attack_during_train fgsm --train_fgsm_alpha 16 --exp_name cifar10_fgsm_16_preActResNet18_add_norm_exp2
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --attack_during_train none --exp_name cifar10_standard_preActResNet18_add_norm_v2_exp1
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --attack_during_train fgsm --train_fgsm_alpha 7 --exp_name cifar10_fgsm_7_preActResNet18_add_norm_v2_exp1
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --attack_during_train fgsm --train_fgsm_alpha 10 --exp_name cifar10_fgsm_10_preActResNet18_add_norm_v2_exp1
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --attack_during_train fgsm --train_fgsm_alpha 16 --exp_name cifar10_fgsm_16_preActResNet18_add_norm_v2_exp1
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --lr 0.01 --lr_schedule multistep --lr_change_epoch 100 150 --num_epochs 200 --attack_during_train fgsm --train_fgsm_alpha 7 --finetune --resumed_model_name cifar10_standard_preActResNet18_add_norm_exp3_final.pth --exp_name cifar10_finetune_fgsm_7_lr_0.01_multi_epoch_200_preActResNet18_exp1
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --lr 0.01 --lr_schedule multistep --lr_change_epoch 100 150 --num_epochs 200 --attack_during_train fgsm --train_fgsm_alpha 16 --finetune --resumed_model_name cifar10_standard_preActResNet18_add_norm_exp3_final.pth --exp_name cifar10_finetune_fgsm_16_lr_0.01_multi_epoch_200_preActResNet18_exp1
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --lr 0.001 --lr_schedule multistep --lr_change_epoch 100 150 --num_epochs 200 --attack_during_train fgsm --train_fgsm_alpha 7 --finetune --resumed_model_name cifar10_standard_preActResNet18_add_norm_exp3_final.pth --exp_name cifar10_finetune_fgsm_7_lr_0.001_multi_epoch_200_preActResNet18_exp1
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --lr 0.001 --lr_schedule multistep --lr_change_epoch 100 150 --num_epochs 200 --attack_during_train fgsm --train_fgsm_alpha 16 --finetune --resumed_model_name cifar10_standard_preActResNet18_add_norm_exp3_final.pth --exp_name cifar10_finetune_fgsm_16_lr_0.001_multi_epoch_200_preActResNet18_exp1
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --attack_during_train none --exp_name cifar10_standard_preActResNet18_add_norm_std_exp4
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --attack_during_train fgsm --train_fgsm_alpha 7 --exp_name cifar10_fgsm_7_preActResNet18_add_norm_std_exp4
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --attack_during_train fgsm --train_fgsm_alpha 10 --exp_name cifar10_fgsm_10_preActResNet18_add_norm_std_exp4
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --attack_during_train fgsm --train_fgsm_alpha 16 --exp_name cifar10_fgsm_16_preActResNet18_add_norm_std_exp4
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 99:99 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --attack_during_train fgsm --train_fgsm_alpha 10 --test_pgd_attack_iters 50 --test_pgd_restarts 10 --exp_name cifar10_fgsm_10_pgd_50_10_preActResNet18_exp1
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 99:99 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --attack_during_train fgsm --train_fgsm_alpha 16 --test_pgd_attack_iters 50 --test_pgd_restarts 10 --exp_name cifar10_fgsm_16_pgd_50_10_preActResNet18_exp1
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 24:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --attack_during_train fgsm --train_fgsm_alpha 7 --attack_during_test deepfool --exp_name test
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 24:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --attack_during_train pgd --attack_during_test deepfool --exp_name test_pgd
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 24:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --attack_during_train pgd --train_pgd_attack_iters 20 --train_pgd_restarts 2 --attack_during_test deepfool --exp_name test_pgd_2
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 24:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --attack_during_train fgsm --train_fgsm_alpha 7 --attack_during_test deepfool --exp_name cifar10_fgsm_7_preActResNet18_deepfool_exp1
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 24:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --attack_during_train fgsm --train_fgsm_alpha 10 --attack_during_test deepfool --exp_name cifar10_fgsm_10_preActResNet18_deepfool_exp1
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 24:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --attack_during_train fgsm --train_fgsm_alpha 16 --attack_during_test deepfool --exp_name cifar10_fgsm_16_preActResNet18_deepfool_exp1
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 24:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --attack_during_train fgsm --train_fgsm_alpha 10 --attack_during_test deepfool --deepfool_classes_num 4 --exp_name cifar10_fgsm_10_preActResNet18_deepfool_4_exp1
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
##bsub -n 8 -W 24:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
##module load $PCOMMAND
##python train_fgsm.py --model PreActResNet18 --attack_during_train fgsm --train_fgsm_alpha 7 --mix --exp_name cifar10_fgsm_7_preActResNet18_mix_exp1
##ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 24:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --attack_during_train fgsm --train_fgsm_alpha 10 --mix --exp_name cifar10_fgsm_10_preActResNet18_mix_exp1
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 24:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --attack_during_train fgsm --train_fgsm_alpha 16 --mix --exp_name cifar10_fgsm_16_preActResNet18_mix_exp1
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 10:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --attack_during_train fgsm --train_fgsm_alpha 10 --adjust_fgsm_alpha --exp_name cifar10_fgsm_10_preActResNet18_adjust_exp1
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 10:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --attack_during_train fgsm --train_fgsm_alpha 16 --adjust_fgsm_alpha --exp_name cifar10_fgsm_16_preActResNet18_adjust_exp2
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --attack_during_train none --exp_name cifar10_standard_preActResNet18_add_norm_std_median_exp1
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --attack_during_train fgsm --train_fgsm_alpha 7 --exp_name cifar10_fgsm_7_preActResNet18_add_norm_std_median_exp1
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --attack_during_train fgsm --train_fgsm_alpha 10 --exp_name cifar10_fgsm_10_preActResNet18_add_norm_std_median_exp1
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
##bsub -n 8 -W 08:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
##module load $PCOMMAND
##python train_fgsm.py --model PreActResNet18 --cure --exp_name cifar10_cure_exp1
##ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --cure --exp_name cifar10_cure_eval_exp1
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --cure --cure_lambda 40 --exp_name cifar10_cure_40_eval_exp1
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --cure --cure_lambda 400 --exp_name cifar10_cure_400_eval_exp1
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --attack_during_train fgsm --train_fgsm_alpha 10 --exp_name cifar10_fgsm_10_preActResNet18_add_norm_std_median_exp2
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --attack_during_train fgsm --train_fgsm_alpha 10 --exp_name cifar10_fgsm_10_preActResNet18_add_norm_std_median_exp3
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --cure --exp_name cifar10_cure_h1.5_eval_exp1
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --cure --cure_lambda 40 --exp_name cifar10_cure_40_h1.5_eval_exp1
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --cure --cure_lambda 400 --exp_name cifar10_cure_400_h1.5_eval_exp1
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --lr 0.001 --lr_schedule constant --num_epochs 20 --cure --finetune --resumed_model_name cifar10_standard_preActResNet18_add_norm_exp3_final.pth --exp_name cifar10_finetune_cure_exp1
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --lr 0.001 --lr_schedule constant --num_epochs 20 --cure --cure_lambda 40 --finetune --resumed_model_name cifar10_standard_preActResNet18_add_norm_exp3_final.pth --exp_name cifar10_finetune_cure_40_exp1
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --lr 0.001 --lr_schedule constant --num_epochs 20 --cure --cure_lambda 400 --finetune --resumed_model_name cifar10_standard_preActResNet18_add_norm_exp3_final.pth --exp_name cifar10_finetune_cure_400_exp1
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --lr 0.01 --lr_schedule constant --num_epochs 20 --cure --finetune --resumed_model_name cifar10_standard_preActResNet18_add_norm_exp3_final.pth --exp_name cifar10_finetune_cure_lr0.01_exp1
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --lr 0.01 --lr_schedule constant --num_epochs 20 --cure --cure_lambda 40 --finetune --resumed_model_name cifar10_standard_preActResNet18_add_norm_exp3_final.pth --exp_name cifar10_finetune_cure_40_lr0.01_exp1
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --lr 0.01 --lr_schedule constant --num_epochs 20 --cure --cure_lambda 400 --finetune --resumed_model_name cifar10_standard_preActResNet18_add_norm_exp3_final.pth --exp_name cifar10_finetune_cure_400_lr0.01_exp1
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --lr 0.001 --lr_schedule constant --num_epochs 20 --cure --cure_h 3 --finetune --resumed_model_name cifar10_standard_preActResNet18_add_norm_exp3_final.pth --exp_name cifar10_finetune_cure_h3_exp1
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --lr 0.001 --lr_schedule constant --num_epochs 20 --cure --cure_h 3 --cure_lambda 40 --finetune --resumed_model_name cifar10_standard_preActResNet18_add_norm_exp3_final.pth --exp_name cifar10_finetune_cure_h3_40_exp1
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --lr 0.001 --lr_schedule constant --num_epochs 20 --cure --cure_h 3 --cure_lambda 400 --finetune --resumed_model_name cifar10_standard_preActResNet18_add_norm_exp3_final.pth --exp_name cifar10_finetune_cure_h3_400_exp1
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --lr 0.01 --lr_schedule constant --num_epochs 60 --cure --cure_h 1.5 --cure_lambda 4 --finetune --resumed_model_name cifar10_standard_preActResNet18_add_norm_exp3_final.pth --exp_name cifar10_finetune_cure_h1.5_sum_withpow_exp1
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --lr 0.01 --lr_schedule constant --num_epochs 60 --cure --cure_h 1.5 --cure_lambda 4 --finetune --resumed_model_name cifar10_standard_preActResNet18_add_norm_exp3_final.pth --exp_name cifar10_finetune_cure_h1.5_sum_withpow_normv2_exp1
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --lr 0.01 --lr_schedule constant --num_epochs 60 --cure --cure_h 1.5 --cure_lambda 4 --finetune --resumed_model_name cifar10_standard_preActResNet18_add_norm_exp3_final.pth --exp_name cifar10_finetune_cure_h1.5_sum_withpow_normv2_warmup_exp1
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --lr 0.01 --lr_schedule constant --num_epochs 60 --cure --cure_h 1.5 --cure_lambda 4 --finetune --resumed_model_name cifar10_standard_preActResNet18_add_norm_exp3_final.pth --exp_name cifar10_finetune_cure_h1.5_sum_normv2_warmup_exp1
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --lr 0.01 --lr_schedule constant --num_epochs 60 --cure --cure_h 1.5 --cure_lambda 4 --finetune --resumed_model_name cifar10_standard_preActResNet18_add_norm_exp3_final.pth --exp_name cifar10_finetune_cure_h1.5_sum_normv2_warmup_exp2
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --lr 0.01 --lr_schedule constant --num_epochs 60 --cure --cure_h 1.5 --cure_lambda 4 --finetune --resumed_model_name cifar10_standard_preActResNet18_add_norm_exp3_final.pth --exp_name cifar10_finetune_cure_h1.5_sum_withpow_normv2_warmup_exp2
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --lr 0.1 --lr_schedule constant --num_epochs 60 --cure --cure_h 1.5 --cure_lambda 4 --finetune --resumed_model_name cifar10_standard_preActResNet18_add_norm_exp3_final.pth --exp_name cifar10_finetune_cure_lr0.1_h1.5_sum_normv2_warmup_exp2
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --lr 0.1 --lr_schedule constant --num_epochs 60 --cure --cure_h 1.5 --cure_lambda 4 --finetune --resumed_model_name cifar10_standard_preActResNet18_add_norm_exp3_final.pth --exp_name cifar10_finetune_cure_lr0.1_h1.5_sum_withpow_normv2_warmup_exp2
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --lr 0.001 --lr_schedule constant --num_epochs 60 --cure --cure_h 1.5 --cure_lambda 4 --finetune --resumed_model_name cifar10_standard_preActResNet18_add_norm_exp3_final.pth --exp_name cifar10_finetune_cure_lr0.001_h1.5_sum_withpow_normv2_warmup_exp2
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --lr 0.0001 --lr_schedule constant --num_epochs 60 --cure --cure_h 1.5 --cure_lambda 4 --finetune --resumed_model_name cifar10_standard_preActResNet18_add_norm_exp3_final.pth --exp_name cifar10_finetune_cure_lr0.0001_h1.5_sum_withpow_normv2_warmup_exp1
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --attack_during_train fgsm --train_fgsm_alpha 10 --exp_name cifar10_fgsm_10_preActResNet18_add_norm_std_median_hist_exp1
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
##bsub -n 8 -W 08:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
##module load $PCOMMAND
##python train_fgsm.py --model PreActResNet18 --attack_during_train fgsm --train_fgsm_alpha 16 --exp_name cifar10_fgsm_16_preActResNet18_add_norm_std_median_hist_exp1
##ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --attack_during_train fgsm --train_fgsm_alpha 16 --exp_name cifar10_fgsm_16_preActResNet18_add_norm_std_median_hist_exp2
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
##bsub -n 8 -W 08:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
##module load $PCOMMAND
##python train_fgsm.py --model PreActResNet18 --attack_during_train fgsm --train_fgsm_alpha 16 --exp_name cifar10_fgsm_16_preActResNet18_add_norm_std_median_hist_exp3
##ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --attack_during_train fgsm --train_fgsm_alpha 16 --num_epochs 20 --exp_name test
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --attack_during_train none --num_epochs 20 --exp_name test_none
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --attack_during_train fgsm --train_fgsm_alpha 10 --attack_during_test deepfool --exp_name deepfool
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 24:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --attack_during_train fgsm --train_fgsm_alpha 10 --exp_name fgsm_pgd_deepfool
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 24:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --attack_during_train fgsm --train_fgsm_alpha 10 --exp_name fgsm_pgd_deepfool_exp2
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 24:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --attack_during_train fgsm --train_fgsm_alpha 10 --exp_name fgsm_pgd_deepfool_exp3
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 24:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python adjust_alpha.py --model PreActResNet18 --attack_during_train fgsm --train_fgsm_alpha 10 --exp_name fgsm_10_adjust_alpha_exp1
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 24:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python adjust_alpha.py --model PreActResNet18 --attack_during_train fgsm --train_fgsm_alpha 16 --exp_name fgsm_16_adjust_alpha_exp1
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 24:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --attack_during_train fgsm --train_fgsm_alpha 10 --exp_name fgsm_10_df_grad_cos
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 24:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --attack_during_train fgsm --train_fgsm_alpha 10 --exp_name fgsm_10_df_grad_cos_exp2
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 24:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --attack_during_train fgsm --train_fgsm_alpha 16 --exp_name fgsm_16_df_grad_cos_exp1
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 24:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --attack_during_train fgsm --train_fgsm_alpha 16 --exp_name fgsm_16_df_grad_cos_exp2
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 24:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --attack_during_train fgsm --train_fgsm_alpha 10 --exp_name fgsm_10_valid_exp1
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 24:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --attack_during_train fgsm --train_fgsm_alpha 16 --exp_name fgsm_16_valid_exp1
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 24:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --attack_during_train fgsm --train_fgsm_alpha 10 --exp_name fgsm_10_valid_exp2
#ENDBSUB
#
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 24:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --attack_during_train fgsm --train_fgsm_alpha 16 --exp_name fgsm_16_valid_exp2 // 只考虑对的类
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 24:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --attack_during_train fgsm --train_fgsm_alpha 10 --exp_name fgsm_10_DF_exp1
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 24:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --attack_during_train fgsm --train_fgsm_alpha 10 --exp_name fgsm_10_DF_exp2  // 正确的FGSM
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 24:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_clean.py --model PreActResNet18 --attack_during_train none --exp_name clean_DF_exp1
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 24:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_clean.py --model PreActResNet18 --attack_during_train none --exp_name clean_DF_exp2
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 24:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --attack_during_train fgsm --train_fgsm_alpha 10 --exp_name fgsm_10_DF_FGSM_PGD_cos_exp1 #######################
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 24:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --attack_during_train fgsm --train_fgsm_alpha 10 --exp_name fgsm_10_DF_FGSM_PGD_cos_exp2
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 24:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --attack_during_train fgsm --train_fgsm_alpha 10 --test_fgsm_alpha 10 --exp_name fgsm_test_fgsm_alpha_10_exp1
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 24:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --attack_during_train fgsm --train_fgsm_alpha 10 --test_fgsm_alpha 10 --exp_name fgsm_test_fgsm_alpha_10_exp2
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 24:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --attack_during_train fgsm --train_fgsm_alpha 10 --test_fgsm_alpha 8 --exp_name fgsm_test_fgsm_alpha_8_exp1
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 24:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --attack_during_train fgsm --train_fgsm_alpha 10 --test_fgsm_alpha 8 --exp_name fgsm_test_fgsm_alpha_8_exp2
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 24:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --attack_during_train fgsm --train_fgsm_alpha 10 --test_fgsm_alpha 6 --exp_name fgsm_test_fgsm_alpha_6_exp1
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 24:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --attack_during_train fgsm --train_fgsm_alpha 10 --test_fgsm_alpha 6 --exp_name fgsm_test_fgsm_alpha_6_exp2
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 24:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --attack_during_train fgsm --train_fgsm_alpha 10 --test_fgsm_alpha 4 --exp_name fgsm_test_fgsm_alpha_4_exp1
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 24:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --attack_during_train fgsm --train_fgsm_alpha 10 --test_fgsm_alpha 4 --exp_name fgsm_test_fgsm_alpha_4_exp2 ##################
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 24:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --exp_name attack_deepfool_exp1 #######################
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 24:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --deepfool_max_iter_train 2 --exp_name attack_deepfool_iter2_exp1
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 24:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --deepfool_max_iter_train 3 --exp_name attack_deepfool_iter3_exp1
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 24:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --deepfool_max_iter_train 4 --exp_name attack_deepfool_iter4_exp1
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 24:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --deepfool_max_iter_train 50 --exp_name attack_deepfool_iter50_exp1
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 24:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --exp_name attack_deepfool_v2_exp1
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 24:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --deepfool_max_iter_train 2 --exp_name attack_deepfool_v2_iter2_exp1
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 24:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --attack_during_train none --exp_name clean_exp1    ##############################
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=2048,ngpus_excl_p=4]"  <<ENDBSUB
#module load $PCOMMAND
#python train_cure.py --model PreActResNet18 --lr 8e-4 --lr_schedule constant --num_epochs 15 --batch_size 1024 --finetune --resumed_model_name cifar10_standard_preActResNet18_add_norm_exp3_final.pth --exp_name cure_exp1
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  <<ENDBSUB
#module load $PCOMMAND
#python train_cure.py --model PreActResNet18 --lr 8e-4 --lr_schedule constant --num_epochs 15 --batch_size 1024 --finetune --resumed_model_name cifar10_standard_preActResNet18_add_norm_exp3_final.pth --exp_name cure_exp2
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  <<ENDBSUB
#module load $PCOMMAND
#python train_cure.py --model PreActResNet18 --lr 8e-4 --lr_schedule constant --num_epochs 15 --batch_size 1024 --finetune --resumed_model_name cifar10_standard_preActResNet18_add_norm_exp3_final.pth --exp_name cure_exp3
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  <<ENDBSUB
#module load $PCOMMAND
#python train_cure.py --model PreActResNet18 --lr 8e-4 --lr_change_epoch 10 15 --num_epochs 20 --batch_size 1024 --finetune --resumed_model_name cifar10_standard_preActResNet18_add_norm_exp3_final.pth --exp_name cure_exp4
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  <<ENDBSUB
#module load $PCOMMAND
#python train_cure.py --model PreActResNet18 --num_epochs 20 --cure_h_max 1.5 --batch_size 1024 --finetune --resumed_model_name cifar10_standard_preActResNet18_add_norm_exp3_final.pth --exp_name cure_exp5
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  <<ENDBSUB
#module load $PCOMMAND
#python train_cure.py --model PreActResNet18 --num_epochs 20 --cure_h_max 1.5 --batch_size 1024 --finetune --resumed_model_name cifar10_standard_preActResNet18_add_norm_exp3_final.pth --exp_name cure_exp6
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  <<ENDBSUB
#module load $PCOMMAND
#python train_cure.py --model PreActResNet18 --num_epochs 20 --cure_h_max 3.0 --batch_size 1024 --finetune --resumed_model_name cifar10_standard_preActResNet18_add_norm_exp3_final.pth --exp_name cure_exp7
#ENDBSUB

# 恒定的lr
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  <<ENDBSUB
#module load $PCOMMAND
#python train_cure.py --model PreActResNet18 --num_epochs 20 --cure_h_max 3.0 --batch_size 1024 --finetune --resumed_model_name cifar10_standard_preActResNet18_add_norm_exp3_final.pth --exp_name cure_exp8
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  <<ENDBSUB
#module load $PCOMMAND
#python train_cure.py --model PreActResNet18 --num_epochs 20 --cure_h_max 3.0 --batch_size 1024 --finetune --resumed_model_name cifar10_standard_preActResNet18_add_norm_exp3_final.pth --exp_name cure_exp9
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  <<ENDBSUB
#module load $PCOMMAND
#python train_cure.py --model PreActResNet18 --num_epochs 20 --cure_h_max 5.0 --batch_size 1024 --finetune --resumed_model_name cifar10_standard_preActResNet18_add_norm_exp3_final.pth --exp_name cure_exp10
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  <<ENDBSUB
#module load $PCOMMAND
#python train_cure.py --model PreActResNet18 --num_epochs 20 --cure_h_max 5.0 --batch_size 1024 --finetune --resumed_model_name cifar10_standard_preActResNet18_add_norm_exp3_final.pth --exp_name cure_exp11
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=8]"  <<ENDBSUB
#module load $PCOMMAND
#python train_cure.py --model PreActResNet18 --num_epochs 20 --cure_h_max 8.0 --batch_size 1024 --finetune --resumed_model_name cifar10_standard_preActResNet18_add_norm_exp3_final.pth --exp_name cure_exp12
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
##bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=8]"  <<ENDBSUB
##module load $PCOMMAND
##python train_cure.py --model PreActResNet18 --num_epochs 20 --cure_h_max 16.0 --batch_size 1024 --finetune --resumed_model_name cifar10_standard_preActResNet18_add_norm_exp3_final.pth --exp_name cure_exp13
##ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=8192,ngpus_excl_p=4]"  <<ENDBSUB
#module load $PCOMMAND
#python train_cure.py --model PreActResNet18 --num_epochs 20 --cure_h_max 16.0 --cure_lambda 8 --batch_size 1024 --finetune --resumed_model_name cifar10_standard_preActResNet18_add_norm_exp3_final.pth --exp_name cure_exp14
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=8192,ngpus_excl_p=4]"  <<ENDBSUB
#module load $PCOMMAND
#python train_cure.py --model PreActResNet18 --num_epochs 20 --cure_h_max 16.0 --cure_lambda 16 --batch_size 1024 --finetune --resumed_model_name cifar10_standard_preActResNet18_add_norm_exp3_final.pth --exp_name cure_exp15
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=2048,ngpus_excl_p=4]"  <<ENDBSUB
#module load $PCOMMAND
#python train_cure.py --model PreActResNet18 --lr 8e-4 --lr_schedule constant --num_epochs 15 --batch_size 512 --cure_h_max 1.5 --finetune --resumed_model_name cifar10_standard_preActResNet18_add_norm_exp3_final.pth --exp_name cure_h1.5_exp1
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  <<ENDBSUB
#module load $PCOMMAND
#python train_cure.py --model PreActResNet18 --lr 8e-4 --lr_schedule constant --num_epochs 15 --batch_size 1024 --cure_h_max 2.0 --finetune --resumed_model_name cifar10_standard_preActResNet18_add_norm_exp3_final.pth --exp_name cure_h2.0_exp1
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  <<ENDBSUB
#module load $PCOMMAND
#python train_clean.py --model ResNet18 --batch_size 1024 --exp_name clean_resnet18_exp1
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  <<ENDBSUB
#module load $PCOMMAND
#python train_clean.py --model ResNet18 --batch_size 256 --exp_name clean_resnet18_exp2
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_clean.py --model ResNet18 --batch_size 128 --exp_name clean_resnet18_exp3
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=8]"  <<ENDBSUB
#module load $PCOMMAND
#python train_cure.py --model ResNet18 --lr 8e-4 --lr_schedule constant --num_epochs 15 --batch_size 1024 --finetune --resumed_model_name clean_resnet18_exp1_final.pth --exp_name cure_ResNet18_exp1
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=8]"  <<ENDBSUB
#module load $PCOMMAND
#python train_cure.py --model ResNet18 --lr 8e-4 --lr_schedule constant --num_epochs 15 --batch_size 1024 --finetune --resumed_model_name clean_resnet18_exp2_final.pth --exp_name cure_ResNet18_exp2
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=8]"  <<ENDBSUB
#module load $PCOMMAND
#python train_cure.py --model ResNet18 --lr 8e-4 --lr_schedule constant --num_epochs 15 --batch_size 1024 --finetune --resumed_model_name clean_resnet18_exp3_final.pth --exp_name cure_ResNet18_exp3
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=8]"  <<ENDBSUB
#module load $PCOMMAND
#python train_cure.py --model ResNet18 --lr 8e-4 --lr_schedule constant --num_epochs 15 --batch_size 1024 --finetune --resumed_model_name clean_resnet18_exp3_final.pth --exp_name cure_ResNet18_exp4
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=8]"  <<ENDBSUB
#module load $PCOMMAND
#python train_cure.py --model ResNet18 --lr 8e-4 --lr_schedule constant --num_epochs 15 --batch_size 1024 --finetune --resumed_model_name clean_resnet18_exp2_final.pth --exp_name cure_ResNet18_exp5
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=8]"  <<ENDBSUB
#module load $PCOMMAND
#python train_cure.py --model ResNet18 --lr 8e-4 --lr_schedule constant --num_epochs 15 --batch_size 1024 --finetune --resumed_model_name clean_resnet18_exp2_final.pth --exp_name cure_ResNet18_exp6
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=8]"  <<ENDBSUB
#module load $PCOMMAND
#python train_cure.py --model ResNet18 --lr 2e-4 --lr_schedule constant --num_epochs 15 --batch_size 1024 --finetune --resumed_model_name clean_resnet18_exp2_final.pth --exp_name cure_ResNet18_exp7
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=8]"  <<ENDBSUB
#module load $PCOMMAND
#python train_cure.py --model ResNet18 --lr 2e-4 --lr_schedule constant --num_epochs 15 --batch_size 1024 --finetune --resumed_model_name clean_resnet18_exp2_final.pth --exp_name cure_ResNet18_exp8
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=8]"  <<ENDBSUB
#module load $PCOMMAND
#python train_cure.py --model ResNet18 --lr 2e-4 --lr_schedule constant --num_epochs 15 --batch_size 1024 --finetune --resumed_model_name clean_resnet18_exp2_final.pth --exp_name cure_ResNet18_exp9
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=8192,ngpus_excl_p=8]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm_cure.py --model PreActResNet18 --batch_size 1024 --exp_name fgsm_cure_exp1
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=8192,ngpus_excl_p=8]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm_cure.py --model PreActResNet18 --batch_size 1024 --cure_lambda 8 --exp_name fgsm_cure_exp2
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=8192,ngpus_excl_p=8]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm_cure.py --model PreActResNet18 --batch_size 1024 --cure_lambda 16 --exp_name fgsm_cure_exp3
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=8192,ngpus_excl_p=8]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm_cure.py --model PreActResNet18 --batch_size 1024 --cure_lambda 2 --exp_name fgsm_cure_exp4
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=8192,ngpus_excl_p=8]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm_cure.py --model PreActResNet18 --batch_size 1024 --cure_lambda 4 --cure_h_max 1.5 --exp_name fgsm_cure_exp5
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=8192,ngpus_excl_p=8]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm_cure.py --model PreActResNet18 --batch_size 1024 --cure_lambda 1.5 --cure_h_max 5.0 --exp_name fgsm_cure_exp6
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=8192,ngpus_excl_p=8]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm_cure.py --model PreActResNet18 --batch_size 1024 --cure_lambda 1.5 --cure_h_max 10.0 --exp_name fgsm_cure_exp7
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=8192,ngpus_excl_p=8]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm_cure.py --model PreActResNet18 --batch_size 1024 --cure_lambda 2.0 --cure_h_max 3.0 --train_fgsm_alpha 16 --exp_name fgsm_cure_exp8
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=8192,ngpus_excl_p=8]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm_cure.py --model PreActResNet18 --batch_size 1024 --cure_lambda 2.0 --cure_h_max 3.0 --train_fgsm_alpha 32 --exp_name fgsm_cure_exp9
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 10 -W 16:00 -R "rusage[mem=8192,ngpus_excl_p=4]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --batch_size 256 --train_fgsm_alpha 7 --eval_fgsm_alpha 7 --train_random_start --eval_random_start --exp_name 1013_fgsm7_rs
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 10 -W 16:00 -R "rusage[mem=8192,ngpus_excl_p=4]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --batch_size 256 --train_fgsm_alpha 10 --eval_fgsm_alpha 10 --train_random_start --eval_random_start --exp_name 1013_fgsm10_rs
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 10 -W 16:00 -R "rusage[mem=8192,ngpus_excl_p=4]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --batch_size 256 --train_fgsm_alpha 16 --eval_fgsm_alpha 16 --train_random_start --eval_random_start --exp_name 1013_fgsm16_rs
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 16 -W 08:00 -R "rusage[mem=8192,ngpus_excl_p=4]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --batch_size 256 --train_fgsm_alpha 7 --eval_fgsm_alpha 7 --exp_name 1013_fgsm7_not_rs
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 10 -W 16:00 -R "rusage[mem=8192,ngpus_excl_p=4]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --batch_size 256 --train_fgsm_alpha 10 --eval_fgsm_alpha 10 --exp_name 1013_fgsm10_not_rs
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 10 -W 16:00 -R "rusage[mem=8192,ngpus_excl_p=4]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --batch_size 256 --train_fgsm_alpha 16 --eval_fgsm_alpha 16 --exp_name 1013_fgsm16_not_rs
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 10 -W 16:00 -R "rusage[mem=8192,ngpus_excl_p=4]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --batch_size 256 --train_fgsm_alpha 7 --eval_fgsm_alpha 7 --train_random_start --eval_random_start --exp_name 1014_fgsm7_rs_exp2
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 10 -W 16:00 -R "rusage[mem=8192,ngpus_excl_p=4]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --batch_size 256 --train_fgsm_alpha 10 --eval_fgsm_alpha 10 --train_random_start --eval_random_start --exp_name 1014_fgsm10_rs_exp2
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 10 -W 16:00 -R "rusage[mem=8192,ngpus_excl_p=4]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --batch_size 256 --train_fgsm_alpha 16 --eval_fgsm_alpha 16 --train_random_start --eval_random_start --exp_name 1014_fgsm16_rs_exp2
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 16 -W 08:00 -R "rusage[mem=8192,ngpus_excl_p=4]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --batch_size 256 --train_fgsm_alpha 7 --eval_fgsm_alpha 7 --exp_name 1014_fgsm7_not_rs_exp2
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 10 -W 16:00 -R "rusage[mem=8192,ngpus_excl_p=4]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --batch_size 256 --train_fgsm_alpha 10 --eval_fgsm_alpha 10 --exp_name 1014_fgsm10_not_rs_exp2
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 10 -W 16:00 -R "rusage[mem=8192,ngpus_excl_p=4]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --batch_size 256 --train_fgsm_alpha 16 --eval_fgsm_alpha 16 --exp_name 1014_fgsm16_not_rs_exp2
#ENDBSUB

##PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
##bsub -n 10 -W 16:00 -R "rusage[mem=8192,ngpus_excl_p=4]"  <<ENDBSUB
##module load $PCOMMAND
##python train_deepfool.py --model PreActResNet18 --num_epochs 500 --batch_size 256 --deepfool_norm_dist l_2 --exp_name df_l_2_not_rs
##ENDBSUB
#
##PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
##bsub -n 10 -W 16:00 -R "rusage[mem=8192,ngpus_excl_p=4]"  <<ENDBSUB
##module load $PCOMMAND
##python train_deepfool.py --model PreActResNet18 --num_epochs 500 --batch_size 256 --deepfool_norm_dist l_2 --exp_name 1014_df_l_2_not_rs
##ENDBSUB
##
##PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
##bsub -n 10 -W 16:00 -R "rusage[mem=8192,ngpus_excl_p=4]"  <<ENDBSUB
##module load $PCOMMAND
##python train_deepfool.py --model PreActResNet18 --num_epochs 500 --batch_size 256 --deepfool_norm_dist l_inf --exp_name 1014_df_l_inf_not_rs
##ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 10 -W 16:00 -R "rusage[mem=8192,ngpus_excl_p=4]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --deepfool_norm_rs l_2 --deepfool_epsilon 2 --deepfool_rs --deepfool_norm_dist l_2 --exp_name 1014_df_l2_rs_l2_2
#ENDBSUB
##
##PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
##bsub -n 10 -W 16:00 -R "rusage[mem=8192,ngpus_excl_p=4]"  <<ENDBSUB
##module load $PCOMMAND
##python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --deepfool_rs --deepfool_epsilon 2 --deepfool_norm_rs l_2 --deepfool_norm_dist l_inf --exp_name 1014_df_l_inf_rs_l2_2
##ENDBSUB
##
##PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
##bsub -n 10 -W 16:00 -R "rusage[mem=8192,ngpus_excl_p=4]"  <<ENDBSUB
##module load $PCOMMAND
##python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --deepfool_norm_rs l_2 --deepfool_epsilon 4 --deepfool_rs --deepfool_norm_dist l_2 --exp_name 1014_df_l2_rs_l2_4
##ENDBSUB
##
##PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
##bsub -n 10 -W 16:00 -R "rusage[mem=8192,ngpus_excl_p=4]"  <<ENDBSUB
##module load $PCOMMAND
##python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --deepfool_rs --deepfool_epsilon 4 --deepfool_norm_rs l_2 --deepfool_norm_dist l_inf --exp_name 1014_df_l_inf_rs_l2_4
##ENDBSUB
##
##PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
##bsub -n 10 -W 16:00 -R "rusage[mem=8192,ngpus_excl_p=4]"  <<ENDBSUB
##module load $PCOMMAND
##python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --deepfool_norm_rs l_2 --deepfool_epsilon 8 --deepfool_rs --deepfool_norm_dist l_2 --exp_name 1014_df_l2_rs_l2_8
##ENDBSUB
##
##PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
##bsub -n 10 -W 16:00 -R "rusage[mem=8192,ngpus_excl_p=4]"  <<ENDBSUB
##module load $PCOMMAND
##python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --deepfool_rs --deepfool_epsilon 8 --deepfool_norm_rs l_2 --deepfool_norm_dist l_inf --exp_name 1014_df_l_inf_rs_l2_8
##ENDBSUB
##
##PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
##bsub -n 10 -W 16:00 -R "rusage[mem=8192,ngpus_excl_p=4]"  <<ENDBSUB
##module load $PCOMMAND
##python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --deepfool_norm_rs l_2 --deepfool_epsilon 16 --deepfool_rs --deepfool_norm_dist l_2 --exp_name 1014_df_l2_rs_l2_16
##ENDBSUB
##
##PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
##bsub -n 10 -W 16:00 -R "rusage[mem=8192,ngpus_excl_p=4]"  <<ENDBSUB
##module load $PCOMMAND
##python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --deepfool_rs --deepfool_epsilon 16 --deepfool_norm_rs l_2 --deepfool_norm_dist l_inf --exp_name 1014_df_l_inf_rs_l2_16
##ENDBSUB
##
##PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
##bsub -n 10 -W 16:00 -R "rusage[mem=8192,ngpus_excl_p=4]"  <<ENDBSUB
##module load $PCOMMAND
##python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --deepfool_norm_rs l_2 --deepfool_epsilon 32 --deepfool_rs --deepfool_norm_dist l_2 --exp_name 1014_df_l2_rs_l2_32
##ENDBSUB
##
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 10 -W 16:00 -R "rusage[mem=8192,ngpus_excl_p=4]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --deepfool_rs --deepfool_epsilon 32 --deepfool_norm_rs l_2 --deepfool_norm_dist l_inf --exp_name 1014_df_l_inf_rs_l2_32
#ENDBSUB
##
##PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
##bsub -n 10 -W 16:00 -R "rusage[mem=8192,ngpus_excl_p=4]"  <<ENDBSUB
##module load $PCOMMAND
##python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --deepfool_norm_rs l_2 --deepfool_epsilon 64 --deepfool_rs --deepfool_norm_dist l_2 --exp_name 1014_df_l2_rs_l2_64
##ENDBSUB
##
##PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
##bsub -n 10 -W 16:00 -R "rusage[mem=8192,ngpus_excl_p=4]"  <<ENDBSUB
##module load $PCOMMAND
##python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --deepfool_rs --deepfool_epsilon 64 --deepfool_norm_rs l_2 --deepfool_norm_dist l_inf --exp_name 1014_df_l_inf_rs_l2_64
##ENDBSUB
##
##PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
##bsub -n 10 -W 16:00 -R "rusage[mem=8192,ngpus_excl_p=4]"  <<ENDBSUB
##module load $PCOMMAND
##python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --deepfool_norm_rs l_2 --deepfool_epsilon 128 --deepfool_rs --deepfool_norm_dist l_2 --exp_name 1014_df_l2_rs_l2_128
##ENDBSUB
##
##PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
##bsub -n 10 -W 16:00 -R "rusage[mem=8192,ngpus_excl_p=4]"  <<ENDBSUB
##module load $PCOMMAND
##python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --deepfool_rs --deepfool_epsilon 128 --deepfool_norm_rs l_2 --deepfool_norm_dist l_inf --exp_name 1014_df_l_inf_rs_l2_128
##ENDBSUB
##
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 10 -W 16:00 -R "rusage[mem=8192,ngpus_excl_p=4]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --deepfool_rs --deepfool_epsilon 2 --deepfool_norm_rs l_inf   --deepfool_norm_dist l_2 --exp_name 1014_df_l2_rs_linf_2
#ENDBSUB
##
##PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
##bsub -n 10 -W 16:00 -R "rusage[mem=8192,ngpus_excl_p=4]"  <<ENDBSUB
##module load $PCOMMAND
##python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --deepfool_rs --deepfool_epsilon 2 --deepfool_norm_rs l_inf --deepfool_norm_dist l_inf --exp_name 1014_df_linf_rs_linf_2
##ENDBSUB
##
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 10 -W 16:00 -R "rusage[mem=8192,ngpus_excl_p=4]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --deepfool_rs --deepfool_epsilon 4 --deepfool_norm_rs l_inf   --deepfool_norm_dist l_2 --exp_name 1014_df_l2_rs_linf_4
#ENDBSUB
##
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 10 -W 16:00 -R "rusage[mem=8192,ngpus_excl_p=4]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --deepfool_rs --deepfool_epsilon 4 --deepfool_norm_rs l_inf --deepfool_norm_dist l_inf --exp_name 1014_df_linf_rs_linf_4
#ENDBSUB
##
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 10 -W 16:00 -R "rusage[mem=8192,ngpus_excl_p=4]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --deepfool_rs --deepfool_epsilon 8 --deepfool_norm_rs l_inf   --deepfool_norm_dist l_2 --exp_name 1014_df_l2_rs_linf_8
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 10 -W 16:00 -R "rusage[mem=8192,ngpus_excl_p=4]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --deepfool_rs --deepfool_epsilon 8 --deepfool_norm_rs l_inf --deepfool_norm_dist l_inf --exp_name 1014_df_linf_rs_linf_8
#ENDBSUB
##
##
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 10 -W 16:00 -R "rusage[mem=8192,ngpus_excl_p=4]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --deepfool_rs --deepfool_epsilon 16 --deepfool_norm_rs l_inf   --deepfool_norm_dist l_2 --exp_name 1014_df_l2_rs_linf_16
#ENDBSUB
##
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 10 -W 16:00 -R "rusage[mem=8192,ngpus_excl_p=4]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --deepfool_rs --deepfool_epsilon 16 --deepfool_norm_rs l_inf --deepfool_norm_dist l_inf --exp_name 1014_df_linf_rs_linf_16
#ENDBSUB
#
##PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
##bsub -n 12 -W 16:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  <<ENDBSUB
##module load $PCOMMAND
##python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256  --deepfool_norm_dist l_2 --deepfool_alpha 1.25 --exp_name 1014_df_l2_not_rs_alpha1.25
##ENDBSUB
##
##PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
##bsub -n 12 -W 16:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  <<ENDBSUB
##module load $PCOMMAND
##python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --deepfool_norm_dist l_inf --deepfool_alpha 1.25 --exp_name 1014_df_linf_not_rs_alpha1.25
##ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 12 -W 16:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256  --deepfool_norm_dist l_2 --deepfool_alpha 1.5 --exp_name 1014_df_l2_not_rs_alpha1.5
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 12 -W 16:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --deepfool_norm_dist l_inf --deepfool_alpha 1.5 --exp_name 1014_df_linf_not_rs_alpha1.5
#ENDBSUB
#
##PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
##bsub -n 12 -W 16:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  <<ENDBSUB
##module load $PCOMMAND
##python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256  --deepfool_norm_dist l_2 --deepfool_alpha 1.75 --exp_name 1014_df_l2_not_rs_alpha1.75
##ENDBSUB
#
##PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
##bsub -n 12 -W 16:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  <<ENDBSUB
##module load $PCOMMAND
##python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --deepfool_norm_dist l_inf --deepfool_alpha 1.75 --exp_name 1014_df_linf_not_rs_alpha1.75
##ENDBSUB
#
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 12 -W 16:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256  --deepfool_norm_dist l_2 --deepfool_alpha 2 --exp_name 1014_df_l2_not_rs_alpha2
#ENDBSUB
#
##PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
##bsub -n 12 -W 16:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  <<ENDBSUB
##module load $PCOMMAND
##python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --deepfool_norm_dist l_inf --deepfool_alpha 2 --exp_name 1014_df_linf_not_rs_alpha2
##ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 12 -W 16:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --deepfool_norm_dist l_2 --exp_name 1015_df_l2_not_rs
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 12 -W 16:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --deepfool_norm_dist l_2 --train_deepfool_classes_num 3 --exp_name 1015_df_l2_not_rs_cls3
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 12 -W 16:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --deepfool_norm_dist l_2 --train_deepfool_classes_num 4 --exp_name 1015_df_l2_not_rs_cls4
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 12 -W 16:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --deepfool_norm_dist l_inf --exp_name 1015_df_linf_not_rs
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 12 -W 16:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --deepfool_rs --deepfool_norm_rs l_inf --deepfool_epsilon 8 --deepfool_norm_dist l_2 --exp_name 1015_df_l2_rs_linf_8
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 12 -W 16:00 -R "rusage[mem=8192,ngpus_excl_p=4]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --deepfool_rs --deepfool_norm_rs l_inf --deepfool_epsilon 16 --deepfool_norm_dist l_2 --exp_name 1015_df_l2_rs_linf_16
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 12 -W 16:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --deepfool_rs --deepfool_norm_rs l_2 --deepfool_epsilon 128 --deepfool_norm_dist l_2 --exp_name 1015_df_l2_rs_l2_128
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 12 -W 16:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --deepfool_rs --deepfool_norm_rs l_inf --deepfool_epsilon 8 --deepfool_norm_dist l_inf --exp_name 1015_df_linf_rs_linf_8
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 12 -W 16:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --deepfool_rs --deepfool_norm_rs l_inf --deepfool_epsilon 8 --deepfool_norm_dist l_2 --train_deepfool_classes_num 3 --exp_name 1015_df_l2_rs_linf8_cls3
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 12 -W 16:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --deepfool_rs --deepfool_norm_rs l_inf --deepfool_epsilon 8 --deepfool_norm_dist l_2 --train_deepfool_classes_num 4 --exp_name 1015_df_l2_rs_linf8_cls4
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 12 -W 16:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --deepfool_rs --deepfool_norm_rs l_inf --deepfool_epsilon 8 --deepfool_norm_dist l_2 --train_deepfool_classes_num 10 --exp_name 1015_df_l2_rs_linf8_cls10
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 12 -W 16:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  <<ENDBSUB
#module load $PCOMMAND
#python train_differ_fgsm.py --model PreActResNet18 --num_epochs 110 --batch_size 256 --exp_name 1016_diff_fgsm
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 12 -W 16:00 -R "rusage[mem=4096,ngpus_excl_p=8]"  <<ENDBSUB
#module load $PCOMMAND
#python train_cure.py --model ResNet18 --lr 8e-4 --lr_schedule constant --num_epochs 15 --batch_size 1024 --finetune --resumed_model_name ckpt_standard_rn18.pth --exp_name 1018_cure
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 12 -W 16:00 -R "rusage[mem=4096,ngpus_excl_p=8]"  <<ENDBSUB
#module load $PCOMMAND
#python train_cure.py --model ResNet18 --lr 8e-4 --lr_schedule constant --num_epochs 15 --batch_size 1024 --finetune --resumed_model_name ckpt_standard_rn18.pth --exp_name 1018_cure_test_pgd
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 12 -W 16:00 -R "rusage[mem=4096,ngpus_excl_p=8]"  <<ENDBSUB
#module load $PCOMMAND
#python train_cure.py --model ResNet18 --lr 8e-4 --lr_schedule constant --num_epochs 15 --batch_size 1024 --finetune --resumed_model_name ckpt_standard_rn18.pth --exp_name 1018_cure_test_pgd_exp1 # 改了pgd中的std
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 12 -W 16:00 -R "rusage[mem=4096,ngpus_excl_p=8]"  <<ENDBSUB
#module load $PCOMMAND
#python train_cure.py --model ResNet18 --lr 8e-4 --lr_schedule constant --num_epochs 15 --batch_size 1024 --finetune --resumed_model_name ckpt_standard_rn18.pth --exp_name 1018_cure_std # pdg+std
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 12 -W 16:00 -R "rusage[mem=4096,ngpus_excl_p=8]"  <<ENDBSUB
#module load $PCOMMAND
#python train_cure.py --model ResNet18 --lr 8e-4 --lr_schedule constant --num_epochs 15 --batch_size 1024 --finetune --resumed_model_name ckpt_standard_rn18.pth --exp_name 1018_cure_pgd_std_z
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 12 -W 16:00 -R "rusage[mem=4096,ngpus_excl_p=8]"  <<ENDBSUB
#module load $PCOMMAND
#python train_cure.py --model ResNet18 --lr 8e-4 --lr_schedule constant --num_epochs 15 --batch_size 1024 --finetune --resumed_model_name ckpt_standard_rn18.pth --exp_name 1018_cure_std_z
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 12 -W 16:00 -R "rusage[mem=4096,ngpus_excl_p=8]"  <<ENDBSUB
#module load $PCOMMAND
#python train_cure.py --model ResNet18 --lr 8e-4 --lr_schedule constant --num_epochs 15 --batch_size 1024 --finetune --resumed_model_name ckpt_standard_rn18.pth --exp_name 1018_replace_all
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 12 -W 16:00 -R "rusage[mem=4096,ngpus_excl_p=8]"  <<ENDBSUB
#module load $PCOMMAND
#python train_cure.py --model ResNet18 --lr 8e-4 --lr_schedule constant --num_epochs 15 --batch_size 1024 --finetune --resumed_model_name ckpt_standard_rn18.pth --exp_name 1018_replace_all_exp1
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 12 -W 16:00 -R "rusage[mem=4096,ngpus_excl_p=8]"  <<ENDBSUB
#module load $PCOMMAND
#python train_cure.py --model ResNet18 --lr 8e-4 --lr_schedule constant --num_epochs 15 --batch_size 1024 --finetune --resumed_model_name ckpt_standard_rn18.pth --exp_name 1019_cure
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 12 -W 16:00 -R "rusage[mem=4096,ngpus_excl_p=8]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm_cure.py --model ResNet18 --lr 8e-4 --lr_schedule constant --num_epochs 15 --batch_size 1024 --finetune --resumed_model_name ckpt_standard_rn18.pth --exp_name 1019_fgsm_cure
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 12 -W 16:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --deepfool_norm_dist l_2 --exp_name 1020_df_l2_not_rs
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 12 -W 16:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --deepfool_norm_dist l_2 --exp_name 1020_df_linf_not_rs
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 12 -W 16:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --deepfool_norm_dist l_2 --exp_name 1020_df_linf_not_rs_eval
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 12 -W 16:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --deepfool_norm_dist l_inf --exp_name 1020_df_linf_not_rs_exp2
#ENDBSUB

#######6th week#################################
################################################
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 12 -W 16:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --deepfool_norm_dist l_2 --exp_name 1020_1iter_DF_l2_sgd_intrain_notrs
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 12 -W 16:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --deepfool_norm_dist l_2 --optimizer adam --exp_name 1020_1iter_DF_l2_adam_intrain_notrs
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 12 -W 16:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --deepfool_norm_dist l_inf --exp_name 1020_1iter_DF_linf_sgd_intrain_notrs
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 12 -W 16:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --deepfool_norm_dist l_inf --optimizer adam --exp_name 1020_1iter_DF_linf_adam_intrain_notrs
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 12 -W 16:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --deepfool_norm_dist l_2  --train_deepfool_eval --exp_name 1020_1iter_DF_l2_sgd_ineval_notrs
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 12 -W 16:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --deepfool_norm_dist l_2 --optimizer adam --train_deepfool_eval --exp_name 1020_1iter_DF_l2_adam_ineval_notrs
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 12 -W 16:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --deepfool_norm_dist l_inf --train_deepfool_eval --exp_name 1020_1iter_DF_linf_sgd_ineval_notrs
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 12 -W 16:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --deepfool_norm_dist l_inf --optimizer adam --train_deepfool_eval --exp_name 1020_1iter_DF_linf_adam_ineval_notrs
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 12 -W 16:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --train_deepfool_max_iter 2 --deepfool_norm_dist l_2 --exp_name 1020_2iter_DF_l2_sgd_intrain_notrs
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 12 -W 16:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --train_deepfool_max_iter 2 --deepfool_norm_dist l_2 --optimizer adam --exp_name 1020_2iter_DF_l2_adam_intrain_notrs
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 12 -W 16:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --train_deepfool_max_iter 2 --deepfool_norm_dist l_inf --exp_name 1020_2iter_DF_linf_sgd_intrain_notrs
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 12 -W 16:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --train_deepfool_max_iter 2 --deepfool_norm_dist l_inf --optimizer adam --exp_name 1020_2iter_DF_linf_adam_intrain_notrs
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 12 -W 16:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --train_deepfool_max_iter 2 --deepfool_norm_dist l_2 --train_deepfool_eval --exp_name 1020_2iter_DF_l2_sgd_ineval_notrs
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 12 -W 16:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --train_deepfool_max_iter 2 --deepfool_norm_dist l_2 --optimizer adam --train_deepfool_eval --exp_name 1020_2iter_DF_l2_adam_ineval_notrs
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 12 -W 16:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --train_deepfool_max_iter 2 --deepfool_norm_dist l_inf --train_deepfool_eval --exp_name 1020_2iter_DF_linf_sgd_ineval_notrs
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 12 -W 16:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --train_deepfool_max_iter 2 --deepfool_norm_dist l_inf --optimizer adam --train_deepfool_eval --exp_name 1020_2iter_DF_linf_adam_ineval_notrs
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 12 -W 16:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --deepfool_norm_dist l_2 --deepfool_rs --exp_name 1020_1iter_DF_l2_sgd_intrain_rs
#ENDBSUB

#11256614
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 12 -W 16:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --deepfool_norm_dist l_2 --deepfool_rs --optimizer adam --exp_name 1020_1iter_DF_l2_adam_intrain_rs
#ENDBSUB

#11256698
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 12 -W 16:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --deepfool_norm_dist l_inf --deepfool_rs --exp_name 1020_1iter_DF_linf_sgd_intrain_rs
#ENDBSUB

#11256699
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 12 -W 16:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --deepfool_norm_dist l_inf --deepfool_rs --optimizer adam --exp_name 1020_1iter_DF_linf_adam_intrain_rs
#ENDBSUB
#
#11256700
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 12 -W 16:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --deepfool_norm_dist l_2 --deepfool_rs --train_deepfool_eval --exp_name 1020_1iter_DF_l2_sgd_ineval_rs
#ENDBSUB
#
#11256701
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 12 -W 16:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --deepfool_norm_dist l_2 --deepfool_rs --optimizer adam --train_deepfool_eval --exp_name 1020_1iter_DF_l2_adam_ineval_rs
#ENDBSUB
#
#11256702
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 12 -W 16:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --deepfool_norm_dist l_inf --deepfool_rs --train_deepfool_eval --exp_name 1020_1iter_DF_linf_sgd_ineval_rs
#ENDBSUB
#

#11256703
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 12 -W 16:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --deepfool_norm_dist l_inf --deepfool_rs --optimizer adam --train_deepfool_eval --exp_name 1020_1iter_DF_linf_adam_ineval_rs
#ENDBSUB
##
##11256704
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 12 -W 16:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --train_deepfool_max_iter 2 --deepfool_rs --deepfool_norm_dist l_2 --exp_name 1020_2iter_DF_l2_sgd_intrain_rs(08)
#ENDBSUB
##
##11256705
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 12 -W 16:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --train_deepfool_max_iter 2 --deepfool_rs --deepfool_norm_dist l_2 --optimizer adam --exp_name 1020_2iter_DF_l2_adam_intrain_rs(11)
#ENDBSUB
##
##11256706
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 12 -W 16:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --train_deepfool_max_iter 2 --deepfool_rs --deepfool_norm_dist l_inf --exp_name 1020_2iter_DF_linf_sgd_intrain_rs(14)
#ENDBSUB
##
##11256707
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 12 -W 16:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --train_deepfool_max_iter 2 --deepfool_rs --deepfool_norm_dist l_inf --optimizer adam --exp_name 1020_2iter_DF_linf_adam_intrain_rs(18)
#ENDBSUB
#
#11256708
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 12 -W 16:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --train_deepfool_max_iter 2 --deepfool_rs --deepfool_norm_dist l_2 --train_deepfool_eval --exp_name 1020_2iter_DF_l2_sgd_ineval_rs(20)
#ENDBSUB
#
#11256709
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 12 -W 16:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --train_deepfool_max_iter 2 --deepfool_rs --deepfool_norm_dist l_2 --optimizer adam --train_deepfool_eval --exp_name 1020_2iter_DF_l2_adam_ineval_rs(26)
#ENDBSUB
#
#11256710
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 12 -W 16:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --train_deepfool_max_iter 2 --deepfool_rs --deepfool_norm_dist l_inf --train_deepfool_eval --exp_name 1020_2iter_DF_linf_sgd_ineval_rs(30)
#ENDBSUB
#
#11256711
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 12 -W 16:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --train_deepfool_max_iter 2 --deepfool_rs --deepfool_norm_dist l_inf --optimizer adam --train_deepfool_eval --exp_name 1020_2iter_DF_linf_adam_ineval_rs(32)
#ENDBSUB

###########1021###############
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 12 -W 16:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --train_epsilon 8 --train_fgsm_alpha 8 --exp_name 1021_fgsm_alpha8_notrs
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 12 -W 16:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --train_epsilon 8 --train_fgsm_alpha 10 --exp_name 1021_fgsm_alpha10_notrs
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 12 -W 16:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  <<ENDBSUBc
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --train_epsilon 8 --train_fgsm_alpha 12 --exp_name 1021_fgsm_alpha12_notrs
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 12 -W 16:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --train_epsilon 8 --train_fgsm_alpha 16 --exp_name 1021_fgsm_alpha16_notrs
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 12 -W 16:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --train_epsilon 8 --train_fgsm_alpha 8 --train_random_start --exp_name 1021_fgsm_alpha8_rs
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 12 -W 16:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --train_epsilon 8 --train_fgsm_alpha 10 --train_random_start --exp_name 1021_fgsm_alpha10_rs
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 12 -W 16:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --train_epsilon 8 --train_fgsm_alpha 12 --train_random_start --exp_name 1021_fgsm_alpha12_rs
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 12 -W 16:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --train_epsilon 8 --train_fgsm_alpha 16 --train_random_start --exp_name 1021_fgsm_alpha16_rs
#ENDBSUB

#11267430
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 12 -W 16:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --train_epsilon 8 --train_fgsm_alpha 7 --percentile 85 --exp_name 1021_fgsm_alpha7_85_notrs
#ENDBSUB

#11267459
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 12 -W 16:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --train_epsilon 8 --train_fgsm_alpha 8 --percentile 85 --exp_name 1021_fgsm_alpha8_85_notrs
#ENDBSUB
#
#11267460
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 12 -W 16:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --train_epsilon 8 --train_fgsm_alpha 10 --percentile 85 --exp_name 1021_fgsm_alpha10_85_notrs
#ENDBSUB
#
#11268901
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 12 -W 16:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --train_epsilon 8 --train_fgsm_alpha 12 --percentile 85 --exp_name 1021_fgsm_alpha12_85_notrs
#ENDBSUB
#
#11267468
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
##bsub -n 12 -W 16:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  <<ENDBSUB
##module load $PCOMMAND
##python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --train_epsilon 8 --train_fgsm_alpha 16 --percentile 85 --exp_name 1021_fgsm_alpha16_85_notrs
##ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 12 -W 16:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --train_epsilon 8 --train_fgsm_alpha 20 --percentile 85 --exp_name 1021_fgsm_alpha20_85_notrs
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 12 -W 16:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --train_epsilon 8 --train_fgsm_alpha 32 --percentile 85 --exp_name 1021_fgsm_alpha32_85_notrs
#ENDBSUB

#
#
#11267463
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 12 -W 16:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --train_epsilon 8 --train_fgsm_alpha 7 --percentile 85 --train_random_start --exp_name 1021_fgsm_alpha7_85_rs
#ENDBSUB
#
#11267464
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 12 -W 16:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --train_epsilon 8 --train_fgsm_alpha 8 --percentile 85 --train_random_start --exp_name 1021_fgsm_alpha8_85_rs
#ENDBSUB
#
#11267466
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 12 -W 16:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --train_epsilon 8 --train_fgsm_alpha 10 --percentile 85 --train_random_start --exp_name 1021_fgsm_alpha10_85_rs
#ENDBSUB
#
#11267469
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 12 -W 16:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --train_epsilon 8 --train_fgsm_alpha 12 --percentile 85 --train_random_start --exp_name 1021_fgsm_alpha12_85_rs
#ENDBSUB
#
#11267470
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 12 -W 16:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --train_epsilon 8 --train_fgsm_alpha 16 --percentile 85 --train_random_start --exp_name 1021_fgsm_alpha16_85_rs
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
##bsub -n 12 -W 16:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  <<ENDBSUB
##module load $PCOMMAND
##python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --train_epsilon 8 --train_fgsm_alpha 20 --percentile 85 --train_random_start --exp_name 1021_fgsm_alpha20_85_rs
##ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 12 -W 16:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --train_epsilon 8 --train_fgsm_alpha 32 --percentile 85 --train_random_start --exp_name 1021_fgsm_alpha32_85_rs
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 12 -W 16:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --train_epsilon 8 --train_fgsm_alpha 32 --percentile 95 --train_random_start --exp_name 1021_fgsm_alpha32_95_rs
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 12 -W 16:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --train_epsilon 8 --train_fgsm_alpha 10 --percentile 75 --exp_name 1021_fgsm_alpha10_75_notrs
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
##bsub -n 12 -W 16:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  <<ENDBSUB
##module load $PCOMMAND
##python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --train_epsilon 8 --train_fgsm_alpha 10 --percentile 65 --exp_name 1021_fgsm_alpha10_65_notrs
##ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 12 -W 16:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --train_epsilon 8 --train_fgsm_alpha 10 --percentile 50 --exp_name 1021_fgsm_alpha10_50_notrs
#ENDBSUB

##############without clamp##############
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 12 -W 16:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --train_epsilon 8 --train_fgsm_alpha 8 --percentile 85 --exp_name 1022_fgsm_alpha8_85_notrs
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 12 -W 16:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --train_epsilon 8 --train_fgsm_alpha 8 --percentile 75 --exp_name 1022_fgsm_alpha8_75_notrs
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 12 -W 16:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --train_epsilon 8 --train_fgsm_alpha 8 --percentile 65 --exp_name 1022_fgsm_alpha8_65_notrs
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 12 -W 16:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --train_epsilon 8 --train_fgsm_alpha 8 --percentile 50 --exp_name 1022_fgsm_alpha8_50_notrs
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
##bsub -n 12 -W 16:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  <<ENDBSUB
##module load $PCOMMAND
##python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --train_epsilon 8 --train_fgsm_alpha 8 --percentile 95 --exp_name 1022_fgsm_alpha8_95_notrs
##ENDBSUB

###########不同step的alpha#########
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 12 -W 16:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --train_epsilon 8 --exp_name 1022_fgsm_diff_alpha_notrs
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 12 -W 16:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --train_epsilon 8 --min 4 --max 10 --exp_name 1022_fgsm_diff_alpha_4_10_notrs
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 12 -W 16:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --train_epsilon 8 --min 7 --max 10 --exp_name 1022_fgsm_diff_alpha_7_10_notrs
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 12 -W 16:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --train_epsilon 8 --min 7 --max 8 --exp_name 1022_fgsm_diff_alpha_7_8_notrs
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 12 -W 16:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --train_epsilon 8 --min 1 --max 8 --exp_name 1022_fgsm_diff_alpha_1_8_notrs
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 12 -W 16:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --train_epsilon 8 --min 2 --max 8 --exp_name 1022_fgsm_diff_alpha_2_8_notrs
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 12 -W 16:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --train_epsilon 8 --min 4 --max 8 --exp_name 1022_fgsm_diff_alpha_4_8_notrs
#ENDBSUB

###########1028###################
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 12 -W 16:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --epsilon 8 --train_fgsm_alpha 8 --eval_pgd_alpha 2 --exp_name 1028_standard_fgsm8
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 12 -W 16:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --epsilon 1 --train_fgsm_alpha 1 --eval_pgd_alpha 0.25 --exp_name 1028_standard_fgsm1
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 12 -W 16:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --epsilon 2 --train_fgsm_alpha 2 --eval_pgd_alpha 0.5 --exp_name 1028_standard_fgsm2
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 12 -W 16:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --epsilon 3 --train_fgsm_alpha 3 --eval_pgd_alpha 0.75 --exp_name 1028_standard_fgsm3
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 12 -W 16:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --epsilon 4 --train_fgsm_alpha 4 --eval_pgd_alpha 1 --exp_name 1028_standard_fgsm4
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 12 -W 16:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --epsilon 5 --train_fgsm_alpha 5 --eval_pgd_alpha 1.25 --exp_name 1028_standard_fgsm5
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --epsilon 6 --train_fgsm_alpha 6 --eval_pgd_alpha 1.5 --exp_name 1028_standard_fgsm6
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
##bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
##module load $PCOMMAND
##python train_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --epsilon 7 --train_fgsm_alpha 7 --eval_pgd_alpha 1.75 --exp_name 1028_standard_fgsm7
##ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --epsilon 9 --train_fgsm_alpha 9 --eval_pgd_alpha 2.25 --exp_name 1028_standard_fgsm9
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --epsilon 10 --train_fgsm_alpha 10 --eval_pgd_alpha 2.5 --exp_name 1028_standard_fgsm10
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --epsilon 11 --train_fgsm_alpha 11 --eval_pgd_alpha 2.75 --exp_name 1028_standard_fgsm11
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --epsilon 12 --train_fgsm_alpha 12 --eval_pgd_alpha 3 --exp_name 1028_standard_fgsm12
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --epsilon 13 --train_fgsm_alpha 13 --eval_pgd_alpha 3.25 --exp_name 1028_standard_fgsm13
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --epsilon 14 --train_fgsm_alpha 14 --eval_pgd_alpha 3.5 --exp_name 1028_standard_fgsm14
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --epsilon 15 --train_fgsm_alpha 15 --eval_pgd_alpha 3.75 --exp_name 1028_standard_fgsm15
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --epsilon 16 --train_fgsm_alpha 16 --eval_pgd_alpha 4 --exp_name 1028_standard_fgsm16
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --epsilon 8 --train_fgsm_alpha 1 --train_random_start --eval_pgd_alpha 2 --exp_name 1028_random_fgsm8_alpha1
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --epsilon 8 --train_fgsm_alpha 2 --train_random_start --eval_pgd_alpha 2 --exp_name 1028_random_fgsm8_alpha2
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --epsilon 8 --train_fgsm_alpha 3 --train_random_start --eval_pgd_alpha 2 --exp_name 1028_random_fgsm8_alpha3
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 14:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --epsilon 8 --train_fgsm_alpha 4 --train_random_start --eval_pgd_alpha 2 --exp_name 1028_random_fgsm8_alpha4
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --epsilon 8 --train_fgsm_alpha 5 --train_random_start --eval_pgd_alpha 2 --exp_name 1028_random_fgsm8_alpha5
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --epsilon 8 --train_fgsm_alpha 6 --train_random_start --eval_pgd_alpha 2 --exp_name 1028_random_fgsm8_alpha6
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 14:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --epsilon 8 --train_fgsm_alpha 7 --train_random_start --eval_pgd_alpha 2 --exp_name 1028_random_fgsm8_alpha7
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 14:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --epsilon 8 --train_fgsm_alpha 8 --train_random_start --eval_pgd_alpha 2 --exp_name 1028_random_fgsm8_alpha8
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 14:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --epsilon 8 --train_fgsm_alpha 9 --train_random_start --eval_pgd_alpha 2 --exp_name 1028_random_fgsm8_alpha9
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 14:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --epsilon 8 --train_fgsm_alpha 10 --train_random_start --eval_pgd_alpha 2 --exp_name 1028_random_fgsm8_alpha10
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
##bsub -n 8 -W 14:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
##module load $PCOMMAND
##python train_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --epsilon 8 --train_fgsm_alpha 11 --train_random_start --eval_pgd_alpha 2 --exp_name 1028_random_fgsm8_alpha11
##ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 14:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --epsilon 8 --train_fgsm_alpha 12 --train_random_start --eval_pgd_alpha 2 --exp_name 1028_random_fgsm8_alpha12
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --epsilon 8 --train_fgsm_alpha 13 --train_random_start --eval_pgd_alpha 2 --exp_name 1028_random_fgsm8_alpha13
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --epsilon 8 --train_fgsm_alpha 14 --train_random_start --eval_pgd_alpha 2 --exp_name 1028_random_fgsm8_alpha14
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --epsilon 8 --train_fgsm_alpha 15 --train_random_start --eval_pgd_alpha 2 --exp_name 1028_random_fgsm8_alpha15
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --epsilon 8 --train_fgsm_alpha 16 --train_random_start --eval_pgd_alpha 2 --exp_name 1028_random_fgsm8_alpha16
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --epsilon 1 --train_fgsm_alpha 1 --train_random_start --eval_pgd_alpha 0.25 --exp_name 1028_random_fgsm1
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --epsilon 2 --train_fgsm_alpha 2 --train_random_start --eval_pgd_alpha 0.5 --exp_name 1028_random_fgsm2
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --epsilon 3 --train_fgsm_alpha 3 --train_random_start --eval_pgd_alpha 0.75 --exp_name 1028_random_fgsm3
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --epsilon 4 --train_fgsm_alpha 4 --train_random_start --eval_pgd_alpha 1 --exp_name 1028_random_fgsm4
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --epsilon 5 --train_fgsm_alpha 5 --train_random_start --eval_pgd_alpha 1.25 --exp_name 1028_random_fgsm5
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --epsilon 6 --train_fgsm_alpha 6 --train_random_start --eval_pgd_alpha 1.5 --exp_name 1028_random_fgsm6
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --epsilon 7 --train_fgsm_alpha 7 --train_random_start --eval_pgd_alpha 1.75 --exp_name 1028_random_fgsm7
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --epsilon 9 --train_fgsm_alpha 9 --train_random_start --eval_pgd_alpha 2.25 --exp_name 1028_random_fgsm9
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --epsilon 10 --train_fgsm_alpha 10 --train_random_start --eval_pgd_alpha 2.5 --exp_name 1028_random_fgsm10
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --epsilon 11 --train_fgsm_alpha 11 --train_random_start --eval_pgd_alpha 2.75 --exp_name 1028_random_fgsm11
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --epsilon 12 --train_fgsm_alpha 12 --train_random_start --eval_pgd_alpha 3 --exp_name 1028_random_fgsm12
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --epsilon 13 --train_fgsm_alpha 13 --train_random_start --eval_pgd_alpha 3.25 --exp_name 1028_random_fgsm13
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --epsilon 14 --train_fgsm_alpha 14 --train_random_start --eval_pgd_alpha 3.5 --exp_name 1028_random_fgsm14
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --epsilon 15 --train_fgsm_alpha 15 --train_random_start --eval_pgd_alpha 3.75 --exp_name 1028_random_fgsm15
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --epsilon 16 --train_fgsm_alpha 16 --train_random_start --eval_pgd_alpha 4 --exp_name 1028_random_fgsm16
#ENDBSUB

####################1030####################
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --epsilon 8 --train_fgsm_alpha 8 --eval_pgd_alpha 2 --exp_name 1030_anisotropy_fgsm8
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --epsilon 9 --train_fgsm_alpha 9 --eval_pgd_alpha 2.25 --exp_name 1030_anisotropy_fgsm9
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --epsilon 10 --train_fgsm_alpha 10 --eval_pgd_alpha 2.5 --exp_name 1030_anisotropy_fgsm10
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --epsilon 11 --train_fgsm_alpha 11 --eval_pgd_alpha 2.75 --exp_name 1030_anisotropy_fgsm11
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --epsilon 12 --train_fgsm_alpha 12 --eval_pgd_alpha 3 --exp_name 1030_anisotropy_fgsm12
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --epsilon 13 --train_fgsm_alpha 13 --eval_pgd_alpha 3.25 --exp_name 1030_anisotropy_fgsm13
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --epsilon 14 --train_fgsm_alpha 14 --eval_pgd_alpha 3.5 --exp_name 1030_anisotropy_fgsm14
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --epsilon 15 --train_fgsm_alpha 15 --eval_pgd_alpha 3.75 --exp_name 1030_anisotropy_fgsm15
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --epsilon 16 --train_fgsm_alpha 16 --eval_pgd_alpha 4 --exp_name 1030_anisotropy_fgsm16
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --epsilon 1 --train_fgsm_alpha 1 --eval_pgd_alpha 0.25 --exp_name 1030_anisotropy_fgsm1
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --epsilon 2 --train_fgsm_alpha 2 --eval_pgd_alpha 0.5 --exp_name 1030_anisotropy_fgsm2
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --epsilon 3 --train_fgsm_alpha 3 --eval_pgd_alpha 0.75 --exp_name 1030_anisotropy_fgsm3
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --epsilon 4 --train_fgsm_alpha 4 --eval_pgd_alpha 1 --exp_name 1030_anisotropy_fgsm4
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --epsilon 5 --train_fgsm_alpha 5 --eval_pgd_alpha 1.25 --exp_name 1030_anisotropy_fgsm5
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --epsilon 6 --train_fgsm_alpha 6 --eval_pgd_alpha 1.5 --exp_name 1030_anisotropy_fgsm6
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --epsilon 7 --train_fgsm_alpha 7 --eval_pgd_alpha 1.75 --exp_name 1030_anisotropy_fgsm7
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --epsilon 16 --train_fgsm_alpha 16 --eval_pgd_alpha 4 --train_random_start --exp_name 1030_anisotropy_fgsm16_rs
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --epsilon 15 --train_fgsm_alpha 15 --eval_pgd_alpha 3.75 --train_random_start --exp_name 1030_anisotropy_fgsm15_rs
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --epsilon 14 --train_fgsm_alpha 14 --eval_pgd_alpha 3.5 --train_random_start --exp_name 1030_anisotropy_fgsm14_rs
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --epsilon 13 --train_fgsm_alpha 13 --eval_pgd_alpha 3.25 --train_random_start --exp_name 1030_anisotropy_fgsm13_rs
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --epsilon 12 --train_fgsm_alpha 12 --eval_pgd_alpha 3 --train_random_start --exp_name 1030_anisotropy_fgsm12_rs
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --epsilon 11 --train_fgsm_alpha 11 --eval_pgd_alpha 2.75 --train_random_start --exp_name 1030_anisotropy_fgsm11_rs
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --epsilon 10 --train_fgsm_alpha 10 --eval_pgd_alpha 2.5 --train_random_start --exp_name 1030_anisotropy_fgsm10_rs
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --epsilon 9 --train_fgsm_alpha 9 --eval_pgd_alpha 2.25 --train_random_start --exp_name 1030_anisotropy_fgsm9_rs
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --epsilon 8 --train_fgsm_alpha 8 --eval_pgd_alpha 2 --train_random_start --exp_name 1030_anisotropy_fgsm8_rs
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --epsilon 7 --train_fgsm_alpha 7 --eval_pgd_alpha 1.75 --train_random_start --exp_name 1030_anisotropy_fgsm7_rs
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --epsilon 6 --train_fgsm_alpha 6 --eval_pgd_alpha 1.5 --train_random_start --exp_name 1030_anisotropy_fgsm6_rs
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --epsilon 5 --train_fgsm_alpha 5 --eval_pgd_alpha 1.25 --train_random_start --exp_name 1030_anisotropy_fgsm5_rs
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
##bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
##module load $PCOMMAND
##python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --epsilon 4 --train_fgsm_alpha 4 --eval_pgd_alpha 1 --train_random_start --exp_name 1030_anisotropy_fgsm4_rs
##ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --epsilon 3 --train_fgsm_alpha 3 --eval_pgd_alpha 0.75 --train_random_start --exp_name 1030_anisotropy_fgsm3_rs
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --epsilon 2 --train_fgsm_alpha 2 --eval_pgd_alpha 0.5 --train_random_start --exp_name 1030_anisotropy_fgsm2_rs
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --epsilon 1 --train_fgsm_alpha 1 --eval_pgd_alpha 0.25 --train_random_start --exp_name 1030_anisotropy_fgsm1_rs
#ENDBSUB

##################cyclic learning rate########################3
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 8 --train_fgsm_alpha 8 --eval_pgd_alpha 2 --exp_name 1030_standard_cyclic_fgsm8
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 1 --train_fgsm_alpha 1 --eval_pgd_alpha 0.25 --exp_name 1030_standard_cyclic_fgsm1
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 2 --train_fgsm_alpha 2 --eval_pgd_alpha 0.5 --exp_name 1030_standard_cyclic_fgsm2
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 3 --train_fgsm_alpha 3 --eval_pgd_alpha 0.75 --exp_name 1030_standard_cyclic_fgsm3
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 4 --train_fgsm_alpha 4 --eval_pgd_alpha 1 --exp_name 1030_standard_cyclic_fgsm4
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 5 --train_fgsm_alpha 5 --eval_pgd_alpha 1.25 --exp_name 1030_standard_cyclic_fgsm5
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 6 --train_fgsm_alpha 6 --eval_pgd_alpha 1.5 --exp_name 1030_standard_cyclic_fgsm6
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 7 --train_fgsm_alpha 7 --eval_pgd_alpha 1.75 --exp_name 1030_standard_cyclic_fgsm7
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 9 --train_fgsm_alpha 9 --eval_pgd_alpha 2.25 --exp_name 1030_standard_cyclic_fgsm9
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 10 --train_fgsm_alpha 10 --eval_pgd_alpha 2.5 --exp_name 1030_standard_cyclic_fgsm10
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 11 --train_fgsm_alpha 11 --eval_pgd_alpha 2.75 --exp_name 1030_standard_cyclic_fgsm11
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 12 --train_fgsm_alpha 12 --eval_pgd_alpha 3 --exp_name 1030_standard_cyclic_fgsm12
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 13 --train_fgsm_alpha 13 --eval_pgd_alpha 3.25 --exp_name 1030_standard_cyclic_fgsm13
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 14 --train_fgsm_alpha 14 --eval_pgd_alpha 3.5 --exp_name 1030_standard_cyclic_fgsm14
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 15 --train_fgsm_alpha 15 --eval_pgd_alpha 3.75 --exp_name 1030_standard_cyclic_fgsm15
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 16 --train_fgsm_alpha 16 --eval_pgd_alpha 4 --exp_name 1030_standard_cyclic_fgsm16
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 8 --train_fgsm_alpha 1 --eval_pgd_alpha 2 --train_random_start --exp_name 1030_random_cyclic_fgsm8_alpha1
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 8 --train_fgsm_alpha 2 --eval_pgd_alpha 2 --train_random_start --exp_name 1030_random_cyclic_fgsm8_alpha2
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 8 --train_fgsm_alpha 3 --eval_pgd_alpha 2 --train_random_start --exp_name 1030_random_cyclic_fgsm8_alpha3
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 8 --train_fgsm_alpha 4 --eval_pgd_alpha 2 --train_random_start --exp_name 1030_random_cyclic_fgsm8_alpha4
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 8 --train_fgsm_alpha 5 --eval_pgd_alpha 2 --train_random_start --exp_name 1030_random_cyclic_fgsm8_alpha5
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 8 --train_fgsm_alpha 6 --eval_pgd_alpha 2 --train_random_start --exp_name 1030_random_cyclic_fgsm8_alpha6
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 8 --train_fgsm_alpha 7 --eval_pgd_alpha 2 --train_random_start --exp_name 1030_random_cyclic_fgsm8_alpha7
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 8 --train_fgsm_alpha 8 --eval_pgd_alpha 2 --train_random_start --exp_name 1030_random_cyclic_fgsm8_alpha8
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 8 --train_fgsm_alpha 9 --eval_pgd_alpha 2 --train_random_start --exp_name 1030_random_cyclic_fgsm8_alpha9
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 8 --train_fgsm_alpha 10 --eval_pgd_alpha 2 --train_random_start --exp_name 1030_random_cyclic_fgsm8_alpha10
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 8 --train_fgsm_alpha 11 --eval_pgd_alpha 2 --train_random_start --exp_name 1030_random_cyclic_fgsm8_alpha11
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 8 --train_fgsm_alpha 12 --eval_pgd_alpha 2 --train_random_start --exp_name 1030_random_cyclic_fgsm8_alpha12
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 8 --train_fgsm_alpha 13 --eval_pgd_alpha 2 --train_random_start --exp_name 1030_random_cyclic_fgsm8_alpha13
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 8 --train_fgsm_alpha 14 --eval_pgd_alpha 2 --train_random_start --exp_name 1030_random_cyclic_fgsm8_alpha14
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 8 --train_fgsm_alpha 15 --eval_pgd_alpha 2 --train_random_start --exp_name 1030_random_cyclic_fgsm8_alpha15
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 8 --train_fgsm_alpha 16 --eval_pgd_alpha 2 --train_random_start --exp_name 1030_random_cyclic_fgsm8_alpha16
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 1 --train_fgsm_alpha 1 --eval_pgd_alpha 0.25 --train_random_start --exp_name 1030_random_cyclic_1_fgsm1
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 2 --train_fgsm_alpha 2 --eval_pgd_alpha 0.5 --train_random_start --exp_name 1030_random_cyclic_1_fgsm2
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 3 --train_fgsm_alpha 3 --eval_pgd_alpha 0.75 --train_random_start --exp_name 1030_random_cyclic_1_fgsm3
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 4 --train_fgsm_alpha 4 --eval_pgd_alpha 1 --train_random_start --exp_name 1030_random_cyclic_1_fgsm4
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 5 --train_fgsm_alpha 5 --eval_pgd_alpha 1.25 --train_random_start --exp_name 1030_random_cyclic_1_fgsm5
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 6 --train_fgsm_alpha 6 --eval_pgd_alpha 1.5 --train_random_start --exp_name 1030_random_cyclic_1_fgsm6
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 7 --train_fgsm_alpha 7 --eval_pgd_alpha 1.75 --train_random_start --exp_name 1030_random_cyclic_1_fgsm7
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 8 --train_fgsm_alpha 8 --eval_pgd_alpha 2 --train_random_start --exp_name 1030_random_cyclic_1_fgsm8
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 9 --train_fgsm_alpha 9 --eval_pgd_alpha 2.25 --train_random_start --exp_name 1030_random_cyclic_1_fgsm9
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 10 --train_fgsm_alpha 10 --eval_pgd_alpha 2.5 --train_random_start --exp_name 1030_random_cyclic_1_fgsm10
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 11 --train_fgsm_alpha 11 --eval_pgd_alpha 2.75 --train_random_start --exp_name 1030_random_cyclic_1_fgsm11
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 12 --train_fgsm_alpha 12 --eval_pgd_alpha 3 --train_random_start --exp_name 1030_random_cyclic_1_fgsm12
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 13 --train_fgsm_alpha 13 --eval_pgd_alpha 3.25 --train_random_start --exp_name 1030_random_cyclic_1_fgsm13
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 14 --train_fgsm_alpha 14 --eval_pgd_alpha 3.5 --train_random_start --exp_name 1030_random_cyclic_1_fgsm14
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 15 --train_fgsm_alpha 15 --eval_pgd_alpha 3.75 --train_random_start --exp_name 1030_random_cyclic_1_fgsm15
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 16 --train_fgsm_alpha 16 --eval_pgd_alpha 4 --train_random_start --exp_name 1030_random_cyclic_1_fgsm16
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 14:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 1 --train_fgsm_alpha 1.25 --eval_pgd_alpha 0.25 --train_random_start --exp_name 1030_random_cyclic_1.25_fgsm1
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 14:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 2 --train_fgsm_alpha 2.5 --eval_pgd_alpha 0.5 --train_random_start --exp_name 1030_random_cyclic_1.25_fgsm2
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 14:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 3 --train_fgsm_alpha 3.75 --eval_pgd_alpha 0.75 --train_random_start --exp_name 1030_random_cyclic_1.25_fgsm3
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 14:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 4 --train_fgsm_alpha 5 --eval_pgd_alpha 1 --train_random_start --exp_name 1030_random_cyclic_1.25_fgsm4
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 14:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 5 --train_fgsm_alpha 6.25 --eval_pgd_alpha 1.25 --train_random_start --exp_name 1030_random_cyclic_1.25_fgsm5
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 14:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 6 --train_fgsm_alpha 7.5 --eval_pgd_alpha 1.5 --train_random_start --exp_name 1030_random_cyclic_1.25_fgsm6
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 14:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 7 --train_fgsm_alpha 8.75 --eval_pgd_alpha 1.75 --train_random_start --exp_name 1030_random_cyclic_1.25_fgsm7
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 14:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 8 --train_fgsm_alpha 10 --eval_pgd_alpha 2 --train_random_start --exp_name 1030_random_cyclic_1.25_fgsm8
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 14:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 9 --train_fgsm_alpha 11.25 --eval_pgd_alpha 2.25 --train_random_start --exp_name 1030_random_cyclic_1.25_fgsm9
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 14:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 10 --train_fgsm_alpha 12.5 --eval_pgd_alpha 2.5 --train_random_start --exp_name 1030_random_cyclic_1.25_fgsm10
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 14:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 11 --train_fgsm_alpha 13.75 --eval_pgd_alpha 2.75 --train_random_start --exp_name 1030_random_cyclic_1.25_fgsm11
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 14:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 12 --train_fgsm_alpha 15 --eval_pgd_alpha 3 --train_random_start --exp_name 1030_random_cyclic_1.25_fgsm12
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 14:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 13 --train_fgsm_alpha 16.25 --eval_pgd_alpha 3.25 --train_random_start --exp_name 1030_random_cyclic_1.25_fgsm13
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 14:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 14 --train_fgsm_alpha 17.5 --eval_pgd_alpha 3.5 --train_random_start --exp_name 1030_random_cyclic_1.25_fgsm14
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 14:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 15 --train_fgsm_alpha 18.75 --eval_pgd_alpha 3.75 --train_random_start --exp_name 1030_random_cyclic_1.25_fgsm15
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 14:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 16 --train_fgsm_alpha 20 --eval_pgd_alpha 4 --train_random_start --exp_name 1030_random_cyclic_1.25_fgsm16
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 1 --train_fgsm_alpha 1 --eval_pgd_alpha 0.25 --exp_name 1030_anisotropy_cyclic_fgsm1
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 2 --train_fgsm_alpha 2 --eval_pgd_alpha 0.5 --exp_name 1030_anisotropy_cyclic_fgsm2
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 3 --train_fgsm_alpha 3 --eval_pgd_alpha 0.75 --exp_name 1030_anisotropy_cyclic_fgsm3
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 4 --train_fgsm_alpha 4 --eval_pgd_alpha 1 --exp_name 1030_anisotropy_cyclic_fgsm4
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 5 --train_fgsm_alpha 5 --eval_pgd_alpha 1.25 --exp_name 1030_anisotropy_cyclic_fgsm5
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 6 --train_fgsm_alpha 6 --eval_pgd_alpha 1.5 --exp_name 1030_anisotropy_cyclic_fgsm6
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 7 --train_fgsm_alpha 7 --eval_pgd_alpha 1.75 --exp_name 1030_anisotropy_cyclic_fgsm7
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 8 --train_fgsm_alpha 8 --eval_pgd_alpha 2 --exp_name 1030_anisotropy_cyclic_fgsm8
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 9 --train_fgsm_alpha 9 --eval_pgd_alpha 2.25 --exp_name 1030_anisotropy_cyclic_fgsm9
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 10 --train_fgsm_alpha 10 --eval_pgd_alpha 2.5 --exp_name 1030_anisotropy_cyclic_fgsm10
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 11 --train_fgsm_alpha 11 --eval_pgd_alpha 2.75 --exp_name 1030_anisotropy_cyclic_fgsm11
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 12 --train_fgsm_alpha 12 --eval_pgd_alpha 3 --exp_name 1030_anisotropy_cyclic_fgsm12
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 13 --train_fgsm_alpha 13 --eval_pgd_alpha 3.25 --exp_name 1030_anisotropy_cyclic_fgsm13
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 14 --train_fgsm_alpha 14 --eval_pgd_alpha 3.5 --exp_name 1030_anisotropy_cyclic_fgsm14
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 15 --train_fgsm_alpha 15 --eval_pgd_alpha 3.75 --exp_name 1030_anisotropy_cyclic_fgsm15
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 16 --train_fgsm_alpha 16 --eval_pgd_alpha 4 --exp_name 1030_anisotropy_cyclic_fgsm16
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 1 --train_fgsm_alpha 1 --eval_pgd_alpha 0.25 --train_random_start --exp_name 1030_anisotropy_cyclic_fgsm1_rs
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 2 --train_fgsm_alpha 2 --eval_pgd_alpha 0.5 --train_random_start --exp_name 1030_anisotropy_cyclic_fgsm2_rs
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 3 --train_fgsm_alpha 3 --eval_pgd_alpha 0.75 --train_random_start --exp_name 1030_anisotropy_cyclic_fgsm3_rs
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 4 --train_fgsm_alpha 4 --eval_pgd_alpha 1 --train_random_start --exp_name 1030_anisotropy_cyclic_fgsm4_rs
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 5 --train_fgsm_alpha 5 --eval_pgd_alpha 1.25 --train_random_start --exp_name 1030_anisotropy_cyclic_fgsm5_rs
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 6 --train_fgsm_alpha 6 --eval_pgd_alpha 1.5 --train_random_start --exp_name 1030_anisotropy_cyclic_fgsm6_rs
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 7 --train_fgsm_alpha 7 --eval_pgd_alpha 1.75 --train_random_start --exp_name 1030_anisotropy_cyclic_fgsm7_rs
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 8 --train_fgsm_alpha 8 --eval_pgd_alpha 2 --train_random_start --exp_name 1030_anisotropy_cyclic_fgsm8_rs
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 9 --train_fgsm_alpha 9 --eval_pgd_alpha 2.25 --train_random_start --exp_name 1030_anisotropy_cyclic_fgsm9_rs
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 10 --train_fgsm_alpha 10 --eval_pgd_alpha 2.5 --train_random_start --exp_name 1030_anisotropy_cyclic_fgsm10_rs
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 11 --train_fgsm_alpha 11 --eval_pgd_alpha 2.75 --train_random_start --exp_name 1030_anisotropy_cyclic_fgsm11_rs
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 12 --train_fgsm_alpha 12 --eval_pgd_alpha 3 --train_random_start --exp_name 1030_anisotropy_cyclic_fgsm12_rs
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 13 --train_fgsm_alpha 13 --eval_pgd_alpha 3.25 --train_random_start --exp_name 1030_anisotropy_cyclic_fgsm13_rs
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 14 --train_fgsm_alpha 14 --eval_pgd_alpha 3.5 --train_random_start --exp_name 1030_anisotropy_cyclic_fgsm14_rs
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 15 --train_fgsm_alpha 15 --eval_pgd_alpha 3.75 --train_random_start --exp_name 1030_anisotropy_cyclic_fgsm15_rs
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 16 --train_fgsm_alpha 16 --eval_pgd_alpha 4 --train_random_start --exp_name 1030_anisotropy_cyclic_fgsm16_rs
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 1 --train_fgsm_alpha 1.25 --eval_pgd_alpha 0.25 --train_random_start --exp_name 1030_anisotropy_cyclic_fgsm1_rs1.25
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 2 --train_fgsm_alpha 2.5 --eval_pgd_alpha 0.5 --train_random_start --exp_name 1030_anisotropy_cyclic_fgsm2_rs1.25
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 3 --train_fgsm_alpha 3.75 --eval_pgd_alpha 0.75 --train_random_start --exp_name 1030_anisotropy_cyclic_fgsm3_rs1.25
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 4 --train_fgsm_alpha 5 --eval_pgd_alpha 1 --train_random_start --exp_name 1030_anisotropy_cyclic_fgsm4_rs1.25
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 5 --train_fgsm_alpha 6.25 --eval_pgd_alpha 1.25 --train_random_start --exp_name 1030_anisotropy_cyclic_fgsm5_rs1.25
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 6 --train_fgsm_alpha 7.5 --eval_pgd_alpha 1.5 --train_random_start --exp_name 1030_anisotropy_cyclic_fgsm6_rs1.25
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 7 --train_fgsm_alpha 8.75 --eval_pgd_alpha 1.75 --train_random_start --exp_name 1030_anisotropy_cyclic_fgsm7_rs1.25
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 8 --train_fgsm_alpha 10 --eval_pgd_alpha 2 --train_random_start --exp_name 1030_anisotropy_cyclic_fgsm8_rs1.25
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 9 --train_fgsm_alpha 11.25 --eval_pgd_alpha 2.25 --train_random_start --exp_name 1030_anisotropy_cyclic_fgsm9_rs1.25
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 10 --train_fgsm_alpha 12.5 --eval_pgd_alpha 2.5 --train_random_start --exp_name 1030_anisotropy_cyclic_fgsm10_rs1.25
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 11 --train_fgsm_alpha 13.75 --eval_pgd_alpha 2.75 --train_random_start --exp_name 1030_anisotropy_cyclic_fgsm11_rs1.25
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 12 --train_fgsm_alpha 15 --eval_pgd_alpha 3 --train_random_start --exp_name 1030_anisotropy_cyclic_fgsm12_rs1.25
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 13 --train_fgsm_alpha 16.25 --eval_pgd_alpha 3.25 --train_random_start --exp_name 1030_anisotropy_cyclic_fgsm13_rs1.25
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 14 --train_fgsm_alpha 17.5 --eval_pgd_alpha 3.5 --train_random_start --exp_name 1030_anisotropy_cyclic_fgsm14_rs1.25
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 15 --train_fgsm_alpha 18.75 --eval_pgd_alpha 3.75 --train_random_start --exp_name 1030_anisotropy_cyclic_fgsm15_rs1.25
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 16 --train_fgsm_alpha 20 --eval_pgd_alpha 4 --train_random_start --exp_name 1030_anisotropy_cyclic_fgsm16_rs1.25
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 14:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 1 --train_fgsm_alpha 0.75 --eval_pgd_alpha 0.25 --train_random_start --exp_name 1030_random_cyclic_0.75_fgsm1
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 14:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 2 --train_fgsm_alpha 1.5 --eval_pgd_alpha 0.5 --train_random_start --exp_name 1030_random_cyclic_0.75_fgsm2
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 14:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 3 --train_fgsm_alpha 2.25 --eval_pgd_alpha 0.75 --train_random_start --exp_name 1030_random_cyclic_0.75_fgsm3
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 14:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 4 --train_fgsm_alpha 3 --eval_pgd_alpha 1 --train_random_start --exp_name 1030_random_cyclic_0.75_fgsm4
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 14:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 5 --train_fgsm_alpha 3.75 --eval_pgd_alpha 1.25 --train_random_start --exp_name 1030_random_cyclic_0.75_fgsm5
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 14:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 6 --train_fgsm_alpha 4.5 --eval_pgd_alpha 1.5 --train_random_start --exp_name 1030_random_cyclic_0.75_fgsm6
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 14:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 7 --train_fgsm_alpha 5.25 --eval_pgd_alpha 1.75 --train_random_start --exp_name 1030_random_cyclic_0.75_fgsm7
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 14:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 8 --train_fgsm_alpha 6 --eval_pgd_alpha 2 --train_random_start --exp_name 1030_random_cyclic_0.75_fgsm8
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 14:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 9 --train_fgsm_alpha 6.75 --eval_pgd_alpha 2.25 --train_random_start --exp_name 1030_random_cyclic_0.75_fgsm9
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 14:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 10 --train_fgsm_alpha 7.5 --eval_pgd_alpha 2.5 --train_random_start --exp_name 1030_random_cyclic_0.75_fgsm10
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 14:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 11 --train_fgsm_alpha 8.25 --eval_pgd_alpha 2.75 --train_random_start --exp_name 1030_random_cyclic_0.75_fgsm11
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 14:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 12 --train_fgsm_alpha 9 --eval_pgd_alpha 3 --train_random_start --exp_name 1030_random_cyclic_0.75_fgsm12
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 14:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 13 --train_fgsm_alpha 9.75 --eval_pgd_alpha 3.25 --train_random_start --exp_name 1030_random_cyclic_0.75_fgsm13
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 14:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 14 --train_fgsm_alpha 10.5 --eval_pgd_alpha 3.5 --train_random_start --exp_name 1030_random_cyclic_0.75_fgsm14
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 14:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 15 --train_fgsm_alpha 11.25 --eval_pgd_alpha 3.75 --train_random_start --exp_name 1030_random_cyclic_0.75_fgsm15
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 14:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 16 --train_fgsm_alpha 12 --eval_pgd_alpha 4 --train_random_start --exp_name 1030_random_cyclic_0.75_fgsm16
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 1 --train_fgsm_alpha 1.5 --eval_pgd_alpha 0.25 --train_random_start --exp_name 1030_anisotropy_cyclic_fgsm1_rs1.5
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 2 --train_fgsm_alpha 3 --eval_pgd_alpha 0.5 --train_random_start --exp_name 1030_anisotropy_cyclic_fgsm2_rs1.5
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 3 --train_fgsm_alpha 4.5 --eval_pgd_alpha 0.75 --train_random_start --exp_name 1030_anisotropy_cyclic_fgsm3_rs1.5
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 4 --train_fgsm_alpha 6 --eval_pgd_alpha 1 --train_random_start --exp_name 1030_anisotropy_cyclic_fgsm4_rs1.5
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 5 --train_fgsm_alpha 7.5 --eval_pgd_alpha 1.25 --train_random_start --exp_name 1030_anisotropy_cyclic_fgsm5_rs1.5
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 6 --train_fgsm_alpha 9 --eval_pgd_alpha 1.5 --train_random_start --exp_name 1030_anisotropy_cyclic_fgsm6_rs1.5
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 7 --train_fgsm_alpha 10.5 --eval_pgd_alpha 1.75 --train_random_start --exp_name 1030_anisotropy_cyclic_fgsm7_rs1.5
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 8 --train_fgsm_alpha 12 --eval_pgd_alpha 2 --train_random_start --exp_name 1030_anisotropy_cyclic_fgsm8_rs1.5
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 9 --train_fgsm_alpha 13.5 --eval_pgd_alpha 2.25 --train_random_start --exp_name 1030_anisotropy_cyclic_fgsm9_rs1.5
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 10 --train_fgsm_alpha 15 --eval_pgd_alpha 2.5 --train_random_start --exp_name 1030_anisotropy_cyclic_fgsm10_rs1.5
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 11 --train_fgsm_alpha 16.5 --eval_pgd_alpha 2.75 --train_random_start --exp_name 1030_anisotropy_cyclic_fgsm11_rs1.5
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 12 --train_fgsm_alpha 18 --eval_pgd_alpha 3 --train_random_start --exp_name 1030_anisotropy_cyclic_fgsm12_rs1.5
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 13 --train_fgsm_alpha 19.5 --eval_pgd_alpha 3.25 --train_random_start --exp_name 1030_anisotropy_cyclic_fgsm13_rs1.5
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 14 --train_fgsm_alpha 21 --eval_pgd_alpha 3.5 --train_random_start --exp_name 1030_anisotropy_cyclic_fgsm14_rs1.5
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 15 --train_fgsm_alpha 22.5 --eval_pgd_alpha 3.75 --train_random_start --exp_name 1030_anisotropy_cyclic_fgsm15_rs1.5
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 16 --train_fgsm_alpha 24 --eval_pgd_alpha 4 --train_random_start --exp_name 1030_anisotropy_cyclic_fgsm16_rs1.5
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 1 --train_fgsm_alpha 2 --eval_pgd_alpha 0.25 --train_random_start --exp_name 1030_anisotropy_cyclic_fgsm1_rs2
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 2 --train_fgsm_alpha 4 --eval_pgd_alpha 0.5 --train_random_start --exp_name 1030_anisotropy_cyclic_fgsm2_rs2
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 3 --train_fgsm_alpha 6 --eval_pgd_alpha 0.75 --train_random_start --exp_name 1030_anisotropy_cyclic_fgsm3_rs2
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 4 --train_fgsm_alpha 8 --eval_pgd_alpha 1 --train_random_start --exp_name 1030_anisotropy_cyclic_fgsm4_rs2
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 5 --train_fgsm_alpha 10 --eval_pgd_alpha 1.25 --train_random_start --exp_name 1030_anisotropy_cyclic_fgsm5_rs2
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 6 --train_fgsm_alpha 12 --eval_pgd_alpha 1.5 --train_random_start --exp_name 1030_anisotropy_cyclic_fgsm6_rs2
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 7 --train_fgsm_alpha 14 --eval_pgd_alpha 1.75 --train_random_start --exp_name 1030_anisotropy_cyclic_fgsm7_rs2
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 8 --train_fgsm_alpha 16 --eval_pgd_alpha 2 --train_random_start --exp_name 1030_anisotropy_cyclic_fgsm8_rs2
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 9 --train_fgsm_alpha 18 --eval_pgd_alpha 2.25 --train_random_start --exp_name 1030_anisotropy_cyclic_fgsm9_rs2
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 10 --train_fgsm_alpha 20 --eval_pgd_alpha 2.5 --train_random_start --exp_name 1030_anisotropy_cyclic_fgsm10_rs2
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 11 --train_fgsm_alpha 22 --eval_pgd_alpha 2.75 --train_random_start --exp_name 1030_anisotropy_cyclic_fgsm11_rs2
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 12 --train_fgsm_alpha 24 --eval_pgd_alpha 3 --train_random_start --exp_name 1030_anisotropy_cyclic_fgsm12_rs2
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 13 --train_fgsm_alpha 26 --eval_pgd_alpha 3.25 --train_random_start --exp_name 1030_anisotropy_cyclic_fgsm13_rs2
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 14 --train_fgsm_alpha 28 --eval_pgd_alpha 3.5 --train_random_start --exp_name 1030_anisotropy_cyclic_fgsm14_rs2
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 15 --train_fgsm_alpha 30 --eval_pgd_alpha 3.75 --train_random_start --exp_name 1030_anisotropy_cyclic_fgsm15_rs2
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 30 --batch_size 256 --lr_schedule cyclic --epsilon 16 --train_fgsm_alpha 32 --eval_pgd_alpha 4 --train_random_start --exp_name 1030_anisotropy_cyclic_fgsm16_rs2
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm_test.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --lr_schedule multistep --exp_name 1031_fgsm_test
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm_test.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --lr_schedule multistep --extend_ratio 2.25 --exp_name 1031_fgsm_test_4_2.25
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm_test.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --lr_schedule multistep --train_fgsm_alpha 6 --extend_ratio 2 --exp_name 1031_fgsm_test_6_2
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm_test.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --lr_schedule multistep --train_fgsm_alpha 6 --extend_ratio 2.25 --exp_name 1031_fgsm_test_6_2.25
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --lr_schedule cyclic --num_epochs 30 --batch_size 256 --train_fgsm_alpha 8 --exp_name 1105_fgsm8
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --lr_schedule cyclic --num_epochs 30 --batch_size 256 --epsilon 8 --train_fgsm_alpha 14 --train_random_start --exp_name 1105_fgsm8_14
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --lr_schedule cyclic --num_epochs 30 --batch_size 256 --epsilon 8 --train_fgsm_alpha 11 --train_random_start --exp_name 1105_fgsm8_11
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --lr_schedule cyclic --num_epochs 30 --batch_size 256 --epsilon 8 --train_fgsm_alpha 12 --train_random_start --exp_name 1105_fgsm8_12
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --lr_schedule cyclic --num_epochs 30 --batch_size 256 --epsilon 8 --train_fgsm_alpha 16 --train_random_start --exp_name 1105_fgsm8_16
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm_test.py --model PreActResNet18 --lr_schedule cyclic --num_epochs 30 --batch_size 256 --epsilon 8 --failed_fgsm_alpha 14 --train_fgsm_alpha 6 --extend_ratio 3 --train_random_start --exp_name 1105_fgsm8_6times3
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm_test.py --model PreActResNet18 --lr_schedule cyclic --num_epochs 30 --batch_size 256 --epsilon 8 --failed_fgsm_alpha 14 --train_fgsm_alpha 6 --extend_ratio 4 --train_random_start --exp_name 1105_fgsm8_6times4
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm_test.py --model PreActResNet18 --lr_schedule cyclic --num_epochs 30 --batch_size 256 --epsilon 8 --failed_fgsm_alpha 11 --train_fgsm_alpha 4 --extend_ratio 3 --train_random_start --exp_name 1105_fgsm8_4times3
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm_test.py --model PreActResNet18 --lr_schedule cyclic --num_epochs 30 --batch_size 256 --epsilon 8 --failed_fgsm_alpha 11 --train_fgsm_alpha 4 --extend_ratio 4 --train_random_start --exp_name 1105_fgsm8_4times4
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm_test.py --model PreActResNet18 --lr_schedule cyclic --num_epochs 30 --batch_size 256 --epsilon 8 --failed_fgsm_alpha 11 --train_fgsm_alpha 3 --extend_ratio 5 --train_random_start --exp_name 1105_fgsm8_3times5
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm_test.py --model PreActResNet18 --lr_schedule cyclic --num_epochs 30 --batch_size 256 --epsilon 8 --failed_fgsm_alpha 11 --train_fgsm_alpha 3 --extend_ratio 4 --train_random_start --exp_name 1105_fgsm8_3times4
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm_test.py --model PreActResNet18 --lr_schedule cyclic --num_epochs 30 --batch_size 256 --epsilon 8 --failed_fgsm_alpha 11 --train_fgsm_alpha 3 --extend_ratio 4.5 --train_random_start --exp_name 1105_fgsm8_3times4.5
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm_test.py --model PreActResNet18 --lr_schedule cyclic --num_epochs 30 --batch_size 256 --epsilon 8 --failed_fgsm_alpha 11 --train_fgsm_alpha 3 --extend_ratio 6 --train_random_start --exp_name 1105_fgsm8_3times6
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm_test.py --model PreActResNet18 --lr_schedule cyclic --num_epochs 30 --batch_size 256 --epsilon 8 --failed_fgsm_alpha 11 --train_fgsm_alpha 2 --extend_ratio 8 --train_random_start --exp_name 1105_fgsm8_2times8
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm_test.py --model PreActResNet18 --lr_schedule cyclic --num_epochs 30 --batch_size 256 --epsilon 8 --failed_fgsm_alpha 11 --train_fgsm_alpha 2 --extend_ratio 6 --train_random_start --exp_name 1105_fgsm8_2times6
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm_test.py --model PreActResNet18 --lr_schedule cyclic --num_epochs 30 --batch_size 256 --epsilon 8 --failed_fgsm_alpha 11 --train_fgsm_alpha 4 --extend_ratio 3.5 --train_random_start --exp_name 1105_fgsm8_4times3.5
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm_test.py --model PreActResNet18 --lr_schedule cyclic --num_epochs 30 --batch_size 256 --epsilon 8 --failed_fgsm_alpha 11 --train_fgsm_alpha 3.5 --extend_ratio 4 --train_random_start --exp_name 1105_fgsm8_3.5times4
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm_test.py --model PreActResNet18 --lr_schedule cyclic --num_epochs 30 --batch_size 256 --epsilon 8 --failed_fgsm_alpha 11 --train_fgsm_alpha 5 --extend_ratio 3 --train_random_start --exp_name 1105_fgsm8_5times3
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm_test.py --model PreActResNet18 --lr_schedule cyclic --num_epochs 30 --batch_size 256 --epsilon 8 --failed_fgsm_alpha 11 --train_fgsm_alpha 5 --extend_ratio 3.5 --train_random_start --exp_name 1105_fgsm8_5times3.5
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm_test.py --model PreActResNet18 --lr_schedule cyclic --num_epochs 30 --batch_size 256 --epsilon 8 --failed_fgsm_alpha 10 --train_fgsm_alpha 8 --extend_ratio 1.5 --train_random_start --exp_name 1105_fgsm8_8times1.5
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm_test.py --model PreActResNet18 --lr_schedule cyclic --num_epochs 30 --batch_size 256 --epsilon 8 --failed_fgsm_alpha 10 --train_fgsm_alpha 8 --extend_ratio 2 --train_random_start --exp_name 1105_fgsm8_8times2
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm_test.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --lr_schedule multistep --epsilon 8 --failed_fgsm_alpha 10 --train_fgsm_alpha 3.5 --extend_ratio 3 --train_random_start --exp_name 1105_fgsm_test_3.5_3
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm_test.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --lr_schedule multistep --epsilon 8 --failed_fgsm_alpha 10 --train_fgsm_alpha 3.5 --extend_ratio 3.5 --train_random_start --exp_name 1105_fgsm_test_3.5_3.5
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm_test.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --lr_schedule multistep --epsilon 8 --failed_fgsm_alpha 10 --train_fgsm_alpha 3.5 --extend_ratio 4 --train_random_start --exp_name 1105_fgsm_test_3.5_4
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm_test.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --lr_schedule multistep --epsilon 8 --failed_fgsm_alpha 10 --train_fgsm_alpha 3 --extend_ratio 3 --train_random_start --exp_name 1105_fgsm_test_3_3
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm_test.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --lr_schedule multistep --epsilon 8 --failed_fgsm_alpha 10 --train_fgsm_alpha 3 --extend_ratio 3.5 --train_random_start --exp_name 1105_fgsm_test_3_3.5
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm_test.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --lr_schedule multistep --epsilon 8 --failed_fgsm_alpha 10 --train_fgsm_alpha 3 --extend_ratio 4 --train_random_start --exp_name 1105_fgsm_test_3_4
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm_test.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --lr_schedule multistep --epsilon 8 --failed_fgsm_alpha 10 --train_fgsm_alpha 3 --extend_ratio 4.5 --train_random_start --exp_name 1105_fgsm_test_3_4.5
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm_test.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --lr_schedule multistep --epsilon 8 --failed_fgsm_alpha 10 --train_fgsm_alpha 3 --extend_ratio 5 --train_random_start --exp_name 1105_fgsm_test_3_5
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm_test.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --lr_schedule multistep --epsilon 8 --failed_fgsm_alpha 10 --train_fgsm_alpha 3 --extend_ratio 6 --train_random_start --exp_name 1105_fgsm_test_3_6
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm_test.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --lr_schedule multistep --epsilon 8 --failed_fgsm_alpha 10 --train_fgsm_alpha 2 --extend_ratio 6 --train_random_start --exp_name 1105_fgsm_test_2_6
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm_test.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --lr_schedule multistep --epsilon 8 --failed_fgsm_alpha 10 --train_fgsm_alpha 2 --extend_ratio 8 --train_random_start --exp_name 1105_fgsm_test_2_8
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm_test.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --lr_schedule multistep --epsilon 8 --failed_fgsm_alpha 10 --train_fgsm_alpha 2 --extend_ratio 5 --train_random_start --exp_name 1105_fgsm_test_2_5
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm_test.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --lr_schedule multistep --epsilon 8 --failed_fgsm_alpha 10 --train_fgsm_alpha 2 --extend_ratio 7 --train_random_start --exp_name 1105_fgsm_test_2_7
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm_test.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --lr_schedule multistep --epsilon 8 --failed_fgsm_alpha 10 --train_fgsm_alpha 3.5 --extend_ratio 3.5 --train_random_start --exp_name 1105_fgsm_test_3.5_3.5_exp1
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm_test.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --lr_schedule multistep --epsilon 8 --failed_fgsm_alpha 10 --train_fgsm_alpha 3.5 --extend_ratio 3.5 --train_random_start --eval_pgd_attack_iters 50 --eval_pgd_restarts 10 --exp_name 1105_fgsm_test_3.5_3.5_exp2
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm_test.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --lr_schedule multistep --epsilon 8 --failed_fgsm_alpha 10 --train_fgsm_alpha 3.5 --extend_ratio 3.5 --train_random_start --eval_pgd_attack_iters 10 --eval_pgd_restarts 1 --exp_name 1105_fgsm_test_3.5_3.5_exp3
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm_test.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --lr_schedule multistep --epsilon 8 --failed_fgsm_alpha 10 --train_fgsm_alpha 6 --extend_ratio 3 --train_random_start --eval_pgd_attack_iters 10 --eval_pgd_restarts 1 --exp_name 1105_fgsm_test_6_3
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm_test.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --lr_schedule multistep --epsilon 8 --failed_fgsm_alpha 10 --train_fgsm_alpha 8 --extend_ratio 1.8 --train_random_start --eval_pgd_attack_iters 10 --eval_pgd_restarts 1 --exp_name 1105_fgsm_test_8_1.8
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm_test.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --lr_schedule multistep --epsilon 8 --failed_fgsm_alpha 10 --train_fgsm_alpha 3 --extend_ratio 3 --train_random_start --eval_pgd_attack_iters 10 --eval_pgd_restarts 1 --exp_name 1107_fgsm_test_3_3
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm_test.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --lr_schedule multistep --epsilon 8 --failed_fgsm_alpha 10 --train_fgsm_alpha 3 --extend_ratio 3 --train_random_start --eval_pgd_attack_iters 10 --eval_pgd_restarts 1 --exp_name 1107_fgsm_test_3_3_test
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 12:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm_test.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --lr_schedule multistep --epsilon 8 --failed_fgsm_alpha 10 --train_fgsm_alpha 3 --extend_ratio 3 --train_random_start --eval_pgd_attack_iters 10 --eval_pgd_restarts 1 --exp_name 1107_fgsm_test_3_3_exp1
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 12:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --lr_schedule multistep --exp_name 1107_deepfool_test (1e-9)
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 12:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --lr_schedule multistep --exp_name 1107_deepfool_l2_test (1e-9)
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 12:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --lr_schedule multistep --exp_name 1107_deepfool_l2_test_1e-4_exp1
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 12:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --lr_schedule multistep --exp_name 1107_deepfool_l2_test_1e-4_exp2
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 12:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --lr_schedule multistep --deepfool_norm_dist l_inf --exp_name 1107_deepfool_linf_test (1e-9)
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 12:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --lr_schedule multistep --deepfool_norm_dist l_inf --exp_name 1107_deepfool_linf_test_exp1 (1e-9)
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 12:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --lr_schedule multistep --deepfool_norm_dist l_inf --exp_name 1107_deepfool_linf_test_exp2 (1e-9)
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
##bsub -n 8 -W 12:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
##module load $PCOMMAND
##python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --lr_schedule multistep --deepfool_norm_dist l_inf --exp_name 1107_deepfool_linf_1e-6_exp1
##ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 12:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --lr_schedule multistep --deepfool_norm_dist l_inf --exp_name 1107_deepfool_linf_1e-5_exp1
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --lr_schedule multistep --deepfool_norm_dist l_inf --overshoot 0.04 --exp_name 1107_deepfool_linf_1e-9_0.04_exp1
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --lr_schedule multistep --deepfool_norm_dist l_inf --overshoot 0.06 --exp_name 1107_deepfool_linf_1e-9_0.06_exp1
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --lr_schedule multistep --deepfool_norm_dist l_inf --overshoot 0.1 --exp_name 1107_deepfool_linf_1e-9_0.1_exp1
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --lr_schedule multistep --deepfool_norm_dist l_inf --overshoot 0.2 --exp_name 1107_deepfool_linf_1e-9_0.2_exp1
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --lr_schedule multistep --deepfool_norm_dist l_inf --overshoot 0.3 --exp_name 1107_deepfool_linf_1e-9_0.3_exp1
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --lr_schedule multistep --deepfool_norm_dist l_inf --overshoot 0.4 --exp_name 1107_deepfool_linf_1e-9_0.4_exp1
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --lr_schedule multistep --deepfool_norm_dist l_inf --overshoot 0.5 --exp_name 1107_deepfool_linf_1e-9_0.5_exp1
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --lr_schedule multistep --deepfool_norm_dist l_inf --overshoot 1 --exp_name 1107_deepfool_linf_1e-9_1_exp1
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --lr_schedule multistep --deepfool_norm_dist l_inf --exp_name 1107_deepfool_linf_1e-9_0.02_clamp_exp1
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 12:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --lr_schedule multistep --exp_name 1108_deepfool_l2_woclamp_wors_exp1
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 12:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --lr_schedule multistep --exp_name 1108_deepfool_l2_woclamp_wors_exp2
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 12:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --lr_schedule multistep --exp_name 1108_deepfool_l2_woclamp_wors_exp3
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 12:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --lr_schedule multistep --deepfool_rs --exp_name 1108_deepfool_l2_woclamp_rs_exp1
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 12:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --lr_schedule multistep --deepfool_rs --exp_name 1108_deepfool_l2_woclamp_rs_exp2
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 12:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --lr_schedule multistep --deepfool_rs --exp_name 1108_deepfool_l2_woclamp_rs_exp3
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 12:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --lr_schedule multistep --df_clamp --exp_name 1108_deepfool_l2_clamp_wors_exp1
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 12:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --lr_schedule multistep --df_clamp --exp_name 1108_deepfool_l2_clamp_wors_exp2
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 12:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --lr_schedule multistep --df_clamp --exp_name 1108_deepfool_l2_clamp_wors_exp3
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 12:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --lr_schedule multistep --df_clamp --deepfool_rs --exp_name 1108_deepfool_l2_clamp_rs_exp1
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 12:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --lr_schedule multistep --df_clamp --deepfool_rs --exp_name 1108_deepfool_l2_clamp_rs_exp2
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 12:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --lr_schedule multistep --df_clamp --deepfool_rs --exp_name 1108_deepfool_l2_clamp_rs_exp3
#ENDBSUB
#
#
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 12:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --deepfool_norm_dist l_inf --lr_schedule multistep --exp_name 1108_deepfool_linf_woclamp_wors_exp1
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 12:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --deepfool_norm_dist l_inf --lr_schedule multistep --exp_name 1108_deepfool_linf_woclamp_wors_exp2
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 12:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --deepfool_norm_dist l_inf --lr_schedule multistep --exp_name 1108_deepfool_linf_woclamp_wors_exp3
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 12:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --deepfool_norm_dist l_inf --lr_schedule multistep --deepfool_rs --exp_name 1108_deepfool_linf_woclamp_rs_exp1
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 12:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --deepfool_norm_dist l_inf --lr_schedule multistep --deepfool_rs --exp_name 1108_deepfool_linf_woclamp_rs_exp2
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 12:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --deepfool_norm_dist l_inf --lr_schedule multistep --deepfool_rs --exp_name 1108_deepfool_linf_woclamp_rs_exp3
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 12:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --deepfool_norm_dist l_inf --lr_schedule multistep --df_clamp --exp_name 1108_deepfool_linf_clamp_wors_exp1
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 12:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --deepfool_norm_dist l_inf --lr_schedule multistep --df_clamp --exp_name 1108_deepfool_linf_clamp_wors_exp2
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 12:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --deepfool_norm_dist l_inf --lr_schedule multistep --df_clamp --exp_name 1108_deepfool_linf_clamp_wors_exp3
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 12:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --deepfool_norm_dist l_inf --lr_schedule multistep --df_clamp --deepfool_rs --exp_name 1108_deepfool_linf_clamp_rs_exp1
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 12:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --deepfool_norm_dist l_inf --lr_schedule multistep --df_clamp --deepfool_rs --exp_name 1108_deepfool_linf_clamp_rs_exp2
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 12:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --deepfool_norm_dist l_inf --lr_schedule multistep --df_clamp --deepfool_rs --exp_name 1108_deepfool_linf_clamp_rs_exp3
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 12:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --lr_schedule multistep --df_clamp --deepfool_rs --deepfool_epsilon 9 --eval_epsilon 9 --eval_pgd_alpha 2.25 --exp_name 1108_deepfool_l2_clamp_rs_9
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 12:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --lr_schedule multistep --df_clamp --deepfool_rs --deepfool_epsilon 10 --eval_epsilon 10 --eval_pgd_alpha 2.5 --exp_name 1108_deepfool_l2_clamp_rs_10
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 12:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --lr_schedule multistep --df_clamp --deepfool_rs --deepfool_epsilon 11 --eval_epsilon 11 --eval_pgd_alpha 2.75 --exp_name 1108_deepfool_l2_clamp_rs_11
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 12:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --lr_schedule multistep --df_clamp --deepfool_rs --deepfool_epsilon 12 --eval_epsilon 12 --eval_pgd_alpha 3 --exp_name 1108_deepfool_l2_clamp_rs_12
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 12:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --lr_schedule multistep --df_clamp --deepfool_rs --deepfool_epsilon 13 --eval_epsilon 13 --eval_pgd_alpha 3.25 --exp_name 1108_deepfool_l2_clamp_rs_13
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 12:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --lr_schedule multistep --df_clamp --deepfool_rs --deepfool_epsilon 14 --eval_epsilon 14 --eval_pgd_alpha 3.5 --exp_name 1108_deepfool_l2_clamp_rs_14
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 12:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --lr_schedule multistep --df_clamp --deepfool_rs --deepfool_epsilon 15 --eval_epsilon 15 --eval_pgd_alpha 3.75 --exp_name 1108_deepfool_l2_clamp_rs_15
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 12:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --lr_schedule multistep --df_clamp --deepfool_rs --deepfool_epsilon 16 --eval_epsilon 16 --eval_pgd_alpha 4 --exp_name 1108_deepfool_l2_clamp_rs_16
#ENDBSUB
#
#
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 12:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --deepfool_norm_dist l_inf --lr_schedule multistep --df_clamp --deepfool_rs --deepfool_epsilon 9 --eval_epsilon 9 --eval_pgd_alpha 2.25 --exp_name 1108_deepfool_linf_clamp_rs_9
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 12:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --deepfool_norm_dist l_inf --lr_schedule multistep --df_clamp --deepfool_rs --deepfool_epsilon 10 --eval_epsilon 10 --eval_pgd_alpha 2.5 --exp_name 1108_deepfool_linf_clamp_rs_10
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 12:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --deepfool_norm_dist l_inf --lr_schedule multistep --df_clamp --deepfool_rs --deepfool_epsilon 11 --eval_epsilon 11 --eval_pgd_alpha 2.75 --exp_name 1108_deepfool_linf_clamp_rs_11
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 12:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --deepfool_norm_dist l_inf --lr_schedule multistep --df_clamp --deepfool_rs --deepfool_epsilon 12 --eval_epsilon 12 --eval_pgd_alpha 3 --exp_name 1108_deepfool_linf_clamp_rs_12
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 12:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --deepfool_norm_dist l_inf --lr_schedule multistep --df_clamp --deepfool_rs --deepfool_epsilon 13 --eval_epsilon 13 --eval_pgd_alpha 3.25 --exp_name 1108_deepfool_linf_clamp_rs_13
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 12:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --deepfool_norm_dist l_inf --lr_schedule multistep --df_clamp --deepfool_rs --deepfool_epsilon 14 --eval_epsilon 14 --eval_pgd_alpha 3.5 --exp_name 1108_deepfool_linf_clamp_rs_14
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 12:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --deepfool_norm_dist l_inf --lr_schedule multistep --df_clamp --deepfool_rs --deepfool_epsilon 15 --eval_epsilon 15 --eval_pgd_alpha 3.75 --exp_name 1108_deepfool_linf_clamp_rs_15
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 12:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --deepfool_norm_dist l_inf --lr_schedule multistep --df_clamp --deepfool_rs --deepfool_epsilon 16 --eval_epsilon 16 --eval_pgd_alpha 4 --exp_name 1108_deepfool_linf_clamp_rs_16
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --lr_schedule multistep --df_clamp --deepfool_rs --deepfool_epsilon 8 --eval_epsilon 8 --eval_pgd_alpha 2 --overshoot 0.04 --exp_name 1108_deepfool_l2_clamp_rs_8_0.04
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --lr_schedule multistep --df_clamp --deepfool_rs --deepfool_epsilon 9 --eval_epsilon 9 --eval_pgd_alpha 2.25 --overshoot 0.04 --exp_name 1108_deepfool_l2_clamp_rs_9_0.04
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --lr_schedule multistep --df_clamp --deepfool_rs --deepfool_epsilon 8 --eval_epsilon 8 --eval_pgd_alpha 2 --overshoot 0.06 --exp_name 1108_deepfool_l2_clamp_rs_8_0.06
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --lr_schedule multistep --df_clamp --deepfool_rs --deepfool_epsilon 9 --eval_epsilon 9 --eval_pgd_alpha 2.25 --overshoot 0.06 --exp_name 1108_deepfool_l2_clamp_rs_9_0.06
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --lr_schedule multistep --df_clamp --deepfool_rs --deepfool_epsilon 8 --eval_epsilon 8 --eval_pgd_alpha 2 --overshoot 0.08 --exp_name 1108_deepfool_l2_clamp_rs_8_0.08
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --lr_schedule multistep --df_clamp --deepfool_rs --deepfool_epsilon 9 --eval_epsilon 9 --eval_pgd_alpha 2.25 --overshoot 0.08 --exp_name 1108_deepfool_l2_clamp_rs_9_0.08
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --lr_schedule multistep --df_clamp --deepfool_rs --deepfool_epsilon 8 --eval_epsilon 8 --eval_pgd_alpha 2 --overshoot 0.1 --exp_name 1108_deepfool_l2_clamp_rs_8_0.1
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --lr_schedule multistep --df_clamp --deepfool_rs --deepfool_epsilon 9 --eval_epsilon 9 --eval_pgd_alpha 2.25 --overshoot 0.1 --exp_name 1108_deepfool_l2_clamp_rs_9_0.1
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --lr_schedule multistep --df_clamp --deepfool_rs --deepfool_epsilon 8 --eval_epsilon 8 --eval_pgd_alpha 2 --overshoot 0.2 --exp_name 1108_deepfool_l2_clamp_rs_8_0.2
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --lr_schedule multistep --df_clamp --deepfool_rs --deepfool_epsilon 9 --eval_epsilon 9 --eval_pgd_alpha 2.25 --overshoot 0.2 --exp_name 1108_deepfool_l2_clamp_rs_9_0.2
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --lr_schedule multistep --df_clamp --deepfool_rs --deepfool_epsilon 8 --eval_epsilon 8 --eval_pgd_alpha 2 --overshoot 0.4 --exp_name 1108_deepfool_l2_clamp_rs_8_0.4
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --lr_schedule multistep --df_clamp --deepfool_rs --deepfool_epsilon 9 --eval_epsilon 9 --eval_pgd_alpha 2.25 --overshoot 0.4 --exp_name 1108_deepfool_l2_clamp_rs_9_0.4
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --deepfool_norm_dist l_inf --lr_schedule multistep --df_clamp --deepfool_rs --deepfool_epsilon 8 --eval_epsilon 8 --eval_pgd_alpha 2 --overshoot 0.04 --exp_name 1108_deepfool_linf_clamp_rs_8_0.04
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --deepfool_norm_dist l_inf --lr_schedule multistep --df_clamp --deepfool_rs --deepfool_epsilon 9 --eval_epsilon 9 --eval_pgd_alpha 2.25 --overshoot 0.04 --exp_name 1108_deepfool_linf_clamp_rs_9_0.04
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --deepfool_norm_dist l_inf --lr_schedule multistep --df_clamp --deepfool_rs --deepfool_epsilon 8 --eval_epsilon 8 --eval_pgd_alpha 2 --overshoot 0.06 --exp_name 1108_deepfool_linf_clamp_rs_8_0.06
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --deepfool_norm_dist l_inf --lr_schedule multistep --df_clamp --deepfool_rs --deepfool_epsilon 9 --eval_epsilon 9 --eval_pgd_alpha 2.25 --overshoot 0.06 --exp_name 1108_deepfool_linf_clamp_rs_9_0.06
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --deepfool_norm_dist l_inf --lr_schedule multistep --df_clamp --deepfool_rs --deepfool_epsilon 8 --eval_epsilon 8 --eval_pgd_alpha 2 --overshoot 0.08 --exp_name 1108_deepfool_linf_clamp_rs_8_0.08
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --deepfool_norm_dist l_inf --lr_schedule multistep --df_clamp --deepfool_rs --deepfool_epsilon 9 --eval_epsilon 9 --eval_pgd_alpha 2.25 --overshoot 0.08 --exp_name 1108_deepfool_linf_clamp_rs_9_0.08
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --deepfool_norm_dist l_inf --lr_schedule multistep --df_clamp --deepfool_rs --deepfool_epsilon 8 --eval_epsilon 8 --eval_pgd_alpha 2 --overshoot 0.1 --exp_name 1108_deepfool_linf_clamp_rs_8_0.1
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --deepfool_norm_dist l_inf --lr_schedule multistep --df_clamp --deepfool_rs --deepfool_epsilon 9 --eval_epsilon 9 --eval_pgd_alpha 2.25 --overshoot 0.1 --exp_name 1108_deepfool_linf_clamp_rs_9_0.1
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --deepfool_norm_dist l_inf --lr_schedule multistep --df_clamp --deepfool_rs --deepfool_epsilon 8 --eval_epsilon 8 --eval_pgd_alpha 2 --overshoot 0.2 --exp_name 1108_deepfool_linf_clamp_rs_8_0.2
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --deepfool_norm_dist l_inf --lr_schedule multistep --df_clamp --deepfool_rs --deepfool_epsilon 9 --eval_epsilon 9 --eval_pgd_alpha 2.25 --overshoot 0.2 --exp_name 1108_deepfool_linf_clamp_rs_9_0.2
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --deepfool_norm_dist l_inf --lr_schedule multistep --df_clamp --deepfool_rs --deepfool_epsilon 8 --eval_epsilon 8 --eval_pgd_alpha 2 --overshoot 0.4 --exp_name 1108_deepfool_linf_clamp_rs_8_0.4
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_deepfool.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --deepfool_norm_dist l_inf --lr_schedule multistep --df_clamp --deepfool_rs --deepfool_epsilon 9 --eval_epsilon 9 --eval_pgd_alpha 2.25 --overshoot 0.4 --exp_name 1108_deepfool_linf_clamp_rs_9_0.4
#ENDBSUB

#################################1109 -1113#################################
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --train_random_start --epsilon 8 --train_fgsm_ratio 0.25 --exp_name 1111_fgsm_8_0.25
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 14:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --train_random_start --epsilon 8 --train_fgsm_ratio 0.5 --exp_name 1111_fgsm_8_0.5
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --train_random_start --epsilon 8 --train_fgsm_ratio 0.75 --exp_name 1111_fgsm_8_0.75
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --train_random_start --epsilon 8 --train_fgsm_ratio 1 --exp_name 1111_fgsm_8_1
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --train_random_start --epsilon 8 --train_fgsm_ratio 1.25 --exp_name 1111_fgsm_8_1.25
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --train_random_start --epsilon 8 --train_fgsm_ratio 1.5 --exp_name 1111_fgsm_8_1.5
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --train_random_start --epsilon 8 --train_fgsm_ratio 1.75 --exp_name 1111_fgsm_8_1.75
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --train_random_start --epsilon 8 --train_fgsm_ratio 2 --exp_name 1111_fgsm_8_2
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --train_random_start --epsilon 7 --train_fgsm_ratio 0.25 --exp_name 1111_fgsm_7_0.25
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --train_random_start --epsilon 7 --train_fgsm_ratio 0.5 --exp_name 1111_fgsm_7_0.5
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --train_random_start --epsilon 7 --train_fgsm_ratio 0.75 --exp_name 1111_fgsm_7_0.75
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --train_random_start --epsilon 7 --train_fgsm_ratio 1 --exp_name 1111_fgsm_7_1
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --train_random_start --epsilon 7 --train_fgsm_ratio 1.25 --exp_name 1111_fgsm_7_1.25
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --train_random_start --epsilon 7 --train_fgsm_ratio 1.5 --exp_name 1111_fgsm_7_1.5
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --train_random_start --epsilon 7 --train_fgsm_ratio 1.75 --exp_name 1111_fgsm_7_1.75
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --train_random_start --epsilon 7 --train_fgsm_ratio 2 --exp_name 1111_fgsm_7_2
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --train_random_start --epsilon 6 --train_fgsm_ratio 0.25 --exp_name 1111_fgsm_6_0.25
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --train_random_start --epsilon 6 --train_fgsm_ratio 0.5 --exp_name 1111_fgsm_6_0.5
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --train_random_start --epsilon 6 --train_fgsm_ratio 0.75 --exp_name 1111_fgsm_6_0.75
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --train_random_start --epsilon 6 --train_fgsm_ratio 1 --exp_name 1111_fgsm_6_1
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --train_random_start --epsilon 6 --train_fgsm_ratio 1.25 --exp_name 1111_fgsm_6_1.25
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --train_random_start --epsilon 6 --train_fgsm_ratio 1.5 --exp_name 1111_fgsm_6_1.5
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --train_random_start --epsilon 6 --train_fgsm_ratio 1.75 --exp_name 1111_fgsm_6_1.75
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --train_random_start --epsilon 6 --train_fgsm_ratio 2 --exp_name 1111_fgsm_6_2
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --train_random_start --epsilon 5 --train_fgsm_ratio 0.25 --exp_name 1111_fgsm_5_0.25
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --train_random_start --epsilon 5 --train_fgsm_ratio 0.5 --exp_name 1111_fgsm_5_0.5
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --train_random_start --epsilon 5 --train_fgsm_ratio 0.75 --exp_name 1111_fgsm_5_0.75
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --train_random_start --epsilon 5 --train_fgsm_ratio 1 --exp_name 1111_fgsm_5_1
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --train_random_start --epsilon 5 --train_fgsm_ratio 1.25 --exp_name 1111_fgsm_5_1.25
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --train_random_start --epsilon 5 --train_fgsm_ratio 1.5 --exp_name 1111_fgsm_5_1.5
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --train_random_start --epsilon 5 --train_fgsm_ratio 1.75 --exp_name 1111_fgsm_5_1.75
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --train_random_start --epsilon 5 --train_fgsm_ratio 2 --exp_name 1111_fgsm_5_2
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --train_random_start --epsilon 4 --train_fgsm_ratio 0.25 --exp_name 1111_fgsm_4_0.25
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --train_random_start --epsilon 4 --train_fgsm_ratio 0.5 --exp_name 1111_fgsm_4_0.5
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --train_random_start --epsilon 4 --train_fgsm_ratio 0.75 --exp_name 1111_fgsm_4_0.75
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --train_random_start --epsilon 4 --train_fgsm_ratio 1 --exp_name 1111_fgsm_4_1
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --train_random_start --epsilon 4 --train_fgsm_ratio 1.25 --exp_name 1111_fgsm_4_1.25
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --train_random_start --epsilon 4 --train_fgsm_ratio 1.5 --exp_name 1111_fgsm_4_1.5
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --train_random_start --epsilon 4 --train_fgsm_ratio 1.75 --exp_name 1111_fgsm_4_1.75
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --train_random_start --epsilon 4 --train_fgsm_ratio 2 --exp_name 1111_fgsm_4_2
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --train_random_start --epsilon 3 --train_fgsm_ratio 0.25 --exp_name 1111_fgsm_3_0.25
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --train_random_start --epsilon 3 --train_fgsm_ratio 0.5 --exp_name 1111_fgsm_3_0.5
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --train_random_start --epsilon 3 --train_fgsm_ratio 0.75 --exp_name 1111_fgsm_3_0.75
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --train_random_start --epsilon 3 --train_fgsm_ratio 1 --exp_name 1111_fgsm_3_1
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --train_random_start --epsilon 3 --train_fgsm_ratio 1.25 --exp_name 1111_fgsm_3_1.25
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --train_random_start --epsilon 3 --train_fgsm_ratio 1.5 --exp_name 1111_fgsm_3_1.5
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --train_random_start --epsilon 3 --train_fgsm_ratio 1.75 --exp_name 1111_fgsm_3_1.75
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --train_random_start --epsilon 3 --train_fgsm_ratio 2 --exp_name 1111_fgsm_3_2
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --train_random_start --epsilon 2 --train_fgsm_ratio 0.25 --exp_name 1111_fgsm_2_0.25
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --train_random_start --epsilon 2 --train_fgsm_ratio 0.5 --exp_name 1111_fgsm_2_0.5
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --train_random_start --epsilon 2 --train_fgsm_ratio 0.75 --exp_name 1111_fgsm_2_0.75
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --train_random_start --epsilon 2 --train_fgsm_ratio 1 --exp_name 1111_fgsm_2_1
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --train_random_start --epsilon 2 --train_fgsm_ratio 1.25 --exp_name 1111_fgsm_2_1.25
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --train_random_start --epsilon 2 --train_fgsm_ratio 1.5 --exp_name 1111_fgsm_2_1.5
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --train_random_start --epsilon 2 --train_fgsm_ratio 1.75 --exp_name 1111_fgsm_2_1.75
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --train_random_start --epsilon 2 --train_fgsm_ratio 2 --exp_name 1111_fgsm_2_2
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --train_random_start --epsilon 1 --train_fgsm_ratio 0.25 --exp_name 1111_fgsm_1_0.25
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --train_random_start --epsilon 1 --train_fgsm_ratio 0.5 --exp_name 1111_fgsm_1_0.5
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --train_random_start --epsilon 1 --train_fgsm_ratio 0.75 --exp_name 1111_fgsm_1_0.75
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --train_random_start --epsilon 1 --train_fgsm_ratio 1 --exp_name 1111_fgsm_1_1
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --train_random_start --epsilon 1 --train_fgsm_ratio 1.25 --exp_name 1111_fgsm_1_1.25
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --train_random_start --epsilon 1 --train_fgsm_ratio 1.5 --exp_name 1111_fgsm_1_1.5
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --train_random_start --epsilon 1 --train_fgsm_ratio 1.75 --exp_name 1111_fgsm_1_1.75
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python train_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --train_random_start --epsilon 1 --train_fgsm_ratio 2 --exp_name 1111_fgsm_1_2
#ENDBSUB

#for i in 0.25 0.5 0.75 1 1.25 1.5 1.75 2
#do
#  PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#  bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --train_random_start --epsilon 9 --train_fgsm_ratio $i --exp_name 1111_fgsm_9_$i"
#done

#for i in 0.25 0.5 0.75 1 1.25 1.5 1.75 2
#do
#  PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#  bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --train_random_start --epsilon 10 --train_fgsm_ratio $i --exp_name 1111_fgsm_10_$i"
#done

#for i in 11 12 13 14
#do
#  for j in 0.25 0.5 0.75 1 1.25 1.5 1.75 2
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --train_random_start --epsilon $i --train_fgsm_ratio $j --exp_name 1111_fgsm_{$i}_$j"
#  done
#done

#for i in 15 16
#do
#  for j in 0.25 0.5 0.75 1 1.25 1.5 1.75 2
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 200 --batch_size 256 --train_random_start --epsilon $i --train_fgsm_ratio $j --exp_name 1111_fgsm_{$i}_$j"
#  done
#done

#for i in 8
#do
#  for j in 0.02
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  "module load $PCOMMAND; python train_deepfool.py --model PreActResNet18 --train_overshoot $j --train_deepfool_rs --epsilon $i --train_df_clamp --exp_name 1112_deepfool_{$i}_$j"
#  done
#done

#for i in 8
#do
#  for j in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  "module load $PCOMMAND; python train_deepfool.py --model PreActResNet18 --train_overshoot $j --train_deepfool_rs --epsilon $i --train_df_clamp --exp_name 1112_deepfool_{$i}_$j"
#  done
#done

#for i in 8 7 6 5 4 3 2 1 9 10 11 12 13 14 15 16
#do
#  for j in 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  "module load $PCOMMAND; python train_deepfool.py --model PreActResNet18 --train_overshoot $j --train_deepfool_rs --epsilon $i --train_df_clamp --exp_name 1112_deepfool_{$i}_$j"
#  done
#done
#
#for i in 13
#do
#  for j in 1
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=2]"  "module load $PCOMMAND; python train_deepfool.py --model PreActResNet18 --train_overshoot $j --train_deepfool_rs --epsilon $i --train_df_clamp --exp_name 1112_deepfool_{$i}_$j"
#  done
#done

#for i in 8
#do
#  for j in 1.25
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 30  --lr_schedule cyclic --lr-max 0.3 --train_random_start --epsilon $i --train_fgsm_ratio $j --exp_name 1115_fgsm_{$i}_$j"
#  done
#done

#for i in 8
#do
#  for j in 0.02
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool.py --model PreActResNet18 --num_epochs 30  --lr_schedule cyclic --lr-max 0.3 --epsilon $i --train_overshoot $j --train_deepfool_rs --train_df_clamp --exp_name 1115_deepfool_{$i}_$j"
#  done
#done

#for i in 8
#do
#  for j in 1 1.5 2 2.5 3 3.5 4
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool.py --model PreActResNet18 --num_epochs 30  --lr_schedule cyclic --lr-max 0.3 --epsilon $i --train_overshoot $j --train_deepfool_rs --train_df_clamp --exp_name 1115_deepfool_{$i}_$j"
#  done
#done

#for i in 8
#do
#  for j in 0.6 0.7 0.8 0.9
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool.py --model PreActResNet18 --num_epochs 30  --lr_schedule cyclic --lr-max $j --epsilon $i --train_overshoot 0.02 --train_deepfool_rs --train_df_clamp --exp_name 1115_deepfool_{$i}_0.02_$j"
#  done
#done

#for i in 8
#do
#  for j in 1
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool.py --model PreActResNet18 --num_epochs 30  --lr_schedule cyclic --lr-max 0.5 --epsilon $i --train_overshoot $j --train_deepfool_rs --train_df_clamp --exp_name 1115_deepfool_30_0.5_{$i}_$j"
#  done
#done

#for i in 8
#do
#  for j in 1
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool.py --model PreActResNet18 --num_epochs 50  --lr_schedule cyclic --lr-max 0.3 --epsilon $i --train_overshoot $j --train_deepfool_rs --train_df_clamp --exp_name 1115_deepfool_50_0.3_{$i}_$j"
#  done
#done
#
#for i in 8
#do
#  for j in 1
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool.py --model PreActResNet18 --num_epochs 50  --lr_schedule cyclic --lr-max 0.5 --epsilon $i --train_overshoot $j --train_deepfool_rs --train_df_clamp --exp_name 1115_deepfool_50_0.5_{$i}_$j"
#  done
#done
#
#for i in 8
#do
#  for j in 1
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool.py --model PreActResNet18 --num_epochs 70  --lr_schedule cyclic --lr-max 0.3 --epsilon $i --train_overshoot $j --train_deepfool_rs --train_df_clamp --exp_name 1115_deepfool_70_0.3_{$i}_$j"
#  done
#done
#
#for i in 8
#do
#  for j in 1
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool.py --model PreActResNet18 --num_epochs 70  --lr_schedule cyclic --lr-max 0.5 --epsilon $i --train_overshoot $j --train_deepfool_rs --train_df_clamp --exp_name 1115_deepfool_70_0.5_{$i}_$j"
#  done
#done

#for i in 8
#do
#  for j in 0.25 0.5 0.75 1 1.5 1.75 2
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 30  --lr_schedule cyclic --lr-max 0.3 --train_random_start --epsilon $i --train_fgsm_ratio $j --exp_name 1115_fgsm_{$i}_$j"
#  done
#done

#for i in 1 2 3 4 5 6 7 9 10 11 12 13 14 15 16
#do
#  for j in 0.25 0.5 0.75 1 1.25 1.5 1.75 2
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 30  --lr_schedule cyclic --lr-max 0.3 --train_random_start --epsilon $i --train_fgsm_ratio $j --exp_name 1115_fgsm_{$i}_$j"
#  done
#done

#for i in 8
#do
#  for j in 1.5 2 2.5 3 3.5 4
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool.py --model PreActResNet18 --num_epochs 30  --lr_schedule cyclic --lr-max 0.5 --epsilon $i --train_overshoot $j --train_deepfool_rs --train_df_clamp --exp_name 1115_deepfool_70_0.5_{$i}_$j"
#  done
#done
#
#for i in 1 2 3 4 5 6 7 9 10 11 12 13 14 15 16
#do
#  for j in 0.02 1 1.5 2 2.5 3 3.5 4
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool.py --model PreActResNet18 --num_epochs 30  --lr_schedule cyclic --lr-max 0.5 --epsilon $i --train_overshoot $j --train_deepfool_rs --train_df_clamp --exp_name 1115_deepfool_70_0.5_{$i}_$j"
#  done
#done

#for i in 0.25 0.5 0.75 1 1.25 1.5 1.75 2
#do
#  PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#  bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate.py --model PreActResNet18 --epsilon 8 --resumed_model_name 1111_fgsm_8_$i --exp_name evaluate_1111_fgsm_8_$i"
#done
#
#for i in 0.25 0.5 0.75 1 1.25 1.5 1.75 2
#do
#  PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#  bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate.py --model PreActResNet18 --epsilon 7 --resumed_model_name 1111_fgsm_7_$i --exp_name evaluate_1111_fgsm_7_$i"
#done
#
#for i in 0.25 0.5 0.75 1 1.25 1.5 1.75 2
#do
#  PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#  bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate.py --model PreActResNet18 --epsilon 6 --resumed_model_name 1111_fgsm_6_$i --exp_name evaluate_1111_fgsm_6_$i"
#done
#
#for i in 0.25 0.5 0.75 1 1.25 1.5 1.75 2
#do
#  PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#  bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate.py --model PreActResNet18 --epsilon 5 --resumed_model_name 1111_fgsm_5_$i --exp_name evaluate_1111_fgsm_5_$i"
#done
#
#for i in 0.25 0.5 0.75 1 1.25 1.5 1.75 2
#do
#  PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#  bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate.py --model PreActResNet18 --epsilon 4 --resumed_model_name 1111_fgsm_4_$i --exp_name evaluate_1111_fgsm_4_$i"
#done
#
#for i in 0.25 0.5 0.75 1 1.25 1.5 1.75 2
#do
#  PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#  bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate.py --model PreActResNet18 --epsilon 3 --resumed_model_name 1111_fgsm_3_$i --exp_name evaluate_1111_fgsm_3_$i"
#done
#
#for i in 0.25 0.5 0.75 1 1.25 1.5 1.75 2
#do
#  PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#  bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate.py --model PreActResNet18 --epsilon 2 --resumed_model_name 1111_fgsm_2_$i --exp_name evaluate_1111_fgsm_2_$i"
#done
#
#for i in 0.25 0.5 0.75 1 1.25 1.5 1.75 2
#do
#  PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#  bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate.py --model PreActResNet18 --epsilon 1 --resumed_model_name 1111_fgsm_1_$i --exp_name evaluate_1111_fgsm_1_$i"
#done
#
#for i in 0.25 0.5 0.75 1 1.25 1.5 1.75 2
#do
#  PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#  bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate.py --model PreActResNet18 --epsilon 9 --resumed_model_name 1111_fgsm_9_$i --exp_name evaluate_1111_fgsm_9_$i"
#done
#
#for i in 0.25 0.5 0.75 1 1.25 1.5 1.75 2
#do
#  PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#  bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate.py --model PreActResNet18 --epsilon 10 --resumed_model_name 1111_fgsm_10_$i --exp_name evaluate_1111_fgsm_10_$i"
#done
#
#for i in 11 12 13 14 15 16
#do
#  for j in 0.25 0.5 0.75 1 1.25 1.5 1.75 2
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate.py --model PreActResNet18 --epsilon $i --resumed_model_name 1111_fgsm_{$i}_$j --exp_name evaluate_1111_fgsm_{$i}_$j"
#  done
#done
#
#for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
#do
#  for j in 0.02 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate.py --model PreActResNet18 --epsilon $i --resumed_model_name 1112_deepfool_{$i}_$j --exp_name evaluate_1112_deepfool_{$i}_$j"
#  done
#done

#for i in 8
#do
#  for j in 0.02
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_Dfgsm_Ldeepfool.py --model PreActResNet18 --num_epochs 30  --lr_schedule cyclic --lr-max 0.3 --epsilon $i --train_overshoot $j --exp_name 1115_fgsm_deepfool_30_0.3_{$i}_$j"
#  done
#done

#for i in 8
#do
#  for j in 0.5 1 1.5 2
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_Dfgsm_Ldeepfool.py --model PreActResNet18 --num_epochs 30  --lr_schedule cyclic --lr-max 0.3 --epsilon $i --train_overshoot $j --exp_name 1115_fgsm_deepfool_30_0.3_{$i}_$j"
#  done
#done

#for i in 16
#do
#  for j in 0.02 0.5 1 1.5 2
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_Dfgsm_Ldeepfool.py --model PreActResNet18 --num_epochs 30  --lr_schedule cyclic --lr-max 0.3 --epsilon $i --train_overshoot $j --exp_name 1115_fgsm_deepfool_30_0.3_{$i}_$j"
#  done
#done

#for i in 8
#do
#  for j in 1
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_Dfgsm_Ldeepfool.py --model PreActResNet18 --num_epochs 30  --lr_schedule cyclic --lr-max 0.2 --epsilon $i --train_overshoot $j --exp_name 1115_fgsm_deepfool_30_0.2_{$i}_$j"
#  done
#done

#for i in 16
#do
#  for j in 0.02
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_Dfgsm_Ldeepfool.py --model PreActResNet18 --num_epochs 30  --lr_schedule cyclic --lr-max 0.3 --epsilon $i --train_overshoot $j --exp_name 1115_fgsm_deepfool_woabs_30_0.3_{$i}_$j"
#  done
#done

#for i in 16
#do
#  for j in 1
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 14:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_Dfgsm_Ldeepfool.py --model PreActResNet18 --num_epochs 30  --lr_schedule cyclic --lr-max 0.3 --epsilon $i --train_overshoot $j --exp_name 1115_fgsm_deepfool_woabs_30_0.3_{$i}_$j"
#  done
#done

#for i in 16
#do
#  for j in 1
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 14:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_Dfgsm_Ldeepfool.py --model PreActResNet18 --num_epochs 30  --lr_schedule cyclic --lr-max 0.3 --train_random_start --epsilon $i --train_overshoot $j --exp_name 1115_fgsm_deepfool_rs_30_0.3_{$i}_$j"
#  done
#done

#for i in 16
#do
#  for j in 0.5
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 14:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_Dfgsm_Ldeepfool.py --model PreActResNet18 --num_epochs 30  --lr_schedule cyclic --lr-max 0.3 --train_random_start --epsilon $i --train_overshoot $j --exp_name 1115_fgsm_deepfool_rs_30_0.3_{$i}_$j"
#  done
#done

#for i in 16
#do
#  for j in 0.02
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 14:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_Dfgsm_Ldeepfool.py --model PreActResNet18 --num_epochs 30  --lr_schedule cyclic --lr-max 0.3 --train_random_start --epsilon $i --train_overshoot $j --exp_name 1115_fgsm_deepfool_rs_30_0.3_{$i}_$j"
#  done
#done

#for i in 16
#do
#  for j in 0.02 0.5 1
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 14:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_Dfgsm_Ldeepfool.py --model PreActResNet18 --num_epochs 30  --lr_schedule cyclic --lr-max 0.3 --train_random_start --epsilon $i --train_overshoot $j --exp_name 1115_fgsm_deepfool_woabsrs_30_0.3_{$i}_$j"
#  done
#done

#for i in 8 16
#do
#  for j in -0.02
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 14:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_Dfgsm_Ldeepfool.py --model PreActResNet18 --num_epochs 30  --lr_schedule cyclic --lr-max 0.3 --epsilon $i --train_overshoot $j --exp_name 1115_fgsm_deepfool_wabs_wors_30_0.3_{$i}_$j"
#  done
#done

#for i in 16
#do
#  for j in 0.02 0.2 0.4 0.6 0.8 1 1.2 1.4 1.6 1.8 2
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate.py --model PreActResNet18 --epsilon $i --resumed_model_name 1112_deepfool_{$i}_$j --exp_name evaluate_1112_deepfool_{$i}_$j"
#  done
#done

#for i in 8 16
#do
#  for j in 0.25 0.5 0.75 1 1.25 1.5 1.75 2
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_test1.py --model PreActResNet18 --num_epochs 200  --lr_schedule multistep --epsilon $i --train_fgsm_ratio $j --exp_name 1116_fgsm_test1_{$i}_$j"
#  done
#done

#for i in 8 16
#do
#  for j in 0.25 0.5 0.75 1 1.25 1.5 1.75 2
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_test2.py --model PreActResNet18 --num_epochs 200  --lr_schedule multistep --epsilon $i --train_fgsm_ratio $j --exp_name 1116_fgsm_test2_{$i}_$j"
#  done
#done

#for i in 8
#do
#  for j in 0.02
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_test1.py --model PreActResNet18 --num_epochs 30 --lr_schedule cyclic --lr-max 0.3 --train_overshoot $j --epsilon $i --train_df_clamp --exp_name 1117_deepfool_test1_wors_{$i}_$j"
#  done
#done

#for i in 8
#do
#  for j in 0.5 1 1.5 2
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_test1.py --model PreActResNet18 --num_epochs 30 --lr_schedule cyclic --lr-max 0.3 --train_overshoot $j --epsilon $i --train_df_clamp --exp_name 1117_deepfool_test1_wors_{$i}_$j"
#  done
#done

#for i in 8
#do
#  for j in 0.02
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_test1.py --model PreActResNet18 --num_epochs 30 --lr_schedule cyclic --lr-max 0.3 --train_overshoot $j --epsilon $i --train_deepfool_rs --train_df_clamp --exp_name 1117_deepfool_test1_rs_{$i}_$j"
#  done
#done

#for i in 8
#do
#  for j in 0.5 1 1.5 2
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_test1.py --model PreActResNet18 --num_epochs 30 --lr_schedule cyclic --lr-max 0.3 --train_overshoot $j --epsilon $i --train_deepfool_rs --train_df_clamp --exp_name 1117_deepfool_test1_rs_{$i}_$j"
#  done
#done

#for i in 8
#do
#  for j in 0.02
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_test2.py --model PreActResNet18 --num_epochs 30 --lr_schedule cyclic --lr-max 0.3 --train_overshoot_pos $j --train_overshoot_neg -0.02 --epsilon $i --train_df_clamp --exp_name 1117_deepfool_test2_wors_-0.02_{$i}_$j"
#  done
#done
#
#for i in 8
#do
#  for j in 0.5
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_test2.py --model PreActResNet18 --num_epochs 30 --lr_schedule cyclic --lr-max 0.3 --train_overshoot_pos $j --train_overshoot_neg -0.5 --epsilon $i --train_df_clamp --exp_name 1117_deepfool_test2_wors_-0.5_{$i}_$j"
#  done
#done
#
#for i in 8
#do
#  for j in 1
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_test2.py --model PreActResNet18 --num_epochs 30 --lr_schedule cyclic --lr-max 0.3 --train_overshoot_pos $j --train_overshoot_neg -1 --epsilon $i --train_df_clamp --exp_name 1117_deepfool_test2_wors_-1_{$i}_$j"
#  done
#done
#
#for i in 8
#do
#  for j in 1.5
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_test2.py --model PreActResNet18 --num_epochs 30 --lr_schedule cyclic --lr-max 0.3 --train_overshoot_pos $j --train_overshoot_neg -1.5 --epsilon $i --train_df_clamp --exp_name 1117_deepfool_test2_wors_-1.5_{$i}_$j"
#  done
#done
#
#for i in 8
#do
#  for j in 2
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_test2.py --model PreActResNet18 --num_epochs 30 --lr_schedule cyclic --lr-max 0.3 --train_overshoot_pos $j --train_overshoot_neg -2 --epsilon $i --train_df_clamp --exp_name 1117_deepfool_test2_wors_-2_{$i}_$j"
#  done
#done
#
#
#for i in 8
#do
#  for j in 0.02
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_test2.py --model PreActResNet18 --num_epochs 30 --lr_schedule cyclic --lr-max 0.3 --train_overshoot_pos $j --train_overshoot_neg -0.02 --epsilon $i --train_deepfool_rs --train_df_clamp --exp_name 1117_deepfool_test2_rs_-0.02_{$i}_$j"
#  done
#done
#
#for i in 8
#do
#  for j in 0.5
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_test2.py --model PreActResNet18 --num_epochs 30 --lr_schedule cyclic --lr-max 0.3 --train_overshoot_pos $j --train_overshoot_neg -0.5 --epsilon $i --train_deepfool_rs --train_df_clamp --exp_name 1117_deepfool_test2_rs_-0.5_{$i}_$j"
#  done
#done
#
#for i in 8
#do
#  for j in 1
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_test2.py --model PreActResNet18 --num_epochs 30 --lr_schedule cyclic --lr-max 0.3 --train_overshoot_pos $j --train_overshoot_neg -1 --epsilon $i --train_deepfool_rs --train_df_clamp --exp_name 1117_deepfool_test2_rs_-1_{$i}_$j"
#  done
#done
#
#for i in 8
#do
#  for j in 1.5
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_test2.py --model PreActResNet18 --num_epochs 30 --lr_schedule cyclic --lr-max 0.3 --train_overshoot_pos $j --train_overshoot_neg -1.5 --epsilon $i --train_deepfool_rs --train_df_clamp --exp_name 1117_deepfool_test2_rs_-1.5_{$i}_$j"
#  done
#done
#
#for i in 8
#do
#  for j in 2
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_test2.py --model PreActResNet18 --num_epochs 30 --lr_schedule cyclic --lr-max 0.3 --train_overshoot_pos $j --train_overshoot_neg -2 --epsilon $i --train_deepfool_rs --train_df_clamp --exp_name 1117_deepfool_test2_rs_-2_{$i}_$j"
#  done
#done

#for i in 8
#do
#  for j in 2.5
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_test2.py --model PreActResNet18 --num_epochs 30 --lr_schedule cyclic --lr-max 0.3 --train_overshoot_pos $j --train_overshoot_neg -2.5 --epsilon $i --train_df_clamp --exp_name 1117_deepfool_test2_wors_-2.5_{$i}_$j"
#  done
#done
#
#for i in 8
#do
#  for j in 3
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_test2.py --model PreActResNet18 --num_epochs 30 --lr_schedule cyclic --lr-max 0.3 --train_overshoot_pos $j --train_overshoot_neg -3 --epsilon $i --train_df_clamp --exp_name 1117_deepfool_test2_wors_-3_{$i}_$j"
#  done
#done

#for i in 8
#do
#  for j in 3.5
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_test2.py --model PreActResNet18 --num_epochs 30 --lr_schedule cyclic --lr-max 0.3 --train_overshoot_pos $j --train_overshoot_neg -3.5 --epsilon $i --train_df_clamp --exp_name 1117_deepfool_test2_wors_-3.5_{$i}_$j"
#  done
#done
#
#for i in 8
#do
#  for j in 4
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_test2.py --model PreActResNet18 --num_epochs 30 --lr_schedule cyclic --lr-max 0.3 --train_overshoot_pos $j --train_overshoot_neg -4 --epsilon $i --train_df_clamp --exp_name 1117_deepfool_test2_wors_-4_{$i}_$j"
#  done
#done

#for i in 8
#do
#  for j in 4.5
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_test2.py --model PreActResNet18 --num_epochs 30 --lr_schedule cyclic --lr-max 0.3 --train_overshoot_pos $j --train_overshoot_neg -4.5 --epsilon $i --train_df_clamp --exp_name 1117_deepfool_test2_wors_-4.5_{$i}_$j"
#  done
#done
#
#for i in 8
#do
#  for j in 5
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_test2.py --model PreActResNet18 --num_epochs 30 --lr_schedule cyclic --lr-max 0.3 --train_overshoot_pos $j --train_overshoot_neg -5 --epsilon $i --train_df_clamp --exp_name 1117_deepfool_test2_wors_-5_{$i}_$j"
#  done
#done

#for i in 8
#do
#  for j in 5.5
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_test2.py --model PreActResNet18 --num_epochs 30 --lr_schedule cyclic --lr-max 0.3 --train_overshoot_pos $j --train_overshoot_neg -5.5 --epsilon $i --train_df_clamp --exp_name 1117_deepfool_test2_wors_-5.5_{$i}_$j"
#  done
#done
#
#for i in 8
#do
#  for j in 6
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_test2.py --model PreActResNet18 --num_epochs 30 --lr_schedule cyclic --lr-max 0.3 --train_overshoot_pos $j --train_overshoot_neg -6 --epsilon $i --train_df_clamp --exp_name 1117_deepfool_test2_wors_-6_{$i}_$j"
#  done
#done

#for i in 8
#do
#  for j in 6.5
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_test2.py --model PreActResNet18 --num_epochs 30 --lr_schedule cyclic --lr-max 0.3 --train_overshoot_pos $j --train_overshoot_neg -6.5 --epsilon $i --train_df_clamp --exp_name 1117_deepfool_test2_wors_-6.5_{$i}_$j"
#  done
#done
#
#for i in 8
#do
#  for j in 7
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_test2.py --model PreActResNet18 --num_epochs 30 --lr_schedule cyclic --lr-max 0.3 --train_overshoot_pos $j --train_overshoot_neg -7 --epsilon $i --train_df_clamp --exp_name 1117_deepfool_test2_wors_-7_{$i}_$j"
#  done
#done

#for i in 8
#do
#  for j in 7.5
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_test2.py --model PreActResNet18 --num_epochs 30 --lr_schedule cyclic --lr-max 0.3 --train_overshoot_pos $j --train_overshoot_neg -7.5 --epsilon $i --train_df_clamp --exp_name 1117_deepfool_test2_wors_-7.5_{$i}_$j"
#  done
#done
#
#for i in 8
#do
#  for j in 8
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_test2.py --model PreActResNet18 --num_epochs 30 --lr_schedule cyclic --lr-max 0.3 --train_overshoot_pos $j --train_overshoot_neg -8 --epsilon $i --train_df_clamp --exp_name 1117_deepfool_test2_wors_-8_{$i}_$j"
#  done
#done
#
#for i in 8
#do
#  for j in 8.5
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_test2.py --model PreActResNet18 --num_epochs 30 --lr_schedule cyclic --lr-max 0.3 --train_overshoot_pos $j --train_overshoot_neg -8.5 --epsilon $i --train_df_clamp --exp_name 1117_deepfool_test2_wors_-8.5_{$i}_$j"
#  done
#done
#
#for i in 8
#do
#  for j in 9
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_test2.py --model PreActResNet18 --num_epochs 30 --lr_schedule cyclic --lr-max 0.3 --train_overshoot_pos $j --train_overshoot_neg -9 --epsilon $i --train_df_clamp --exp_name 1117_deepfool_test2_wors_-9_{$i}_$j"
#  done
#done
#
#for i in 8
#do
#  for j in 9.5
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_test2.py --model PreActResNet18 --num_epochs 30 --lr_schedule cyclic --lr-max 0.3 --train_overshoot_pos $j --train_overshoot_neg -9.5 --epsilon $i --train_df_clamp --exp_name 1117_deepfool_test2_wors_-9.5_{$i}_$j"
#  done
#done
#
#for i in 8
#do
#  for j in 10
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_test2.py --model PreActResNet18 --num_epochs 30 --lr_schedule cyclic --lr-max 0.3 --train_overshoot_pos $j --train_overshoot_neg -10 --epsilon $i --train_df_clamp --exp_name 1117_deepfool_test2_wors_-10_{$i}_$j"
#  done
#done

#for i in 8
#do
#  for j in 10.5
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_test2.py --model PreActResNet18 --num_epochs 30 --lr_schedule cyclic --lr-max 0.3 --train_overshoot_pos $j --train_overshoot_neg -10.5 --epsilon $i --train_df_clamp --exp_name 1117_deepfool_test2_wors_-10.5_{$i}_$j"
#  done
#done
#
#for i in 8
#do
#  for j in 11
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_test2.py --model PreActResNet18 --num_epochs 30 --lr_schedule cyclic --lr-max 0.3 --train_overshoot_pos $j --train_overshoot_neg -11 --epsilon $i --train_df_clamp --exp_name 1117_deepfool_test2_wors_-11_{$i}_$j"
#  done
#done
#
#for i in 8
#do
#  for j in 11.5
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_test2.py --model PreActResNet18 --num_epochs 30 --lr_schedule cyclic --lr-max 0.3 --train_overshoot_pos $j --train_overshoot_neg -11.5 --epsilon $i --train_df_clamp --exp_name 1117_deepfool_test2_wors_-11.5_{$i}_$j"
#  done
#done
#
#for i in 8
#do
#  for j in 12
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_test2.py --model PreActResNet18 --num_epochs 30 --lr_schedule cyclic --lr-max 0.3 --train_overshoot_pos $j --train_overshoot_neg -12 --epsilon $i --train_df_clamp --exp_name 1117_deepfool_test2_wors_-12_{$i}_$j"
#  done
#done

#for i in 8
#do
#  for j in 12.5
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_test2.py --model PreActResNet18 --num_epochs 30 --lr_schedule cyclic --lr-max 0.3 --train_overshoot_pos $j --train_overshoot_neg -$j --epsilon $i --train_df_clamp --exp_name 1117_deepfool_test2_wors_{$i}_$j"
#  done
#done

#for i in 8
#do
#  for j in 2.5
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_test2.py --model PreActResNet18 --num_epochs 30 --lr_schedule cyclic --lr-max 0.3 --train_overshoot_pos $j --train_overshoot_neg -2.5 --epsilon $i --train_deepfool_rs --train_df_clamp --exp_name 1117_deepfool_test2_rs_-2.5_{$i}_$j"
#  done
#done
#
#for i in 8
#do
#  for j in 3
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_test2.py --model PreActResNet18 --num_epochs 30 --lr_schedule cyclic --lr-max 0.3 --train_overshoot_pos $j --train_overshoot_neg -3 --epsilon $i --train_deepfool_rs --train_df_clamp --exp_name 1117_deepfool_test2_rs_-3_{$i}_$j"
#  done
#done

#for i in 8
#do
#  for j in 3.5
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_test2.py --model PreActResNet18 --num_epochs 30 --lr_schedule cyclic --lr-max 0.3 --train_overshoot_pos $j --train_overshoot_neg -3.5 --epsilon $i --train_deepfool_rs --train_df_clamp --exp_name 1117_deepfool_test2_rs_-3.5_{$i}_$j"
#  done
#done
#
#for i in 8
#do
#  for j in 4
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_test2.py --model PreActResNet18 --num_epochs 30 --lr_schedule cyclic --lr-max 0.3 --train_overshoot_pos $j --train_overshoot_neg -4 --epsilon $i --train_deepfool_rs --train_df_clamp --exp_name 1117_deepfool_test2_rs_-4_{$i}_$j"
#  done
#done

#for i in 8
#do
#  for j in 4.5
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_test2.py --model PreActResNet18 --num_epochs 30 --lr_schedule cyclic --lr-max 0.3 --train_overshoot_pos $j --train_overshoot_neg -4.5 --epsilon $i --train_deepfool_rs --train_df_clamp --exp_name 1117_deepfool_test2_rs_-4.5_{$i}_$j"
#  done
#done
#
#for i in 8
#do
#  for j in 5
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_test2.py --model PreActResNet18 --num_epochs 30 --lr_schedule cyclic --lr-max 0.3 --train_overshoot_pos $j --train_overshoot_neg -5 --epsilon $i --train_deepfool_rs --train_df_clamp --exp_name 1117_deepfool_test2_rs_-5_{$i}_$j"
#  done
#done

#for i in 8
#do
#  for j in 5.5
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_test2.py --model PreActResNet18 --num_epochs 30 --lr_schedule cyclic --lr-max 0.3 --train_overshoot_pos $j --train_overshoot_neg -5.5 --epsilon $i --train_deepfool_rs --train_df_clamp --exp_name 1117_deepfool_test2_rs_-5.5_{$i}_$j"
#  done
#done
#
#for i in 8
#do
#  for j in 6
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_test2.py --model PreActResNet18 --num_epochs 30 --lr_schedule cyclic --lr-max 0.3 --train_overshoot_pos $j --train_overshoot_neg -6 --epsilon $i --train_deepfool_rs --train_df_clamp --exp_name 1117_deepfool_test2_rs_-6_{$i}_$j"
#  done
#done

#for i in 8
#do
#  for j in 6.5
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_test2.py --model PreActResNet18 --num_epochs 30 --lr_schedule cyclic --lr-max 0.3 --train_overshoot_pos $j --train_overshoot_neg -6.5 --epsilon $i --train_deepfool_rs --train_df_clamp --exp_name 1117_deepfool_test2_rs_-6.5_{$i}_$j"
#  done
#done
#
#for i in 8
#do
#  for j in 7
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_test2.py --model PreActResNet18 --num_epochs 30 --lr_schedule cyclic --lr-max 0.3 --train_overshoot_pos $j --train_overshoot_neg -7 --epsilon $i --train_deepfool_rs --train_df_clamp --exp_name 1117_deepfool_test2_rs_-7_{$i}_$j"
#  done
#done

#for i in 8 16
#do
#  for j in 0.02 0.4 0.8 1.2 1.6 2
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate.py --model PreActResNet18 --epsilon $i --eval_pgd_ratio 0.25 --eval_pgd_attack_iters 50 --eval_pgd_restarts 1 --resumed_model_name 1112_deepfool_{$i}_$j --exp_name 1117_evaluate_1112_deepfool_0.25_50_1_{$i}_$j"
#  done
#done
#
#for i in 8 16
#do
#  for j in 0.02 0.4 0.8 1.2 1.6 2
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate.py --model PreActResNet18 --epsilon $i --eval_pgd_ratio 0.25 --eval_pgd_attack_iters 10 --eval_pgd_restarts 10 --resumed_model_name 1112_deepfool_{$i}_$j --exp_name 1117_evaluate_1112_deepfool_0.25_10_10_{$i}_$j"
#  done
#done
#
#for i in 8 16
#do
#  for j in 0.02 0.4 0.8 1.2 1.6 2
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate.py --model PreActResNet18 --epsilon $i --eval_pgd_ratio 0.125 --eval_pgd_attack_iters 10 --eval_pgd_restarts 1 --resumed_model_name 1112_deepfool_{$i}_$j --exp_name 1117_evaluate_1112_deepfool_0.125_10_1_{$i}_$j"
#  done
#done

#for i in 16
#do
#  for j in 0.02 0.5 1 1.5 2 2.5 3 3.5 4 4.5 5 5.5 6 6.5 7 7.5 8 8.5 9 9.5 10 10.5 11 11.5 12
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_test2.py --model PreActResNet18 --num_epochs 30 --lr_schedule cyclic --lr-max 0.3 --train_overshoot_pos $j --train_overshoot_neg -$j --epsilon $i --train_df_clamp --exp_name 1117_deepfool_test2_wors_{$i}_$j"
#  done
#done

#for i in 16
#do
#  for j in 12.5 13 13.5 14 14.5 15 15.5 16
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_test2.py --model PreActResNet18 --num_epochs 30 --lr_schedule cyclic --lr-max 0.3 --train_overshoot_pos $j --train_overshoot_neg -$j --epsilon $i --train_df_clamp --exp_name 1117_deepfool_test2_wors_{$i}_$j"
#  done
#done

#for i in 16
#do
#  for j in 16.5 17 17.5 18 18.5 19 19.5 20
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_test2.py --model PreActResNet18 --num_epochs 30 --lr_schedule cyclic --lr-max 0.3 --train_overshoot_pos $j --train_overshoot_neg -$j --epsilon $i --train_df_clamp --exp_name 1117_deepfool_test2_wors_{$i}_$j"
#  done
#done

#for i in 16
#do
#  for j in 20.5 21 21.5 22 22.5 23 23.5 24 24.5 25 25.5 26 26.5 27
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_test2.py --model PreActResNet18 --num_epochs 30 --lr_schedule cyclic --lr-max 0.3 --train_overshoot_pos $j --train_overshoot_neg -$j --epsilon $i --train_df_clamp --exp_name 1117_deepfool_test2_wors_{$i}_$j"
#  done
#done

#
#for i in 8
#do
#  for j in 13 13.5 14 14.5 15 15.5 16
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_test2.py --model PreActResNet18 --num_epochs 30 --lr_schedule cyclic --lr-max 0.3 --train_overshoot_pos $j --train_overshoot_neg -$j --epsilon $i --train_df_clamp --exp_name 1117_deepfool_test2_wors_{$i}_$j"
#  done
#done

#for i in 8
#do
#  for j in 16.5 17 17.5 18 18.5 19 19.5 20
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_test2.py --model PreActResNet18 --num_epochs 30 --lr_schedule cyclic --lr-max 0.3 --train_overshoot_pos $j --train_overshoot_neg -$j --epsilon $i --train_df_clamp --exp_name 1117_deepfool_test2_wors_{$i}_$j"
#  done
#done

#for i in 16
#do
#  for j in 0.02 0.5 1 1.5 2 2.5 3 3.5 4 4.5 5 5.5 6 6.5 7 7.5 8 8.5 9 9.5 10 10.5 11 11.5 12 12.5 13 13.5 14 14.5 15 15.5 16
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_test2.py --model PreActResNet18 --num_epochs 30 --lr_schedule cyclic --lr-max 0.2 --train_overshoot_pos $j --train_overshoot_neg -$j --epsilon $i --train_df_clamp --exp_name 1117_deepfool_test2_wors_lr0.2_{$i}_$j"
#  done
#done

#for i in 16
#do
#  for j in 12.5 13 13.5 14 14.5 15 15.5 16 16.5 17 17.5 18 18.5 19 19.5 20
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_test2.py --model PreActResNet18 --num_epochs 30 --lr_schedule cyclic --lr-max 0.15 --train_overshoot_pos $j --train_overshoot_neg -$j --epsilon $i --train_df_clamp --exp_name 1117_deepfool_test2_wors_lr0.15_{$i}_$j"
#  done
#done

#for i in 16
#do
#  for j in 20.5 21 21.5 22 22.5 23 23.5 24 24.5 25 25.5 26 26.5 27
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_test2.py --model PreActResNet18 --num_epochs 30 --lr_schedule cyclic --lr-max 0.15 --train_overshoot_pos $j --train_overshoot_neg -$j --epsilon $i --train_df_clamp --exp_name 1117_deepfool_test2_wors_lr0.15_{$i}_$j"
#  done
#done

#for i in 16
#do
#  for j in 12.5 13 13.5 14 14.5 15 15.5 16 16.5 17 17.5 18 18.5 19 19.5 20 20.5 21 21.5 22 22.5 23 23.5 24 24.5 25 25.5 26 26.5 27
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_test2.py --model PreActResNet18 --num_epochs 30 --lr_schedule cyclic --lr-max 0.1 --train_overshoot_pos $j --train_overshoot_neg -$j --epsilon $i --train_df_clamp --exp_name 1117_deepfool_test2_wors_lr0.1_{$i}_$j"
#  done
#done

#for i in 16
#do
#  for j in 27.5 28 28.5 29 29.5 30 30.5 31 31.5 32 32.5 33 33.5 34 34.5 35
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_test2.py --model PreActResNet18 --num_epochs 30 --lr_schedule cyclic --lr-max 0.1 --train_overshoot_pos $j --train_overshoot_neg -$j --epsilon $i --train_df_clamp --exp_name 1117_deepfool_test2_wors_lr0.1_{$i}_$j"
#  done
#done

#for i in 16
#do
#  for j in 35.5 36 36.5 37 37.5 38 38.5 39 39.5 40 40.5 41 41.5 42
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_test2.py --model PreActResNet18 --num_epochs 30 --lr_schedule cyclic --lr-max 0.1 --train_overshoot_pos $j --train_overshoot_neg -$j --epsilon $i --train_df_clamp --exp_name 1117_deepfool_test2_wors_lr0.1_{$i}_$j"
#  done
#done
#
#for i in 16
#do
#  for j in 100
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_test2.py --model PreActResNet18 --num_epochs 30 --lr_schedule cyclic --lr-max 0.3 --train_overshoot_pos $j --train_overshoot_neg -$j --epsilon $i --train_df_clamp --exp_name 1117_deepfool_test2_wors_{$i}_$j"
#  done
#done

#for i in 8
#do
#  for j in 40 60 80 100
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_test2.py --model PreActResNet18 --num_epochs 30 --lr_schedule cyclic --lr-max 0.3 --train_overshoot_pos $j --train_overshoot_neg -$j --epsilon $i --train_df_clamp --exp_name 1117_deepfool_test2_wors_{$i}_$j"
#  done
#done

#for i in 8
#do
#  for j in 150 200 250 300 350 400
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_test2.py --model PreActResNet18 --num_epochs 30 --lr_schedule cyclic --lr-max 0.3 --train_overshoot_pos $j --train_overshoot_neg -$j --epsilon $i --train_df_clamp --exp_name 1117_deepfool_test2_wors_{$i}_$j"
#  done
#done

#for i in 8
#do
#  for j in 1000 10000
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_test2.py --model PreActResNet18 --num_epochs 30 --lr_schedule cyclic --lr-max 0.3 --train_overshoot_pos $j --train_overshoot_neg -$j --epsilon $i --train_df_clamp --exp_name 1117_deepfool_test2_wors_{$i}_$j"
#  done
#done

#for i in 8
#do
#  for j in 100000
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_test2.py --model PreActResNet18 --num_epochs 30 --lr_schedule cyclic --lr-max 0.3 --train_overshoot_pos $j --train_overshoot_neg -$j --epsilon $i --train_df_clamp --exp_name 1117_deepfool_test2_wors_{$i}_$j"
#  done
#done

#for i in 16
#do
#  for j in 150 200 250 300 350 400
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_test2.py --model PreActResNet18 --num_epochs 30 --lr_schedule cyclic --lr-max 0.3 --train_overshoot_pos $j --train_overshoot_neg -$j --epsilon $i --train_df_clamp --exp_name 1117_deepfool_test2_wors_{$i}_$j"
#  done
#done

#for i in 16
#do
#  for j in 50 100 150 200 250 300 350 400
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_test2.py --model PreActResNet18 --num_epochs 30 --lr_schedule cyclic --lr-max 0.2 --train_overshoot_pos $j --train_overshoot_neg -$j --epsilon $i --train_df_clamp --exp_name 1117_deepfool_test2_wors_lr0.2_{$i}_$j"
#  done
#done
#
#for i in 16
#do
#  for j in 500 600 700 800 900 1000
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_test2.py --model PreActResNet18 --num_epochs 30 --lr_schedule cyclic --lr-max 0.2 --train_overshoot_pos $j --train_overshoot_neg -$j --epsilon $i --train_df_clamp --exp_name 1117_deepfool_test2_wors_lr0.2_{$i}_$j"
#  done
#done

#for i in 16
#do
#  for j in 10000 100000
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_test2.py --model PreActResNet18 --num_epochs 30 --lr_schedule cyclic --lr-max 0.2 --train_overshoot_pos $j --train_overshoot_neg -$j --epsilon $i --train_df_clamp --exp_name 1117_deepfool_test2_wors_lr0.2_{$i}_$j"
#  done
#done

#for i in 16
#do
#  for j in 10000 100000
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_test2.py --model PreActResNet18 --num_epochs 30 --lr_schedule cyclic --lr-max 0.2 --train_overshoot_pos $j --train_overshoot_neg -$j --epsilon $i --train_df_clamp --exp_name 1117_deepfool_test2_wors_lr0.2_{$i}_$j"
#  done
#done

#for i in 16
#do
#  for j in 0.02
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_test1.py --model PreActResNet18 --num_epochs 30 --lr_schedule cyclic --lr-max 0.3 --train_overshoot $j --epsilon $i --train_df_clamp --exp_name 1117_deepfool_test1_wors_{$i}_$j"
#  done
#done

#for i in 16
#do
#  for j in 0.5 1 1.5 2
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_test1.py --model PreActResNet18 --num_epochs 30 --lr_schedule cyclic --lr-max 0.3 --train_overshoot $j --epsilon $i --train_df_clamp --exp_name 1117_deepfool_test1_wors_{$i}_$j"
#  done
#done

#for i in 8 16
#do
#  for j in 0.02 1 2 3 4 5 6 7 8 9 10 15 20 25 30 35 40 45 50 55 60 100 1000 10000
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_test2.py --model PreActResNet18 --num_epochs 30 --lr_schedule cyclic --lr-max 0.3 --train_overshoot_pos $j --train_overshoot_neg -$j --epsilon $i --train_deepfool_norm_dist l_2 --train_deepfool_rs --train_df_clamp --exp_name 1117_deepfool_test2_l2_rs_{$i}_$j"
#  done
#done
#
#for i in 8 16
#do
#  for j in 0.02 1 2 3 4 5 6 7 8 9 10 15 20 25 30 35 40 45 50 55 60 100 1000 10000
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_test2.py --model PreActResNet18 --num_epochs 30 --lr_schedule cyclic --lr-max 0.3 --train_overshoot_pos $j --train_overshoot_neg -$j --epsilon $i --train_deepfool_norm_dist l_2 --train_df_clamp --exp_name 1117_deepfool_test2_l2_wors_{$i}_$j"
#  done
#done
#
#for i in 8 16
#do
#  for j in 0.02 1 2 3 4 5 6 7 8 9 10 15 20 25 30 35 40 45 50 55 60 100 1000 10000
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_test2.py --model PreActResNet18 --num_epochs 30 --lr_schedule cyclic --lr-max 0.2 --train_overshoot_pos $j --train_overshoot_neg -$j --epsilon $i --train_deepfool_norm_dist l_2 --train_deepfool_rs --train_df_clamp --exp_name 1117_deepfool_test2_l2_rs_lr0.2_{$i}_$j"
#  done
#done
#
#for i in 8 16
#do
#  for j in 0.02 1 2 3 4 5 6 7 8 9 10 15 20 25 30 35 40 45 50 55 60 100 1000 10000
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_test2.py --model PreActResNet18 --num_epochs 30 --lr_schedule cyclic --lr-max 0.2 --train_overshoot_pos $j --train_overshoot_neg -$j --epsilon $i --train_deepfool_norm_dist l_2 --train_df_clamp --exp_name 1117_deepfool_test2_l2_wors_lr0.2_{$i}_$j"
#  done
#done

#for i in 8 16
#do
#  for j in 0.02 1 2 3 4 5 6 7 8 9 10 15 20 25 30 35 40 45 50 55 60 100 1000 10000
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_test1.py --model PreActResNet18 --num_epochs 30 --lr_schedule cyclic --lr-max 0.3 --train_overshoot $j --epsilon $i --train_deepfool_norm_dist l_2 --train_deepfool_rs --train_df_clamp --exp_name 1117_deepfool_test1_l2_rs_{$i}_$j"
#  done
#done
#
#for i in 8 16
#do
#  for j in 0.02 1 2 3 4 5 6 7 8 9 10 15 20 25 30 35 40 45 50 55 60 100 1000 10000
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_test1.py --model PreActResNet18 --num_epochs 30 --lr_schedule cyclic --lr-max 0.3 --train_overshoot $j --epsilon $i --train_deepfool_norm_dist l_2 --train_df_clamp --exp_name 1117_deepfool_test1_l2_wors_{$i}_$j"
#  done
#done
#
#for i in 8 16
#do
#  for j in 0.02 1 2 3 4 5 6 7 8 9 10 15 20 25 30 35 40 45 50 55 60 100 1000 10000
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_test1.py --model PreActResNet18 --num_epochs 30 --lr_schedule cyclic --lr-max 0.2 --train_overshoot -$j --epsilon $i --train_deepfool_norm_dist l_2 --train_deepfool_rs --train_df_clamp --exp_name 1117_deepfool_test1_l2_rs_lr0.2_{$i}_$j"
#  done
#done
#
#for i in 8 16
#do
#  for j in 0.02 1 2 3 4 5 6 7 8 9 10 15 20 25 30 35 40 45 50 55 60 100 1000 10000
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_test1.py --model PreActResNet18 --num_epochs 30 --lr_schedule cyclic --lr-max 0.2 --train_overshoot $j --epsilon $i --train_deepfool_norm_dist l_2 --train_df_clamp --exp_name 1117_deepfool_test1_l2_wors_lr0.2_{$i}_$j"
#  done
#done

#for i in 8 16
#do
#  for j in 0.02 0.4 1.2 2 4 8 16
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_test2.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --train_overshoot_pos $j --train_overshoot_neg -$j --epsilon $i --train_df_clamp --exp_name 1118_deepfool_test2_wors_{$i}_$j"
#  done
#done

#for i in 8 16
#do
#  for j in 0.02 0.4 1.2 2 4 8 16
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_test2.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --train_overshoot_pos $j --train_overshoot_neg -$j --epsilon $i --train_deepfool_norm_dist l_2 --train_df_clamp --exp_name 1118_deepfool_test2_l2_wors_{$i}_$j"
#  done
#done

#for i in 8 16
#do
#  for j in 0.02 0.4 1.2 2 4 8 16
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --train_overshoot $j --epsilon $i --train_deepfool_norm_dist l_2 --train_df_clamp --exp_name 1118_deepfool_l2_wors_{$i}_$j"
#  done
#done

#for i in 8 16
#do
#  for j in 0.02 0.4 1.2 2 4 8 16
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_test2.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --train_overshoot_pos $j --train_overshoot_neg -$j --epsilon $i --train_deepfool_norm_dist l_2 --train_deepfool_rs --train_df_clamp --exp_name 1118_deepfool_test2_l2_rs_{$i}_$j"
#  done
#done
#
#for i in 8 16
#do
#  for j in 0.02 0.4 1.2 2 4 8 16
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --train_overshoot $j --epsilon $i --train_deepfool_norm_dist l_2 --train_deepfool_rs --train_df_clamp --exp_name 1118_deepfool_l2_rs_{$i}_$j"
#  done
#done

#for i in 8 16
#do
#  for j in 0.02 0.4 1.2 2 4 8 16
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_test2.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --train_overshoot_pos $j --train_overshoot_neg -$j --epsilon $i --train_deepfool_norm_dist l_inf --train_deepfool_rs --train_df_clamp --exp_name 1118_deepfool_test2_linf_rs_{$i}_$j"
#  done
#done
#
#for i in 8 16
#do
#  for j in 0.02 0.4 1.2 2 4 8 16
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --train_overshoot $j --epsilon $i --train_deepfool_norm_dist l_inf --train_deepfool_rs --train_df_clamp --exp_name 1118_deepfool_linf_rs_{$i}_$j"
#  done
#done

#for i in 8
#do
#  for j in 0.02
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_Dfgsm_Ldeepfool.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --train_overshoot $j --epsilon $i --train_deepfool_norm_dist l_2 --exp_name 1118_fgsm_deepfool_{$i}_$j"
#  done
#done

#for i in 8
#do
#  for j in 0.4 1.2 2 4 8 16
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_Dfgsm_Ldeepfool.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --train_overshoot $j --epsilon $i --train_deepfool_norm_dist l_2 --exp_name 1118_fgsm_deepfool_{$i}_$j"
#  done
#done

#for i in 8
#do
#  for j in 0.02
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_Dfgsm_Ldeepfool.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --train_overshoot $j --epsilon $i --train_deepfool_norm_dist l_2 --exp_name test_1118_fgsm_deepfool_{$i}_$j"
#  done
#done

#for i in 8
#do
#  for j in 0.4 1.2 2 4 8 16
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_Dfgsm_Ldeepfool.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --train_overshoot $j --epsilon $i --train_deepfool_norm_dist l_2 --exp_name test_1118_fgsm_deepfool_{$i}_$j"
#  done
#done

#for i in 0.02 0.4 1.2
#do
#  PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#  bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate.py --model PreActResNet18 --epsilon 8 --resumed_model_name 1118_deepfool_linf_rs_{8}_$i --exp_name evaluate_1118_deepfool_linf_rs_{8}_$i"
#done
#
#for i in 0.02 8 16
#do
#  PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#  bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate.py --model PreActResNet18 --epsilon 16 --resumed_model_name 1118_deepfool_linf_rs_{16}_$i --exp_name evaluate_1118_deepfool_linf_rs_{16}_$i"
#done
#
#for i in 16
#do
#  PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#  bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate.py --model PreActResNet18 --epsilon 8 --resumed_model_name 1118_deepfool_test2_linf_rs_{8}_$i --exp_name evaluate_1118_deepfool_test2_linf_rs_{8}_$i"
#done
#
#for i in 0.4 8 16
#do
#  PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#  bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate.py --model PreActResNet18 --epsilon 16 --resumed_model_name 1118_deepfool_test2_linf_rs_{16}_$i --exp_name evaluate_1118_deepfool_test2_linf_rs_{16}_$i"
#done

#for i in 8 16
#do
#  for j in 32 64
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_test2.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --train_overshoot_pos $j --train_overshoot_neg -$j --epsilon $i --train_deepfool_norm_dist l_2 --train_df_clamp --exp_name 1118_deepfool_test2_l2_wors_{$i}_$j"
#  done
#done
#
#for i in 8 16
#do
#  for j in 32 64
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --train_overshoot $j --epsilon $i --train_deepfool_norm_dist l_2 --train_df_clamp --exp_name 1118_deepfool_l2_wors_{$i}_$j"
#  done
#done
#
#
#for i in 8 16
#do
#  for j in 32 64
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_test2.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --train_overshoot_pos $j --train_overshoot_neg -$j --epsilon $i --train_deepfool_norm_dist l_2 --train_deepfool_rs --train_df_clamp --exp_name 1118_deepfool_test2_l2_rs_{$i}_$j"
#  done
#done
#
#for i in 8 16
#do
#  for j in 32 64
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --train_overshoot $j --epsilon $i --train_deepfool_norm_dist l_2 --train_deepfool_rs --train_df_clamp --exp_name 1118_deepfool_l2_rs_{$i}_$j"
#  done
#done
#
#
#for i in 8 16
#do
#  for j in 32 64
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_test2.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --train_overshoot_pos $j --train_overshoot_neg -$j --epsilon $i --train_deepfool_norm_dist l_inf --train_deepfool_rs --train_df_clamp --exp_name 1118_deepfool_test2_linf_rs_{$i}_$j"
#  done
#done
#
#for i in 8 16
#do
#  for j in 32 64
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --train_overshoot $j --epsilon $i --train_deepfool_norm_dist l_inf --train_deepfool_rs --train_df_clamp --exp_name 1118_deepfool_linf_rs_{$i}_$j"
#  done
#done

#for i in 8
#do
#  for j in 0.02 0.4 1.2 2 4 8 16
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_Dfgsm_Ldeepfool.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --train_overshoot $j --epsilon $i --train_deepfool_norm_dist l_2 --train_df_clamp --exp_name test_1118_fgsm_deepfool_clamp_{$i}_$j"
#  done
#done

#for i in 8
#do
#  for j in 0.02 0.4 1.2 2 4 8 16
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_Dfgsm_Ldeepfool.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --train_overshoot $j --epsilon $i --train_deepfool_norm_dist l_2 --train_df_clamp --exp_name test1_1118_fgsm_deepfool_clamp_{$i}_$j"
#  done
#done

#for i in 16
#do
#  for j in 0.02 0.4 1.2 2 4 8 16
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_Dfgsm_Ldeepfool.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --train_overshoot $j --epsilon $i --train_deepfool_norm_dist l_2 --train_df_clamp --exp_name test1_1118_fgsm_deepfool_clamp_{$i}_$j"
#  done
#done

#for i in 8
#do
#  for j in 0.02 0.4 1.2 2 4 8 16
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_Dfgsm_Ldeepfool.py --model PreActResNet18 --num_epochs 30 --lr_schedule cyclic --train_overshoot $j --epsilon $i --train_deepfool_norm_dist l_2 --train_df_clamp --exp_name test1_1118_fgsm_deepfool_clamp_cyclic_{$i}_$j"
#  done
#done

#for i in 16
#do
#  for j in 0.02 0.4 1.2 2 4 8 16
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_Dfgsm_Ldeepfool.py --model PreActResNet18 --num_epochs 30 --lr_schedule cyclic --train_overshoot $j --epsilon $i --train_deepfool_norm_dist l_2 --train_df_clamp --exp_name test1_1118_fgsm_deepfool_clamp_cyclic_{$i}_$j"
#  done
#done

#for i in 8 16
#do
#  for j in 0.25 0.5 0.75 1 1.25 1.5 1.75 2
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_test0.py --model PreActResNet18 --num_epochs 200  --lr_schedule multistep --epsilon $i --train_fgsm_ratio $j --exp_name 1120_fgsm_test0_{$i}_$j"
#  done
#done

#for i in 8 16
#do
#  for j in 0.25 0.5 0.75 1 1.25 1.5 1.75 2
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 200  --lr_schedule multistep --percentile 85 --epsilon $i --train_fgsm_ratio $j --exp_name 1120_anisotropy_85_{$i}_$j"
#  done
#done

#for i in 8 16
#do
#  for j in 0.25 0.5 0.75 1 1.25 1.5 1.75 2
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 200  --lr_schedule multistep --percentile 75 --epsilon $i --train_fgsm_ratio $j --exp_name 1120_anisotropy_75_{$i}_$j"
#  done
#done
#
#for i in 8 16
#do
#  for j in 0.25 0.5 0.75 1 1.25 1.5 1.75 2
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 200  --lr_schedule multistep --percentile 65 --epsilon $i --train_fgsm_ratio $j --exp_name 1120_anisotropy_65_{$i}_$j"
#  done
#done
#
#for i in 8 16
#do
#  for j in 0.25 0.5 0.75 1 1.25 1.5 1.75 2
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 200  --lr_schedule multistep --percentile 55 --epsilon $i --train_fgsm_ratio $j --exp_name 1120_anisotropy_55_{$i}_$j"
#  done
#done
#
#for i in 8 16
#do
#  for j in 0.25 0.5 0.75 1 1.25 1.5 1.75 2
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 200  --lr_schedule multistep --percentile 45 --epsilon $i --train_fgsm_ratio $j --exp_name 1120_anisotropy_45_{$i}_$j"
#  done
#done

#for i in 8 16
#do
#  for j in 0.25 0.5 0.75 1 1.25 1.5 1.75 2
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 200  --lr_schedule multistep --percentile 85 --epsilon $i --train_fgsm_ratio $j --train_random_start --exp_name 1120_anisotropy_rs_85_{$i}_$j"
#  done
#done

#for i in 16
#do
#  for j in 0.0625 0.125 0.1875 0.25 0.3125 0.375 0.4375 0.5 0.5625 0.625 0.6875 0.75 0.8125 0.875 0.9375 1 1.0625 1.125 1.1875 1.25 1.3125 1.375 1.4375 1.5 1.5625 1.625 1.6875 1.75 1.8125 1.875 1.9375 2
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 200  --lr_schedule multistep --percentile 85 --epsilon $i --train_fgsm_ratio $j --exp_name 1121_anisotropy_85_{$i}_$j"
#  done
#done
#
#for i in 16
#do
#  for j in 0.0625 0.125 0.1875 0.25 0.3125 0.375 0.4375 0.5 0.5625 0.625 0.6875 0.75 0.8125 0.875 0.9375 1 1.0625 1.125 1.1875 1.25 1.3125 1.375 1.4375 1.5 1.5625 1.625 1.6875 1.75 1.8125 1.875 1.9375 2
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 200  --lr_schedule multistep --percentile 95 --epsilon $i --train_fgsm_ratio $j --exp_name 1121_anisotropy_95_{$i}_$j"
#  done
#done

#for i in 8 16
#do
#  for j in 0 -0.02 -0.4 -0.8 -1.2
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --train_overshoot $j --epsilon $i --train_deepfool_norm_dist l_2 --train_df_clamp --exp_name 1118_deepfool_l2_wors_{$i}_$j"
#  done
#done
#
#
#for i in 8 16
#do
#  for j in 0 -0.02 -0.4 -0.8 -1.2
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --train_overshoot $j --epsilon $i --train_deepfool_norm_dist l_2 --train_deepfool_rs --train_df_clamp --exp_name 1118_deepfool_l2_rs_{$i}_$j"
#  done
#done
#
#
#for i in 8 16
#do
#  for j in 0 -0.02 -0.4 -0.8 -1.2
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --train_overshoot $j --epsilon $i --train_deepfool_norm_dist l_inf --train_deepfool_rs --train_df_clamp --exp_name 1118_deepfool_linf_rs_{$i}_$j"
#  done
#done

#for i in 8 16
#do
#  for j in 0.02 2 4 8 16 32 64
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_rs_test1.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --train_overshoot $j --epsilon $i --train_deepfool_norm_dist l_2 --exp_name 1122_deepfool_l2_rs_test1_{$i}_$j"
#  done
#done

#for i in 8 16
#do
#  for j in 0.02 2 4 8 16 32 64
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_rs_test2.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --train_overshoot $j --epsilon $i --train_deepfool_norm_dist l_2 --exp_name 1122_deepfool_l2_rs_test2_{$i}_$j"
#  done
#done

#for i in 8
#do
#  for j in 32 64
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_rs.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --train_overshoot $j --epsilon $i --train_deepfool_norm_dist l_2 --exp_name 1122_deepfool_l2_rs_{$i}_$j"
#  done
#done
#
#for i in 8
#do
#  for j in 32 64
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_rs.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --train_overshoot $j --epsilon $i --train_deepfool_norm_dist l_inf --exp_name 1122_deepfool_linf_rs_{$i}_$j"
#  done
#done

#for i in 8
#do
#  for j in -1.2 -0.8 -0.4 -0.02 0 0.02 0.4 1.2 2 4 8 16
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_rs.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --train_overshoot $j --epsilon $i --train_deepfool_norm_dist l_2 --exp_name 1122_deepfool_l2_rs_{$i}_$j"
#  done
#done
#
#for i in 8
#do
#  for j in -1.2 -0.8 -0.4 -0.02 0 0.02 0.4 1.2 2 4 8 16
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_rs.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --train_overshoot $j --epsilon $i --train_deepfool_norm_dist l_inf --exp_name 1122_deepfool_linf_rs_{$i}_$j"
#  done
#done

#for i in 16
#do
#  for j in -1.2 -0.8 -0.4 -0.02 0 0.02 0.4 1.2 2 4 8 16 32 64
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_rs.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --train_overshoot $j --epsilon $i --train_deepfool_norm_dist l_2 --exp_name 1122_deepfool_l2_rs_{$i}_$j"
#  done
#done
#
#for i in 16
#do
#  for j in -1.2 -0.8 -0.4 -0.02 0 0.02 0.4 1.2 2 4 8 16 32 64
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_rs.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --train_overshoot $j --epsilon $i --train_deepfool_norm_dist l_inf --exp_name 1122_deepfool_linf_rs_{$i}_$j"
#  done
#done

#for i in 8
#do
#  for j in 96 128 256
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_rs.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --train_overshoot $j --epsilon $i --train_deepfool_norm_dist l_2 --exp_name 1122_deepfool_l2_rs_{$i}_$j"
#  done
#done

#for i in 1
#do
#  PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#  bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate.py --model PreActResNet18 --epsilon 8 --resumed_model_name 1120_anisotropy_65_{8}_1 --exp_name evaluate_1120_anisotropy_65_{8}_1"
#done

#for i in 1 1.25 1.5 1.75 2
#do
#  for j in 99
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 200  --lr_schedule multistep --percentile $j --epsilon 16 --train_fgsm_ratio $i --exp_name 1123_anisotropy_16_{$i}_$j"
#  done
#done

#for i in 8 16
#do
#  for j in 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 200  --lr_schedule multistep --percentile $j --epsilon $i --exp_name 1123_anisotropy_wors_{$i}_$j"
#  done
#done
#
#for i in 8 16
#do
#  for j in 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 200  --lr_schedule multistep --percentile $j --epsilon $i --train_random_start --exp_name 1123_anisotropy_rs_{$i}_$j"
#  done
#done

#for i in 8 16
#do
#  for j in 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 30 --lr_schedule cyclic --percentile $j --epsilon $i --exp_name 1123_anisotropy_cyclic_wors_{$i}_$j"
#  done
#done
#
#for i in 8 16
#do
#  for j in 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_anisotropy_fgsm.py --model PreActResNet18 --num_epochs 30 --lr_schedule cyclic --percentile $j --epsilon $i --train_random_start --exp_name 1123_anisotropy_cyclic_rs_{$i}_$j"
#  done
#done

#for i in 8 16
#do
#  for j in 0.25 0.5 0.75 1 1.25 1.5 1.75 2
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 30 --lr_schedule cyclic --epsilon $i --train_fgsm_ratio $j --exp_name 1123_fgsm_cyclic_wors_{$i}_$j"
#  done
#done
#
#
#for i in 8 16
#do
#  for j in 0.25 0.5 0.75 1 1.25 1.5 1.75 2
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 30 --lr_schedule cyclic --epsilon $i --train_fgsm_ratio $j --train_random_start --exp_name 1123_fgsm_cyclic_rs_{$i}_$j"
#  done
#done

#for i in 8
#do
#  for j in 0.02 -0.02 -0.4 -0.8 -1.2
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon $i --train_overshoot $j --train_deepfool_norm_dist l_inf --exp_name 1124_deepfool_linf_wors_{$i}_$j"
#  done
#done

#for i in 8
#do
#  for j in 0.02 64
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon $i --train_overshoot $j --train_deepfool_norm_dist l_2 --exp_name 1124_deepfool_l2_wors_{$i}_$j"
#  done
#done

#for i in 8 16
#do
#  for j in -0.8 -0.4 -0.02 0 0.02 0.4 0.8 1.2 1.6 2 4 8 16
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_Dfgsm_Ldeepfool.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon $i --train_overshoot $j --train_deepfool_norm_dist l_inf --exp_name 1124_fgsm_deepfool_linf_wors_{$i}_$j"
#  done
#done

#for i in 8 16
#do
#  for j in -0.8 -0.4 -0.02 0 0.02 0.4 0.8 1.2 1.6 2 4 8 16
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_Dfgsm_Ldeepfool.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon $i --train_overshoot $j --train_deepfool_norm_dist l_2 --exp_name 1124_fgsm_deepfool_l2_wors_{$i}_$j"
#  done
#done

#for i in 8
#do
#  for j in 0.02 0 -0.02 -0.4 -0.8
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon $i --train_overshoot $j --train_deepfool_norm_dist l_inf --exp_name 1125_deepfool_linf_wors_{$i}_$j"
#  done
#done

#for i in 8
#do
#  for j in 0.02 64
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon $i --train_overshoot $j --train_deepfool_norm_dist l_2 --exp_name 1125_deepfool_l2_wors_{$i}_$j"
#  done
#done

#for i in 8
#do
#  PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#  bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_test.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon $i --train_overshoot 0.02 --train_overshoot_m 2 --train_overshoot_l 200 --train_deepfool_norm_dist l_2 --exp_name 1125_deepfool_l2_wors_{$i}_0.02_2_200"
#done

#for i in 8
#do
#  PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#  bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_test.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon $i --train_overshoot 8 --train_overshoot_m 16 --train_overshoot_l 32 --train_deepfool_norm_dist l_2 --exp_name 1125_deepfool_l2_wors_{$i}_8_16_32"
#done

#for i in 8
#do
#  for j in 0.02 2
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon $i --train_overshoot $j --train_deepfool_norm_dist l_inf --train_deepfool_rs --exp_name 1125_deepfool_linf_rs_{$i}_$j"
#  done
#done

#for i in 8
#do
#  for j in 0 0.02 2 -0.4 -0.02 4 8 0.4 0.8 1.2
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_1127_1.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon $i --train_overshoot $j --train_deepfool_norm_dist l_inf --train_deepfool_rs --exp_name 1127_1_deepfool_linf_rs_{$i}_$j"
#  done
#done

#for i in 8
#do
#  for j in 0 0.02 2 -0.4 -0.02 4 8 0.4 0.8 1.2
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_1127_2.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon $i --train_overshoot $j --train_deepfool_norm_dist l_inf --train_deepfool_rs --exp_name 1127_2_deepfool_linf_rs_{$i}_$j"
#  done
#done

#for i in 8
#do
#  for j in 0 0.02 2 -0.4 -0.02 4 8 0.4 0.8 1.2
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_1127_3.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon $i --train_overshoot $j --train_deepfool_norm_dist l_inf --train_deepfool_rs --exp_name 1127_3_deepfool_linf_rs_{$i}_$j"
#  done
#done

#for i in 8
#do
#  for j in 0 0.02 2 -0.4 -0.02 4 8 16 0.4 0.8 1.2
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_1127_1.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon $i --train_overshoot $j --train_deepfool_norm_dist l_2 --train_deepfool_rs --exp_name 1127_1_deepfool_l2_rs_{$i}_$j"
#  done
#done
#
#
#for i in 8
#do
#  for j in 0 0.02 2 -0.4 -0.02 4 8 16 0.4 0.8 1.2
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_1127_2.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon $i --train_overshoot $j --train_deepfool_norm_dist l_2 --train_deepfool_rs --exp_name 1127_2_deepfool_l2_rs_{$i}_$j"
#  done
#done

#for i in 8
#do
#  for j in 0 0.02 2 -0.4 -0.02 4 8 16 0.4 0.8 1.2
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_1127_3.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon $i --train_overshoot $j --train_deepfool_norm_dist l_2 --train_deepfool_rs --exp_name 1127_3_deepfool_l2_rs_{$i}_$j"
#  done
#done

#for i in 8
#do
#  for j in 0.02 2 4 8 16 32 64 128 -0.02 0 0.4 1.2 -1.2 -0.8 -0.4 256 512
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_1128_1.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon $i --train_overshoot $j --train_deepfool_norm_dist l_2 --train_deepfool_rs --exp_name 1128_1_deepfool_l2_rs_{$i}_$j"
#  done
#done
#
#for i in 8
#do
#  for j in 0.02 2 4 8 16 32 64 128 -0.02 0 0.4 1.2 -1.2 -0.8 -0.4 256 512
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_1128_1.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon $i --train_overshoot $j --train_deepfool_norm_dist l_inf --train_deepfool_rs --exp_name 1128_1_deepfool_linf_rs_{$i}_$j"
#  done
#done
#
#for i in 8
#do
#  for j in 0.02 2 4 8 16 32 64 128 -0.02 0 0.4 1.2 -1.2 -0.8 -0.4 256 512
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_1128_2.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon $i --train_overshoot $j --train_deepfool_norm_dist l_2 --train_deepfool_rs --exp_name 1128_2_deepfool_l2_rs_{$i}_$j"
#  done
#done
#
#for i in 8
#do
#  for j in 0.02 2 4 8 16 32 64 128 -0.02 0 0.4 1.2 -1.2 -0.8 -0.4 256 512
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_1128_2.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon $i --train_overshoot $j --train_deepfool_norm_dist l_inf --train_deepfool_rs --exp_name 1128_2_deepfool_linf_rs_{$i}_$j"
#  done
#done

#for i in 0.02 2 4 8 16 32 64 128 -0.02 0 0.4 1.2 -1.2 -0.8 -0.4 256 512
#do
#  PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#  bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate.py --model PreActResNet18 --epsilon 8 --resumed_model_name 1128_1_deepfool_l2_rs_{8}_$i --exp_name evaluate_1128_1_deepfool_l2_rs_{8}_$i"
#done
#
#
#for i in 0.02 2 4 8 16 32 64 128 -0.02 0 0.4 1.2 -1.2 -0.8 -0.4 256 512
#do
#  PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#  bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate.py --model PreActResNet18 --epsilon 8 --resumed_model_name 1128_1_deepfool_linf_rs_{8}_$i --exp_name evaluate_1128_1_deepfool_linf_rs_{8}_$i"
#done
#
#
#for i in 0.02 2 4 8 16 32 64 128 -0.02 0 0.4 1.2 -1.2 -0.8 -0.4 256 512
#do
#  PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#  bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate.py --model PreActResNet18 --epsilon 8 --resumed_model_name 1128_2_deepfool_l2_rs_{8}_$i --exp_name evaluate_1128_2_deepfool_l2_rs_{8}_$i"
#done
#
#
#for i in 0.02 2 4 8 16 32 64 128 -0.02 0 0.4 1.2 -1.2 -0.8 -0.4 256 512
#do
#  PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#  bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate.py --model PreActResNet18 --epsilon 8 --resumed_model_name 1128_2_deepfool_linf_rs_{8}_$i --exp_name evaluate_1128_2_deepfool_linf_rs_{8}_$i"
#done

########################################################### 30/11/2020-04/12/2020 #########################################################################
#for i in 8
#do
#  for j in 1
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon $i --train_fgsm_ratio $j --exp_name 1201_fgsm_wors_{$i}_$j"
#  done
#done

#for i in 8
#do
#  for j in 0.02
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon $i --train_overshoot $j --exp_name 1201_deepfool_linf_wors_{$i}_$j"
#  done
#done

#for i in 8
#do
#  for j in 0.02
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_Dfgsm_Ldeepfool.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon $i --train_overshoot $j --exp_name 1201_fgsm_deepfool_linf_wors_{$i}_$j"
#  done
#done

#for i in 8
#do
#  for j in 0.02
#  do
#    PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#    bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_Ddeepfool_Lfgsm.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon $i --train_overshoot $j --train_fgsm_ratio 1 --exp_name 1201_deepfool_fgsm_linf_wors_{$i}_$j"
#  done
#done

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy" ;
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate.py --model PreActResNet18 --epsilon 8 --resumed_model_name 1201_fgsm_wors_{8}_1 --exp_name evaluate_1201_fgsm_wors_{8}_1"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate.py --model PreActResNet18 --epsilon 8 --resumed_model_name 1201_deepfool_linf_wors_{8}_0.02 --exp_name evaluate_1201_deepfool_linf_wors_{8}_0.02"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate.py --model PreActResNet18 --epsilon 8 --resumed_model_name 1201_fgsm_deepfool_linf_wors_{8}_0.02 --exp_name evaluate_1201_fgsm_deepfool_linf_wors_{8}_0.02"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate.py --model PreActResNet18 --epsilon 8 --resumed_model_name 1201_deepfool_fgsm_linf_wors_{8}_0.02 --exp_name evaluate_1201_deepfool_fgsm_linf_wors_{8}_0.02"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate.py --model PreActResNet18 --epsilon 8 --resumed_model_name 1201_fgsm_rs_{8}_1 --exp_name evaluate_1201_fgsm_rs_{8}_1"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate.py --model PreActResNet18 --epsilon 8 --resumed_model_name 1201_deepfool_l2_wors_{8}_0.02 --exp_name evaluate_1201_deepfool_l2_wors_{8}_0.02"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate.py --model PreActResNet18 --epsilon 8 --resumed_model_name 1201_fgsm_rs_{8}_1.25 --exp_name evaluate_1201_fgsm_rs_{8}_1.25"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate.py --model PreActResNet18 --epsilon 8 --resumed_model_name 1201_deepfool_l2_wors_{8}_64 --exp_name evaluate_1201_deepfool_l2_wors_{8}_64"

#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python draw_decision_boundary.py --model PreActResNet18 --epochs 6 11 16 61 196 --epsilon 8 --num 20 --resumed_model_name 1201_fgsm_wors_{8}_1 --exp_name draw_1201_fgsm_wors_{8}_1"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python draw_decision_boundary.py --model PreActResNet18 --epochs 6 31 61 196 --epsilon 8 --num 20 --resumed_model_name 1201_deepfool_linf_wors_{8}_0.02 --exp_name draw_1201_deepfool_linf_wors_{8}_0.02"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python draw_decision_boundary.py --model PreActResNet18 --epochs 6 46 51 61 196 --epsilon 8 --num 20 --resumed_model_name 1201_fgsm_deepfool_linf_wors_{8}_0.02 --exp_name draw_1201_fgsm_deepfool_linf_wors_{8}_0.02"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python draw_decision_boundary.py --model PreActResNet18 --epochs 6 41 46 61 196 --epsilon 8 --num 20 --resumed_model_name 1201_deepfool_fgsm_linf_wors_{8}_0.02 --exp_name draw_1201_deepfool_fgsm_linf_wors_{8}_0.02"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python draw_decision_boundary.py --model PreActResNet18 --epochs 6 141 151 196 --epsilon 8 --num 20 --resumed_model_name 1201_fgsm_rs_{8}_1 --exp_name draw_1201_fgsm_rs_{8}_1"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python draw_decision_boundary.py --model PreActResNet18 --epochs 6 61 121 196 --epsilon 8 --num 20 --resumed_model_name 1201_deepfool_l2_wors_{8}_0.02 --exp_name draw_1201_deepfool_l2_wors_{8}_0.02"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python draw_decision_boundary.py --model PreActResNet18 --epochs 86 91 6 196 --epsilon 8 --num 20 --train_fgsm_ratio 1.25 --resumed_model_name 1201_fgsm_rs_{8}_1.25 --exp_name draw_1201_fgsm_rs_{8}_1.25"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python draw_decision_boundary.py --model PreActResNet18 --epochs 111 136 6 196 --epsilon 8 --num 20 --train_overshoot 64 --resumed_model_name 1201_deepfool_l2_wors_{8}_64 --exp_name 1201_deepfool_l2_wors_{8}_64"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python draw_decision_boundary.py --model PreActResNet18 --epochs 41 46 136 141 56 61 66 196 --epsilon 8 --num 20 --train_fgsm_ratio 1.25 --resumed_model_name 1201_fgsm_rs_{8}_1.25_exp2 --exp_name draw_1201_fgsm_rs_{8}_1.25_exp2_1"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python draw_decision_boundary.py --model PreActResNet18 --epochs 36 41 --epsilon 8 --num 20 --train_fgsm_ratio 1.25 --resumed_model_name 1201_fgsm_rs_{8}_1.25_exp3 --exp_name draw_1201_fgsm_rs_{8}_1.25_exp3"

#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --train_random_start --exp_name 1201_fgsm_rs_{8}_1"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1.25 --train_random_start --exp_name 1201_fgsm_rs_{8}_1.25"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1.25 --train_random_start --exp_name 1201_fgsm_rs_{8}_1.25_exp2"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1.25 --train_random_start --exp_name 1201_fgsm_rs_{8}_1.25_exp3"

#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_overshoot 0.02 --train_deepfool_norm_dist l_2 --exp_name 1201_deepfool_l2_wors_{8}_0.02"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_overshoot 64 --train_deepfool_norm_dist l_2 --exp_name 1201_deepfool_l2_wors_{8}_64"

########################################################### 07/12/2020-11/12/2020 #########################################################################
PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_1209.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.5 --exp_name 1209_fgsm_{8}_0.5"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_1209.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.25 --exp_name 1209_fgsm_{8}_0.25"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_1209.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.75 --exp_name 1209_fgsm_{8}_0.75"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_1209.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --exp_name 1209_fgsm_{8}_1"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_1209.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1.5 --exp_name 1209_fgsm_{8}_1.5"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_1209.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1.25 --exp_name 1209_fgsm_{8}_1.25"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_1209.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1.75 --exp_name 1209_fgsm_{8}_1.75"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_1209.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 2 --exp_name 1209_fgsm_{8}_2"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_1209.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_overshoot 2 --train_deepfool_norm_dist l_inf --exp_name 1209_deepfool_linf_{8}_2"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_1209.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_overshoot 8 --train_deepfool_norm_dist l_2 --exp_name 1209_deepfool_l2_{8}_8"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_1209.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_overshoot 16 --train_deepfool_norm_dist l_2 --exp_name 1209_deepfool_l2_{8}_16"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_1209.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_overshoot 0.02 --train_deepfool_norm_dist l_inf --exp_name 1209_deepfool_linf_{8}_0.02"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_1209.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_overshoot 4 --train_deepfool_norm_dist l_inf --exp_name 1209_deepfool_linf_{8}_4"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_1209.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_overshoot 8 --train_deepfool_norm_dist l_inf --exp_name 1209_deepfool_linf_{8}_8"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_1209.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_overshoot 4 --train_deepfool_norm_dist l_2 --exp_name 1209_deepfool_l2_{8}_4"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_1209.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_overshoot 32 --train_deepfool_norm_dist l_2 --exp_name 1209_deepfool_l2_{8}_32"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_1209.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_overshoot 16 --train_deepfool_norm_dist l_inf --exp_name 1209_deepfool_linf_{8}_16"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_1209.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_overshoot 64 --train_deepfool_norm_dist l_2 --exp_name 1209_deepfool_l2_{8}_64"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_1209.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_overshoot 32 --train_deepfool_norm_dist l_inf --exp_name 1209_deepfool_linf_{8}_32"

#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate.py --model PreActResNet18 --epsilon 8 --resumed_model_name 1209_fgsm_{8}_0.5 --exp_name evaluate_1209_fgsm_{8}_0.5"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate.py --model PreActResNet18 --epsilon 8 --resumed_model_name 1209_fgsm_{8}_0.25 --exp_name evaluate_1209_fgsm_{8}_0.25"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate.py --model PreActResNet18 --epsilon 8 --resumed_model_name 1209_fgsm_{8}_0.75 --exp_name evaluate_1209_fgsm_{8}_0.75"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate.py --model PreActResNet18 --epsilon 8 --resumed_model_name 1209_fgsm_{8}_1 --exp_name evaluate_1209_fgsm_{8}_1"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate.py --model PreActResNet18 --epsilon 8 --resumed_model_name 1209_fgsm_{8}_1.5 --exp_name evaluate_1209_fgsm_{8}_1.5"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate.py --model PreActResNet18 --epsilon 8 --resumed_model_name 1209_fgsm_{8}_1.25 --exp_name evaluate_1209_fgsm_{8}_1.25"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate.py --model PreActResNet18 --epsilon 8 --resumed_model_name 1209_fgsm_{8}_1.75 --exp_name evaluate_1209_fgsm_{8}_1.75"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate.py --model PreActResNet18 --epsilon 8 --resumed_model_name 1209_fgsm_{8}_2 --exp_name evaluate_1209_fgsm_{8}_2"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate.py --model PreActResNet18 --epsilon 8 --resumed_model_name 1209_deepfool_linf_{8}_2 --exp_name evaluate_1209_deepfool_linf_{8}_2"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate.py --model PreActResNet18 --epsilon 8 --resumed_model_name 1209_deepfool_l2_{8}_8 --exp_name evaluate_1209_deepfool_l2_{8}_8"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate.py --model PreActResNet18 --epsilon 8 --resumed_model_name 1209_deepfool_l2_{8}_16 --exp_name evaluate_1209_deepfool_l2_{8}_16"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate.py --model PreActResNet18 --epsilon 8 --resumed_model_name 1209_deepfool_linf_{8}_0.02 --exp_name evaluate_1209_deepfool_linf_{8}_0.02"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate.py --model PreActResNet18 --epsilon 8 --resumed_model_name 1209_deepfool_linf_{8}_4 --exp_name evaluate_1209_deepfool_linf_{8}_4"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate.py --model PreActResNet18 --epsilon 8 --resumed_model_name 1209_deepfool_linf_{8}_8 --exp_name evaluate_1209_deepfool_linf_{8}_8"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate.py --model PreActResNet18 --epsilon 8 --resumed_model_name 1209_deepfool_l2_{8}_4 --exp_name evaluate_1209_deepfool_l2_{8}_4"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate.py --model PreActResNet18 --epsilon 8 --resumed_model_name 1209_deepfool_l2_{8}_32 --exp_name evaluate_1209_deepfool_l2_{8}_32"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate.py --model PreActResNet18 --epsilon 8 --resumed_model_name 1209_deepfool_linf_{8}_16 --exp_name evaluate_1209_deepfool_linf_{8}_16"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate.py --model PreActResNet18 --epsilon 8 --resumed_model_name 1209_deepfool_l2_{8}_64 --exp_name evaluate_1209_deepfool_l2_{8}_64"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate.py --model PreActResNet18 --epsilon 8 --resumed_model_name 1209_deepfool_linf_{8}_32 --exp_name evaluate_1209_deepfool_linf_{8}_32"

#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_1209.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 16 --train_fgsm_ratio 0.5 --exp_name 1209_fgsm_{16}_0.5"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_1209.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 16 --train_fgsm_ratio 0.25 --exp_name 1209_fgsm_{16}_0.25"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_1209.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 16 --train_fgsm_ratio 0.75 --exp_name 1209_fgsm_{16}_0.75"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_1209.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 16 --train_fgsm_ratio 1 --exp_name 1209_fgsm_{16}_1"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_1209.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 16 --train_fgsm_ratio 1.5 --exp_name 1209_fgsm_{16}_1.5"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_1209.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 16 --train_fgsm_ratio 1.25 --exp_name 1209_fgsm_{16}_1.25"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_1209.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 16 --train_fgsm_ratio 1.75 --exp_name 1209_fgsm_{16}_1.75"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_1209.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 16 --train_fgsm_ratio 2 --exp_name 1209_fgsm_{16}_2"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_1209.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 16 --train_overshoot 2 --train_deepfool_norm_dist l_inf --exp_name 1209_deepfool_linf_{16}_2"
##bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_1209.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 16 --train_overshoot 8 --train_deepfool_norm_dist l_2 --exp_name 1209_deepfool_l2_{16}_8"
##bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_1209.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 16 --train_overshoot 16 --train_deepfool_norm_dist l_2 --exp_name 1209_deepfool_l2_{16}_16"
##bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_1209.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 16 --train_overshoot 0.02 --train_deepfool_norm_dist l_inf --exp_name 1209_deepfool_linf_{16}_0.02"
##bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_1209.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 16 --train_overshoot 4 --train_deepfool_norm_dist l_inf --exp_name 1209_deepfool_linf_{16}_4"
##bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_1209.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 16 --train_overshoot 8 --train_deepfool_norm_dist l_inf --exp_name 1209_deepfool_linf_{16}_8"
##bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_1209.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 16 --train_overshoot 4 --train_deepfool_norm_dist l_2 --exp_name 1209_deepfool_l2_{16}_4"
##bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_1209.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 16 --train_overshoot 32 --train_deepfool_norm_dist l_2 --exp_name 1209_deepfool_l2_{16}_32"
##bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_1209.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 16 --train_overshoot 16 --train_deepfool_norm_dist l_inf --exp_name 1209_deepfool_linf_{16}_16"
##bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_1209.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 16 --train_overshoot 64 --train_deepfool_norm_dist l_2 --exp_name 1209_deepfool_l2_{16}_64"
##bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_1209.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 16 --train_overshoot 32 --train_deepfool_norm_dist l_inf --exp_name 1209_deepfool_linf_{16}_32"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate.py --model PreActResNet18 --epsilon 16 --resumed_model_name 1209_deepfool_linf_{16}_2 --exp_name evaluate_1209_deepfool_linf_{16}_2"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate.py --model PreActResNet18 --epsilon 16 --resumed_model_name 1209_deepfool_l2_{16}_8 --exp_name evaluate_1209_deepfool_l2_{16}_8"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate.py --model PreActResNet18 --epsilon 16 --resumed_model_name 1209_deepfool_l2_{16}_16 --exp_name evaluate_1209_deepfool_l2_{16}_16"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate.py --model PreActResNet18 --epsilon 16 --resumed_model_name 1209_deepfool_linf_{16}_0.02 --exp_name evaluate_1209_deepfool_linf_{16}_0.02"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate.py --model PreActResNet18 --epsilon 16 --resumed_model_name 1209_deepfool_linf_{16}_4 --exp_name evaluate_1209_deepfool_linf_{16}_4"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate.py --model PreActResNet18 --epsilon 16 --resumed_model_name 1209_deepfool_linf_{16}_8 --exp_name evaluate_1209_deepfool_linf_{16}_8"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate.py --model PreActResNet18 --epsilon 16 --resumed_model_name 1209_deepfool_l2_{16}_4 --exp_name evaluate_1209_deepfool_l2_{16}_4"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate.py --model PreActResNet18 --epsilon 16 --resumed_model_name 1209_deepfool_l2_{16}_32 --exp_name evaluate_1209_deepfool_l2_{16}_32"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate.py --model PreActResNet18 --epsilon 16 --resumed_model_name 1209_deepfool_linf_{16}_16 --exp_name evaluate_1209_deepfool_linf_{16}_16"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate.py --model PreActResNet18 --epsilon 16 --resumed_model_name 1209_deepfool_l2_{16}_64 --exp_name evaluate_1209_deepfool_l2_{16}_64"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate.py --model PreActResNet18 --epsilon 16 --resumed_model_name 1209_deepfool_linf_{16}_32 --exp_name evaluate_1209_deepfool_linf_{16}_32"

#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_1209_1.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_overshoot 0.02 --train_deepfool_norm_dist l_inf --exp_name 1209_1_deepfool_linf_{8}_0.02"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_1209_1.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_overshoot 2 --train_deepfool_norm_dist l_inf --exp_name 1209_1_deepfool_linf_{8}_2"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_1209_1.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_overshoot 4 --train_deepfool_norm_dist l_2 --exp_name 1209_1_deepfool_l2_{8}_4"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_1209_1.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_overshoot 8 --train_deepfool_norm_dist l_2 --exp_name 1209_1_deepfool_l2_{8}_8"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_1209_1.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_overshoot 16 --train_deepfool_norm_dist l_2 --exp_name 1209_1_deepfool_l2_{8}_16"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_1209_1.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 16 --train_overshoot 0.02 --train_deepfool_norm_dist l_inf --exp_name 1209_1_deepfool_linf_{16}_0.02"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_1209_1.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 16 --train_overshoot 2 --train_deepfool_norm_dist l_inf --exp_name 1209_1_deepfool_linf_{16}_2"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_1209_1.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 16 --train_overshoot 4 --train_deepfool_norm_dist l_2 --exp_name 1209_1_deepfool_l2_{16}_4"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_1209_1.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 16 --train_overshoot 8 --train_deepfool_norm_dist l_2 --exp_name 1209_1_deepfool_l2_{16}_8"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_1209_1.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 16 --train_overshoot 16 --train_deepfool_norm_dist l_2 --exp_name 1209_1_deepfool_l2_{16}_16"

#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 16 --train_fgsm_ratio 0.25 --exp_name 1209_fgsm_wors_{16}_0.25"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 16 --train_fgsm_ratio 0.5 --exp_name 1209_fgsm_wors_{16}_0.5"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 16 --train_fgsm_ratio 1 --exp_name 1209_fgsm_wors_{16}_1"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 16 --train_fgsm_ratio 0.25 --train_random_start --exp_name 1209_fgsm_rs_{16}_0.25"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 16 --train_fgsm_ratio 0.5 --train_random_start --exp_name 1209_fgsm_rs_{16}_0.5"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 16 --train_fgsm_ratio 1 --train_random_start --exp_name 1209_fgsm_rs_{16}_1"
#
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 16 --train_overshoot 0.02 --train_deepfool_norm_dist l_inf --exp_name 1209_deepfool_linf_wors_{16}_0.02"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 16 --train_overshoot 0.02 --train_deepfool_norm_dist l_2 --exp_name 1209_deepfool_l2_wors_{16}_0.02"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 16 --train_overshoot 2 --train_deepfool_norm_dist l_2 --exp_name 1209_deepfool_l2_wors_{16}_2"

#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python draw_decision_boundary.py --model PreActResNet18 --epochs 1 --epsilon 8 --num 20 --train_fgsm_ratio 1 --resumed_model_name 1201_fgsm_wors_{8}_1 --exp_name draw_1201_fgsm_wors_{8}_1"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python draw_decision_boundary.py --model PreActResNet18 --epochs 1 185 199 --epsilon 16 --num 20 --train_fgsm_ratio 0.25 --resumed_model_name 1209_fgsm_wors_{16}_0.25 --exp_name draw_1209_fgsm_wors_{16}_0.25"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python draw_decision_boundary.py --model PreActResNet18 --epochs 1 15 13 199 --epsilon 16 --num 20 --train_fgsm_ratio 0.5 --resumed_model_name 1209_fgsm_wors_{16}_0.5 --exp_name draw_1209_fgsm_wors_{16}_0.5"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python draw_decision_boundary.py --model PreActResNet18 --epochs 3 1 199 --epsilon 16 --num 20 --train_fgsm_ratio 1 --resumed_model_name 1209_fgsm_wors_{16}_1 --exp_name draw_1209_fgsm_wors_{16}_1"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python draw_decision_boundary.py --model PreActResNet18 --epochs 5 7 13 101 --epsilon 16 --num 20 --train_fgsm_ratio 1 --resumed_model_name 1209_fgsm_wors_{16}_1 --exp_name draw_1209_fgsm_wors_{16}_1_supply"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python draw_decision_boundary.py --model PreActResNet18 --epochs 199 --epsilon 16 --num 20 --train_fgsm_ratio 0.25 --resumed_model_name 1209_fgsm_rs_{16}_0.25 --exp_name draw_1209_fgsm_rs_{16}_0.25"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python draw_decision_boundary.py --model PreActResNet18 --epochs 199 --epsilon 16 --num 20 --train_fgsm_ratio 0.5 --resumed_model_name 1209_fgsm_rs_{16}_0.5 --exp_name draw_1209_fgsm_rs_{16}_0.5"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python draw_decision_boundary.py --model PreActResNet18 --epochs 13 9 11 199 --epsilon 16 --num 20 --train_fgsm_ratio 1 --resumed_model_name 1209_fgsm_rs_{16}_1 --exp_name draw_1209_fgsm_rs_{16}_1"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python draw_decision_boundary.py --model PreActResNet18 --epochs 15 9 199 --epsilon 16 --num 20 --train_fgsm_ratio 1 --train_overshoot 0.02 --resumed_model_name 1209_deepfool_linf_wors_{16}_0.02 --exp_name draw_1209_deepfool_linf_wors_{16}_0.02"

#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate_diff_fgsm.py --model PreActResNet18 --epsilon 8 --interval 5 --resumed_model_name 1201_fgsm_wors_{8}_1 --exp_name diff_fgsm_1201_fgsm_wors_{8}_1"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate_diff_fgsm.py --model PreActResNet18 --epsilon 8 --interval 5 --resumed_model_name 1201_deepfool_linf_wors_{8}_0.02 --exp_name diff_fgsm_1201_deepfool_linf_wors_{8}_0.02"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate_diff_fgsm.py --model PreActResNet18 --epsilon 8 --interval 5 --resumed_model_name 1201_fgsm_deepfool_linf_wors_{8}_0.02 --exp_name diff_fgsm_1201_fgsm_deepfool_linf_wors_{8}_0.02"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate_diff_fgsm.py --model PreActResNet18 --epsilon 8 --interval 5 --resumed_model_name 1201_deepfool_fgsm_linf_wors_{8}_0.02 --exp_name diff_fgsm_1201_deepfool_fgsm_linf_wors_{8}_0.02"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate_diff_deepfool.py --model PreActResNet18 --epsilon 8 --interval 5 --resumed_model_name 1201_deepfool_linf_wors_{8}_0.02 --exp_name diff_deepfool_1201_deepfool_linf_wors_{8}_0.02"

#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_overshoot 0.02 --train_deepfool_max_iter 2 --train_deepfool_norm_dist l_inf --exp_name 1211_deepfool_linf2_wors_{8}_0.02"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 16 --train_overshoot 0.02 --train_deepfool_max_iter 2 --train_deepfool_norm_dist l_inf --exp_name 1211_deepfool_linf2_wors_{16}_0.02"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_overshoot 0.02 --train_deepfool_max_iter 2 --train_deepfool_norm_dist l_inf --train_deepfool_rs --exp_name 1211_deepfool_linf2_rs_{8}_0.02"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 16 --train_overshoot 0.02 --train_deepfool_max_iter 2 --train_deepfool_norm_dist l_inf --train_deepfool_rs --exp_name 1211_deepfool_linf2_rs_{16}_0.02"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_overshoot 0.02 --train_deepfool_max_iter 1 --train_deepfool_norm_dist l_inf --exp_name 1211_deepfool_linf1_wors_{8}_0.02"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 16 --train_overshoot 0.02 --train_deepfool_max_iter 1 --train_deepfool_norm_dist l_inf --exp_name 1211_deepfool_linf1_wors_{16}_0.02"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_overshoot 0.02 --train_deepfool_max_iter 1 --train_deepfool_norm_dist l_inf --train_deepfool_rs --exp_name 1211_deepfool_linf1_rs_{8}_0.02"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 16 --train_overshoot 0.02 --train_deepfool_max_iter 1 --train_deepfool_norm_dist l_inf --train_deepfool_rs --exp_name 1211_deepfool_linf1_rs_{16}_0.02"

#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --exp_name 1212_fgsm_wors_{8}_1"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python draw_decision_boundary.py --model PreActResNet18 --epochs 14 15 16 17 18 1 100 101 --epsilon 8 --num 20 --train_fgsm_ratio 1 --resumed_model_name 1212_fgsm_wors_{8}_1 --exp_name draw_1212_fgsm_wors_{8}_1"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate_diff_fgsm.py --model PreActResNet18 --epsilon 8 --interval 1 --resumed_model_name 1212_fgsm_wors_{8}_1 --exp_name diff_fgsm_1212_fgsm_wors_{8}_1"

#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_Ddeepfool_Lfgsm.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --exp_name 1212_deepfool_fgsm_wors_{8}_1"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python draw_decision_boundary.py --model PreActResNet18 --epochs 1 45 46 47 48 49 50 --epsilon 8 --num 20 --train_fgsm_ratio 1 --resumed_model_name 1212_deepfool_fgsm_wors_{8}_1 --exp_name draw_1212_deepfool_fgsm_wors_{8}_1"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate_diff_deepfool_fgsm.py --model PreActResNet18 --epsilon 8 --interval 1 --resumed_model_name 1212_deepfool_fgsm_wors_{8}_1 --exp_name diff_deepfool_1212_deepfool_fgsm_wors_{8}_1"

#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --exp_name 1212_deepfool_linf_{8}_0.02"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python draw_decision_boundary.py --model PreActResNet18 --epochs 35 36 37 38 39 40 41 42 43 44 45 46  47 --epsilon 8 --num 20 --train_fgsm_ratio 1 --resumed_model_name 1212_deepfool_linf_{8}_0.02 --exp_name draw_1212_deepfool_linf_{8}_0.02"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate_diff_deepfool.py --model PreActResNet18 --epsilon 8 --interval 1 --resumed_model_name 1212_deepfool_linf_{8}_0.02 --exp_name diff_deepfool_1212_deepfool_linf_{8}_0.02"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate_diff_deepfool_fgsm.py --model PreActResNet18 --epsilon 8 --interval 1 --resumed_model_name 1212_deepfool_linf_{8}_0.02 --exp_name diff_deepfool_fgsm_1212_deepfool_linf_{8}_0.02"

#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_Dfgsm_Ldeepfool.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --exp_name 1212_fgsm_deepfool_wors_{8}_1"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python draw_decision_boundary.py --model PreActResNet18 --epochs 32 33 34 35 36 37 38 39 42 43 --epsilon 8 --num 20 --train_fgsm_ratio 1 --resumed_model_name 1212_fgsm_deepfool_wors_{8}_1 --exp_name draw_1212_fgsm_deepfool_wors_{8}_1"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python draw_decision_boundary.py --model PreActResNet18 --epochs 1 --epsilon 8 --num 20 --train_fgsm_ratio 1 --resumed_model_name 1212_fgsm_deepfool_wors_{8}_1 --exp_name draw_1212_fgsm_deepfool_wors_{8}_1"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate_diff_fgsm_deepfool.py --model PreActResNet18 --epsilon 8 --interval 1 --resumed_model_name 1212_fgsm_deepfool_wors_{8}_1 --exp_name diff_deepfool_1212_fgsm_deepfool_wors_{8}_1"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate_diff_fgsm_deepfool.py --model PreActResNet18 --epsilon 8 --interval 5 --resumed_model_name 1201_fgsm_deepfool_linf_wors_{8}_0.02 --exp_name diff_deepfool_1201_fgsm_deepfool_linf_wors_{8}_0.02"

#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --train_random_start --exp_name 1212_fgsm_rs_{8}_1"

############ 14.12.2020-18.12.2020 ########
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_dynamic.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_overshoot_min 0 --train_overshoot_max 2 --train_deepfool_max_iter 1 --train_deepfool_norm_dist l_inf --exp_name 1215_deepfool_dynamic_linf_wors_{8}_0_2"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python draw_decision_boundary.py --model PreActResNet18 --epochs 34 35 36 37 38 39 40 41 42 --epsilon 8 --num 20 --train_fgsm_ratio 1 --resumed_model_name 1215_deepfool_dynamic_linf_wors_{8}_0_2 --exp_name draw_1215_deepfool_dynamic_linf_wors_{8}_0_2"

#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --exp_name 1215_deepfool_linf_{8}_0.02"

#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate_diff_fgsm.py --model PreActResNet18 --epsilon 8 --interval 1 --resumed_model_name 1212_fgsm_deepfool_wors_{8}_1 --exp_name diff_fgsm_1212_fgsm_deepfool_wors_{8}_1"

#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_Dfgsm_Ldeepfool.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --fgsm_ratio 1 --exp_name 1217_fgsm1_deepfool_linf_{8}_0.02"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python draw_decision_boundary.py --model PreActResNet18 --epochs 42 41 38 39 40 --epsilon 8 --num 20 --train_fgsm_ratio 1 --resumed_model_name 1217_fgsm1_deepfool_linf_{8}_0.02 --exp_name draw_1217_fgsm1_deepfool_linf_{8}_0.02"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_Dfgsm_Ldeepfool.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --fgsm_ratio 0.75 --exp_name 1217_fgsm0.75_deepfool_linf_{8}_0.02"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_Dfgsm_Ldeepfool.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --fgsm_ratio 0.5 --exp_name 1217_fgsm0.5_deepfool_linf_{8}_0.02"

####### add l2 regularization
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_Dfgsm_Ldeepfool.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --fgsm_ratio 1 --l2 0.1 --exp_name 1217_fgsm1_deepfool_linf_{8}_0.02_0.1"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_Dfgsm_Ldeepfool.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --fgsm_ratio 1 --l2 0.01 --exp_name 1217_fgsm1_deepfool_linf_{8}_0.02_0.01"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_Dfgsm_Ldeepfool.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --fgsm_ratio 1 --l2 0.001 --exp_name 1217_fgsm1_deepfool_linf_{8}_0.02_0.001"

#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_reg.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --delta_lambda 1000 --exp_name 1217_fgsm_wors_{8}_1_1000"

#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.25 --exp_name 1219_fgsm_wors_{8}_0.25"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.5 --exp_name 1219_fgsm_wors_{8}_0.5"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.75 --exp_name 1219_fgsm_wors_{8}_0.75"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --exp_name 1219_fgsm_wors_{8}_1"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate_diff_fgsm.py --model PreActResNet18 --epsilon 8 --interval 1 --resumed_model_name 1219_fgsm_wors_{8}_0.25 --exp_name diff_fgsm_1219_fgsm_wors_{8}_0.25"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate_diff_fgsm.py --model PreActResNet18 --epsilon 8 --interval 1 --resumed_model_name 1219_fgsm_wors_{8}_0.5 --exp_name diff_fgsm_1219_fgsm_wors_{8}_0.5"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate_diff_fgsm.py --model PreActResNet18 --epsilon 8 --interval 1 --resumed_model_name 1219_fgsm_wors_{8}_0.75 --exp_name diff_fgsm_1219_fgsm_wors_{8}_0.75"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate_diff_fgsm.py --model PreActResNet18 --epsilon 8 --interval 1 --resumed_model_name 1219_fgsm_wors_{8}_1 --exp_name diff_fgsm_1219_fgsm_wors_{8}_1"

#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_Dfgsm_Ldeepfool.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --clamp_ratio 0.25 --train_overshoot 0.02 --exp_name 1219_fgsm0.25_deepfool_linf_{8}_0.02"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_Dfgsm_Ldeepfool.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --clamp_ratio 0.5 --train_overshoot 0.02 --exp_name 1219_fgsm0.5_deepfool_linf_{8}_0.02"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_Dfgsm_Ldeepfool.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --clamp_ratio 0.75 --train_overshoot 0.02 --exp_name 1219_fgsm0.75_deepfool_linf_{8}_0.02"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_Dfgsm_Ldeepfool.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --clamp_ratio 1 --train_overshoot 0.02 --exp_name 1219_fgsm1_deepfool_linf_{8}_0.02"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_Dfgsm_Ldeepfool.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --clamp_ratio 0.25 --train_overshoot 2 --exp_name 1219_fgsm0.25_deepfool_linf_{8}_2"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_Dfgsm_Ldeepfool.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --clamp_ratio 0.5 --train_overshoot 2 --exp_name 1219_fgsm0.5_deepfool_linf_{8}_2"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_Dfgsm_Ldeepfool.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --clamp_ratio 0.75 --train_overshoot 2 --exp_name 1219_fgsm0.75_deepfool_linf_{8}_2"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_Dfgsm_Ldeepfool.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --clamp_ratio 1 --train_overshoot 2 --exp_name 1219_fgsm1_deepfool_linf_{8}_2"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate_diff_fgsm.py --model PreActResNet18 --epsilon 8 --interval 1 --resumed_model_name 1219_fgsm0.25_deepfool_linf_{8}_0.02 --exp_name diff_fgsm_1219_fgsm0.25_deepfool_linf_{8}_0.02"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate_diff_fgsm.py --model PreActResNet18 --epsilon 8 --interval 1 --resumed_model_name 1219_fgsm0.5_deepfool_linf_{8}_0.02 --exp_name diff_fgsm_1219_fgsm0.5_deepfool_linf_{8}_0.02"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate_diff_fgsm.py --model PreActResNet18 --epsilon 8 --interval 1 --resumed_model_name 1219_fgsm0.75_deepfool_linf_{8}_0.02 --exp_name diff_fgsm_1219_fgsm0.75_deepfool_linf_{8}_0.02"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate_diff_fgsm.py --model PreActResNet18 --epsilon 8 --interval 1 --resumed_model_name 1219_fgsm1_deepfool_linf_{8}_0.02 --exp_name diff_fgsm_1219_fgsm1_deepfool_linf_{8}_0.02"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate_diff_fgsm.py --model PreActResNet18 --epsilon 8 --interval 1 --resumed_model_name 1219_fgsm0.25_deepfool_linf_{8}_2 --exp_name diff_fgsm_1219_fgsm0.25_deepfool_linf_{8}_2"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate_diff_fgsm.py --model PreActResNet18 --epsilon 8 --interval 1 --resumed_model_name 1219_fgsm0.5_deepfool_linf_{8}_2 --exp_name diff_fgsm_1219_fgsm0.5_deepfool_linf_{8}_2"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate_diff_fgsm.py --model PreActResNet18 --epsilon 8 --interval 1 --resumed_model_name 1219_fgsm0.75_deepfool_linf_{8}_2 --exp_name diff_fgsm_1219_fgsm0.75_deepfool_linf_{8}_2"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate_diff_fgsm.py --model PreActResNet18 --epsilon 8 --interval 1 --resumed_model_name 1219_fgsm1_deepfool_linf_{8}_2 --exp_name diff_fgsm_1219_fgsm1_deepfool_linf_{8}_2"

#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_reg.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --delta_lambda 1000 --exp_name 1219_fgsm_wors_{8}_1_input1000"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_reg_perturbed.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --delta_lambda 1000 --exp_name 1219_fgsm_wors_{8}_1_perturbed1000"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_sep.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --exp_name 1219_fgsm_sep_wors_{8}_1"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate_diff_fgsm.py --model PreActResNet18 --epsilon 8 --interval 1 --resumed_model_name 1219_fgsm_sep_wors_{8}_1 --exp_name diff_fgsm_1219_fgsm_sep_wors_{8}_1"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_Dfgsm_Ldeepfool_sep.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --clamp_ratio 1 --train_overshoot 0.02 --exp_name 1219_fgsm1_deepfool_sep_linf_{8}_0.02"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate_diff_fgsm_deepfool.py --model PreActResNet18 --epsilon 8 --interval 1 --clamp_ratio 1 --resumed_model_name 1219_fgsm1_deepfool_sep_linf_{8}_0.02 --exp_name diff_fgsm_1219_fgsm1_deepfool_sep_linf_{8}_0.02"

#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.125 --exp_name 1225_fgsm_wors_{8}_0.125"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.25 --exp_name 1225_fgsm_wors_{8}_0.25"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.375 --exp_name 1225_fgsm_wors_{8}_0.375"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.5 --exp_name 1225_fgsm_wors_{8}_0.5"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.625 --exp_name 1225_fgsm_wors_{8}_0.625"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.75 --exp_name 1225_fgsm_wors_{8}_0.75"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.875 --exp_name 1225_fgsm_wors_{8}_0.875"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --exp_name 1225_fgsm_wors_{8}_1"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate_diff_fgsm.py --model PreActResNet18 --epsilon 8 --interval 1 --train_fgsm_ratio 1 --resumed_model_name 1225_fgsm_wors_{8}_1 --exp_name diff_fgsm_1225_fgsm_wors_{8}_1"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --exp_name 1225_fgsm_wors_{8}_1_exp1"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate_diff_fgsm.py --model PreActResNet18 --epsilon 8 --interval 1 --train_fgsm_ratio 1 --resumed_model_name 1225_fgsm_wors_{8}_1_exp1 --exp_name diff_fgsm_1225_fgsm_wors_{8}_1_exp1"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate_diff_deepfool_fgsm.py --model PreActResNet18 --epsilon 8 --interval 1 --resumed_model_name 1225_fgsm_wors_{8}_1_exp1 --exp_name diff_df_fgsm_1225_fgsm_wors_{8}_1_exp1"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python draw_decision_boundary.py --model PreActResNet18 --epochs 15 16 17 18 19 20 31 --epsilon 8 --num 20 --train_fgsm_ratio 1 --resumed_model_name 1225_fgsm_wors_{8}_1_exp1 --exp_name draw_1225_fgsm_wors_{8}_1_exp1"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_batch.py --model PreActResNet18 --num_epochs 5 --epsilon 8 --train_fgsm_ratio 1 --resumed_model_name 1225_fgsm_wors_{8}_1_exp1_14.pth --exp_name batch_1225_fgsm_wors_{8}_1_exp1"

#
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.125 --train_random_start --exp_name 1225_fgsm_rs_{8}_0.125"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.25 --train_random_start --exp_name 1225_fgsm_rs_{8}_0.25"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.375 --train_random_start --exp_name 1225_fgsm_rs_{8}_0.375"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.5 --train_random_start --exp_name 1225_fgsm_rs_{8}_0.5"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.625 --train_random_start --exp_name 1225_fgsm_rs_{8}_0.625"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.75 --train_random_start --exp_name 1225_fgsm_rs_{8}_0.75"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.875 --train_random_start --exp_name 1225_fgsm_rs_{8}_0.875"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --train_random_start --exp_name 1225_fgsm_rs_{8}_1"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1.125 --train_random_start --exp_name 1225_fgsm_rs_{8}_1.125"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1.25 --train_random_start --exp_name 1225_fgsm_rs_{8}_1.25"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1.375 --train_random_start --exp_name 1225_fgsm_rs_{8}_1.375"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1.5 --train_random_start --exp_name 1225_fgsm_rs_{8}_1.5"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1.625 --train_random_start --exp_name 1225_fgsm_rs_{8}_1.625"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1.75 --train_random_start --exp_name 1225_fgsm_rs_{8}_1.75"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1.875 --train_random_start --exp_name 1225_fgsm_rs_{8}_1.875"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 2 --train_random_start --exp_name 1225_fgsm_rs_{8}_2"
#
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --clamp 0.125 --exp_name 1225_deepfool_linf_{8}_0.02_0.125"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --clamp 0.25 --exp_name 1225_deepfool_linf_{8}_0.02_0.25"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --clamp 0.375 --exp_name 1225_deepfool_linf_{8}_0.02_0.375"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --clamp 0.5 --exp_name 1225_deepfool_linf_{8}_0.02_0.5"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --clamp 0.625 --exp_name 1225_deepfool_linf_{8}_0.02_0.625"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --clamp 0.75 --exp_name 1225_deepfool_linf_{8}_0.02_0.75"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --clamp 0.875 --exp_name 1225_deepfool_linf_{8}_0.02_0.875"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --clamp 1 --exp_name 1225_deepfool_linf_{8}_0.02_1"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate_diff_deepfool.py --model PreActResNet18 --epsilon 8 --interval 1 --clamp 1 --resumed_model_name 1225_deepfool_linf_{8}_0.02_1 --exp_name diff_fgsm_1225_deepfool_linf_{8}_0.02_1"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate_diff_deepfool_fgsm.py --model PreActResNet18 --epsilon 8 --interval 1 --resumed_model_name 1225_deepfool_linf_{8}_0.02_1 --exp_name diff_df_fgsm_1225_deepfool_linf_{8}_0.02_1"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate_diff_fgsm.py --model PreActResNet18 --epsilon 8 --interval 1 --train_fgsm_ratio 1 --resumed_model_name 1225_deepfool_linf_{8}_0.02_1 --exp_name diff_fgsm_fgsm_1225_deepfool_linf_{8}_0.02_1"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python draw_decision_boundary.py --model PreActResNet18 --epochs 49 51 53 62 --epsilon 8 --num 20 --train_fgsm_ratio 1 --resumed_model_name 1225_deepfool_linf_{8}_0.02_1 --exp_name draw_1225_deepfool_linf_{8}_0.02_1"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --clamp 1 --train_deepfool_rs --exp_name 1225_deepfool_rs_linf_{8}_0.02_1"

#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_sep.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --exp_name 1225_fgsm_wors_{8}_1_sep"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_pgd.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_pgd_ratio 0.5 --exp_name 1225_pgd2_wors_{8}_0.5"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_pgd.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_pgd_ratio 1 --exp_name 1225_pgd2_wors_{8}_1"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --clamp 1 --train_deepfool_rs --exp_name 1225_deepfool_rs_linf_{8}_0.02_1"

#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_rs_test1.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --clamp 1 --exp_name 0103_deepfool_linf_test1_{8}_0.02_1"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_rs_test2.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --clamp 1 --exp_name 0103_deepfool_linf_test2_{8}_0.02_1"

#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_extend.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --train_random_start --exp_name 0103_fgsm_rs_{8}_1_extend"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_extend.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --train_random_start --ratio 0.95 --exp_name 0103_fgsm_rs_{8}_1_extend_0.95"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_extend.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --train_random_start --ratio 0.9 --exp_name 0103_fgsm_rs_{8}_1_extend_0.9"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_extend.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --train_random_start --ratio 0.875 --exp_name 0103_fgsm_rs_{8}_1_extend_0.875"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_extend.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --train_random_start --ratio 0.85 --exp_name 0103_fgsm_rs_{8}_1_extend_0.85"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_extend.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.875 --train_random_start --ratio 1 --exp_name 0103_fgsm_rs_{8}_0.875_extend_1"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_extend.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.75 --train_random_start --ratio 1 --exp_name 0103_fgsm_rs_{8}_0.75_extend_1"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_extend.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.625 --train_random_start --ratio 1 --exp_name 0103_fgsm_rs_{8}_0.625_extend_1"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_extend.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.5 --train_random_start --ratio 1 --exp_name 0103_fgsm_rs_{8}_0.5_extend_1"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_extend.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.375 --train_random_start --ratio 1 --exp_name 0103_fgsm_rs_{8}_0.375_extend_1"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_extend_1.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --train_random_start --ratio 1.25 --exp_name 0103_fgsm_rs_{8}_1_extend_1.25"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_extend_1.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --train_random_start --ratio 1.125 --exp_name 0103_fgsm_rs_{8}_1_extend_1.125"

#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_extend.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 16 --train_fgsm_ratio 0.125 --train_random_start --ratio 1 --exp_name 0108_fgsm_rs_{16}_0.125_extend_1"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_extend.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 16 --train_fgsm_ratio 0.25 --train_random_start --ratio 1 --exp_name 0108_fgsm_rs_{16}_0.25_extend_1"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_extend.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 16 --train_fgsm_ratio 0.375 --train_random_start --ratio 1 --exp_name 0108_fgsm_rs_{16}_0.375_extend_1"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_extend.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 16 --train_fgsm_ratio 0.5 --train_random_start --ratio 1 --exp_name 0108_fgsm_rs_{16}_0.5_extend_1"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_extend.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 16 --train_fgsm_ratio 0.625 --train_random_start --ratio 1 --exp_name 0108_fgsm_rs_{16}_0.625_extend_1"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_extend.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 16 --train_fgsm_ratio 0.75 --train_random_start --ratio 1 --exp_name 0108_fgsm_rs_{16}_0.75_extend_1"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_extend.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 16 --train_fgsm_ratio 0.875 --train_random_start --ratio 1 --exp_name 0108_fgsm_rs_{16}_0.875_extend_1"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_extend.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 16 --train_fgsm_ratio 1 --train_random_start --ratio 1 --exp_name 0108_fgsm_rs_{16}_1_extend_1"

#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_rs_test1.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --clamp 1 --finetune --resumed_model_name 1225_deepfool_linf_{8}_0.02_1_50.pth --exp_name 0108_resume_deepfool_linf_test1_{8}_0.02_1"
##bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_rs_test2.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --clamp 1 --finetune --resumed_model_name 1225_deepfool_linf_{8}_0.02_1_50.pth --exp_name 0108_resume_deepfool_linf_test2_{8}_0.02_1"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_rs_test1.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --clamp 1 --finetune --resumed_model_name 1225_deepfool_linf_{8}_0.02_1_49.pth --exp_name 0108_resume49_deepfool_linf_test1_{8}_0.02_1"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_rs_test2.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --clamp 1 --finetune --resumed_model_name 1225_deepfool_linf_{8}_0.02_1_49.pth --exp_name 0108_resume49_deepfool_linf_test2_{8}_0.02_1"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_rs_test1.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --clamp 1 --finetune --resumed_model_name 1225_deepfool_linf_{8}_0.02_1_48.pth --exp_name 0108_resume48_deepfool_linf_test1_{8}_0.02_1"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_rs_test2.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --clamp 1 --finetune --resumed_model_name 1225_deepfool_linf_{8}_0.02_1_48.pth --exp_name 0108_resume48_deepfool_linf_test2_{8}_0.02_1"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_rs_test1.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --clamp 1 --finetune --resumed_model_name 1225_deepfool_linf_{8}_0.02_1_60.pth --exp_name 0108_resume60_deepfool_linf_test1_{8}_0.02_1"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_rs_test2.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --clamp 1 --finetune --resumed_model_name 1225_deepfool_linf_{8}_0.02_1_60.pth --exp_name 0108_resume60_deepfool_linf_test2_{8}_0.02_1"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_rs_test1.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --clamp 1 --finetune --resumed_model_name 1225_deepfool_linf_{8}_0.02_1_70.pth --exp_name 0108_resume70_deepfool_linf_test1_{8}_0.02_1"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool_rs_test2.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --clamp 1 --finetune --resumed_model_name 1225_deepfool_linf_{8}_0.02_1_70.pth --exp_name 0108_resume70_deepfool_linf_test2_{8}_0.02_1"

#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_test1.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.125 --exp_name 1225_fgsm_test1_{8}_0.125"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_test1.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.25 --exp_name 1225_fgsm_test1_{8}_0.25"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_test1.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.375 --exp_name 1225_fgsm_test1_{8}_0.375"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_test1.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.5 --exp_name 1225_fgsm_test1_{8}_0.5"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_test1.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.625 --exp_name 1225_fgsm_test1_{8}_0.625"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_test1.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.75 --exp_name 1225_fgsm_test1_{8}_0.75"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_test1.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.875 --exp_name 1225_fgsm_test1_{8}_0.875"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_test1.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --exp_name 1225_fgsm_test1_{8}_1"
#
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_test2.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.125 --exp_name 1225_fgsm_test2_{8}_0.125"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_test2.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.25 --exp_name 1225_fgsm_test2_{8}_0.25"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_test2.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.375 --exp_name 1225_fgsm_test2_{8}_0.375"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_test2.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.5 --exp_name 1225_fgsm_test2_{8}_0.5"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_test2.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.625 --exp_name 1225_fgsm_test2_{8}_0.625"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_test2.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.75 --exp_name 1225_fgsm_test2_{8}_0.75"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_test2.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.875 --exp_name 1225_fgsm_test2_{8}_0.875"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_test2.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --exp_name 1225_fgsm_test2_{8}_1"

#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate_linear_app.py --model PreActResNet18 --epsilon 8 --interval 1 --train_fgsm_ratio 1 --resumed_model_name 1225_fgsm_wors_{8}_1_exp1 --exp_name app_1225_fgsm_wors_{8}_1_exp1"

#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_test1_and_test2.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.125 --exp_name 1225_fgsm_test12_{8}_0.125"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_test1_and_test2.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.25 --exp_name 1225_fgsm_test12_{8}_0.25"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_test1_and_test2.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.375 --exp_name 1225_fgsm_test12_{8}_0.375"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_test1_and_test2.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.5 --exp_name 1225_fgsm_test12_{8}_0.5"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_test1_and_test2.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.625 --exp_name 1225_fgsm_test12_{8}_0.625"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_test1_and_test2.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.75 --exp_name 1225_fgsm_test12_{8}_0.75"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_test1_and_test2.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.875 --exp_name 1225_fgsm_test12_{8}_0.875"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_test1_and_test2.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --exp_name 1225_fgsm_test12_{8}_1"

#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --train_random_start --exp_name 0109_fgsm_rs_{8}_1"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.875 --train_random_start --exp_name 0109_fgsm_rs_{8}_0.875"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.75 --train_random_start --exp_name 0109_fgsm_rs_{8}_0.75"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.625 --train_random_start --exp_name 0109_fgsm_rs_{8}_0.625"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1.125 --train_random_start --exp_name 0109_fgsm_rs_{8}_1.125"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1.25 --train_random_start --exp_name 0109_fgsm_rs_{8}_1.25"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1.375 --train_random_start --exp_name 0109_fgsm_rs_{8}_1.375"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1.5 --train_random_start --exp_name 0109_fgsm_rs_{8}_1.5"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1.625 --train_random_start --exp_name 0109_fgsm_rs_{8}_1.625"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1.75 --train_random_start --exp_name 0109_fgsm_rs_{8}_1.75"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1.875 --train_random_start --exp_name 0109_fgsm_rs_{8}_1.875"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 2 --train_random_start --exp_name 0109_fgsm_rs_{8}_2"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 30 --lr_schedule cyclic --epsilon 8 --train_fgsm_ratio 1.125 --train_random_start --exp_name 0109_fgsm_rs_cyclic_{8}_1.125"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 30 --lr_schedule cyclic --epsilon 8 --train_fgsm_ratio 1.25 --train_random_start --exp_name 0109_fgsm_rs_cyclic_{8}_1.25"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 30 --lr_schedule cyclic --epsilon 8 --train_fgsm_ratio 1.5 --train_random_start --exp_name 0109_fgsm_rs_cyclic_{8}_1.5"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --train_random_start --exp_name 0110_fgsm_rs_{8}_1"

#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 16 --train_fgsm_ratio 1 --train_random_start --exp_name 0109_fgsm_rs_{16}_1"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 16 --train_fgsm_ratio 1 --train_random_start --exp_name 0109_fgsm_rs_{16}_1_exp1"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 16 --train_fgsm_ratio 1 --train_random_start --exp_name 0109_fgsm_rs_{16}_1_exp2"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 16 --train_fgsm_ratio 1 --train_random_start --exp_name 0109_fgsm_rs_{16}_1_exp3"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 16 --train_fgsm_ratio 0.875 --train_random_start --exp_name 0109_fgsm_rs_{16}_0.875" ############没有clamp
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 16 --train_fgsm_ratio 0.8125 --train_random_start --exp_name 0109_fgsm_rs_{16}_0.8125"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 16 --train_fgsm_ratio 0.75 --train_random_start --exp_name 0109_fgsm_rs_{16}_0.75"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 16 --train_fgsm_ratio 0.625 --train_random_start --exp_name 0109_fgsm_rs_{16}_0.625"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 16 --train_fgsm_ratio 0.5 --train_random_start --exp_name 0109_fgsm_rs_{16}_0.5"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_test1.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --exp_name 0109_fgsm_test1_{8}_1"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 30 --lr_schedule cyclic --epsilon 16 --train_fgsm_ratio 0.75 --train_random_start --exp_name 0109_fgsm_rs_cyclic_{16}_0.75"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 30 --lr_schedule cyclic --epsilon 16 --train_fgsm_ratio 0.875 --train_random_start --exp_name 0109_fgsm_rs_cyclic_{16}_0.875"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 30 --lr_schedule cyclic --epsilon 16 --train_fgsm_ratio 1 --train_random_start --exp_name 0109_fgsm_rs_cyclic_{16}_1"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 16 --train_fgsm_ratio 1 --train_random_start --exp_name 0110_fgsm_rs_{16}_1"   ######## 随机变量在 boundary 上
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 16 --train_fgsm_ratio 0.875 --train_random_start --exp_name 0110_fgsm_rs_{16}_0.875"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 16 --train_fgsm_ratio 0.75 --train_random_start --exp_name 0110_fgsm_rs_{16}_0.75"

#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_0109.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --train_random_start --exp_name 0109_fgsm_rs_ratio_{8}_1"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_0109.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 2 --train_random_start --exp_name 0109_fgsm_rs_ratio_{8}_2"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_0109.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 16 --train_fgsm_ratio 1 --train_random_start --exp_name 0109_fgsm_rs_ratio_{16}_1"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_0109.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 16 --train_fgsm_ratio 2 --train_random_start --exp_name 0109_fgsm_rs_ratio_{16}_2"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_0109.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 16 --train_fgsm_ratio 0.5 --train_random_start --exp_name 0109_fgsm_rs_ratio_{16}_0.5"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_0109.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 16 --train_fgsm_ratio 1.5 --train_random_start --exp_name 0109_fgsm_rs_ratio_{16}_1.5"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_0109.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 16 --train_fgsm_ratio 1.75 --train_random_start --exp_name 0109_fgsm_rs_ratio_{16}_1.75"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_0109.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 16 --train_fgsm_ratio 1.25 --train_random_start --exp_name 0109_fgsm_rs_ratio_{16}_1.25"

#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_recover.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --adjust_fgsm_alpha --exp_name 0110_fgsm_wors_adjust_{8}_1"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_recover.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --adjust_fgsm_alpha --exp_name 0110_fgsm_wors_adjust_{8}_1_exp1"
#
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_test1_without_project.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.125 --exp_name 0113_fgsm_test1_wo_project_{8}_0.125"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_test1_without_project.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.25 --exp_name 0113_fgsm_test1_wo_project_{8}_0.25"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_test1_without_project.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.375 --exp_name 0113_fgsm_test1_wo_project_{8}_0.375"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_test1_without_project.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.5 --exp_name 0113_fgsm_test1_wo_project_{8}_0.5"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_test1_without_project.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.625 --exp_name 0113_fgsm_test1_wo_project_{8}_0.625"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_test1_without_project.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.75 --exp_name 0113_fgsm_test1_wo_project_{8}_0.75"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_test1_without_project.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.875 --exp_name 0113_fgsm_test1_wo_project_{8}_0.875"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_test1_without_project.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --exp_name 0113_fgsm_test1_wo_project_{8}_1"
#
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_test1_and_test2_without_project.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.125 --exp_name 0113_fgsm_test12_wo_project_{8}_0.125"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_test1_and_test2_without_project.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.25 --exp_name 0113_fgsm_test12_wo_project_{8}_0.25"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_test1_and_test2_without_project.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.375 --exp_name 0113_fgsm_test12_wo_project_{8}_0.375"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_test1_and_test2_without_project.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.5 --exp_name 0113_fgsm_test12_wo_project_{8}_0.5"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_test1_and_test2_without_project.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.625 --exp_name 0113_fgsm_test12_wo_project_{8}_0.625"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_test1_and_test2_without_project.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.75 --exp_name 0113_fgsm_test12_wo_project_{8}_0.75"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_test1_and_test2_without_project.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.875 --exp_name 0113_fgsm_test12_wo_project_{8}_0.875"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_test1_and_test2_without_project.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --exp_name 0113_fgsm_test12_wo_project_{8}_1"

#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_remove_all_project.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.125 --train_random_start --exp_name 0113_fgsm_rs_remove_all_project_{8}_0.125"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_remove_all_project.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.25 --train_random_start --exp_name 0113_fgsm_rs_remove_all_project_{8}_0.25"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_remove_all_project.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.375 --train_random_start --exp_name 0113_fgsm_rs_remove_all_project_{8}_0.375"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_remove_all_project.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.5 --train_random_start --exp_name 0113_fgsm_rs_remove_all_project_{8}_0.5"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_remove_all_project.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.625 --train_random_start --exp_name 0113_fgsm_rs_remove_all_project_{8}_0.625"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_remove_all_project.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.75 --train_random_start --exp_name 0113_fgsm_rs_remove_all_project_{8}_0.75"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_remove_all_project.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.875 --train_random_start --exp_name 0113_fgsm_rs_remove_all_project_{8}_0.875"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_remove_all_project.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --train_random_start --exp_name 0113_fgsm_rs_remove_all_project_{8}_1"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_remove_all_project.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1.125 --train_random_start --exp_name 0113_fgsm_rs_remove_all_project_{8}_1.125"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_remove_all_project.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1.25 --train_random_start --exp_name 0113_fgsm_rs_remove_all_project_{8}_1.25"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_remove_all_project.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1.375 --train_random_start --exp_name 0113_fgsm_rs_remove_all_project_{8}_1.375"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_remove_all_project.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1.5 --train_random_start --exp_name 0113_fgsm_rs_remove_all_project_{8}_1.5"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_remove_all_project.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1.625 --train_random_start --exp_name 0113_fgsm_rs_remove_all_project_{8}_1.625"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_remove_all_project.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1.75 --train_random_start --exp_name 0113_fgsm_rs_remove_all_project_{8}_1.75"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_remove_all_project.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1.875 --train_random_start --exp_name 0113_fgsm_rs_remove_all_project_{8}_1.875"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_remove_all_project.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 2 --train_random_start --exp_name 0113_fgsm_rs_remove_all_project_{8}_2"

#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_test2_half.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.25 --exp_name 0113_fgsm_test2_half_{8}_0.25"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_test2_half.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.5 --exp_name 0113_fgsm_test2_half_{8}_0.5"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_test2_half.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.75 --exp_name 0113_fgsm_test2_half_{8}_0.75"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_test2_half.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --exp_name 0113_fgsm_test2_half_{8}_1"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_test2_half.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.125 --exp_name 0113_fgsm_test2_half_{8}_0.125"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_test2_half.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.375 --exp_name 0113_fgsm_test2_half_{8}_0.375"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_test2_half.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.625 --exp_name 0113_fgsm_test2_half_{8}_0.625"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_test2_half.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.875 --exp_name 0113_fgsm_test2_half_{8}_0.875"

#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate_loss_diff.py --model PreActResNet18 --epsilon 8 --interval 1 --train_fgsm_ratio 0.875 --resumed_model_name 1225_fgsm_wors_{8}_0.875 --exp_name 0113_diff_1225_fgsm_wors_{8}_0.875"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate_loss_diff.py --model PreActResNet18 --epsilon 8 --interval 1 --train_fgsm_ratio 1 --resumed_model_name 1225_fgsm_wors_{8}_1_exp1 --exp_name 0113_diff_1225_fgsm_wors_{8}_1_exp1"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate_loss_diff.py --model PreActResNet18 --epsilon 8 --interval 1 --train_fgsm_ratio 1 --resumed_model_name 1225_fgsm_rs_{8}_1 --exp_name 0113_diff_1225_fgsm_rs_{8}_1"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate_loss_diff.py --model PreActResNet18 --epsilon 8 --interval 1 --train_fgsm_ratio 1.25 --resumed_model_name 1225_fgsm_rs_{8}_1.25 --exp_name 0113_diff_1225_fgsm_rs_{8}_1.25"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate_loss_diff.py --model PreActResNet18 --epsilon 8 --interval 1 --train_fgsm_ratio 1 --resumed_model_name 0109_fgsm_rs_{8}_1 --exp_name 0113_diff_0109_fgsm_rs_{8}_1"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate_loss_diff.py --model PreActResNet18 --epsilon 8 --interval 1 --train_fgsm_ratio 1.25 --resumed_model_name 0109_fgsm_rs_{8}_1.25 --exp_name 0113_diff_0109_fgsm_rs_{8}_1.25"

#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate_loss_diff.py --model PreActResNet18 --epsilon 8 --interval 1 --train_fgsm_ratio 0.875 --resumed_model_name 1225_fgsm_wors_{8}_0.875 --exp_name 0113_diff_test_eval_1225_fgsm_wors_{8}_0.875_train"

#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate_loss_diff.py --model PreActResNet18 --epsilon 8 --interval 1 --train_fgsm_ratio 0.875 --mode train --resumed_model_name 1225_fgsm_wors_{8}_0.875 --exp_name 0113_diff_test_train_1225_fgsm_wors_{8}_0.875"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate_loss_diff.py --model PreActResNet18 --epsilon 8 --interval 1 --train_fgsm_ratio 1 --mode train --resumed_model_name 1225_fgsm_wors_{8}_1_exp1 --exp_name 0113_diff_test_train_1225_fgsm_wors_{8}_1_exp1"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate_loss_diff.py --model PreActResNet18 --epsilon 8 --interval 1 --train_fgsm_ratio 1 --mode train --resumed_model_name 1225_fgsm_rs_{8}_1 --exp_name 0113_diff_test_train_1225_fgsm_rs_{8}_1"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate_loss_diff.py --model PreActResNet18 --epsilon 8 --interval 1 --train_fgsm_ratio 1.25 --mode train --resumed_model_name 1225_fgsm_rs_{8}_1.25 --exp_name 0113_diff_test_train_1225_fgsm_rs_{8}_1.25"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate_loss_diff.py --model PreActResNet18 --epsilon 8 --interval 1 --train_fgsm_ratio 1 --mode train --resumed_model_name 0109_fgsm_rs_{8}_1 --exp_name 0113_diff_test_train_0109_fgsm_rs_{8}_1"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate_loss_diff.py --model PreActResNet18 --epsilon 8 --interval 1 --train_fgsm_ratio 1.25 --mode train --resumed_model_name 0109_fgsm_rs_{8}_1.25 --exp_name 0113_diff_test_train_0109_fgsm_rs_{8}_1.25"

#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate_loss_diff.py --model PreActResNet18 --epsilon 8 --interval 1 --train_fgsm_ratio 0.875 --type train --resumed_model_name 1225_fgsm_wors_{8}_0.875 --exp_name 0113_diff_train_1225_fgsm_wors_{8}_0.875"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate_loss_diff.py --model PreActResNet18 --epsilon 8 --interval 1 --train_fgsm_ratio 1 --type train --resumed_model_name 1225_fgsm_wors_{8}_1_exp1 --exp_name 0113_diff_train_1225_fgsm_wors_{8}_1_exp1"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate_loss_diff.py --model PreActResNet18 --epsilon 8 --interval 1 --train_fgsm_ratio 1 --type train --resumed_model_name 1225_fgsm_rs_{8}_1 --exp_name 0113_diff_train_1225_fgsm_rs_{8}_1"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate_loss_diff.py --model PreActResNet18 --epsilon 8 --interval 1 --train_fgsm_ratio 1.25 --type train --resumed_model_name 1225_fgsm_rs_{8}_1.25 --exp_name 0113_diff_train_1225_fgsm_rs_{8}_1.25"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate_loss_diff.py --model PreActResNet18 --epsilon 8 --interval 1 --train_fgsm_ratio 1 --type train --resumed_model_name 0109_fgsm_rs_{8}_1 --exp_name 0113_diff_train_0109_fgsm_rs_{8}_1"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate_loss_diff.py --model PreActResNet18 --epsilon 8 --interval 1 --train_fgsm_ratio 1.25 --type train --resumed_model_name 0109_fgsm_rs_{8}_1.25 --exp_name 0113_diff_train_0109_fgsm_rs_{8}_1.25"

#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate_loss_diff.py --model PreActResNet18 --epsilon 8 --interval 1 --train_fgsm_ratio 0.875 --type train --mode eval --resumed_model_name 1225_fgsm_wors_{8}_0.875 --exp_name 0113_diff_train_eval_1225_fgsm_wors_{8}_0.875"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate_loss_diff.py --model PreActResNet18 --epsilon 8 --interval 1 --train_fgsm_ratio 1 --type train --mode eval --resumed_model_name 1225_fgsm_wors_{8}_1_exp1 --exp_name 0113_diff_train_eval_1225_fgsm_wors_{8}_1_exp1"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate_loss_diff.py --model PreActResNet18 --epsilon 8 --interval 1 --train_fgsm_ratio 1 --type train --mode eval --resumed_model_name 1225_fgsm_rs_{8}_1 --exp_name 0113_diff_train_eval_1225_fgsm_rs_{8}_1"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate_loss_diff.py --model PreActResNet18 --epsilon 8 --interval 1 --train_fgsm_ratio 1.25 --type train --mode eval --resumed_model_name 1225_fgsm_rs_{8}_1.25 --exp_name 0113_diff_train_eval_1225_fgsm_rs_{8}_1.25"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate_loss_diff.py --model PreActResNet18 --epsilon 8 --interval 1 --train_fgsm_ratio 1 --type train --mode eval --resumed_model_name 0109_fgsm_rs_{8}_1 --exp_name 0113_diff_train_eval_0109_fgsm_rs_{8}_1"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate_loss_diff.py --model PreActResNet18 --epsilon 8 --interval 1 --train_fgsm_ratio 1.25 --type train --mode eval --resumed_model_name 0109_fgsm_rs_{8}_1.25 --exp_name 0113_diff_train_eval_0109_fgsm_rs_{8}_1.25"

#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_eval.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.125 --clamp --exp_name 0115_fgsm_wors_{8}_0.125"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_eval.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.25 --clamp --exp_name 0115_fgsm_wors_{8}_0.25"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_eval.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.375 --clamp --exp_name 0115_fgsm_wors_{8}_0.375"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_eval.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.5 --clamp --exp_name 0115_fgsm_wors_{8}_0.5"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_eval.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.625 --clamp --exp_name 0115_fgsm_wors_{8}_0.625"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_eval.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.75 --clamp --exp_name 0115_fgsm_wors_{8}_0.75"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_eval.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.875 --clamp  --exp_name 0115_fgsm_wors_{8}_0.875"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_eval.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --clamp --exp_name 1225_0115_wors_{8}_1"

#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_eval.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.125 --train_random_start --clamp --exp_name 0115_fgsm_rs_{8}_0.125"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_eval.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.25 --train_random_start --clamp --exp_name 0115_fgsm_rs_{8}_0.25"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_eval.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.375 --train_random_start --clamp --exp_name 0115_fgsm_rs_{8}_0.375"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_eval.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.5 --train_random_start --clamp --exp_name 0115_fgsm_rs_{8}_0.5"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_eval.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.625 --train_random_start --clamp --exp_name 0115_fgsm_rs_{8}_0.625"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_eval.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.75 --train_random_start --clamp --exp_name 0115_fgsm_rs_{8}_0.75"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_eval.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.875 --train_random_start --clamp --exp_name 0115_fgsm_rs_{8}_0.875"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_eval.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --train_random_start --clamp --exp_name 0115_fgsm_rs_{8}_1"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_eval.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1.125 --train_random_start --clamp --exp_name 0115_fgsm_rs_{8}_1.125"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_eval.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1.25 --train_random_start --clamp --exp_name 0115_fgsm_rs_{8}_1.25"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_eval.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1.375 --train_random_start --clamp --exp_name 0115_fgsm_rs_{8}_1.375"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_eval.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1.5 --train_random_start --clamp --exp_name 0115_fgsm_rs_{8}_1.5"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_eval.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1.625 --train_random_start --clamp --exp_name 0115_fgsm_rs_{8}_1.625"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_eval.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1.75 --train_random_start --clamp --exp_name 0115_fgsm_rs_{8}_1.75"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_eval.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1.875 --train_random_start --clamp --exp_name 0115_fgsm_rs_{8}_1.875"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_eval.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 2 --train_random_start --clamp --exp_name 0115_fgsm_rs_{8}_2"

#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_eval.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.125 --train_random_start --exp_name 0115_fgsm_rs_wo_clamp_{8}_0.125"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_eval.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.25 --train_random_start --exp_name 0115_fgsm_rs_wo_clamp_{8}_0.25"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_eval.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.375 --train_random_start --exp_name 0115_fgsm_rs_wo_clamp_{8}_0.375"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_eval.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.5 --train_random_start --exp_name 0115_fgsm_rs_wo_clamp_{8}_0.5"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_eval.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.625 --train_random_start --exp_name 0115_fgsm_rs_wo_clamp_{8}_0.625"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_eval.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.75 --train_random_start --exp_name 0115_fgsm_rs_wo_clamp_{8}_0.75"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_eval.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.875 --train_random_start --exp_name 0115_fgsm_rs_wo_clamp_{8}_0.875"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_eval.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --train_random_start --exp_name 0115_fgsm_rs_wo_clamp_{8}_1"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_eval.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1.125 --train_random_start --exp_name 0115_fgsm_rs_wo_clamp_{8}_1.125"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_eval.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1.25 --train_random_start --exp_name 0115_fgsm_rs_wo_clamp_{8}_1.25"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_eval.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1.375 --train_random_start --exp_name 0115_fgsm_rs_wo_clamp_{8}_1.375"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_eval.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1.5 --train_random_start --exp_name 0115_fgsm_rs_wo_clamp_{8}_1.5"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_eval.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1.625 --train_random_start --exp_name 0115_fgsm_rs_wo_clamp_{8}_1.625"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_eval.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1.75 --train_random_start --exp_name 0115_fgsm_rs_wo_clamp_{8}_1.75"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_eval.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1.875 --train_random_start --exp_name 0115_fgsm_rs_wo_clamp_{8}_1.875"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_eval.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 2 --train_random_start --exp_name 0115_fgsm_rs_wo_clamp_{8}_2"

#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_uniform.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.125 --exp_name 0116_fgsm_uni_{8}_0.125"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_uniform.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.25 --exp_name 0116_fgsm_uni_{8}_0.25"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_uniform.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.375 --exp_name 0116_fgsm_uni_{8}_0.375"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_uniform.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.5 --exp_name 0116_fgsm_uni_{8}_0.5"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_uniform.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.625 --exp_name 0116_fgsm_uni_{8}_0.625"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_uniform.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.75 --exp_name 0116_fgsm_uni_{8}_0.75"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_uniform.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.875 --exp_name 0116_fgsm_uni_{8}_0.875"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_uniform.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --exp_name 0116_fgsm_uni_{8}_1"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_uniform.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1.125 --exp_name 0116_fgsm_uni_{8}_1.125"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_uniform.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1.25 --exp_name 0116_fgsm_uni_{8}_1.25"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_uniform.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1.375 --exp_name 0116_fgsm_uni_{8}_1.375"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_uniform.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1.5 --exp_name 0116_fgsm_uni_{8}_1.5"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_uniform.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1.625 --exp_name 0116_fgsm_uni_{8}_1.625"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_uniform.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1.75 --exp_name 0116_fgsm_uni_{8}_1.75"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_uniform.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1.875 --exp_name 0116_fgsm_uni_{8}_1.875"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_uniform.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 2 --exp_name 0116_fgsm_uni_{8}_2"

#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_boundary.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.125 --train_random_start --clamp --exp_name 0116_fgsm_bound_{8}_0.125"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_boundary.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.25 --train_random_start --clamp --exp_name 0116_fgsm_bound_{8}_0.25"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_boundary.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.375 --train_random_start --clamp --exp_name 0116_fgsm_bound_{8}_0.375"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_boundary.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.5 --train_random_start --clamp --exp_name 0116_fgsm_bound_{8}_0.5"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_boundary.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.625 --train_random_start --clamp --exp_name 0116_fgsm_bound_{8}_0.625"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_boundary.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.75 --train_random_start --clamp --exp_name 0116_fgsm_bound_{8}_0.75"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_boundary.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.875 --train_random_start --clamp --exp_name 0116_fgsm_bound_{8}_0.875"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_boundary.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --train_random_start --clamp --exp_name 0116_fgsm_bound_{8}_1"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_boundary.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1.125 --train_random_start --clamp --exp_name 0116_fgsm_bound_{8}_1.125"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_boundary.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1.25 --train_random_start --clamp --exp_name 0116_fgsm_bound_{8}_1.25"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_boundary.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1.375 --train_random_start --clamp --exp_name 0116_fgsm_bound_{8}_1.375"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_boundary.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1.5 --train_random_start --clamp --exp_name 0116_fgsm_bound_{8}_1.5"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_boundary.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1.625 --train_random_start --clamp --exp_name 0116_fgsm_bound_{8}_1.625"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_boundary.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1.75 --train_random_start --clamp --exp_name 0116_fgsm_bound_{8}_1.75"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_boundary.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1.875 --train_random_start --clamp --exp_name 0116_fgsm_bound_{8}_1.875"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_boundary.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 2 --train_random_start --clamp --exp_name 0116_fgsm_bound_{8}_2"
#
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_boundary.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.125 --train_random_start --exp_name 0116_fgsm_bound_wo_clamp_{8}_0.125"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_boundary.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.25 --train_random_start --exp_name 0116_fgsm_bound_wo_clamp_{8}_0.25"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_boundary.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.375 --train_random_start --exp_name 0116_fgsm_bound_wo_clamp_{8}_0.375"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_boundary.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.5 --train_random_start --exp_name 0116_fgsm_bound_wo_clamp_{8}_0.5"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_boundary.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.625 --train_random_start --exp_name 0116_fgsm_bound_wo_clamp_{8}_0.625"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_boundary.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.75 --train_random_start --exp_name 0116_fgsm_bound_wo_clamp_{8}_0.75"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_boundary.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.875 --train_random_start --exp_name 0116_fgsm_bound_wo_clamp_{8}_0.875"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_boundary.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --train_random_start --exp_name 0116_fgsm_bound_wo_clamp_{8}_1"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_boundary.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1.125 --train_random_start --exp_name 0116_fgsm_bound_wo_clamp_{8}_1.125"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_boundary.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1.25 --train_random_start --exp_name 0116_fgsm_bound_wo_clamp_{8}_1.25"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_boundary.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1.375 --train_random_start --exp_name 0116_fgsm_bound_wo_clamp_{8}_1.375"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_boundary.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1.5 --train_random_start --exp_name 0116_fgsm_bound_wo_clamp_{8}_1.5"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_boundary.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1.625 --train_random_start --exp_name 0116_fgsm_bound_wo_clamp_{8}_1.625"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_boundary.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1.75 --train_random_start --exp_name 0116_fgsm_bound_wo_clamp_{8}_1.75"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_boundary.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1.875 --train_random_start --exp_name 0116_fgsm_bound_wo_clamp_{8}_1.875"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_boundary.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 2 --train_random_start --exp_name 0116_fgsm_bound_wo_clamp_{8}_2"

#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_reverse.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.125 --train_random_start --clamp --exp_name 0116_fgsm_reverse_{8}_0.125"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_reverse.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.25 --train_random_start --clamp --exp_name 0116_fgsm_reverse_{8}_0.25"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_reverse.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.375 --train_random_start --clamp --exp_name 0116_fgsm_reverse_{8}_0.375"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_reverse.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.5 --train_random_start --clamp --exp_name 0116_fgsm_reverse_{8}_0.5"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_reverse.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.625 --train_random_start --clamp --exp_name 0116_fgsm_reverse_{8}_0.625"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_reverse.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.75 --train_random_start --clamp --exp_name 0116_fgsm_reverse_{8}_0.75"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_reverse.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.875 --train_random_start --clamp --exp_name 0116_fgsm_reverse_{8}_0.875"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_reverse.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --train_random_start --clamp --exp_name 0116_fgsm_reverse_{8}_1"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_reverse.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1.125 --train_random_start --clamp --exp_name 0116_fgsm_reverse_{8}_1.125"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_reverse.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1.25 --train_random_start --clamp --exp_name 0116_fgsm_reverse_{8}_1.25"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_reverse.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1.375 --train_random_start --clamp --exp_name 0116_fgsm_reverse_{8}_1.375"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_reverse.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1.5 --train_random_start --clamp --exp_name 0116_fgsm_reverse_{8}_1.5"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_reverse.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1.625 --train_random_start --clamp --exp_name 0116_fgsm_reverse_{8}_1.625"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_reverse.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1.75 --train_random_start --clamp --exp_name 0116_fgsm_reverse_{8}_1.75"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_reverse.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1.875 --train_random_start --clamp --exp_name 0116_fgsm_reverse_{8}_1.875"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_reverse.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 2 --train_random_start --clamp --exp_name 0116_fgsm_reverse_{8}_2"
#
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_reverse.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.125 --train_random_start --exp_name 0116_fgsm_reverse_wo_clamp_{8}_0.125"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_reverse.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.25 --train_random_start --exp_name 0116_fgsm_reverse_wo_clamp_{8}_0.25"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_reverse.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.375 --train_random_start --exp_name 0116_fgsm_reverse_wo_clamp_{8}_0.375"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_reverse.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.5 --train_random_start --exp_name 0116_fgsm_reverse_wo_clamp_{8}_0.5"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_reverse.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.625 --train_random_start --exp_name 0116_fgsm_reverse_wo_clamp_{8}_0.625"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_reverse.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.75 --train_random_start --exp_name 0116_fgsm_reverse_wo_clamp_{8}_0.75"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_reverse.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.875 --train_random_start --exp_name 0116_fgsm_reverse_wo_clamp_{8}_0.875"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_reverse.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --train_random_start --exp_name 0116_fgsm_reverse_wo_clamp_{8}_1"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_reverse.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1.125 --train_random_start --exp_name 0116_fgsm_reverse_wo_clamp_{8}_1.125"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_reverse.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1.25 --train_random_start --exp_name 0116_fgsm_reverse_wo_clamp_{8}_1.25"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_reverse.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1.375 --train_random_start --exp_name 0116_fgsm_reverse_wo_clamp_{8}_1.375"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_reverse.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1.5 --train_random_start --exp_name 0116_fgsm_reverse_wo_clamp_{8}_1.5"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_reverse.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1.625 --train_random_start --exp_name 0116_fgsm_reverse_wo_clamp_{8}_1.625"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_reverse.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1.75 --train_random_start --exp_name 0116_fgsm_reverse_wo_clamp_{8}_1.75"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_reverse.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1.875 --train_random_start --exp_name 0116_fgsm_reverse_wo_clamp_{8}_1.875"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_reverse.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 2 --train_random_start --exp_name 0116_fgsm_reverse_wo_clamp_{8}_2"

#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_uniform.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1.125 --clamp --exp_name 0116_fgsm_uni_w_clamp_{8}_1.125"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_uniform.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1.25 --clamp --exp_name 0116_fgsm_uni_w_clamp_{8}_1.25"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_uniform.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1.375 --clamp --exp_name 0116_fgsm_uni_w_clamp_{8}_1.375"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_uniform.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1.5 --clamp --exp_name 0116_fgsm_uni_w_clamp_{8}_1.5"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_uniform.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1.625 --clamp --exp_name 0116_fgsm_uni_w_clamp_{8}_1.625"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_uniform.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1.75 --clamp --exp_name 0116_fgsm_uni_w_clamp_{8}_1.75"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_uniform.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1.875 --clamp --exp_name 0116_fgsm_uni_w_clamp_{8}_1.875"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_uniform.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 2 --clamp --exp_name 0116_fgsm_uni_w_clamp_{8}_2"
#
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_uniform.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.125 --train_random_start --clamp --exp_name 0116_fgsm_uni_rs_w_clamp_{8}_0.125"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_uniform.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.25 --train_random_start --clamp --exp_name 0116_fgsm_uni_rs_w_clamp_{8}_0.25"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_uniform.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.375 --train_random_start --clamp --exp_name 0116_fgsm_uni_rs_w_clamp_{8}_0.375"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_uniform.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.5 --train_random_start --clamp --exp_name 0116_fgsm_uni_rs_w_clamp_{8}_0.5"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_uniform.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.625 --train_random_start --clamp --exp_name 0116_fgsm_uni_rs_w_clamp_{8}_0.625"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_uniform.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.75 --train_random_start --clamp --exp_name 0116_fgsm_uni_rs_w_clamp_{8}_0.75"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_uniform.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.875 --train_random_start --clamp --exp_name 0116_fgsm_uni_rs_w_clamp_{8}_0.875"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_uniform.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --train_random_start --clamp --exp_name 0116_fgsm_uni_rs_w_clamp_{8}_1"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_uniform.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1.125 --train_random_start --clamp --exp_name 0116_fgsm_uni_rs_w_clamp_{8}_1.125"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_uniform.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1.25 --train_random_start --clamp --exp_name 0116_fgsm_uni_rs_w_clamp_{8}_1.25"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_uniform.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1.375 --train_random_start --clamp --exp_name 0116_fgsm_uni_rs_w_clamp_{8}_1.375"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_uniform.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1.5 --train_random_start --clamp --exp_name 0116_fgsm_uni_rs_w_clamp_{8}_1.5"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_uniform.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1.625 --train_random_start --clamp --exp_name 0116_fgsm_uni_rs_w_clamp_{8}_1.625"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_uniform.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1.75 --train_random_start --clamp --exp_name 0116_fgsm_uni_rs_w_clamp_{8}_1.75"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_uniform.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1.875 --train_random_start --clamp --exp_name 0116_fgsm_uni_rs_w_clamp_{8}_1.875"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_uniform.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 2 --train_random_start --clamp --exp_name 0116_fgsm_uni_rs_w_clamp_{8}_2"
#
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_uniform.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.125 --train_random_start --exp_name 0116_fgsm_uni_rs_wo_clamp_{8}_0.125"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_uniform.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.25 --train_random_start --exp_name 0116_fgsm_uni_rs_wo_clamp_{8}_0.25"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_uniform.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.375 --train_random_start --exp_name 0116_fgsm_uni_rs_wo_clamp_{8}_0.375"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_uniform.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.5 --train_random_start --exp_name 0116_fgsm_uni_rs_wo_clamp_{8}_0.5"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_uniform.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.625 --train_random_start --exp_name 0116_fgsm_uni_rs_wo_clamp_{8}_0.625"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_uniform.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.75 --train_random_start --exp_name 0116_fgsm_uni_rs_wo_clamp_{8}_0.75"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_uniform.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.875 --train_random_start --exp_name 0116_fgsm_uni_rs_wo_clamp_{8}_0.875"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_uniform.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --train_random_start --exp_name 0116_fgsm_uni_rs_wo_clamp_{8}_1"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_uniform.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1.125 --train_random_start --exp_name 0116_fgsm_uni_rs_wo_clamp_{8}_1.125"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_uniform.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1.25 --train_random_start --exp_name 0116_fgsm_uni_rs_wo_clamp_{8}_1.25"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_uniform.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1.375 --train_random_start --exp_name 0116_fgsm_uni_rs_wo_clamp_{8}_1.375"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_uniform.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1.5 --train_random_start --exp_name 0116_fgsm_uni_rs_wo_clamp_{8}_1.5"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_uniform.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1.625 --train_random_start --exp_name 0116_fgsm_uni_rs_wo_clamp_{8}_1.625"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_uniform.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1.75 --train_random_start --exp_name 0116_fgsm_uni_rs_wo_clamp_{8}_1.75"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_uniform.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1.875 --train_random_start --exp_name 0116_fgsm_uni_rs_wo_clamp_{8}_1.875"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_uniform.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 2 --train_random_start --exp_name 0116_fgsm_uni_rs_wo_clamp_{8}_2"

#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_probability.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --probability 0.5 --train_fgsm_ratio 0.25 --clamp --exp_name 0116_fgsm_prob_w_clamp_{8}_0.25"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_probability.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --probability 0.5 --train_fgsm_ratio 0.5 --clamp --exp_name 0116_fgsm_prob_w_clamp_{8}_0.5"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_probability.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --probability 0.5 --train_fgsm_ratio 0.75 --clamp --exp_name 0116_fgsm_prob_w_clamp_{8}_0.75"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_probability.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --probability 0.5 --train_fgsm_ratio 1 --clamp --exp_name 0116_fgsm_prob_w_clamp_{8}_1"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_probability.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --probability 0.5 --train_fgsm_ratio 1.25 --clamp --exp_name 0116_fgsm_prob_w_clamp_{8}_1.25"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_probability.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --probability 0.5 --train_fgsm_ratio 1.5 --clamp --exp_name 0116_fgsm_prob_w_clamp_{8}_1.5"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_probability.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --probability 0.5 --train_fgsm_ratio 1.75 --clamp --exp_name 0116_fgsm_prob_w_clamp_{8}_1.75"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_probability.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --probability 0.5 --train_fgsm_ratio 2 --clamp --exp_name 0116_fgsm_prob_w_clamp_{8}_2"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate.py --model PreActResNet18 --epsilon 8 --resumed_model_name 0116_fgsm_prob_w_clamp_{8}_0.25 --exp_name evaluate_0116_fgsm_prob_w_clamp_{8}_0.25"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate.py --model PreActResNet18 --epsilon 8 --resumed_model_name 0116_fgsm_prob_w_clamp_{8}_0.5 --exp_name evaluate_0116_fgsm_prob_w_clamp_{8}_0.5"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate.py --model PreActResNet18 --epsilon 8 --resumed_model_name 0116_fgsm_prob_w_clamp_{8}_0.75 --exp_name evaluate_0116_fgsm_prob_w_clamp_{8}_0.75"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate.py --model PreActResNet18 --epsilon 8 --resumed_model_name 0116_fgsm_prob_w_clamp_{8}_1 --exp_name evaluate_0116_fgsm_prob_w_clamp_{8}_1"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate.py --model PreActResNet18 --epsilon 8 --resumed_model_name 0116_fgsm_prob_w_clamp_{8}_1.25 --exp_name evaluate_0116_fgsm_prob_w_clamp_{8}_1.25"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate.py --model PreActResNet18 --epsilon 8 --resumed_model_name 0116_fgsm_prob_w_clamp_{8}_1.5 --exp_name evaluate_0116_fgsm_prob_w_clamp_{8}_1.5"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate.py --model PreActResNet18 --epsilon 8 --resumed_model_name 0116_fgsm_prob_w_clamp_{8}_1.75 --exp_name evaluate_0116_fgsm_prob_w_clamp_{8}_1.75"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate.py --model PreActResNet18 --epsilon 8 --resumed_model_name 0116_fgsm_prob_w_clamp_{8}_2 --exp_name evaluate_0116_fgsm_prob_w_clamp_{8}_2"

#
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_probability.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --probability 0.5 --train_fgsm_ratio 0.25 --exp_name 0116_fgsm_prob_wo_clamp_{8}_0.25"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_probability.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --probability 0.5 --train_fgsm_ratio 0.5 --exp_name 0116_fgsm_prob_wo_clamp_{8}_0.5"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_probability.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --probability 0.5 --train_fgsm_ratio 0.75 --exp_name 0116_fgsm_prob_wo_clamp_{8}_0.75"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_probability.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --probability 0.5 --train_fgsm_ratio 1 --exp_name 0116_fgsm_prob_wo_clamp_{8}_1"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_probability.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --probability 0.5 --train_fgsm_ratio 1.25 --exp_name 0116_fgsm_prob_wo_clamp_{8}_1.25"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_probability.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --probability 0.5 --train_fgsm_ratio 1.5 --exp_name 0116_fgsm_prob_wo_clamp_{8}_1.5"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_probability.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --probability 0.5 --train_fgsm_ratio 1.75 --exp_name 0116_fgsm_prob_wo_clamp_{8}_1.75"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_probability.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --probability 0.5 --train_fgsm_ratio 2 --exp_name 0116_fgsm_prob_wo_clamp_{8}_2"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate.py --model PreActResNet18 --epsilon 8 --resumed_model_name 0116_fgsm_prob_wo_clamp_{8}_0.25 --exp_name evaluate_0116_fgsm_prob_wo_clamp_{8}_0.25"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate.py --model PreActResNet18 --epsilon 8 --resumed_model_name 0116_fgsm_prob_wo_clamp_{8}_0.5 --exp_name evaluate_0116_fgsm_prob_wo_clamp_{8}_0.5"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate.py --model PreActResNet18 --epsilon 8 --resumed_model_name 0116_fgsm_prob_wo_clamp_{8}_0.75 --exp_name evaluate_0116_fgsm_prob_wo_clamp_{8}_0.75"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate.py --model PreActResNet18 --epsilon 8 --resumed_model_name 0116_fgsm_prob_wo_clamp_{8}_1 --exp_name evaluate_0116_fgsm_prob_wo_clamp_{8}_1"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate.py --model PreActResNet18 --epsilon 8 --resumed_model_name 0116_fgsm_prob_wo_clamp_{8}_1.25 --exp_name evaluate_0116_fgsm_prob_wo_clamp_{8}_1.25"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate.py --model PreActResNet18 --epsilon 8 --resumed_model_name 0116_fgsm_prob_wo_clamp_{8}_1.5 --exp_name evaluate_0116_fgsm_prob_wo_clamp_{8}_1.5"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate.py --model PreActResNet18 --epsilon 8 --resumed_model_name 0116_fgsm_prob_wo_clamp_{8}_1.75 --exp_name evaluate_0116_fgsm_prob_wo_clamp_{8}_1.75"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate.py --model PreActResNet18 --epsilon 8 --resumed_model_name 0116_fgsm_prob_wo_clamp_{8}_2 --exp_name evaluate_0116_fgsm_prob_wo_clamp_{8}_2"

#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_clip.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --clip 100 --exp_name 0116_fgsm_{8}_clip100"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_clip.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --clip 1.5 --exp_name 0116_fgsm_{8}_clip1.5_exp1"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_clip.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --clip 0.5 --exp_name 0116_fgsm_{8}_clip0.5_exp1"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_clip.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --clip 0.25 --exp_name 0116_fgsm_{8}_clip0.25_exp1"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_clip.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --clip 1 --exp_name 0116_fgsm_{8}_clip1_exp1"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_clip.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --clip 0.75 --exp_name 0116_fgsm_{8}_clip0.75_exp1"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_clip.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --clip 0.1 --exp_name 0116_fgsm_{8}_clip0.1_exp1"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_clip.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --clip 0.15 --exp_name 0116_fgsm_{8}_clip0.15_exp1"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_clip.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --clip 0.2 --exp_name 0116_fgsm_{8}_clip0.2_exp1"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_clip.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --clip 0.05 --exp_name 0116_fgsm_{8}_clip0.05_exp1"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate.py --model PreActResNet18 --epsilon 8 --resumed_model_name 0116_fgsm_{8}_clip0.05_exp1 --exp_name evaluate_0116_fgsm_{8}_clip0.05_exp1"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_clip.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --clip 0.05 --exp_name 0116_fgsm_{8}_clip0.05_exp2"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_clip.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --clip 0.075 --exp_name 0116_fgsm_{8}_clip0.075_exp2"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_clip.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 16 --train_fgsm_ratio 1 --clip 100 --exp_name 0116_fgsm_{16}_clip100"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_clip.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 16 --train_fgsm_ratio 1 --clip 0.05 --exp_name 0116_fgsm_{16}_clip0.05"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_clip.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 16 --train_fgsm_ratio 1 --clip 0.075 --exp_name 0116_fgsm_{16}_clip0.075"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_clip.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 16 --train_fgsm_ratio 1 --clip 0.75 --exp_name 0116_fgsm_{16}_clip0.75"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_clip.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 16 --train_fgsm_ratio 1 --clip 1 --exp_name 0116_fgsm_{16}_clip1"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_clip.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 16 --train_fgsm_ratio 1 --clip 0.025 --exp_name 0116_fgsm_{16}_clip0.025"

#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_only_random.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --random_type uniform --exp_name 0116_fgsm_uniform_random_{8}"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_only_random.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 16 --random_type uniform --exp_name 0116_fgsm_uniform_random_{16}"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_only_random.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --random_type normal --exp_name 0116_fgsm_normal_random_{8}"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_only_random.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 16 --random_type normal --exp_name 0116_fgsm_normal_random_{16}"

#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_batch.py --model PreActResNet18 --num_epochs 4 --epsilon 8 --train_fgsm_ratio 1 --resumed_model_name 1225_fgsm_wors_{8}_1_exp1_14.pth --exp_name 0119_batch_1225_fgsm_wors_{8}_1_exp1"

#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_batch.py --model PreActResNet18 --num_epochs 4 --epsilon 8 --train_fgsm_ratio 1 --resumed_model_name 1225_fgsm_wors_{8}_1_exp1_14.pth --exp_name 0119_batch_1225_fgsm_wors_{8}_1_exp2"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_batch.py --model PreActResNet18 --num_epochs 4 --epsilon 8 --train_fgsm_ratio 1 --resumed_model_name 1225_fgsm_wors_{8}_1_exp1_14.pth --exp_name 0119_batch_1225_fgsm_wors_{8}_1_exp3"

#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --train_random_start --finetune --resumed_model_name 0116_fgsm_uni_{8}_2_46.pth --exp_name 0123_resume46_0116_fgsm_uni_{8}_2_rs"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --train_random_start --finetune --resumed_model_name 0116_fgsm_uni_{8}_2_47.pth --exp_name 0123_resume47_0116_fgsm_uni_{8}_2_rs"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --train_random_start --finetune --resumed_model_name 0116_fgsm_uni_{8}_2_48.pth --exp_name 0123_resume48_0116_fgsm_uni_{8}_2_rs"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --train_random_start --finetune --resumed_model_name 0116_fgsm_uni_{8}_2_49.pth --exp_name 0123_resume49_0116_fgsm_uni_{8}_2_rs"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --train_random_start --finetune --resumed_model_name 0116_fgsm_uni_{8}_2_70.pth --exp_name 0123_resume70_0116_fgsm_uni_{8}_2_rs"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --train_random_start --finetune --resumed_model_name 0116_fgsm_uni_{8}_2_46.pth --exp_name 0123_resume46_0116_fgsm_uni_{8}_2_rs_exp1"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --train_random_start --finetune --resumed_model_name 0116_fgsm_uni_{8}_2_47.pth --exp_name 0123_resume47_0116_fgsm_uni_{8}_2_rs_exp1"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --train_random_start --finetune --resumed_model_name 0116_fgsm_uni_{8}_2_48.pth --exp_name 0123_resume48_0116_fgsm_uni_{8}_2_rs_exp1"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --train_random_start --finetune --resumed_model_name 0116_fgsm_uni_{8}_2_49.pth --exp_name 0123_resume49_0116_fgsm_uni_{8}_2_rs_exp1"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --train_random_start --finetune --resumed_model_name 0116_fgsm_uni_{8}_2_70.pth --exp_name 0123_resume70_0116_fgsm_uni_{8}_2_rs_exp1"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --train_random_start --finetune --resumed_model_name 0116_fgsm_uni_{8}_2_46.pth --exp_name 0123_resume46_0116_fgsm_uni_{8}_2_rs_exp2"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --train_random_start --finetune --resumed_model_name 0116_fgsm_uni_{8}_2_47.pth --exp_name 0123_resume47_0116_fgsm_uni_{8}_2_rs_exp2"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --train_random_start --finetune --resumed_model_name 0116_fgsm_uni_{8}_2_48.pth --exp_name 0123_resume48_0116_fgsm_uni_{8}_2_rs_exp2"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --train_random_start --finetune --resumed_model_name 0116_fgsm_uni_{8}_2_49.pth --exp_name 0123_resume49_0116_fgsm_uni_{8}_2_rs_exp2"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --train_random_start --finetune --resumed_model_name 0116_fgsm_uni_{8}_2_70.pth --exp_name 0123_resume70_0116_fgsm_uni_{8}_2_rs_exp2"

#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_boundary.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --train_random_start --finetune --resumed_model_name 0116_fgsm_uni_{8}_2_46.pth --exp_name 0123_resume46_0116_fgsm_uni_{8}_2_bound"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_boundary.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --train_random_start --finetune --resumed_model_name 0116_fgsm_uni_{8}_2_47.pth --exp_name 0123_resume47_0116_fgsm_uni_{8}_2_bound"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_boundary.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --train_random_start --finetune --resumed_model_name 0116_fgsm_uni_{8}_2_48.pth --exp_name 0123_resume48_0116_fgsm_uni_{8}_2_bound"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_boundary.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --train_random_start --finetune --resumed_model_name 0116_fgsm_uni_{8}_2_49.pth --exp_name 0123_resume49_0116_fgsm_uni_{8}_2_bound"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_boundary.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --train_random_start --finetune --resumed_model_name 0116_fgsm_uni_{8}_2_70.pth --exp_name 0123_resume70_0116_fgsm_uni_{8}_2_bound"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_boundary.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --train_random_start --finetune --resumed_model_name 0116_fgsm_uni_{8}_2_46.pth --exp_name 0123_resume46_0116_fgsm_uni_{8}_2_bound_exp1"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_boundary.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --train_random_start --finetune --resumed_model_name 0116_fgsm_uni_{8}_2_47.pth --exp_name 0123_resume47_0116_fgsm_uni_{8}_2_bound_exp1"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_boundary.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --train_random_start --finetune --resumed_model_name 0116_fgsm_uni_{8}_2_48.pth --exp_name 0123_resume48_0116_fgsm_uni_{8}_2_bound_exp1"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_boundary.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --train_random_start --finetune --resumed_model_name 0116_fgsm_uni_{8}_2_49.pth --exp_name 0123_resume49_0116_fgsm_uni_{8}_2_bound_exp1"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_boundary.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --train_random_start --finetune --resumed_model_name 0116_fgsm_uni_{8}_2_70.pth --exp_name 0123_resume70_0116_fgsm_uni_{8}_2_bound_exp1"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_boundary.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --train_random_start --finetune --resumed_model_name 0116_fgsm_uni_{8}_2_46.pth --exp_name 0123_resume46_0116_fgsm_uni_{8}_2_bound_exp2"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_boundary.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --train_random_start --finetune --resumed_model_name 0116_fgsm_uni_{8}_2_47.pth --exp_name 0123_resume47_0116_fgsm_uni_{8}_2_bound_exp2"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_boundary.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --train_random_start --finetune --resumed_model_name 0116_fgsm_uni_{8}_2_48.pth --exp_name 0123_resume48_0116_fgsm_uni_{8}_2_bound_exp2"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_boundary.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --train_random_start --finetune --resumed_model_name 0116_fgsm_uni_{8}_2_49.pth --exp_name 0123_resume49_0116_fgsm_uni_{8}_2_bound_exp2"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_boundary.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --train_random_start --finetune --resumed_model_name 0116_fgsm_uni_{8}_2_70.pth --exp_name 0123_resume70_0116_fgsm_uni_{8}_2_bound_exp2"

#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_boundary.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 16 --train_fgsm_ratio 0.125 --train_random_start --exp_name 0116_fgsm_bound_wo_clamp_{16}_0.125"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_boundary.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 16 --train_fgsm_ratio 0.25 --train_random_start --exp_name 0116_fgsm_bound_wo_clamp_{16}_0.25"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_boundary.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 16 --train_fgsm_ratio 0.375 --train_random_start --exp_name 0116_fgsm_bound_wo_clamp_{16}_0.375"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_boundary.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 16 --train_fgsm_ratio 0.5 --train_random_start --exp_name 0116_fgsm_bound_wo_clamp_{16}_0.5"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_boundary.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 16 --train_fgsm_ratio 0.625 --train_random_start --exp_name 0116_fgsm_bound_wo_clamp_{16}_0.625"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_boundary.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 16 --train_fgsm_ratio 0.75 --train_random_start --exp_name 0116_fgsm_bound_wo_clamp_{16}_0.75"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_boundary.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 16 --train_fgsm_ratio 0.875 --train_random_start --exp_name 0116_fgsm_bound_wo_clamp_{16}_0.875"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_boundary.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 16 --train_fgsm_ratio 1 --train_random_start --exp_name 0116_fgsm_bound_wo_clamp_{16}_1"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_boundary.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 16 --train_fgsm_ratio 1.125 --train_random_start --exp_name 0116_fgsm_bound_wo_clamp_{16}_1.125"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_boundary.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 16 --train_fgsm_ratio 1.25 --train_random_start --exp_name 0116_fgsm_bound_wo_clamp_{16}_1.25"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_boundary.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 16 --train_fgsm_ratio 1.375 --train_random_start --exp_name 0116_fgsm_bound_wo_clamp_{16}_1.375"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_boundary.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 16 --train_fgsm_ratio 1.5 --train_random_start --exp_name 0116_fgsm_bound_wo_clamp_{16}_1.5"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_boundary.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 16 --train_fgsm_ratio 1.625 --train_random_start --exp_name 0116_fgsm_bound_wo_clamp_{16}_1.625"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_boundary.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 16 --train_fgsm_ratio 1.75 --train_random_start --exp_name 0116_fgsm_bound_wo_clamp_{16}_1.75"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_boundary.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 16 --train_fgsm_ratio 1.875 --train_random_start --exp_name 0116_fgsm_bound_wo_clamp_{16}_1.875"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_boundary.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 16 --train_fgsm_ratio 2 --train_random_start --exp_name 0116_fgsm_bound_wo_clamp_{16}_2"

#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.125 --train_random_start --exp_name 0116_fgsm_rs_wo_clamp_{8}_0.125"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.25 --train_random_start --exp_name 0116_fgsm_rs_wo_clamp_{8}_0.25"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.375 --train_random_start --exp_name 0116_fgsm_rs_wo_clamp_{8}_0.375"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.5 --train_random_start --exp_name 0116_fgsm_rs_wo_clamp_{8}_0.5"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.625 --train_random_start --exp_name 0116_fgsm_rs_wo_clamp_{8}_0.625"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.75 --train_random_start --exp_name 0116_fgsm_rs_wo_clamp_{8}_0.75"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.875 --train_random_start --exp_name 0116_fgsm_rs_wo_clamp_{8}_0.875"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --train_random_start --exp_name 0116_fgsm_rs_wo_clamp_{8}_1"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1.125 --train_random_start --exp_name 0116_fgsm_rs_wo_clamp_{8}_1.125"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1.25 --train_random_start --exp_name 0116_fgsm_rs_wo_clamp_{8}_1.25"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1.375 --train_random_start --exp_name 0116_fgsm_rs_wo_clamp_{8}_1.375"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1.5 --train_random_start --exp_name 0116_fgsm_rs_wo_clamp_{8}_1.5"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1.625 --train_random_start --exp_name 0116_fgsm_rs_wo_clamp_{8}_1.625"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1.75 --train_random_start --exp_name 0116_fgsm_rs_wo_clamp_{8}_1.75"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1.875 --train_random_start --exp_name 0116_fgsm_rs_wo_clamp_{8}_1.875"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 2 --train_random_start --exp_name 0116_fgsm_rs_wo_clamp_{8}_2"
#
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 16 --train_fgsm_ratio 0.125 --train_random_start --exp_name 0116_fgsm_rs_wo_clamp_{16}_0.125"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 16 --train_fgsm_ratio 0.25 --train_random_start --exp_name 0116_fgsm_rs_wo_clamp_{16}_0.25"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 16 --train_fgsm_ratio 0.375 --train_random_start --exp_name 0116_fgsm_rs_wo_clamp_{16}_0.375"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 16 --train_fgsm_ratio 0.5 --train_random_start --exp_name 0116_fgsm_rs_wo_clamp_{16}_0.5"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 16 --train_fgsm_ratio 0.625 --train_random_start --exp_name 0116_fgsm_rs_wo_clamp_{16}_0.625"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 16 --train_fgsm_ratio 0.75 --train_random_start --exp_name 0116_fgsm_rs_wo_clamp_{16}_0.75"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 16 --train_fgsm_ratio 0.875 --train_random_start --exp_name 0116_fgsm_rs_wo_clamp_{16}_0.875"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 16 --train_fgsm_ratio 1 --train_random_start --exp_name 0116_fgsm_rs_wo_clamp_{16}_1"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 16 --train_fgsm_ratio 1.125 --train_random_start --exp_name 0116_fgsm_rs_wo_clamp_{16}_1.125"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 16 --train_fgsm_ratio 1.25 --train_random_start --exp_name 0116_fgsm_rs_wo_clamp_{16}_1.25"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 16 --train_fgsm_ratio 1.375 --train_random_start --exp_name 0116_fgsm_rs_wo_clamp_{16}_1.375"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 16 --train_fgsm_ratio 1.5 --train_random_start --exp_name 0116_fgsm_rs_wo_clamp_{16}_1.5"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 16 --train_fgsm_ratio 1.625 --train_random_start --exp_name 0116_fgsm_rs_wo_clamp_{16}_1.625"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 16 --train_fgsm_ratio 1.75 --train_random_start --exp_name 0116_fgsm_rs_wo_clamp_{16}_1.75"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 16 --train_fgsm_ratio 1.875 --train_random_start --exp_name 0116_fgsm_rs_wo_clamp_{16}_1.875"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 16 --train_fgsm_ratio 2 --train_random_start --exp_name 0116_fgsm_rs_wo_clamp_{16}_2"

#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_large_random.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.625 --train_random_start --lower_ratio 0.5 --upper_ratio 1.5 --exp_name 0124_fgsm_large_rs_wo_clamp_{8}_0.625"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_large_random.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.75 --train_random_start --lower_ratio 0.5 --upper_ratio 1.5 --exp_name 0124_fgsm_large_rs_wo_clamp_{8}_0.75"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_large_random.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 0.875 --train_random_start --lower_ratio 0.5 --upper_ratio 1.5 --exp_name 0124_fgsm_large_rs_wo_clamp_{8}_0.875"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_large_random.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --train_random_start --lower_ratio 0.5 --upper_ratio 1.5 --exp_name 0124_fgsm_large_rs_wo_clamp_{8}_1"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_large_random.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1.125 --train_random_start --lower_ratio 0.5 --upper_ratio 1.5 --exp_name 0124_fgsm_large_rs_wo_clamp_{8}_1.125"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_large_random.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1.25 --train_random_start --lower_ratio 0.5 --upper_ratio 1.5 --exp_name 0124_fgsm_large_rs_wo_clamp_{8}_1.25"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_large_random.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1.375 --train_random_start --lower_ratio 0.5 --upper_ratio 1.5 --exp_name 0124_fgsm_large_rs_wo_clamp_{8}_1.375"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_large_random.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1.5 --train_random_start --lower_ratio 0.5 --upper_ratio 1.5 --exp_name 0124_fgsm_large_rs_wo_clamp_{8}_1.5"

#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_large_random.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 16 --train_fgsm_ratio 0.375 --train_random_start --lower_ratio 0.5 --upper_ratio 1.5 --exp_name 0124_fgsm_large_rs_wo_clamp_{16}_0.375"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_large_random.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 16 --train_fgsm_ratio 0.5 --train_random_start --lower_ratio 0.5 --upper_ratio 1.5 --exp_name 0124_fgsm_large_rs_wo_clamp_{16}_0.5"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_large_random.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 16 --train_fgsm_ratio 0.625 --train_random_start --lower_ratio 0.5 --upper_ratio 1.5 --exp_name 0124_fgsm_large_rs_wo_clamp_{16}_0.625"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_large_random.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 16 --train_fgsm_ratio 0.75 --train_random_start --lower_ratio 0.5 --upper_ratio 1.5 --exp_name 0124_fgsm_large_rs_wo_clamp_{16}_0.75"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_large_random.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 16 --train_fgsm_ratio 0.875 --train_random_start --lower_ratio 0.5 --upper_ratio 1.5 --exp_name 0124_fgsm_large_rs_wo_clamp_{16}_0.875"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_large_random.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 16 --train_fgsm_ratio 1 --train_random_start --lower_ratio 0.5 --upper_ratio 1.5 --exp_name 0124_fgsm_large_rs_wo_clamp_{16}_1"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_large_random.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 16 --train_fgsm_ratio 1.125 --train_random_start --lower_ratio 0.5 --upper_ratio 1.5 --exp_name 0124_fgsm_large_rs_wo_clamp_{16}_1.125"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_large_random.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 16 --train_fgsm_ratio 1.25 --train_random_start --lower_ratio 0.5 --upper_ratio 1.5 --exp_name 0124_fgsm_large_rs_wo_clamp_{16}_1.25"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_large_random.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 16 --train_fgsm_ratio 0.375 --train_random_start --lower_ratio 0.5 --upper_ratio 1 --exp_name 0124_fgsm_large_rs_wo_clamp_{16}_0.375_0.5_1"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_large_random.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 16 --train_fgsm_ratio 0.5 --train_random_start --lower_ratio 0.5 --upper_ratio 1 --exp_name 0124_fgsm_large_rs_wo_clamp_{16}_0.5_0.5_1"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_large_random.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 16 --train_fgsm_ratio 0.625 --train_random_start --lower_ratio 0.5 --upper_ratio 1 --exp_name 0124_fgsm_large_rs_wo_clamp_{16}_0.625_0.5_1"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_large_random.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 16 --train_fgsm_ratio 0.75 --train_random_start --lower_ratio 0.5 --upper_ratio 1 --exp_name 0124_fgsm_large_rs_wo_clamp_{16}_0.75_0.5_1"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_large_random.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 16 --train_fgsm_ratio 0.875 --train_random_start --lower_ratio 0.5 --upper_ratio 1 --exp_name 0124_fgsm_large_rs_wo_clamp_{16}_0.875_0.5_1"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_large_random.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 16 --train_fgsm_ratio 1 --train_random_start --lower_ratio 0.5 --upper_ratio 1 --exp_name 0124_fgsm_large_rs_wo_clamp_{16}_1_0.5_1"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_large_random.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 16 --train_fgsm_ratio 1.125 --train_random_start --lower_ratio 0.5 --upper_ratio 1 --exp_name 0124_fgsm_large_rs_wo_clamp_{16}_1.125_0.5_1"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_large_random.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 16 --train_fgsm_ratio 1.25 --train_random_start --lower_ratio 0.5 --upper_ratio 1 --exp_name 0124_fgsm_large_rs_wo_clamp_{16}_1.25_0.5_1"

#bsub -n 8 -W 24:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_eval_batch.py --model PreActResNet18 --num_epochs 200 --epsilon 8 --train_fgsm_ratio 1 --train_random_start --clamp --exp_name 0127_fgsm_rs_batch_w_clamp_{8}_1"
#bsub -n 8 -W 24:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_eval_batch.py --model PreActResNet18 --num_epochs 200 --epsilon 8 --train_fgsm_ratio 1 --train_random_start --exp_name 0127_fgsm_rs_batch_wo_clamp_{8}_1"
#bsub -n 8 -W 24:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_eval_batch.py --model PreActResNet18 --num_epochs 200 --epsilon 8 --train_fgsm_ratio 1 --exp_name 0127_fgsm_wors_batch_{8}_1"

#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_rs_test1.py --model PreActResNet18 --num_epochs 200 --epsilon 8 --train_fgsm_ratio 1 --train_random_start --exp_name 0129_fgsm_rs_test1_wo_clamp_{8}_1"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_test1_rs.py --model PreActResNet18 --num_epochs 200 --epsilon 8 --train_fgsm_ratio 1 --train_random_start --exp_name 0129_fgsm_test1_rs_wo_clamp_{8}_1"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_test1_rs.py --model PreActResNet18 --num_epochs 200 --epsilon 8 --train_fgsm_ratio 1 --train_random_start --exp_name 0129_fgsm_test1_rs_wo_clamp_{8}_1_exp1"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_test1_rs.py --model PreActResNet18 --num_epochs 200 --epsilon 8 --train_fgsm_ratio 1 --exp_name 0131_fgsm_test1_rs_wo_clamp_{8}_1_exp2"

#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --train_random_start --finetune --resumed_model_name 0129_fgsm_test1_rs_wo_clamp_{8}_1_46.pth --exp_name 0130_resume_rs_0129_fgsm_test1_rs_wo_clamp_{8}_1_46_exp1"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --train_random_start --finetune --resumed_model_name 0129_fgsm_test1_rs_wo_clamp_{8}_1_46.pth --exp_name 0130_resume_rs_0129_fgsm_test1_rs_wo_clamp_{8}_1_46_exp2"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --train_random_start --finetune --resumed_model_name 0129_fgsm_test1_rs_wo_clamp_{8}_1_46.pth --exp_name 0130_resume_rs_0129_fgsm_test1_rs_wo_clamp_{8}_1_46_exp3"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --train_random_start --finetune --resumed_model_name 0129_fgsm_test1_rs_wo_clamp_{8}_1_47.pth --exp_name 0130_resume_rs_0129_fgsm_test1_rs_wo_clamp_{8}_1_47_exp1"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --train_random_start --finetune --resumed_model_name 0129_fgsm_test1_rs_wo_clamp_{8}_1_47.pth --exp_name 0130_resume_rs_0129_fgsm_test1_rs_wo_clamp_{8}_1_47_exp2"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --train_random_start --finetune --resumed_model_name 0129_fgsm_test1_rs_wo_clamp_{8}_1_47.pth --exp_name 0130_resume_rs_0129_fgsm_test1_rs_wo_clamp_{8}_1_47_exp3"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --train_random_start --finetune --resumed_model_name 0129_fgsm_test1_rs_wo_clamp_{8}_1_48.pth --exp_name 0130_resume_rs_0129_fgsm_test1_rs_wo_clamp_{8}_1_48_exp1"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --train_random_start --finetune --resumed_model_name 0129_fgsm_test1_rs_wo_clamp_{8}_1_48.pth --exp_name 0130_resume_rs_0129_fgsm_test1_rs_wo_clamp_{8}_1_48_exp2"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --train_random_start --finetune --resumed_model_name 0129_fgsm_test1_rs_wo_clamp_{8}_1_48.pth --exp_name 0130_resume_rs_0129_fgsm_test1_rs_wo_clamp_{8}_1_48_exp3"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --train_random_start --finetune --resumed_model_name 0129_fgsm_test1_rs_wo_clamp_{8}_1_70.pth --exp_name 0130_resume_rs_0129_fgsm_test1_rs_wo_clamp_{8}_1_70_exp1"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --train_random_start --finetune --resumed_model_name 0129_fgsm_test1_rs_wo_clamp_{8}_1_70.pth --exp_name 0130_resume_rs_0129_fgsm_test1_rs_wo_clamp_{8}_1_70_exp2"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --train_random_start --finetune --resumed_model_name 0129_fgsm_test1_rs_wo_clamp_{8}_1_70.pth --exp_name 0130_resume_rs_0129_fgsm_test1_rs_wo_clamp_{8}_1_70_exp3"
#
#
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_boundary.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --train_random_start --finetune --resumed_model_name 0129_fgsm_test1_rs_wo_clamp_{8}_1_46.pth --exp_name 0130_resume_bound_0129_fgsm_test1_rs_wo_clamp_{8}_1_46_exp1"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_boundary.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --train_random_start --finetune --resumed_model_name 0129_fgsm_test1_rs_wo_clamp_{8}_1_46.pth --exp_name 0130_resume_bound_0129_fgsm_test1_rs_wo_clamp_{8}_1_46_exp2"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_boundary.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --train_random_start --finetune --resumed_model_name 0129_fgsm_test1_rs_wo_clamp_{8}_1_46.pth --exp_name 0130_resume_bound_0129_fgsm_test1_rs_wo_clamp_{8}_1_46_exp3"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_boundary.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --train_random_start --finetune --resumed_model_name 0129_fgsm_test1_rs_wo_clamp_{8}_1_47.pth --exp_name 0130_resume_bound_0129_fgsm_test1_rs_wo_clamp_{8}_1_47_exp1"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_boundary.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --train_random_start --finetune --resumed_model_name 0129_fgsm_test1_rs_wo_clamp_{8}_1_47.pth --exp_name 0130_resume_bound_0129_fgsm_test1_rs_wo_clamp_{8}_1_47_exp2"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_boundary.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --train_random_start --finetune --resumed_model_name 0129_fgsm_test1_rs_wo_clamp_{8}_1_47.pth --exp_name 0130_resume_bound_0129_fgsm_test1_rs_wo_clamp_{8}_1_47_exp3"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_boundary.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --train_random_start --finetune --resumed_model_name 0129_fgsm_test1_rs_wo_clamp_{8}_1_48.pth --exp_name 0130_resume_bound_0129_fgsm_test1_rs_wo_clamp_{8}_1_48_exp1"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_boundary.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --train_random_start --finetune --resumed_model_name 0129_fgsm_test1_rs_wo_clamp_{8}_1_48.pth --exp_name 0130_resume_bound_0129_fgsm_test1_rs_wo_clamp_{8}_1_48_exp2"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_boundary.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --train_random_start --finetune --resumed_model_name 0129_fgsm_test1_rs_wo_clamp_{8}_1_48.pth --exp_name 0130_resume_bound_0129_fgsm_test1_rs_wo_clamp_{8}_1_48_exp3"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_boundary.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --train_random_start --finetune --resumed_model_name 0129_fgsm_test1_rs_wo_clamp_{8}_1_70.pth --exp_name 0130_resume_bound_0129_fgsm_test1_rs_wo_clamp_{8}_1_70_exp1"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_boundary.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --train_random_start --finetune --resumed_model_name 0129_fgsm_test1_rs_wo_clamp_{8}_1_70.pth --exp_name 0130_resume_bound_0129_fgsm_test1_rs_wo_clamp_{8}_1_70_exp2"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_boundary.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --train_random_start --finetune --resumed_model_name 0129_fgsm_test1_rs_wo_clamp_{8}_1_70.pth --exp_name 0130_resume_bound_0129_fgsm_test1_rs_wo_clamp_{8}_1_70_exp3"

#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_diff_rs_rs.py --model PreActResNet18 --num_epochs 200 --epsilon 8 --train_fgsm_ratio 1 --train_random_start --exp_name 0130_fgsm_diff_rs_rs_wo_clamp_{8}_1"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_diff_rs_rs.py --model PreActResNet18 --num_epochs 200 --epsilon 8 --train_fgsm_ratio 1 --exp_name 0131_fgsm_diff_rs_rs_wo_clamp_{8}_1_exp2"

#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_wors_rs.py --model PreActResNet18 --num_epochs 200 --epsilon 8 --train_fgsm_ratio 1 --exp_name 0130_fgsm_wors_rs_wo_clamp_{8}_1"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_wors_rs.py --model PreActResNet18 --num_epochs 200 --epsilon 8 --train_fgsm_ratio 1 --exp_name 0131_fgsm_wors_rs_wo_clamp_{8}_1_exp2"

#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_boundary.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --random_ratio 0.5 --train_fgsm_ratio 0.125 --train_random_start --clamp --exp_name 0204_fgsm_half_bound_{8}_0.125"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_boundary.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --random_ratio 0.5 --train_fgsm_ratio 0.25 --train_random_start --clamp --exp_name 0204_fgsm_half_bound_{8}_0.25"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_boundary.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --random_ratio 0.5 --train_fgsm_ratio 0.375 --train_random_start --clamp --exp_name 0204_fgsm_half_bound_{8}_0.375"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_boundary.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --random_ratio 0.5 --train_fgsm_ratio 0.5 --train_random_start --clamp --exp_name 0204_fgsm_half_bound_{8}_0.5"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_boundary.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --random_ratio 0.5 --train_fgsm_ratio 0.625 --train_random_start --clamp --exp_name 0204_fgsm_half_bound_{8}_0.625"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_boundary.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --random_ratio 0.5 --train_fgsm_ratio 0.75 --train_random_start --clamp --exp_name 0204_fgsm_half_bound_{8}_0.75"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_boundary.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --random_ratio 0.5 --train_fgsm_ratio 0.875 --train_random_start --clamp --exp_name 0204_fgsm_half_bound_{8}_0.875"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_boundary.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --random_ratio 0.5 --train_fgsm_ratio 1 --train_random_start --clamp --exp_name 0204_fgsm_half_bound_{8}_1"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_boundary.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --random_ratio 0.5 --train_fgsm_ratio 1.125 --train_random_start --clamp --exp_name 0204_fgsm_half_bound_{8}_1.125"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_boundary.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --random_ratio 0.5 --train_fgsm_ratio 1.25 --train_random_start --clamp --exp_name 0204_fgsm_half_bound_{8}_1.25"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_boundary.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --random_ratio 0.5 --train_fgsm_ratio 1.375 --train_random_start --clamp --exp_name 0204_fgsm_half_bound_{8}_1.375"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_boundary.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --random_ratio 0.5 --train_fgsm_ratio 1.5 --train_random_start --clamp --exp_name 0204_fgsm_half_bound_{8}_1.5"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_boundary.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --random_ratio 0.5 --train_fgsm_ratio 1.625 --train_random_start --clamp --exp_name 0204_fgsm_half_bound_{8}_1.625"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_boundary.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --random_ratio 0.5 --train_fgsm_ratio 1.75 --train_random_start --clamp --exp_name 0204_fgsm_half_bound_{8}_1.75"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_boundary.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --random_ratio 0.5 --train_fgsm_ratio 1.875 --train_random_start --clamp --exp_name 0204_fgsm_half_bound_{8}_1.875"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_boundary.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --random_ratio 0.5 --train_fgsm_ratio 2 --train_random_start --clamp --exp_name 0204_fgsm_half_bound_{8}_2"
#
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_boundary.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --random_ratio 0.5 --train_fgsm_ratio 0.125 --train_random_start --exp_name 0204_fgsm_half_bound_wo_clamp_{8}_0.125"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_boundary.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --random_ratio 0.5 --train_fgsm_ratio 0.25 --train_random_start --exp_name 0204_fgsm_half_bound_wo_clamp_{8}_0.25"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_boundary.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --random_ratio 0.5 --train_fgsm_ratio 0.375 --train_random_start --exp_name 0204_fgsm_half_bound_wo_clamp_{8}_0.375"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_boundary.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --random_ratio 0.5 --train_fgsm_ratio 0.5 --train_random_start --exp_name 0204_fgsm_half_bound_wo_clamp_{8}_0.5"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_boundary.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --random_ratio 0.5 --train_fgsm_ratio 0.625 --train_random_start --exp_name 0204_fgsm_half_bound_wo_clamp_{8}_0.625"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_boundary.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --random_ratio 0.5 --train_fgsm_ratio 0.75 --train_random_start --exp_name 0204_fgsm_half_bound_wo_clamp_{8}_0.75"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_boundary.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --random_ratio 0.5 --train_fgsm_ratio 0.875 --train_random_start --exp_name 0204_fgsm_half_bound_wo_clamp_{8}_0.875"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_boundary.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --random_ratio 0.5 --train_fgsm_ratio 1 --train_random_start --exp_name 0204_fgsm_half_bound_wo_clamp_{8}_1"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_boundary.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --random_ratio 0.5 --train_fgsm_ratio 1.125 --train_random_start --exp_name 0204_fgsm_half_bound_wo_clamp_{8}_1.125"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_boundary.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --random_ratio 0.5 --train_fgsm_ratio 1.25 --train_random_start --exp_name 0204_fgsm_half_bound_wo_clamp_{8}_1.25"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_boundary.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --random_ratio 0.5 --train_fgsm_ratio 1.375 --train_random_start --exp_name 0204_fgsm_half_bound_wo_clamp_{8}_1.375"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_boundary.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --random_ratio 0.5 --train_fgsm_ratio 1.5 --train_random_start --exp_name 0204_fgsm_half_bound_wo_clamp_{8}_1.5"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_boundary.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --random_ratio 0.5 --train_fgsm_ratio 1.625 --train_random_start --exp_name 0204_fgsm_half_bound_wo_clamp_{8}_1.625"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_boundary.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --random_ratio 0.5 --train_fgsm_ratio 1.75 --train_random_start --exp_name 0204_fgsm_half_bound_wo_clamp_{8}_1.75"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_boundary.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --random_ratio 0.5 --train_fgsm_ratio 1.875 --train_random_start --exp_name 0204_fgsm_half_bound_wo_clamp_{8}_1.875"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_boundary.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --random_ratio 0.5 --train_fgsm_ratio 2 --train_random_start --exp_name 0204_fgsm_half_bound_wo_clamp_{8}_2"

#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=1]"  "module load $PCOMMAND; python train_mnist.py --epochs 10 --attack fgsm_uni --epsilon 0.3 --alpha_ratio 1 --clamp --pgd_attack_iters 40 --pgd_alpha_value 1e-2 --pgd_restarts 1 --exp_name 0205_mnist_uniform_clamp"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=1]"  "module load $PCOMMAND; python train_mnist.py --epochs 10 --attack fgsm_uni --epsilon 0.3 --alpha_ratio 1 --clamp --pgd_attack_iters 40 --pgd_alpha_value 1e-2 --pgd_restarts 1 --seed 1 --exp_name 0205_mnist_uniform_clamp_exp1"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=1]"  "module load $PCOMMAND; python train_mnist.py --epochs 10 --attack fgsm_uni --epsilon 0.3 --alpha_ratio 1 --clamp --pgd_attack_iters 40 --pgd_alpha_value 1e-2 --pgd_restarts 1 --seed 10 --exp_name 0205_mnist_uniform_clamp_exp2"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=1]"  "module load $PCOMMAND; python train_mnist.py --epochs 10 --attack fgsm_uni --epsilon 0.3 --alpha_ratio 1 --clamp --pgd_attack_iters 40 --pgd_alpha_value 1e-2 --pgd_restarts 1 --seed 100 --exp_name 0205_mnist_uniform_clamp_exp3"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=1]"  "module load $PCOMMAND; python train_mnist.py --epochs 10 --attack fgsm_uni --epsilon 0.3 --alpha_ratio 1 --clamp --pgd_attack_iters 40 --pgd_alpha_value 1e-2 --pgd_restarts 1 --seed 1000 --exp_name 0205_mnist_uniform_clamp_exp4"

#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=1]"  "module load $PCOMMAND; python train_mnist.py --epochs 10 --attack fgsm_uni --epsilon 0.3 --alpha_ratio 0.5 --clamp --pgd_attack_iters 40 --pgd_alpha_value 1e-2 --pgd_restarts 1 --exp_name 0205_mnist_uniform_0.5_clamp"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=1]"  "module load $PCOMMAND; python train_mnist.py --epochs 10 --attack fgsm_uni --epsilon 0.3 --alpha_ratio 0.5 --clamp --pgd_attack_iters 40 --pgd_alpha_value 1e-2 --pgd_restarts 1 --seed 1 --exp_name 0205_mnist_uniform_0.5_clamp_exp1"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=1]"  "module load $PCOMMAND; python train_mnist.py --epochs 10 --attack fgsm_uni --epsilon 0.3 --alpha_ratio 0.5 --clamp --pgd_attack_iters 40 --pgd_alpha_value 1e-2 --pgd_restarts 1 --seed 10 --exp_name 0205_mnist_uniform_0.5_clamp_exp2"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=1]"  "module load $PCOMMAND; python train_mnist.py --epochs 10 --attack fgsm_uni --epsilon 0.3 --alpha_ratio 0.5 --clamp --pgd_attack_iters 40 --pgd_alpha_value 1e-2 --pgd_restarts 1 --seed 100 --exp_name 0205_mnist_uniform_0.5_clamp_exp3"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=1]"  "module load $PCOMMAND; python train_mnist.py --epochs 10 --attack fgsm_uni --epsilon 0.3 --alpha_ratio 0.5 --clamp --pgd_attack_iters 40 --pgd_alpha_value 1e-2 --pgd_restarts 1 --seed 1000 --exp_name 0205_mnist_uniform_0.5_clamp_exp4"

#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=1]"  "module load $PCOMMAND; python train_mnist.py --epochs 10 --attack fgsm_half_bound --epsilon 0.3 --alpha_ratio 0.5 --clamp --pgd_attack_iters 40 --pgd_alpha_value 1e-2 --pgd_restarts 1 --exp_name 0205_mnist_half_bound_clamp"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=1]"  "module load $PCOMMAND; python train_mnist.py --epochs 10 --attack fgsm_half_bound --epsilon 0.3 --alpha_ratio 0.5 --clamp --pgd_attack_iters 40 --pgd_alpha_value 1e-2 --pgd_restarts 1 --seed 1 --exp_name 0205_mnist_half_bound_clamp_exp1"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=1]"  "module load $PCOMMAND; python train_mnist.py --epochs 10 --attack fgsm_half_bound --epsilon 0.3 --alpha_ratio 0.5 --clamp --pgd_attack_iters 40 --pgd_alpha_value 1e-2 --pgd_restarts 1 --seed 10 --exp_name 0205_mnist_half_bound_clamp_exp2"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=1]"  "module load $PCOMMAND; python train_mnist.py --epochs 10 --attack fgsm_half_bound --epsilon 0.3 --alpha_ratio 0.5 --clamp --pgd_attack_iters 40 --pgd_alpha_value 1e-2 --pgd_restarts 1 --seed 100 --exp_name 0205_mnist_half_bound_clamp_exp3"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=1]"  "module load $PCOMMAND; python train_mnist.py --epochs 10 --attack fgsm_half_bound --epsilon 0.3 --alpha_ratio 0.5 --clamp --pgd_attack_iters 40 --pgd_alpha_value 1e-2 --pgd_restarts 1 --seed 1000 --exp_name 0205_mnist_half_bound_clamp_exp4"

#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=1]"  "module load $PCOMMAND; python train_mnist.py --epochs 10 --attack fgsm_half_bound --epsilon 0.3 --alpha_ratio 1 --clamp --pgd_attack_iters 40 --pgd_alpha_value 1e-2 --pgd_restarts 1 --exp_name 0205_mnist_half_bound_1_clamp"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=1]"  "module load $PCOMMAND; python train_mnist.py --epochs 10 --attack fgsm_half_bound --epsilon 0.3 --alpha_ratio 1 --clamp --pgd_attack_iters 40 --pgd_alpha_value 1e-2 --pgd_restarts 1 --seed 1 --exp_name 0205_mnist_half_bound_1_clamp_exp1"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=1]"  "module load $PCOMMAND; python train_mnist.py --epochs 10 --attack fgsm_half_bound --epsilon 0.3 --alpha_ratio 1 --clamp --pgd_attack_iters 40 --pgd_alpha_value 1e-2 --pgd_restarts 1 --seed 10 --exp_name 0205_mnist_half_bound_1_clamp_exp2"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=1]"  "module load $PCOMMAND; python train_mnist.py --epochs 10 --attack fgsm_half_bound --epsilon 0.3 --alpha_ratio 1 --clamp --pgd_attack_iters 40 --pgd_alpha_value 1e-2 --pgd_restarts 1 --seed 100 --exp_name 0205_mnist_half_bound_1_clamp_exp3"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=1]"  "module load $PCOMMAND; python train_mnist.py --epochs 10 --attack fgsm_half_bound --epsilon 0.3 --alpha_ratio 1 --clamp --pgd_attack_iters 40 --pgd_alpha_value 1e-2 --pgd_restarts 1 --seed 1000 --exp_name 0205_mnist_half_bound_1_clamp_exp4"

#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=1]"  "module load $PCOMMAND; python train_mnist.py --epochs 10 --attack fgsm_bound --epsilon 0.3 --alpha_ratio 0.5 --clamp --pgd_attack_iters 40 --pgd_alpha_value 1e-2 --pgd_restarts 1 --exp_name 0205_mnist_bound_clamp"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=1]"  "module load $PCOMMAND; python train_mnist.py --epochs 10 --attack fgsm_bound --epsilon 0.3 --alpha_ratio 0.5 --clamp --pgd_attack_iters 40 --pgd_alpha_value 1e-2 --pgd_restarts 1 --seed 1 --exp_name 0205_mnist_bound_clamp_exp1"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=1]"  "module load $PCOMMAND; python train_mnist.py --epochs 10 --attack fgsm_bound --epsilon 0.3 --alpha_ratio 0.5 --clamp --pgd_attack_iters 40 --pgd_alpha_value 1e-2 --pgd_restarts 1 --seed 10 --exp_name 0205_mnist_bound_clamp_exp2"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=1]"  "module load $PCOMMAND; python train_mnist.py --epochs 10 --attack fgsm_bound --epsilon 0.3 --alpha_ratio 0.5 --clamp --pgd_attack_iters 40 --pgd_alpha_value 1e-2 --pgd_restarts 1 --seed 100 --exp_name 0205_mnist_bound_clamp_exp3"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=1]"  "module load $PCOMMAND; python train_mnist.py --epochs 10 --attack fgsm_bound --epsilon 0.3 --alpha_ratio 0.5 --clamp --pgd_attack_iters 40 --pgd_alpha_value 1e-2 --pgd_restarts 1 --seed 1000 --exp_name 0205_mnist_bound_clamp_exp4"

#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=1]"  "module load $PCOMMAND; python train_mnist.py --epochs 10 --attack fgsm_bound --epsilon 0.3 --alpha_ratio 1 --clamp --pgd_attack_iters 40 --pgd_alpha_value 1e-2 --pgd_restarts 1 --exp_name 0205_mnist_bound_1_clamp"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=1]"  "module load $PCOMMAND; python train_mnist.py --epochs 10 --attack fgsm_bound --epsilon 0.3 --alpha_ratio 1 --clamp --pgd_attack_iters 40 --pgd_alpha_value 1e-2 --pgd_restarts 1 --seed 1 --exp_name 0205_mnist_bound_1_clamp_exp1"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=1]"  "module load $PCOMMAND; python train_mnist.py --epochs 10 --attack fgsm_bound --epsilon 0.3 --alpha_ratio 1 --clamp --pgd_attack_iters 40 --pgd_alpha_value 1e-2 --pgd_restarts 1 --seed 10 --exp_name 0205_mnist_bound_1_clamp_exp2"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=1]"  "module load $PCOMMAND; python train_mnist.py --epochs 10 --attack fgsm_bound --epsilon 0.3 --alpha_ratio 1 --clamp --pgd_attack_iters 40 --pgd_alpha_value 1e-2 --pgd_restarts 1 --seed 100 --exp_name 0205_mnist_bound_1_clamp_exp3"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=1]"  "module load $PCOMMAND; python train_mnist.py --epochs 10 --attack fgsm_bound --epsilon 0.3 --alpha_ratio 1 --clamp --pgd_attack_iters 40 --pgd_alpha_value 1e-2 --pgd_restarts 1 --seed 1000 --exp_name 0205_mnist_bound_1_clamp_exp4"

#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=1]"  "module load $PCOMMAND; python train_mnist.py --epochs 10 --attack fgsm_bound --epsilon 0.3 --alpha_ratio 1.5 --clamp --pgd_attack_iters 40 --pgd_alpha_value 1e-2 --pgd_restarts 1 --exp_name 0205_mnist_bound_1.5_clamp"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=1]"  "module load $PCOMMAND; python train_mnist.py --epochs 10 --attack fgsm_bound --epsilon 0.3 --alpha_ratio 1.5 --clamp --pgd_attack_iters 40 --pgd_alpha_value 1e-2 --pgd_restarts 1 --seed 1 --exp_name 0205_mnist_bound_1.5_clamp_exp1"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=1]"  "module load $PCOMMAND; python train_mnist.py --epochs 10 --attack fgsm_bound --epsilon 0.3 --alpha_ratio 1.5 --clamp --pgd_attack_iters 40 --pgd_alpha_value 1e-2 --pgd_restarts 1 --seed 10 --exp_name 0205_mnist_bound_1.5_clamp_exp2"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=1]"  "module load $PCOMMAND; python train_mnist.py --epochs 10 --attack fgsm_bound --epsilon 0.3 --alpha_ratio 1.5 --clamp --pgd_attack_iters 40 --pgd_alpha_value 1e-2 --pgd_restarts 1 --seed 100 --exp_name 0205_mnist_bound_1.5_clamp_exp3"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=1]"  "module load $PCOMMAND; python train_mnist.py --epochs 10 --attack fgsm_bound --epsilon 0.3 --alpha_ratio 1.5 --clamp --pgd_attack_iters 40 --pgd_alpha_value 1e-2 --pgd_restarts 1 --seed 1000 --exp_name 0205_mnist_bound_1.5_clamp_exp4"

#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=1]"  "module load $PCOMMAND; python train_mnist.py --epochs 10 --attack fgsm_bound --epsilon 0.3 --alpha_ratio 1.75 --clamp --pgd_attack_iters 40 --pgd_alpha_value 1e-2 --pgd_restarts 1 --exp_name 0205_mnist_bound_1.75_clamp"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=1]"  "module load $PCOMMAND; python train_mnist.py --epochs 10 --attack fgsm_bound --epsilon 0.3 --alpha_ratio 1.75 --clamp --pgd_attack_iters 40 --pgd_alpha_value 1e-2 --pgd_restarts 1 --seed 1 --exp_name 0205_mnist_bound_1.75_clamp_exp1"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=1]"  "module load $PCOMMAND; python train_mnist.py --epochs 10 --attack fgsm_bound --epsilon 0.3 --alpha_ratio 1.75 --clamp --pgd_attack_iters 40 --pgd_alpha_value 1e-2 --pgd_restarts 1 --seed 10 --exp_name 0205_mnist_bound_1.75_clamp_exp2"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=1]"  "module load $PCOMMAND; python train_mnist.py --epochs 10 --attack fgsm_bound --epsilon 0.3 --alpha_ratio 1.75 --clamp --pgd_attack_iters 40 --pgd_alpha_value 1e-2 --pgd_restarts 1 --seed 100 --exp_name 0205_mnist_bound_1.75_clamp_exp3"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=1]"  "module load $PCOMMAND; python train_mnist.py --epochs 10 --attack fgsm_bound --epsilon 0.3 --alpha_ratio 1.75 --clamp --pgd_attack_iters 40 --pgd_alpha_value 1e-2 --pgd_restarts 1 --seed 1000 --exp_name 0205_mnist_bound_1.75_clamp_exp4"

#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=1]"  "module load $PCOMMAND; python train_mnist.py --epochs 10 --attack fgsm_bound --epsilon 0.3 --alpha_ratio 2 --clamp --pgd_attack_iters 40 --pgd_alpha_value 1e-2 --pgd_restarts 1 --exp_name 0205_mnist_bound_2_clamp"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=1]"  "module load $PCOMMAND; python train_mnist.py --epochs 10 --attack fgsm_bound --epsilon 0.3 --alpha_ratio 2 --clamp --pgd_attack_iters 40 --pgd_alpha_value 1e-2 --pgd_restarts 1 --seed 1 --exp_name 0205_mnist_bound_2_clamp_exp1"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=1]"  "module load $PCOMMAND; python train_mnist.py --epochs 10 --attack fgsm_bound --epsilon 0.3 --alpha_ratio 2 --clamp --pgd_attack_iters 40 --pgd_alpha_value 1e-2 --pgd_restarts 1 --seed 10 --exp_name 0205_mnist_bound_2_clamp_exp2"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=1]"  "module load $PCOMMAND; python train_mnist.py --epochs 10 --attack fgsm_bound --epsilon 0.3 --alpha_ratio 2 --clamp --pgd_attack_iters 40 --pgd_alpha_value 1e-2 --pgd_restarts 1 --seed 100 --exp_name 0205_mnist_bound_2_clamp_exp3"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=1]"  "module load $PCOMMAND; python train_mnist.py --epochs 10 --attack fgsm_bound --epsilon 0.3 --alpha_ratio 2 --clamp --pgd_attack_iters 40 --pgd_alpha_value 1e-2 --pgd_restarts 1 --seed 1000 --exp_name 0205_mnist_bound_2_clamp_exp4"

#
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=1]"  "module load $PCOMMAND; python train_mnist.py --epochs 10 --attack fgsm_bound --epsilon 0.3 --alpha_ratio 0.5 --pgd_attack_iters 40 --pgd_alpha_value 1e-2 --pgd_restarts 1 --exp_name 0205_mnist_bound_wo_clamp"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=1]"  "module load $PCOMMAND; python train_mnist.py --epochs 10 --attack fgsm_bound --epsilon 0.3 --alpha_ratio 0.5 --pgd_attack_iters 40 --pgd_alpha_value 1e-2 --pgd_restarts 1 --seed 1 --exp_name 0205_mnist_bound_wo_clamp_exp1"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=1]"  "module load $PCOMMAND; python train_mnist.py --epochs 10 --attack fgsm_bound --epsilon 0.3 --alpha_ratio 0.5 --pgd_attack_iters 40 --pgd_alpha_value 1e-2 --pgd_restarts 1 --seed 10 --exp_name 0205_mnist_bound_wo_clamp_exp2"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=1]"  "module load $PCOMMAND; python train_mnist.py --epochs 10 --attack fgsm_bound --epsilon 0.3 --alpha_ratio 0.5 --pgd_attack_iters 40 --pgd_alpha_value 1e-2 --pgd_restarts 1 --seed 100 --exp_name 0205_mnist_bound_wo_clamp_exp3"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=1]"  "module load $PCOMMAND; python train_mnist.py --epochs 10 --attack fgsm_bound --epsilon 0.3 --alpha_ratio 0.5 --pgd_attack_iters 40 --pgd_alpha_value 1e-2 --pgd_restarts 1 --seed 1000 --exp_name 0205_mnist_bound_wo_clamp_exp4"
#
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=1]"  "module load $PCOMMAND; python train_mnist.py --epochs 10 --attack fgsm_bound --epsilon 0.3 --alpha_ratio 1 --pgd_attack_iters 40 --pgd_alpha_value 1e-2 --pgd_restarts 1 --exp_name 0205_mnist_bound_1_wo_clamp"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=1]"  "module load $PCOMMAND; python train_mnist.py --epochs 10 --attack fgsm_bound --epsilon 0.3 --alpha_ratio 1 --pgd_attack_iters 40 --pgd_alpha_value 1e-2 --pgd_restarts 1 --seed 1 --exp_name 0205_mnist_bound_1_wo_clamp_exp1"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=1]"  "module load $PCOMMAND; python train_mnist.py --epochs 10 --attack fgsm_bound --epsilon 0.3 --alpha_ratio 1 --pgd_attack_iters 40 --pgd_alpha_value 1e-2 --pgd_restarts 1 --seed 10 --exp_name 0205_mnist_bound_1_wo_clamp_exp2"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=1]"  "module load $PCOMMAND; python train_mnist.py --epochs 10 --attack fgsm_bound --epsilon 0.3 --alpha_ratio 1 --pgd_attack_iters 40 --pgd_alpha_value 1e-2 --pgd_restarts 1 --seed 100 --exp_name 0205_mnist_bound_1_wo_clamp_exp3"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=1]"  "module load $PCOMMAND; python train_mnist.py --epochs 10 --attack fgsm_bound --epsilon 0.3 --alpha_ratio 1 --pgd_attack_iters 40 --pgd_alpha_value 1e-2 --pgd_restarts 1 --seed 1000 --exp_name 0205_mnist_bound_1_wo_clamp_exp4"
#
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=1]"  "module load $PCOMMAND; python train_mnist.py --epochs 10 --attack fgsm_uni --epsilon 0.3 --alpha_ratio 1 --clamp --pgd_attack_iters 40 --pgd_alpha_value 1e-2 --pgd_restarts 1 --exp_name 0205_mnist_uniform_clamp2"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=1]"  "module load $PCOMMAND; python train_mnist.py --epochs 10 --attack fgsm_uni --epsilon 0.3 --alpha_ratio 1 --clamp --pgd_attack_iters 40 --pgd_alpha_value 1e-2 --pgd_restarts 1 --seed 1 --exp_name 0205_mnist_uniform_clamp_exp12"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=1]"  "module load $PCOMMAND; python train_mnist.py --epochs 10 --attack fgsm_uni --epsilon 0.3 --alpha_ratio 1 --clamp --pgd_attack_iters 40 --pgd_alpha_value 1e-2 --pgd_restarts 1 --seed 10 --exp_name 0205_mnist_uniform_clamp_exp22"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=1]"  "module load $PCOMMAND; python train_mnist.py --epochs 10 --attack fgsm_uni --epsilon 0.3 --alpha_ratio 1 --clamp --pgd_attack_iters 40 --pgd_alpha_value 1e-2 --pgd_restarts 1 --seed 100 --exp_name 0205_mnist_uniform_clamp_exp32"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=1]"  "module load $PCOMMAND; python train_mnist.py --epochs 10 --attack fgsm_uni --epsilon 0.3 --alpha_ratio 1 --clamp --pgd_attack_iters 40 --pgd_alpha_value 1e-2 --pgd_restarts 1 --seed 1000 --exp_name 0205_mnist_uniform_clamp_exp42"
#
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=1]"  "module load $PCOMMAND; python train_mnist.py --epochs 10 --attack fgsm_half_bound --epsilon 0.3 --alpha_ratio 0.5 --clamp --pgd_attack_iters 40 --pgd_alpha_value 1e-2 --pgd_restarts 1 --exp_name 0205_mnist_half_bound_clamp2"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=1]"  "module load $PCOMMAND; python train_mnist.py --epochs 10 --attack fgsm_half_bound --epsilon 0.3 --alpha_ratio 0.5 --clamp --pgd_attack_iters 40 --pgd_alpha_value 1e-2 --pgd_restarts 1 --seed 1 --exp_name 0205_mnist_half_bound_clamp_exp12"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=1]"  "module load $PCOMMAND; python train_mnist.py --epochs 10 --attack fgsm_half_bound --epsilon 0.3 --alpha_ratio 0.5 --clamp --pgd_attack_iters 40 --pgd_alpha_value 1e-2 --pgd_restarts 1 --seed 10 --exp_name 0205_mnist_half_bound_clamp_exp22"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=1]"  "module load $PCOMMAND; python train_mnist.py --epochs 10 --attack fgsm_half_bound --epsilon 0.3 --alpha_ratio 0.5 --clamp --pgd_attack_iters 40 --pgd_alpha_value 1e-2 --pgd_restarts 1 --seed 100 --exp_name 0205_mnist_half_bound_clamp_exp32"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=1]"  "module load $PCOMMAND; python train_mnist.py --epochs 10 --attack fgsm_half_bound --epsilon 0.3 --alpha_ratio 0.5 --clamp --pgd_attack_iters 40 --pgd_alpha_value 1e-2 --pgd_restarts 1 --seed 1000 --exp_name 0205_mnist_half_bound_clamp_exp42"

#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=1]"  "module load $PCOMMAND; python train_mnist.py --epochs 10 --attack fgsm_bound --epsilon 0.3 --alpha_ratio 0.5 --clamp --pgd_attack_iters 40 --pgd_alpha_value 1e-2 --pgd_restarts 1 --exp_name 0205_mnist_bound_clamp2"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=1]"  "module load $PCOMMAND; python train_mnist.py --epochs 10 --attack fgsm_bound --epsilon 0.3 --alpha_ratio 0.5 --clamp --pgd_attack_iters 40 --pgd_alpha_value 1e-2 --pgd_restarts 1 --seed 1 --exp_name 0205_mnist_bound_clamp_exp12"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=1]"  "module load $PCOMMAND; python train_mnist.py --epochs 10 --attack fgsm_bound --epsilon 0.3 --alpha_ratio 0.5 --clamp --pgd_attack_iters 40 --pgd_alpha_value 1e-2 --pgd_restarts 1 --seed 10 --exp_name 0205_mnist_bound_clamp_exp22"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=1]"  "module load $PCOMMAND; python train_mnist.py --epochs 10 --attack fgsm_bound --epsilon 0.3 --alpha_ratio 0.5 --clamp --pgd_attack_iters 40 --pgd_alpha_value 1e-2 --pgd_restarts 1 --seed 100 --exp_name 0205_mnist_bound_clamp_exp32"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=1]"  "module load $PCOMMAND; python train_mnist.py --epochs 10 --attack fgsm_bound --epsilon 0.3 --alpha_ratio 0.5 --clamp --pgd_attack_iters 40 --pgd_alpha_value 1e-2 --pgd_restarts 1 --seed 1000 --exp_name 0205_mnist_bound_clamp_exp42"
#
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=1]"  "module load $PCOMMAND; python train_mnist.py --epochs 10 --attack fgsm_bound --epsilon 0.3 --alpha_ratio 1 --clamp --pgd_attack_iters 40 --pgd_alpha_value 1e-2 --pgd_restarts 1 --exp_name 0205_mnist_bound_1_clamp2"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=1]"  "module load $PCOMMAND; python train_mnist.py --epochs 10 --attack fgsm_bound --epsilon 0.3 --alpha_ratio 1 --clamp --pgd_attack_iters 40 --pgd_alpha_value 1e-2 --pgd_restarts 1 --seed 1 --exp_name 0205_mnist_bound_1_clamp_exp12"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=1]"  "module load $PCOMMAND; python train_mnist.py --epochs 10 --attack fgsm_bound --epsilon 0.3 --alpha_ratio 1 --clamp --pgd_attack_iters 40 --pgd_alpha_value 1e-2 --pgd_restarts 1 --seed 10 --exp_name 0205_mnist_bound_1_clamp_exp22"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=1]"  "module load $PCOMMAND; python train_mnist.py --epochs 10 --attack fgsm_bound --epsilon 0.3 --alpha_ratio 1 --clamp --pgd_attack_iters 40 --pgd_alpha_value 1e-2 --pgd_restarts 1 --seed 100 --exp_name 0205_mnist_bound_1_clamp_exp32"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=1]"  "module load $PCOMMAND; python train_mnist.py --epochs 10 --attack fgsm_bound --epsilon 0.3 --alpha_ratio 1 --clamp --pgd_attack_iters 40 --pgd_alpha_value 1e-2 --pgd_restarts 1 --seed 1000 --exp_name 0205_mnist_bound_1_clamp_exp42"

#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=1]"  "module load $PCOMMAND; python train_mnist.py --epochs 10 --attack fgsm_bound --epsilon 0.3 --alpha_ratio 0.5 --pgd_attack_iters 40 --pgd_alpha_value 1e-2 --pgd_restarts 1 --exp_name 0205_mnist_bound_wo_clamp2"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=1]"  "module load $PCOMMAND; python train_mnist.py --epochs 10 --attack fgsm_bound --epsilon 0.3 --alpha_ratio 0.5 --pgd_attack_iters 40 --pgd_alpha_value 1e-2 --pgd_restarts 1 --seed 1 --exp_name 0205_mnist_bound_wo_clamp_exp12"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=1]"  "module load $PCOMMAND; python train_mnist.py --epochs 10 --attack fgsm_bound --epsilon 0.3 --alpha_ratio 0.5 --pgd_attack_iters 40 --pgd_alpha_value 1e-2 --pgd_restarts 1 --seed 10 --exp_name 0205_mnist_bound_wo_clamp_exp22"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=1]"  "module load $PCOMMAND; python train_mnist.py --epochs 10 --attack fgsm_bound --epsilon 0.3 --alpha_ratio 0.5 --pgd_attack_iters 40 --pgd_alpha_value 1e-2 --pgd_restarts 1 --seed 100 --exp_name 0205_mnist_bound_wo_clamp_exp32"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=1]"  "module load $PCOMMAND; python train_mnist.py --epochs 10 --attack fgsm_bound --epsilon 0.3 --alpha_ratio 0.5 --pgd_attack_iters 40 --pgd_alpha_value 1e-2 --pgd_restarts 1 --seed 1000 --exp_name 0205_mnist_bound_wo_clamp_exp42"

#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=1]"  "module load $PCOMMAND; python train_mnist.py --epochs 10 --attack fgsm_bound --epsilon 0.3 --alpha_ratio 0.625 --pgd_attack_iters 40 --pgd_alpha_value 1e-2 --pgd_restarts 1 --exp_name 0205_mnist_bound_0.625_wo_clamp2"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=1]"  "module load $PCOMMAND; python train_mnist.py --epochs 10 --attack fgsm_bound --epsilon 0.3 --alpha_ratio 0.625 --pgd_attack_iters 40 --pgd_alpha_value 1e-2 --pgd_restarts 1 --seed 1 --exp_name 0205_mnist_bound_0.625_wo_clamp_exp12"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=1]"  "module load $PCOMMAND; python train_mnist.py --epochs 10 --attack fgsm_bound --epsilon 0.3 --alpha_ratio 0.625 --pgd_attack_iters 40 --pgd_alpha_value 1e-2 --pgd_restarts 1 --seed 10 --exp_name 0205_mnist_bound_0.625_wo_clamp_exp22"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=1]"  "module load $PCOMMAND; python train_mnist.py --epochs 10 --attack fgsm_bound --epsilon 0.3 --alpha_ratio 0.625 --pgd_attack_iters 40 --pgd_alpha_value 1e-2 --pgd_restarts 1 --seed 100 --exp_name 0205_mnist_bound_0.625_wo_clamp_exp32"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=1]"  "module load $PCOMMAND; python train_mnist.py --epochs 10 --attack fgsm_bound --epsilon 0.3 --alpha_ratio 0.625 --pgd_attack_iters 40 --pgd_alpha_value 1e-2 --pgd_restarts 1 --seed 1000 --exp_name 0205_mnist_bound_0.625_wo_clamp_exp42"

#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=1]"  "module load $PCOMMAND; python train_mnist.py --epochs 10 --attack fgsm_bound --epsilon 0.3 --alpha_ratio 0.75 --pgd_attack_iters 40 --pgd_alpha_value 1e-2 --pgd_restarts 1 --exp_name 0205_mnist_bound_0.75_wo_clamp2"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=1]"  "module load $PCOMMAND; python train_mnist.py --epochs 10 --attack fgsm_bound --epsilon 0.3 --alpha_ratio 0.75 --pgd_attack_iters 40 --pgd_alpha_value 1e-2 --pgd_restarts 1 --seed 1 --exp_name 0205_mnist_bound_0.75_wo_clamp_exp12"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=1]"  "module load $PCOMMAND; python train_mnist.py --epochs 10 --attack fgsm_bound --epsilon 0.3 --alpha_ratio 0.75 --pgd_attack_iters 40 --pgd_alpha_value 1e-2 --pgd_restarts 1 --seed 10 --exp_name 0205_mnist_bound_0.75_wo_clamp_exp22"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=1]"  "module load $PCOMMAND; python train_mnist.py --epochs 10 --attack fgsm_bound --epsilon 0.3 --alpha_ratio 0.75 --pgd_attack_iters 40 --pgd_alpha_value 1e-2 --pgd_restarts 1 --seed 100 --exp_name 0205_mnist_bound_0.75_wo_clamp_exp32"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=1]"  "module load $PCOMMAND; python train_mnist.py --epochs 10 --attack fgsm_bound --epsilon 0.3 --alpha_ratio 0.75 --pgd_attack_iters 40 --pgd_alpha_value 1e-2 --pgd_restarts 1 --seed 1000 --exp_name 0205_mnist_bound_0.75_wo_clamp_exp42"

#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=1]"  "module load $PCOMMAND; python train_mnist.py --epochs 10 --attack fgsm_bound --epsilon 0.3 --alpha_ratio 1 --pgd_attack_iters 40 --pgd_alpha_value 1e-2 --pgd_restarts 1 --exp_name 0205_mnist_bound_1_wo_clamp2"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=1]"  "module load $PCOMMAND; python train_mnist.py --epochs 10 --attack fgsm_bound --epsilon 0.3 --alpha_ratio 1 --pgd_attack_iters 40 --pgd_alpha_value 1e-2 --pgd_restarts 1 --seed 1 --exp_name 0205_mnist_bound_1_wo_clamp_exp12"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=1]"  "module load $PCOMMAND; python train_mnist.py --epochs 10 --attack fgsm_bound --epsilon 0.3 --alpha_ratio 1 --pgd_attack_iters 40 --pgd_alpha_value 1e-2 --pgd_restarts 1 --seed 10 --exp_name 0205_mnist_bound_1_wo_clamp_exp22"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=1]"  "module load $PCOMMAND; python train_mnist.py --epochs 10 --attack fgsm_bound --epsilon 0.3 --alpha_ratio 1 --pgd_attack_iters 40 --pgd_alpha_value 1e-2 --pgd_restarts 1 --seed 100 --exp_name 0205_mnist_bound_1_wo_clamp_exp32"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=1]"  "module load $PCOMMAND; python train_mnist.py --epochs 10 --attack fgsm_bound --epsilon 0.3 --alpha_ratio 1 --pgd_attack_iters 40 --pgd_alpha_value 1e-2 --pgd_restarts 1 --seed 1000 --exp_name 0205_mnist_bound_1_wo_clamp_exp42"

#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_test1.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --clamp --epsilon 8 --train_fgsm_ratio 1 --seed 0 --exp_name 0206_fgsm_test1_{8}_1"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_test1.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --seed 0 --exp_name 0213_fgsm_test1_wo_clamp_{8}_1"

#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_test2.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --seed 0 --exp_name 0206_fgsm_test2_{8}_1"


##
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_test1_and_test2.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --seed 0 --ratio 0 --exp_name 0213_fgsm_diff_rs_wo_clamp_0_{8}_1"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_test1_and_test2.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --seed 0 --ratio 0.2 --exp_name 0213_fgsm_diff_rs_wo_clamp_0.2_{8}_1"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_test1_and_test2.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --seed 0 --ratio 0.4 --exp_name 0213_fgsm_diff_rs_wo_clamp_0.4_{8}_1"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_test1_and_test2.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --seed 0 --ratio 0.6 --exp_name 0213_fgsm_diff_rs_wo_clamp_0.6_{8}_1"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_test1_and_test2.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --seed 0 --ratio 0.8 --exp_name 0213_fgsm_diff_rs_wo_clamp_0.8_{8}_1"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_test1_and_test2.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --seed 0 --ratio 1 --exp_name 0213_fgsm_diff_rs_wo_clamp_1_{8}_1"

#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_test1_and_test2.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --seed 10 --clamp --ratio 0 --exp_name 0206_fgsm_diff_rs_0_{8}_1_s10"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_test1_and_test2.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --seed 10 --clamp --ratio 0.2 --exp_name 0206_fgsm_diff_rs_0.2_{8}_1_s10"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_test1_and_test2.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --seed 10 --clamp --ratio 0.4 --exp_name 0206_fgsm_diff_rs_0.4_{8}_1_s10"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_test1_and_test2.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --seed 10 --clamp --ratio 0.6 --exp_name 0206_fgsm_diff_rs_0.6_{8}_1_s10"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_test1_and_test2.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --seed 10 --clamp --ratio 0.8 --exp_name 0206_fgsm_diff_rs_0.8_{8}_1_s10"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_test1_and_test2.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --seed 10 --clamp --ratio 1 --exp_name 0206_fgsm_diff_rs_1_{8}_1_s10"
##
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_test1_and_test2.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --seed 10 --ratio 0 --exp_name 0213_fgsm_diff_rs_wo_clamp_0_{8}_1_s10"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_test1_and_test2.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --seed 10 --ratio 0.2 --exp_name 0213_fgsm_diff_rs_wo_clamp_0.2_{8}_1_s10"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_test1_and_test2.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --seed 10 --ratio 0.4 --exp_name 0213_fgsm_diff_rs_wo_clamp_0.4_{8}_1_s10"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_test1_and_test2.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --seed 10 --ratio 0.6 --exp_name 0213_fgsm_diff_rs_wo_clamp_0.6_{8}_1_s10"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_test1_and_test2.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --seed 10 --ratio 0.8 --exp_name 0213_fgsm_diff_rs_wo_clamp_0.8_{8}_1_s10"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_test1_and_test2.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --seed 10 --ratio 1 --exp_name 0213_fgsm_diff_rs_wo_clamp_1_{8}_1_s10"

#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_mixed.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --seed 0 --change_epoch 1 --clamp --exp_name 0207_fgsm_mixed_{8}_1_1"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_mixed.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --seed 10 --change_epoch 1 --clamp --exp_name 0207_fgsm_mixed_{8}_1_1_exp2"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_mixed.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --seed 100 --change_epoch 1 --clamp --exp_name 0207_fgsm_mixed_{8}_1_1_exp3"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_mixed.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --seed 1000 --change_epoch 1 --clamp --exp_name 0207_fgsm_mixed_{8}_1_1_exp4"

#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_mixed.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --seed 0 --change_epoch 16 --clamp --exp_name 0207_fgsm_mixed_{8}_1_16"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_mixed.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --seed 0 --change_epoch 60 --clamp --exp_name 0207_fgsm_mixed_{8}_1_60"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_mixed.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --seed 0 --change_epoch 120 --clamp --exp_name 0207_fgsm_mixed_{8}_1_120"
##bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_mixed.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --seed 0 --change_epoch 16 --exp_name 0207_fgsm_mixed_wo_clamp_{8}_1_16"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_mixed.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --seed 0 --change_epoch 60 --exp_name 0207_fgsm_mixed_wo_clamp_{8}_1_60"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_mixed.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --seed 0 --change_epoch 120 --exp_name 0207_fgsm_mixed_wo_clamp_{8}_1_120"

#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_mixed.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --seed 1 --change_epoch 1 --clamp --exp_name 0207_fgsm_mixed_{8}_1_1_exp1"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_mixed.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --seed 1 --change_epoch 16 --exp_name 0207_fgsm_mixed_wo_clamp_{8}_1_16_exp1"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_mixed.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --seed 1 --change_epoch 60 --exp_name 0207_fgsm_mixed_wo_clamp_{8}_1_60_exp1"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_mixed.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --seed 1 --change_epoch 90 --exp_name 0207_fgsm_mixed_wo_clamp_{8}_1_90_exp1"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_mixed.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --seed 1 --change_epoch 201 --exp_name 0207_fgsm_mixed_wo_clamp_{8}_1_201_exp1"

#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_mixed.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --seed 10000 --change_epoch 0 --save_epoch --clamp --exp_name 0207_fgsm_mixed_{8}_1_1_exp5"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --epsilon 8 --train_random_start --train_fgsm_ratio 1 --seed 10000 --finetune --resumed_model_name 0207_fgsm_mixed_{8}_1_1_exp5_16.pth --exp_name 0207_fgsm_resumed_wo_clamp_{8}_1_16_exp5"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --epsilon 8 --train_random_start --train_fgsm_ratio 1 --seed 10000 --finetune --resumed_model_name 0207_fgsm_mixed_{8}_1_1_exp5_17.pth --exp_name 0207_fgsm_resumed_wo_clamp_{8}_1_17_exp5"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --epsilon 8 --train_random_start --train_fgsm_ratio 1 --seed 10000 --finetune --resumed_model_name 0207_fgsm_mixed_{8}_1_1_exp5_18.pth --exp_name 0207_fgsm_resumed_wo_clamp_{8}_1_18_exp5"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_mixed.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --seed 10000 --change_epoch 11 --exp_name 0207_fgsm_mixed_wo_clamp_{8}_1_11_exp5"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_mixed.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --seed 10000 --change_epoch 11 --change_type test1 --exp_name 0221_fgsm_mixed_test1_wo_clamp_{8}_1_11_exp5"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_mixed.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --seed 10000 --change_epoch 11 --change_type test2 --exp_name 0221_fgsm_mixed_test2_wo_clamp_{8}_1_11_exp5"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_mixed.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --seed 10000 --change_epoch 17 --exp_name 0207_fgsm_mixed_wo_clamp_{8}_1_17_exp5"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_mixed.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --seed 10000 --change_epoch 31 --exp_name 0207_fgsm_mixed_wo_clamp_{8}_1_31_exp5"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_mixed.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --seed 10000 --change_epoch 31 --save_epoch --exp_name 0214_fgsm_mixed_wo_clamp_{8}_1_31_exp5"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_mixed.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --seed 10000 --change_epoch 31 --save_epoch --change_type test1 --exp_name 0214_fgsm_mixed_test1_wo_clamp_{8}_1_31_exp5"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_mixed.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --seed 10000 --change_epoch 31 --save_epoch --change_type test2 --exp_name 0214_fgsm_mixed_test2_wo_clamp_{8}_1_31_exp5"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_mixed.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --seed 10000 --change_epoch 60 --exp_name 0207_fgsm_mixed_wo_clamp_{8}_1_60_exp5"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_mixed.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --seed 10000 --change_epoch 61 --exp_name 0207_fgsm_mixed_wo_clamp_{8}_1_61_exp5"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_mixed.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --seed 10000 --change_epoch 61 --save_epoch --exp_name 0214_fgsm_mixed_wo_clamp_{8}_1_61_exp5"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_mixed.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --seed 10000 --change_epoch 61 --save_epoch --change_type test1 --exp_name 0214_fgsm_mixed_test1_wo_clamp_{8}_1_61_exp5"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_mixed.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --seed 10000 --change_epoch 61 --save_epoch --change_type test2 --exp_name 0214_fgsm_mixed_test2_wo_clamp_{8}_1_61_exp5"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_mixed.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --seed 10000 --change_epoch 90 --exp_name 0207_fgsm_mixed_wo_clamp_{8}_1_90_exp5"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_mixed.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --seed 10000 --change_epoch 91 --exp_name 0207_fgsm_mixed_wo_clamp_{8}_1_91_exp5"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_mixed.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --seed 10000 --change_epoch 101 --exp_name 0207_fgsm_mixed_wo_clamp_{8}_1_101_exp5"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_mixed.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --seed 10000 --change_epoch 101 --change_type test1 --exp_name 0221_fgsm_mixed_test1_wo_clamp_{8}_1_101_exp5"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_mixed.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --seed 10000 --change_epoch 101 --change_type test2 --exp_name 0221_fgsm_mixed_test2_wo_clamp_{8}_1_101_exp5"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_mixed.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --seed 10000 --change_epoch 121 --exp_name 0207_fgsm_mixed_wo_clamp_{8}_1_121_exp5"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_mixed.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --seed 10000 --change_epoch 121 --change_type test1 --exp_name 0221_fgsm_mixed_test1_wo_clamp_{8}_1_121_exp5"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_mixed.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --seed 10000 --change_epoch 121 --change_type test2 --exp_name 0221_fgsm_mixed_test2_wo_clamp_{8}_1_121_exp5"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_mixed.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --seed 10000 --change_epoch 151 --exp_name 0207_fgsm_mixed_wo_clamp_{8}_1_151_exp5"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_mixed.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --seed 10000 --change_epoch 151 --change_type test1 --exp_name 0221_fgsm_mixed_test1_wo_clamp_{8}_1_151_exp5"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_mixed.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --seed 10000 --change_epoch 151 --change_type test2 --exp_name 0221_fgsm_mixed_test2_wo_clamp_{8}_1_151_exp5"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_mixed.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --seed 10000 --change_epoch 171 --exp_name 0207_fgsm_mixed_wo_clamp_{8}_1_171_exp5"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_mixed.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --seed 10000 --change_epoch 171 --change_type test1 --exp_name 0221_fgsm_mixed_test1_wo_clamp_{8}_1_171_exp5"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_mixed.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --seed 10000 --change_epoch 171 --change_type test2 --exp_name 0221_fgsm_mixed_test2_wo_clamp_{8}_1_171_exp5"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_mixed.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --seed 10000 --change_epoch 201 --exp_name 0207_fgsm_mixed_wo_clamp_{8}_1_201_exp5"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm.py --epsilon 8 --train_fgsm_ratio 1 --seed 10000 --finetune --resumed_model_name 0207_fgsm_mixed_{8}_1_1_exp5_16.pth --exp_name 0207_fgsm_resumed_wors_wo_clamp_{8}_1_16_exp5"

#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_frozen.py --epsilon 8 --train_fgsm_ratio 1 --seed 10000 --frozen_epoch 5 --exp_name 0209_fgsm_wors_fixed_5"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_frozen.py --epsilon 8 --train_fgsm_ratio 1 --seed 10000 --frozen_epoch 10 --exp_name 0209_fgsm_wors_fixed_10"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_frozen.py --epsilon 8 --train_fgsm_ratio 1 --seed 10000 --frozen_epoch 10 --exp_name 0209_fgsm_wors_fixed_10_exp1"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_frozen.py --epsilon 8 --train_fgsm_ratio 1 --seed 10000 --frozen_epoch 13 --exp_name 0209_fgsm_wors_fixed_13"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_frozen.py --epsilon 8 --train_fgsm_ratio 1 --seed 10000 --frozen_epoch 14 --exp_name 0209_fgsm_wors_fixed_14"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_frozen.py --epsilon 8 --train_fgsm_ratio 1 --seed 10000 --frozen_epoch 15 --exp_name 0209_fgsm_wors_fixed_15"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_frozen.py --epsilon 8 --train_fgsm_ratio 1 --seed 10000 --frozen_epoch 16 --exp_name 0209_fgsm_wors_fixed_16"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_frozen.py --epsilon 8 --train_fgsm_ratio 1 --seed 10000 --frozen_epoch 17 --exp_name 0209_fgsm_wors_fixed_17"

#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_frozen.py --epsilon 8 --train_fgsm_ratio 1 --train_random_start --seed 10000 --frozen_epoch 10 --exp_name 0209_fgsm_rs_fixed_10"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_frozen.py --epsilon 8 --train_fgsm_ratio 1 --train_random_start --seed 10000 --frozen_epoch 30 --exp_name 0209_fgsm_rs_fixed_30"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_frozen.py --epsilon 8 --train_fgsm_ratio 1 --train_random_start --seed 10000 --frozen_epoch 60 --exp_name 0209_fgsm_rs_fixed_60"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_frozen.py --epsilon 8 --train_fgsm_ratio 1 --train_random_start --seed 10000 --frozen_epoch 60 --add_random --exp_name 0214_fgsm_rs_fixed_random_60" ### without clamp
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_frozen.py --epsilon 8 --train_fgsm_ratio 1 --train_random_start --seed 10000 --frozen_epoch 90 --exp_name 0209_fgsm_rs_fixed_90"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_frozen.py --epsilon 8 --train_fgsm_ratio 1 --train_random_start --seed 10000 --frozen_epoch 100 --exp_name 0209_fgsm_rs_fixed_100"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_frozen.py --epsilon 8 --train_fgsm_ratio 1 --train_random_start --seed 10000 --frozen_epoch 120 --exp_name 0209_fgsm_rs_fixed_120"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_frozen.py --epsilon 8 --train_fgsm_ratio 1 --train_random_start --seed 10000 --frozen_epoch 150 --exp_name 0209_fgsm_rs_fixed_150"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_frozen.py --epsilon 8 --train_fgsm_ratio 1 --train_random_start --seed 10000 --frozen_epoch 170 --exp_name 0209_fgsm_rs_fixed_170"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_frozen.py --epsilon 8 --train_fgsm_ratio 1 --train_random_start --seed 10000 --frozen_epoch 30 --exp_name 0214_fgsm_rs_fixed_30"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_frozen.py --epsilon 8 --train_fgsm_ratio 1 --train_random_start --seed 10000 --frozen_epoch 30 --add_random --exp_name 0214_fgsm_rs_fixed_random_30" ### without clamp
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_frozen.py --epsilon 8 --train_fgsm_ratio 1 --train_random_start --seed 10000 --frozen_epoch 30 --add_random --exp_name 0214_fgsm_rs_fixed_random_30_exp1" ### with clamp



#
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_all.py --dataset cifar100 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --epsilon 8 --train_fgsm_ratio 0.125 --project --random_start_type none --seed 0 --exp_name 0220_cifar100_wors_8_0.125"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_all.py --dataset cifar100 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --epsilon 8 --train_fgsm_ratio 0.25 --project --random_start_type none --seed 0 --exp_name 0220_cifar100_wors_8_0.25"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_all.py --dataset cifar100 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --epsilon 8 --train_fgsm_ratio 0.375 --project --random_start_type none --seed 0 --exp_name 0220_cifar100_wors_8_0.375"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_all.py --dataset cifar100 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --epsilon 8 --train_fgsm_ratio 0.5 --project --random_start_type none --seed 0 --exp_name 0220_cifar100_wors_8_0.5"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_all.py --dataset cifar100 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --epsilon 8 --train_fgsm_ratio 0.625 --project --random_start_type none --seed 0 --exp_name 0220_cifar100_wors_8_0.625"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_all.py --dataset cifar100 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --epsilon 8 --train_fgsm_ratio 0.75 --project --random_start_type none --seed 0 --exp_name 0220_cifar100_wors_8_0.75"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_all.py --dataset cifar100 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --epsilon 8 --train_fgsm_ratio 0.875 --project --random_start_type none --seed 0 --exp_name 0220_cifar100_wors_8_0.875"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_all.py --dataset cifar100 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --epsilon 8 --train_fgsm_ratio 1 --project --random_start_type none --seed 0 --exp_name 0220_cifar100_wors_8_1"
#
#for i in 8
#do
#  for j in 1.125 1.25 1.375 1.5
#  do
#    bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_all.py --dataset cifar100 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --epsilon 8 --train_fgsm_ratio $j --project --random_start_type uniform --seed 0 --exp_name 0220_cifar100_uniform_8_$j"
#  done
#done
##
##
#for i in 8
#do
#  for j in 0.125 0.25 0.375 0.5 0.625 0.75 0.875 1 1.125 1.25 1.375 1.5
#  do
#    bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_all.py --dataset cifar100 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --epsilon 8 --train_fgsm_ratio $j --random_start_type uniform --seed 0 --exp_name 0220_cifar100_uniform_wo_proj_8_$j"
#  done
#done
##
##
#for i in 8
#do
#  for j in 0.125 0.25 0.625 0.75 0.875 1 1.125 1.25 1.375 1.625 1.75 1.875 2
#  do
#    bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_all.py --dataset cifar100 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --epsilon 8 --train_fgsm_ratio $j --project --random_start_type boundary --seed 0 --exp_name 0220_cifar100_boundary_8_$j"
#  done
#done
###
###
#for i in 8
#do
#  for j in 0.125 0.25 0.5 0.625 0.75 0.875 1 1.125 1.25 1.375 1.5
#  do
#    bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_all.py --dataset cifar100 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --epsilon 8 --train_fgsm_ratio $j --random_start_type boundary --seed 0 --exp_name 0220_cifar100_boundary_wo_proj_8_$j"
#  done
#done

###################################### 8 12 ######################################

#for i in 0.125 0.25 0.375 0.5 0.625 0.75 0.875 1
#do
#  bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_all.py --dataset svhn --model PreActResNet18 --lr_schedule cyclic --lr-min 0 --lr-max 0.05 --num_epochs 15 --epsilon 12 --train_fgsm_ratio $i --project --random_start_type none --seed 0 --exp_name 0220_svhn_wors_12_$i"
#done
#for i in 0.125 0.25 0.375 0.5 0.625 0.75 0.875 1 1.125 1.25 1.375 1.5 1.625 1.75 1.875 2
#do
#  bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_all.py --dataset svhn --model PreActResNet18 --lr_schedule cyclic --lr-min 0 --lr-max 0.05 --num_epochs 15 --epsilon 12 --train_fgsm_ratio $i --project --random_start_type uniform --seed 0 --exp_name 0220_svhn_uniform_12_$i"
#  bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_all.py --dataset svhn --model PreActResNet18 --lr_schedule cyclic --lr-min 0 --lr-max 0.05 --num_epochs 15 --epsilon 12 --train_fgsm_ratio $i --random_start_type uniform --seed 0 --exp_name 0220_svhn_uniform_wo_proj_12_$i"
#  bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_all.py --dataset svhn --model PreActResNet18 --lr_schedule cyclic --lr-min 0 --lr-max 0.05 --num_epochs 15 --epsilon 12 --train_fgsm_ratio $i --project --random_start_type boundary --seed 0 --exp_name 0220_svhn_boundary_12_$i"
#  bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_all.py --dataset svhn --model PreActResNet18 --lr_schedule cyclic --lr-min 0 --lr-max 0.05 --num_epochs 15 --epsilon 12 --train_fgsm_ratio $i --random_start_type boundary --seed 0 --exp_name 0220_svhn_boundary_wo_proj_12_$i"
#done

#for i in 0 1 2 3 4
#do
#  bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_all.py --dataset svhn --model PreActResNet18 --lr_schedule cyclic --lr-min 0 --lr-max 0.05 --num_epochs 15 --epsilon 8 --train_fgsm_ratio 1 --project --random_start_type none --seed $i --exp_name 0220_svhn_wors_8_seed_$i"
#  bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_all.py --dataset svhn --model PreActResNet18 --lr_schedule cyclic --lr-min 0 --lr-max 0.05 --num_epochs 15 --epsilon 12 --train_fgsm_ratio 1 --project --random_start_type none --seed $i --exp_name 0220_svhn_wors_12_seed_$i"
#  bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_all.py --dataset svhn --model PreActResNet18 --lr_schedule cyclic --lr-min 0 --lr-max 0.05 --num_epochs 15 --epsilon 16 --train_fgsm_ratio 1 --project --random_start_type none --seed $i --exp_name 0220_svhn_wors_16_seed_$i"

#  bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_all.py --dataset svhn --model PreActResNet18 --lr_schedule cyclic --lr-min 0 --lr-max 0.05 --num_epochs 15 --epsilon 8 --train_fgsm_ratio 1 --project --random_start_type uniform --seed $i --exp_name 0220_svhn_uniform_8_seed_$i"
#  bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_all.py --dataset svhn --model PreActResNet18 --lr_schedule cyclic --lr-min 0 --lr-max 0.05 --num_epochs 15 --epsilon 12 --train_fgsm_ratio 1 --project --random_start_type uniform --seed $i --exp_name 0220_svhn_uniform_12_seed_$i"
#  bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_all.py --dataset svhn --model PreActResNet18 --lr_schedule cyclic --lr-min 0 --lr-max 0.05 --num_epochs 15 --epsilon 16 --train_fgsm_ratio 1 --project --random_start_type uniform --seed $i --exp_name 0220_svhn_uniform_16_seed_$i"

#  bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_all.py --dataset svhn --model PreActResNet18 --lr_schedule cyclic --lr-min 0 --lr-max 0.05 --num_epochs 15 --epsilon 8 --train_fgsm_ratio 1 --random_start_type uniform --seed $i --exp_name 0220_svhn_uniform_wo_proj_8_seed_$i"
#  bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_all.py --dataset svhn --model PreActResNet18 --lr_schedule cyclic --lr-min 0 --lr-max 0.05 --num_epochs 15 --epsilon 12 --train_fgsm_ratio 1 --random_start_type uniform --seed $i --exp_name 0220_svhn_uniform_wo_proj_12_seed_$i"
#  bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_all.py --dataset svhn --model PreActResNet18 --lr_schedule cyclic --lr-min 0 --lr-max 0.05 --num_epochs 15 --epsilon 16 --train_fgsm_ratio 1 --random_start_type uniform --seed $i --exp_name 0220_svhn_uniform_wo_proj_16_seed_$i"

#  bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_all.py --dataset svhn --model PreActResNet18 --lr_schedule cyclic --lr-min 0 --lr-max 0.05 --num_epochs 15 --epsilon 8 --train_fgsm_ratio 1 --random_start_type boundary --seed $i --exp_name 0220_svhn_boundary_wo_proj_8_seed_$i"
#  bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_all.py --dataset svhn --model PreActResNet18 --lr_schedule cyclic --lr-min 0 --lr-max 0.05 --num_epochs 15 --epsilon 12 --train_fgsm_ratio 1 --random_start_type boundary --seed $i --exp_name 0220_svhn_boundary_wo_proj_12_seed_$i"
#  bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_all.py --dataset svhn --model PreActResNet18 --lr_schedule cyclic --lr-min 0 --lr-max 0.05 --num_epochs 15 --epsilon 16 --train_fgsm_ratio 1 --random_start_type boundary --seed $i --exp_name 0220_svhn_boundary_wo_proj_16_seed_$i"
#done



#for i in 1 2 3 4
#do
#  bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_all.py --dataset cifar100 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --epsilon 8 --train_fgsm_ratio 1 --project --random_start_type none --seed $i --exp_name 0220_cifar100_wors_8_seed_$i"
#  bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_all.py --dataset cifar100 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --epsilon 8 --train_fgsm_ratio 1 --project --random_start_type uniform --seed $i --exp_name 0220_cifar100_uniform_8_seed_$i"
#  bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_all.py --dataset cifar100 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --epsilon 8 --train_fgsm_ratio 1 --random_start_type uniform --seed $i --exp_name 0220_cifar100_uniform_wo_proj_8_seed_$i"
#  bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_all.py --dataset cifar100 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --epsilon 8 --train_fgsm_ratio 1 --random_start_type boundary --seed $i --exp_name 0220_cifar100_boundary_wo_proj_8_seed_$i"
#
#  bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_all.py --dataset cifar100 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --epsilon 16 --train_fgsm_ratio 1 --project --random_start_type none --seed $i --exp_name 0220_cifar100_wors_16_seed_$i"
#  bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_all.py --dataset cifar100 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --epsilon 16 --train_fgsm_ratio 1 --project --random_start_type uniform --seed $i --exp_name 0220_cifar100_uniform_16_seed_$i"
#  bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_all.py --dataset cifar100 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --epsilon 16 --train_fgsm_ratio 1 --random_start_type uniform --seed $i --exp_name 0220_cifar100_uniform_wo_proj_16_seed_$i"
#  bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_all.py --dataset cifar100 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --epsilon 16 --train_fgsm_ratio 1 --random_start_type boundary --seed $i --exp_name 0220_cifar100_boundary_wo_proj_16_seed_$i"
#done

#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_save_perturbation.py --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --epsilon 8 --train_fgsm_ratio 1 --train_random_start --seed 0 --exp_name 0221_save_perturbation"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_custom_dataset.py --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --epsilon 8 --resumed_epoch 10 --seed 0 --exp_name 0221_train_fixed_perturbation_10"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_custom_dataset.py --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --epsilon 8 --resumed_epoch 30 --seed 0 --exp_name 0221_train_fixed_perturbation_30"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_custom_dataset.py --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --epsilon 8 --resumed_epoch 60 --seed 0 --exp_name 0221_train_fixed_perturbation_60"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_custom_dataset.py --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --epsilon 8 --resumed_epoch 90 --seed 0 --exp_name 0221_train_fixed_perturbation_90"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_custom_dataset.py --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --epsilon 8 --resumed_epoch 120 --seed 0 --exp_name 0221_train_fixed_perturbation_120"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_custom_dataset.py --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --epsilon 8 --resumed_epoch 170 --seed 0 --exp_name 0221_train_fixed_perturbation_170"



############################################# used by report##################################################################
#for i in 8 16 1 2 3 4 5 6 7 9 10 11 12 13 14 15
#do
#  bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_all.py --dataset cifar10 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --epsilon $i --train_fgsm_ratio 1 --project --save_epoch --random_start_type none --seed 0 --exp_name 0226_cifar10_none_w_proj_eps_$i"
#  bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_all.py --dataset cifar10 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --epsilon $i --train_fgsm_ratio 1 --project --save_epoch --random_start_type uniform --seed 0 --exp_name 0226_cifar10_uniform_w_proj_eps_$i"
#  bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_all.py --dataset cifar10 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --epsilon $i --train_fgsm_ratio 1 --project --save_epoch --random_start_type boundary --seed 0 --exp_name 0226_cifar10_boundary1_w_proj_eps_$i"
#  bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_all.py --dataset cifar10 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --epsilon $i --train_fgsm_ratio 1.5 --project --save_epoch --random_start_type boundary --seed 0 --exp_name 0226_cifar10_boundary1.5_w_proj_eps_$i"
#  bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_all.py --dataset cifar10 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --epsilon $i --train_fgsm_ratio 1 --random_start_type uniform --seed 0 --exp_name 0226_cifar10_uniform_wo_proj_eps_$i"
#  bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_all.py --dataset cifar10 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --epsilon $i --train_fgsm_ratio 1 --random_start_type boundary --seed 0 --exp_name 0226_cifar10_boundary_wo_proj_eps_$i"
#  bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_pgd.py --dataset cifar10 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --epsilon $i --train_pgd_ratio 0.5 --train_pgd_attack_iters 2 --train_pgd_restarts 1 --save_epoch --seed 0 --exp_name 0226_cifar10_pgd2_$i"   ######## random start false
#  bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool.py --dataset cifar10 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --epsilon $i --clamp_ratio 1 --save_epoch --seed 0 --exp_name 0226_cifar10_deepfool_$i"
#  bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_deepfool.py --dataset cifar10 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --epsilon $i --clamp_ratio 1 --train_deepfool_rs --seed 0 --exp_name 0226_cifar10_deepfool_rs_$i"
#  bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_grad_align.py --dataset cifar10 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --epsilon $i --train_fgsm_ratio 1 --grad_align_cos_lambda 2 --seed 0 --exp_name 0306_cifar10_grad_align_$i"
#  bsub -n 8 -W 24:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_pgd.py --dataset cifar10 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --epsilon $i --train_pgd_ratio 0.2 --train_pgd_attack_iters 10 --train_pgd_restarts 1 --seed 0 --exp_name 0306_cifar10_pgd10_$i"
#done
#
#
#for i in 0.125 0.25 0.375 0.5 0.625 0.75 0.875 1
#do
#  bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_all.py --dataset cifar10 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --epsilon 8 --train_fgsm_ratio $i --project --save_epoch --random_start_type none --seed 0 --exp_name 0226_cifar10_none_w_proj_step_$i"
#done
#

#for i in 0.125 0.25 0.375 0.5 0.625 0.75 0.875 1 1.125 1.25 1.375 1.5 1.625 1.75 1.875 2
#do
#  bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_all.py --dataset cifar10 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --epsilon 8 --train_fgsm_ratio $i --project --save_epoch --random_start_type uniform --seed 0 --exp_name 0226_cifar10_uniform_w_proj_step_$i"
#  bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_all.py --dataset cifar10 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --epsilon 8 --train_fgsm_ratio $i --random_start_type uniform --seed 0 --exp_name 0306_cifar10_uniform_wo_proj_step_$i"
#  bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_all.py --dataset cifar10 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --epsilon 16 --train_fgsm_ratio $i --project --save_epoch --random_start_type uniform --seed 0 --exp_name 0226_cifar10_uniform_w_proj_16_step_$i"
#  bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_all.py --dataset cifar10 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --epsilon 8 --train_fgsm_ratio $i --project --random_start_type boundary --seed 0 --exp_name 0306_cifar10_boundary_w_proj_step_$i"
#  bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_all.py --dataset cifar10 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --epsilon 8 --train_fgsm_ratio $i --random_start_type boundary --seed 0 --exp_name 0306_cifar10_boundary_wo_proj_step_$i"
#done

#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate_diff_fgsm.py --model PreActResNet18 --epsilon 8 --interval 1 --resumed_model_name 0226_cifar10_none_w_proj_eps_8 --exp_name diff_fgsm_0226_cifar10_none_w_proj_eps_8"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate_diff_fgsm.py --model PreActResNet18 --epsilon 8 --interval 1 --resumed_model_name 0226_cifar10_deepfool_8 --exp_name diff_fgsm_0226_cifar10_deepfool_8"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate_diff_deepfool_fgsm.py --model PreActResNet18 --epsilon 8 --interval 1 --resumed_model_name 0226_cifar10_none_w_proj_eps_8 --exp_name diff_deepfool_0226_cifar10_none_w_proj_eps_8"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate_diff_deepfool_fgsm.py --model PreActResNet18 --epsilon 8 --interval 1 --resumed_model_name 0226_cifar10_deepfool_8 --exp_name diff_deepfool_0226_cifar10_deepfool_8"
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate_diff_deepfool.py --model PreActResNet18 --epsilon 8 --interval 1 --resumed_model_name 0226_cifar10_none_w_proj_eps_8 --exp_name diff_df_df_0226_cifar10_none_w_proj_eps_8"   #diff_fgsm_0226_cifar10_deepfool_8
#bsub -n 8 -W 04:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python evaluate_diff_deepfool.py --model PreActResNet18 --epsilon 8 --interval 1 --resumed_model_name 0226_cifar10_deepfool_8 --exp_name diff_df_df_0226_cifar10_deepfool_8"                       ########## evaluate accuracy under different perturbation length, chapter 4

#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_extend.py --dataset cifar10 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --project --epsilon 8 --train_fgsm_ratio 0.25 --train_random_start --seed 0 --exp_name 0304_cifar10_8_0.25"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_extend.py --dataset cifar10 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --project --epsilon 8 --train_fgsm_ratio 0.5 --train_random_start --seed 0 --exp_name 0304_cifar10_8_0.5"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_extend.py --dataset cifar10 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --project --epsilon 8 --train_fgsm_ratio 0.75 --train_random_start --seed 0 --exp_name 0304_cifar10_8_0.75"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_extend.py --dataset cifar10 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --project --epsilon 8 --train_fgsm_ratio 1 --train_random_start --seed 0 --exp_name 0304_cifar10_8_1"
#
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_extend.py --dataset cifar10 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --project --epsilon 10 --train_fgsm_ratio 0.25 --train_random_start --seed 0 --exp_name 0304_cifar10_10_0.25"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_extend.py --dataset cifar10 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --project --epsilon 10 --train_fgsm_ratio 0.5 --train_random_start --seed 0 --exp_name 0304_cifar10_10_0.5"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_extend.py --dataset cifar10 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --project --epsilon 10 --train_fgsm_ratio 0.75 --train_random_start --seed 0 --exp_name 0304_cifar10_10_0.75"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_extend.py --dataset cifar10 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --project --epsilon 10 --train_fgsm_ratio 1 --train_random_start --seed 0 --exp_name 0304_cifar10_10_1"
#
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_extend.py --dataset cifar10 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --project --epsilon 12 --train_fgsm_ratio 0.25 --train_random_start --seed 0 --exp_name 0304_cifar10_12_0.25"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_extend.py --dataset cifar10 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --project --epsilon 12 --train_fgsm_ratio 0.5 --train_random_start --seed 0 --exp_name 0304_cifar10_12_0.5"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_extend.py --dataset cifar10 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --project --epsilon 12 --train_fgsm_ratio 0.75 --train_random_start --seed 0 --exp_name 0304_cifar10_12_0.75"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_extend.py --dataset cifar10 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --project --epsilon 12 --train_fgsm_ratio 1 --train_random_start --seed 0 --exp_name 0304_cifar10_12_1"   ###########with the gradient of random input

#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_extend.py --dataset cifar10 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --project --epsilon 6 --train_fgsm_ratio 0.25 --train_random_start --seed 0 --exp_name 0306_cifar10_6_0.25"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_extend.py --dataset cifar10 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --project --epsilon 6 --train_fgsm_ratio 0.5 --train_random_start --seed 0 --exp_name 0306_cifar10_6_0.5"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_extend.py --dataset cifar10 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --project --epsilon 6 --train_fgsm_ratio 0.75 --train_random_start --seed 0 --exp_name 0306_cifar10_6_0.75"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_extend.py --dataset cifar10 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --project --epsilon 6 --train_fgsm_ratio 1 --train_random_start --seed 0 --exp_name 0306_cifar10_6_1"

#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_extend.py --dataset cifar10 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --project --epsilon 8 --train_fgsm_ratio 0.25 --train_random_start --seed 0 --exp_name 0306_cifar10_8_0.25"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_extend.py --dataset cifar10 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --project --epsilon 8 --train_fgsm_ratio 0.5 --train_random_start --seed 0 --exp_name 0306_cifar10_8_0.5"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_extend.py --dataset cifar10 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --project --epsilon 8 --train_fgsm_ratio 0.75 --train_random_start --seed 0 --exp_name 0306_cifar10_8_0.75"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_extend.py --dataset cifar10 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --project --epsilon 8 --train_fgsm_ratio 1 --train_random_start --seed 0 --exp_name 0306_cifar10_8_1"
#
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_extend.py --dataset cifar10 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --project --epsilon 10 --train_fgsm_ratio 0.25 --train_random_start --seed 0 --exp_name 0306_cifar10_10_0.25"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_extend.py --dataset cifar10 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --project --epsilon 10 --train_fgsm_ratio 0.5 --train_random_start --seed 0 --exp_name 0306_cifar10_10_0.5"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_extend.py --dataset cifar10 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --project --epsilon 10 --train_fgsm_ratio 0.75 --train_random_start --seed 0 --exp_name 0306_cifar10_10_0.75"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_extend.py --dataset cifar10 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --project --epsilon 10 --train_fgsm_ratio 1 --train_random_start --seed 0 --exp_name 0306_cifar10_10_1"
#
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_extend.py --dataset cifar10 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --project --epsilon 12 --train_fgsm_ratio 0.25 --train_random_start --seed 0 --exp_name 0306_cifar10_12_0.25"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_extend.py --dataset cifar10 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --project --epsilon 12 --train_fgsm_ratio 0.5 --train_random_start --seed 0 --exp_name 0306_cifar10_12_0.5"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_extend.py --dataset cifar10 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --project --epsilon 12 --train_fgsm_ratio 0.75 --train_random_start --seed 0 --exp_name 0306_cifar10_12_0.75"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_extend.py --dataset cifar10 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --project --epsilon 12 --train_fgsm_ratio 1 --train_random_start --seed 0 --exp_name 0306_cifar10_12_1"

#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_extend.py --dataset cifar10 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --project --epsilon 14 --train_fgsm_ratio 0.25 --train_random_start --seed 0 --exp_name 0306_cifar10_14_0.25"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_extend.py --dataset cifar10 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --project --epsilon 14 --train_fgsm_ratio 0.5 --train_random_start --seed 0 --exp_name 0306_cifar10_14_0.5"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_extend.py --dataset cifar10 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --project --epsilon 14 --train_fgsm_ratio 0.75 --train_random_start --seed 0 --exp_name 0306_cifar10_14_0.75"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_extend.py --dataset cifar10 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --project --epsilon 14 --train_fgsm_ratio 1 --train_random_start --seed 0 --exp_name 0306_cifar10_14_1"

#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_extend.py --dataset cifar10 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --project --epsilon 16 --train_fgsm_ratio 0.25 --train_random_start --seed 0 --exp_name 0306_cifar10_16_0.25"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_extend.py --dataset cifar10 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --project --epsilon 16 --train_fgsm_ratio 0.5 --train_random_start --seed 0 --exp_name 0306_cifar10_16_0.5"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_extend.py --dataset cifar10 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --project --epsilon 16 --train_fgsm_ratio 0.75 --train_random_start --seed 0 --exp_name 0306_cifar10_16_0.75"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_extend.py --dataset cifar10 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --project --epsilon 16 --train_fgsm_ratio 1 --train_random_start --seed 0 --exp_name 0306_cifar10_16_1"
# ################with the gradient of clean input

#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_test1_and_test2.py --dataset cifar10 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --ratio 0 --epsilon 8 --train_fgsm_ratio 1 --project --save_epoch --seed 0 --exp_name 0226_cifar10_diff_rs_w_proj_8_0"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_test1_and_test2.py --dataset cifar10 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --ratio 0.2 --epsilon 8 --train_fgsm_ratio 1 --project --save_epoch --seed 0 --exp_name 0226_cifar10_diff_rs_w_proj_8_0.2"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_test1_and_test2.py --dataset cifar10 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --ratio 0.4 --epsilon 8 --train_fgsm_ratio 1 --project --save_epoch --seed 0 --exp_name 0226_cifar10_diff_rs_w_proj_8_0.4"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_test1_and_test2.py --dataset cifar10 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --ratio 0.6 --epsilon 8 --train_fgsm_ratio 1 --project --save_epoch --seed 0 --exp_name 0226_cifar10_diff_rs_w_proj_8_0.6"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_test1_and_test2.py --dataset cifar10 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --ratio 0.8 --epsilon 8 --train_fgsm_ratio 1 --project --save_epoch --seed 0 --exp_name 0226_cifar10_diff_rs_w_proj_8_0.8"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_test1_and_test2.py --dataset cifar10 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --ratio 1 --epsilon 8 --train_fgsm_ratio 1 --project --save_epoch --seed 0 --exp_name 0226_cifar10_diff_rs_w_proj_8_1"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_test1_and_test2.py --dataset cifar10 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --ratio 0 --epsilon 8 --train_fgsm_ratio 1 --save_epoch --seed 0 --exp_name 0226_cifar10_diff_rs_wo_proj_8_0"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_test1_and_test2.py --dataset cifar10 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --ratio 0.2 --epsilon 8 --train_fgsm_ratio 1 --save_epoch --seed 0 --exp_name 0226_cifar10_diff_rs_wo_proj_8_0.2"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_test1_and_test2.py --dataset cifar10 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --ratio 0.4 --epsilon 8 --train_fgsm_ratio 1 --save_epoch --seed 0 --exp_name 0226_cifar10_diff_rs_wo_proj_8_0.4"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_test1_and_test2.py --dataset cifar10 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --ratio 0.6 --epsilon 8 --train_fgsm_ratio 1 --save_epoch --seed 0 --exp_name 0226_cifar10_diff_rs_wo_proj_8_0.6"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_test1_and_test2.py --dataset cifar10 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --ratio 0.8 --epsilon 8 --train_fgsm_ratio 1 --save_epoch --seed 0 --exp_name 0226_cifar10_diff_rs_wo_proj_8_0.8"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_test1_and_test2.py --dataset cifar10 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --ratio 1 --epsilon 8 --train_fgsm_ratio 1 --save_epoch --seed 0 --exp_name 0226_cifar10_diff_rs_wo_proj_8_1"

#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_test1_and_test2.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --seed 0 --clamp --ratio 0 --exp_name 0206_fgsm_diff_rs_0_{8}_1"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_test1_and_test2.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --seed 0 --clamp --ratio 0.2 --exp_name 0206_fgsm_diff_rs_0.2_{8}_1"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_test1_and_test2.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --seed 0 --clamp --ratio 0.4 --exp_name 0206_fgsm_diff_rs_0.4_{8}_1"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_test1_and_test2.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --seed 0 --clamp --ratio 0.6 --exp_name 0206_fgsm_diff_rs_0.6_{8}_1"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_test1_and_test2.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --seed 0 --clamp --ratio 0.8 --exp_name 0206_fgsm_diff_rs_0.8_{8}_1"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_test1_and_test2.py --model PreActResNet18 --num_epochs 200 --lr_schedule multistep --epsilon 8 --train_fgsm_ratio 1 --seed 0 --clamp --ratio 1 --exp_name 0206_fgsm_diff_rs_1_{8}_1"


##############################gradient align##############################
#for i in 1 2 3 4
#do
#  bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_grad_align.py --dataset cifar10 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --epsilon 8 --train_fgsm_ratio 1 --grad_align_cos_lambda 2 --seed $i --exp_name 0306_cifar10_grad_align_8_2_$i"
#  bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_grad_align.py --dataset cifar10 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --epsilon 16 --train_fgsm_ratio 1 --grad_align_cos_lambda 2 --seed $i --exp_name 0306_cifar10_grad_align_16_2_$i"
#  bsub -n 8 -W 24:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_pgd.py --dataset cifar10 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --epsilon 8 --train_pgd_ratio 0.2 --train_pgd_attack_iters 10 --train_pgd_restarts 1 --seed $i --exp_name 0306_cifar10_pgd10_8_seed_$i"
#  bsub -n 8 -W 24:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_pgd.py --dataset cifar10 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --epsilon 16 --train_pgd_ratio 0.2 --train_pgd_attack_iters 10 --train_pgd_restarts 1 --seed $i --exp_name 0306_cifar10_pgd10_16_seed_$i"
#done

#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_all.py --dataset cifar10 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --epsilon 8 --train_fgsm_ratio 1 --project --random_start_type none --seed 0 --exp_name 0220_cifar10_wors_8_1"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_all.py --dataset cifar10 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --epsilon 8 --train_fgsm_ratio 1 --project --random_start_type uniform --seed 0 --exp_name 0220_cifar10_uniform_8_1"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_all.py --dataset cifar10 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --epsilon 8 --train_fgsm_ratio 1 --random_start_type uniform --seed 0 --exp_name 0220_cifar10_uniform_wo_proj_8_1"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_all.py --dataset cifar10 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --epsilon 8 --train_fgsm_ratio 1 --project --random_start_type boundary --seed 0 --exp_name 0220_cifar10_boundary_8_1"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_all.py --dataset cifar10 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --epsilon 8 --train_fgsm_ratio 1 --random_start_type boundary --seed 0 --exp_name 0220_cifar10_boundary_wo_proj_8_1"

#for i in 0 1 2 3 4
#do
#  bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_all.py --dataset cifar10 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --epsilon 8 --train_fgsm_ratio 1 --project --random_start_type none --seed $i --exp_name 0220_cifar10_wors_8_seed_$i"
#  bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_all.py --dataset cifar10 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --epsilon 8 --train_fgsm_ratio 1 --project --random_start_type uniform --seed $i --exp_name 0220_cifar10_uniform_8_seed_$i"
#  bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_all.py --dataset cifar10 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --epsilon 8 --train_fgsm_ratio 1 --random_start_type uniform --seed $i --exp_name 0220_cifar10_uniform_wo_proj_8_seed_$i"
#  bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_all.py --dataset cifar10 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --epsilon 8 --train_fgsm_ratio 1 --random_start_type boundary --seed $i --exp_name 0220_cifar10_boundary_wo_proj_8_seed_$i"
#
#  bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_all.py --dataset cifar10 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --epsilon 16 --train_fgsm_ratio 1 --project --random_start_type none --seed $i --exp_name 0220_cifar10_wors_16_seed_$i"
#  bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_all.py --dataset cifar10 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --epsilon 16 --train_fgsm_ratio 1 --project --random_start_type uniform --seed $i --exp_name 0220_cifar10_uniform_16_seed_$i"
#  bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_all.py --dataset cifar10 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --epsilon 16 --train_fgsm_ratio 1 --random_start_type uniform --seed $i --exp_name 0220_cifar10_uniform_wo_proj_16_seed_$i"
#  bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_all.py --dataset cifar10 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --epsilon 16 --train_fgsm_ratio 1 --random_start_type boundary --seed $i --exp_name 0220_cifar10_boundary_wo_proj_16_seed_$i"
#done


########################################################### MARCH #############################################################
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_all.py --dataset cifar10 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --epsilon 8 --train_fgsm_ratio 1 --project --save_epoch --random_start_type none --seed 0 --exp_name 0325_cifar10_none_w_proj_eps_8"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_all.py --dataset cifar10 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --epsilon 8 --train_fgsm_ratio 1 --project --save_epoch --random_start_type uniform --seed 0 --exp_name 0325_cifar10_uniform_w_proj_eps_8"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_all.py --dataset cifar10 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --epsilon 8 --train_fgsm_ratio 1 --project --save_epoch --random_start_type none --seed 0 --not_track_running_stats --exp_name 0401_cifar10_nottrack_none_w_proj_eps_8"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_all.py --dataset cifar10 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --epsilon 8 --train_fgsm_ratio 1 --project --random_start_type uniform --seed 0 --not_track_running_stats --exp_name 0401_cifar10_nottrack_uniform_w_proj_eps_8"

#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_noise.py --dataset cifar10 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --epsilon 8 --train_fgsm_ratio 1 --project --random_start_type none --noise_strength 0.1 --seed 0 --exp_name 0402_cifar10_eps8_noise0.1"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_noise.py --dataset cifar10 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --epsilon 8 --train_fgsm_ratio 1 --project --random_start_type none --noise_strength 0.05 --seed 0 --exp_name 0402_cifar10_eps8_noise0.05"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_noise.py --dataset cifar10 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --epsilon 8 --train_fgsm_ratio 1 --project --random_start_type none --noise_strength 0.01 --seed 0 --exp_name 0402_cifar10_eps8_noise0.01"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_noise.py --dataset cifar10 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --epsilon 8 --train_fgsm_ratio 1 --project --random_start_type none --noise_strength 0.005 --seed 0 --exp_name 0402_cifar10_eps8_noise0.005"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_noise.py --dataset cifar10 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --epsilon 8 --train_fgsm_ratio 1 --project --random_start_type none --noise_strength 0.001 --seed 0 --exp_name 0402_cifar10_eps8_noise0.001"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_noise.py --dataset cifar10 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --epsilon 8 --train_fgsm_ratio 1 --project --random_start_type none --noise_strength 0.0008 --seed 0 --exp_name 0402_cifar10_eps8_noise0.0008"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_noise.py --dataset cifar10 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --epsilon 8 --train_fgsm_ratio 1 --project --random_start_type none --noise_strength 0.0005 --seed 0 --exp_name 0402_cifar10_eps8_noise0.0005"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_noise.py --dataset cifar10 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --epsilon 8 --train_fgsm_ratio 1 --project --random_start_type none --noise_strength 0.0001 --seed 0 --exp_name 0402_cifar10_eps8_noise0.0001"


#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_noise1.py --dataset cifar10 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --epsilon 8 --train_fgsm_ratio 1 --project --random_start_type none --noise_strength 0.1 --seed 0 --exp_name 0402_1_cifar10_eps8_noise0.1"
##bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_noise1.py --dataset cifar10 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --epsilon 8 --train_fgsm_ratio 1 --project --random_start_type none --noise_strength 0.05 --seed 0 --exp_name 0402_1_cifar10_eps8_noise0.05"
##bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_noise1.py --dataset cifar10 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --epsilon 8 --train_fgsm_ratio 1 --project --random_start_type none --noise_strength 0.005 --seed 0 --exp_name 0402_1_cifar10_eps8_noise0.005"
##bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_noise1.py --dataset cifar10 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --epsilon 8 --train_fgsm_ratio 1 --project --random_start_type none --noise_strength 0.004 --seed 0 --exp_name 0402_1_cifar10_eps8_noise0.004"
##bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_noise1.py --dataset cifar10 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --epsilon 8 --train_fgsm_ratio 1 --project --random_start_type none --noise_strength 0.003 --seed 0 --exp_name 0402_1_cifar10_eps8_noise0.003"
##bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_noise1.py --dataset cifar10 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --epsilon 8 --train_fgsm_ratio 1 --project --random_start_type none --noise_strength 0.002 --seed 0 --exp_name 0402_1_cifar10_eps8_noise0.002"
##bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_noise1.py --dataset cifar10 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --epsilon 8 --train_fgsm_ratio 1 --project --random_start_type none --noise_strength 0.001 --seed 0 --exp_name 0402_1_cifar10_eps8_noise0.001"
##bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_noise1.py --dataset cifar10 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --epsilon 8 --train_fgsm_ratio 1 --project --random_start_type none --noise_strength 0.0008 --seed 0 --exp_name 0402_1_cifar10_eps8_noise0.0008"
##bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_noise1.py --dataset cifar10 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --epsilon 8 --train_fgsm_ratio 1 --project --random_start_type none --noise_strength 0.0005 --seed 0 --exp_name 0402_1_cifar10_eps8_noise0.0005"
##bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_noise1.py --dataset cifar10 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --epsilon 8 --train_fgsm_ratio 1 --project --random_start_type none --noise_strength 0.0001 --seed 0 --exp_name 0402_1_cifar10_eps8_noise0.0001"


#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_noise1.py --dataset cifar10 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --epsilon 8 --train_fgsm_ratio 1 --project --random_start_type none --noise_strength 0.1 --seed 0 --not_track_running_stats --exp_name 0405_1_nottrack_cifar10_eps8_noise0.1"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_noise1.py --dataset cifar10 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --epsilon 8 --train_fgsm_ratio 1 --project --random_start_type none --noise_strength 0.05 --seed 0 --not_track_running_stats --exp_name 0405_1_nottrack_cifar10_eps8_noise0.05"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_noise1.py --dataset cifar10 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --epsilon 8 --train_fgsm_ratio 1 --project --random_start_type none --noise_strength 0.005 --seed 0 --not_track_running_stats --exp_name 0405_1_nottrack_cifar10_eps8_noise0.005"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_noise1.py --dataset cifar10 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --epsilon 8 --train_fgsm_ratio 1 --project --random_start_type none --noise_strength 0.004 --seed 0 --not_track_running_stats --exp_name 0405_1_nottrack_cifar10_eps8_noise0.004"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_noise1.py --dataset cifar10 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --epsilon 8 --train_fgsm_ratio 1 --project --random_start_type none --noise_strength 0.003 --seed 0 --not_track_running_stats --exp_name 0405_1_nottrack_cifar10_eps8_noise0.003"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_noise1.py --dataset cifar10 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --epsilon 8 --train_fgsm_ratio 1 --project --random_start_type none --noise_strength 0.002 --seed 0 --not_track_running_stats --exp_name 0405_1_nottrack_cifar10_eps8_noise0.002"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_noise1.py --dataset cifar10 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --epsilon 8 --train_fgsm_ratio 1 --project --random_start_type none --noise_strength 0.001 --seed 0 --not_track_running_stats --exp_name 0405_1_nottrack_cifar10_eps8_noise0.001"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_noise1.py --dataset cifar10 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --epsilon 8 --train_fgsm_ratio 1 --project --random_start_type none --noise_strength 0.0008 --seed 0 --not_track_running_stats --exp_name 0405_1_nottrack_cifar10_eps8_noise0.0008"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_noise1.py --dataset cifar10 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --epsilon 8 --train_fgsm_ratio 1 --project --random_start_type none --noise_strength 0.0005 --seed 0 --not_track_running_stats --exp_name 0405_1_nottrack_cifar10_eps8_noise0.0005"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_noise1.py --dataset cifar10 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --epsilon 8 --train_fgsm_ratio 1 --project --random_start_type none --noise_strength 0.0001 --seed 0 --not_track_running_stats --exp_name 0405_1_nottrack_cifar10_eps8_noise0.0001"

#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_noise11.py --dataset cifar10 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --epsilon 8 --train_fgsm_ratio 1 --project --random_start_type none --noise_strength 0.1 --seed 0 --not_track_running_stats --exp_name 0405_1_nottrack_decrease_cifar10_eps8_noise0.1"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_noise11.py --dataset cifar10 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --epsilon 8 --train_fgsm_ratio 1 --project --random_start_type none --noise_strength 0.05 --seed 0 --not_track_running_stats --exp_name 0405_1_nottrack_decrease_cifar10_eps8_noise0.05"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_noise11.py --dataset cifar10 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --epsilon 8 --train_fgsm_ratio 1 --project --random_start_type none --noise_strength 0.005 --seed 0 --not_track_running_stats --exp_name 0405_1_nottrack_decrease_cifar10_eps8_noise0.005"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_noise11.py --dataset cifar10 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --epsilon 8 --train_fgsm_ratio 1 --project --random_start_type none --noise_strength 0.004 --seed 0 --not_track_running_stats --exp_name 0405_1_nottrack_decrease_cifar10_eps8_noise0.004"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_noise11.py --dataset cifar10 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --epsilon 8 --train_fgsm_ratio 1 --project --random_start_type none --noise_strength 0.003 --seed 0 --not_track_running_stats --exp_name 0405_1_nottrack_decrease_cifar10_eps8_noise0.003"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_noise11.py --dataset cifar10 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --epsilon 8 --train_fgsm_ratio 1 --project --random_start_type none --noise_strength 0.002 --seed 0 --not_track_running_stats --exp_name 0405_1_nottrack_decrease_cifar10_eps8_noise0.002"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_noise11.py --dataset cifar10 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --epsilon 8 --train_fgsm_ratio 1 --project --random_start_type none --noise_strength 0.001 --seed 0 --not_track_running_stats --exp_name 0405_1_nottrack_decrease_cifar10_eps8_noise0.001"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_noise11.py --dataset cifar10 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --epsilon 8 --train_fgsm_ratio 1 --project --random_start_type none --noise_strength 0.0008 --seed 0 --not_track_running_stats --exp_name 0405_1_nottrack_decrease_cifar10_eps8_noise0.0008"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_noise11.py --dataset cifar10 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --epsilon 8 --train_fgsm_ratio 1 --project --random_start_type none --noise_strength 0.0005 --seed 0 --not_track_running_stats --exp_name 0405_1_nottrack_decrease_cifar10_eps8_noise0.0005"
#bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_noise11.py --dataset cifar10 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --epsilon 8 --train_fgsm_ratio 1 --project --random_start_type none --noise_strength 0.0001 --seed 0 --not_track_running_stats --exp_name 0405_1_nottrack_decrease_cifar10_eps8_noise0.0001"

bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_noise11.py --dataset cifar10 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --epsilon 8 --train_fgsm_ratio 1 --project --random_start_type none --noise_strength 0.1 --seed 0 --exp_name 0405_1_decrease_cifar10_eps8_noise0.1"
bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_noise11.py --dataset cifar10 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --epsilon 8 --train_fgsm_ratio 1 --project --random_start_type none --noise_strength 0.05 --seed 0 --exp_name 0405_1_decrease_cifar10_eps8_noise0.05"
bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_noise11.py --dataset cifar10 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --epsilon 8 --train_fgsm_ratio 1 --project --random_start_type none --noise_strength 0.005 --seed 0 --exp_name 0405_1_decrease_cifar10_eps8_noise0.005"
bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_noise11.py --dataset cifar10 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --epsilon 8 --train_fgsm_ratio 1 --project --random_start_type none --noise_strength 0.004 --seed 0 --exp_name 0405_1_decrease_cifar10_eps8_noise0.004"
bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_noise11.py --dataset cifar10 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --epsilon 8 --train_fgsm_ratio 1 --project --random_start_type none --noise_strength 0.003 --seed 0 --exp_name 0405_1_decrease_cifar10_eps8_noise0.003"
bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_noise11.py --dataset cifar10 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --epsilon 8 --train_fgsm_ratio 1 --project --random_start_type none --noise_strength 0.002 --seed 0 --exp_name 0405_1_decrease_cifar10_eps8_noise0.002"
bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_noise11.py --dataset cifar10 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --epsilon 8 --train_fgsm_ratio 1 --project --random_start_type none --noise_strength 0.001 --seed 0 --exp_name 0405_1_decrease_cifar10_eps8_noise0.001"
bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_noise11.py --dataset cifar10 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --epsilon 8 --train_fgsm_ratio 1 --project --random_start_type none --noise_strength 0.0008 --seed 0 --exp_name 0405_1_decrease_cifar10_eps8_noise0.0008"
bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_noise11.py --dataset cifar10 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --epsilon 8 --train_fgsm_ratio 1 --project --random_start_type none --noise_strength 0.0005 --seed 0 --exp_name 0405_1_decrease_cifar10_eps8_noise0.0005"
bsub -n 8 -W 08:00 -R "rusage[mem=4096,ngpus_excl_p=4]"  "module load $PCOMMAND; python train_fgsm_noise11.py --dataset cifar10 --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --epsilon 8 --train_fgsm_ratio 1 --project --random_start_type none --noise_strength 0.0001 --seed 0 --exp_name 0405_1_decrease_cifar10_eps8_noise0.0001"