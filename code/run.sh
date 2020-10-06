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
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python main.py --model PreActResNet18 --lr 0.001 --lr_schedule multistep --lr_change_epoch 100 150 --num_epochs 200 --attack_during_train fgsm --train_fgsm_alpha 16 --finetune --resumed_model_name cifar10_standard_preActResNet18_add_norm_exp3_final.pth --exp_name cifar10_finetune_fgsm_16_lr_0.001_multi_epoch_200_preActResNet18_exp1
#ENDBSUB


#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python main.py --model PreActResNet18 --attack_during_train none --exp_name cifar10_standard_preActResNet18_add_norm_std_exp4
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python main.py --model PreActResNet18 --attack_during_train fgsm --train_fgsm_alpha 7 --exp_name cifar10_fgsm_7_preActResNet18_add_norm_std_exp4
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python main.py --model PreActResNet18 --attack_during_train fgsm --train_fgsm_alpha 10 --exp_name cifar10_fgsm_10_preActResNet18_add_norm_std_exp4
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python main.py --model PreActResNet18 --attack_during_train fgsm --train_fgsm_alpha 16 --exp_name cifar10_fgsm_16_preActResNet18_add_norm_std_exp4
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 99:99 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python main.py --model PreActResNet18 --attack_during_train fgsm --train_fgsm_alpha 10 --test_pgd_attack_iters 50 --test_pgd_restarts 10 --exp_name cifar10_fgsm_10_pgd_50_10_preActResNet18_exp1
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 99:99 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python main.py --model PreActResNet18 --attack_during_train fgsm --train_fgsm_alpha 16 --test_pgd_attack_iters 50 --test_pgd_restarts 10 --exp_name cifar10_fgsm_16_pgd_50_10_preActResNet18_exp1
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 24:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python main.py --model PreActResNet18 --attack_during_train fgsm --train_fgsm_alpha 7 --attack_during_test deepfool --exp_name test
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 24:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python main.py --model PreActResNet18 --attack_during_train pgd --attack_during_test deepfool --exp_name test_pgd
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 24:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python main.py --model PreActResNet18 --attack_during_train pgd --train_pgd_attack_iters 20 --train_pgd_restarts 2 --attack_during_test deepfool --exp_name test_pgd_2
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 24:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python main.py --model PreActResNet18 --attack_during_train fgsm --train_fgsm_alpha 7 --attack_during_test deepfool --exp_name cifar10_fgsm_7_preActResNet18_deepfool_exp1
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 24:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python main.py --model PreActResNet18 --attack_during_train fgsm --train_fgsm_alpha 10 --attack_during_test deepfool --exp_name cifar10_fgsm_10_preActResNet18_deepfool_exp1
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 24:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python main.py --model PreActResNet18 --attack_during_train fgsm --train_fgsm_alpha 16 --attack_during_test deepfool --exp_name cifar10_fgsm_16_preActResNet18_deepfool_exp1
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 24:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python main.py --model PreActResNet18 --attack_during_train fgsm --train_fgsm_alpha 10 --attack_during_test deepfool --deepfool_classes_num 4 --exp_name cifar10_fgsm_10_preActResNet18_deepfool_4_exp1
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
##bsub -n 8 -W 24:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
##module load $PCOMMAND
##python main.py --model PreActResNet18 --attack_during_train fgsm --train_fgsm_alpha 7 --mix --exp_name cifar10_fgsm_7_preActResNet18_mix_exp1
##ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 24:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python main.py --model PreActResNet18 --attack_during_train fgsm --train_fgsm_alpha 10 --mix --exp_name cifar10_fgsm_10_preActResNet18_mix_exp1
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 24:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python main.py --model PreActResNet18 --attack_during_train fgsm --train_fgsm_alpha 16 --mix --exp_name cifar10_fgsm_16_preActResNet18_mix_exp1
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 10:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python main.py --model PreActResNet18 --attack_during_train fgsm --train_fgsm_alpha 10 --adjust_fgsm_alpha --exp_name cifar10_fgsm_10_preActResNet18_adjust_exp1
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 10:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python main.py --model PreActResNet18 --attack_during_train fgsm --train_fgsm_alpha 16 --adjust_fgsm_alpha --exp_name cifar10_fgsm_16_preActResNet18_adjust_exp2
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python main.py --model PreActResNet18 --attack_during_train none --exp_name cifar10_standard_preActResNet18_add_norm_std_median_exp1
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python main.py --model PreActResNet18 --attack_during_train fgsm --train_fgsm_alpha 7 --exp_name cifar10_fgsm_7_preActResNet18_add_norm_std_median_exp1
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python main.py --model PreActResNet18 --attack_during_train fgsm --train_fgsm_alpha 10 --exp_name cifar10_fgsm_10_preActResNet18_add_norm_std_median_exp1
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
##bsub -n 8 -W 08:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
##module load $PCOMMAND
##python main.py --model PreActResNet18 --cure --exp_name cifar10_cure_exp1
##ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python main.py --model PreActResNet18 --cure --exp_name cifar10_cure_eval_exp1
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python main.py --model PreActResNet18 --cure --cure_lambda 40 --exp_name cifar10_cure_40_eval_exp1
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python main.py --model PreActResNet18 --cure --cure_lambda 400 --exp_name cifar10_cure_400_eval_exp1
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python main.py --model PreActResNet18 --attack_during_train fgsm --train_fgsm_alpha 10 --exp_name cifar10_fgsm_10_preActResNet18_add_norm_std_median_exp2
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python main.py --model PreActResNet18 --attack_during_train fgsm --train_fgsm_alpha 10 --exp_name cifar10_fgsm_10_preActResNet18_add_norm_std_median_exp3
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python main.py --model PreActResNet18 --cure --exp_name cifar10_cure_h1.5_eval_exp1
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python main.py --model PreActResNet18 --cure --cure_lambda 40 --exp_name cifar10_cure_40_h1.5_eval_exp1
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python main.py --model PreActResNet18 --cure --cure_lambda 400 --exp_name cifar10_cure_400_h1.5_eval_exp1
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python main.py --model PreActResNet18 --lr 0.001 --lr_schedule constant --num_epochs 20 --cure --finetune --resumed_model_name cifar10_standard_preActResNet18_add_norm_exp3_final.pth --exp_name cifar10_finetune_cure_exp1
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python main.py --model PreActResNet18 --lr 0.001 --lr_schedule constant --num_epochs 20 --cure --cure_lambda 40 --finetune --resumed_model_name cifar10_standard_preActResNet18_add_norm_exp3_final.pth --exp_name cifar10_finetune_cure_40_exp1
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python main.py --model PreActResNet18 --lr 0.001 --lr_schedule constant --num_epochs 20 --cure --cure_lambda 400 --finetune --resumed_model_name cifar10_standard_preActResNet18_add_norm_exp3_final.pth --exp_name cifar10_finetune_cure_400_exp1
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python main.py --model PreActResNet18 --lr 0.01 --lr_schedule constant --num_epochs 20 --cure --finetune --resumed_model_name cifar10_standard_preActResNet18_add_norm_exp3_final.pth --exp_name cifar10_finetune_cure_lr0.01_exp1
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python main.py --model PreActResNet18 --lr 0.01 --lr_schedule constant --num_epochs 20 --cure --cure_lambda 40 --finetune --resumed_model_name cifar10_standard_preActResNet18_add_norm_exp3_final.pth --exp_name cifar10_finetune_cure_40_lr0.01_exp1
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python main.py --model PreActResNet18 --lr 0.01 --lr_schedule constant --num_epochs 20 --cure --cure_lambda 400 --finetune --resumed_model_name cifar10_standard_preActResNet18_add_norm_exp3_final.pth --exp_name cifar10_finetune_cure_400_lr0.01_exp1
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python main.py --model PreActResNet18 --lr 0.001 --lr_schedule constant --num_epochs 20 --cure --cure_h 3 --finetune --resumed_model_name cifar10_standard_preActResNet18_add_norm_exp3_final.pth --exp_name cifar10_finetune_cure_h3_exp1
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python main.py --model PreActResNet18 --lr 0.001 --lr_schedule constant --num_epochs 20 --cure --cure_h 3 --cure_lambda 40 --finetune --resumed_model_name cifar10_standard_preActResNet18_add_norm_exp3_final.pth --exp_name cifar10_finetune_cure_h3_40_exp1
#ENDBSUB
#
#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 04:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python main.py --model PreActResNet18 --lr 0.001 --lr_schedule constant --num_epochs 20 --cure --cure_h 3 --cure_lambda 400 --finetune --resumed_model_name cifar10_standard_preActResNet18_add_norm_exp3_final.pth --exp_name cifar10_finetune_cure_h3_400_exp1
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python main.py --model PreActResNet18 --lr 0.01 --lr_schedule constant --num_epochs 60 --cure --cure_h 1.5 --cure_lambda 4 --finetune --resumed_model_name cifar10_standard_preActResNet18_add_norm_exp3_final.pth --exp_name cifar10_finetune_cure_h1.5_sum_withpow_exp1
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python main.py --model PreActResNet18 --lr 0.01 --lr_schedule constant --num_epochs 60 --cure --cure_h 1.5 --cure_lambda 4 --finetune --resumed_model_name cifar10_standard_preActResNet18_add_norm_exp3_final.pth --exp_name cifar10_finetune_cure_h1.5_sum_withpow_normv2_exp1
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python main.py --model PreActResNet18 --lr 0.01 --lr_schedule constant --num_epochs 60 --cure --cure_h 1.5 --cure_lambda 4 --finetune --resumed_model_name cifar10_standard_preActResNet18_add_norm_exp3_final.pth --exp_name cifar10_finetune_cure_h1.5_sum_withpow_normv2_warmup_exp1
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python main.py --model PreActResNet18 --lr 0.01 --lr_schedule constant --num_epochs 60 --cure --cure_h 1.5 --cure_lambda 4 --finetune --resumed_model_name cifar10_standard_preActResNet18_add_norm_exp3_final.pth --exp_name cifar10_finetune_cure_h1.5_sum_normv2_warmup_exp1
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python main.py --model PreActResNet18 --lr 0.01 --lr_schedule constant --num_epochs 60 --cure --cure_h 1.5 --cure_lambda 4 --finetune --resumed_model_name cifar10_standard_preActResNet18_add_norm_exp3_final.pth --exp_name cifar10_finetune_cure_h1.5_sum_normv2_warmup_exp2
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python main.py --model PreActResNet18 --lr 0.01 --lr_schedule constant --num_epochs 60 --cure --cure_h 1.5 --cure_lambda 4 --finetune --resumed_model_name cifar10_standard_preActResNet18_add_norm_exp3_final.pth --exp_name cifar10_finetune_cure_h1.5_sum_withpow_normv2_warmup_exp2
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python main.py --model PreActResNet18 --lr 0.1 --lr_schedule constant --num_epochs 60 --cure --cure_h 1.5 --cure_lambda 4 --finetune --resumed_model_name cifar10_standard_preActResNet18_add_norm_exp3_final.pth --exp_name cifar10_finetune_cure_lr0.1_h1.5_sum_normv2_warmup_exp2
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python main.py --model PreActResNet18 --lr 0.1 --lr_schedule constant --num_epochs 60 --cure --cure_h 1.5 --cure_lambda 4 --finetune --resumed_model_name cifar10_standard_preActResNet18_add_norm_exp3_final.pth --exp_name cifar10_finetune_cure_lr0.1_h1.5_sum_withpow_normv2_warmup_exp2
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python main.py --model PreActResNet18 --lr 0.001 --lr_schedule constant --num_epochs 60 --cure --cure_h 1.5 --cure_lambda 4 --finetune --resumed_model_name cifar10_standard_preActResNet18_add_norm_exp3_final.pth --exp_name cifar10_finetune_cure_lr0.001_h1.5_sum_withpow_normv2_warmup_exp2
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python main.py --model PreActResNet18 --lr 0.0001 --lr_schedule constant --num_epochs 60 --cure --cure_h 1.5 --cure_lambda 4 --finetune --resumed_model_name cifar10_standard_preActResNet18_add_norm_exp3_final.pth --exp_name cifar10_finetune_cure_lr0.0001_h1.5_sum_withpow_normv2_warmup_exp1
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python main.py --model PreActResNet18 --attack_during_train fgsm --train_fgsm_alpha 10 --exp_name cifar10_fgsm_10_preActResNet18_add_norm_std_median_hist_exp1
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
##bsub -n 8 -W 08:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
##module load $PCOMMAND
##python main.py --model PreActResNet18 --attack_during_train fgsm --train_fgsm_alpha 16 --exp_name cifar10_fgsm_16_preActResNet18_add_norm_std_median_hist_exp1
##ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python main.py --model PreActResNet18 --attack_during_train fgsm --train_fgsm_alpha 16 --exp_name cifar10_fgsm_16_preActResNet18_add_norm_std_median_hist_exp2
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
##bsub -n 8 -W 08:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
##module load $PCOMMAND
##python main.py --model PreActResNet18 --attack_during_train fgsm --train_fgsm_alpha 16 --exp_name cifar10_fgsm_16_preActResNet18_add_norm_std_median_hist_exp3
##ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python main.py --model PreActResNet18 --attack_during_train fgsm --train_fgsm_alpha 16 --num_epochs 20 --exp_name test
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python main.py --model PreActResNet18 --attack_during_train none --num_epochs 20 --exp_name test_none
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 08:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python main.py --model PreActResNet18 --attack_during_train fgsm --train_fgsm_alpha 10 --attack_during_test deepfool --exp_name deepfool
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 24:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python main.py --model PreActResNet18 --attack_during_train fgsm --train_fgsm_alpha 10 --exp_name fgsm_pgd_deepfool
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 24:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python main.py --model PreActResNet18 --attack_during_train fgsm --train_fgsm_alpha 10 --exp_name fgsm_pgd_deepfool_exp2
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 24:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python main.py --model PreActResNet18 --attack_during_train fgsm --train_fgsm_alpha 10 --exp_name fgsm_pgd_deepfool_exp3
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
#python main.py --model PreActResNet18 --attack_during_train fgsm --train_fgsm_alpha 10 --exp_name fgsm_10_df_grad_cos
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 24:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python main.py --model PreActResNet18 --attack_during_train fgsm --train_fgsm_alpha 10 --exp_name fgsm_10_df_grad_cos_exp2
#ENDBSUB

#PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
#bsub -n 8 -W 24:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
#module load $PCOMMAND
#python main.py --model PreActResNet18 --attack_during_train fgsm --train_fgsm_alpha 16 --exp_name fgsm_16_df_grad_cos_exp1
#ENDBSUB

PCOMMAND="gcc/6.3.0 python_gpu/3.8.5 eth_proxy"
bsub -n 8 -W 24:00 -R "rusage[mem=2048,ngpus_excl_p=2]"  <<ENDBSUB
module load $PCOMMAND
python main.py --model PreActResNet18 --attack_during_train fgsm --train_fgsm_alpha 16 --exp_name fgsm_16_df_grad_cos_exp2
ENDBSUB
