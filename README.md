# robust_machine_learning

This is for master thesis **Understanding Catastrophic Overfitting in Adversarial Training**

## Environment

This project is run on ETHz [Leonhard cluster](https://scicomp.ethz.ch/wiki/Leonhard) but you can also run on your own enviroment:
* **start_jupyter_nb.sh** is taken from [ethz gitlab](https://gitlab.ethz.ch/sfux/Jupyter-on-Euler-or-Leonhard-Open) and used to start a jupyter notebook from a local computer on Euler/Leonhard Open 
* **start_tensorboard.sh** is used to start tensorboard from a local computer on Euler/Leonhard Open

If you do not use Leonhard cluster environment, you can ignore these two scripts.

## train_fgsm.py
train_fgsm.py implements the FGSM adversarial training.

--random_start_type is used to specify the random initialization type
* **none**: use vanilla FGSM and do not have random initialization
* **uniform**: use FGSM with unifrom initialzation. For more details about this, you can check paper [Fast is better than free: Revisiting adversarial training]( https://arxiv.org/abs/2001.03994)
* **boundary**: use FGSM with boundary initialization. For more details about this , you check section 5.1.2 in my [report](https://github.com/kangpl/robust_machine_learning/blob/master/report/Master%20Thesis_Understanding%20Catastrophic%20Overfitting%20in%20Adversarial%20Training_Peilin%20Kang.pdf)

This command can reproduce catastrophic overfitting.
```
python train_fgsm.py  --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --epsilon 8 --random_start_type none --seed 0 --exp_name reporduce_co
```

## train_deepfool.py
train_deepfool.py implements the adversarial training with 1 iteration $l_{\infty}$ [DeepFool](https://arxiv.org/abs/1511.04599)

Example
```
python train_deepfool.py  --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --epsilon 8 --seed 0 --exp_name deepfool
```

## train_pgd.py
train_pgd.py implements the PGD adversarial training.

Example
```
python train_pgd.py --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --epsilon 8 --train_pgd_ratio 0.2 --train_pgd_attack_iters 10 --train_pgd_restarts 1 --seed 0 --exp_name pgd
```

## train_fgsm_grad_align.py
train_fgsm_grad_align.py implements the FGSM with gradient alignment. For more details about gradient alignment, you can check this paper [Understanding and Improving Fast Adversarial Training](https://arxiv.org/abs/2007.02617)

Example
```
python train_fgsm_grad_align.py --model PreActResNet18 --lr 0.1 --lr_schedule multistep --num_epochs 200 --epsilon 8 --train_fgsm_ratio 1 --grad_align_cos_lambda 2 --seed 0 --exp_name grad_alignment
```
