class Metrics(object):
    def __init__(self):
        self.metrics = {}

    def init_metrics(self):
        self.metrics = {"train_clean_loss": 0,
                        "train_clean_correct": 0,
                        "train_clean_grad_norm": [],
                        "train_fgsm_loss": 0,
                        "train_fgsm_correct": 0,
                        "train_fgsm_grad_norm": [],
                        "train_pgd_loss": 0,
                        "train_pgd_correct": 0,
                        "train_pgd_grad_norm": [],
                        "train_df_loop": [],
                        "train_df_perturbation_norm": [],
                        "train_df_grad_norm": [],
                        "train_cos_clean_df": [],
                        "train_cos_clean_fgsm": [],
                        "train_cos_clean_pgd": [],
                        "train_cos_fgsm_pgd": [],
                        "train_total": 0,
                        "test_clean_loss": 0,
                        "test_clean_correct": 0,
                        "test_clean_grad_norm": [],
                        "test_fgsm_loss": 0,
                        "test_fgsm_correct": 0,
                        "test_fgsm_grad_norm": [],
                        "test_pgd_loss": 0,
                        "test_pgd_correct": 0,
                        "test_pgd_grad_norm": [],
                        "test_df_loop": [],
                        "test_df_perturbation_norm": [],
                        "test_df_grad_norm": [],
                        "test_cos_clean_df": [],
                        "test_cos_clean_fgsm": [],
                        "test_cos_clean_pgd": [],
                        "test_cos_fgsm_pgd": [],
                        "test_total": 0}

    def clear_metrics(self):
        self.init_metrics()
