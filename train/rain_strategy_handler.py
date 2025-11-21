# src/training/train_strategy_handler.py

import torch

from .pretrain_ob import pretrain_ob
from .pretrain_rcm import pretrain_rcm
from .finetune import finetune_model

# # configs/train_strategy/pt_rcm_ft_rcm_ob.yaml
# strategy: pt_rcm_ft_rcm_ob
# pretrain:
#   module: RCM
# finetune:
#   module: [RCM, OB]


def run_train_strategy(model, train_loader, val_loader, cfg, device):

    strategy = cfg["strategy"]

    # ================
    # 1. END-TO-END FROM SCRATCH
    # ================
    if strategy == "scratch":
        print("[Strategy] OB + RCM (Scratch)")
        return finetune_model(model, train_loader, val_loader, device)

    # ================
    # 2. PRETRAIN OB, THEN FT RCM + OB
    # ================
    if strategy == "pt_ob_ft_rcm_ob":
        print("[Strategy] PT-OB → FT-RCM+OB")

        model = pretrain_ob(model, train_loader, device)
        return finetune_model(model, train_loader, val_loader, device,
                              finetune_parts=["RCM", "OB"])

    # ================
    # 3. PRETRAIN RCM, THEN FT RCM + OB
    # ================
    if strategy == "pt_rcm_ft_rcm_ob":
        print("[Strategy] PT-RCM → FT-RCM+OB")

        model = pretrain_rcm(model, train_loader, device)
        return finetune_model(model, train_loader, val_loader, device,
                              finetune_parts=["RCM", "OB"])

    # ================
    # 4. PRETRAIN RCM, THEN FT OB ONLY
    # ================
    if strategy == "pt_rcm_ft_ob":
        print("[Strategy] PT-RCM → FT-OB")

        model = pretrain_rcm(model, train_loader, device)
        return finetune_model(model, train_loader, val_loader, device,
                              finetune_parts=["OB"])

    raise ValueError(f"Unknown strategy: {strategy}")
