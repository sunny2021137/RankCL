import sys
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from scripts.run_rcm import run_image, run_tabular
from scripts.run_baseline_deep import run_tabular_baseline_deep, run_image_baseline_deep
from scripts.run_baseline import run_image_baseline, run_tabular_baseline
from scripts.run_ptob_ftrcmob import run_ptob_ftrcmob_image, run_ptob_ftrcmob_tabular
from scripts.run_ptrcm_ftob import run_ptrcm_ftob_image, run_ptrcm_ftob_tabular
from scripts.run_ptrcm_ftrcmob import run_ptrcm_ftrcmob_image, run_ptrcm_ftrcmob_tabular
import argparse
from src.utils import load_yaml


def run(base_cfg):
    
    if base_cfg["method_type"] == "RankCL":
        if base_cfg["dataset"]["dataset_type"] == "tabular":
            run_tabular(base_cfg)
        elif base_cfg["dataset"]["dataset_type"] == "image":
            run_image(base_cfg)
        else:
            raise ValueError("Unsupported dataset type")
    
    elif base_cfg["method_type"] == "BaselineDeep":
        if base_cfg["dataset"]["dataset_type"] == "tabular":
            run_tabular_baseline_deep(base_cfg)
        elif base_cfg["dataset"]["dataset_type"] == "image":
            run_image_baseline_deep(base_cfg)
        else:
            raise ValueError("Unsupported dataset type")
        
    elif base_cfg["method_type"] == "Baseline":
        if base_cfg["dataset"]["dataset_type"] == "tabular":
            run_tabular_baseline(base_cfg)
        elif base_cfg["dataset"]["dataset_type"] == "image":
            run_image_baseline(base_cfg)
        else:
            raise ValueError("Unsupported dataset type")
    
    elif base_cfg["method_type"] == "PT-OB + FT-RCM-OB":
        if base_cfg["dataset"]["dataset_type"] == "tabular":
            run_ptob_ftrcmob_tabular(base_cfg)
        elif base_cfg["dataset"]["dataset_type"] == "image":
            run_ptob_ftrcmob_image(base_cfg)
        else:
            raise ValueError("Unsupported dataset type")
    
    elif base_cfg["method_type"] == "PT-RCM + FT-OB":
        if base_cfg["dataset"]["dataset_type"] == "tabular":
            run_ptrcm_ftob_tabular(base_cfg)
        elif base_cfg["dataset"]["dataset_type"] == "image":
            run_ptrcm_ftob_image(base_cfg)
        else:
            raise ValueError("Unsupported dataset type")
        
    elif base_cfg["method_type"] == "PT-RCM + FT-RCM-OB":
        if base_cfg["dataset"]["dataset_type"] == "tabular":
            run_ptrcm_ftrcmob_tabular(base_cfg)
        elif base_cfg["dataset"]["dataset_type"] == "image":
            run_ptrcm_ftrcmob_image(base_cfg)
        else:
            raise ValueError("Unsupported dataset type")
    else:
        raise ValueError("Unsupported method type")

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RankCL Framework")
    parser.add_argument("--config", type=str, default="configs/default/default.yaml")
    args = parser.parse_args()

    base_cfg = load_yaml(args.config)
    run(base_cfg)