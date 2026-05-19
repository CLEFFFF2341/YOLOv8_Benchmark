import sys
from pathlib import Path

import torch

repo_dir = Path(__file__).resolve().parent
src_dir = repo_dir / "ultralytics_src"
ultralytics_dir = src_dir if src_dir.exists() else repo_dir
sys.path.insert(0, str(ultralytics_dir))

from ultralytics import YOLO
from ultralytics.nn.modules import SG_LCA


def main():
    module = SG_LCA(c1=64)
    x = torch.randn(2, 64, 80, 80)
    y = module(x)
    assert y.shape == x.shape, f"SG_LCA shape mismatch: input={x.shape}, output={y.shape}"

    model_yaml = ultralytics_dir / "ultralytics" / "cfg" / "models" / "v8" / "yolov8n_sglca.yaml"
    model = YOLO(str(model_yaml))
    print(f"SG_LCA module OK: {tuple(y.shape)}")
    print(model.model)


if __name__ == "__main__":
    main()
