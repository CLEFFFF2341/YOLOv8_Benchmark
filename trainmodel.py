import sys
from pathlib import Path
from multiprocessing import freeze_support

repo_dir = Path(__file__).resolve().parent
src_dir = repo_dir / "ultralytics_src"
ultralytics_dir = src_dir if src_dir.exists() else repo_dir
sys.path.insert(0, str(ultralytics_dir))

from ultralytics import YOLO


def main():
    root_dir = Path(r"D:\PCB\Data_YOLO")
    yaml_path = root_dir / "data.yaml"
    model_yaml = ultralytics_dir / "ultralytics" / "cfg" / "models" / "v8" / "yolov8-ema.yaml"

    model = YOLO(str(model_yaml)).load("yolov8n.pt")

    model.train(
        data=str(yaml_path),
        epochs=80,
        imgsz=640,
        batch=32,
        device=0,
        workers=4,
        project=r"D:\PCB\runs",
        name="yolov8n_ema_backbone",
        pretrained=False,
        optimizer="auto",
        lr0=0.005,
        close_mosaic=25,
        patience=20,
        seed=42,
        deterministic=True,
        verbose=True
    )


if __name__ == "__main__":
    freeze_support()
    main()
