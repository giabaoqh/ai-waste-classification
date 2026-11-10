"""Run a local bounded YOLO pilot without replacing the production classifier."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "logs/ultralytics_config"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "logs/matplotlib"))
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("COMET_MODE", "DISABLED")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", choices=["yolo26n.pt", "yolo11n.pt"], default="yolo26n.pt")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--name", required=True)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()
    from ultralytics import YOLO, settings
    import torch

    settings.update({key: False for key in ("sync", "wandb", "comet", "clearml", "mlflow", "dvc", "tensorboard") if key in settings})
    if not 1 <= args.epochs <= 30:
        raise SystemExit("Pilot is limited to 1-30 epochs")
    if args.batch < 1 or args.batch > 8:
        raise SystemExit("Pilot batch must be 1-8")
    if not args.name or Path(args.name).name != args.name or args.name in (".", ".."):
        raise SystemExit("Run name must be a single directory name")
    data_path = args.data.resolve()
    dataset = data_path.parent
    audit = json.loads((dataset / "audit.json").read_text(encoding="utf-8"))
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable; pilot will not silently switch to CPU")
    for item in manifest:
        if sha256(dataset / item["image_file"]) != item["sha256"]:
            raise SystemExit(f"Image changed since audit: {item['image_file']}")
        for line in (dataset / item["label_file"]).read_text().splitlines():
            values = line.split()
            if len(values) != 5 or not 0 <= int(values[0]) < 9 or any(not 0 <= float(v) <= 1 for v in values[1:]):
                raise SystemExit(f"Invalid YOLO label: {item['label_file']}")
    run_dir = ROOT / "runs/waste_pilot" / args.name
    if run_dir.exists():
        raise SystemExit(f"Run already exists; choose a fresh name: {run_dir}")
    model_cache = ROOT / "checkpoints/yolo_pretrained"
    model_cache.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(model_cache / args.model))
    initial_checkpoint_hash = sha256(model_cache / args.model)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model.train(data=str(data_path), epochs=args.epochs, imgsz=640, batch=args.batch,
                device=0, workers=args.workers, amp=True, seed=42, deterministic=True,
                optimizer="AdamW", lr0=.001, lrf=.01, cos_lr=True,
                patience=10, close_mosaic=min(3, args.epochs), plots=False,
                cache=False, project=str(ROOT / "runs/waste_pilot"), name=args.name,
                exist_ok=False, save=True, val=True, verbose=False)
    train_seconds = time.perf_counter() - started
    best_path = run_dir / "weights/best.pt"
    trained = YOLO(str(best_path))
    # Validation only. The reserved test images remain unopened by this script.
    validation = trained.val(data=str(data_path), split="val", imgsz=640,
                             batch=args.batch, device=0, workers=0, plots=False,
                             project=str(run_dir), name="validation", verbose=False)
    per_class = []
    for index, class_id in enumerate(validation.box.ap_class_index):
        p, r, ap50, ap = validation.box.class_result(index)
        per_class.append({"class_id": int(class_id), "class_name": CLASSES[int(class_id)],
                          "precision": float(p), "recall": float(r), "mAP50": float(ap50), "mAP50_95": float(ap)})
    # Warm up, then time full model.predict calls on validation inputs only.
    val_paths = [dataset / item["image_file"] for item in manifest if item["split"] == "val"]
    for index in range(5):
        trained.predict(str(val_paths[index % len(val_paths)]), imgsz=640, device=0, verbose=False, save=False)
    timings = []
    for index in range(30):
        torch.cuda.synchronize()
        tick = time.perf_counter()
        trained.predict(str(val_paths[index % len(val_paths)]), imgsz=640, device=0, verbose=False, save=False)
        torch.cuda.synchronize()
        timings.append((time.perf_counter() - tick) * 1000)
    timings.sort()
    summary = {"run": args.name, "model": args.model, "initial_weights_sha256": initial_checkpoint_hash,
               "best_checkpoint": str(best_path), "best_checkpoint_sha256": sha256(best_path),
               "epochs_requested": args.epochs, "seed": 42, "batch": args.batch, "imgsz": 640,
               "gpu": torch.cuda.get_device_name(0), "train_seconds": train_seconds,
               "peak_allocated_vram_mib": torch.cuda.max_memory_allocated() / 1024**2,
               "validation_metrics": {k: float(v) for k, v in validation.results_dict.items()},
               "per_class": per_class, "dataset": str(data_path), "dataset_yaml_sha256": sha256(data_path),
               "manifest_sha256": sha256(dataset / "manifest.json"), "image_counts": audit["image_counts"],
               "packages": {p: importlib.metadata.version(p) for p in ("ultralytics", "torch", "torchvision", "numpy", "pillow")},
               "python": platform.python_version(),
               "pilot_latency": {"warmup_calls": 5, "timed_calls": 30, "median_ms": timings[len(timings)//2], "p95_ms": timings[int(.95 * (len(timings)-1))], "scope": "batch1 model.predict including file decode, default FP32; diagnostic only, not acceptance benchmark"},
               "test_evaluated": False, "human_review": "pending", "production_model_replaced": False,
               "limitations": audit["limitations"]}
    (run_dir / "pilot_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    freeze = subprocess.run([os.sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True)
    (run_dir / "environment.txt").write_text(freeze.stdout, encoding="utf-8")
    print("PILOT_COMPLETE " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    from prepare_taco_pilot import CLASSES
    main()
