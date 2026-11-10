"""Read-only local setup diagnosis. Exit 0: ready, 1: error, 2: missing model."""

import importlib.metadata
import json
from pathlib import Path
import shutil
import sys


ROOT = Path(__file__).resolve().parents[1]


def main():
    report = {"root": str(ROOT), "python": sys.executable}
    errors = []
    versions = {}
    for package in ("fastapi", "uvicorn", "python-multipart", "torch",
                    "torchvision", "pillow", "numpy", "pydantic", "pytest", "httpx"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            errors.append(f"Missing Python package: {package}")
    report["packages"] = versions
    report["node"] = shutil.which("node")
    report["npm"] = shutil.which("npm.cmd") or shutil.which("npm")
    report["frontend_installed"] = (ROOT / "frontend/node_modules/.package-lock.json").is_file()
    if not report["node"] or not report["npm"] or not report["frontend_installed"]:
        errors.append("Node/npm or frontend dependencies are missing; run npm.cmd ci in frontend.")

    checkpoint = ROOT / "checkpoints/mobilenetv3_best.pth"
    report["checkpoint"] = str(checkpoint)
    report["checkpoint_exists"] = checkpoint.is_file()
    report["model_loaded"] = False
    try:
        import torch
        import torchvision

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        report["device"] = str(device)
        report["cuda_runtime"] = torch.version.cuda
        if device.type == "cuda":
            report["gpu"] = torch.cuda.get_device_name(0)
        probe = torch.tensor([1.0, 2.0], device=device) * 2
        if probe.cpu().tolist() != [2.0, 4.0]:
            raise RuntimeError("Tensor computation failed")
        torchvision.ops.nms(torch.tensor([[0., 0., 1., 1.]], device=device),
                            torch.tensor([1.], device=device), 0.5)
        report["tensor_and_torchvision_ops"] = "ok"
        if checkpoint.is_file():
            sys.path.insert(0, str(ROOT))
            from backend.app.main import load_model_from_checkpoint

            model, classes = load_model_from_checkpoint(checkpoint, device)
            with torch.inference_mode():
                output = model(torch.zeros(1, 3, 224, 224, device=device))
            if output.shape != (1, len(classes)):
                raise RuntimeError("Classifier output does not match checkpoint classes")
            report["model_loaded"] = True
            report["target_classes"] = classes
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")

    report["errors"] = errors
    report["status"] = "error" if errors else ("ready" if report["model_loaded"] else "missing_checkpoint")
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return 1 if errors else (0 if report["model_loaded"] else 2)


if __name__ == "__main__":
    raise SystemExit(main())
