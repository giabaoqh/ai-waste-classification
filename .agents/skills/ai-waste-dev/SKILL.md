---
name: ai-waste-dev
description: Set up or modify this AI waste classification repository, especially PyTorch inference, datasets, calibration, Grad-CAM, multi-object detection, and FastAPI integration. Use only within giabaoqh/ai-waste-classification.
---

# AI waste development

Read root `AGENTS.md`, `docs/handover/AGENT_RULES.md` and
`docs/LOCAL_SETUP.md`. Resolve paths from the repository root. Inspect Git status
and the relevant current implementation; older handover docs describe a six-class
model and an obsolete uncommitted working tree.

For the isolated detector experiment, read `docs/YOLO_PILOT.md` and its results.
Use `experiments/yolo/.venv` for YOLO; its nine TACO object groups are separate
from the seven production classifier labels. Preserve the reserved test split
and source/derived image hashes; human label review remains a separate step.

## Setup and diagnosis

Use `backend/.venv` and the commands in `docs/LOCAL_SETUP.md`. Keep dependencies
isolated. Preserve the frontend lockfile with `npm.cmd ci`. Run
`backend\.venv\Scripts\python.exe scripts\check_setup.py`; exit 2 means trained
weights are missing, not a successful inference setup. An HTTP 200 health check
also needs both model-loaded flags to establish readiness.

## Model, data and prediction changes

- The classifier is MobileNetV3-Large; the final layer is `classifier[3]`.
  Current training labels include `medical` as index 6. Verify checkpoint
  `target_classes` and head output size instead of assuming six or seven outputs.
- Inference uses Resize(256), CenterCrop(224), ToTensor and ImageNet mean/std
  from `backend/app/main.py`. Keep training/evaluation preprocessing compatible.
- Check `backend/app/main.py` for temperature scaling, top-two margin and
  uncertainty logic before adjusting thresholds. Recorded metrics belong to
  their own checkpoint and split; do not reuse them as validation of new weights.
- Dataset preparation is in `backend/scripts/prepare_dataset.py` and
  `prepare_combined_dataset.py`; training consumes CSV manifests under
  `data/processed`. Preserve split provenance and keep test images out of training.
  Inspect arguments and output paths before running any data-writing operation.
- Multi-object detection uses pretrained SSDLite COCO proposals followed by crop
  classification, not a detector trained specifically on every waste class.
  The ImageNet non-waste gatekeeper expects an existing user-profile cache and
  otherwise falls back to heuristics. Distinguish these dependencies from the
  project's fine-tuned classifier checkpoint.
- Preserve Vietnamese recommendations, image quality and uncertainty fields, and
  the separation between visual classification and disposal guidance. Test
  hazardous-item overrides when changing `knowledge_base.py` or its JSON.

## Validation

Run the checks applicable to the changed layer from `docs/LOCAL_SETUP.md`.
For model-related work, use genuine inputs and trained weights; disclose missing
checkpoints or datasets rather than fabricating predictions. Do not retrain or
replace checkpoints unless the user requested it. Include observed test results,
class order and remaining limitations when handing work back.
