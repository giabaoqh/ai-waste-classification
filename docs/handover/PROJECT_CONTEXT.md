# PROJECT CONTEXT

## 1. Project Identity

| Field | Value |
|-------|-------|
| Name | AI-Based Waste Recognition and Recycling Recommendation System |
| Type | University capstone project |
| Repository | `https://github.com/giabaoqh/ai-waste-classification.git` |
| Root Path | `D:\AI-Based Waste Recognition and Recycling Recommendation System` |
| Language | Vietnamese UI/docs, English code/APIs |

## 2. Team

| Member | Role | Branch |
|--------|------|--------|
| Huy | Team Leader — PR reviews, integration, `main` management | `main` |
| Bảo | AI Model & Dataset — training, evaluation, data preparation | `feature/ai-model` |
| Minh | Backend, Frontend & Integration — FastAPI, React, E2E | `feature/backend-api`, `feature/frontend` |

**Note**: The 3 members share a single Windows machine (see ADR-005 in `docs/DECISIONS.md`).

## 3. Tech Stack (Actual)

### Backend
- **Python 3.13.16** via `backend\.venv`
- **FastAPI** `>=0.110.0` + **Uvicorn** `>=0.28.0`
- **PyTorch** `>=2.0.0` + **Torchvision** `>=0.15.0` (CUDA-accelerated on RTX 3050)
- **Pillow** `>=10.0.0`, **Pydantic** `>=2.0.0`, **python-multipart** `>=0.0.9`
- **NumPy** (transitive dependency)
- **SQLite3** (stdlib, WAL mode)

### Frontend
- **React 19.3.1** + **React DOM 19.3.1**
- **Vite 8.3.4** (dev server + bundler)
- **Vanilla CSS** (no Tailwind, no component library)
- **OxLint** for linting

### AI Model
- **Architecture**: MobileNetV3-Large (torchvision pretrained backbone)
- **Fine-tuning**: Transfer learning, replaced final classifier `nn.Linear(1280, 6)`
- **Checkpoint**: `checkpoints/mobilenetv3_best.pth` (~16 MB)
- **Backup**: `checkpoints/mobilenetv3_best_backup.pth`
- **Training data**: Combined RealWaste + Kaggle Garbage Classification datasets
  - Split: 3,327 train / 712 val / 713 test images
  - Mapped to 6 target classes
- **Preprocessing**: `Resize(256) → CenterCrop(224) → ToTensor() → Normalize(ImageNet)`

### Hardware
- **GPU**: NVIDIA GeForce RTX 3050 Laptop GPU (4 GB VRAM)
- **OS**: Windows 11 (64-bit)
- **Driver**: NVIDIA `616.64`

## 4. Architecture Overview

```
User → React UI (Port 5173)
         ↓ POST /api/predict (multipart/form-data)
       FastAPI Backend (Port 8000)
         ├── Image validation (format, size, quality)
         ├── MobileNetV3 inference (CUDA/CPU)
         ├── Softmax → Top-3 probabilities
         ├── Uncertainty check (threshold: 0.65)
         ├── Vietnamese recycling recommendations (in-memory dict)
         ├── Save to SQLite (metadata only, NO image binary)
         └── Return PredictionResponse JSON
```

### API Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/api/health` | Service & model health check |
| `POST` | `/api/predict` | Upload image → classify → recommend |
| `GET` | `/api/history` | Paginated prediction history (filters: class, date range) |
| `GET` | `/api/history/stats` | Quick aggregation stats |
| `GET` | `/api/history/{id}` | Single history record by ID |
| `DELETE` | `/api/history/{id}` | Delete single record |
| `DELETE` | `/api/history` | Clear all history |
| `GET` | `/api/analytics/dashboard` | Full analytics dashboard data (filters: date range) |

## 5. Model Performance (Test Set)

| Metric | Value |
|--------|-------|
| Accuracy | **95.98%** |
| Macro Precision | 95.41% |
| Macro Recall | 95.85% |
| Macro F1 | 95.61% |
| Weighted F1 | 95.99% |
| Best Epoch | 10 |
| Best Val Accuracy | 96.36% |
| Training Time | ~1,421 seconds (~24 minutes) |

### Per-Class F1 Scores
| Class | F1 |
|-------|----|
| plastic | 0.9219 |
| paper | 0.9747 |
| metal | 0.9253 |
| glass | 0.9645 |
| organic | 0.9767 |
| other | 0.9736 |

## 6. Key Constraints

1. **Local-only deployment** — no cloud, no Docker, no auth/multi-user
2. **Model checkpoint is sacred** — never delete, overwrite, or replace without explicit request
3. **No image storage** — database stores metadata only (privacy by design)
4. **Vietnamese guidance is hardcoded** in `RECOMMENDATIONS` dict in `main.py`, not a separate DB
5. **Git changes require user approval** — do not commit, push, merge, or switch branches
