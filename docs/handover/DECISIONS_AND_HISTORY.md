# DECISIONS AND HISTORY — Architecture Decision Records

This document combines existing ADRs from `docs/DECISIONS.md` with decisions made during Phases 1–4 implementation.

---

## Original ADRs (from project setup)

### ADR-001: Feature-Based Git Branching
- **Decision**: Use `feature/frontend`, `feature/ai-model`, `feature/backend-api` branches instead of person-named branches.
- **Rationale**: Avoids confusion when team members help across features. Follows industry-standard Git Flow.

### ADR-002: Python 3.13 over 3.14
- **Decision**: Use Python 3.13.16 for backend and AI.
- **Rationale**: PyTorch has stable CUDA pre-built wheels for 3.13. Python 3.14 lacks stable binaries for many scientific libraries.

### ADR-003: Separate AI Model from Recycling Knowledge Base
- **Decision**: Model only predicts class + confidence. Recycling rules are managed in a separate knowledge base.
- **Rationale**: Recycling rules can change by locality without retraining the neural network.

### ADR-004: Dataset & Weights Excluded from Git
- **Decision**: `.gitignore` excludes `*.pth`, `data/`, `dataset/`.
- **Rationale**: GitHub has a 100MB file limit. Model weights are ~16MB each. Datasets are much larger.

### ADR-005: Shared Machine Identity Management
- **Decision**: Each team member sets `git config user.name/email` locally when they take a turn.
- **Rationale**: 3 members share one Windows laptop.

---

## Phase 1–4 Decisions (made during implementation)

### ADR-006: MobileNetV3-Large for Classification
- **Decision**: Use MobileNetV3-Large with transfer learning from ImageNet.
- **Rationale**: Best accuracy/VRAM tradeoff for 4GB RTX 3050. ~16MB checkpoint, fast inference (~50ms/image).
- **Alternatives considered**: EfficientNet (higher VRAM), ResNet50 (larger model), Vision Transformer (too heavy).

### ADR-007: Uncertainty Threshold at 0.65
- **Decision**: Flag predictions as "uncertain" when softmax confidence < 65%.
- **Rationale**: Validated on 1,978 validation images. Below 0.65, accuracy drops to ~55-63%. Configurable via env var.

### ADR-008: Image Quality Rejection vs Warning Thresholds
- **Decision**: Two tiers:
  - **Hard reject** (HTTP 400): <64px dimension, brightness <6.0 or >252.0 (unusable images)
  - **Soft warning** (proceed with inference): <160px, brightness <40 or >230, blur (Laplacian variance <45)
- **Rationale**: Prevents false rejections of valid but unusual images. Thresholds derived from validation set distribution analysis.

### ADR-009: SQLite over PostgreSQL/Supabase
- **Decision**: Use SQLite (stdlib) instead of the originally planned PostgreSQL/Supabase.
- **Rationale**: Local-only capstone project. No multi-user, no cloud deployment. SQLite requires zero setup, runs embedded, and handles the expected scale (hundreds of predictions, not millions).

### ADR-010: No Image Binary Storage
- **Decision**: Database stores only prediction metadata (filename, dimensions, format, size). Never stores the image file itself.
- **Rationale**: Privacy by design. Uploaded images are processed in-memory and discarded. Users don't expect their waste photos to be stored permanently.

### ADR-011: Vietnamese Recycling Guidance In-Memory
- **Decision**: Vietnamese recycling recommendations are defined as a Python dict (`RECOMMENDATIONS`) in `main.py`, not in a database table or external JSON file.
- **Rationale**: Only 6 classes, each with static guidance text. A database lookup adds complexity without benefit. If more classes are added or localization is needed, this should be refactored.

### ADR-012: Single-File Frontend (App.jsx)
- **Decision**: All React UI in one file (~67KB).
- **Rationale**: Rapid prototyping during capstone. Acknowledged as technical debt (see KNOWN_ISSUES.md #5).
- **Recommendation**: Refactor into components if the project continues.

### ADR-013: Analytics Dashboard Ethical Disclaimers
- **Decision**: Dashboard explicitly states that prediction counts ≠ actual waste recycled, and makes no carbon/environmental impact claims.
- **Rationale**: The system counts how many images were classified, not how much waste was actually recycled. Making environmental claims would be misleading.

### ADR-014: Combined Dataset Training
- **Decision**: Merge RealWaste and Kaggle Garbage Classification datasets, mapping their original categories to the 6 target classes.
- **Rationale**: RealWaste alone had imbalanced class distribution and limited variety. Adding Kaggle data improved generalization.

---

## Chronological History

| Date | Event |
|------|-------|
| 2026-10-19 | Project repository initialized, initial documentation created |
| 2026-10-19 | Git remote connected, initial commit `7bb1647` pushed |
| 2026-10-19 | Feature branches created (`feature/frontend`, `feature/ai-model`, `feature/backend-api`) |
| 2026-10-19 | React + Vite frontend scaffolded, mock prediction UI built |
| 2026-10-19 | Backend venv created, FastAPI app + MobileNetV3 inference implemented |
| 2026-10-19 | Dataset research: RealWaste + Kaggle Garbage Classification |
| 2026-10-19 | Model trained (10 epochs, cosine LR, 95.98% test accuracy) |
| 2026-10-19 | Phase 1: Confidence-aware predictions (top-3, uncertainty flagging) |
| 2026-10-19 | Phase 2: Image quality validation (brightness, blur, resolution) |
| 2026-10-19 | Phase 3: SQLite prediction history (CRUD, pagination, filtering) |
| 2026-10-19 | Phase 4: Analytics dashboard (KPIs, charts, trends, disclaimers) |
| 2026-10-19 | Full test suite: 33 backend tests passing |
| 2026-10-26 | Handover documentation created for VS Code migration |
