# FILE MAP — Project Structure Reference

> Every significant file in the project with its purpose.

---

## Root Directory

```
D:\AI-Based Waste Recognition and Recycling Recommendation System\
│
├── .git/                          # Git repository data
├── .gitignore                     # Excludes: .venv, node_modules, *.pth, data/, etc.
├── README.md                      # Project overview, team roles, quickstart guide
├── RUN_GUIDE.md                   # Detailed system startup documentation
├── run_app.bat                    # One-click startup script (backend + frontend)
│
├── backend/                       # Python backend (FastAPI + AI inference)
├── checkpoints/                   # Trained model weights & metadata
├── data/                          # SQLite database & training datasets
├── docs/                          # Project documentation
└── frontend/                      # React + Vite web application
```

---

## `backend/`

```
backend/
├── .venv/                         # Python 3.13 virtual environment (NOT committed)
├── requirements.txt               # pip dependencies (fastapi, torch, torchvision, pillow, pydantic)
│
├── app/
│   ├── __init__.py                # Package marker (64 bytes)
│   ├── main.py                    # ★ CORE: FastAPI app, all endpoints, inference logic (842 lines)
│   │                              #   - Lifespan: model loading, warmup, DB init
│   │                              #   - POST /api/predict: upload → validate → infer → recommend → save
│   │                              #   - GET/DELETE /api/history*: CRUD history
│   │                              #   - GET /api/analytics/dashboard: aggregated stats
│   │                              #   - RECOMMENDATIONS dict (Vietnamese recycling guidance)
│   │                              #   - assess_image_quality(): brightness, blur, resolution checks
│   │                              #   - Pydantic schemas for all request/response models
│   └── database.py                # ★ CORE: SQLite module (526 lines)
│                                  #   - init_db(): create table + indexes
│                                  #   - save_prediction_record(): insert with JSON serialization
│                                  #   - get_history_records(): paginated + filtered queries
│                                  #   - get_analytics_summary(): dashboard aggregation
│                                  #   - get_db_connection(): context manager with WAL mode
│
├── scripts/
│   ├── download_data.py           # Dataset downloader (Kaggle API integration)
│   ├── prepare_dataset.py         # RealWaste dataset preparation & splitting
│   ├── prepare_combined_dataset.py # Combined dataset merger (RealWaste + Kaggle)
│   └── train.py                   # MobileNetV3 training script (cosine LR, early stopping)
│
└── tests/
    ├── test_confidence_prediction.py  # Phase 1: softmax, top-3, uncertainty tests
    ├── test_image_quality.py          # Phase 2: brightness, blur, dimension tests
    ├── test_prediction_history.py     # Phase 3: CRUD, pagination, filtering tests
    ├── test_api_integration.py        # Phase 3: end-to-end API tests
    └── test_analytics_dashboard.py    # Phase 4: dashboard analytics tests
```

---

## `checkpoints/`

```
checkpoints/
├── mobilenetv3_best.pth           # ★ CRITICAL: Trained model weights (~16 MB) — DO NOT DELETE
├── mobilenetv3_best_backup.pth    # Safety backup of the above (~16 MB)
├── class_mapping.json             # Class name ↔ index mapping (6 classes)
├── test_evaluation.json           # Test set metrics (accuracy: 95.98%, per-class F1)
└── training_history.json          # Epoch-by-epoch training/validation metrics (10 epochs)
```

---

## `data/`

```
data/
├── waste_history.db               # SQLite database for prediction history (~28 KB)
├── raw/                           # Raw downloaded datasets (NOT committed)
└── processed/                     # Processed train/val/test splits (NOT committed)
```

---

## `frontend/`

```
frontend/
├── .env.example                   # Environment variable template
├── .gitignore                     # Frontend-specific ignores
├── .oxlintrc.json                 # OxLint configuration
├── README.md                      # Frontend-specific readme
├── index.html                     # HTML entry point (Vite SPA)
├── package.json                   # npm deps: react 19, vite 8, oxlint
├── package-lock.json              # Lockfile
├── vite.config.js                 # Vite config (React plugin)
│
├── dist/                          # Production build output (NOT committed)
├── node_modules/                  # npm packages (NOT committed)
├── public/                        # Static assets
│
└── src/
    ├── main.jsx                   # React entry point (ReactDOM.createRoot)
    ├── App.jsx                    # ★ CORE: Entire UI in one file (~67 KB, ~1800+ lines)
    │                              #   - Upload/camera/drag-drop interface
    │                              #   - Prediction results display with Top-3
    │                              #   - Uncertainty warnings
    │                              #   - Image quality feedback
    │                              #   - Prediction history tab with pagination
    │                              #   - Analytics dashboard tab with charts
    │                              #   - Vietnamese text throughout
    ├── App.css                    # Component-level styles (~2.9 KB)
    ├── index.css                  # ★ CORE: Global styles + dashboard styles (~32 KB)
    └── assets/                    # Static image assets
```

---

## `docs/`

```
docs/
├── PROJECT_SPEC.md                # Original project specification (6 classes, scope)
├── ARCHITECTURE.md                # System architecture diagram & data flow
├── TASKS.md                       # Task breakdown by feature branch
├── PROGRESS.md                    # Progress log (partially outdated — only covers initial setup)
├── DECISIONS.md                   # Architecture Decision Records (ADR-001 through ADR-005)
├── DATASET_RESEARCH.md            # Dataset research notes (RealWaste, Kaggle options)
│
└── handover/                      # ← YOU ARE HERE: Agent handover documentation
    ├── START_HERE.md
    ├── PROJECT_CONTEXT.md
    ├── CURRENT_STATE.md
    ├── FILE_MAP.md
    ├── KNOWN_ISSUES.md
    ├── DECISIONS_AND_HISTORY.md
    ├── TASK_BACKLOG.md
    └── AGENT_RULES.md
```
