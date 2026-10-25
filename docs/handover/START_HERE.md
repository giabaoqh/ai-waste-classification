# START HERE — Agent Handover Entry Point

> **Purpose**: Quick orientation for a new AI coding agent continuing this project.
> **Last verified**: 2026-10-26 (against live repository state)

---

## 1. What Is This Project?

**AI-Based Waste Recognition and Recycling Recommendation System** — a capstone project that classifies waste images into 6 categories using a fine-tuned MobileNetV3-Large model and provides Vietnamese-language recycling guidance.

| Layer | Technology | Location |
|-------|-----------|----------|
| Frontend | React 19 + Vite 8 | `frontend/` |
| Backend API | Python 3.13 + FastAPI | `backend/app/main.py` |
| AI Model | PyTorch MobileNetV3-Large | `checkpoints/mobilenetv3_best.pth` |
| Database | SQLite (WAL mode) | `data/waste_history.db` |

**Six waste classes**: `plastic`, `paper`, `metal`, `glass`, `organic`, `other`

---

## 2. Quick Start (One Command)

```powershell
# From project root:
.\run_app.bat
```

This script:
1. Activates `backend\.venv` and starts FastAPI on `http://127.0.0.1:8000`
2. Waits for backend health check to pass
3. Starts React dev server on `http://127.0.0.1:5173`
4. Opens browser automatically

**Manual startup** (if bat file doesn't work):
```powershell
# Terminal 1 — Backend
backend\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload

# Terminal 2 — Frontend
cd frontend && npm run dev
```

---

## 3. Current Git State

- **Active branch**: `feature/frontend`
- **Commits**: 4 total (`7bb1647` → `a95ea96`)
- **Uncommitted changes**: Significant — all Phase 1–4 implementations are uncommitted
- **Modified files**: `.gitignore`, `frontend/src/App.jsx`, `frontend/src/index.css`
- **Untracked files**: `backend/`, `run_app.bat`, `RUN_GUIDE.md`, `docs/DATASET_RESEARCH.md`

> ⚠️ **DO NOT** commit, push, merge, or switch branches without explicit user instruction.

---

## 4. Handover Document Index

Read these in order for full context:

| # | Document | Purpose |
|---|----------|---------|
| 1 | [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md) | Tech stack, architecture, team info, constraints |
| 2 | [CURRENT_STATE.md](CURRENT_STATE.md) | What's implemented (Phases 1–4), test results |
| 3 | [FILE_MAP.md](FILE_MAP.md) | Every file in the project with purpose annotations |
| 4 | [DECISIONS_AND_HISTORY.md](DECISIONS_AND_HISTORY.md) | Key architectural decisions and their rationale |
| 5 | [KNOWN_ISSUES.md](KNOWN_ISSUES.md) | Known bugs, limitations, technical debt |
| 6 | [TASK_BACKLOG.md](TASK_BACKLOG.md) | Remaining work and improvement ideas |
| 7 | [AGENT_RULES.md](AGENT_RULES.md) | Mandatory constraints for any AI agent working here |

---

## 5. Critical Things Not To Do

1. **Do NOT delete or overwrite** `checkpoints/mobilenetv3_best.pth` (16MB trained model)
2. **Do NOT retrain the model** unless explicitly requested
3. **Do NOT commit/push/merge** without user permission
4. **Do NOT store image binaries** in the database
5. **Do NOT fabricate** confidence scores or prediction data
