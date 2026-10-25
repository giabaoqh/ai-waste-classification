# AGENT RULES — Mandatory Constraints for AI Agents

> These rules apply to ANY AI coding agent working on this project.
> Violating these rules may cause irreversible damage to the project.

---

## 🔴 NEVER DO (Hard Constraints)

### Model & Checkpoint
1. **NEVER delete, overwrite, or replace** `checkpoints/mobilenetv3_best.pth` or `mobilenetv3_best_backup.pth` unless the user explicitly requests it.
2. **NEVER retrain the model** without explicit user instruction. Training takes ~24 minutes on GPU and could overwrite the current best checkpoint.
3. **NEVER fabricate** prediction probabilities, confidence scores, or model evaluation metrics. All values must come from actual model inference or verified database records.

### Git & Version Control
4. **NEVER commit, push, merge, or rebase** without explicit user approval.
5. **NEVER switch branches** without explicit user approval.
6. **NEVER rewrite git history** (force push, interactive rebase, amend commits).

### Data & Privacy
7. **NEVER store image binaries** in the database. The system is designed for metadata-only storage.
8. **NEVER expose** user-uploaded images or prediction history via public URLs.
9. **NEVER hardcode or fabricate** production analytics data. Dashboard must show only real database-backed data.

### Ethics
10. **NEVER claim environmental impact** (carbon savings, recycling weight, etc.) from prediction counts. The system counts classifications, not actual recycling outcomes.

---

## 🟡 ALWAYS DO (Mandatory Practices)

### Before Making Changes
11. **ALWAYS inspect** the current file state before modifying any file.
12. **ALWAYS check** `git status` before starting work to understand uncommitted changes.
13. **ALWAYS preserve** existing comments, docstrings, and functionality unless specifically asked to change them.

### Code Quality
14. **ALWAYS run tests** after making backend changes:
    ```powershell
    backend\.venv\Scripts\python.exe -m pytest backend\tests\ -v
    ```
15. **ALWAYS verify** frontend builds after making UI changes:
    ```powershell
    cd frontend && npm run build
    ```
16. **ALWAYS use parameterized SQL queries**. Never interpolate user input into SQL strings.

### API Design
17. **ALWAYS return** the `is_uncertain` flag and `uncertainty_note` in prediction responses when confidence < threshold.
18. **ALWAYS include** Vietnamese text (`label_vi`, `guidance`, `tips`) in prediction responses.
19. **ALWAYS return** the `quality_assessment` object in prediction responses.

### Documentation
20. **ALWAYS update** relevant documentation when adding features.
21. **ALWAYS use Vietnamese** for user-facing text and **English** for code identifiers and API field names.

---

## 🟢 PREFER (Best Practices)

22. **Prefer** modifying existing files over creating new ones when the change is small.
23. **Prefer** adding tests for any new backend functionality.
24. **Prefer** using existing Pydantic schemas over creating ad-hoc response dicts.
25. **Prefer** environment variables for configurable thresholds over hardcoded values.
26. **Prefer** logging important operations via the existing `logger` rather than print statements.

---

## Key File Locations (Quick Reference)

| Purpose | Path |
|---------|------|
| Backend entry point | `backend/app/main.py` |
| Database module | `backend/app/database.py` |
| Frontend UI | `frontend/src/App.jsx` |
| Global styles | `frontend/src/index.css` |
| Model checkpoint | `checkpoints/mobilenetv3_best.pth` |
| SQLite database | `data/waste_history.db` |
| Backend tests | `backend/tests/` |
| Dependencies | `backend/requirements.txt`, `frontend/package.json` |
| Startup script | `run_app.bat` |

---

## Running the Application

```powershell
# One-click startup (recommended):
.\run_app.bat

# Manual startup:
# Terminal 1 — Backend (from project root):
backend\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload

# Terminal 2 — Frontend:
cd frontend && npm run dev -- --host 127.0.0.1
```

**URLs**:
- Frontend: `http://127.0.0.1:5173`
- Backend API: `http://127.0.0.1:8000`
- Swagger Docs: `http://127.0.0.1:8000/docs`
- Health Check: `http://127.0.0.1:8000/api/health`
