# KNOWN ISSUES — Bugs, Limitations & Technical Debt

---

## Critical Issues

### 1. All Phase 1–4 Work Is Uncommitted
- **Impact**: High risk of data loss if working directory is reset.
- **Detail**: The git history only has 4 commits (basic setup + frontend UI). The entire backend (`backend/`), database module, training scripts, and all Phase 1–4 features exist only in the git working tree as untracked/modified files.
- **Recommendation**: The user should commit this work to preserve it. Do NOT commit without explicit user instruction.

---

## Model Limitations

### 2. Six-Class Limitation
- The model classifies into exactly 6 categories: `plastic`, `paper`, `metal`, `glass`, `organic`, `other`.
- No sub-classification (e.g., PET vs HDPE plastic, cardboard vs newspaper).
- No contamination-level detection.

### 3. "Other" Class Is a Catch-All
- `other` is a residual category for mixed materials, rubber, fabric, and anything not in the other 5 classes.
- Confidence for `other` class predictions may be inflated because it's a broad catch-all.

### 4. Plastic and Metal Classes Have Lowest F1 Scores
- `plastic` F1 = 0.9219, `metal` F1 = 0.9253 — lower than other classes.
- The confusion matrix shows plastic↔metal and plastic↔glass confusion.
- These classes may benefit from more training data or harder augmentation.

---

## Frontend Issues

### 5. Monolithic App.jsx (~67 KB, ~1800+ lines)
- **Impact**: Hard to maintain and navigate.
- **Detail**: The entire UI (upload, results, history, dashboard) is in a single `App.jsx` file.
- **Recommendation**: Refactor into separate components (e.g., `UploadPanel.jsx`, `ResultCard.jsx`, `HistoryTab.jsx`, `DashboardTab.jsx`).

### 6. No Automated Frontend Tests
- Only backend has tests. No React component tests, no E2E browser tests (Playwright, Cypress, etc.).

### 7. No Error Boundary
- If any component throws a runtime error, the entire app crashes without a recovery UI.

---

## Backend Issues

### 8. Image Quality Warning Query Is Fragile
- `database.py:491` — The analytics quality warning count uses `LIKE '%"has_warnings": true%'` against JSON text.
- This string-matching approach may break if JSON serialization changes whitespace or key ordering.
- **Better approach**: Add a dedicated `has_quality_warnings` INTEGER column.

### 9. No NumPy in requirements.txt
- NumPy is imported directly in `main.py` (`import numpy as np`) but not listed in `requirements.txt`.
- It currently installs as a transitive dependency of torch/torchvision, but should be explicit.

### 10. Database VACUUM Not Executed After Bulk Delete
- `clear_all_history()` has a comment about VACUUM but doesn't execute it (VACUUM requires autocommit mode).
- After bulk deletes, SQLite file size won't shrink until VACUUM is run manually.

---

## Infrastructure Issues

### 11. No Authentication / Multi-User Support
- Prediction history is shared across all users on the same machine.
- No user isolation, session management, or access control.
- Acknowledged as out-of-scope for local capstone deployment.

### 12. CORS Origins Are Hardcoded
- `main.py:366-371` — Only allows `localhost:5173`, `127.0.0.1:5173`, `localhost:3000`, `127.0.0.1:3000`.
- If deployed on a different host/port, CORS will block requests.
- Should be configurable via environment variable.

### 13. No Rate Limiting
- The predict endpoint has no throttling. Rapid-fire requests could overwhelm GPU memory on the 4GB RTX 3050.

---

## Documentation Issues

### 14. `docs/PROGRESS.md` Is Outdated
- Only covers initial setup (Phase 0 / environment inspection). All Phase 1–4 progress is not reflected.

### 15. `docs/TASKS.md` Checkboxes Are Outdated  
- Most tasks are still marked `[ ]` (incomplete) despite being fully implemented.

### 16. README Tech Stack Section Lists PostgreSQL/Supabase
- `README.md:56` mentions PostgreSQL/Supabase as planned database, but SQLite was chosen and implemented.
