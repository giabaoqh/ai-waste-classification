# TASK BACKLOG — Remaining Work & Improvement Ideas

---

## Priority Levels
- 🔴 **P0 — Critical**: Must be done for project submission
- 🟡 **P1 — Important**: Improves quality significantly
- 🟢 **P2 — Nice-to-have**: Enhancements for extra credit or future work

---

## 🔴 P0 — Critical (Submission-Required)

### T-001: Commit All Phase 1–4 Work
- All Phase 1–4 implementations are uncommitted (working tree only).
- Create structured commits on appropriate branches.
- Merge feature branches to `main` following the team's review process.
- **Risk**: Working tree could be lost to accidental `git reset` or disk failure.

### T-002: Update Outdated Documentation
- `docs/PROGRESS.md` — Only covers initial setup. Update to reflect Phases 1–4 completion.
- `docs/TASKS.md` — Most checkboxes still `[ ]`. Mark completed items.
- `README.md` — Still mentions "PostgreSQL or Supabase" as database. Update to SQLite.
- These files will be visible to course graders.

### T-003: Final Presentation Materials
- Prepare demo flow and screenshots for capstone presentation.
- Document model metrics and system capabilities.

---

## 🟡 P1 — Important (Quality Improvements)

### T-004: Refactor App.jsx into Components
- Split the ~67KB monolithic `App.jsx` into focused components:
  - `UploadPanel.jsx` — Drag-drop, file input, camera capture
  - `PredictionResult.jsx` — Classification result card, top-3, recommendations
  - `HistoryTab.jsx` — Prediction history list with pagination
  - `DashboardTab.jsx` — Analytics dashboard with charts
  - `QualityWarning.jsx` — Image quality warning alerts
- Extract shared state into custom hooks or context.

### T-005: Add NumPy to requirements.txt
- `import numpy as np` in `main.py` but not in `requirements.txt`.
- Add explicit dependency: `numpy>=1.24.0`.

### T-006: Make CORS Origins Configurable
- Currently hardcoded to `localhost:5173/3000` and `127.0.0.1:5173/3000`.
- Add environment variable `CORS_ORIGINS` for flexible deployment.

### T-007: Add `has_quality_warnings` Column to Database
- Replace fragile `LIKE '%"has_warnings": true%'` JSON string matching with a proper boolean column.
- Requires schema migration (add column, backfill existing records).

### T-008: Add Frontend Error Boundary
- Wrap React tree in an error boundary component.
- Show recovery UI instead of white screen on runtime errors.

### T-009: Add Frontend Automated Tests
- Set up testing framework (Vitest + React Testing Library).
- Add tests for key user flows: upload, view results, history navigation.

---

## 🟢 P2 — Nice-to-Have (Future Enhancements)

### T-010: Image Upload History Thumbnails
- Option to store small thumbnails (not original images) for history browsing.
- Requires privacy policy consideration and user consent.

### T-011: Export Analytics to CSV/PDF
- Add export functionality to the analytics dashboard.
- Useful for course reports and presentations.

### T-012: Model Retraining Pipeline
- Automated pipeline to retrain model with new images.
- Include data validation, augmentation, and evaluation steps.
- Currently, training scripts exist but require manual execution.

### T-013: Dark Mode Toggle
- Frontend currently has dark theme elements but no toggle.
- Add persistent user preference for light/dark mode.

### T-014: Internationalization (i18n)
- All UI text is hardcoded in Vietnamese.
- Add i18n support for English and other languages.
- Move `RECOMMENDATIONS` dict to a translatable format.

### T-015: PWA (Progressive Web App) Support
- Add service worker for offline capability.
- Add app manifest for "Add to Home Screen" on mobile devices.

### T-016: Database Backup/Restore
- Add endpoint to export/import prediction history as JSON.
- Useful for migration between machines.

### T-017: Model Performance Monitoring
- Track inference latency, GPU memory usage, and prediction distribution over time.
- Alert if model performance degrades (confidence drift detection).

### T-018: Sub-Class Classification
- Extend model to classify specific waste subtypes (e.g., PET vs HDPE plastic).
- Requires new training data and potentially a hierarchical classifier.

---

## Notes for New Agent

When picking up tasks:
1. Always check current git status before making changes.
2. Run the existing test suite before and after modifications.
3. Consult `AGENT_RULES.md` for mandatory constraints.
4. Ask the user before committing, switching branches, or modifying the model checkpoint.
