# Working on AI Waste Classification

The multisource training runs completed on 2026-11-10. Read
`CHATGPT_HANDOVER.md`, `PROJECT_STATUS.md` and `docs/RESUME_MULTISOURCE.md`
for the saved state; do not restart historical training/download commands.
The user's latest decision is to stop prioritizing MobileNet restoration and
continue the YOLO/battery direction. Existing MobileNet code remains legacy;
this decision does not itself request deleting code or checkpoint binaries.

Read `docs/handover/AGENT_RULES.md` before changing this project. Use Vietnamese
for user-facing text and English for identifiers. Treat handover status reports as
historical context; verify claims against the current code and Git state.

## Repository map

- `backend/app/main.py`: FastAPI schemas, startup, prediction, history and map APIs.
- `backend/app/detector.py`: SSDLite object proposals and crop classification.
- `backend/app/non_waste_detector.py`: ImageNet gatekeeper and OOD heuristics.
- `backend/app/gradcam.py`: model explanations.
- `backend/app/database.py`: SQLite metadata, feedback and litter reports.
- `backend/app/knowledge_base.py` and `waste_disposal_kb.json`: disposal guidance.
- `backend/app/battery_screen.py` and `safety_routes.py`: experimental two-label
  battery screening; separate from the legacy classifier and nine-group detector.
- `backend/app/yolo_detector.py`, `yolo_routes.py`, `yolo_detector_manifest.json`:
  Task 02A verified object-group detection endpoint `/api/yolo/detect` and runtime readiness.
- `backend/app/safety_guidance.py` and `collection_contacts.json`: user-reported
  precautionary guidance and source-backed, currently unconfirmed collection candidates.
- `backend/scripts/`: dataset preparation and training.
- `frontend/src/App.jsx` and `frontend/src/components/`: React interface.
- `docs/LOCAL_SETUP.md`: verified local commands and current limitations.
- `docs/YOLO_PILOT.md`: isolated YOLO experiment commands and TACO pilot scope.

## Model contract

The current training script and `checkpoints/class_mapping.json` define seven
classes: `plastic`, `paper`, `metal`, `glass`, `organic`, `other`, `medical`.
`class_mapping_6class.json` and older documentation refer to the earlier model.
Runtime class order comes from checkpoint `target_classes`; verify the classifier
head and matching evaluation metadata before changing labels or replacing weights.
Do not present stored evaluation JSON as a new measurement on this machine.

The YOLO multisource experiment uses nine object groups, not these seven material labels.
Its weights live under `runs/waste_pilot`; never load them into the MobileNet
checkpoint path or describe pilot validation as production/test-set accuracy.
Task 02A exposes it via a separate experimental backend endpoint. Verify the
pinned manifest/hash/label order; no automatic mapping to materials or hazards.
Task 02B completed the experimental main YOLO UI. Task 03A added the isolated
Teacher–Student preparation/review pipeline; no Teacher inference or new training
has run. See `PROJECT_STATUS.md`, `docs/API_CONTRACT.md` and
`docs/TEACHER_STUDENT_PIPELINE.md` before continuing.

The battery experiment uses `battery`, `not_battery` under `runs/battery_screen`.
Require a measured report and matching checkpoint hash; never infer safety from
a negative result. Medical/radiation guidance is based on user-reported suspicion,
not a trained medical detector, infection assay or radiation measurement.
Read `docs/MULTISOURCE_BATTERY_RESULTS.md` and `docs/HAZARD_ROADMAP_DANANG.md`.
The user confirmed there are no local Da Nang photos yet.

Weights and datasets are excluded from Git. A successful clone or health HTTP 200
does not establish inference readiness: check `checkpoint_loaded` and
`ai_model_loaded`. Never create random weights as a substitute for the trained model.

## Development and verification

Use the repo-local `.agents/skills/ai-waste-dev/SKILL.md` for setup, AI integration,
dataset, calibration or prediction changes. Run commands from the repository root
using `backend\.venv\Scripts\python.exe`, not the global Python installation.

- Environment: `backend\.venv\Scripts\python.exe scripts\check_setup.py`
- Backend changes: `backend\.venv\Scripts\python.exe -m pytest backend\tests -v`
- Frontend changes: `npm.cmd --prefix frontend run build` and
  `npm.cmd --prefix frontend run lint`.

Task 01 standardized the API contract; read `docs/API_CONTRACT.md`. Pytest now
starts an isolated local API/temporary database for marked integration cases.
Use `-m "not integration and not model"` for offline units. Model inference is
opt-in with `--run-model-tests` and requires genuine already-local artifacts;
missing legacy weights are explicitly skipped, never restored or downloaded.
Standalone benchmark scripts are excluded from collection to prevent import
side effects. See `docs/LOCAL_SETUP.md`; report skipped prerequisites honestly.

Preserve checkpoint binaries and data. Training requires an explicit request.
Keep Git mutations within the user's request and the existing handover rules.
