# Decision Log

One entry per decision. Newest at the bottom.  
Source of truth: `PROTOCOL.md` (tag: protocol-v1-sprint).

---

## D-001: Protocol locked
- Date: 2026-11-12
- Decision: PROTOCOL.md (protocol-v1-sprint) is fixed. Any change requires team agreement + new dated entry here.
- Owner: All

## D-002: Branching and PRs
- Date: 2026-11-12
- Decision: Each member works on their own branch. PRs into main. Smoke test must pass.
- Owner: All

## D-003: Raw data never committed
- Date: 2026-11-12
- Decision: `data/raw/` and `data/processed/` are git-ignored. Only `splits/*.json` and small result CSVs are tracked.
- Owner: All

## D-004: Signal & preprocessing (locked by protocol)
- 3-axis accelerometer only, units in **g**
- Anti-alias filter + resample to **20 Hz**
- Windows: 50 samples (2.5 s), stride 25
- Never across gap > 0.5 s, subject change, or activity change
- Classes: 0 walking, 1 upstairs, 2 downstairs, 3 sitting, 4 standing

## D-005: Normalization
- Pipelines save **raw g** only.
- Normalization (source-train mean/std) happens only inside the harness and is saved in the checkpoint.

## D-006: Tuning rule
- 4 random configs per model
- Only on UCI-HAR, split 0, seed 0, max 10 epochs
- Best config is reused for WISDM and PAMAP2-hand

2026-11-15 (Sharanya): PAMAP2-hand uses the hand IMU +/-16 g accelerometer (columns 5-7). Class map from the readme activity list (II.2): walking=4, upstairs=12, downstairs=13, sitting=2, standing=3; activity 0 and all other activities dropped. Only the Protocol folder is read; the Optional folder holds none of the 5 classes.

2026-11-15 (Sharanya): PAMAP2 NaN rule: runs of up to 50 samples (0.5 s at 100 Hz) are filled by linear interpolation inside a single-activity segment; longer runs cut the segment; NaN at segment edges is trimmed. In the real data the longest run is 28 samples, so nothing is cut. Alternative considered: drop every window that contains a NaN.

2026-11-15 (Sharanya): PAMAP2 subject 109 has no windows for the 5 classes (Protocol: rope jumping and transient only; Optional: TV, computer work, car driving, folding laundry, house cleaning, soccer). Usable subjects: 101-108. Needs a team decision on the 5/1/3 splits. See docs/challenges_log.md.
