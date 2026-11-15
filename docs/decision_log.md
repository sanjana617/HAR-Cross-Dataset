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
# Decision Log


- Best config is reused for WISDM and PAMAP2-hand

## WISDM Dataset

- Dataset file: `wisdm.npz`
- SHA-256: `25aa9ab426c0f1f4c662566059321e5a3c5c3d1772c8a4143882c2b65b5c7ac0`
- Dataset contains 29,553 windows from 36 subjects.
- Input shape: `(29553, 50, 3)`.
- Sampling frequency: 20 Hz.
- Window size: 50 samples.
- Units: g.
- Data is not normalized.
- Preprocessing: interpolation within segments; jogging dropped.
- All five activity classes are present.
- Generated `splits/wisdm.json` using seed 0 and real subject IDs.
- All three splits passed subject-count, disjointness and class-presence checks.
- Metadata currently records `git_commit: unknown`. Awaiting confirmation from the dataset owner.

## UCI-HAR Dataset

- Dataset contains 8,355 windows from 30 subjects.
- Input shape: `(8355, 50, 3)`.
- Sampling frequency: 20 Hz.
- Window size: 50 samples.
- Units: g.
- Data is not normalized.
- Preprocessing method: continuous signal stitching.

2026-11-15 (Shreya): UCI-HAR uses the preferred stitch method (amendment 5), not the fallback. Train 7352 rows chained into 600 segments, test 2947 rows into 257. Windows never cross a gap in the overlap. Output: 8355 windows, 30 subjects. See `docs/data_cards/uci_har.md`.
