# Data card: WISDM (pocket)

**Source:** WISDM Lab, Activity Prediction dataset v1.1 (2010), Fordham University.
Paper to cite: Kwapisz, Weiss and Moore (2010), "Activity Recognition using Cell Phone Accelerometers", KDD-10 sensor workshop (check the exact details against the readme before citing).
**License:** none stated in the readme. It asks that the readme travel with the data and that the paper above is cited.
**Checksum (SHA-256, raw file):** 9aba4b1eaece56ab6d9187b16fc010c176e464750a4ec44c9d833514ecc1fa2e

## What it is
3-axis phone accelerometer, phone in the pocket, 36 users, original rate about 20 Hz, original units m/s^2.

## Processing (script: wisdm_preprocess.py)
- Cleaned as listed in docs/DECISIONS.md (malformed lines, timestamp 0, duplicates, sorting).
- Converted to g (divide by 9.80665).
- Interpolated onto a uniform 20 Hz grid inside continuous segments; a new segment starts after any gap over 0.5 s.
- Windows of 50 samples (2.5 s) with stride 25. No window crosses a gap, a user or an activity.
- Classes kept: 0 Walking, 1 Upstairs, 2 Downstairs, 3 Sitting, 4 Standing. Jogging dropped.
- Not normalized: the file holds raw g.

## Output
data/processed/wisdm.npz: X (29553, 50, 3) float32, y, subject_id, meta.

## Class counts (windows)
| Class | Windows | Share | Users with this class |
|---|---|---|---|
| 0 Walking | 16,562 | 56.0% | 36 |
| 1 Upstairs | 4,808 | 16.3% | 32 |
| 2 Downstairs | 3,925 | 13.3% | 32 |
| 3 Sitting | 2,361 | 8.0% | 23 |
| 4 Standing | 1,897 | 6.4% | 24 |

## Checks that passed
Shapes and dtype, no NaN, labels 0 to 4, user ids 1 to 36. Sitting and standing windows have mean magnitude 1.0026 g.

## Known issues
- Strongly imbalanced (Walking is 56% of windows), so macro F1 is the headline metric.
- Not every user has every class (only 23 users have Sitting, 24 have Standing). Some test users will lack classes.
- git_commit in meta is “unknown” because the file was built from a local folder that is not a git repository. The script src/data/wisdm_preprocess.py produces it (see D-entry “WISDM pipeline decisions” in docs/DECISIONS.md).
