
"""Make fixed subject-wise splits for each HAR dataset.

Run:
    python scripts/make_splits.py
    python scripts/make_splits.py --real

Output: splits/{domain}.json
"""

import argparse
import json
from pathlib import Path

import numpy as np

SEED = 0
N_SPLITS = 3

SIZES = {
    "uci": (21, 4, 5),
    "wisdm": (25, 5, 6),
    "pamap2_hand": (None, None, None),
}

PAMAP2_SPLITS = [
    {
        "split": 0,
        "train": [102, 106, 107, 108],
        "val": [101],
        "test": [103, 104, 105],
    },
    {
        "split": 1,
        "train": [102, 103, 104, 105],
        "val": [108],
        "test": [101, 106, 107],
    },
    {
        "split": 2,
        "train": [101, 103, 104, 106, 107],
        "val": [105],
        "test": [102, 108],
    },
]


def get_ids(domain, real):
    if real:
        ids = np.unique(
            np.load(f"data/processed/{domain}.npz")["subject_id"]
        )
    elif domain == "pamap2_hand":
        ids = np.array([101, 102, 103, 104, 105, 106, 107, 108])
    else:
        ids = np.arange(1, sum(SIZES[domain]) + 1)

    return sorted(int(i) for i in ids)


def make_splits(domain, ids):
    if domain == "pamap2_hand":
        return PAMAP2_SPLITS

    n_train, n_val, n_test = SIZES[domain]
    rng = np.random.default_rng(SEED)
    splits = []

    for k in range(N_SPLITS):
        order = [int(s) for s in rng.permutation(ids)]
        test = order[:n_test]
        val = order[n_test:n_test + n_val]
        train = order[n_test + n_val:]

        splits.append({
            "split": k,
            "train": sorted(train),
            "val": sorted(val),
            "test": sorted(test),
        })

    return splits


def check(domain, ids, splits):
    assert len(splits) == N_SPLITS

    for split in splits:
        train = set(split["train"])
        val = set(split["val"])
        test = set(split["test"])

        assert not (train & val)
        assert not (train & test)
        assert not (val & test)
        assert train | val | test == set(ids)

    if domain == "pamap2_hand":
        all_test = [
            subject
            for split in splits
            for subject in split["test"]
        ]
        assert sorted(all_test) == sorted(ids)
        assert len(all_test) == len(set(all_test))
    else:
        n_train, n_val, n_test = SIZES[domain]
        for split in splits:
            assert len(split["train"]) == n_train
            assert len(split["val"]) == n_val
            assert len(split["test"]) == n_test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true")
    args = parser.parse_args()

    Path("splits").mkdir(exist_ok=True)

    for domain in SIZES:
        ids = get_ids(domain, args.real)
        splits = make_splits(domain, ids)
        check(domain, ids, splits)

        output = {
            "domain": domain,
            "seed": SEED,
            "splits": splits,
        }

        with open(f"splits/{domain}.json", "w") as file:
            json.dump(output, file, indent=2)

        print(f"{domain}: OK ({len(ids)} subjects)")


if __name__ == "__main__":
    main()
