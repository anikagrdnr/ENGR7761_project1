"""
run this ONCE to split data (grouped into folder classes) into train test and val (70,15,15) 
then delete old data 
"""

# split_data.py  — run once, then delete or archive
import shutil, random
from pathlib import Path



SRC   = Path("data")
SEED  = 42
SPLIT = {"train": 0.70, "val": 0.15, "test": 0.15}

random.seed(SEED)
for cls_dir in sorted(SRC.iterdir()):
    if not cls_dir.is_dir() or cls_dir.name in ("train", "val", "test"):
        continue
    images = sorted(cls_dir.glob("*"))
    random.shuffle(images)
    n = len(images)
    cuts = [int(n * SPLIT["train"]), int(n * (SPLIT["train"] + SPLIT["val"]))]
    parts = {"train": images[:cuts[0]],
             "val":   images[cuts[0]:cuts[1]],
             "test":  images[cuts[1]:]}
    for split, files in parts.items():
        dest = SRC / split / cls_dir.name
        dest.mkdir(parents=True, exist_ok=True)
        for f in files:
            shutil.copy(f, dest / f.name)

