import shutil, random
from pathlib import Path
RAW = Path("data/raw")
PROCESSED = Path("data/processed")
ARCHIVE = RAW / "archive"
HANDLOOM = RAW / "handloom_sarees"

if PROCESSED.exists():
    shutil.rmtree(PROCESSED)

for p in ["train/handloom_sarees","train/normal_sarees","val/handloom_sarees","val/normal_sarees"]:
    (PROCESSED/p).mkdir(parents=True, exist_ok=True)

def get_images(p):
    return list(p.rglob("*.jpg")) + list(p.rglob("*.jpeg")) + list(p.rglob("*.png"))

handloom_files = get_images(HANDLOOM)
archive_files = get_images(ARCHIVE)
print(f"Handloom: {len(handloom_files)}, Archive: {len(archive_files)}")

random.seed(42)
random.shuffle(handloom_files)
random.shuffle(archive_files)

def split_copy(files, train_dir, val_dir):
    n = int(len(files)*0.8)
    for f in files[:n]: shutil.copy2(f, train_dir / f.name)
    for f in files[n:]: shutil.copy2(f, val_dir / f.name)

split_copy(handloom_files, PROCESSED/"train"/"handloom_sarees", PROCESSED/"val"/"handloom_sarees")
split_copy(archive_files, PROCESSED/"train"/"normal_sarees", PROCESSED/"val"/"normal_sarees")
print("DONE! Data ready for training")