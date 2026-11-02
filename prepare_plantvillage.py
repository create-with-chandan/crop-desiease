from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and export PlantVillage into image folders.")
    parser.add_argument("--output", type=Path, default=Path("training-data"))
    parser.add_argument("--archive", type=Path, default=Path("datasets/plantvillage.zip"))
    args = parser.parse_args()

    if not args.archive.is_file():
        raise SystemExit(f"PlantVillage archive does not exist: {args.archive}")
    args.output.mkdir(parents=True, exist_ok=True)

    exported = 0
    classes: set[str] = set()
    with zipfile.ZipFile(args.archive) as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            path = Path(member.filename)
            if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            parts = path.parts
            if len(parts) < 2:
                continue
            label = parts[-2]
            destination = args.output / label
            destination.mkdir(parents=True, exist_ok=True)
            target = destination / f"plantvillage_{exported:06d}{path.suffix.lower()}"
            with archive.open(member) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
            classes.add(label)
            exported += 1

    print(f"Exported {exported} labeled images into {args.output}")
    print(f"Exported {len(classes)} classes")


if __name__ == "__main__":
    main()
