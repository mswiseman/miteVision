#!/usr/bin/env python3
import argparse
from pathlib import Path

def parse_class_map(spec: str) -> dict:
    """
    Parse a mapping spec like: "0:0,1:1,2:drop,3:2,4:3"
    Returns a dict {old_cls: new_cls or None}, where None means "drop".
    """
    mapping = {}
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            old_s, new_s = item.split(":")
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"Invalid mapping entry '{item}'. Use 'old:new' or 'old:drop'."
            )
        old = int(old_s)
        if new_s.lower() == "drop":
            mapping[old] = None
        else:
            try:
                mapping[old] = int(new_s)
            except ValueError:
                raise argparse.ArgumentTypeError(
                    f"Invalid target '{new_s}' in '{item}'. Use an int or 'drop'."
                )
    if not mapping:
        raise argparse.ArgumentTypeError("Class map cannot be empty.")
    return mapping

def remap_labels(label_dir: Path, class_map: dict, ext: str = ".txt") -> None:
    """
    Remap YOLO class IDs in all label files in label_dir according to class_map.
    Any class mapped to None is dropped.
    """
    ext = ext if ext.startswith(".") else f".{ext}"
    files = [p for p in label_dir.iterdir() if p.is_file() and p.suffix.lower() == ext.lower()]
    if not files:
        print(f"No '{ext}' files found in {label_dir}")
        return

    for path_in in sorted(files):
        updated_lines = []
        try:
            with path_in.open("r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        # Skip malformed lines
                        continue
                    try:
                        old_cls = int(parts[0])
                    except ValueError:
                        continue

                    # If class not in map, keep as-is or drop? Default: drop.
                    if old_cls not in class_map:
                        # comment the next line to *keep* unknown classes as-is:
                        continue
                        # To keep unknown classes: new_cls = old_cls
                    new_cls = class_map[old_cls]

                    if new_cls is None:
                        # drop this annotation
                        continue

                    coords = parts[1:5]  # x_center y_center width height (normalized)
                    updated_lines.append(f"{new_cls} {' '.join(coords)}\n")

            with path_in.open("w") as f:
                f.writelines(updated_lines)

            print(f"Processed {path_in.name}: kept/modified {len(updated_lines)} line(s)")
        except Exception as e:
            print(f"Error processing {path_in}: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Remap YOLO class IDs in label files; supports dropping classes."
    )
    parser.add_argument(
        "--label_dir",
        type=Path,
        required=True,
        help="Directory containing YOLO .txt label files.",
    )
    parser.add_argument(
        "--class_map",
        type=parse_class_map,
        required=True,
        help="Mapping spec, e.g. '0:0,1:1,2:drop,3:2,4:3'. Use 'drop' to remove a class.",
    )
    parser.add_argument(
        "--ext",
        type=str,
        default=".txt",
        help="Label file extension (default: .txt).",
    )
    args = parser.parse_args()

    if not args.label_dir.is_dir():
        raise SystemExit(f"Label directory not found: {args.label_dir}")

    remap_labels(args.label_dir, args.class_map, args.ext)
    print("Done remapping classes.")

if __name__ == "__main__":
    main()
