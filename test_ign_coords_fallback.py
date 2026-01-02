from __future__ import annotations

import argparse
from pathlib import Path

from src.pipeline.ign.coords_fallback import (
    build_sorted_records_with_fallback,
    extract_xy_from_ign_filename,
    infer_xy_from_pdal_bounds,
)


def _parse_input_list(p: Path) -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        if "," in line:
            filename, url = line.split(",", 1)
            items.append((filename.strip(), url.strip()))
        else:
            # allow bare filenames for local testing
            items.append((line, line))
    return items


def main() -> int:
    ap = argparse.ArgumentParser(description="Test IGN coords fallback (PDAL bounds -> XXXX_YYYY rename + sorted records)")
    ap.add_argument("--input", required=True, help="Path to IGN input list (filename,url per line)")
    ap.add_argument(
        "--dalles-dir",
        required=True,
        help="Directory containing already-downloaded LAZ/LAS (used for PDAL bounds inference + optional rename)",
    )
    ap.add_argument(
        "--compare",
        action="store_true",
        help="Compare coords extracted from filename vs coords inferred via PDAL bounds.",
    )
    ap.add_argument(
        "--fail-on-mismatch",
        action="store_true",
        help="Exit with non-zero code if any mismatch is detected (only with --compare).",
    )
    ap.add_argument(
        "--no-rename",
        action="store_true",
        help="Do not rename files when building sorted records (comparison still runs).",
    )
    ap.add_argument(
        "--write-sorted",
        default="",
        help="Optional output path for fichier_tri.txt (will be written as filename,url sorted)",
    )
    args = ap.parse_args()

    input_path = Path(args.input)
    dalles_dir = Path(args.dalles_dir)

    if not input_path.exists() or not input_path.is_file():
        raise FileNotFoundError(input_path)
    if not dalles_dir.exists() or not dalles_dir.is_dir():
        raise FileNotFoundError(dalles_dir)

    file_list = _parse_input_list(input_path)

    def log(msg: str) -> None:
        print(msg)

    mismatch_count = 0
    missing_name_coords = 0
    missing_pdal_coords = 0

    if args.compare:
        print("=== Compare name coords vs PDAL coords ===")
        for filename, _url in file_list:
            name_xy = extract_xy_from_ign_filename(filename)
            if name_xy is None:
                missing_name_coords += 1
            pdal_xy = infer_xy_from_pdal_bounds(dalles_dir / filename)
            if pdal_xy is None:
                missing_pdal_coords += 1

            if name_xy is not None and pdal_xy is not None and tuple(name_xy) != tuple(pdal_xy):
                mismatch_count += 1
                print(
                    f"MISMATCH {filename}: name={name_xy[0]:04d}_{name_xy[1]:04d} vs pdal={pdal_xy[0]:04d}_{pdal_xy[1]:04d}"
                )
            else:
                # keep output compact
                pass

        print(f"missing_name_coords={missing_name_coords}")
        print(f"missing_pdal_coords={missing_pdal_coords}")
        print(f"mismatch_count={mismatch_count}")

        if args.fail_on_mismatch and mismatch_count > 0:
            return 3

    # Build records for sorting / optional writing.
    # If user requested no rename, we do a light version: keep original filename in output.
    if args.no_rename:
        records = []
        for filename, url in file_list:
            xy = extract_xy_from_ign_filename(filename)
            if xy is None:
                inferred = infer_xy_from_pdal_bounds(dalles_dir / filename)
                if inferred is None:
                    log(f"Impossible d'inférer les coordonnées via PDAL: {filename}")
                    continue
                xy = inferred
            records.append(type("R", (), {"x_km": int(xy[0]), "y_km": int(xy[1]), "filename": filename, "url": url})())
        records = sorted(records, key=lambda r: (r.x_km, r.y_km))
    else:
        records = build_sorted_records_with_fallback(file_list=file_list, dalles_dir=dalles_dir, log=log)
    if not records:
        print("No records produced (no coords found and/or PDAL inference failed).")
        return 2

    print("\n=== Sorted records ===")
    for r in records:
        print(f"{r.x_km:04d}_{r.y_km:04d} -> {r.filename}")

    if args.write_sorted:
        out = Path(args.write_sorted)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(f"{r.filename},{r.url}\n")
        print(f"\nWritten: {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
