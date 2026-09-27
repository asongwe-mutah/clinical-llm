#!/usr/bin/env python3
"""Install a downloaded adapter into outputs/clinical-qlora, safely.

Google Drive folder downloads arrive as one or more zips with names like
``clinical-qlora-20260927T034859Z-1-001.zip``, sometimes nested, sometimes
carrying checkpoint-* directories alongside the final adapter. This finds the
real adapter, backs up whatever is currently installed, and puts the new one
in place.

    python scripts/install_adapter.py                 # auto-find in ~/Downloads
    python scripts/install_adapter.py path/to.zip     # or point at one
    python scripts/install_adapter.py --dry-run       # show the plan only

Nothing is deleted: the existing adapter is renamed, never removed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEST = REPO / "outputs" / "clinical-qlora"
MARKER = "adapter_config.json"


def find_sources(explicit: list[str]) -> list[tuple[Path, float]]:
    """Return (path, adapter-save-time) candidates, newest first.

    Browsers may auto-extract a Drive download, so the source can be a
    DIRECTORY rather than a zip -- and globbing only *.zip would then match
    older archives lying around in Downloads and silently install a stale
    adapter. Both shapes are considered, ranked by when the adapter was
    actually saved.
    """
    if explicit:
        cands = [Path(p).expanduser().resolve() for p in explicit]
    else:
        dl = Path.home() / "Downloads"
        cands = [
            p
            for p in dl.glob("clinical-qlora*")
            if ".download" not in str(p) and not p.name.endswith((".crdownload", ".part"))
        ]

    out: list[tuple[Path, float]] = []
    for c in cands:
        if c.is_dir():
            cfg = c / MARKER
            if cfg.exists():
                out.append((c, cfg.stat().st_mtime))
            else:
                ck = [d / MARKER for d in c.glob("checkpoint-*") if (d / MARKER).exists()]
                if ck:
                    out.append((c, max(f.stat().st_mtime for f in ck)))
        elif c.suffix == ".zip":
            try:
                with zipfile.ZipFile(c) as zf:
                    times = [
                        dt.datetime(*i.date_time).timestamp()
                        for i in zf.infolist()
                        if i.filename.endswith(MARKER)
                    ]
                if times:
                    out.append((c, max(times)))
            except zipfile.BadZipFile:
                continue
    out.sort(key=lambda t: t[1], reverse=True)
    return out


def extract_all(zips: list[Path], into: Path) -> None:
    for z in zips:
        print(f"  extracting {z.name} ({z.stat().st_size / 1e6:.0f} MB)")
        try:
            with zipfile.ZipFile(z) as zf:
                zf.extractall(into)
        except zipfile.BadZipFile:
            sys.exit(
                f"\n{z.name} is not a valid zip -- if the download is still "
                "running, wait for it to finish and re-run."
            )


def find_adapter(root: Path) -> Path | None:
    """Prefer a final adapter; fall back to the newest checkpoint."""
    finals, ckpts = [], []
    for cfg in root.rglob(MARKER):
        d = cfg.parent
        (ckpts if d.name.startswith("checkpoint-") else finals).append(d)
    if finals:
        # Shallowest wins: the top-level save, not something nested.
        return min(finals, key=lambda p: len(p.parts))
    if ckpts:
        newest = max(ckpts, key=lambda p: int(p.name.rsplit("-", 1)[-1]))
        print(f"  no final adapter in the download; using {newest.name}")
        return newest
    return None


def describe(d: Path, skip_checkpoints: bool = True) -> str:
    cfg = d / MARKER
    try:
        j = json.loads(cfg.read_text())
        base, r = j.get("base_model_name_or_path"), j.get("r")
    except Exception:
        base, r = "?", "?"
    when = dt.datetime.fromtimestamp(cfg.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    size = sum(
        f.stat().st_size
        for f in d.rglob("*")
        if f.is_file()
        and not (skip_checkpoints and any(p.startswith("checkpoint-") for p in f.parts))
    ) / 1e6
    return f"base={base} r={r} saved={when} size={size:.0f}MB"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("zips", nargs="*", help="zip(s) to install; default: search ~/Downloads")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    sources = find_sources(args.zips)
    if not sources:
        sys.exit(
            "No clinical-qlora source found in ~/Downloads (looked for both a\n"
            "folder and a zip). If a download is still running, wait for the\n"
            "*.download bundle to disappear, then re-run."
        )

    print("candidates (newest adapter first):")
    for i, (p, t) in enumerate(sources):
        when = dt.datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M")
        kind = "dir" if p.is_dir() else "zip"
        mark = "  <- using" if i == 0 else "     skipped"
        print(f"  [{when}] {kind:3} {p.name}{mark}")
    chosen = sources[0][0]
    if len(sources) > 1:
        print("\n  (pass a path explicitly to override)")
    print()

    stage = Path(tempfile.mkdtemp(prefix="adapter-"))
    try:
        if chosen.is_dir():
            src = find_adapter(chosen)
        else:
            extract_all([chosen], stage)
            src = find_adapter(stage)
        if src is None:
            sys.exit(f"\nNo {MARKER} anywhere in the download. Wrong archive?")

        print(f"\n  incoming: {describe(src)}")
        if DEST.exists() and (DEST / MARKER).exists():
            print(f"  current : {describe(DEST)}")
        else:
            print("  current : (nothing installed)")

        if args.dry_run:
            print("\n--dry-run: nothing changed.")
            return

        if DEST.exists():
            stamp = dt.datetime.fromtimestamp(
                (DEST / MARKER).stat().st_mtime
            ).strftime("%Y%m%d")
            backup = DEST.with_name(f"clinical-qlora-backup-{stamp}")
            n = 1
            while backup.exists():
                n += 1
                backup = DEST.with_name(f"clinical-qlora-backup-{stamp}-{n}")
            shutil.move(str(DEST), str(backup))
            print(f"\n  backed up old adapter -> {backup.name}/")

        DEST.parent.mkdir(parents=True, exist_ok=True)
        # Leave checkpoint-* behind: each carries optimizer state (hundreds of
        # MB) that inference never reads. The adapter itself is ~140 MB.
        shutil.copytree(src, DEST, ignore=shutil.ignore_patterns("checkpoint-*"))
        print(f"  installed -> {DEST.relative_to(REPO)}/  (checkpoints skipped)")

        print(f"\nverify: {describe(DEST)}")
        files = sorted(p.name for p in DEST.iterdir() if p.is_file())
        print("files :", ", ".join(files))
        missing = [f for f in (MARKER, "adapter_model.safetensors") if not (DEST / f).exists()]
        if missing:
            sys.exit(f"\nMISSING: {missing} -- do not evaluate this.")
        print("\nReady. Run the eval next.")
    finally:
        shutil.rmtree(stage, ignore_errors=True)


if __name__ == "__main__":
    main()
