#!/usr/bin/env python3
"""Download MPS SLAM points + observations only for the episodes in a prediction file.

Space-efficient: skips VRS / body / video / etc. Downloads into ``$FOM_NYMERIA_PLUS``.

    python -m fom.visualization.pointcloud.download_mps --url-json urls.json --preds preds.pkl --list
    python -m fom.visualization.pointcloud.download_mps ... --one         # test one episode
    python -m fom.visualization.pointcloud.download_mps ... --all-used -y
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from fom.evaluation.io import load_sequences
from fom.preprocessing.nymeria.io.download import download_sequence
from fom.visualization.common import nymeria_plus_root

# Minimal groups for observation-filtered semidense PCD.
GROUPS = ("slam", "slam_semidense_observations")


def pred_episodes(preds_path: Path) -> list[str]:
    preds = load_sequences(preds_path)
    return sorted(
        {
            k.split("_vgoal_seq_")[0].split("_mvgoal_seq_")[0]
            for k in preds
        }
    )


def parse_ep_from_key(key: str) -> str:
    take = key.split("___")[0]
    for marker in ("_vgoal_seq_", "_mvgoal_seq_"):
        if marker in take:
            return take.split(marker, 1)[0]
    return take


def episode_has_mps(ep: str) -> bool:
    slam = nymeria_plus_root() / ep / "recording_head" / "mps" / "slam"
    return (slam / "semidense_points.csv.gz").is_file() and (
        slam / "semidense_observations.csv.gz"
    ).is_file()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url-json", type=Path, required=True)
    parser.add_argument("--preds", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, default=None, help="Default: $FOM_NYMERIA_PLUS")
    parser.add_argument(
        "--one",
        action="store_true",
        help="Download only the first missing used episode (recommended test).",
    )
    parser.add_argument(
        "--all-used",
        action="store_true",
        help="Download slam+obs for every episode present in preds.",
    )
    parser.add_argument(
        "--match",
        type=str,
        default="",
        help="Force a specific episode name (must be in preds).",
    )
    parser.add_argument(
        "--seq-json",
        type=Path,
        default=None,
        help="pcd_sequences.json (or similar) with pred_key/episode fields - "
        "download only those episodes.",
    )
    parser.add_argument("--list", action="store_true", help="Print plan and exit.")
    parser.add_argument("-y", "--yes", action="store_true")
    args = parser.parse_args()

    if not args.url_json.is_file():
        raise SystemExit(f"Missing URL JSON: {args.url_json}")

    data = json.loads(args.url_json.read_text())
    seqs = data["sequences"]

    if args.seq_json is not None:
        payload = json.loads(args.seq_json.read_text())
        if not isinstance(payload, list):
            raise SystemExit("--seq-json must be a list of sequence dicts")
        eps = sorted(
            {
                (item.get("episode") or parse_ep_from_key(item["pred_key"]))
                for item in payload
            }
        )
    else:
        eps = pred_episodes(args.preds)
    missing = [e for e in eps if e not in seqs]
    if missing:
        print(f"WARNING: {len(missing)} pred eps not in URL JSON (e.g. {missing[:3]})")

    usable = [e for e in eps if e in seqs]
    need = [e for e in usable if not episode_has_mps(e)]
    have = [e for e in usable if episode_has_mps(e)]

    def size_gb(ep: str) -> float:
        return sum(seqs[ep][g]["file_size_bytes"] for g in GROUPS) / 1e9

    print(f"pred episodes: {len(eps)}")
    print(f"in URL JSON:   {len(usable)}")
    print(f"already local: {len(have)}")
    print(f"still needed:  {len(need)}")
    if need:
        print(f"needed total:  {sum(size_gb(e) for e in need):.1f} GB (compressed download)")
        print(f"groups:        {', '.join(GROUPS)}  (+ tiny LICENSE/metadata)")

    if args.list:
        for e in need[:20]:
            print(f"  {e}  {size_gb(e):.2f} GB")
        if len(need) > 20:
            print(f"  ... +{len(need)-20} more")
        return

    if args.match:
        targets = [args.match]
        if args.match not in usable:
            raise SystemExit(f"--match {args.match!r} not in preds&URL JSON")
    elif args.one:
        if not need:
            print("Nothing to download; all used episodes already have MPS assets.")
            return
        # Prefer smallest remaining for the first test.
        targets = [min(need, key=size_gb)]
    elif args.all_used or args.seq_json is not None:
        targets = need
        if not targets:
            print("Nothing to download.")
            return
    else:
        raise SystemExit("Pass --list, --one, --all-used, --seq-json, or --match <ep>")

    print("Targets:")
    for e in targets:
        status = "HAVE" if episode_has_mps(e) else "NEED"
        print(f"  [{status}] {e}  {size_gb(e):.2f} GB")

    args.outdir = args.outdir or nymeria_plus_root()
    for ep in targets:
        if episode_has_mps(ep) and not args.match:
            print(f"skip {ep} (already complete)")
            continue
        print(f"\n=== downloading {ep} ===", flush=True)
        download_sequence(
            url_json=args.url_json,
            out_root=args.outdir,
            match_key=ep,
            groups=list(GROUPS),
            ignore_existing=True,
            yes=args.yes or args.one,  # --one implies confirm
        )
        # Slam zips unpack all Aria devices; we only need head semidense.
        _prune_episode_to_head_mps(args.outdir / ep)
        if episode_has_mps(ep):
            print(f"OK: MPS assets ready for {ep}")
        else:
            slam = args.outdir / ep / "recording_head" / "mps" / "slam"
            print(f"FAILED: expected files under {slam}", file=sys.stderr)
            raise SystemExit(1)


def _prune_episode_to_head_mps(ep_dir: Path) -> None:
    """Delete wrist/observer MPS and unused head traj CSVs to save space."""
    if not ep_dir.is_dir():
        return
    for name in ("recording_lwrist", "recording_rwrist", "recording_observer"):
        p = ep_dir / name
        if p.is_dir():
            import shutil

            shutil.rmtree(p)
            print(f"pruned {name}/")
    slam = ep_dir / "recording_head" / "mps" / "slam"
    for name in (
        "closed_loop_trajectory.csv",
        "open_loop_trajectory.csv",
        "online_calibration.jsonl",
    ):
        f = slam / name
        if f.is_file():
            f.unlink()
            print(f"pruned recording_head/mps/slam/{name}")


if __name__ == "__main__":
    main()
