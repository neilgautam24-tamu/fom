#!/usr/bin/env python3
"""Download one Nymeria Plus episode, extract visible-goal bundles, delete raw data.

Designed for SLURM array jobs: each task processes a single episode end-to-end.

Pipeline per episode
--------------------
1. Download required Plus groups into ``<temp-root>/<episode>/``
2. Convert SMPL -> SMPL-X (``xdata_smplx_neutral.npz``)
3. Detect hand-object contacts
4. Extract + export ``visible_goal_sequence_bundles``
5. Move outputs to ``<out-root>/<episode>/`` (including original ``body/xdata_smpl_neutral.npz``)
6. Delete ``<temp-root>/<episode>/``

Usage
-----
    python -m fom.preprocessing.nymeria.pipeline.process_episode --episode-index 0
    python -m fom.preprocessing.nymeria.pipeline.process_episode --episode-name 20230607_s0_james_johnson_act0_e72nhq
    python -m fom.preprocessing.nymeria.pipeline.process_episode --episode-index 0 --skip-download --force
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REQUIRED_DOWNLOAD_GROUPS = [
    "body_processed",
    "slam",
    "timesync_and_imu",
    "object_bounding_box",
    "recording_head_data_data_vrs",
]

FINAL_OUTPUT_NAMES = [
    "visible_goal_sequence_bundles",
    "visible_goal_sequences.json",
    "visible_goal_sequences.csv",
    "hand_object_contacts.csv",
    "hand_object_contact_intervals.csv",
    "contact_summary.json",
    "contact_episode_counts.json",
    "contact_episode_counts_by_object.csv",
]

# Original Nymeria Plus body motion from ``body_processed`` (SMPL, not SMPL-X).
ORIGINAL_BODY_FILES = [
    "xdata_smpl_neutral.npz",
]

# Episode-level SMPL-X archive (written after high-quality conversion).
ARCHIVED_BODY_FILES = [
    *ORIGINAL_BODY_FILES,
    "xdata_smplx_neutral.npz",
]

CONTACT_OUTPUT_NAMES = [
    "hand_object_contacts.csv",
    "hand_object_contact_intervals.csv",
    "contact_summary.json",
    "contact_episode_counts.json",
    "contact_episode_counts_by_object.csv",
]

SEQUENCE_OUTPUT_NAMES = [
    "visible_goal_sequences.json",
    "visible_goal_sequences.csv",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_episode_list(url_json: Path, max_episodes: int) -> list[str]:
    data = json.loads(url_json.read_text())
    episodes = sorted(data["sequences"].keys())
    if max_episodes > 0:
        episodes = episodes[:max_episodes]
    return episodes


def setup_episode_logger(episode: str, log_root: Path) -> logging.Logger:
    log_root.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"episode.{episode}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(log_root / f"{episode}.log")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def update_pipeline_state(
    state_path: Path,
    episode: str,
    *,
    status: str,
    details: dict,
) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state: dict = {}
    if state_path.is_file():
        try:
            state = json.loads(state_path.read_text())
        except json.JSONDecodeError:
            state = {}
    state.setdefault("episodes", {})
    state["episodes"][episode] = {
        "status": status,
        "updated_at": utc_now(),
        **details,
    }
    state["updated_at"] = utc_now()
    state_path.write_text(json.dumps(state, indent=2))


def episode_is_done(out_dir: Path, required_smplx_mode: str) -> bool:
    needs, _ = episode_needs_work(out_dir, required_smplx_mode, force=False)
    return not needs


def detect_episode_gaps(out_dir: Path, required_smplx_mode: str) -> dict[str, bool]:
    """Return True for each artifact that is missing or below the required SMPL-X quality."""
    bundles_root = out_dir / "visible_goal_sequence_bundles"
    manifest = bundles_root / "manifest.json"

    gaps = {
        "manifest": not manifest.is_file(),
        "contacts": not (out_dir / "hand_object_contacts.csv").is_file(),
        "body_smpl": not (out_dir / "body" / "xdata_smpl_neutral.npz").is_file(),
        "body_smplx": not (out_dir / "body" / "xdata_smplx_neutral.npz").is_file(),
        "smplx_quality": True,
        "smpl_motion_in_bundles": False,
    }

    status_mode = "fast"
    status_path = out_dir / "episode_status.json"
    if status_path.is_file():
        try:
            status_mode = json.loads(status_path.read_text()).get("smplx_mode", "fast")
        except json.JSONDecodeError:
            pass
    gaps["smplx_quality"] = status_mode != required_smplx_mode

    if manifest.is_file():
        bundle_dirs = [
            p for p in bundles_root.iterdir() if p.is_dir() and p.name.startswith("vgoal_seq_")
        ]
        if bundle_dirs:
            gaps["smpl_motion_in_bundles"] = any(
                not (p / "smpl_motion.npz").is_file() for p in bundle_dirs
            )
        else:
            gaps["smpl_motion_in_bundles"] = True
    else:
        gaps["smpl_motion_in_bundles"] = True

    return gaps


def episode_needs_work(
    out_dir: Path,
    required_smplx_mode: str,
    *,
    force: bool,
) -> tuple[bool, dict[str, bool]]:
    if force:
        return True, {}
    gaps = detect_episode_gaps(out_dir, required_smplx_mode)
    return any(gaps.values()), gaps


def build_episode_plan(
    out_dir: Path,
    required_smplx_mode: str,
    *,
    force: bool,
) -> dict[str, object]:
    needs, gaps = episode_needs_work(out_dir, required_smplx_mode, force=force)
    if not needs:
        return {"skip": True, "gaps": gaps}

    if force:
        return {
            "skip": False,
            "gaps": gaps,
            "reconvert_smplx": True,
            "reextract_contacts": True,
            "reextract_sequences": True,
            "reexport_bundles": True,
            "skip_extract_goals": False,
        }

    reconvert_smplx = bool(
        gaps["smplx_quality"] or gaps["manifest"] or gaps["body_smplx"]
    )
    reextract_contacts = reconvert_smplx or gaps["contacts"]
    reextract_sequences = reextract_contacts or gaps["manifest"]
    reexport_bundles = bool(
        gaps["manifest"]
        or gaps["smpl_motion_in_bundles"]
        or gaps["body_smpl"]
        or reconvert_smplx
    )

    # Backfill only per-sequence SMPL / archived body motion without redoing detection.
    if (
        not gaps["smplx_quality"]
        and not gaps["manifest"]
        and not gaps["contacts"]
        and (gaps["smpl_motion_in_bundles"] or gaps["body_smpl"])
    ):
        reconvert_smplx = False
        reextract_contacts = False
        reextract_sequences = False
        reexport_bundles = True

    return {
        "skip": False,
        "gaps": gaps,
        "reconvert_smplx": reconvert_smplx,
        "reextract_contacts": reextract_contacts,
        "reextract_sequences": reextract_sequences,
        "reexport_bundles": reexport_bundles,
        "skip_extract_goals": not reextract_sequences,
    }


def restore_body_to_temp(seq_dir: Path, out_dir: Path, logger: logging.Logger) -> bool:
    """Copy archived episode-level SMPL from ``processed_data`` into the temp tree."""
    smpl_dst = seq_dir / "body" / "xdata_smpl_neutral.npz"
    if smpl_dst.is_file():
        return True
    smpl_src = out_dir / "body" / "xdata_smpl_neutral.npz"
    if not smpl_src.is_file():
        return False
    smpl_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(smpl_src, smpl_dst)
    logger.info("Restored archived SMPL -> %s", smpl_dst)
    return True


def seed_work_dir_from_output(
    out_dir: Path,
    work_dir: Path,
    plan: dict[str, object],
    logger: logging.Logger,
) -> None:
    """Reuse prior CSV/JSON outputs when only re-exporting bundles."""
    work_dir.mkdir(parents=True, exist_ok=True)
    if not plan["reextract_contacts"]:
        for name in CONTACT_OUTPUT_NAMES:
            src = out_dir / name
            if src.is_file():
                shutil.copy2(src, work_dir / name)
                logger.info("Reused contact artifact -> %s", work_dir / name)
    if plan["skip_extract_goals"]:
        for name in SEQUENCE_OUTPUT_NAMES:
            src = out_dir / name
            if src.is_file():
                shutil.copy2(src, work_dir / name)
                logger.info("Reused sequence artifact -> %s", work_dir / name)


def temp_episode_ready(seq_dir: Path) -> bool:
    """True when downloaded raw data is present for contact + RGB export."""
    return (
        (seq_dir / "body" / "xdata_smpl_neutral.npz").is_file()
        and (seq_dir / "recording_head" / "data" / "data.vrs").is_file()
        and (seq_dir / "recording_head" / "mps" / "slam" / "closed_loop_trajectory.csv").is_file()
    )


def run_cmd(
    cmd: list[str],
    *,
    logger: logging.Logger,
    cwd: Path | None = None,
    env: dict | None = None,
) -> None:
    logger.info("RUN %s", " ".join(cmd))
    merged_env = None
    if env is not None:
        merged_env = {**os.environ, **env}
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=merged_env, check=True)


def download_episode(
    *,
    python: str,
    url_json: Path,
    temp_root: Path,
    episode: str,
    logger: logging.Logger,
) -> None:
    logger.info("Downloading episode %s -> %s", episode, temp_root)
    cmd = [
        python,
        "-m",
        "fom.preprocessing.nymeria.io.download",
        "--url-json",
        str(url_json),
        "--outdir",
        str(temp_root),
        "--match",
        episode,
        "--groups",
        *REQUIRED_DOWNLOAD_GROUPS,
        "-y",
    ]
    run_cmd(cmd, logger=logger)


def convert_smpl_to_smplx(
    *,
    python: str,
    seq_dir: Path,
    smplx_mode: str,
    device: str,
    logger: logging.Logger,
    force_reconvert: bool = False,
) -> Path:
    smpl_npz = seq_dir / "body" / "xdata_smpl_neutral.npz"
    smplx_npz = seq_dir / "body" / "xdata_smplx_neutral.npz"
    if not smpl_npz.is_file():
        raise FileNotFoundError(f"Missing SMPL motion: {smpl_npz}")
    if smplx_npz.is_file():
        if force_reconvert:
            logger.info("Removing existing %s for %s reconversion", smplx_npz, smplx_mode)
            smplx_npz.unlink()
        else:
            logger.info("Reusing existing %s", smplx_npz)
            return smplx_npz

    cmd = [
        python,
        "-m",
        "fom.preprocessing.nymeria.body.smpl_to_smplx",
        "--input",
        str(smpl_npz),
        "--output",
        str(smplx_npz),
        "--mode",
        smplx_mode,
        "--device",
        device,
    ]
    run_cmd(cmd, logger=logger)
    if not smplx_npz.is_file():
        raise FileNotFoundError(f"SMPL-X conversion did not create {smplx_npz}")
    return smplx_npz


def extract_contacts(
    *,
    python: str,
    seq_dir: Path,
    work_dir: Path,
    stride: int,
    device_torch: str,
    logger: logging.Logger,
) -> None:
    smplx_npz = seq_dir / "body" / "xdata_smplx_neutral.npz"
    cmd = [
        python,
        "-m",
        "fom.preprocessing.nymeria.contacts.extract",
        "--npz",
        str(smplx_npz),
        "--sequence-dir",
        str(seq_dir),
        "--output-dir",
        str(work_dir),
        "--stride",
        str(stride),
        "--no-hand-vertices",
        "--device-torch",
        device_torch,
    ]
    run_cmd(cmd, logger=logger)


def extract_visible_goals(
    *,
    python: str,
    seq_dir: Path,
    work_dir: Path,
    stride: int,
    rgb_stride: int,
    max_rgb_edge: int,
    logger: logging.Logger,
    skip_extract: bool = False,
) -> None:
    smplx_npz = seq_dir / "body" / "xdata_smplx_neutral.npz"
    cmd = [
        python,
        "-m",
        "fom.preprocessing.nymeria.goals.extract",
        "--npz",
        str(smplx_npz),
        "--sequence-dir",
        str(seq_dir),
        "--input-dir",
        str(work_dir),
        "--output-dir",
        str(work_dir),
        "--stride",
        str(stride),
        "--rgb-stride",
        str(rgb_stride),
        "--max-rgb-edge",
        str(max_rgb_edge),
    ]
    if skip_extract:
        cmd.append("--skip-extract")
    run_cmd(cmd, logger=logger)


def preserve_original_body_motion(
    *,
    seq_dir: Path,
    out_dir: Path,
    logger: logging.Logger,
) -> dict[str, str]:
    """Copy episode-level SMPL / SMPL-X motion into ``processed_data/<episode>/body/``."""
    body_src = seq_dir / "body"
    body_dst = out_dir / "body"
    body_dst.mkdir(parents=True, exist_ok=True)
    saved: dict[str, str] = {}

    for name in ARCHIVED_BODY_FILES:
        src = body_src / name
        if not src.is_file():
            if name in ORIGINAL_BODY_FILES:
                raise FileNotFoundError(f"Missing original body motion file: {src}")
            continue
        dst = body_dst / name
        shutil.copy2(src, dst)
        logger.info("Saved body motion -> %s", dst)
        if name == "xdata_smpl_neutral.npz":
            saved["smpl_npz"] = str(dst)
        elif name == "xdata_smplx_neutral.npz":
            saved["smplx_npz"] = str(dst)

    return saved


def finalize_outputs(
    *,
    work_dir: Path,
    out_dir: Path,
    seq_dir: Path,
    episode: str,
    logger: logging.Logger,
    extra_status: dict,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    body_paths = preserve_original_body_motion(seq_dir=seq_dir, out_dir=out_dir, logger=logger)

    for name in FINAL_OUTPUT_NAMES:
        src = work_dir / name
        if not src.exists():
            continue
        dst = out_dir / name
        if dst.exists():
            if dst.is_dir():
                shutil.rmtree(dst)
            else:
                dst.unlink()
        shutil.move(str(src), str(dst))
        logger.info("Moved %s -> %s", src.name, dst)

    manifest = out_dir / "visible_goal_sequence_bundles" / "manifest.json"
    num_sequences = 0
    if manifest.is_file():
        num_sequences = int(json.loads(manifest.read_text()).get("num_sequences", 0))

    status = {
        "episode": episode,
        "status": "success",
        "finished_at": utc_now(),
        "num_visible_goal_sequences": num_sequences,
        **body_paths,
        **extra_status,
    }
    (out_dir / "episode_status.json").write_text(json.dumps(status, indent=2))
    logger.info("Wrote episode_status.json (%d sequences)", num_sequences)


def cleanup_temp(seq_dir: Path, logger: logging.Logger) -> None:
    if seq_dir.is_dir():
        logger.info("Deleting raw episode data: %s", seq_dir)
        shutil.rmtree(seq_dir)


def process_episode(
    *,
    episode: str,
    url_json: Path,
    temp_root: Path,
    out_root: Path,
    log_root: Path,
    state_path: Path,
    python: str,
    smplx_mode: str,
    device: str,
    device_torch: str,
    stride: int,
    rgb_stride: int,
    max_rgb_edge: int,
    skip_download: bool,
    keep_temp: bool,
    force: bool,
) -> dict:
    logger = setup_episode_logger(episode, log_root)
    seq_dir = temp_root / episode
    work_dir = seq_dir / "_pipeline_work"
    out_dir = out_root / episode
    t0 = time.time()

    logger.info("=" * 72)
    logger.info("Episode: %s", episode)
    logger.info("Temp:    %s", seq_dir)
    logger.info("Output:  %s", out_dir)

    plan = build_episode_plan(out_dir, smplx_mode, force=force)
    if plan["skip"]:
        logger.info("Already complete for mode=%s - skipping", smplx_mode)
        return {"status": "skipped", "episode": episode, "gaps": plan["gaps"]}

    logger.info("Work plan: %s", json.dumps({k: v for k, v in plan.items() if k != "gaps"}, indent=2))
    if plan["gaps"]:
        logger.info("Detected gaps: %s", plan["gaps"])

    update_pipeline_state(
        state_path,
        episode,
        status="running",
        details={"started_at": utc_now(), "plan": {k: v for k, v in plan.items() if k != "gaps"}},
    )

    try:
        work_dir.mkdir(parents=True, exist_ok=True)
        seed_work_dir_from_output(out_dir, work_dir, plan, logger)

        has_smpl = restore_body_to_temp(seq_dir, out_dir, logger)
        if not temp_episode_ready(seq_dir):
            if skip_download:
                raise FileNotFoundError(
                    f"Temp episode incomplete and --skip-download set: {seq_dir}"
                )
            download_episode(
                python=python,
                url_json=url_json,
                temp_root=temp_root,
                episode=episode,
                logger=logger,
            )
        elif not has_smpl and not (seq_dir / "body" / "xdata_smpl_neutral.npz").is_file():
            raise FileNotFoundError(
                f"Episode SMPL missing in temp and output; re-download required: {seq_dir}"
            )

        smplx_npz = convert_smpl_to_smplx(
            python=python,
            seq_dir=seq_dir,
            smplx_mode=smplx_mode,
            device=device,
            logger=logger,
            force_reconvert=bool(plan["reconvert_smplx"]),
        )
        if plan["reextract_contacts"]:
            extract_contacts(
                python=python,
                seq_dir=seq_dir,
                work_dir=work_dir,
                stride=stride,
                device_torch=device_torch,
                logger=logger,
            )
        if plan["reexport_bundles"]:
            extract_visible_goals(
                python=python,
                seq_dir=seq_dir,
                work_dir=work_dir,
                stride=stride,
                rgb_stride=rgb_stride,
                max_rgb_edge=max_rgb_edge,
                logger=logger,
                skip_extract=bool(plan["skip_extract_goals"]),
            )
        elapsed_s = time.time() - t0
        finalize_outputs(
            work_dir=work_dir,
            out_dir=out_dir,
            seq_dir=seq_dir,
            episode=episode,
            logger=logger,
            extra_status={
                "elapsed_s": round(elapsed_s, 1),
                "smplx_npz": str(smplx_npz),
                "smplx_mode": smplx_mode,
                "stride": stride,
                "rgb_stride": rgb_stride,
            },
        )
        if work_dir.is_dir():
            shutil.rmtree(work_dir, ignore_errors=True)
        if not keep_temp:
            cleanup_temp(seq_dir, logger)

        update_pipeline_state(
            state_path,
            episode,
            status="success",
            details={"elapsed_s": round(elapsed_s, 1), "out_dir": str(out_dir)},
        )
        logger.info("Done in %.1fs", elapsed_s)
        return {"status": "success", "episode": episode, "elapsed_s": elapsed_s}

    except Exception as exc:
        elapsed_s = time.time() - t0
        logger.exception("FAILED after %.1fs: %s", elapsed_s, exc)
        update_pipeline_state(
            state_path,
            episode,
            status="failed",
            details={"error": str(exc), "elapsed_s": round(elapsed_s, 1)},
        )
        fail_status = {
            "episode": episode,
            "status": "failed",
            "finished_at": utc_now(),
            "error": str(exc),
            "elapsed_s": round(elapsed_s, 1),
        }
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "episode_status.json").write_text(json.dumps(fail_status, indent=2))
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url-json", type=Path, required=True, help="Nymeria-Plus signed download URL json")
    parser.add_argument("--temp-root", type=Path, required=True, help="Scratch dir for raw downloads")
    parser.add_argument("--out-root", type=Path, required=True, help="Processed episode output dir")
    parser.add_argument("--log-root", type=Path, default=None, help="Default: <out-root>/logs")
    parser.add_argument("--state-path", type=Path, default=None, help="Default: <out-root>/pipeline_state.json")
    parser.add_argument("--max-episodes", type=int, default=200)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--episode-index", type=int, help="0-based index into sorted episode list")
    group.add_argument("--episode-name", type=str, help="Exact Nymeria Plus sequence id")
    parser.add_argument("--python", type=str, default=sys.executable, help="Interpreter for sub-steps")
    parser.add_argument(
        "--smplx-mode",
        default="quality-h100",
        choices=["fast", "balanced", "quality", "quality-h100", "ultra-h100"],
    )
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--device-torch", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--rgb-stride", type=int, default=4)
    parser.add_argument("--max-rgb-edge", type=int, default=512)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--keep-temp", action="store_true", help="Do not delete raw episode after success")
    parser.add_argument("--force", action="store_true", help="Reprocess even if episode_status.json is success")
    args = parser.parse_args()
    args.log_root = args.log_root or args.out_root / "logs"
    args.state_path = args.state_path or args.out_root / "pipeline_state.json"

    episodes = load_episode_list(args.url_json.resolve(), args.max_episodes)
    if args.episode_name:
        episode = args.episode_name
        if episode not in episodes:
            raise SystemExit(f"Episode {episode!r} not in first {args.max_episodes} episodes")
    else:
        idx = int(args.episode_index)
        if idx < 0 or idx >= len(episodes):
            raise SystemExit(f"--episode-index {idx} out of range [0, {len(episodes) - 1}]")
        episode = episodes[idx]

    args.temp_root.mkdir(parents=True, exist_ok=True)
    args.out_root.mkdir(parents=True, exist_ok=True)
    args.log_root.mkdir(parents=True, exist_ok=True)

    result = process_episode(
        episode=episode,
        url_json=args.url_json.resolve(),
        temp_root=args.temp_root.resolve(),
        out_root=args.out_root.resolve(),
        log_root=args.log_root.resolve(),
        state_path=args.state_path.resolve(),
        python=args.python,
        smplx_mode=args.smplx_mode,
        device=args.device,
        device_torch=args.device_torch,
        stride=int(args.stride),
        rgb_stride=int(args.rgb_stride),
        max_rgb_edge=int(args.max_rgb_edge),
        skip_download=bool(args.skip_download),
        keep_temp=bool(args.keep_temp),
        force=bool(args.force),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
