#!/usr/bin/env python3
"""Download one Nymeria Plus episode and extract multi-object visible-goal bundles.

Same end-to-end flow as :mod:`fom.preprocessing.nymeria.pipeline.process_episode`, but writes
``multi_object_visible_goal_bundles`` under ``processed_data_multiobject/``.

Designed for SLURM array jobs on episode indices 200-250 (disjoint from the
first 200 single-object episodes).

Usage
-----
    python -m fom.preprocessing.nymeria.pipeline.process_multi_object_episode --episode-index 200
    python -m fom.preprocessing.nymeria.pipeline.process_multi_object_episode --episode-name 20230724_s0_tyler_ayers_act0_bhdzn3
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path

from fom.preprocessing.nymeria.pipeline.process_episode import (
    CONTACT_OUTPUT_NAMES,
    cleanup_temp,
    convert_smpl_to_smplx,
    download_episode,
    extract_contacts,
    preserve_original_body_motion,
    restore_body_to_temp,
    run_cmd,
    setup_episode_logger,
    temp_episode_ready,
    update_pipeline_state,
    utc_now,
)

FINAL_OUTPUT_NAMES = [
    "multi_object_visible_goal_bundles",
    "multi_object_visible_goal_sequences.json",
    "hand_object_contacts.csv",
    "hand_object_contact_intervals.csv",
    "contact_summary.json",
    "contact_episode_counts.json",
    "contact_episode_counts_by_object.csv",
]

MULTI_OBJECT_SEQUENCE_JSON = "multi_object_visible_goal_sequences.json"


def load_episode_list(url_json: Path) -> list[str]:
    data = json.loads(url_json.read_text())
    return sorted(data["sequences"].keys())


def detect_episode_gaps(out_dir: Path, required_smplx_mode: str) -> dict[str, bool]:
    bundles_root = out_dir / "multi_object_visible_goal_bundles"
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
            p
            for p in bundles_root.iterdir()
            if p.is_dir() and p.name.startswith("mvgoal_seq_")
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


def seed_work_dir_from_output_multi(
    out_dir: Path,
    work_dir: Path,
    plan: dict[str, object],
    logger: logging.Logger,
) -> None:
    work_dir.mkdir(parents=True, exist_ok=True)
    if not plan["reextract_contacts"]:
        for name in CONTACT_OUTPUT_NAMES:
            src = out_dir / name
            if src.is_file():
                shutil.copy2(src, work_dir / name)
                logger.info("Reused contact artifact -> %s", work_dir / name)
    if plan["skip_extract_goals"]:
        src = out_dir / MULTI_OBJECT_SEQUENCE_JSON
        if src.is_file():
            shutil.copy2(src, work_dir / MULTI_OBJECT_SEQUENCE_JSON)
            logger.info("Reused sequence artifact -> %s", work_dir / MULTI_OBJECT_SEQUENCE_JSON)


def extract_multi_object_goals(
    *,
    python: str,
    episode: str,
    seq_dir: Path,
    work_dir: Path,
    url_json: Path,
    stride: int,
    rgb_stride: int,
    max_rgb_edge: int,
    min_objects: int,
    max_objects: int,
    logger: logging.Logger,
) -> None:
    smplx_npz = seq_dir / "body" / "xdata_smplx_neutral.npz"
    cmd = [
        python,
        "-m",
        "fom.preprocessing.nymeria.goals.extract_multi_object",
        "--url-json",
        str(url_json),
        "--episode-name",
        episode,
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
        "--min-objects",
        str(min_objects),
        "--max-objects",
        str(max_objects),
    ]
    run_cmd(cmd, logger=logger)


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

    manifest = out_dir / "multi_object_visible_goal_bundles" / "manifest.json"
    num_sequences = 0
    if manifest.is_file():
        num_sequences = int(json.loads(manifest.read_text()).get("num_sequences", 0))

    status = {
        "episode": episode,
        "status": "success",
        "finished_at": utc_now(),
        "num_multi_object_sequences": num_sequences,
        **body_paths,
        **extra_status,
    }
    (out_dir / "episode_status.json").write_text(json.dumps(status, indent=2))
    logger.info("Wrote episode_status.json (%d multi-object sequences)", num_sequences)


def process_episode(
    *,
    episode: str,
    episode_index: int,
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
    min_objects: int,
    max_objects: int,
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
    logger.info("Episode index: %d", episode_index)
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
        details={
            "episode_index": episode_index,
            "started_at": utc_now(),
            "plan": {k: v for k, v in plan.items() if k != "gaps"},
        },
    )

    try:
        work_dir.mkdir(parents=True, exist_ok=True)
        seed_work_dir_from_output_multi(out_dir, work_dir, plan, logger)

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
            extract_multi_object_goals(
                python=python,
                episode=episode,
                seq_dir=seq_dir,
                work_dir=work_dir,
                url_json=url_json,
                stride=stride,
                rgb_stride=rgb_stride,
                max_rgb_edge=max_rgb_edge,
                min_objects=min_objects,
                max_objects=max_objects,
                logger=logger,
            )
        elapsed_s = time.time() - t0
        finalize_outputs(
            work_dir=work_dir,
            out_dir=out_dir,
            seq_dir=seq_dir,
            episode=episode,
            logger=logger,
            extra_status={
                "episode_index": episode_index,
                "elapsed_s": round(elapsed_s, 1),
                "smplx_npz": str(smplx_npz),
                "smplx_mode": smplx_mode,
                "stride": stride,
                "rgb_stride": rgb_stride,
                "min_objects": min_objects,
                "max_objects": max_objects,
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
            details={
                "episode_index": episode_index,
                "elapsed_s": round(elapsed_s, 1),
                "out_dir": str(out_dir),
            },
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
            details={
                "episode_index": episode_index,
                "error": str(exc),
                "elapsed_s": round(elapsed_s, 1),
            },
        )
        fail_status = {
            "episode": episode,
            "episode_index": episode_index,
            "status": "failed",
            "finished_at": utc_now(),
            "error": str(exc),
            "elapsed_s": round(elapsed_s, 1),
        }
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "episode_status.json").write_text(json.dumps(fail_status, indent=2))
        raise


def resolve_episode(
    episodes: list[str],
    *,
    episode_index: int | None,
    episode_name: str | None,
    index_min: int,
    index_max: int,
) -> tuple[int, str]:
    if episode_name:
        if episode_name not in episodes:
            raise SystemExit(f"Episode {episode_name!r} not found in URL JSON")
        idx = episodes.index(episode_name)
    else:
        idx = int(episode_index)
    if idx < index_min or idx > index_max:
        raise SystemExit(
            f"Episode index {idx} outside allowed range [{index_min}, {index_max}]"
        )
    return idx, episodes[idx]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url-json", type=Path, required=True, help="Nymeria-Plus signed download URL json")
    parser.add_argument("--temp-root", type=Path, required=True, help="Scratch dir for raw downloads")
    parser.add_argument("--out-root", type=Path, required=True, help="Processed episode output dir")
    parser.add_argument("--log-root", type=Path, default=None, help="Default: <out-root>/logs")
    parser.add_argument("--state-path", type=Path, default=None, help="Default: <out-root>/pipeline_state.json")
    parser.add_argument("--index-min", type=int, default=200)
    parser.add_argument("--index-max", type=int, default=250)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--episode-index", type=int, help="0-based index into full episode list")
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
    parser.add_argument("--min-objects", type=int, default=2)
    parser.add_argument("--max-objects", type=int, default=4)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--keep-temp", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.log_root = args.log_root or args.out_root / "logs"
    args.state_path = args.state_path or args.out_root / "pipeline_state.json"

    episodes = load_episode_list(args.url_json.resolve())
    episode_index, episode = resolve_episode(
        episodes,
        episode_index=args.episode_index,
        episode_name=args.episode_name,
        index_min=int(args.index_min),
        index_max=int(args.index_max),
    )

    single_object_episodes = set(episodes[: int(args.index_min)])
    if episode in single_object_episodes:
        raise SystemExit(
            f"Episode {episode!r} belongs to the single-object batch (indices 0..{args.index_min - 1})"
        )

    args.temp_root.mkdir(parents=True, exist_ok=True)
    args.out_root.mkdir(parents=True, exist_ok=True)
    args.log_root.mkdir(parents=True, exist_ok=True)

    result = process_episode(
        episode=episode,
        episode_index=episode_index,
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
        min_objects=int(args.min_objects),
        max_objects=int(args.max_objects),
        skip_download=bool(args.skip_download),
        keep_temp=bool(args.keep_temp),
        force=bool(args.force),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
