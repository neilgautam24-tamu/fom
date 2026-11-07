#!/usr/bin/env python3
"""Download Nymeria Plus sequences from a Nymeria Plus URL JSON file."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from zipfile import ZipFile, is_zipfile

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

NYMERIA_PLUS_PREFIX = "NymeriaPlus_v1.0_"
CHUNK_SIZE = 8192
RETRY = 5
BACKOFF = 3

# JSON group key -> relative path inside the sequence folder (non-zip files).
UNGROUPED_DEST: dict[str, str] = {
    "LICENSE": "LICENSE",
    "metadata_json": "metadata.json",
    "recording_head_data_data_vrs": "recording_head/data/data.vrs",
    "recording_lwrist_data_data_vrs": "recording_lwrist/data/data.vrs",
    "recording_rwrist_data_data_vrs": "recording_rwrist/data/data.vrs",
    "recording_observer_data_data_vrs": "recording_observer/data/data.vrs",
}


class DlStatus(str, Enum):
    SUCCESS = "success"
    IGNORED = "ignored, file already downloaded"
    WARN_NOTFOUND = "warning, missing download link"
    ERR_SHA1SUM = "error, sha1sum mismatch"
    ERR_MEMORY = "error, insufficient disk space"
    ERR_NETWORK = "error, network"


@dataclass
class DlLink:
    filename: str
    sha1sum: str
    file_size_bytes: int
    download_url: str
    group_name: str
    status: DlStatus | None = None

    def __post_init__(self) -> None:
        if NYMERIA_PLUS_PREFIX not in self.filename:
            raise ValueError(
                f"Expected {NYMERIA_PLUS_PREFIX!r} prefix in {self.filename!r}"
            )
        self.filename = self.filename.replace(NYMERIA_PLUS_PREFIX, "")

    @property
    def seq_name(self) -> str:
        return "_".join(self.filename.split("_")[0:6])

    def destination_path(self, outdir: Path) -> Path | None:
        if self.group_name in UNGROUPED_DEST:
            return outdir / UNGROUPED_DEST[self.group_name]
        return None

    def get(self, outdir: Path, ignore_existing: bool = True) -> None:
        flag = outdir / "logs" / self.group_name
        if flag.is_file() and ignore_existing:
            self.status = DlStatus.IGNORED
            return

        if outdir.name != self.seq_name:
            raise ValueError(
                f"Output directory {outdir.name!r} != sequence {self.seq_name!r}"
            )
        outdir.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir) / self.filename
            print(f"Downloading {self.group_name} ({self.file_size_bytes / 1e9:.2f} GB)")

            session = requests.Session()
            retries = Retry(
                total=RETRY,
                backoff_factor=BACKOFF,
                status_forcelist=[429, 500, 502, 503, 504],
            )
            session.mount("https://", HTTPAdapter(max_retries=retries))

            with session.get(self.download_url, stream=True, timeout=60) as response:
                free_out = shutil.disk_usage(outdir).free
                free_tmp = shutil.disk_usage(tmpdir).free
                if free_out < self.file_size_bytes or free_tmp < self.file_size_bytes:
                    self.status = DlStatus.ERR_MEMORY
                    raise RuntimeError(
                        f"Insufficient disk space for {self.file_size_bytes} bytes"
                    )

                sha1 = hashlib.sha1()
                progress = (
                    tqdm(total=self.file_size_bytes, unit="iB", unit_scale=True)
                    if tqdm
                    else None
                )
                with open(tmp_path, "wb") as handle:
                    for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                        if not chunk:
                            continue
                        handle.write(chunk)
                        sha1.update(chunk)
                        if progress:
                            progress.update(len(chunk))
                if progress:
                    progress.close()

                computed = sha1.hexdigest()
                if computed != self.sha1sum:
                    self.status = DlStatus.ERR_SHA1SUM
                    raise RuntimeError(
                        f"sha1sum mismatch: got {computed}, expected {self.sha1sum}"
                    )
                response.raise_for_status()

            if is_zipfile(tmp_path):
                print(f"  extracting {self.group_name}")
                with ZipFile(tmp_path) as archive:
                    archive.extractall(outdir)
            else:
                dst = self.destination_path(outdir)
                if dst is None:
                    dst = outdir / self.filename
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(tmp_path), str(dst))

        self.status = DlStatus.SUCCESS
        flag.parent.mkdir(parents=True, exist_ok=True)
        flag.touch()
        print(f"  done: {self.group_name}")


def load_json(path: Path) -> dict:
    with open(path, "r") as handle:
        return json.load(handle)


def list_groups(data: dict, seq_name: str) -> list[str]:
    return sorted(data["sequences"][seq_name].keys())


def download_sequence(
    url_json: Path,
    out_root: Path,
    match_key: str,
    groups: list[str] | None,
    ignore_existing: bool,
    yes: bool,
) -> None:
    data = load_json(url_json)
    sequences = data.get("sequences", {})
    if not sequences:
        raise SystemExit("No sequences found in JSON")

    matched = [name for name in sequences if match_key in name]
    if not matched:
        raise SystemExit(f"No sequence matching {match_key!r}")

    if groups:
        selected_groups: set[str] | None = set(groups) | {"LICENSE", "metadata_json"}
    else:
        selected_groups = None

    total_bytes = 0
    plan: list[tuple[str, str, dict]] = []
    for seq_name in matched:
        for group_name, entry in sequences[seq_name].items():
            if selected_groups is not None and group_name not in selected_groups:
                continue
            plan.append((seq_name, group_name, entry))
            total_bytes += entry.get("file_size_bytes", 0)

    free_gb = shutil.disk_usage(out_root).free / (2**30)
    print("Download plan")
    print(f"  JSON:       {url_json}")
    print(f"  Output:     {out_root}")
    print(f"  Sequences:  {matched}")
    group_label = "all" if selected_groups is None else sorted(selected_groups)
    print(f"  Groups:     {group_label}")
    print(f"  Total (GB): {total_bytes / (2**30):.2f}")
    print(f"  Free (GB):  {free_gb:.2f}")

    if not yes:
        answer = input("Proceed? [y/n] ").strip().lower()
        if answer != "y":
            raise SystemExit("Aborted")

    out_root.mkdir(parents=True, exist_ok=True)
    logs: dict = {}

    for seq_name, group_name, entry in plan:
        logs.setdefault(seq_name, {})
        outdir = out_root / seq_name
        dl = DlLink(group_name=group_name, **entry)
        try:
            dl.get(outdir, ignore_existing=ignore_existing)
        except Exception as exc:
            print(f"  FAILED {group_name}: {exc}", file=sys.stderr)
            dl.status = DlStatus.ERR_NETWORK
        logs[seq_name][group_name] = (
            dl.status.value if dl.status else DlStatus.ERR_NETWORK.value
        )

        summary_path = out_root / "download_summary.json"
        with open(summary_path, "w") as handle:
            json.dump(logs, handle, indent=2)

    print(f"Summary written to {out_root / 'download_summary.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url-json",
        type=Path,
        default=Path(__file__).with_name("nymeria_plus_download_urls.json"),
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path(__file__).parent,
    )
    parser.add_argument(
        "--match",
        default="",
        help="Substring to match sequence name (default: only sequence in JSON)",
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        help="Data groups to download (default: all). Always includes LICENSE and metadata_json.",
    )
    parser.add_argument(
        "--list-groups",
        action="store_true",
        help="List available groups for matched sequence and exit",
    )
    parser.add_argument("--no-ignore-existing", action="store_true")
    parser.add_argument("-y", "--yes", action="store_true", help="Skip confirmation")
    args = parser.parse_args()

    data = load_json(args.url_json)
    sequences = data["sequences"]
    match_key = args.match or next(iter(sequences))

    if args.list_groups:
        for seq_name in sequences:
            if match_key in seq_name:
                print(seq_name)
                for group in list_groups(data, seq_name):
                    size_gb = sequences[seq_name][group]["file_size_bytes"] / 1e9
                    print(f"  {group}: {size_gb:.2f} GB")
        return

    download_sequence(
        url_json=args.url_json,
        out_root=args.outdir,
        match_key=match_key,
        groups=args.groups,
        ignore_existing=not args.no_ignore_existing,
        yes=args.yes,
    )


if __name__ == "__main__":
    main()
