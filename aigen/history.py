"""生成履歴の保存・読み込みを行うモジュール。

生成した画像・動画をファイルとして保存すると同時に、プロンプトやシード値などの
メタデータを outputs/history.jsonl に追記する。「生成履歴」タブから一覧表示し、
過去の生成結果とその設定を後から見返せるようにする。
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from PIL import Image

from .utils import ensure_output_dir, timestamped_filename

HISTORY_FILENAME = "history.jsonl"


def save_image_output(
    output_dir: str,
    kind: str,
    kind_label: str,
    image: Image.Image,
    metadata: dict[str, Any],
) -> str:
    """生成画像をPNGとして保存し、履歴に記録する。保存先パスを返す。"""
    out_dir = ensure_output_dir(output_dir)
    path = out_dir / timestamped_filename(kind, "png")
    image.save(path)
    _append_history(out_dir, kind, kind_label, str(path), metadata)
    return str(path)


def record_existing_output(
    output_dir: str,
    kind: str,
    kind_label: str,
    output_path: str,
    metadata: dict[str, Any],
) -> None:
    """既にファイルとして保存済みの生成物（I2Vの動画など）を履歴に記録する。"""
    out_dir = ensure_output_dir(output_dir)
    _append_history(out_dir, kind, kind_label, output_path, metadata)


def _append_history(
    out_dir: Path,
    kind: str,
    kind_label: str,
    output_path: str,
    metadata: dict[str, Any],
) -> None:
    record = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "kind": kind,
        "kind_label": kind_label,
        "path": output_path,
        **metadata,
    }
    history_path = out_dir / HISTORY_FILENAME
    with open(history_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_history(output_dir: str, limit: int = 200) -> list[dict[str, Any]]:
    """履歴を新しい順に読み込む。壊れた行があってもスキップして継続する。"""
    history_path = Path(output_dir) / HISTORY_FILENAME
    if not history_path.exists():
        return []

    records: list[dict[str, Any]] = []
    with open(history_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    records.reverse()
    return records[:limit]
