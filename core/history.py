"""
history.py - 本地历史记录管理（JSON 持久化，支持按用户隔离）
"""

import json
import threading
from datetime import datetime
from pathlib import Path

_lock = threading.Lock()


def _history_file(data_dir: Path) -> Path:
    return data_dir / "history.json"


def load_history(data_dir: Path, user_id: str = "") -> list:
    """
    加载历史记录列表，按时间倒序返回。
    若提供 user_id，只返回该用户的记录。
    """
    hf = _history_file(data_dir)
    if not hf.exists():
        return []
    with _lock:
        try:
            with open(hf, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, Exception):
            return []

    if user_id:
        data = [r for r in data if r.get("user_id", "") == user_id]

    return sorted(data, key=lambda x: x.get("timestamp", ""), reverse=True)


def save_job(data_dir: Path, job_info: dict):
    """追加一条历史记录。"""
    hf = _history_file(data_dir)
    with _lock:
        existing = []
        if hf.exists():
            try:
                with open(hf, "r", encoding="utf-8") as f:
                    existing = json.load(f)
            except Exception:
                existing = []
        existing.append(job_info)
        with open(hf, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)


def delete_job(data_dir: Path, job_id: str, user_id: str = "") -> bool:
    """
    删除一条历史记录。
    若提供 user_id，只允许删除属于该用户的记录。
    """
    hf = _history_file(data_dir)
    with _lock:
        if not hf.exists():
            return False
        try:
            with open(hf, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except Exception:
            return False

        new_list = []
        deleted = False
        for j in existing:
            if j.get("id") == job_id:
                # 若指定了 user_id，检查归属
                if user_id and j.get("user_id", "") != user_id:
                    new_list.append(j)  # 不允许删除别人的记录
                else:
                    deleted = True      # 删除
            else:
                new_list.append(j)

        if not deleted:
            return False
        with open(hf, "w", encoding="utf-8") as f:
            json.dump(new_list, f, ensure_ascii=False, indent=2)
        return True


def remove_job_ids(data_dir: Path, job_ids: set):
    """批量删除历史记录（用于自动清理过期任务）。"""
    if not job_ids:
        return
    hf = _history_file(data_dir)
    with _lock:
        if not hf.exists():
            return
        try:
            with open(hf, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except Exception:
            return
        new_list = [j for j in existing if j.get("id") not in job_ids]
        with open(hf, "w", encoding="utf-8") as f:
            json.dump(new_list, f, ensure_ascii=False, indent=2)


def make_job_record(job_id: str, original_filename: str, entity_count: int,
                    categories: dict, desensitized_filename: str,
                    mapping_filename: str, user_id: str = "") -> dict:
    return {
        "id": job_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "user_id": user_id,
        "original_filename": original_filename,
        "entity_count": entity_count,
        "categories": categories,
        "desensitized_filename": desensitized_filename,
        "mapping_filename": mapping_filename,
    }
