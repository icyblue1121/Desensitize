"""
feedback.py - 用户纠错反馈的存储、检索与训练数据导出

职责：
1. 存储用户标记的「漏识别」和「误识别」案例（feedback.json）
2. 为 detector.py 提供 few-shot 示例注入 —— 实现 Prompt 层面的持续学习
3. 导出 LoRA 微调用的 instruction JSONL，供 Unsloth / LLaMA-Factory 训练
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional


FEEDBACK_FILE      = "feedback.json"
AUTO_SAMPLES_FILE  = "auto_training_samples.json"   # 每次任务自动保存的训练样本
TRAINING_FILE      = "training_data.jsonl"

# 注入 prompt 时最多使用的示例条数（避免 prompt 过长）
MAX_FEW_SHOT = 20


# ─── 基础读写 ──────────────────────────────────────────────────────────────────

def _load(data_dir: Path) -> list:
    p = data_dir / FEEDBACK_FILE
    if not p.exists():
        return []
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def _save(data_dir: Path, records: list):
    p = data_dir / FEEDBACK_FILE
    with open(p, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


# ─── 写入反馈 ──────────────────────────────────────────────────────────────────

def save_feedback(data_dir: Path,
                  job_id: str,
                  missed: list[dict],
                  false_positive: list[dict]) -> str:
    """
    保存一次反馈记录。

    missed        : 漏识别项，格式 [{"original": "...", "category": "..."}]
    false_positive: 误识别项，格式 [{"original": "..."}]

    返回新建记录的 feedback_id。
    """
    records = _load(data_dir)
    fid = str(uuid.uuid4())[:8]
    record = {
        "feedback_id": fid,
        "job_id": job_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "missed": missed,
        "false_positive": false_positive,
    }
    records.append(record)
    _save(data_dir, records)
    return fid


# ─── Few-shot 示例生成（用于 Prompt 注入）─────────────────────────────────────

def build_few_shot_section(data_dir: Path, max_items: int = MAX_FEW_SHOT) -> str:
    """
    从历史反馈中提取「漏识别」案例，生成可直接插入 DETECTION_PROMPT 的文本块。
    按时间倒序取最近 max_items 条，每条简要说明原文中出现了什么、属于哪类。
    """
    records = _load(data_dir)
    if not records:
        return ""

    # 收集所有漏识别示例，去重（以 original 为 key）
    seen: set[str] = set()
    examples: list[dict] = []
    for rec in reversed(records):           # 最新的优先
        for item in rec.get("missed", []):
            orig = item.get("original", "").strip()
            cat  = item.get("category", "").strip()
            if orig and orig not in seen:
                seen.add(orig)
                examples.append({"original": orig, "category": cat})
                if len(examples) >= max_items:
                    break
        if len(examples) >= max_items:
            break

    if not examples:
        return ""

    lines = ["【历史纠错示例 - 这些类型在过往任务中曾被漏识别，请务必识别同类内容】"]
    for ex in examples:
        line = f"- [{ex['category']}] 原文内容「{ex['original']}」（此类内容须识别并脱敏）"
        lines.append(line)

    return "\n".join(lines)


# ─── 误识别过滤器 ──────────────────────────────────────────────────────────────

def get_false_positive_set(data_dir: Path) -> set[str]:
    """
    返回所有被用户标记为「误识别」的原文字符串集合。
    detector 可在最终过滤时排除这些项。
    """
    records = _load(data_dir)
    fp_set: set[str] = set()
    for rec in records:
        for item in rec.get("false_positive", []):
            orig = item.get("original", "").strip()
            if orig:
                fp_set.add(orig)
    return fp_set


# ─── 统计信息 ──────────────────────────────────────────────────────────────────

def get_stats(data_dir: Path) -> dict:
    """返回反馈库的统计摘要。"""
    records = _load(data_dir)
    total_missed = sum(len(r.get("missed", [])) for r in records)
    total_fp     = sum(len(r.get("false_positive", [])) for r in records)

    category_counts: dict[str, int] = {}
    for rec in records:
        for item in rec.get("missed", []):
            cat = item.get("category", "未知")
            category_counts[cat] = category_counts.get(cat, 0) + 1

    return {
        "total_feedback_records": len(records),
        "total_missed_items": total_missed,
        "total_false_positives": total_fp,
        "missed_by_category": category_counts,
    }


# ─── LoRA 训练数据导出 ─────────────────────────────────────────────────────────

_INSTRUCTION_TEMPLATE = (
    "你是合同文档脱敏助手。分析以下合同文本，识别敏感实体并生成虚假替代内容。"
    "以JSON格式返回：{{\"entities\": [{{\"original\": \"原始文本\", "
    "\"category\": \"类别\", \"replacement\": \"替代内容\"}}]}}"
)

# 自动保存样本的最大文本片段长度（字符）
_SNIPPET_LEN = 1500


def save_auto_sample(data_dir: Path, job_id: str, doc_text: str, entities: list[dict]):
    """
    每次脱敏任务完成后自动调用，将模型识别结果保存为训练样本。

    doc_text : 原始文档全文（取前 _SNIPPET_LEN 字作为训练输入）
    entities : detect_entities() 返回的完整实体列表
    """
    if not entities or not doc_text.strip():
        return

    p = data_dir / AUTO_SAMPLES_FILE
    try:
        with open(p, "r", encoding="utf-8") as f:
            samples = json.load(f)
    except Exception:
        samples = []

    # 避免同一 job 重复写入
    existing_ids = {s.get("job_id") for s in samples}
    if job_id in existing_ids:
        return

    samples.append({
        "job_id":    job_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "snippet":   doc_text[:_SNIPPET_LEN],
        "entities":  [
            {"original": e["original"],
             "category": e.get("category", "未知"),
             "replacement": e["replacement"]}
            for e in entities
            if e.get("original") and e.get("replacement")
        ],
    })

    with open(p, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)


def _load_auto_samples(data_dir: Path) -> list:
    p = data_dir / AUTO_SAMPLES_FILE
    if not p.exists():
        return []
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def export_training_data(data_dir: Path) -> tuple[int, Path]:
    """
    将训练样本导出为 LoRA 微调用的 instruction JSONL。

    来源优先级（合并后去重）：
    1. 自动样本（auto_training_samples.json）—— 每次脱敏任务自动保存，含完整 replacement
    2. 用户手动纠错（feedback.json）—— 仅含用户填写了 replacement 的条目

    格式（alpaca-style，兼容 Unsloth / LLaMA-Factory）：
      {"instruction": "...", "input": "<文档片段>", "output": "{\"entities\":[...]}"}

    返回 (导出条数, 输出文件路径)。
    """
    output_path = data_dir / TRAINING_FILE
    count = 0

    with open(output_path, "w", encoding="utf-8") as f:

        # ── 1. 自动样本（质量最高，全量 replacement）──────────────────────────
        for sample in _load_auto_samples(data_dir):
            entities_out = sample.get("entities", [])
            snippet = sample.get("snippet", "")
            if not entities_out or not snippet:
                continue
            row = {
                "instruction": _INSTRUCTION_TEMPLATE,
                "input":  snippet,
                "output": json.dumps({"entities": entities_out}, ensure_ascii=False),
                "_source": "auto",
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1

        # ── 2. 用户纠错（仅含有 replacement 的条目）──────────────────────────
        for rec in _load(data_dir):
            missed = rec.get("missed", [])
            snippet = rec.get("doc_snippet", "")
            if not missed or not snippet:
                continue
            entities_out = [
                {"original": m["original"],
                 "category": m.get("category", "未知"),
                 "replacement": m["replacement"]}
                for m in missed
                if m.get("original") and m.get("replacement", "").strip()
            ]
            if not entities_out:
                continue
            row = {
                "instruction": _INSTRUCTION_TEMPLATE,
                "input":  snippet,
                "output": json.dumps({"entities": entities_out}, ensure_ascii=False),
                "_source": "correction",
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1

    return count, output_path


def load_all(data_dir: Path) -> list:
    """返回完整反馈记录列表（供管理接口使用）。"""
    return _load(data_dir)
