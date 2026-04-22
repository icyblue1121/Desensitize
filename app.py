"""
app.py - 合同文档脱敏系统（局域网多用户版）
基于 Tornado + 本地 AI 服务（兼容 Ollama / LM Studio / OpenAI 兼容接口）
"""

import asyncio
import concurrent.futures
import json
import os
import re
import shutil
import sys
import threading
import time
import urllib.parse
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import tornado.ioloop
import tornado.web
import tornado.gen

from core import detector, doc_handler, history as hist_module, feedback as fb_module
from core.queue_manager import (
    enqueue, get_position,
    cancel_job, reorder_job,
    is_cancelled, clear_cancelled,
)

# ─── 路径配置 ─────────────────────────────────────────────────────────────────
APP_DIR  = Path(__file__).parent.resolve()
DATA_DIR = APP_DIR / "data"
JOBS_DIR = DATA_DIR / "jobs"
CONFIG_FILE = APP_DIR / "config.json"
STATIC_DIR  = APP_DIR / "static"

DATA_DIR.mkdir(exist_ok=True)
JOBS_DIR.mkdir(exist_ok=True)

# 线程池（doc_handler 使用，不限制 AI 服务并发，队列已保证串行）
executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

# 内存任务状态表
jobs: dict = {}

ALLOWED_SUFFIXES = {".docx", ".xlsx", ".xls", ".csv", ".pdf"}

# ─── 配置读写 ─────────────────────────────────────────────────────────────────

DEFAULT_MODELS = ["qwen3.5:35b"]


def _normalize_models(cfg: dict) -> list:
    """从 cfg 读取 models，兼容旧版 model/model_2/dual_model 字段。"""
    models = cfg.get("models")
    if isinstance(models, list):
        cleaned = [str(m).strip() for m in models if str(m).strip()]
        if cleaned:
            return cleaned
    legacy = [str(cfg.get("model", "")).strip()]
    if cfg.get("dual_model") and str(cfg.get("model_2", "")).strip():
        legacy.append(str(cfg["model_2"]).strip())
    legacy = [m for m in legacy if m]
    return legacy or list(DEFAULT_MODELS)


def load_config() -> dict:
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                cfg = json.load(f)
                cfg["models"] = _normalize_models(cfg)
                return cfg
        except Exception:
            pass
    return {
        "api_url": "http://localhost:11434",
        "models": list(DEFAULT_MODELS),
        "cleanup_days": 7,
    }


def save_config(cfg: dict):
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


# ─── 自动清理过期任务文件 ─────────────────────────────────────────────────────

def cleanup_old_jobs(days: int = 7):
    """删除超过 days 天的任务文件及对应历史记录。"""
    cutoff = datetime.now() - timedelta(days=days)
    expired_ids: set = set()

    if JOBS_DIR.exists():
        for job_dir in JOBS_DIR.iterdir():
            if not job_dir.is_dir():
                continue
            try:
                mtime = datetime.fromtimestamp(job_dir.stat().st_mtime)
                if mtime < cutoff:
                    expired_ids.add(job_dir.name)
                    shutil.rmtree(job_dir, ignore_errors=True)
            except Exception:
                pass

    if expired_ids:
        hist_module.remove_job_ids(DATA_DIR, expired_ids)


def _schedule_cleanup():
    """启动时立即执行一次清理，之后每小时执行一次。"""
    cfg = load_config()
    days = cfg.get("cleanup_days", 7)
    cleanup_old_jobs(days)
    while True:
        time.sleep(3600)
        cleanup_old_jobs(load_config().get("cleanup_days", 7))


threading.Thread(target=_schedule_cleanup, daemon=True).start()


# ─── 任务状态持久化（让重启后的任务状态可以从磁盘恢复）────────────────────────

def _save_state(job_id: str):
    """将当前内存中的任务状态写入 JOBS_DIR/<job_id>/state.json。"""
    job = jobs.get(job_id)
    if not job:
        return
    try:
        state_path = JOBS_DIR / job_id / "state.json"
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(job, f, ensure_ascii=False)
    except Exception:
        pass


def _load_state(job_id: str) -> "dict | None":
    """从磁盘读取任务状态。若文件不存在则返回 None。"""
    state_path = JOBS_DIR / job_id / "state.json"
    if not state_path.exists():
        return None
    try:
        with open(state_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


# ─── 核心处理函数（在线程池中运行）────────────────────────────────────────────

def _process_desensitize(job_id: str, upload_path: Path,
                         original_filename: str, custom_instructions: str,
                         username: str):
    try:
        jobs[job_id]["progress"] = "正在读取文档内容..."
        jobs[job_id]["progress_pct"] = 5
        jobs[job_id]["started_at"] = time.time()

        text = doc_handler.extract_text(upload_path)
        if not text.strip():
            raise ValueError("文档内容为空或无法提取文字（PDF 请确认为可选中文字的格式）")

        # ── 检查点 1：读取文档后，调用 AI 前 ──
        if is_cancelled(job_id):
            clear_cancelled(job_id)
            jobs[job_id].update({"status": "cancelled", "progress": "已取消", "progress_pct": 0})
            _save_state(job_id)
            return

        cfg = load_config()
        models = _normalize_models(cfg)
        total_models = len(models)

        jobs[job_id]["progress"] = (
            f"正在调用 AI 识别敏感信息（{total_models} 个模型依次运行）..."
            if total_models > 1 else
            "正在调用 AI 识别敏感信息（本地模型，请耐心等待）..."
        )
        jobs[job_id]["progress_pct"] = 20

        # ── 从反馈库加载 few-shot 示例 + 误识别过滤集 ────────────────────────
        few_shot   = fb_module.build_few_shot_section(DATA_DIR)
        fp_set     = fb_module.get_false_positive_set(DATA_DIR)

        def _on_pass(idx, total, model_name):
            jobs[job_id]["progress"] = f"正在运行模型 {idx + 1}/{total}：{model_name}..."
            jobs[job_id]["progress_pct"] = 20 + int(60 * idx / max(total, 1))

        entities = detector.detect_entities(
            text,
            custom_instructions,
            models=models,
            ollama_url=cfg.get("api_url", "http://localhost:11434"),
            job_id=job_id,
            few_shot_section=few_shot,
            false_positive_set=fp_set if fp_set else None,
            progress_cb=_on_pass,
        )

        if total_models > 1:
            jobs[job_id]["progress"] = f"{total_models} 个模型识别完成，共 {len(entities)} 个实体，正在生成文档..."

        jobs[job_id]["progress_pct"] = 80

        # ── 检查点 2：AI 调用返回后 ──
        if is_cancelled(job_id):
            clear_cancelled(job_id)
            jobs[job_id].update({"status": "cancelled", "progress": "已取消", "progress_pct": 0})
            _save_state(job_id)
            return

        if not entities:
            jobs[job_id].update({
                "status": "completed",
                "progress_pct": 100,
                "entity_count": 0,
                "categories": {},
                "desensitized_filename": None,
                "mapping_filename": None,
                "warning": "未识别到需要脱敏的敏感信息，请确认文档内容或调整自定义指令。"
            })
            _save_state(job_id)   # 持久化"未识别到实体"的完成状态
            return

        jobs[job_id]["progress"] = f"识别到 {len(entities)} 个敏感实体，正在生成脱敏文档..."

        mapping = {e["original"]: e["replacement"] for e in entities}

        stem = Path(original_filename).stem
        out_suffix = doc_handler.get_output_suffix(upload_path.suffix)

        # 对文件名同样应用脱敏映射（按长度降序避免短串先替换）
        desensitized_stem = stem
        for orig, repl in sorted(mapping.items(), key=lambda x: len(x[0]), reverse=True):
            desensitized_stem = desensitized_stem.replace(orig, repl)
        # 清除文件名中的非法字符
        desensitized_stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', desensitized_stem)

        desensitized_filename = f"脱敏_{desensitized_stem}{out_suffix}"
        mapping_filename     = f"映射表_{desensitized_stem}.xlsx"

        job_dir = JOBS_DIR / job_id
        job_dir.mkdir(exist_ok=True)

        output_path  = job_dir / desensitized_filename
        mapping_path = job_dir / mapping_filename

        doc_handler.apply_desensitization(upload_path, output_path, mapping, original_filename)

        # PDF → docx 路径修正
        if upload_path.suffix.lower() == ".pdf" and not output_path.exists():
            output_path = output_path.with_suffix(".docx")
            desensitized_filename = output_path.name

        jobs[job_id]["progress_pct"] = 92
        doc_handler.create_mapping_excel(entities, mapping_path)

        categories: dict = {}
        for e in entities:
            cat = e.get("category", "未知")
            categories[cat] = categories.get(cat, 0) + 1

        record = hist_module.make_job_record(
            job_id, original_filename, len(entities),
            categories, desensitized_filename, mapping_filename,
            user_id=username
        )
        hist_module.save_job(DATA_DIR, record)

        # 自动保存训练样本（用于 LoRA 微调导出）
        fb_module.save_auto_sample(DATA_DIR, job_id, text, entities)

        jobs[job_id].update({
            "status": "completed",
            "progress_pct": 100,
            "entity_count": len(entities),
            "categories": categories,
            "desensitized_filename": desensitized_filename,
            "mapping_filename": mapping_filename,
        })
        _save_state(job_id)   # 持久化"已完成"状态

    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e)
        _save_state(job_id)   # 持久化"出错"状态


def _process_restore(job_id: str, doc_path: Path, mapping_path: Path,
                     original_filename: str):
    try:
        jobs[job_id]["progress"] = "正在读取映射表..."
        jobs[job_id]["progress_pct"] = 10
        jobs[job_id]["started_at"] = time.time()

        reverse_mapping = doc_handler.read_mapping_excel(mapping_path)

        if not reverse_mapping:
            raise ValueError("映射表为空或格式不正确，请上传由本系统生成的映射表 Excel 文件")

        # ── 检查点：读取映射表后 ──
        if is_cancelled(job_id):
            clear_cancelled(job_id)
            jobs[job_id].update({"status": "cancelled", "progress": "已取消", "progress_pct": 0})
            _save_state(job_id)
            return

        jobs[job_id]["progress"] = f"正在还原文档（共 {len(reverse_mapping)} 个替换项）..."
        jobs[job_id]["progress_pct"] = 40

        stem = Path(original_filename).stem
        out_suffix = doc_path.suffix.lower()
        restored_filename = f"还原_{stem}{out_suffix}"

        job_dir = JOBS_DIR / job_id
        job_dir.mkdir(exist_ok=True)
        output_path = job_dir / restored_filename

        doc_handler.restore_document(doc_path, output_path, reverse_mapping)

        jobs[job_id].update({
            "status": "completed",
            "progress_pct": 100,
            "restored_filename": restored_filename,
            "replacement_count": len(reverse_mapping),
        })
        _save_state(job_id)

    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e)
        _save_state(job_id)


# ─── 工具函数 ─────────────────────────────────────────────────────────────────

def get_username(handler) -> str:
    """从请求头中获取用户名，匿名用户返回 'anonymous'。
    前端使用 encodeURIComponent 编码，此处解码以支持中文姓名。"""
    raw = handler.request.headers.get("X-Username", "").strip()
    if not raw:
        return "anonymous"
    try:
        from urllib.parse import unquote
        return unquote(raw)
    except Exception:
        return raw


# ─── Tornado Handlers ─────────────────────────────────────────────────────────

class BaseHandler(tornado.web.RequestHandler):
    def set_default_headers(self):
        self.set_header("Content-Type", "application/json; charset=UTF-8")
        self.set_header("Cache-Control", "no-cache")

    def write_error(self, status_code: int, **kwargs):
        """覆盖默认 HTML 错误页，始终返回 JSON。
        防止 Tornado 异常时返回 HTML 导致前端 resp.json() 解析失败，
        出现误导性的"网络错误"提示。"""
        msg = self._reason
        if "exc_info" in kwargs:
            exc_value = kwargs["exc_info"][1]
            if exc_value:
                msg = str(exc_value)
        self.set_header("Content-Type", "application/json; charset=UTF-8")
        self.finish(json.dumps({"error": msg}, ensure_ascii=False))

    def write_json(self, data: dict, status: int = 200):
        self.set_status(status)
        self.write(json.dumps(data, ensure_ascii=False))

    def write_error_json(self, msg: str, status: int = 400):
        self.set_status(status)
        self.write(json.dumps({"error": msg}, ensure_ascii=False))


class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.set_header("Content-Type", "text/html; charset=UTF-8")
        with open(STATIC_DIR / "index.html", "r", encoding="utf-8") as f:
            self.write(f.read())


class SettingsHandler(BaseHandler):
    def get(self):
        cfg = load_config()
        api_url = cfg.get("api_url", "http://localhost:11434")
        models  = _normalize_models(cfg)

        statuses = [detector.check_ollama(api_url, m) for m in models]
        ready = all(s["ok"] for s in statuses) if statuses else False
        if len(models) == 1:
            msg = statuses[0]["message"]
        else:
            msg = "；".join(f"{models[i]}：{statuses[i]['message']}" for i in range(len(models)))

        self.write_json({
            "ready": ready,
            "message": msg,
            "api_url": api_url,
            "models": models,
            "model": models[0] if models else "",
            "cleanup_days": cfg.get("cleanup_days", 7),
        })

    def post(self):
        try:
            body = json.loads(self.request.body or b"{}")
        except Exception:
            self.write_error_json("无效请求体")
            return
        cfg = load_config()
        if "api_url" in body:
            cfg["api_url"] = str(body["api_url"]).strip() or cfg.get("api_url", "http://localhost:11434")
        if "models" in body:
            raw = body["models"]
            if not isinstance(raw, list):
                self.write_error_json("models 必须为数组")
                return
            cleaned = [str(m).strip() for m in raw if str(m).strip()]
            if not cleaned:
                self.write_error_json("至少需要一个模型")
                return
            cfg["models"] = cleaned
        if "cleanup_days" in body:
            try:
                cfg["cleanup_days"] = max(1, int(body["cleanup_days"]))
            except Exception:
                pass
        for legacy in ("model", "model_2", "dual_model"):
            cfg.pop(legacy, None)
        save_config(cfg)
        self.write_json({"ok": True, "models": cfg["models"], "api_url": cfg.get("api_url", "")})


class SecureLlmHandler(BaseHandler):
    """
    POST /api/secure_llm
    连贯链路：本地识别并脱敏 -> 外部 AI 处理 -> 自动还原结果。
    """
    def post(self):
        try:
            body = json.loads(self.request.body or b"{}")
        except Exception:
            self.write_error_json("无效请求体")
            return

        text = str(body.get("text", "")).strip()
        task = str(body.get("task", "")).strip()
        custom_instructions = str(body.get("custom_instructions", "")).strip()
        passthrough_if_empty = bool(body.get("passthrough_if_empty", True))
        return_tokenized = bool(body.get("return_tokenized", False))

        if not text:
            self.write_error_json("缺少 text")
            return
        if not task:
            self.write_error_json("缺少 task")
            return

        cfg = load_config()
        local_models = _normalize_models(cfg)
        api_url = str(body.get("external_api_url", "")).strip() or cfg.get("api_url", "http://localhost:11434")
        model = str(body.get("external_model", "")).strip() or (local_models[0] if local_models else "qwen3.5:35b")
        system_prompt = str(body.get("external_system_prompt", "")).strip()
        api_key = str(body.get("external_api_key", "")).strip()

        few_shot = fb_module.build_few_shot_section(DATA_DIR)
        fp_set = fb_module.get_false_positive_set(DATA_DIR)

        entities = detector.detect_entities(
            text,
            custom_instructions,
            models=local_models,
            ollama_url=cfg.get("api_url", "http://localhost:11434"),
            few_shot_section=few_shot,
            false_positive_set=fp_set if fp_set else None,
        )

        mapping = {e["original"]: e["replacement"] for e in entities}
        tokenized_text = doc_handler.tokenize_text_with_mapping(text, mapping)

        # 无敏感项时可直接透传原文，避免不必要失败
        if not mapping and passthrough_if_empty:
            tokenized_text = text

        ai_output = detector.call_openai_compatible(
            task_prompt=task,
            text=tokenized_text,
            model=model,
            api_url=api_url,
            system_prompt=system_prompt,
            api_key=api_key,
        )

        reverse_mapping = {v: k for k, v in mapping.items()}
        restored_output = doc_handler.restore_text_with_mapping(ai_output, reverse_mapping)

        resp = {
            "result": restored_output,
            "entity_count": len(entities),
            "mapping_count": len(mapping),
            "used_model": model,
            "used_api_url": api_url,
        }
        if return_tokenized:
            resp["tokenized_text"] = tokenized_text
            resp["tokenized_result"] = ai_output
        self.write_json(resp)


class DesensitizeTextHandler(BaseHandler):
    """POST /api/desensitize_text — 对文本直接脱敏，返回脱敏后文本及实体映射。"""
    def post(self):
        try:
            body = json.loads(self.request.body or b"{}")
        except Exception:
            self.write_error_json("无效请求体")
            return

        text = str(body.get("text", "")).strip()
        custom_instructions = str(body.get("custom_instructions", "")).strip()

        if not text:
            self.write_error_json("缺少 text")
            return

        cfg = load_config()
        few_shot = fb_module.build_few_shot_section(DATA_DIR)
        fp_set = fb_module.get_false_positive_set(DATA_DIR)

        try:
            entities = detector.detect_entities(
                text,
                custom_instructions,
                models=_normalize_models(cfg),
                ollama_url=cfg.get("api_url", "http://localhost:11434"),
                few_shot_section=few_shot,
                false_positive_set=fp_set if fp_set else None,
            )
        except RuntimeError as e:
            self.write_error_json(str(e), 502)
            return

        mapping = {e["original"]: e["replacement"] for e in entities}
        desensitized = doc_handler.tokenize_text_with_mapping(text, mapping)

        self.write_json({
            "desensitized_text": desensitized,
            "entity_count": len(entities),
            "mapping": mapping,
        })


class DesensitizeHandler(BaseHandler):
    async def post(self):
        username = get_username(self)
        files = self.request.files.get("file", [])
        if not files:
            self.write_error_json("请上传文件")
            return

        file_info = files[0]
        original_filename = file_info["filename"]
        suffix = Path(original_filename).suffix.lower()

        if suffix not in ALLOWED_SUFFIXES:
            self.write_error_json(f"不支持的文件格式 {suffix}，支持：.docx .xlsx .xls .csv .pdf")
            return

        job_id  = str(uuid.uuid4())
        job_dir = JOBS_DIR / job_id
        job_dir.mkdir(exist_ok=True)

        upload_path = job_dir / f"upload_{original_filename}"
        # 使用线程池异步写文件，避免阻塞 Tornado IOLoop（对局域网大文件尤为重要）
        file_body = file_info["body"]
        loop = asyncio.get_running_loop()   # Python 3.10+ 推荐用法
        await loop.run_in_executor(executor, upload_path.write_bytes, file_body)

        custom_instructions = self.get_argument("custom_instructions", "").strip()

        # 先写入 queued 状态（同时持久化到磁盘，重启后可恢复）
        jobs[job_id] = {"status": "queued", "progress": "等待处理...", "progress_pct": 0}
        _save_state(job_id)

        # 入队（队列保证串行执行）
        enqueue(
            job_id,
            _process_desensitize,
            (job_id, upload_path, original_filename, custom_instructions, username),
            jobs
        )

        # 返回初始队列位置
        position = get_position(job_id)
        self.write_json({"job_id": job_id, "queue_position": position})


class RestoreHandler(BaseHandler):
    async def post(self):
        doc_files     = self.request.files.get("doc_file", [])
        mapping_files = self.request.files.get("mapping_file", [])

        if not doc_files:
            self.write_error_json("请上传需要还原的脱敏文档")
            return
        if not mapping_files:
            self.write_error_json("请上传对应的映射表 Excel 文件")
            return

        doc_info = doc_files[0]
        map_info = mapping_files[0]
        original_filename = doc_info["filename"]
        suffix = Path(original_filename).suffix.lower()

        if suffix not in ALLOWED_SUFFIXES:
            self.write_error_json(f"不支持的文件格式 {suffix}")
            return

        job_id  = str(uuid.uuid4())
        job_dir = JOBS_DIR / job_id
        job_dir.mkdir(exist_ok=True)

        doc_path     = job_dir / f"upload_{original_filename}"
        mapping_path = job_dir / f"mapping_{map_info['filename']}"

        loop = asyncio.get_running_loop()   # Python 3.10+ 推荐用法
        await loop.run_in_executor(executor, doc_path.write_bytes, doc_info["body"])
        await loop.run_in_executor(executor, mapping_path.write_bytes, map_info["body"])

        jobs[job_id] = {"status": "queued", "progress": "等待处理...", "progress_pct": 0}
        _save_state(job_id)

        enqueue(
            job_id,
            _process_restore,
            (job_id, doc_path, mapping_path, original_filename),
            jobs
        )

        position = get_position(job_id)
        self.write_json({"job_id": job_id, "queue_position": position})


class RestoreFromJobHandler(BaseHandler):
    """从历史记录还原：复用服务器上已有的映射表，用户只需上传脱敏文档。"""
    async def post(self):
        doc_files     = self.request.files.get("doc_file", [])
        source_job_id = self.get_body_argument("source_job_id", "").strip()

        if not doc_files:
            self.write_error_json("请上传需要还原的脱敏文档")
            return
        if not source_job_id:
            self.write_error_json("缺少来源任务 ID")
            return

        # 找到来源任务的映射表
        source_dir = JOBS_DIR / source_job_id
        if not source_dir.exists():
            self.write_error_json("来源任务不存在，文件可能已过期")
            return

        mapping_files_found = sorted(source_dir.glob("映射表_*.xlsx"))
        if not mapping_files_found:
            self.write_error_json("该历史记录的映射表文件已过期或不存在")
            return

        mapping_path = mapping_files_found[0]  # 取第一个（通常只有一个）

        doc_info = doc_files[0]
        original_filename = doc_info["filename"]
        suffix = Path(original_filename).suffix.lower()

        if suffix not in ALLOWED_SUFFIXES:
            self.write_error_json(f"不支持的文件格式 {suffix}")
            return

        job_id  = str(uuid.uuid4())
        job_dir = JOBS_DIR / job_id
        job_dir.mkdir(exist_ok=True)

        doc_path = job_dir / f"upload_{original_filename}"

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(executor, doc_path.write_bytes, doc_info["body"])

        jobs[job_id] = {"status": "queued", "progress": "等待处理...", "progress_pct": 0}
        _save_state(job_id)

        enqueue(
            job_id,
            _process_restore,
            (job_id, doc_path, mapping_path, original_filename),
            jobs
        )

        position = get_position(job_id)
        self.write_json({"job_id": job_id, "queue_position": position})


class StatusHandler(BaseHandler):
    def get(self, job_id):
        job = jobs.get(job_id)

        # 内存中找不到时，从 state.json 恢复（服务器重启场景）
        if not job:
            saved = _load_state(job_id)
            if saved:
                status = saved.get("status", "unknown")
                if status in ("queued", "processing"):
                    # 任务在处理中被重启打断，标记为错误
                    saved["status"] = "error"
                    saved["error"] = "服务器已重启，该任务未能完成，请重新提交文件"
                # 恢复到内存，避免下次再读磁盘
                jobs[job_id] = saved
                job = saved
            else:
                self.write_error_json("任务不存在", status=404)
                return

        result = dict(job)

        if job["status"] == "queued":
            pos = get_position(job_id)
            result["queue_position"] = pos
            if pos > 0:
                result["progress"] = f"排队等待中，前方还有 {pos} 个任务..."
            else:
                result["progress"] = "即将开始处理..."

        self.write_json(result)


class DownloadHandler(tornado.web.RequestHandler):
    def get(self, job_id, file_type):
        job = jobs.get(job_id)
        if not job:
            history = hist_module.load_history(DATA_DIR)
            job = next((h for h in history if h.get("id") == job_id), None)
            if not job:
                self.set_status(404)
                self.write("任务不存在")
                return

        filename_map = {
            "desensitized": job.get("desensitized_filename"),
            "mapping":      job.get("mapping_filename"),
            "restored":     job.get("restored_filename"),
        }
        filename = filename_map.get(file_type)
        if not filename:
            self.set_status(404)
            self.write("文件类型不存在")
            return

        file_path = JOBS_DIR / job_id / filename
        if not file_path.exists():
            self.set_status(404)
            self.write("文件不存在，可能已被自动清理（超过保留期限）")
            return

        encoded = urllib.parse.quote(filename.encode("utf-8"))
        # 同时提供 filename（ASCII 回退）和 filename*（RFC 5987 UTF-8 编码）
        # Edge 对纯 filename* 的支持存在兼容性问题，必须同时提供两者
        try:
            ascii_name = filename.encode("ascii").decode()
            ascii_name = ascii_name.replace('"', '\\"')   # 转义引号
        except UnicodeEncodeError:
            ascii_name = "download"
        self.set_header(
            "Content-Disposition",
            f'attachment; filename="{ascii_name}"; filename*=UTF-8\'\'{encoded}'
        )
        self.set_header("Content-Type", "application/octet-stream")
        self.set_header("X-Content-Type-Options", "nosniff")
        with open(file_path, "rb") as f:
            self.write(f.read())


class CancelJobHandler(BaseHandler):
    """DELETE /api/job/<job_id> — 取消排队中或运行中的任务。"""
    def delete(self, job_id):
        job = jobs.get(job_id)
        if not job:
            self.write_error_json("任务不存在", status=404)
            return
        status = job.get("status", "")
        if status not in ("queued", "processing"):
            self.write_error_json("只能取消排队中或处理中的任务")
            return

        ok = cancel_job(job_id)
        if ok:
            if status == "queued":
                # 排队中 → 已从队列移除，直接标记已取消
                jobs[job_id].update({
                    "status": "cancelled",
                    "progress": "已取消",
                    "progress_pct": 0,
                })
                _save_state(job_id)
            else:
                # 运行中 → 标记取消标志，处理函数在检查点生效
                jobs[job_id]["progress"] = "正在取消..."
        self.write_json({"ok": ok})


class ReorderQueueHandler(BaseHandler):
    """POST /api/queue/reorder — 调整排队中任务的顺序。"""
    def post(self):
        try:
            body = json.loads(self.request.body or b"{}")
        except Exception:
            self.write_error_json("无效请求体")
            return
        job_id    = body.get("job_id", "")
        direction = body.get("direction", "")
        if direction not in ("up", "down"):
            self.write_error_json("direction 必须为 'up' 或 'down'")
            return
        ok = reorder_job(job_id, direction)
        self.write_json({"ok": ok})


class HistoryHandler(BaseHandler):
    def get(self):
        username = get_username(self)
        records  = hist_module.load_history(DATA_DIR, user_id=username)
        for r in records:
            jid = r.get("id", "")
            r["files_exist"] = {
                "desensitized": bool(r.get("desensitized_filename") and
                                     (JOBS_DIR / jid / r["desensitized_filename"]).exists()),
                "mapping":      bool(r.get("mapping_filename") and
                                     (JOBS_DIR / jid / r["mapping_filename"]).exists()),
            }
        self.write_json({"history": records})


class HistoryDeleteHandler(BaseHandler):
    def delete(self, job_id):
        username = get_username(self)
        ok = hist_module.delete_job(DATA_DIR, job_id, user_id=username)
        self.write_json({"ok": ok})


# ─── 反馈 / 持续学习 Handlers ──────────────────────────────────────────────────

class FeedbackHandler(BaseHandler):
    """
    POST /api/feedback
    提交用户对某个任务结果的纠错反馈。

    请求体 JSON：
    {
      "job_id": "...",
      "missed": [{"original": "420802199307020624", "category": "证件编号"}],
      "false_positive": [{"original": "杭州市"}]
    }
    """
    def post(self):
        try:
            body = json.loads(self.request.body or b"{}")
        except Exception:
            self.write_error_json("无效请求体")
            return

        job_id   = body.get("job_id", "").strip()
        missed   = body.get("missed", [])
        fp_items = body.get("false_positive", [])

        if not job_id:
            self.write_error_json("缺少 job_id")
            return
        if not missed and not fp_items:
            self.write_error_json("missed 和 false_positive 不能同时为空")
            return

        fid = fb_module.save_feedback(DATA_DIR, job_id, missed, fp_items)
        stats = fb_module.get_stats(DATA_DIR)
        self.write_json({
            "ok": True,
            "feedback_id": fid,
            "stats": stats,
        })

    def get(self):
        """GET /api/feedback — 返回反馈统计和最近记录。"""
        stats   = fb_module.get_stats(DATA_DIR)
        records = fb_module.load_all(DATA_DIR)
        self.write_json({"stats": stats, "records": records[-50:]})


class ExportTrainingDataHandler(BaseHandler):
    """
    POST /api/export-training-data
    将累积的纠错反馈导出为 LoRA 微调用 JSONL 文件，可直接下载。
    """
    def post(self):
        count, out_path = fb_module.export_training_data(DATA_DIR)
        if count == 0:
            self.write_error_json("暂无可导出的训练数据（需先完成至少一次脱敏任务）")
            return

        encoded = __import__("urllib.parse", fromlist=["quote"]).quote(
            out_path.name.encode("utf-8")
        )
        self.set_header(
            "Content-Disposition",
            f'attachment; filename="training_data.jsonl"; filename*=UTF-8\'\'{encoded}'
        )
        self.set_header("Content-Type", "application/jsonlines")
        with open(out_path, "rb") as f:
            self.write(f.read())


class ReprocessHandler(BaseHandler):
    """
    POST /api/reprocess/<job_id>
    以原始上传文件为基础，新建一个脱敏处理任务（不修改原有记录）。
    """
    def post(self, job_id):
        job_dir = JOBS_DIR / job_id
        if not job_dir.exists():
            self.write_error_json("任务目录不存在，无法重新处理")
            return

        upload_files = list(job_dir.glob("upload_*"))
        if not upload_files:
            self.write_error_json("原始上传文件不存在，无法重新处理")
            return

        upload_file = upload_files[0]
        original_filename = upload_file.name[len("upload_"):]

        new_job_id  = str(uuid.uuid4())
        new_job_dir = JOBS_DIR / new_job_id
        new_job_dir.mkdir(parents=True, exist_ok=True)

        new_upload_path = new_job_dir / upload_file.name
        shutil.copy2(str(upload_file), str(new_upload_path))

        username = get_username(self)
        cfg = load_config()
        custom_instructions = cfg.get("custom_instructions", "")

        jobs[new_job_id] = {
            "status": "queued",
            "progress": "等待处理（重新生成）...",
            "progress_pct": 0,
        }
        _save_state(new_job_id)
        enqueue(new_job_id, _process_desensitize,
                (new_job_id, new_upload_path, original_filename,
                 custom_instructions, username), jobs)

        position = get_position(new_job_id)
        self.write_json({
            "job_id": new_job_id,
            "filename": original_filename,
            "queue_position": position,
        })


# ─── 应用组装 ─────────────────────────────────────────────────────────────────

def make_app():
    return tornado.web.Application([
        (r"/",                         MainHandler),
        (r"/api/settings",             SettingsHandler),
        (r"/api/secure_llm",           SecureLlmHandler),
        (r"/api/desensitize_text",     DesensitizeTextHandler),
        (r"/api/desensitize",          DesensitizeHandler),
        (r"/api/restore",              RestoreHandler),
        (r"/api/restore_from_job",     RestoreFromJobHandler),
        (r"/api/status/([^/]+)",       StatusHandler),
        (r"/api/download/([^/]+)/([^/]+)", DownloadHandler),
        (r"/api/history",              HistoryHandler),
        (r"/api/history/([^/]+)",      HistoryDeleteHandler),
        (r"/api/job/([^/]+)",          CancelJobHandler),
        (r"/api/queue/reorder",        ReorderQueueHandler),
        (r"/api/feedback",             FeedbackHandler),
        (r"/api/export-training-data", ExportTrainingDataHandler),
        (r"/api/reprocess/([^/]+)",    ReprocessHandler),
    ], debug=False)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app  = make_app()
    app.listen(port, address="0.0.0.0")   # 监听所有网卡，局域网可访问

    # 获取本机所有局域网 IP（显示全部，避免多网卡/DHCP变化导致找不到入口）
    import socket
    lan_ips: list[str] = []
    try:
        hostname = socket.gethostname()
        for info in socket.getaddrinfo(hostname, None):
            ip = info[4][0]
            # 只保留 IPv4、非回环、非链路本地
            if ':' not in ip and not ip.startswith('127.') and not ip.startswith('169.254.'):
                if ip not in lan_ips:
                    lan_ips.append(ip)
    except Exception:
        pass
    # 兜底：通过 UDP 探测路由获取主网卡 IP
    if not lan_ips:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            lan_ips = [s.getsockname()[0]]
            s.close()
        except Exception:
            pass

    print(f"\n{'='*60}")
    print(f"  合同文档脱敏系统（局域网版）已启动")
    print(f"  本机访问：    http://localhost:{port}")
    if lan_ips:
        for ip in lan_ips:
            print(f"  局域网访问：  http://{ip}:{port}")
    else:
        print(f"  局域网 IP 获取失败，请手动查看本机 IP")
    print(f"")
    print(f"  ⚠️  若IP每次重启后变化，请在路由器后台为本机设置静态IP")
    print(f"{'='*60}\n")

    try:
        tornado.ioloop.IOLoop.current().start()
    except KeyboardInterrupt:
        print("\n服务已停止")
        sys.exit(0)
