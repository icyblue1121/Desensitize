"""
queue_manager.py - 串行任务队列（保证 AI 服务同一时间只处理一个请求）

特性：
- Condition 变量驱动的列表队列，支持动态排序和取消
- 工作线程看门狗：若线程意外崩溃，下次入队时自动重启
- cancel_job()  : 取消排队中或运行中的任务
- reorder_job() : 调整排队中任务的顺序（上移/下移）
- is_cancelled() / clear_cancelled() : 供处理函数在关键检查点查询
- 每个任务携带唯一 seed + user 标识，确保模型服务在任务间清除 KV 缓存，
  避免前缀缓存跨任务复用，实现任务间完全隔离（见 detector.detect_entities）
"""

import threading
import logging

logger = logging.getLogger(__name__)

# ── 内部状态 ──────────────────────────────────────────────────────────────
_lock      = threading.Lock()
_condition = threading.Condition(_lock)   # 持有 _lock，用于通知工作线程

_pending_order: list = []    # 按提交顺序排列的 job_id 列表（等待中）
_pending_data:  dict = {}    # job_id -> (fn, args, jobs_dict)
_cancelled_jobs: set = set() # 请求取消的 job_id（含排队和运行中）

_currently_running: "str | None" = None   # 当前正在处理的任务 ID
_worker_thread: "threading.Thread | None" = None


# ── 工作线程 ─────────────────────────────────────────────────────────────

def _worker():
    """后台工作线程主循环，捕获所有异常保证不会意外退出。"""
    global _currently_running
    while True:
        job_id    = None
        fn        = None
        args      = None
        jobs_dict = None

        # 等待队列中出现任务
        with _condition:
            while not _pending_order:
                _condition.wait()
            job_id = _pending_order.pop(0)
            fn, args, jobs_dict = _pending_data.pop(job_id, (None, None, None))
            _currently_running = job_id

        if fn is None:
            # 数据异常，跳过
            with _lock:
                _currently_running = None
            continue

        # ── 在真正开始前检查是否已被取消 ──
        pre_cancelled = False
        with _lock:
            if job_id in _cancelled_jobs:
                _cancelled_jobs.discard(job_id)
                pre_cancelled = True

        if pre_cancelled:
            if jobs_dict and job_id in jobs_dict:
                jobs_dict[job_id].update({
                    "status": "cancelled",
                    "progress": "已取消",
                    "progress_pct": 0,
                })
            with _lock:
                _currently_running = None
            continue

        # ── 正式执行任务 ──
        try:
            if jobs_dict and job_id in jobs_dict:
                jobs_dict[job_id]["status"] = "processing"
            fn(*args)
        except Exception as e:
            logger.exception("任务 %s 执行失败: %s", job_id, e)
            if jobs_dict and job_id in jobs_dict:
                jobs_dict[job_id]["status"] = "error"
                jobs_dict[job_id]["error"]  = str(e)
        finally:
            with _lock:
                _currently_running = None


def _start_worker():
    """启动一个新的工作线程。"""
    global _worker_thread
    t = threading.Thread(target=_worker, daemon=True)
    t.name = "desensitize-worker"
    t.start()
    _worker_thread = t
    return t


def _ensure_worker_alive():
    """
    检查工作线程是否存活；若已死亡则重新启动。
    必须在持有 _condition（即 _lock）的情况下调用。
    """
    global _worker_thread
    if _worker_thread is None or not _worker_thread.is_alive():
        logger.warning("工作线程未运行，正在重启...")
        _start_worker()


# ── 公开接口 ─────────────────────────────────────────────────────────────

def enqueue(job_id: str, fn, args: tuple, jobs_dict: dict):
    """将任务加入队列，立即返回。入队前确认工作线程存活。"""
    with _condition:
        _pending_order.append(job_id)
        _pending_data[job_id] = (fn, args, jobs_dict)
        _ensure_worker_alive()
        _condition.notify()


def cancel_job(job_id: str) -> bool:
    """
    取消任务：
    - 排队中：从队列移除，立即生效，返回 True
    - 运行中：标记取消标志，处理函数下次到检查点时生效，返回 True
    - 其他状态：返回 False
    """
    with _condition:
        # 排队中 → 直接移除
        if job_id in _pending_order:
            _pending_order.remove(job_id)
            _pending_data.pop(job_id, None)
            _cancelled_jobs.discard(job_id)
            return True
        # 运行中 → 标记取消
        if _currently_running == job_id:
            _cancelled_jobs.add(job_id)
            return True
    return False


def reorder_job(job_id: str, direction: str) -> bool:
    """
    调整排队中任务的顺序。
    direction: 'up'（提前）或 'down'（推后）
    返回 True 表示移动成功，False 表示无法移动（不在队列或已到边界）。
    """
    with _lock:
        try:
            idx = _pending_order.index(job_id)
        except ValueError:
            return False

        if direction == "up" and idx > 0:
            _pending_order[idx], _pending_order[idx - 1] = (
                _pending_order[idx - 1], _pending_order[idx]
            )
            return True
        elif direction == "down" and idx < len(_pending_order) - 1:
            _pending_order[idx], _pending_order[idx + 1] = (
                _pending_order[idx + 1], _pending_order[idx]
            )
            return True
    return False


def is_cancelled(job_id: str) -> bool:
    """处理函数在关键检查点调用，判断是否被请求取消。"""
    with _lock:
        return job_id in _cancelled_jobs


def clear_cancelled(job_id: str):
    """处理函数确认取消后调用，清除标志。"""
    with _lock:
        _cancelled_jobs.discard(job_id)


def get_position(job_id: str) -> int:
    """
    返回该任务前方还有多少个任务（含正在执行的任务）。
    0  = 即将开始（无任务在前）
    N  = 前方还有 N 个任务（含正在跑的那个）
    -1 = 该任务不在等待队列中（已完成/已出错/已取消）
    """
    with _lock:
        try:
            idx = _pending_order.index(job_id)
            extra = 1 if _currently_running is not None else 0
            return idx + extra
        except ValueError:
            return -1


def get_currently_running() -> "str | None":
    """返回当前正在处理的任务 ID，如无则返回 None。"""
    with _lock:
        return _currently_running


def queue_depth() -> int:
    """当前等待队列长度（不含正在执行的任务）。"""
    with _lock:
        return len(_pending_order)


# 首次启动工作线程
_start_worker()
