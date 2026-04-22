"""
detector.py - 调用本地 AI API（兼容 Ollama / LM Studio / 任意 OpenAI 兼容服务）识别敏感信息并生成替换映射
"""

import json
import random
import re
import requests


DETECTION_PROMPT = """你是合同文档脱敏助手。分析以下合同文本，识别敏感实体并生成虚假替代内容。

【识别类别】
1. 公司/机构名称 - 甲乙双方公司全称；**简称须单独列出**
2. 人名 - 法定代表人、联系人、经办人、签字人（含"王总"中的姓）
3. 地址 - 注册/办公/项目/收件/通信等各类地址；正文中嵌入的地址片段同样须识别
4. 联系方式 - 手机号、固话、传真
5. 电子邮箱
6. 网址/域名
7. 账号/用户名
8. 银行及金融信息 - 账号、开户行、SWIFT/CNAPS行号
9. 证件编号 - 营业执照号、统一社会信用代码、身份证号
10. 金额 - 具体数字（保留币种单位，如"500万元"→"320万元"）
11. 项目/产品名称 - **同一名称的不同括号形式须各自独立列出**（如《X》和<X>分别记录）
12. 合同编号/出版物号/ISBN/软著号/著作权编号
13. 游戏名称

【括号变体规则】
同一名称可能以《》""''""''<>「」等多种括号出现，每种变体单独列出，替换内容括号形式与原文保持一致。

【替换规则】
- 公司全称→符合中国命名习惯的虚假名称；简称→对应全称简称
- 人名→真实感中文姓名（2-3字）
- 地址→真实行政区划+虚构门牌（与公司城市一致）
- 手机→合法号段11位；固话→带区号虚构号码
- 银行账号→16-19位纯数字；邮箱→格式合法虚构地址
- 统一社会信用代码→18位合规格式；合同编号→保持相近格式
- 同一实体只记录一次，映射到同一替代内容

【注意】
- original须与原文完全一致（含标点/全半角/括号）
- "甲方""乙方"等代称本身不是敏感信息
- 公司简称必须单独列出（系统逐词精确匹配）
- 括号变体必须各自独立列出

【示例】
{{"entities": [
  {{"original": "北京某科技有限公司", "category": "公司/机构名称", "replacement": "鸿达建设集团有限公司"}},
  {{"original": "某科技", "category": "公司/机构名称", "replacement": "鸿达建设"}},
  {{"original": "张三", "category": "人名", "replacement": "李建国"}},
  {{"original": "13812345678", "category": "联系方式", "replacement": "15067891234"}},
  {{"original": "《极略三国》", "category": "项目/产品名称", "replacement": "《天龙传说》"}},
  {{"original": "<极略三国>", "category": "项目/产品名称", "replacement": "<天龙传说>"}}
]}}

{custom_section}
【合同文本】
{document_text}

严格按以下JSON格式返回，不含其他文字或markdown：
{{"entities": [{{"original": "原始文本", "category": "类别", "replacement": "替代内容"}}]}}"""


def _strip_think_tags(text: str) -> str:
    """去除推理模型输出的 <think>...</think> 标签。"""
    return re.sub(r'<think>[\s\S]*?</think>', '', text, flags=re.IGNORECASE).strip()


# ─── 公司简称自动补全 ────────────────────────────────────────────────────────

_CORP_SUFFIXES = [
    '股份有限公司', '有限责任公司', '（集团）有限公司', '集团股份有限公司',
    '集团有限公司', '有限公司', '国际集团', '股份公司',
]

# 常见省份/直辖市/主要城市
_GEO_PREFIXES = [
    '东北', '华北', '华南', '华东', '华中', '西北', '西南', '中国',
    '黑龙江', '内蒙古', '新疆', '广东', '浙江', '江苏', '四川',
    '湖北', '湖南', '山东', '山西', '陕西', '辽宁', '福建', '安徽',
    '北京', '上海', '天津', '重庆', '深圳', '广州', '成都', '杭州',
    '南京', '武汉', '西安', '苏州', '青岛', '大连', '宁波', '厦门',
]


def _derive_abbrev(full_name: str) -> list[str]:
    """从公司全称推导常见简称，返回候选列表（按优先级）。"""
    name = full_name.strip()
    stripped = name
    for suffix in sorted(_CORP_SUFFIXES, key=len, reverse=True):
        if stripped.endswith(suffix):
            stripped = stripped[:-len(suffix)]
            break

    geo_stripped = stripped
    for prefix in sorted(_GEO_PREFIXES, key=len, reverse=True):
        if geo_stripped.startswith(prefix):
            candidate = geo_stripped[len(prefix):]
            if candidate and candidate[0] in '省市区县':
                candidate = candidate[1:]
            if candidate and len(candidate) >= 2:
                geo_stripped = candidate
            break

    candidates = []
    if geo_stripped and geo_stripped != name and geo_stripped != stripped and len(geo_stripped) >= 2:
        candidates.append(geo_stripped)
    if stripped and stripped != name and stripped != geo_stripped and len(stripped) >= 2:
        candidates.append(stripped)

    seen = set()
    result = []
    for c in candidates:
        if c not in seen and c != full_name:
            seen.add(c)
            result.append(c)
    return result


# ─── 括号变体自动补全 ────────────────────────────────────────────────────────

_BRACKET_DEFS = [
    (re.compile(r'^《([^《》]+)》$'), '《', '》'),
    (re.compile(r'^"([^""]+)"$'),    '\u201c', '\u201d'),
    (re.compile(r"^'([^'']+)'$"),    '\u2018', '\u2019'),
    (re.compile(r'^"([^"]+)"$'),     '"',  '"'),
    (re.compile(r"^'([^']+)'$"),     "'",  "'"),
    (re.compile(r'^<([^<>]{1,40})>$'), '<',  '>'),
    (re.compile(r'^「([^「」]+)」$'), '「', '」'),
]


def _extract_bracket_core(text: str):
    for pattern, lb, rb in _BRACKET_DEFS:
        m = pattern.match(text)
        if m:
            return m.group(1), lb, rb
    return None, None, None


def _auto_expand_bracket_variants(entities: list, text: str) -> list:
    """补全 AI 漏识别的括号变体（如《》与<>混用）。"""
    existing = {e['original'] for e in entities}
    new_entities = []
    for entity in list(entities):
        original, replacement = entity['original'], entity['replacement']
        category = entity.get('category', '')
        core_orig, _, _ = _extract_bracket_core(original)
        core_orig = core_orig or original
        core_repl, _, _ = _extract_bracket_core(replacement)
        core_repl = core_repl or replacement
        for _, lb, rb in _BRACKET_DEFS:
            v_orig = lb + core_orig + rb
            if v_orig in text and v_orig not in existing and v_orig != original:
                new_entities.append({'original': v_orig, 'category': category,
                                     'replacement': lb + core_repl + rb})
                existing.add(v_orig)
    return entities + new_entities


def _auto_expand_abbreviations(entities: list, text: str) -> list:
    """补全 AI 漏识别的公司简称。"""
    existing = {e['original'] for e in entities}
    new_entities = []
    for entity in entities:
        if '公司' not in entity.get('category', '') and '机构' not in entity.get('category', ''):
            continue
        orig_full, repl_full = entity['original'], entity['replacement']
        orig_abbrevs = _derive_abbrev(orig_full)
        repl_abbrevs = _derive_abbrev(repl_full)
        for i, orig_abbrev in enumerate(orig_abbrevs):
            if orig_abbrev in existing:
                continue
            if orig_abbrev not in text.replace(orig_full, ''):
                continue
            repl_abbrev = repl_abbrevs[i] if i < len(repl_abbrevs) else repl_full[:4]
            new_entities.append({'original': orig_abbrev, 'category': '公司/机构名称',
                                  'replacement': repl_abbrev})
            existing.add(orig_abbrev)
    return entities + new_entities


def _run_detection_pass(prompt: str, model: str, ollama_url: str,
                        job_id: str, text: str, pass_idx: int) -> list:
    """对单个模型执行一次识别，返回解析后的实体列表。"""
    endpoint = ollama_url.rstrip("/") + "/v1/chat/completions"
    task_seed = random.randint(1, 2**31 - 1)
    task_user = f"job-{job_id}-p{pass_idx}" if job_id else f"job-{task_seed}"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "你是严格按JSON格式输出结果的助手，不输出任何额外文字。/no_think"},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "temperature": 0.1,
        "max_tokens": 16384,
        "seed": task_seed,
        "user": task_user,
    }

    try:
        response = requests.post(endpoint, headers={"Content-Type": "application/json"},
                                 json=payload, timeout=600)
    except requests.exceptions.Timeout:
        raise RuntimeError(f"AI 服务响应超时（模型 {model}，超过10分钟）")
    except requests.exceptions.ConnectionError:
        raise RuntimeError(f"无法连接到 AI 服务（{ollama_url}），请确认服务已启动且模型已加载")

    if response.status_code == 404:
        raise RuntimeError(f"模型 {model} 未找到，请确认模型已在 AI 服务中加载")
    elif response.status_code != 200:
        raise RuntimeError(f"AI 服务返回错误 {response.status_code}（模型 {model}）：{response.text[:300]}")

    raw = response.json()["choices"][0]["message"]["content"]
    response_text = _strip_think_tags(raw)

    if response_text.startswith("```"):
        response_text = re.sub(r'^```[a-z]*\s*', '', response_text)
        response_text = re.sub(r'\s*```$', '', response_text).strip()

    json_match = re.search(r'\{[\s\S]*\}', response_text)
    if json_match:
        response_text = json_match.group(0)

    try:
        data = json.loads(response_text)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"识别结果格式解析失败（模型 {model}：{e}），请重试或检查模型是否支持 JSON 输出")

    results = []
    for entity in data.get("entities", []):
        original = entity.get("original", "").strip()
        replacement = entity.get("replacement", "").strip()
        category = entity.get("category", "未知").strip()
        if original and replacement and original in text:
            results.append({"original": original, "category": category, "replacement": replacement})
    return results


def detect_entities(text: str, custom_instructions: str, api_key: str = "",
                    models: "list[str] | None" = None,
                    ollama_url: str = "http://localhost:11434",
                    job_id: str = "",
                    few_shot_section: str = "",
                    false_positive_set: "set | None" = None,
                    progress_cb=None) -> list:
    """
    调用本地 AI API（OpenAI 兼容接口）识别文档中的敏感实体并生成替换映射。
    兼容 Ollama、LM Studio 及其他 OpenAI 兼容服务。

    models            : 本地模型列表。每个模型用同一提示词独立跑一次，结果按 original 去重合并。
                        默认 ["qwen3.5:35b"]，即只跑一次。
    progress_cb       : 可选回调 fn(idx, total, model_name)，在每个模型开始前调用。
    """
    model_list = [m.strip() for m in (models or ["qwen3.5:35b"]) if m and m.strip()]
    if not model_list:
        model_list = ["qwen3.5:35b"]

    # ── 构造 custom_section（自定义指令 + 历史 few-shot 示例）──────────────────
    parts = []
    if custom_instructions and custom_instructions.strip():
        parts.append(
            f"【本次任务额外指令（优先于上述所有默认规则）】\n{custom_instructions.strip()}\n"
        )
    if few_shot_section and few_shot_section.strip():
        parts.append(f"{few_shot_section.strip()}\n")
    custom_section = "\n".join(parts) if parts else ""

    prompt = DETECTION_PROMPT.format(document_text=text, custom_section=custom_section)

    # ── 按模型顺序执行 N 次识别，同一 prompt，结果合并去重 ────────────────────
    valid_entities: list = []
    seen: set = set()
    for idx, model in enumerate(model_list):
        if progress_cb:
            try:
                progress_cb(idx, len(model_list), model)
            except Exception:
                pass
        pass_results = _run_detection_pass(prompt, model, ollama_url, job_id, text, idx)
        for entity in pass_results:
            if entity["original"] not in seen:
                valid_entities.append(entity)
                seen.add(entity["original"])

    valid_entities = _auto_expand_abbreviations(valid_entities, text)
    valid_entities = _auto_expand_bracket_variants(valid_entities, text)

    # ── 过滤历史误识别项 ───────────────────────────────────────────────────────
    if false_positive_set:
        valid_entities = [
            e for e in valid_entities
            if e["original"] not in false_positive_set
        ]

    return valid_entities


def check_ollama(ollama_url: str, model: str) -> dict:
    """检查 AI 服务状态及模型是否已加载。兼容 Ollama 和 LM Studio（OpenAI 兼容接口）。"""
    base_url = ollama_url.rstrip("/")
    try:
        # 优先使用 OpenAI 兼容端点 /v1/models（LM Studio / Ollama 均支持）
        r = requests.get(base_url + "/v1/models", timeout=5)
        if r.status_code == 200:
            models_data = r.json()
            # OpenAI 格式：{"data": [{"id": "model-name", ...}]}
            if "data" in models_data:
                model_ids = [m.get("id", "") for m in models_data.get("data", [])]
            else:
                # Ollama 兼容格式兜底
                model_ids = [m.get("id", "") or m.get("name", "") for m in models_data.get("models", [])]
            base = model.split(":")[0]
            if model_ids and not any(base in mid for mid in model_ids):
                return {"ok": False, "message": f"模型 {model} 未加载，请在 AI 服务中加载该模型"}
            return {"ok": True, "message": f"AI 服务就绪，模型 {model} 已加载"}
        # /v1/models 不可用时回退：尝试 Ollama 专有端点 /api/tags
        r2 = requests.get(base_url + "/api/tags", timeout=5)
        if r2.status_code == 200:
            model_names = [m.get("name", "") for m in r2.json().get("models", [])]
            base = model.split(":")[0]
            if not any(base in name for name in model_names):
                return {"ok": False, "message": f"模型 {model} 未找到，请先运行：ollama pull {model}"}
            return {"ok": True, "message": f"AI 服务就绪，模型 {model} 已加载"}
        return {"ok": False, "message": f"AI 服务异常（HTTP {r.status_code}）"}
    except requests.exceptions.ConnectionError:
        return {"ok": False, "message": f"无法连接到 AI 服务（{ollama_url}），请确认服务已启动"}
    except Exception as e:
        return {"ok": False, "message": f"检查失败：{e}"}


def call_openai_compatible(task_prompt: str, text: str,
                           model: str = "qwen3.5:35b",
                           api_url: str = "http://localhost:11434",
                           system_prompt: str = "") -> str:
    """
    调用 OpenAI 兼容 chat completions 接口处理文本。
    适用于“先本地脱敏，再发外部 AI”的下游任务。
    """
    endpoint = api_url.rstrip("/") + "/v1/chat/completions"
    sys_prompt = system_prompt.strip() if system_prompt else "你是一个专业的文档处理助手，请严格按用户要求处理文本。"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {
                "role": "user",
                "content": f"任务要求：\n{task_prompt.strip()}\n\n待处理文本：\n{text}"
            }
        ],
        "stream": False,
        "temperature": 0.2,
        "max_tokens": 8192,
    }

    try:
        response = requests.post(
            endpoint,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=300
        )
    except requests.exceptions.Timeout:
        raise RuntimeError("外部 AI 服务响应超时（超过5分钟）")
    except requests.exceptions.ConnectionError:
        raise RuntimeError(f"无法连接到外部 AI 服务（{api_url}）")

    if response.status_code == 404:
        raise RuntimeError(f"外部 AI 模型 {model} 未找到")
    if response.status_code != 200:
        raise RuntimeError(f"外部 AI 服务返回错误 {response.status_code}：{response.text[:300]}")

    try:
        raw = response.json()["choices"][0]["message"]["content"]
    except Exception:
        raise RuntimeError("外部 AI 返回格式不符合 OpenAI 兼容规范")
    return _strip_think_tags(raw).strip()
