# Desensitize

本项目用于文档脱敏与还原，支持本地/自托管 OpenAI 兼容模型服务（如 Ollama、LM Studio）。

## 启动

```bash
bash start.sh
```

或：

```bash
pip3 install -r requirements.txt
python3 app.py
```

## 配置说明

配置文件：`config.json`

- `api_url`: OpenAI 兼容接口地址（默认 `http://localhost:11434`）
- `model`: 主模型（用于本地敏感信息识别）
- `model_2`: 第二模型（双模型补漏时使用）
- `dual_model`: 是否启用双模型补漏

## 新增端到端接口：`/api/secure_llm`

能力：`原文 -> 本地脱敏 -> 外部AI处理 -> 自动还原`

### 请求示例

```bash
curl -X POST "http://localhost:8000/api/secure_llm" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "甲方北京某科技有限公司，联系人张三，手机号13812345678。",
    "task": "请提炼成3条要点",
    "external_api_url": "http://localhost:11434",
    "external_model": "qwen3.5:35b",
    "external_system_prompt": "你是严谨的合同分析助手",
    "custom_instructions": "",
    "passthrough_if_empty": true,
    "return_tokenized": false
  }'
```

### 返回字段

- `result`: 自动还原后的最终结果
- `entity_count`: 识别到的敏感实体数量
- `mapping_count`: 映射项数量
- `used_model`: 本次外部AI调用模型
- `used_api_url`: 本次外部AI调用地址
- `tokenized_text` / `tokenized_result`: 仅在 `return_tokenized=true` 时返回
