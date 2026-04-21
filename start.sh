#!/bin/bash
# 合同文档脱敏系统 - 一键启动脚本
# 在 Mac 终端中运行：bash start.sh

set -e
cd "$(dirname "$0")"

echo ""
echo "====================================="
echo "  合同文档脱敏系统 - 启动中..."
echo "====================================="
echo ""

# 检查 Python3
if ! command -v python3 &>/dev/null; then
  echo "❌ 未找到 Python3，请先安装 Python 3.9+（https://python.org）"
  exit 1
fi

PYTHON_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "✅ Python $PYTHON_VER"

# 检查并安装依赖
echo "📦 检查依赖..."
MISSING=0
for pkg in tornado requests docx openpyxl pdfplumber; do
  python3 -c "import $pkg" 2>/dev/null || MISSING=1
done

if [ $MISSING -eq 1 ]; then
  echo "⚙️  安装缺失依赖..."
  pip3 install -r requirements.txt
  echo "✅ 依赖安装完成"
else
  echo "✅ 依赖已就绪"
fi

# 启动服务
echo ""
echo "🚀 启动服务（端口 8000）..."
echo "📌 按 Ctrl+C 停止服务"
echo ""
python3 app.py
