#!/bin/bash

# GraphRAG 一键停止脚本
# 使用 docker compose 停止所有服务

set -e

echo "=========================================="
echo "  停止 GraphRAG 服务"
echo "=========================================="

# 检查 docker compose 是否可用
if ! command -v docker &> /dev/null; then
    echo "错误: 未找到 docker 命令，请先安装 Docker"
    exit 1
fi

# 检查 docker compose 是否可用
if ! docker compose version &> /dev/null; then
    echo "错误: docker compose 不可用，请确保 Docker Compose 已安装"
    exit 1
fi

# 停止服务
echo ""
echo "正在停止服务..."
docker compose down

echo ""
echo "=========================================="
echo "  服务已停止"
echo "=========================================="
echo ""
echo "提示:"
echo "  - 如需完全清理（包括数据卷）: docker compose down -v"
echo "  - 重新启动服务: ./start.sh"
echo ""



