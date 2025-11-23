#!/bin/bash

# GraphRAG 一键启动脚本
# 使用 docker compose 启动所有服务

set -e

echo "=========================================="
echo "  启动 GraphRAG 服务"
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

# 检查是否存在 .env 文件
if [ ! -f .env ]; then
    echo "警告: 未找到 .env 文件，将使用默认配置"
    echo "提示: 建议创建 .env 文件并配置必要的环境变量"
fi

# 启动服务
echo ""
echo "正在启动服务..."
docker compose up -d

# 等待服务启动
echo ""
echo "等待服务启动..."
sleep 5

# 检查服务状态
echo ""
echo "=========================================="
echo "  服务状态"
echo "=========================================="
docker compose ps

echo ""
echo "=========================================="
echo "  启动完成！"
echo "=========================================="
echo ""
echo "服务访问地址："
echo "  - 前端: http://localhost:3000"
echo "  - 后端 API: http://localhost:8000"
echo "  - Neo4j Browser: http://localhost:7474"
echo ""
echo "查看日志: docker compose logs -f"
echo "停止服务: ./stop.sh"
echo ""



