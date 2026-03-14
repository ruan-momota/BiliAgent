#!/bin/bash
# BiliAgent EC2 一键部署脚本
# 在 Ubuntu 24.04 EC2 实例上运行

set -e

echo "===== 1. 更新系统 ====="
sudo apt-get update && sudo apt-get upgrade -y

echo "===== 2. 安装 Docker ====="
sudo apt-get install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# 让当前用户不用 sudo 也能运行 docker
sudo usermod -aG docker $USER

echo "===== 3. 安装 Nginx ====="
sudo apt-get install -y nginx

echo "===== 4. 配置 Nginx ====="
sudo cp /home/ubuntu/BiliAgent/deploy/nginx.conf /etc/nginx/sites-available/biliagent
sudo ln -sf /etc/nginx/sites-available/biliagent /etc/nginx/sites-enabled/biliagent
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t && sudo systemctl restart nginx

echo "===== 安装完成！====="
echo "接下来请执行:"
echo "  1. cd ~/BiliAgent"
echo "  2. cp .env.example .env && nano .env   # 填入你的密钥"
echo "  3. newgrp docker                       # 刷新 docker 权限"
echo "  4. docker compose up -d --build        # 启动服务"
echo "  5. docker compose logs -f              # 查看日志"
