/**
 * ecosystem.config.js — PM2 部署配置
 *
 * 所有路径均可通过环境变量覆盖：
 *   APP_ROOT      — 项目根目录（默认: /root/theta_code/theta_project）
 *   APP_PORT      — 服务监听端口（默认: 8000）
 *
 * 用法:
 *   # 直接启动（使用默认值）
 *   pm2 start ecosystem.config.js
 *
 *   # 自定义路径
 *   APP_ROOT=/opt/theta/backend pm2 start ecosystem.config.js
 */

const path = require("path");

const APP_ROOT = process.env.APP_ROOT || path.join(process.env.HOME || "/root", "theta_code", "theta_project");
const APP_PORT = process.env.APP_PORT || "8000";

// 查找 venv python 解释器
const VENV_PYTHON = path.join(APP_ROOT, "venv", "bin", "python");
const PROJECT_MAIN = path.join(APP_ROOT, "main.py");

module.exports = {
  apps: [
    {
      name: "theta-backend",
      script: VENV_PYTHON,
      args: `-m uvicorn main:app --host 0.0.0.0 --port ${APP_PORT}`,
      cwd: APP_ROOT,
      env: {
        // 通过 dotenv 自动加载 .env 中的所有变量
        DOTENV: path.join(APP_ROOT, ".env"),
        PATH: [
          path.join(APP_ROOT, "venv", "bin"),
          "/usr/local/sbin",
          "/usr/local/bin",
          "/usr/sbin",
          "/usr/bin",
          "/sbin",
          "/bin",
        ].join(":"),
      },
      interpreter: "none",
      instance_var: "INSTANCE_ID",
      // 日志
      error_file: path.join(APP_ROOT, "logs", "error.log"),
      out_file: path.join(APP_ROOT, "logs", "out.log"),
      time: true,
      // 监控
      autorestart: true,
      max_restarts: 10,
      min_uptime: "10s",
    },
  ],
};
