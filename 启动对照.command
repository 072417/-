#!/bin/bash
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ] || [ ! -d dist ]; then
  bash scripts/setup.sh || exit 1
fi
if ! curl -fsS --max-time 2 http://127.0.0.1:8765/api/health >/dev/null; then
  nohup .venv/bin/python -m uvicorn server.app:app --host 127.0.0.1 --port 8765 </dev/null >/tmp/duizhao-8765.log 2>&1 &
fi
echo '正在启动对照，请稍候。云端文件首次恢复可能需要几分钟。'
for ((i=0; i<180; i++)); do
  if curl -fsS --max-time 2 http://127.0.0.1:8765/api/health >/dev/null 2>&1; then
    open http://127.0.0.1:8765/
    exit 0
  fi
  sleep 2
done
echo '服务尚未就绪，请查看 /tmp/duizhao-8765.log'
read -r -p '按回车关闭窗口'
