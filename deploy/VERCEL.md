# Vercel + Python 容器部署

Vercel 发布静态前端，并将 `/api/*`、`/assets-store/*` 转发到独立 HTTPS 后端。浏览器始终请求同一个域名，匿名会话 Cookie、截图和报告不会经过公共缓存。后台任务仍在常驻 Python 进程执行。

## 1. 先启动后端

在支持 Docker 和持久磁盘的云平台使用本仓库 Dockerfile。容器端口为 8765，健康检查 `/api/health`，磁盘挂载 `/data`。使用一个实例、一个 Uvicorn worker，禁止自动休眠和自动横向扩容；当前 SQLite/进程任务队列不支持多副本。

环境变量：`PUBLIC_MODE=true`、`COOKIE_SECURE=true`、`DESIGN_REVIEW_DATA=/data`，以及 `ALLOWED_ORIGINS=https://你的正式前端域名`。多个受信任域名用英文逗号分隔，不要使用通配符。API Key 仅在后端平台填写，Vercel 无需模型密钥。

自有服务器可运行 `docker compose -f compose.backend.yaml up -d --build`，并将 HTTPS 反向代理转发到本机 8765。不要将此端口直接暴露给公网。Dockerfile 包含中文、英文 Tesseract OCR 和中文字体。

## 2. Vercel 导入

导入同一仓库，使用仓库中的 vercel.json 设置：Build Command 为 `npm run build:vercel`，框架 Other。在 Vercel 环境变量中填写 `BACKEND_ORIGIN=https://你的后端域名`，不带路径、用户名、密码。缺少该值时构建会停止，避免上线只有页面没有识别功能的站点。

构建使用 Vercel Build Output API，生成 `.vercel/output/static` 与路由配置。后端地址改变后需重新构建部署。

## 3. 上线验证

从正式前端访问 `/api/health`，应看到 `ok=true`、`ocr=true`、`publicMode=true`。验证上传、分析完成、截图、导出；另开隐私窗口确认看不到原窗口记录；重启容器后确认已完成记录还在。重启时未完成任务会标记失败，需重新执行。

后端必须有独立 HTTPS 地址和持久磁盘。免费休眠服务不能保证后台任务或历史文件可靠保存。此适配本身不创建服务器、不购买资源，也不代表后端已经上线。Vercel 和后端在大陆的访问情况仍需实际网络验证。
