# 对照 · 移动端 App 设计走查（服务器版）

React/TypeScript 工作台 + Python/FastAPI 图像分析 + SQLite/本地文件保存。公开浏览器版在相邻 `design-review-web` 目录，两种版本的识别引擎不同，不能混用测试准确率。

## 本机启动

此目录已完成安装和构建，可运行：

```sh
npm start
```

访问 `http://127.0.0.1:8765/`。macOS 也可双击 `启动对照.command`。首次安装执行 `bash scripts/setup.sh`。需要 Node.js 22、Python 3.9+；macOS OCR 使用本机 Apple Vision，初始化脚本会编译识别助手。Linux 容器使用 Tesseract 中英文 OCR。

## 使用

添加两张图片，确认逻辑宽度（360、375、390、414、430 或自定义），点击开始。可先体验示例。点击问题查看局部证据及测量值，修改人工状态后自动保存。替换截图会创建新分析版本。导出 JSON、标注 PNG 或含局部证据的 Markdown ZIP。

已实现四模式画布、缩放/平移、筛选、历史、复核状态、手动范围与忽略框、底部锚定比较、安全区参考线、超时/取消及可选服务器视觉模型解释。

## 模型配置

复制 `.env.example` 为 `.env`，仅在服务器填写支持图片输入及 JSON 输出的 OpenAI 兼容接口的 `VISION_BASE_URL`（含版本前缀，例如 /v1）、`VISION_MODEL` 和 `VISION_API_KEY`。重启后可在设置中选择是否发送候选局部图。未配置时本地图像链路可运行，AI 调用未作真实凭据验证。不要把密钥放入前端。

## 公开 Linux 部署准备

`Dockerfile` 和 `compose.yaml` 提供 Python/Tesseract 与 Caddy HTTPS 反向代理。将域名 A/AAAA 解析至服务器，开放 80/443，在 `.env` 设置 `SITE_DOMAIN`，运行 `docker compose up -d --build`。默认不对公网直接开放 API 端口。

公共模式 `PUBLIC_MODE=true` 使用随机 HttpOnly 会话 cookie 隔离访客图片和历史，不要求登录。清除 cookie 后无法访问此前会话记录，请先导出。默认本机模式保留已有本地记录。

镜像尚未在实际 Linux/Docker 主机验证；本机没有 Docker。服务器需访问镜像/包仓库进行初次构建，站点运行不依赖外部字体或图像 CDN。国内域名及备案条件需要按选定云平台确认。

## 已验证与限制

- 50 个本机测试通过：五档宽度、六类差异、真实字体字重、跨宽适配估算、CBase/LEGO 浮层忽略、裁剪坐标及取消。
- 2 个公共会话/API 测试通过：访客隔离、访问/修改限制、导出、删除、坏文件、跨站写入拒绝。
- 浏览器主流程测试通过：真实示例分析、问题定位、复核持久化、四种画布模式、导出、范围/设置、1280/390 布局。
- 测试仅证明合成样本及流程；没有获得多款真实 App 的人工标注集，不宣称普遍准确率。
- 字重会综合笔画宽度与字形墨色密度估算；它仍是截图证据，不能直接读取 CSS 或原生字体属性。
- 设计稿与截图宽度不一致时，会自动识别 360/375/390/414/430 逻辑宽度，并按左对齐、右对齐、居中或左右边距关系估算适配位置。换行和业务断点仍需人工复核。
- OCR 识别到的 CBase、LEGO 测试浮层会自动排除，不参与差异数量和热力图。
- 浏览器公共版不继承服务器版测试指标，另行验证。

测试命令：`.venv/bin/python -m pytest tests/test_engine.py -q`、`.venv/bin/python -m pytest tests/test_public.py -q`、`npm run test:ui`。公共会话测试须独立进程运行，测试隔离目录不含实际用户数据。

## Vercel 前端与独立后端

详见 [部署说明](deploy/VERCEL.md)。必须先配置具有持久磁盘的 HTTPS Python 后端，再设置 Vercel 的 `BACKEND_ORIGIN`。
