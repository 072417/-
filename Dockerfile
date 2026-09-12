FROM node:22-bookworm-slim AS frontend
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY tsconfig.json vite.config.ts index.html ./
COPY src ./src
COPY public ./public
RUN npm run build

FROM python:3.11-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-chi-sim fonts-noto-cjk && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY server ./server
COPY --from=frontend /app/dist ./dist
RUN rm -f server/ocr-helper && useradd --uid 10001 --create-home appuser && mkdir /data && chown appuser:appuser /data
ENV DESIGN_REVIEW_DATA=/data PUBLIC_MODE=true COOKIE_SECURE=true PYTHONUNBUFFERED=1
USER appuser
EXPOSE 8765
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/api/health')"
CMD ["python","-m","uvicorn","server.app:app","--host","0.0.0.0","--port","8765","--proxy-headers","--forwarded-allow-ips","*"]
