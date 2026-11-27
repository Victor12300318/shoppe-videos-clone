FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

WORKDIR /app

# O wheel do PyAV já traz o FFmpeg com libx264: não precisa instalar ffmpeg no sistema
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY montar_video_mockup.py limpar_e_enviar_telegram.py webhook_server.py Mock.jpg ./

RUN useradd --create-home --uid 1000 app && chown -R app /app
USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"8000\")}/health', timeout=4)"

CMD ["python", "webhook_server.py"]
