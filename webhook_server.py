"""
Webhook do pipeline: recebe a URL de um vídeo, baixa, edita no mockup,
remove os metadados e envia ao Telegram (reusa limpar_e_enviar_telegram.py).

Iniciar:
    python webhook_server.py            # escuta em 0.0.0.0:8000 (PORTA no .env muda)

Disparar:
    POST /webhook   {"url": "https://.../video.mp4", "legenda": "Texto do post"}
    -> 202 {"job": "<id>"}   (o processamento roda em segundo plano, um vídeo por vez)

    GET  /status/<id>  -> {"status": "na_fila|baixando|editando|limpando|enviando|ok|erro", ...}

.env (opcional):
    WEBHOOK_SECRET=algo   # se definido, exige o header  X-Webhook-Secret: algo
    PORTA=8000
"""

from __future__ import annotations

import json
import os
import queue
import re
import shutil
import sys
import tempfile
import threading
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import requests

from limpar_e_enviar_telegram import (
    CHAT_ID_PADRAO,
    PASTA_SCRIPT,
    carregar_env,
    editar_dentro_do_limite,
    enviar_telegram,
    remover_metadados,
)

LIMITE_DOWNLOAD_BYTES = 500 * 1024 * 1024
MOCKUP = PASTA_SCRIPT / "Mock.jpg"

jobs: dict[str, dict] = {}
fila: "queue.Queue[str]" = queue.Queue()


NAVEGADOR = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36"}
RE_OG_VIDEO = re.compile(r'<meta[^>]+property=["\']og:video(?::url|:secure_url)?["\'][^>]+content=["\']([^"\']+)', re.I)
RE_MP4 = re.compile(r'https?:(?://|\\u002F\\u002F)[^"\'\s<>]+?\.mp4(?:\?[^"\'\s<>]*)?', re.I)


RE_DOWNLOAD_URL = re.compile(r'"downloadUrl"\s*:\s*"([^"]+)"')
RE_SUFIXO_MARCA = re.compile(r"(?:\.\d+){2}(?=\.mp4)")  # Shopee: "<id>.1600...369.4932.mp4" = com marca


def existe(url: str) -> bool:
    try:
        r = requests.head(url, timeout=15, allow_redirects=True, headers=NAVEGADOR)
        return r.ok and r.headers.get("Content-Type", "").startswith("video/")
    except requests.RequestException:
        return False


def achar_video_na_pagina(html: str) -> str | None:
    """Link de página (Shopee Video etc.): acha o .mp4 dentro do HTML, de preferência sem marca d'água."""
    m = RE_DOWNLOAD_URL.search(html)
    if m:
        return m.group(1).replace("\\u002F", "/")
    m = RE_OG_VIDEO.search(html) or RE_MP4.search(html)
    if not m:
        return None
    url = (m.group(1) if m.re is RE_OG_VIDEO else m.group(0)).replace("\\u002F", "/").replace("&amp;", "&")
    sem_marca = RE_SUFIXO_MARCA.sub("", url)
    if sem_marca != url and existe(sem_marca):
        return sem_marca
    return url


def baixar(url: str, destino: Path, _redirecionado: bool = False) -> None:
    with requests.get(url, stream=True, timeout=(15, 120), allow_redirects=True, headers=NAVEGADOR) as r:
        r.raise_for_status()
        if "text/html" in r.headers.get("Content-Type", ""):
            video_url = None if _redirecionado else achar_video_na_pagina(r.text)
            if not video_url:
                raise ValueError("A URL é uma página e não achei o vídeo nela; mande o link direto do .mp4.")
            print(f"      Página detectada; vídeo encontrado: {video_url}")
            return baixar(video_url, destino, _redirecionado=True)
        baixado = 0
        with destino.open("wb") as f:
            for bloco in r.iter_content(1024 * 1024):
                baixado += len(bloco)
                if baixado > LIMITE_DOWNLOAD_BYTES:
                    raise ValueError("Vídeo maior que 500 MB.")
                f.write(bloco)
    if baixado == 0:
        raise ValueError("Download vazio.")


def processar(job_id: str) -> None:
    job = jobs[job_id]
    tmp = Path(tempfile.mkdtemp(prefix=f"webhook_{job_id}_"))
    try:
        job["status"] = "baixando"
        bruto = tmp / "bruto.mp4"
        baixar(job["url"], bruto)

        job["status"] = "editando"
        editado = editar_dentro_do_limite(bruto, MOCKUP, tmp / "editado.mp4", 16)

        job["status"] = "limpando"
        limpo = remover_metadados(editado, tmp / "video.mp4")

        job["status"] = "enviando"
        token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        chat_id = (os.getenv("TELEGRAM_CHAT_ID", "").strip() or CHAT_ID_PADRAO)
        msg = enviar_telegram(limpo, job["legenda"], token, chat_id)

        job.update(status="ok", message_id=msg.get("message_id"))
        print(f"[job {job_id}] OK message_id={msg.get('message_id')}")
    except Exception as e:
        job.update(status="erro", erro=str(e))
        print(f"[job {job_id}] ERRO: {e}", file=sys.stderr)
        traceback.print_exc()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def worker() -> None:
    while True:
        processar(fila.get())
        fila.task_done()


class Handler(BaseHTTPRequestHandler):
    def _responder(self, codigo: int, corpo: dict) -> None:
        dados = json.dumps(corpo, ensure_ascii=False).encode("utf-8")
        self.send_response(codigo)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(dados)))
        self.end_headers()
        self.wfile.write(dados)

    def do_POST(self) -> None:
        if self.path.rstrip("/") != "/webhook":
            return self._responder(404, {"erro": "rota não encontrada"})

        segredo = os.getenv("WEBHOOK_SECRET", "").strip()
        if segredo and self.headers.get("X-Webhook-Secret") != segredo:
            return self._responder(401, {"erro": "segredo inválido"})

        try:
            tamanho = int(self.headers.get("Content-Length", 0))
            corpo = json.loads(self.rfile.read(tamanho) or b"{}")
        except (ValueError, json.JSONDecodeError):
            return self._responder(400, {"erro": "JSON inválido"})

        url = str(corpo.get("url", "")).strip()
        if urlparse(url).scheme not in ("http", "https"):
            return self._responder(400, {"erro": "campo 'url' (http/https) é obrigatório"})

        job_id = uuid.uuid4().hex[:8]
        jobs[job_id] = {"status": "na_fila", "url": url, "legenda": str(corpo.get("legenda", ""))}
        fila.put(job_id)
        print(f"[job {job_id}] recebido: {url}")
        self._responder(202, {"job": job_id, "status_url": f"/status/{job_id}"})

    def do_GET(self) -> None:
        if self.path.rstrip("/") == "/health":
            return self._responder(200, {"ok": True, "na_fila": fila.qsize()})
        if self.path.startswith("/status/"):
            job = jobs.get(self.path.split("/")[-1])
            if job:
                return self._responder(200, job)
        self._responder(404, {"erro": "não encontrado"})


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    carregar_env(PASTA_SCRIPT / ".env")
    if not os.getenv("TELEGRAM_BOT_TOKEN", "").strip():
        sys.exit("ERRO: defina TELEGRAM_BOT_TOKEN no arquivo .env")

    if not os.getenv("WEBHOOK_SECRET", "").strip():
        print("AVISO: WEBHOOK_SECRET não definido; qualquer um com a URL pode disparar o webhook.")

    threading.Thread(target=worker, daemon=True).start()
    porta = int(os.getenv("PORT") or os.getenv("PORTA") or "8000")
    print(f"Webhook ouvindo em http://0.0.0.0:{porta}/webhook")
    ThreadingHTTPServer(("0.0.0.0", porta), Handler).serve_forever()


if __name__ == "__main__":
    main()
