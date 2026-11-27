"""
Pipeline completo para a VPS:
    1) Edita o vídeo bruto no mockup (usa montar_video_mockup.py)
    2) Remove TODOS os metadados do MP4 final (sem perder qualidade)
    3) Envia o vídeo limpo para o Telegram (substitui o nó "Send a video" do n8n)

Como os sites "limpam" metadados de vídeo:
    - Remux: copiam o áudio/vídeo para um MP4 novo SEM recomprimir, descartando
      as tags do container (encoder, creation_time, GPS/location, make/model do
      celular, handler_name, capítulos, faixas extras de dados/legenda).
    - Removem o SEI do H.264: o x264 grava DENTRO do vídeo um texto com a versão
      e todos os parâmetros do encoder ("x264 - core 165 ... crf=16.0").
    - Alguns ainda reencodam o vídeo inteiro (perde qualidade; aqui não precisa).
  Este script faz os dois primeiros passos com PyAV (mesma lib do editor).

Configuração (.env na mesma pasta do script):
    TELEGRAM_BOT_TOKEN=123456:ABC...
    TELEGRAM_CHAT_ID=-5381746349

Uso:
    # Editar + limpar + enviar
    python limpar_e_enviar_telegram.py --video bruto.mp4 --legenda "Texto do post"

    # Vídeo já editado: só limpar + enviar
    python limpar_e_enviar_telegram.py --ja-editado video_final.mp4 --legenda "Texto"

    # Só limpar (sem enviar), útil para testar
    python limpar_e_enviar_telegram.py --ja-editado video_final.mp4 --nao-enviar
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

import av
import requests

PASTA_SCRIPT = Path(__file__).resolve().parent
sys.path.insert(0, str(PASTA_SCRIPT))

LIMITE_TELEGRAM_BYTES = 50 * 1024 * 1024  # limite do Bot API oficial para upload
LIMITE_LEGENDA = 1024                      # limite de caracteres da legenda
CHAT_ID_PADRAO = "-5381746349"

# SEI = NAL units de "informação extra" (onde fica a assinatura do encoder)
FILTRO_SEI = {
    "h264": "filter_units=remove_types=6",
    "hevc": "filter_units=remove_types=39|40",
}


# =============================================================================
# .env
# =============================================================================
def carregar_env(caminho: Path) -> None:
    """Carrega o .env (usa python-dotenv se instalado, senão um parser simples)."""
    if not caminho.is_file():
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(caminho)
        return
    except ImportError:
        pass
    for linha in caminho.read_text(encoding="utf-8").splitlines():
        linha = linha.strip()
        if not linha or linha.startswith("#") or "=" not in linha:
            continue
        chave, valor = linha.split("=", 1)
        os.environ.setdefault(chave.strip(), valor.strip().strip('"').strip("'"))


# =============================================================================
# Remoção de metadados
# =============================================================================
def remover_metadados(entrada: Path, saida: Path) -> Path:
    """
    Remux sem perda: copia só a 1ª faixa de vídeo e a 1ª de áudio para um MP4 novo,
    sem tags, sem capítulos, sem faixas extras e sem o SEI do encoder.
    Equivalente a:
      ffmpeg -i entrada -map 0:v:0 -map 0:a:0? -c copy -map_metadata -1 -map_chapters -1 \
             -bsf:v filter_units=remove_types=6 -fflags +bitexact \
             -flags:v +bitexact -flags:a +bitexact -movflags +faststart saida
    """
    saida.parent.mkdir(parents=True, exist_ok=True)

    with av.open(str(entrada)) as inp:
        if not inp.streams.video:
            raise ValueError(f"Nenhum vídeo encontrado em {entrada}")
        in_v = inp.streams.video[0]
        in_a = inp.streams.audio[0] if inp.streams.audio else None

        # +bitexact: o muxer não grava "encoder=Lavf..." nem outras assinaturas
        with av.open(
            str(saida),
            mode="w",
            format="mp4",
            options={"movflags": "+faststart", "fflags": "+bitexact"},
        ) as out:
            out.metadata.clear()

            out_v = out.add_stream_from_template(in_v)
            out_v.metadata.clear()
            out_a = None
            if in_a is not None:
                out_a = out.add_stream_from_template(in_a)
                out_a.metadata.clear()

            bsf = None
            filtro = FILTRO_SEI.get(in_v.codec_context.name)
            if filtro and filtro.split("=")[0] in av.bitstream.bitstream_filters_available:
                bsf = av.bitstream.BitStreamFilterContext(filtro, in_v, out_v)

            streams = [in_v] + ([in_a] if in_a is not None else [])
            for pkt in inp.demux(streams):
                if pkt.dts is None:
                    continue
                if pkt.stream is in_v:
                    pacotes = bsf.filter(pkt) if bsf else [pkt]
                    for p in pacotes:
                        p.stream = out_v
                        out.mux(p)
                else:
                    pkt.stream = out_a
                    out.mux(pkt)

            if bsf:
                for p in bsf.filter(None):
                    p.stream = out_v
                    out.mux(p)

    return saida


def info_video(caminho: Path) -> dict:
    """Largura, altura e duração (Telegram usa isso para exibir o player certo)."""
    with av.open(str(caminho)) as c:
        v = c.streams.video[0]
        duracao = int(round(c.duration / av.time_base)) if c.duration else 0
        return {"width": v.width, "height": v.height, "duration": duracao}


# =============================================================================
# Telegram
# =============================================================================
def enviar_telegram(video: Path, legenda: str, token: str, chat_id: str, tentativas: int = 4) -> dict:
    """POST multipart em sendVideo (mesmo que o nó do n8n com binaryData=true)."""
    if video.stat().st_size > LIMITE_TELEGRAM_BYTES:
        raise ValueError(
            f"{video.name} tem {video.stat().st_size / 1e6:.1f} MB; o Bot API aceita até 50 MB."
        )

    if len(legenda) > LIMITE_LEGENDA:
        legenda = legenda[: LIMITE_LEGENDA - 1] + "…"

    url = f"https://api.telegram.org/bot{token}/sendVideo"
    dados = {"chat_id": chat_id, "caption": legenda, "supports_streaming": "true", **info_video(video)}

    for tentativa in range(1, tentativas + 1):
        try:
            with video.open("rb") as f:
                resp = requests.post(
                    url,
                    data=dados,
                    files={"video": (video.name, f, "video/mp4")},
                    # O 1º valor também é o tempo MÁXIMO para escrever o upload inteiro
                    # (sendall único). Dimensiona para uploads lentos, ~50 KB/s.
                    timeout=(60 + video.stat().st_size / 50_000, 600),
                )
            corpo = resp.json()
        except (requests.RequestException, ValueError) as e:
            # nunca mostra o token no log
            corpo = {"ok": False, "description": str(e).replace(token, "***")}
            resp = None

        if corpo.get("ok"):
            return corpo["result"]

        # Flood control do Telegram: espera o tempo pedido
        espera = corpo.get("parameters", {}).get("retry_after")
        erro_cliente = resp is not None and 400 <= resp.status_code < 500 and resp.status_code != 429
        if erro_cliente or tentativa == tentativas:
            raise RuntimeError(f"Telegram recusou o envio: {corpo.get('description')}")
        espera = espera or 2**tentativa
        print(f"      Falha ({corpo.get('description')}). Tentando de novo em {espera}s...")
        time.sleep(espera)

    raise RuntimeError("Falha no envio ao Telegram.")


# =============================================================================
# Pipeline
# =============================================================================
def editar_dentro_do_limite(video_bruto: Path, mockup: Path, destino: Path, crf_inicial: int) -> Path:
    """Roda o editor; se passar de 50 MB, re-renderiza com CRF maior (arquivo menor)."""
    from montar_video_mockup import processar_video

    # Teto de bitrate calculado pela duração para caber em ~45 MB já na 1ª renderização
    # (o áudio é copiado; reserva 192 kbps para ele). Teto máximo de 12 Mbps.
    with av.open(str(video_bruto)) as c:
        segundos = max(1.0, (c.duration or 0) / av.time_base)
    max_kbps = int(min(12_000, 45 * 8_000 / segundos - 192))

    crf = crf_inicial
    while True:
        processar_video(
            caminho_mockup=mockup, caminho_video=video_bruto, caminho_saida=destino, crf=crf, max_kbps=max_kbps
        )
        tamanho = destino.stat().st_size
        if tamanho <= LIMITE_TELEGRAM_BYTES * 0.98 or crf >= 28:
            return destino
        crf += 3
        print(f"      {tamanho / 1e6:.1f} MB passa do limite do Telegram. Re-renderizando com CRF={crf}...")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    p = argparse.ArgumentParser(description="Edita, remove metadados e envia o vídeo ao Telegram.")
    origem = p.add_mutually_exclusive_group(required=True)
    origem.add_argument("--video", type=Path, help="Vídeo bruto (será editado no mockup).")
    origem.add_argument("--ja-editado", type=Path, help="Vídeo já editado (pula a edição).")
    p.add_argument("--mockup", type=Path, default=PASTA_SCRIPT / "Mock.jpg")
    p.add_argument("--legenda", default="", help="Legenda do vídeo no Telegram.")
    p.add_argument("--saida", type=Path, help="Onde salvar o vídeo limpo (padrão: pasta temporária).")
    p.add_argument("--crf", type=int, default=16, help="Qualidade inicial do editor (padrão 16).")
    p.add_argument("--nao-enviar", action="store_true", help="Só edita/limpa, não envia.")
    p.add_argument("--manter-arquivos", action="store_true", help="Não apaga os temporários.")
    args = p.parse_args()

    carregar_env(PASTA_SCRIPT / ".env")
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = (os.getenv("TELEGRAM_CHAT_ID", "").strip() or CHAT_ID_PADRAO)
    if not args.nao_enviar and not token:
        print("ERRO: defina TELEGRAM_BOT_TOKEN no arquivo .env", file=sys.stderr)
        return 2

    tmp = Path(tempfile.mkdtemp(prefix="video_tg_"))
    temporarios: list[Path] = []

    try:
        if args.video:
            print("== Editando no mockup ==")
            editado = editar_dentro_do_limite(args.video, args.mockup, tmp / "editado.mp4", args.crf)
            temporarios.append(editado)
        else:
            editado = args.ja_editado
            if not editado.is_file():
                raise FileNotFoundError(f"Arquivo não encontrado: {editado}")

        print("== Removendo metadados ==")
        limpo = args.saida or (tmp / "video.mp4")
        remover_metadados(editado, limpo)
        if not args.saida:
            temporarios.append(limpo)
        print(f"      OK: {limpo} ({limpo.stat().st_size / 1e6:.1f} MB)")

        if args.nao_enviar:
            if not args.saida:
                temporarios.remove(limpo)
            print(f"[OK] Vídeo limpo salvo em: {limpo}")
            return 0

        print(f"== Enviando ao Telegram (chat {chat_id}) ==")
        msg = enviar_telegram(limpo, args.legenda, token, chat_id)
        print(f"[OK] Enviado. message_id={msg.get('message_id')}")
        return 0
    except Exception as e:
        print(f"ERRO: {e}", file=sys.stderr)
        return 1
    finally:
        if not args.manter_arquivos:
            for f in temporarios:
                f.unlink(missing_ok=True)
            try:
                tmp.rmdir()
            except OSError:
                pass


if __name__ == "__main__":
    sys.exit(main())
