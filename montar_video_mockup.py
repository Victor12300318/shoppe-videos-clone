"""
Insere um vídeo bruto exatamente sobre o quadrado cinza de um mockup (ex: Mock.jpg)
preservando o máximo de qualidade para publicação no Instagram Reels.

Destaques técnicos de qualidade:
- Detecta automaticamente o quadrado cinza do mockup.
- Exporta por padrão em 1080x1920 (Full HD 9:16, padrão do Instagram Reels) sem
  distorcer o mockup nem o vídeo.
- Faz recorte (crop) e redimensionamento Lanczos diretamente em YUV420p BT.709,
  evitando conversão desnecessária YUV -> RGB -> YUV no vídeo.
- Codifica vídeo em H.264 (libx264) com CRF 16, preset slow, perfil High,
  espaço de cor BT.709 e movflags +faststart.
- Mantém o áudio AAC original em stream copy (0% de recompressão/perda).

Uso rápido (edite o bloco CONFIGURACOES abaixo ou passe argumentos no terminal):
    python montar_video_mockup.py
    python montar_video_mockup.py --video "meu_video.mp4" --saida "saida.mp4"
    python montar_video_mockup.py --offset-y -0.15 --preview
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
from av.video.reformatter import ColorRange, Colorspace
from PIL import Image, ImageDraw

# =============================================================================
# CONFIGURAÇÕES PADRÃO (edite aqui para testar rapidamente)
# =============================================================================
MOCKUP_PADRAO = "Mock.jpg"
VIDEO_PADRAO = "WhatsApp Video 2026-11-21 at 12.28.34.mp4"
SAIDA_PADRAO = "video_teste_mockup.mp4"

# Resolução final:
#   "1080x1920" -> Full HD 9:16 (RECOMENDADO para Instagram Reels, evita reduzir o vídeo)
#   "mockup"    -> Mantém a resolução exata da imagem do mockup
RESOLUCAO_SAIDA = "1080x1920"

# Layout:
#   "instagram" -> canvas 1080x1920 montado pelo script: o cabeçalho do mockup (logo, nome, frase)
#                  vai logo acima de uma área de vídeo maior (AREA_INSTAGRAM). Recomendado para Reels.
#   "mockup"    -> usa o mockup inteiro como está e encaixa o vídeo no quadrado cinza.
LAYOUT = "instagram"

# Área do vídeo no layout "instagram" (x, y, largura, altura) no canvas 1080x1920.
# 900x1125 = 4:5 (logo fica dentro do recorte 3:4 da grade do perfil). O cabeçalho é posicionado automaticamente acima, com ESPACO_CABECALHO px de folga.
AREA_INSTAGRAM = (90, 620, 900, 1125)
ESPACO_CABECALHO = 36

# Como encaixar o vídeo na área do quadrado cinza:
#   "contain" -> Mantém o formato original do vídeo (retangular), menorzinho dentro da área cinza
#   "cover"   -> Recorta o vídeo para preencher 100% da área (padrão)
MODO_ENCAIXE = "cover"

# Escala extra do vídeo no modo "contain" (1.0 = cabe exatamente na área cinza;
# valores > 1.0, ex: 1.25, aumentam o vídeo para baixo aproveitando o espaço branco sem cobrir o texto)
ESCALA_VIDEO = 1.25

# Se True, mantém o quadrado cinza original visível atrás do vídeo no modo "contain"
MANTER_FUNDO_CINZA = False

# Deslocamento do recorte no modo "cover" (-1.0 a +1.0):
#   0.0 = centralizado
#   offset_y negativo = puxa o recorte para cima | positivo = puxa para baixo
#   offset_x negativo = puxa para a esquerda     | positivo = puxa para a direita
OFFSET_X = 0.0
OFFSET_Y = 0.0

# Arredondamento opcional dos cantos do vídeo (em pixels na resolução final; 0 = canto reto)
RAIO_BORDA = 0

# Qualidade H.264 (CRF menor = maior qualidade e arquivo maior; 14-18 = visualmente sem perdas)
CRF = 16
PRESET = "slow"  # ultrafast, fast, medium, slow, slower
# =============================================================================


@dataclass(frozen=True)
class Retangulo:
    x: int
    y: int
    w: int
    h: int


def _maior_segmento_continuo(mask_1d: np.ndarray) -> tuple[int, int]:
    """Retorna (inicio, fim_inclusivo) da maior sequência contínua True em um vetor booleano."""
    melhor_tam = 0
    melhor_ini = 0
    melhor_fim = -1
    atual_ini: int | None = None

    for i, val in enumerate(mask_1d):
        if val and atual_ini is None:
            atual_ini = i
        elif not val and atual_ini is not None:
            tam = i - atual_ini
            if tam > melhor_tam:
                melhor_tam, melhor_ini, melhor_fim = tam, atual_ini, i - 1
            atual_ini = None

    if atual_ini is not None:
        tam = len(mask_1d) - atual_ini
        if tam > melhor_tam:
            melhor_tam, melhor_ini, melhor_fim = tam, atual_ini, len(mask_1d) - 1

    if melhor_tam == 0:
        raise ValueError("Não foi possível detectar automaticamente o quadrado cinza no mockup.")
    return melhor_ini, melhor_fim


def detectar_quadrado_cinza(img_rgb: np.ndarray) -> tuple[Retangulo, tuple[int, int, int]]:
    """
    Detecta o maior retângulo cinza uniforme (acromático) no mockup e a cor de fundo da imagem.
    Retorna (Retangulo(x, y, w, h), cor_fundo_rgb).
    """
    h, w, _ = img_rgb.shape

    # Cor de fundo estimada pelas bordas externas do mockup
    bordas = np.concatenate(
        [img_rgb[0, :], img_rgb[-1, :], img_rgb[:, 0], img_rgb[:, -1]],
        axis=0,
    )
    cor_fundo = tuple(int(c) for c in np.median(bordas, axis=0))

    # Pixels cinza acromáticos (baixa saturação, luminosidade intermediária)
    max_c = img_rgb.max(axis=2).astype(np.int16)
    min_c = img_rgb.min(axis=2).astype(np.int16)
    chroma = max_c - min_c
    luma = img_rgb.mean(axis=2)
    mask_cinza = (chroma <= 15) & (luma >= 80) & (luma <= 240)

    # 1) Maior faixa vertical de linhas preenchidas com cinza
    linhas_cinza = mask_cinza.sum(axis=1) > (w * 0.25)
    y1, y2 = _maior_segmento_continuo(linhas_cinza)

    # 2) Maior faixa horizontal de colunas preenchidas com cinza dentro dessas linhas
    colunas_cinza = mask_cinza[y1 : y2 + 1, :].sum(axis=0) > ((y2 - y1 + 1) * 0.5)
    x1, x2 = _maior_segmento_continuo(colunas_cinza)

    # Valida uniformidade do miolo para garantir que achou o retângulo certo
    miolo = img_rgb[y1 + 4 : max(y1 + 5, y2 - 3), x1 + 4 : max(x1 + 5, x2 - 3)]
    if miolo.size == 0 or float(np.std(miolo)) > 25.0:
        raise ValueError(
            "A área detectada não parece um retângulo cinza uniforme. "
            "Verifique o mockup ou informe a área manualmente."
        )

    return Retangulo(x=x1, y=y1, w=x2 - x1 + 1, h=y2 - y1 + 1), cor_fundo


def _para_par(valor: float, minimo: int = 0, maximo: int | None = None) -> int:
    v = int(round(valor / 2.0)) * 2
    if v < minimo:
        v = minimo + (minimo % 2)
    if maximo is not None and v > maximo:
        v = maximo - (maximo % 2)
    return v


def preparar_mockup_canvas(
    caminho_mockup: Path,
    resolucao: str,
    manter_fundo_cinza: bool = False,
) -> tuple[Image.Image, Retangulo, tuple[int, int, int]]:
    """
    Abre o mockup, detecta o quadrado cinza, apaga a área cinza (incluindo anti-aliasing
    de borda) com a cor de fundo antes de escalar para evitar halos do filtro Lanczos
    (a menos que manter_fundo_cinza=True), e retorna (imagem_canvas_rgb, retangulo_cinza_par, cor_fundo_rgb).
    """
    img = Image.open(caminho_mockup).convert("RGB")
    arr = np.array(img)
    orig_h, orig_w, _ = arr.shape

    box_orig, cor_fundo = detectar_quadrado_cinza(arr)

    if manter_fundo_cinza:
        img_limpa = img
    else:
        # Apaga o quadrado cinza + 2px de margem anti-aliasing na imagem original antes do upscale
        # (assim o Lanczos não cria halo cinza/escuro nas bordas onde o vídeo vai encaixar)
        margem_limpeza = 2
        ly1 = max(0, box_orig.y - margem_limpeza)
        ly2 = min(orig_h, box_orig.y + box_orig.h + margem_limpeza)
        lx1 = max(0, box_orig.x - margem_limpeza)
        lx2 = min(orig_w, box_orig.x + box_orig.w + margem_limpeza)
        arr_limpo = arr.copy()
        arr_limpo[ly1:ly2, lx1:lx2] = cor_fundo
        img_limpa = Image.fromarray(arr_limpo, mode="RGB")

    if resolucao.lower() == "mockup":
        target_w = _para_par(orig_w, minimo=2)
        target_h = _para_par(orig_h, minimo=2)
    else:
        partes = resolucao.lower().split("x")
        if len(partes) != 2:
            raise ValueError(f"Resolução inválida: {resolucao!r}. Use ex: '1080x1920' ou 'mockup'.")
        target_w = _para_par(int(partes[0]), minimo=2)
        target_h = _para_par(int(partes[1]), minimo=2)

    # Escala preservando 100% a proporção do mockup (sem distorcer logo ou texto)
    escala = min(target_w / orig_w, target_h / orig_h)
    scaled_w = int(round(orig_w * escala))
    scaled_h = int(round(orig_h * escala))

    if (scaled_w, scaled_h) != (orig_w, orig_h):
        img_escalada = img_limpa.resize((scaled_w, scaled_h), resample=Image.Resampling.LANCZOS)
    else:
        img_escalada = img_limpa

    pad_x = (target_w - scaled_w) // 2
    pad_y = (target_h - scaled_h) // 2

    canvas = Image.new("RGB", (target_w, target_h), cor_fundo)
    canvas.paste(img_escalada, (pad_x, pad_y))

    # Mapeia as coordenadas do quadrado cinza para o canvas final (tudo múltiplo de 2 para YUV420p)
    raw_x = pad_x + box_orig.x * escala
    raw_y = pad_y + box_orig.y * escala
    raw_w = box_orig.w * escala
    raw_h = box_orig.h * escala

    # Se o quadrado original é praticamente 1:1 (diferença <= 2px), garante quadrado exato
    if abs(box_orig.w - box_orig.h) <= 2:
        lado = _para_par((raw_w + raw_h) / 2.0, minimo=2, maximo=min(target_w, target_h))
        box_w = lado
        box_h = lado
    else:
        box_w = _para_par(raw_w, minimo=2, maximo=target_w)
        box_h = _para_par(raw_h, minimo=2, maximo=target_h)

    centro_x = raw_x + raw_w / 2.0
    centro_y = raw_y + raw_h / 2.0
    box_x = _para_par(centro_x - box_w / 2.0, minimo=0, maximo=target_w - box_w)
    box_y = _para_par(centro_y - box_h / 2.0, minimo=0, maximo=target_h - box_h)

    return canvas, Retangulo(x=box_x, y=box_y, w=box_w, h=box_h), cor_fundo


def preparar_canvas_instagram(
    caminho_mockup: Path,
    area: tuple[int, int, int, int] = AREA_INSTAGRAM,
    espaco: int = ESPACO_CABECALHO,
    largura: int = 1080,
    altura: int = 1920,
) -> tuple[Image.Image, Retangulo, tuple[int, int, int]]:
    """
    Monta o canvas do layout "instagram": recorta o cabeçalho do mockup (tudo que há acima do
    quadrado cinza), escala para a largura do canvas e cola logo acima da área do vídeo.
    """
    img = Image.open(caminho_mockup).convert("RGB")
    arr = np.array(img)
    orig_w = arr.shape[1]
    box_orig, cor_fundo = detectar_quadrado_cinza(arr)

    conteudo = np.abs(arr[: box_orig.y].astype(np.int16) - np.array(cor_fundo, dtype=np.int16)).sum(axis=2) > 40
    linhas = np.where(conteudo.any(axis=1))[0]
    if linhas.size == 0:
        raise ValueError("Não achei cabeçalho (logo/texto) acima do quadrado cinza no mockup.")
    margem = 6  # pega o anti-aliasing das bordas do texto
    y0 = max(0, int(linhas[0]) - margem)
    y1 = min(box_orig.y, int(linhas[-1]) + 1 + margem)

    escala = largura / orig_w
    cabecalho = img.crop((0, y0, orig_w, y1))
    cabecalho = cabecalho.resize((largura, round((y1 - y0) * escala)), resample=Image.Resampling.LANCZOS)

    ax, ay, aw, ah = (_para_par(v) for v in area)
    topo = ay - espaco - cabecalho.height
    if topo < 0 or ay + ah > altura or ax + aw > largura:
        raise ValueError(f"AREA_INSTAGRAM {area} não cabe no canvas {largura}x{altura} com o cabeçalho.")

    canvas = Image.new("RGB", (largura, altura), cor_fundo)
    canvas.paste(cabecalho, (0, topo))
    return canvas, Retangulo(x=ax, y=ay, w=aw, h=ah), cor_fundo


def rgb_image_para_yuv420p_bt709(img_rgb: Image.Image) -> np.ndarray:
    """
    Converte uma imagem PIL RGB para array NumPy contínuo (H*3//2, W) em yuv420p BT.709 (TV range),
    usando libswscale do FFmpeg para correspondência exata com o vídeo.
    """
    arr_rgb = np.asarray(img_rgb, dtype=np.uint8)
    h, w, _ = arr_rgb.shape
    frame_rgb = av.VideoFrame.from_ndarray(arr_rgb, format="rgb24")

    graph = av.filter.Graph()
    buf = graph.add_buffer(width=w, height=h, format="rgb24", time_base="1/30")
    scale = graph.add(
        "scale",
        "out_color_matrix=bt709:out_range=tv:flags=lanczos+accurate_rnd+full_chroma_int",
    )
    fmt = graph.add("format", "yuv420p")
    sink = graph.add("buffersink")
    buf.link_to(scale)
    scale.link_to(fmt)
    fmt.link_to(sink)
    graph.configure()

    buf.push(frame_rgb)
    buf.push(None)
    yuv_frame = sink.pull()
    return extrair_yuv420p_continuo(yuv_frame)


def extrair_planos_yuv420p(frame: av.VideoFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extrai os planos Y, U, V respeitando o stride (line_size) do FFmpeg."""
    w, h = frame.width, frame.height
    y = np.frombuffer(frame.planes[0], dtype=np.uint8).reshape(h, frame.planes[0].line_size)[:, :w]
    u = np.frombuffer(frame.planes[1], dtype=np.uint8).reshape(h // 2, frame.planes[1].line_size)[
        :, : w // 2
    ]
    v = np.frombuffer(frame.planes[2], dtype=np.uint8).reshape(h // 2, frame.planes[2].line_size)[
        :, : w // 2
    ]
    return y, u, v


def extrair_yuv420p_continuo(frame: av.VideoFrame) -> np.ndarray:
    """Empacota os planos Y, U, V de um VideoFrame yuv420p no layout (H*3//2, W) do PyAV."""
    y, u, v = extrair_planos_yuv420p(frame)
    w, h = frame.width, frame.height
    out = np.empty((h * 3 // 2, w), dtype=np.uint8)
    out[:h, :] = y
    out[h : h + h // 4, :] = u.reshape(h // 4, w)
    out[h + h // 4 :, :] = v.reshape(h // 4, w)
    return out


def obter_rotacao_video(stream: av.video.stream.VideoStream) -> int:
    """Lê metadado de rotação (0, 90, 180, 270) se existir no stream de vídeo."""
    rot = stream.metadata.get("rotate")
    if rot is not None:
        try:
            return int(float(rot)) % 360
        except ValueError:
            pass
    return 0


def criar_mascaras_cantos_arredondados(
    w: int, h: int, raio: int
) -> tuple[np.ndarray, np.ndarray]:
    """Cria máscaras alpha anti-aliased (float32 [0..1]) para os planos Y e UV."""
    ss = 4  # supersampling 4x para borda perfeitamente suave
    mask_hi = Image.new("L", (w * ss, h * ss), 0)
    draw = ImageDraw.Draw(mask_hi)
    draw.rounded_rectangle(
        (0, 0, w * ss - 1, h * ss - 1),
        radius=raio * ss,
        fill=255,
    )
    mask_y = np.asarray(mask_hi.resize((w, h), resample=Image.Resampling.LANCZOS), dtype=np.float32) / 255.0
    mask_uv = (
        np.asarray(mask_hi.resize((w // 2, h // 2), resample=Image.Resampling.LANCZOS), dtype=np.float32)
        / 255.0
    )
    return mask_y, mask_uv


def construir_filter_graph_video(
    in_stream: av.video.stream.VideoStream,
    box: Retangulo,
    canvas_w: int,
    canvas_h: int,
    modo: str,
    escala_video: float,
    offset_x: float,
    offset_y: float,
) -> tuple[av.filter.Graph, av.filter.FilterContext, av.filter.FilterContext, Retangulo]:
    """
    Cria o grafo de filtros FFmpeg:
    rotação (se houver) -> crop (no modo cover) -> scale Lanczos (BT.709 TV) -> format yuv420p.
    Retorna também o Retangulo exato (dest_box, múltiplo de 2) onde o vídeo será colado no canvas.
    """
    rotacao = obter_rotacao_video(in_stream)
    src_w = in_stream.width
    src_h = in_stream.height

    graph = av.filter.Graph()
    buf = graph.add_buffer(template=in_stream)
    atual = buf

    # 1) Corrige rotação se o vídeo veio com tag rotate
    if rotacao == 90:
        f_rot = graph.add("transpose", "clock")
        atual.link_to(f_rot)
        atual = f_rot
        src_w, src_h = src_h, src_w
    elif rotacao == 270:
        f_rot = graph.add("transpose", "cclock")
        atual.link_to(f_rot)
        atual = f_rot
        src_w, src_h = src_h, src_w
    elif rotacao == 180:
        f_h = graph.add("hflip")
        f_v = graph.add("vflip")
        atual.link_to(f_h)
        f_h.link_to(f_v)
        atual = f_v

    offset_x = max(-1.0, min(1.0, offset_x))
    offset_y = max(-1.0, min(1.0, offset_y))

    flags_scale = "lanczos+accurate_rnd+full_chroma_int+full_chroma_inp"
    target_ar = box.w / box.h
    src_ar = src_w / src_h

    if modo == "cover":
        # Recorta o vídeo exatamente na proporção do quadrado cinza antes de escalar
        if src_ar > target_ar:
            # Vídeo mais largo que o quadrado -> corta nas laterais
            crop_h = src_h - (src_h % 2)
            crop_w = _para_par(crop_h * target_ar, minimo=2, maximo=src_w)
        else:
            # Vídeo mais alto que o quadrado (ex: 9:16 em quadrado 1:1) -> corta em cima e embaixo
            crop_w = src_w - (src_w % 2)
            crop_h = _para_par(crop_w / target_ar, minimo=2, maximo=src_h)

        sobra_x = src_w - crop_w
        sobra_y = src_h - crop_h
        crop_x = _para_par((sobra_x / 2.0) * (1.0 + offset_x), minimo=0, maximo=sobra_x)
        crop_y = _para_par((sobra_y / 2.0) * (1.0 + offset_y), minimo=0, maximo=sobra_y)

        dest_box = box
        f_crop = graph.add("crop", f"{crop_w}:{crop_h}:{crop_x}:{crop_y}")
        f_scale = graph.add(
            "scale",
            f"w={dest_box.w}:h={dest_box.h}:flags={flags_scale}:out_color_matrix=bt709:out_range=tv",
        )
        atual.link_to(f_crop)
        f_crop.link_to(f_scale)
        atual = f_scale
    elif modo == "contain":
        # Mantém o formato original do vídeo (retangular, sem cortar) cabendo dentro da área cinza
        if escala_video <= 0:
            raise ValueError(f"escala_video deve ser > 0, recebido: {escala_video}")

        if src_ar > target_ar:
            base_w = box.w * escala_video
            base_h = base_w / src_ar
        else:
            base_h = box.h * escala_video
            base_w = base_h * src_ar

        # Limita ao canvas preservando o texto acima da área cinza
        max_w = canvas_w
        max_h = canvas_h - box.y if base_h > box.h else box.h
        fator_limite = min(1.0, max_w / base_w, max_h / base_h)
        fit_w = _para_par(base_w * fator_limite, minimo=2, maximo=max_w)
        fit_h = _para_par(base_h * fator_limite, minimo=2, maximo=max_h)

        # Centraliza na área cinza; se escala > 1 ultrapassar a altura da caixa, mantém o topo preso em box.y
        dest_x = _para_par(box.x + (box.w - fit_w) / 2.0, minimo=0, maximo=canvas_w - fit_w)
        if fit_h <= box.h:
            dest_y = _para_par(box.y + (box.h - fit_h) / 2.0, minimo=0, maximo=canvas_h - fit_h)
        else:
            dest_y = _para_par(box.y, minimo=0, maximo=canvas_h - fit_h)

        dest_box = Retangulo(x=dest_x, y=dest_y, w=fit_w, h=fit_h)
        f_scale = graph.add(
            "scale",
            f"w={dest_box.w}:h={dest_box.h}:flags={flags_scale}:out_color_matrix=bt709:out_range=tv",
        )
        atual.link_to(f_scale)
        atual = f_scale
    else:
        raise ValueError(f"Modo inválido: {modo!r}. Use 'cover' ou 'contain'.")

    f_sar = graph.add("setsar", "1")
    f_fmt = graph.add("format", "yuv420p")
    sink = graph.add("buffersink")

    atual.link_to(f_sar)
    f_sar.link_to(f_fmt)
    f_fmt.link_to(sink)
    graph.configure()

    return graph, buf, sink, dest_box


def compor_frame_no_canvas(
    canvas_yuv: np.ndarray,
    base_box_y: np.ndarray | None,
    base_box_u: np.ndarray | None,
    base_box_v: np.ndarray | None,
    canvas_w: int,
    canvas_h: int,
    box: Retangulo,
    video_frame: av.VideoFrame,
    mask_y: np.ndarray | None,
    mask_uv: np.ndarray | None,
) -> av.VideoFrame:
    """
    Cola os planos Y, U, V do vídeo dentro do canvas YUV420p do mockup e devolve um novo VideoFrame.
    """
    vid_y, vid_u, vid_v = extrair_planos_yuv420p(video_frame)

    # Views dos planos dentro do buffer contínuo (H*3//2, W)
    dst_y = canvas_yuv[:canvas_h, :]
    dst_u = canvas_yuv[canvas_h : canvas_h + canvas_h // 4, :].reshape(canvas_h // 2, canvas_w // 2)
    dst_v = canvas_yuv[canvas_h + canvas_h // 4 :, :].reshape(canvas_h // 2, canvas_w // 2)

    y0, y1 = box.y, box.y + box.h
    x0, x1 = box.x, box.x + box.w
    uy0, uy1 = y0 // 2, y1 // 2
    ux0, ux1 = x0 // 2, x1 // 2

    if mask_y is None or mask_uv is None:
        # Cópia direta bit-exata nos planos Y, U, V (sem perda nenhuma)
        dst_y[y0:y1, x0:x1] = vid_y
        dst_u[uy0:uy1, ux0:ux1] = vid_u
        dst_v[uy0:uy1, ux0:ux1] = vid_v
    else:
        assert base_box_y is not None and base_box_u is not None and base_box_v is not None
        dst_y[y0:y1, x0:x1] = np.clip(
            base_box_y * (1.0 - mask_y) + vid_y * mask_y + 0.5, 0, 255
        ).astype(np.uint8)
        dst_u[uy0:uy1, ux0:ux1] = np.clip(
            base_box_u * (1.0 - mask_uv) + vid_u * mask_uv + 0.5, 0, 255
        ).astype(np.uint8)
        dst_v[uy0:uy1, ux0:ux1] = np.clip(
            base_box_v * (1.0 - mask_uv) + vid_v * mask_uv + 0.5, 0, 255
        ).astype(np.uint8)

    out_frame = av.VideoFrame.from_ndarray(canvas_yuv, format="yuv420p")
    out_frame.colorspace = Colorspace.ITU709
    out_frame.color_range = ColorRange.MPEG
    return out_frame


def processar_video(
    caminho_mockup: Path,
    caminho_video: Path,
    caminho_saida: Path,
    resolucao: str = RESOLUCAO_SAIDA,
    modo: str = MODO_ENCAIXE,
    escala_video: float = ESCALA_VIDEO,
    manter_fundo_cinza: bool = MANTER_FUNDO_CINZA,
    offset_x: float = OFFSET_X,
    offset_y: float = OFFSET_Y,
    raio_borda: int = RAIO_BORDA,
    crf: int = CRF,
    preset: str = PRESET,
    apenas_preview: bool = False,
    layout: str = LAYOUT,
    area_instagram: tuple[int, int, int, int] = AREA_INSTAGRAM,
    max_kbps: int | None = None,
) -> None:
    inicio = time.perf_counter()

    if not caminho_mockup.is_file():
        raise FileNotFoundError(f"Mockup não encontrado: {caminho_mockup}")
    if not caminho_video.is_file():
        raise FileNotFoundError(f"Vídeo não encontrado: {caminho_video}")

    print(f"[1/3] Preparando mockup: {caminho_mockup.name} (layout {layout})")
    if layout == "instagram":
        canvas_img, box_cinza, _cor_fundo = preparar_canvas_instagram(caminho_mockup, area_instagram)
    elif layout == "mockup":
        canvas_img, box_cinza, _cor_fundo = preparar_mockup_canvas(
            caminho_mockup,
            resolucao=resolucao,
            manter_fundo_cinza=manter_fundo_cinza,
        )
    else:
        raise ValueError(f"Layout inválido: {layout!r}. Use 'instagram' ou 'mockup'.")
    canvas_w, canvas_h = canvas_img.size
    canvas_yuv = rgb_image_para_yuv420p_bt709(canvas_img)

    with av.open(str(caminho_video)) as inp:
        if not inp.streams.video:
            raise ValueError(f"Nenhum stream de vídeo encontrado em: {caminho_video}")

        in_vs = inp.streams.video[0]
        in_as = inp.streams.audio[0] if inp.streams.audio else None
        in_vs.thread_type = "AUTO"

        fps = in_vs.base_rate or in_vs.average_rate or Fraction(30, 1)
        total_frames = in_vs.frames or 0

        _graph, buf_in, buf_out, dest_box = construir_filter_graph_video(
            in_stream=in_vs,
            box=box_cinza,
            canvas_w=canvas_w,
            canvas_h=canvas_h,
            modo=modo,
            escala_video=escala_video,
            offset_x=offset_x,
            offset_y=offset_y,
        )

        print(
            f"      Canvas final: {canvas_w}x{canvas_h} | "
            f"Área cinza: x={box_cinza.x}, y={box_cinza.y}, {box_cinza.w}x{box_cinza.h}"
        )
        print(
            f"      Vídeo posicionado ({modo}): x={dest_box.x}, y={dest_box.y}, "
            f"{dest_box.w}x{dest_box.h}"
        )
        print(
            f"[2/3] Vídeo bruto: {in_vs.width}x{in_vs.height} | "
            f"{float(fps):.2f} fps | codec={in_vs.codec_context.name} | "
            f"áudio={in_as.codec_context.name if in_as else 'sem áudio'}"
        )

        mask_y = mask_uv = None
        base_box_y = base_box_u = base_box_v = None
        if raio_borda > 0:
            mask_y, mask_uv = criar_mascaras_cantos_arredondados(dest_box.w, dest_box.h, raio_borda)
            base_y = canvas_yuv[:canvas_h, :]
            base_u = canvas_yuv[canvas_h : canvas_h + canvas_h // 4, :].reshape(canvas_h // 2, canvas_w // 2)
            base_v = canvas_yuv[canvas_h + canvas_h // 4 :, :].reshape(canvas_h // 2, canvas_w // 2)
            base_box_y = base_y[dest_box.y : dest_box.y + dest_box.h, dest_box.x : dest_box.x + dest_box.w].astype(
                np.float32
            )
            base_box_u = base_u[
                dest_box.y // 2 : (dest_box.y + dest_box.h) // 2,
                dest_box.x // 2 : (dest_box.x + dest_box.w) // 2,
            ].astype(np.float32)
            base_box_v = base_v[
                dest_box.y // 2 : (dest_box.y + dest_box.h) // 2,
                dest_box.x // 2 : (dest_box.x + dest_box.w) // 2,
            ].astype(np.float32)

        # Modo preview rápido: salva apenas um PNG do frame central para conferir o enquadramento
        if apenas_preview:
            alvo = max(0, total_frames // 2)
            preview_path = caminho_saida.with_suffix(".png")
            for idx, frame in enumerate(inp.decode(in_vs)):
                if idx < alvo:
                    continue
                buf_in.push(frame)
                buf_in.push(None)
                f_filt = buf_out.pull()
                out_f = compor_frame_no_canvas(
                    canvas_yuv, base_box_y, base_box_u, base_box_v,
                    canvas_w, canvas_h, dest_box, f_filt, mask_y, mask_uv,
                )
                out_f.to_image().save(preview_path)
                print(f"[OK] Preview salvo em: {preview_path}")
                return
            raise RuntimeError("Não foi possível ler frames do vídeo para gerar preview.")

        caminho_saida.parent.mkdir(parents=True, exist_ok=True)
        print(f"[3/3] Renderizando (libx264 CRF={crf}, preset={preset}, áudio sem perda)...")

        with av.open(str(caminho_saida), mode="w", options={"movflags": "+faststart"}) as out:
            # Configura stream de vídeo em alta qualidade (padrão Instagram Reels)
            out_vs = out.add_stream("libx264", rate=fps)
            out_vs.width = canvas_w
            out_vs.height = canvas_h
            out_vs.pix_fmt = "yuv420p"
            out_vs.time_base = in_vs.time_base
            out_vs.codec_context.colorspace = Colorspace.ITU709
            out_vs.codec_context.color_primaries = 1  # bt709
            out_vs.codec_context.color_trc = 1        # bt709
            out_vs.codec_context.color_range = ColorRange.MPEG
            opcoes_x264 = {
                "crf": str(crf),
                "preset": preset,
                "profile": "high",
                "level": "4.1",
                "x264-params": "colorprim=bt709:transfer=bt709:colormatrix=bt709:fullrange=off",
            }
            if max_kbps:
                # Teto de bitrate (VBV): o CRF manda, mas nunca passa disso -> tamanho previsível
                opcoes_x264.update(maxrate=f"{max_kbps}k", bufsize=f"{max_kbps * 2}k")
            out_vs.codec_context.options = opcoes_x264

            # Configura áudio: stream copy direto se já for AAC, senão re-encoda para AAC 192k
            out_as = None
            copiar_audio = False
            audio_resampler = None
            if in_as is not None:
                if in_as.codec_context.name == "aac":
                    out_as = out.add_stream_from_template(in_as)
                    copiar_audio = True
                else:
                    out_as = out.add_stream("aac", rate=in_as.rate or 44100)
                    out_as.bit_rate = 192_000
                    audio_resampler = av.AudioResampler(
                        format="fltp",
                        layout=in_as.layout.name or "stereo",
                        rate=in_as.rate or 44100,
                    )

            streams_demux = [in_vs] + ([in_as] if in_as is not None else [])
            frames_processados = 0
            ultimo_pct = -1

            def _drenar_filtro_e_encodar() -> None:
                nonlocal frames_processados, ultimo_pct
                while True:
                    try:
                        f_filt = buf_out.pull()
                    except (av.BlockingIOError, av.EOFError):
                        break

                    out_frame = compor_frame_no_canvas(
                        canvas_yuv, base_box_y, base_box_u, base_box_v,
                        canvas_w, canvas_h, dest_box, f_filt, mask_y, mask_uv,
                    )
                    out_frame.pts = f_filt.pts
                    out_frame.time_base = f_filt.time_base

                    for pkt_out in out_vs.encode(out_frame):
                        out.mux(pkt_out)

                    frames_processados += 1
                    if total_frames > 0:
                        pct = int(frames_processados * 100 / total_frames)
                        if pct >= ultimo_pct + 10 or frames_processados == total_frames:
                            ultimo_pct = pct
                            print(f"      Progresso: {frames_processados}/{total_frames} frames ({pct}%)")

            for packet in inp.demux(streams_demux):
                if packet.dts is None:
                    continue

                if packet.stream == in_vs:
                    for frame in packet.decode():
                        buf_in.push(frame)
                        _drenar_filtro_e_encodar()
                elif out_as is not None and packet.stream == in_as:
                    if copiar_audio:
                        # Stream copy: zero perda de qualidade no áudio original
                        packet.stream = out_as
                        out.mux(packet)
                    else:
                        assert audio_resampler is not None
                        for a_frame in packet.decode():
                            for resampled in audio_resampler.resample(a_frame):
                                for pkt_a in out_as.encode(resampled):
                                    out.mux(pkt_a)

            # Flush do decoder de vídeo -> filter graph -> encoder
            for frame in in_vs.codec_context.decode(None):
                buf_in.push(frame)
                _drenar_filtro_e_encodar()

            buf_in.push(None)
            _drenar_filtro_e_encodar()

            for pkt_out in out_vs.encode(None):
                out.mux(pkt_out)

            if out_as is not None and not copiar_audio:
                assert audio_resampler is not None
                for resampled in audio_resampler.resample(None):
                    for pkt_a in out_as.encode(resampled):
                        out.mux(pkt_a)
                for pkt_a in out_as.encode(None):
                    out.mux(pkt_a)

    duracao = time.perf_counter() - inicio
    tam_mb = caminho_saida.stat().st_size / (1024 * 1024)
    print(f"\n[OK] Vídeo salvo com sucesso: {caminho_saida}")
    print(f"     Frames: {frames_processados} | Tamanho: {tam_mb:.2f} MB | Tempo: {duracao:.1f}s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Encaixa um vídeo bruto exatamente sobre o quadrado cinza do mockup sem perder qualidade."
    )
    parser.add_argument("--mockup", type=Path, default=Path(MOCKUP_PADRAO), help="Caminho da imagem de mockup.")
    parser.add_argument("--video", type=Path, default=Path(VIDEO_PADRAO), help="Caminho do vídeo bruto.")
    parser.add_argument("--saida", type=Path, default=Path(SAIDA_PADRAO), help="Caminho do vídeo de saída (.mp4).")
    parser.add_argument(
        "--resolucao",
        default=RESOLUCAO_SAIDA,
        help="Resolução final, ex: '1080x1920' (padrão Reels) ou 'mockup' (resolução original da imagem).",
    )
    parser.add_argument(
        "--layout",
        choices=("instagram", "mockup"),
        default=LAYOUT,
        help="'instagram' = cabeçalho do mockup + vídeo grande (AREA_INSTAGRAM); 'mockup' = mockup original.",
    )
    parser.add_argument(
        "--area",
        type=lambda s: tuple(int(v) for v in s.lower().replace("x", ",").split(",")),
        default=AREA_INSTAGRAM,
        help="Área do vídeo no layout instagram: 'x,y,largura,altura' (padrão 90,620,900,1125).",
    )
    parser.add_argument(
        "--modo",
        choices=("contain", "cover"),
        default=MODO_ENCAIXE,
        help="'contain' mantém o formato original menorzinho dentro da área cinza; 'cover' recorta para preencher o quadrado.",
    )
    parser.add_argument(
        "--escala",
        type=float,
        default=ESCALA_VIDEO,
        help="Escala do vídeo no modo 'contain' (1.0 = cabe exatamente na área cinza; ex: 1.2 = um pouco maior para baixo).",
    )
    parser.add_argument(
        "--manter-cinza",
        action="store_true",
        default=MANTER_FUNDO_CINZA,
        help="Mantém o quadrado cinza visível atrás do vídeo no modo 'contain' (por padrão o fundo fica branco limpo).",
    )
    parser.add_argument(
        "--offset-x",
        type=float,
        default=OFFSET_X,
        help="Deslocamento horizontal do recorte no modo 'cover' (-1.0 esquerda, 0.0 centro, +1.0 direita).",
    )
    parser.add_argument(
        "--offset-y",
        type=float,
        default=OFFSET_Y,
        help="Deslocamento vertical do recorte no modo 'cover' (-1.0 topo, 0.0 centro, +1.0 base).",
    )
    parser.add_argument(
        "--raio-borda",
        type=int,
        default=RAIO_BORDA,
        help="Raio de arredondamento dos cantos do vídeo em pixels (0 = canto reto).",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=CRF,
        help="Fator de qualidade H.264 (menor = mais qualidade; padrão 16).",
    )
    parser.add_argument(
        "--preset",
        default=PRESET,
        help="Velocidade x eficiência do libx264 (fast, medium, slow, slower; padrão slow).",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Gera apenas uma imagem PNG do meio do vídeo para testar o enquadramento rapidamente.",
    )
    return parser.parse_args()


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    processar_video(
        caminho_mockup=args.mockup,
        caminho_video=args.video,
        caminho_saida=args.saida,
        resolucao=args.resolucao,
        modo=args.modo,
        escala_video=args.escala,
        manter_fundo_cinza=args.manter_cinza,
        offset_x=args.offset_x,
        offset_y=args.offset_y,
        raio_borda=args.raio_borda,
        crf=args.crf,
        preset=args.preset,
        apenas_preview=args.preview,
        layout=args.layout,
        area_instagram=args.area,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
