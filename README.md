# shoppe-videos-clone

Webhook que recebe o link de um vídeo (inclusive link de compartilhamento da Shopee), baixa a versão
sem marca d'água, monta no layout do mockup para Instagram Reels (1080x1920), remove os metadados
e envia ao Telegram.

## API

```
POST /webhook
Header: X-Webhook-Secret: <WEBHOOK_SECRET>   (se definido)
{"url": "https://br.shp.ee/xxxx", "legenda": "Texto do post"}
-> 202 {"job": "ab12cd34", "status_url": "/status/ab12cd34"}

GET /status/<job>  -> {"status": "na_fila|baixando|editando|limpando|enviando|ok|erro", ...}
GET /health        -> {"ok": true, "na_fila": 0}
```

Os vídeos são processados em segundo plano, um por vez. Um vídeo de ~40s leva cerca de 1 minuto
para editar, mais o tempo de upload ao Telegram.

## Variáveis de ambiente

| Variável | Obrigatória | Descrição |
|---|---|---|
| `TELEGRAM_BOT_TOKEN` | sim | Token do bot (@BotFather) |
| `TELEGRAM_CHAT_ID` | sim | Chat/grupo de destino (grupos começam com `-`) |
| `WEBHOOK_SECRET` | recomendado | Exige o header `X-Webhook-Secret` |
| `PORT` | não | Porta HTTP (padrão `8000`) |

## Deploy no Easypanel

1. **Create Service → App**.
2. **Source → GitHub**: este repositório, branch `main`. **Build → Dockerfile** (caminho `Dockerfile`).
3. **Environment**: cole as variáveis acima.
4. **Domains**: adicione o domínio e coloque **Port = 8000** (porta do container para onde o Traefik encaminha).
5. **Deploy**. Teste com `GET https://<seu-domínio>/health`.

## Rodar local

```bash
cp .env.example .env   # preencha o token
pip install -r requirements.txt
python webhook_server.py
```

Ou com Docker:

```bash
docker build -t shoppe-videos .
docker run --env-file .env -p 8000:8000 shoppe-videos
```

## Ajustes de layout

Em `montar_video_mockup.py`: `AREA_INSTAGRAM` (área do vídeo, padrão 900x1125 em y=620),
`OFFSET_Y` (altura do corte) e `LAYOUT` (`instagram` ou `mockup`). O cabeçalho é recortado de `Mock.jpg`;
troque a imagem para mudar logo/texto.
