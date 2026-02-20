# Asset Compressor

A web app for the Creative/Design team to compress assets before handing them off to engineering. Supports **MP4 videos**, **Lottie animations**, and **images** (PNG, JPG, WebP).

Upload a file, pick a compression preset, preview the result side-by-side, and download.

## Quick start (local)

**Prerequisites:** Python 3.10+, FFmpeg, and Pillow.

```bash
# Install FFmpeg (macOS)
brew install ffmpeg

# Install Python dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

Open [http://localhost:5050](http://localhost:5050).

## Quick start (Docker)

```bash
docker compose up --build
```

Open [http://localhost:5050](http://localhost:5050).

## Supported file types

| Type | Formats | How it compresses |
|---|---|---|
| Video | MP4, MOV, WebM | FFmpeg H.264 re-encode with CRF control, FPS cap, resolution scaling |
| Lottie | .lottie | Recompresses embedded raster frames to WebP, reduces FPS, optional scaling |
| Image | PNG, JPG, WebP | Format conversion (PNG->JPEG), quality control, resolution scaling |

## Compression presets

### Video

| Preset | CRF | Typical reduction | Best for |
|---|---|---|---|
| Light | 23 | 40-60% | Hero placements, Home screen |
| Balanced | 27 | 65-80% | Most animations |
| Maximum | 31 | 80-90% | Small placements, deep screens |

### Lottie

| Preset | WebP quality | FPS | Scale | Best for |
|---|---|---|---|---|
| Light | 80 | Original | 1.0 | High-visibility placements |
| Balanced | 65 | 15 | 1.0 | Most animations |
| Maximum | 55 | 12 | 0.75 | Deep screens, small placements |

### Image

| Preset | Quality | Typical reduction | Best for |
|---|---|---|---|
| Light | 95 | 30-50% | Hero images, high-traffic screens |
| Balanced | 90 | 50-70% | Most images |
| Maximum | 85 | 65-80% | Thumbnails, deep screens |

## Deployment options

### Option A: Docker on any VM/server

```bash
git clone <this-repo>
cd mp4-compressor
docker compose up -d --build
```

For HTTPS, put it behind Caddy (auto-TLS) or nginx + Let's Encrypt.

### Option B: Google Cloud Run (serverless, scales to zero)

```bash
gcloud builds submit --tag gcr.io/YOUR_PROJECT/asset-compressor
gcloud run deploy asset-compressor \
  --image gcr.io/YOUR_PROJECT/asset-compressor \
  --port 5050 \
  --memory 1Gi \
  --cpu 2 \
  --timeout 120 \
  --allow-unauthenticated
```

### Option C: Railway / Render / Fly.io (one-click PaaS)

These auto-detect the Dockerfile and deploy with minimal config:

- **Railway**: `railway up`
- **Render**: Connect Git repo, point to Dockerfile, set port 5050
- **Fly.io**: `fly launch` then `fly deploy`
