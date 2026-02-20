import io
import json
import math
import os
import subprocess
import tempfile
import uuid
import zipfile
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file
from PIL import Image

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024

UPLOAD_DIR = Path(tempfile.gettempdir()) / "asset_compressor"
UPLOAD_DIR.mkdir(exist_ok=True)

ASSET_TYPES = {
    ".mp4": "video",
    ".mov": "video",
    ".webm": "video",
    ".lottie": "lottie",
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".webp": "image",
}

VIDEO_PRESETS = {
    "light":      {"crf": 23, "preset": "slow"},
    "balanced":   {"crf": 27, "preset": "slow"},
    "aggressive": {"crf": 31, "preset": "slow"},
}

LOTTIE_PRESETS = {
    "light":      {"quality": 80, "fps": None, "scale": 1.0},
    "balanced":   {"quality": 65, "fps": 15,   "scale": 1.0},
    "aggressive": {"quality": 55, "fps": 12,   "scale": 0.75},
}

IMAGE_PRESETS = {
    "light":      {"quality": 95, "fmt": "jpg"},
    "balanced":   {"quality": 90, "fmt": "jpg"},
    "aggressive": {"quality": 85, "fmt": "jpg"},
}


# ---------------------------------------------------------------------------
# Video helpers
# ---------------------------------------------------------------------------

def analyze_video(file_path: str) -> dict:
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_format", "-show_streams", file_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)

    video_stream = next(
        (s for s in data.get("streams", []) if s["codec_type"] == "video"), None
    )
    audio_stream = next(
        (s for s in data.get("streams", []) if s["codec_type"] == "audio"), None
    )

    file_size = int(data["format"].get("size", 0))
    duration = float(data["format"].get("duration", 0))

    info = {
        "file_size": file_size,
        "duration": round(duration, 2),
        "has_audio": audio_stream is not None,
    }

    if video_stream:
        info["width"] = int(video_stream.get("width", 0))
        info["height"] = int(video_stream.get("height", 0))
        info["codec"] = video_stream.get("codec_name", "unknown")
        fps_parts = video_stream.get("r_frame_rate", "0/1").split("/")
        info["fps"] = round(int(fps_parts[0]) / max(int(fps_parts[1]), 1), 2)

    return info


def compress_video(
    input_path: str, output_path: str, *,
    crf: int = 27, preset: str = "slow",
    scale: float | None = None, strip_audio: bool = True, max_fps: int | None = None,
) -> dict:
    vf_filters = []
    if scale and scale < 1.0:
        vf_filters.append(f"scale=iw*{scale}:ih*{scale}:flags=lanczos")
    if max_fps:
        vf_filters.append(f"fps={max_fps}")

    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-c:v", "libx264", "-crf", str(crf), "-preset", preset,
        "-pix_fmt", "yuv420p",
    ]
    if vf_filters:
        cmd.extend(["-vf", ",".join(vf_filters)])
    if strip_audio:
        cmd.append("-an")
    else:
        cmd.extend(["-c:a", "aac", "-b:a", "128k"])
    cmd.extend(["-movflags", "+faststart", output_path])

    subprocess.run(cmd, capture_output=True, text=True, check=True)

    orig = os.path.getsize(input_path)
    new = os.path.getsize(output_path)
    return {"original_size": orig, "compressed_size": new, "reduction_pct": round((1 - new / orig) * 100, 1)}


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def analyze_image(file_path: str) -> dict:
    p = Path(file_path)
    file_size = p.stat().st_size
    img = Image.open(p)
    fmt = img.format or p.suffix.lstrip(".").upper()

    has_alpha = img.mode in ("RGBA", "LA", "PA")
    alpha_used = False
    if has_alpha:
        alpha_used = img.getchannel("A").getextrema()[0] < 255

    estimates = {}
    if not alpha_used:
        for q in [85, 90, 95]:
            buf = io.BytesIO()
            img.convert("RGB").save(buf, "JPEG", quality=q, optimize=True)
            estimates[f"jpg_q{q}"] = buf.tell()
    for q in [70, 80, 90]:
        buf = io.BytesIO()
        img.save(buf, "WEBP", quality=q)
        estimates[f"webp_q{q}"] = buf.tell()

    return {
        "file_size": file_size,
        "width": img.width,
        "height": img.height,
        "format": fmt,
        "mode": img.mode,
        "has_alpha": has_alpha,
        "alpha_used": alpha_used,
        "estimates": estimates,
    }


def compress_image(
    input_path: str, output_path: str, *,
    quality: int = 90, fmt: str = "jpg", scale: float = 1.0,
) -> dict:
    img = Image.open(input_path)

    if scale != 1.0:
        img = img.resize(
            (math.floor(img.width * scale), math.floor(img.height * scale)),
            Image.LANCZOS,
        )

    fmt_upper = fmt.upper()
    if fmt_upper in ("JPG", "JPEG"):
        img.convert("RGB").save(output_path, "JPEG", quality=quality, optimize=True)
    elif fmt_upper == "WEBP":
        img.save(output_path, "WEBP", quality=quality)
    elif fmt_upper == "PNG":
        img.save(output_path, "PNG", optimize=True)
    else:
        raise ValueError(f"Unsupported format: {fmt}")

    orig = os.path.getsize(input_path)
    new = os.path.getsize(output_path)
    return {"original_size": orig, "compressed_size": new, "reduction_pct": round((1 - new / orig) * 100, 1)}


# ---------------------------------------------------------------------------
# Lottie helpers
# ---------------------------------------------------------------------------

def analyze_lottie(file_path: str) -> dict:
    p = Path(file_path)
    file_size = p.stat().st_size

    with zipfile.ZipFile(p, "r") as zf:
        names = zf.namelist()
        image_names = [n for n in names if n.startswith("images/")]
        image_total = sum(zf.getinfo(n).file_size for n in image_names)

        with zf.open("animations/animation.json") as f:
            anim = json.load(f)

        img_w, img_h = anim.get("w", 0), anim.get("h", 0)
        estimates = {}

        if image_names:
            mid = image_names[len(image_names) // 2]
            with zf.open(mid) as img_file:
                img = Image.open(img_file)
                img.load()
                img_w, img_h = img.size

            mid_orig = zf.getinfo(mid).file_size
            for q in [50, 65, 80]:
                buf = io.BytesIO()
                img.save(buf, "WEBP", quality=q)
                estimates[f"q{q}"] = round(buf.tell() / mid_orig, 3)

    fps = anim["fr"]
    frames = int(anim["op"] - anim["ip"])
    duration = frames / fps if fps else 0
    is_image_sequence = len(image_names) > 1

    return {
        "file_size": file_size,
        "canvas_w": anim.get("w"),
        "canvas_h": anim.get("h"),
        "image_w": img_w,
        "image_h": img_h,
        "fps": fps,
        "frames": frames,
        "duration": round(duration, 2),
        "image_count": len(image_names),
        "image_total_bytes": image_total,
        "is_image_sequence": is_image_sequence,
        "compression_ratios": estimates,
    }


def compress_lottie(
    input_path: str, output_path: str, *,
    quality: int = 65, target_fps: int | None = None, scale: float = 1.0,
) -> dict:
    input_path = Path(input_path)
    orig_size = input_path.stat().st_size

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        with zipfile.ZipFile(input_path, "r") as zf:
            zf.extractall(tmpdir)

        anim_path = tmpdir / "animations" / "animation.json"
        with open(anim_path) as f:
            anim = json.load(f)

        orig_fps = anim["fr"]
        assets = anim.get("assets", [])
        assets_by_id = {a["id"]: a for a in assets}
        orig_layers = anim.get("layers", [])
        orig_asset_uri_prefix = assets[0]["u"] if assets else "/images/"

        keep_every_n = 1
        new_fps = orig_fps
        if target_fps and target_fps < orig_fps:
            keep_every_n = round(orig_fps / target_fps)
            new_fps = orig_fps / keep_every_n

        image_layers = [l for l in orig_layers if l.get("ty") == 2 and "refId" in l]
        non_image_layers = [l for l in orig_layers if l.get("ty") != 2 or "refId" not in l]

        kept_image_layers = [l for i, l in enumerate(image_layers) if i % keep_every_n == 0]

        images_dir = tmpdir / "images"
        for f in images_dir.iterdir():
            f.unlink()

        for new_idx, layer in enumerate(kept_image_layers, start=1):
            old_asset = assets_by_id[str(layer["refId"])]
            with zipfile.ZipFile(input_path, "r") as zf:
                with zf.open(f"images/{old_asset['p']}") as img_file:
                    img = Image.open(img_file)
                    img.load()

            if scale != 1.0:
                img = img.resize(
                    (math.floor(img.width * scale), math.floor(img.height * scale)),
                    Image.LANCZOS,
                )
            img.save(images_dir / f"{new_idx}.webp", "WEBP", quality=quality)

        new_frame_count = len(kept_image_layers)

        new_assets = []
        for new_idx, layer in enumerate(kept_image_layers, start=1):
            old_asset = assets_by_id[str(layer["refId"])].copy()
            old_asset["id"] = str(new_idx)
            old_asset["p"] = f"{new_idx}.webp"
            old_asset["u"] = orig_asset_uri_prefix
            if scale != 1.0:
                old_asset["w"] = math.floor(old_asset["w"] * scale)
                old_asset["h"] = math.floor(old_asset["h"] * scale)
            new_assets.append(old_asset)

        new_layers = []
        for new_idx, layer in enumerate(kept_image_layers):
            nl = layer.copy()
            nl["refId"] = str(new_idx + 1)
            nl["ip"] = new_idx
            nl["op"] = new_idx + 1
            new_layers.append(nl)
        for layer in non_image_layers:
            adj = layer.copy()
            adj["ip"] = 0
            adj["op"] = new_frame_count
            new_layers.append(adj)

        anim["fr"] = new_fps
        anim["ip"] = 0
        anim["op"] = new_frame_count
        anim["assets"] = new_assets
        anim["layers"] = new_layers
        if scale != 1.0:
            anim["w"] = math.floor(anim["w"] * scale)
            anim["h"] = math.floor(anim["h"] * scale)

        with open(anim_path, "w") as f:
            json.dump(anim, f, separators=(",", ":"))

        with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(tmpdir):
                for file in files:
                    fp = Path(root) / file
                    zf.write(fp, fp.relative_to(tmpdir))

    new_size = Path(output_path).stat().st_size
    return {
        "original_size": orig_size,
        "compressed_size": new_size,
        "reduction_pct": round((1 - new_size / orig_size) * 100, 1),
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "No file selected"}), 400

    ext = Path(file.filename).suffix.lower()
    asset_type = ASSET_TYPES.get(ext)
    if not asset_type:
        return jsonify({"error": f"Unsupported format ({ext}). Supported: MP4, MOV, WebM, Lottie, PNG, JPG, WebP."}), 400

    file_id = str(uuid.uuid4())
    save_path = UPLOAD_DIR / f"{file_id}{ext}"
    file.save(str(save_path))

    try:
        if asset_type == "video":
            info = analyze_video(str(save_path))
        elif asset_type == "image":
            info = analyze_image(str(save_path))
        elif asset_type == "lottie":
            info = analyze_lottie(str(save_path))
        else:
            raise ValueError("Unknown asset type")
    except Exception as e:
        save_path.unlink(missing_ok=True)
        return jsonify({"error": f"Could not read file: {e}"}), 400

    return jsonify({
        "file_id": file_id,
        "original_name": file.filename,
        "ext": ext,
        "asset_type": asset_type,
        **info,
    })


@app.route("/api/compress", methods=["POST"])
def compress():
    data = request.get_json()
    file_id = data.get("file_id")
    asset_type = data.get("asset_type")
    preset_name = data.get("preset", "balanced")

    if not file_id:
        return jsonify({"error": "Missing file_id"}), 400

    matches = list(UPLOAD_DIR.glob(f"{file_id}.*"))
    if not matches:
        return jsonify({"error": "File not found. Please re-upload."}), 404
    input_path = str(matches[0])

    try:
        if asset_type == "video":
            output_path = str(UPLOAD_DIR / f"{file_id}_compressed.mp4")
            p = VIDEO_PRESETS.get(preset_name, VIDEO_PRESETS["balanced"])
            result = compress_video(
                input_path, output_path,
                crf=data.get("crf", p["crf"]),
                preset=p["preset"],
                scale=data.get("scale"),
                strip_audio=data.get("strip_audio", True),
                max_fps=data.get("max_fps"),
            )

        elif asset_type == "image":
            fmt = data.get("fmt", IMAGE_PRESETS.get(preset_name, IMAGE_PRESETS["balanced"])["fmt"])
            ext = fmt.lower().replace("jpeg", "jpg")
            output_path = str(UPLOAD_DIR / f"{file_id}_compressed.{ext}")
            p = IMAGE_PRESETS.get(preset_name, IMAGE_PRESETS["balanced"])
            result = compress_image(
                input_path, output_path,
                quality=data.get("quality", p["quality"]),
                fmt=fmt,
                scale=data.get("scale", 1.0),
            )

        elif asset_type == "lottie":
            output_path = str(UPLOAD_DIR / f"{file_id}_compressed.lottie")
            p = LOTTIE_PRESETS.get(preset_name, LOTTIE_PRESETS["balanced"])
            result = compress_lottie(
                input_path, output_path,
                quality=data.get("quality", p["quality"]),
                target_fps=data.get("target_fps", p["fps"]),
                scale=data.get("scale", p["scale"]),
            )
        else:
            return jsonify({"error": f"Unknown asset type: {asset_type}"}), 400

    except subprocess.CalledProcessError as e:
        return jsonify({"error": f"Compression failed: {e.stderr}"}), 500
    except Exception as e:
        return jsonify({"error": f"Compression failed: {e}"}), 500

    return jsonify({"file_id": file_id, "asset_type": asset_type, **result})


@app.route("/api/download/<file_id>")
def download(file_id):
    for pattern in (f"{file_id}_compressed.*",):
        matches = list(UPLOAD_DIR.glob(pattern))
        if matches:
            out = matches[0]
            originals = [m for m in UPLOAD_DIR.glob(f"{file_id}.*") if "_compressed" not in m.name]
            dl_name = f"{originals[0].stem}_compressed{out.suffix}" if originals else f"compressed{out.suffix}"
            return send_file(str(out), as_attachment=True, download_name=dl_name)
    return jsonify({"error": "Compressed file not found"}), 404


@app.route("/api/preview/<file_id>")
def preview_original(file_id):
    for m in UPLOAD_DIR.glob(f"{file_id}.*"):
        if "_compressed" not in m.name:
            mime = _mime_for(m.suffix)
            return send_file(str(m), mimetype=mime)
    return jsonify({"error": "File not found"}), 404


@app.route("/api/preview/<file_id>/compressed")
def preview_compressed(file_id):
    for m in UPLOAD_DIR.glob(f"{file_id}_compressed.*"):
        mime = _mime_for(m.suffix)
        return send_file(str(m), mimetype=mime)
    return jsonify({"error": "Compressed file not found"}), 404


def _mime_for(ext: str) -> str:
    return {
        ".mp4": "video/mp4",
        ".mov": "video/quicktime",
        ".webm": "video/webm",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".lottie": "application/octet-stream",
    }.get(ext.lower(), "application/octet-stream")


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5050)
