"""Standalone Quran recitation service or mountable ASGI app. See CLAUDE.md."""
from __future__ import annotations

import argparse
from contextlib import asynccontextmanager
import hmac
import os
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory

import anyio
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from starlette.concurrency import run_in_threadpool
from starlette.requests import ClientDisconnect

if __package__:
    from . import quran_checker as checker
else:
    import quran_checker as checker

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"
MAX_BYTES = 24 * 1024 * 1024
AUDIO_FORMATS = "wav,mp3,mov,matroska,webm,ogg,flac,aac"
FORMATS = {"audio", "video"}
CONTAINER_TYPES = {"application/octet-stream", "application/ogg", "application/mp4", "application/webm"}


def decode_upload(path):
    import numpy as np
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    command = [ffmpeg, "-nostdin", "-v", "error", "-protocol_whitelist", "file",
               "-format_whitelist", AUDIO_FORMATS,
               "-i", str(path), "-map", "0:a:0", "-t", str(checker.MAX_AUDIO + 1),
               "-ac", "1", "-ar", str(checker.SAMPLE_RATE), "-f", "f32le", "pipe:1"]
    try:
        run = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=75)
    except subprocess.TimeoutExpired as exc:
        raise checker.CheckError("استغرقت قراءة التسجيل وقتاً طويلاً. جرّب مقطعاً أقصر.") from exc
    if run.returncode != 0:
        raise checker.CheckError("تعذّرت قراءة الصوت. جرّب تسجيل WAV أو MP3 أو M4A أو WebM.")
    return np.frombuffer(run.stdout, dtype="<f4")


class Analyzer:
    """One model and one mutable Quran index per worker; callers serialize access."""
    def __init__(self):
        self.model_dir = os.getenv("QURAN_MODEL_DIR") or None
        self.model = checker.AcousticModel(self.model_dir, os.getenv("QURAN_DEVICE", "auto"))
        self.quran = checker.Quran(checker.options(None))

    def scan(self, path, ayah, words):
        wave = decode_upload(path)
        spans = checker.split_audio(wave)
        selected = self.quran.words(*ayah, word_range=words) if ayah else None
        results = checker.analyze_spans(wave, spans, self.quran, selected, self.model)
        return checker.report_for(results, selected, self.model_dir)


def create_app():
    # Lazily initialized as well: mounting a subapp does not run its own lifespan.
    # The first request then warms the same runtime under the same lock.
    @asynccontextmanager
    async def lifespan(app):
        app.state.analyzer = await run_in_threadpool(Analyzer)
        try:
            yield
        finally:
            app.state.analyzer = None

    app = FastAPI(title="فحص التلاوة", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    # A single worker serializes the model and prevents request peaks from loading copies.
    # Build asyncio objects in the event loop (compatible with mounting/testing).
    app.state.slot = None
    app.state.analyzer = None

    @app.get("/", include_in_schema=False)
    async def home():
        return RedirectResponse("recitation/", status_code=307)

    @app.get("/recitation/", include_in_schema=False)
    async def page():
        return FileResponse(ASSETS / "index.html")

    @app.get("/recitation/quran-widget.js", include_in_schema=False)
    async def widget():
        return FileResponse(ASSETS / "quran-widget.js", media_type="text/javascript")

    @app.get("/health")
    async def health():
        return {"status": "ready" if app.state.analyzer else "awaiting_first_request"}

    @app.post("/api/recitation/analyze")
    async def analyze(request: Request, ayah: str | None = Query(None, max_length=24),
                      words: str | None = Query(None, max_length=24)):
        secret = os.getenv("QURAN_API_TOKEN")
        if secret and not hmac.compare_digest(request.headers.get("Authorization", "").encode(),
                                               ("Bearer " + secret).encode()):
            raise HTTPException(401, "يتطلب المسار اعتماد الخادم.")
        content_type = request.headers.get("Content-Type", "").lower().partition(";")[0].strip()
        if content_type.split("/", 1)[0] not in FORMATS and content_type not in CONTAINER_TYPES:
            raise HTTPException(415, "أرسل الملف الصوتي مباشرة في جسم الطلب.")
        try:
            where = checker.select_ayah(ayah) if ayah else None
            interval = checker.select_range(words) if words else None
            if interval and (not where or where[1] != where[2]):
                raise checker.CheckError("حدّد آية واحدة لاختيار نطاق كلماتها.")
            size = int(request.headers.get("Content-Length", 0))
            if size < 0:
                raise ValueError
        except (checker.CheckError, argparse.ArgumentTypeError) as exc:
            raise HTTPException(422, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(400, "بيانات الطلب غير صالحة.") from exc
        if size > MAX_BYTES:
            raise HTTPException(413, "الحد الأقصى لحجم الصوت 24 ميبيبايت.")
        if app.state.slot is None:
            app.state.slot = anyio.CapacityLimiter(1)
        try:
            app.state.slot.acquire_nowait()
        except anyio.WouldBlock as exc:
            raise HTTPException(429, "جارٍ تحليل تسجيل آخر. أعد المحاولة بعد قليل.",
                                headers={"Retry-After": "3"}) from exc
        try:
            with TemporaryDirectory(prefix="recitation-") as directory:
                path = Path(directory) / "audio"
                received = 0
                with path.open("wb") as out:
                    try:
                        with anyio.fail_after(90):
                            async for block in request.stream():
                                received += len(block)
                                if received > MAX_BYTES:
                                    raise HTTPException(413, "الحد الأقصى لحجم الصوت 24 ميبيبايت.")
                                if block:
                                    await run_in_threadpool(out.write, block)
                    except TimeoutError as exc:
                        raise HTTPException(408, "استغرق إرسال الصوت وقتاً طويلاً.") from exc
                    except ClientDisconnect as exc:
                        raise HTTPException(400, "انقطع إرسال الصوت.") from exc
                if not received:
                    raise HTTPException(422, "اختر تسجيلاً صوتياً أولاً.")
                if app.state.analyzer is None:
                    app.state.analyzer = await run_in_threadpool(Analyzer)
                try:
                    report = await run_in_threadpool(app.state.analyzer.scan, path, where, interval)
                except checker.CheckError as exc:
                    raise HTTPException(422, str(exc)) from exc
            return JSONResponse(report, headers={"Cache-Control": "no-store"})
        finally:
            app.state.slot.release()

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "8000")), workers=1)
