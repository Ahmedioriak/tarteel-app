"""Run from anywhere:  python3 quran_recitation/tests/service_test.py
Needs quran-transcript, numpy, fastapi, httpx, imageio-ffmpeg. No acoustic model or torch: a stand-in model hears a given phonetic string, so
everything after the model runs for real. The real model is only exercised on a server with the full dependencies."""
"""Run service.py's HTTP API for real (FastAPI + ffmpeg decoding + breath splitting) with the stand-in model."""
import sys, io, json, wave as wavmod
import os
ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import types, os, quran_transcript
_qm = os.path.join(os.path.dirname(os.path.dirname(quran_transcript.__file__)), "quran_muaalem")
for name, path in (("quran_muaalem", _qm), ("quran_muaalem.modeling", os.path.join(_qm, "modeling"))):
    m = types.ModuleType(name); m.__path__ = [path]; sys.modules[name] = m
import numpy as np
import quran_checker as qc

HEARD = {}
class FakeModel:
    def __init__(self, directory, device): pass
    def listen(self, wave, left, right, offset_s=0.0):
        heard = HEARD["next"]
        start, end = offset_s + left / qc.SAMPLE_RATE, offset_s + right / qc.SAMPLE_RATE
        step = (end - start) / max(1, len(heard))
        return qc.Breath(start, end, {"phonemes": [qc.Emission(c, start + (i + .5) * step, .97) for i, c in enumerate(heard)]})
qc.AcousticModel = FakeModel
import service
from fastapi.testclient import TestClient

def wav_bytes(seconds=2.0):
    t = np.arange(int(16000 * seconds)) / 16000
    x = (0.3 * np.sin(2 * np.pi * 220 * t) * (np.sin(2 * np.pi * 1.5 * t) > -0.2)).astype(np.float32)
    buf = io.BytesIO()
    with wavmod.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes((x * 32767).astype("<i2").tobytes())
    return buf.getvalue()

q = qc.Quran(qc.options(None))
ref = q.reference(qc.joined(q.words(112, 1, 1)), 112).phonemes
with TestClient(service.app) as c:
    print("health:", c.get("/health").json())
    print("page:", c.get("/recitation/").status_code, "| widget:", c.get("/recitation/quran-widget.js").status_code)
    HEARD["next"] = ref
    r = c.post("/api/recitation/analyze?ayah=112:1", content=wav_bytes(1.6), headers={"Content-Type": "audio/wav"})
    print("analyze 112:1:", r.status_code, [x["status"] for x in r.json()["results"]])
    HEARD["next"] = ref.replace("هُوَ", "هِوَ", 1)
    r = c.post("/api/recitation/analyze?ayah=112:1", content=wav_bytes(1.6), headers={"Content-Type": "audio/wav"})
    res = r.json()["results"][0]
    print("analyze vowel error:", r.status_code, res["status"], [(f["word"], f["expected"], f["heard"]) for f in res["findings"]])
    print("bad type:", c.post("/api/recitation/analyze", content=b"x", headers={"Content-Type": "text/plain"}).status_code)
    print("empty:", c.post("/api/recitation/analyze", content=b"", headers={"Content-Type": "audio/wav"}).status_code)
    print("bad ayah:", c.post("/api/recitation/analyze?ayah=999:1", content=wav_bytes(1), headers={"Content-Type": "audio/wav"}).status_code)
