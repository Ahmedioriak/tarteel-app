"""Run from anywhere:  python3 quran_recitation/tests/service_test.py
Needs quran-transcript, numpy, fastapi, httpx, imageio-ffmpeg. Runs service.py's HTTP API for real (FastAPI, ffmpeg
decoding, breath splitting) with a stand-in model that hears a given phonetic string; the real model is only
exercised on a server with the full dependencies."""
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

failed = []
def check(name, got, want):
    ok = got == want
    print(("ok  " if ok else "FAIL"), name, "->", json.dumps(got, ensure_ascii=False))
    if not ok:
        failed.append(name)

q = qc.Quran(qc.options(None))
ref = q.reference(qc.joined(q.words(112, 1, 1)), 112).phonemes
post = lambda c, url, body, kind="audio/wav": c.post(url, content=body, headers={"Content-Type": kind})
with TestClient(service.app) as c:
    check("health", c.get("/health").json(), {"status": "ready"})
    check("page and widget", (c.get("/recitation/").status_code, c.get("/recitation/quran-widget.js").status_code), (200, 200))
    HEARD["next"] = ref
    r = post(c, "/api/recitation/analyze?ayah=112:1", wav_bytes(1.6))
    check("correct 112:1", (r.status_code, [x["status"] for x in r.json()["results"]], r.json()["judged"]),
          (200, ["no_difference_detected"], "words"))
    HEARD["next"] = ref.replace("هُوَ", "هِوَ", 1)
    r = post(c, "/api/recitation/analyze?ayah=112:1", wav_bytes(1.6))
    check("vowel هُوَ→هِوَ, words only: not judged", (r.status_code, r.json()["results"][0]["status"]), (200, "no_difference_detected"))
    r = post(c, "/api/recitation/analyze?ayah=112:1&harakat=1", wav_bytes(1.6))
    res = r.json()["results"][0]
    check("vowel هُوَ→هِوَ, harakat=1: judged", (r.status_code, res["status"], [f["word"] for f in res["findings"]], r.json()["judged"]),
          (200, "possible_difference", ["هُوَ"], "full"))
    HEARD["next"] = ref.replace("قُل", "كُن", 1)
    r = post(c, "/api/recitation/analyze?ayah=112:1", wav_bytes(1.6))
    res = r.json()["results"][0]
    check("قُلْ→كُنْ: the word is wrong", (res["status"], [f["label"] for f in res["findings"]]), ("possible_difference", ["الكلمة"]))
    check("bad type", post(c, "/api/recitation/analyze", b"x", "text/plain").status_code, 415)
    check("empty", post(c, "/api/recitation/analyze", b"").status_code, 422)
    check("bad ayah", post(c, "/api/recitation/analyze?ayah=999:1", wav_bytes(1)).status_code, 422)
print("all passed" if not failed else f"{len(failed)} failed")
sys.exit(1 if failed else 0)
