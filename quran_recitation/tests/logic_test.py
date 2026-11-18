"""Run from anywhere:  python3 quran_recitation/tests/logic_test.py
Needs quran-transcript, numpy. No acoustic model or torch: a stand-in model hears a given phonetic string, so
everything after the model runs for real. The real model is only exercised on a server with the full dependencies."""
"""Exercise quran_checker.py's analysis without the 2.4 GB acoustic model: a stand-in model 'hears' a given phonetic
string (as the real CTC heads would emit it), so everything after the model runs for real."""
import sys, json
import os
ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
# test scaffolding: load quran_muaalem.modeling.vocab without the package __init__ (which needs torch for inference)
import types, os, quran_transcript
_qm = os.path.join(os.path.dirname(os.path.dirname(quran_transcript.__file__)), "quran_muaalem")
for name, path in (("quran_muaalem", _qm), ("quran_muaalem.modeling", os.path.join(_qm, "modeling"))):
    m = types.ModuleType(name); m.__path__ = [path]; sys.modules[name] = m
import numpy as np
import quran_checker as qc
from quran_transcript import quran_phonetizer

quran = qc.Quran(qc.options(None))

class FakeModel:
    """Emits `heard` on the phonemes head and the reference's own sifat on the sifat heads."""
    def __init__(self): self.script = []
    def listen(self, wave, left, right, offset_s=0.0):
        heard, sifat = self.script.pop(0)
        start, end = offset_s + left / qc.SAMPLE_RATE, offset_s + right / qc.SAMPLE_RATE
        step = (end - start) / max(1, len(heard))
        phones = [qc.Emission(c, start + (i + 0.5) * step, 0.97) for i, c in enumerate(heard)]
        heads = {"phonemes": phones}
        if sifat is not None:
            from quran_muaalem.modeling.vocab import SIFAT_ATTR_TO_ARABIC
            from quran_transcript import chunck_phonemes
            groups = chunck_phonemes(heard); pos = 0
            for level in qc.SIFA_NAMES:
                evs = []; pos = 0
                for g, s in zip(groups, sifat):
                    t = start + (pos + len(g) / 2) * step; pos += len(g)
                    evs.append(qc.Emission(SIFAT_ATTR_TO_ARABIC.get(getattr(s, level, None), "[?]"), t, 0.97))
                heads[level] = evs
        return qc.Breath(start, end, heads)

def ref_of(s, a):
    words = quran.words(s, a, a)
    return words, quran.reference(qc.joined(words), s)

wave = np.zeros(qc.SAMPLE_RATE * 3, np.float32)
spans = [(0, len(wave))]
model = FakeModel()

def run(s, a, heard, sifat=None, ayah=True):
    model.script.append((heard, sifat))
    sel = quran.words(s, a, a) if ayah else None
    res = qc.analyze_spans(wave, spans, quran, sel, model)
    return qc.report_for(res, sel, None)

out = {}
words, ref = ref_of(112, 1)
r = run(112, 1, ref.phonemes, ref.sifat); out["1_correct_112_1"] = (r["results"][0]["status"], len(r["results"][0].get("findings", [])))
words, ref = ref_of(112, 3)
bad = ref.phonemes.replace("يَلِد", "يَلُد", 1)
r = run(112, 3, bad); f = r["results"][0].get("findings", [])
out["2_vowel_error_112_3"] = (r["results"][0]["status"], [(x["word"], x["kind"], x["label"], x["expected"], x["heard"], x["status"]) for x in f])
bad = ref.phonemes.replace("يَلِد", "يَلِت", 1)
r = run(112, 3, bad); f = r["results"][0].get("findings", [])
out["3_letter_error_112_3"] = (r["results"][0]["status"], [(x["word"], x["kind"], x["expected"], x["heard"]) for x in f])
words, ref = ref_of(67, 1)
r = run(67, 1, ref.phonemes, None, ayah=False)
out["4_auto_identify_67_1"] = (r["results"][0].get("first_ayah"), r["results"][0]["status"])
words, ref = ref_of(55, 13)
r = run(55, 13, ref.phonemes, None, ayah=False)
out["5_auto_repeated_55_13"] = (r["results"][0]["status"], r["results"][0].get("candidate_count"))
words, ref = ref_of(1, 2)
skip = ref.phonemes.replace("رَببِ", "", 1)
r = run(1, 2, skip); f = r["results"][0].get("findings", [])
out["6_skipped_word_1_2"] = (r["results"][0]["status"], [(x["word"], x["operation"], x["status"]) for x in f])
print(json.dumps({"keys": list(r.keys()), "notice": r.get("notice", "")[:40]}, ensure_ascii=False))
for k, v in out.items(): print(k, "->", json.dumps(v, ensure_ascii=False))
