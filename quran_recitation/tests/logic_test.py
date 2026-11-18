"""Run from anywhere:  python3 quran_recitation/tests/logic_test.py
Needs quran-transcript, numpy. No acoustic model or torch: a stand-in model 'hears' a given phonetic string (as the
real CTC heads would emit it), so everything after the model runs for real. The real model is only exercised on a
server with the full dependencies."""
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

def run(s, a, heard, sifat=None, ayah=True, harakat=False):
    model.script.append((heard, sifat))
    sel = quran.words(s, a, a) if ayah else None
    res = qc.analyze_spans(wave, spans, quran, sel, model, harakat=harakat)
    return qc.report_for(res, sel, None, harakat)

failed = []
def check(name, got, want):
    ok = got == want
    print(("ok  " if ok else "FAIL"), name, "->", json.dumps(got, ensure_ascii=False))
    if not ok:
        failed.append(name)

def status(r):
    return r["results"][0]["status"]

def found(r):
    return [(x["word"], x["label"]) for x in r["results"][0].get("findings", [])]

# words only (the default): vowels, tanween, shadda, close letters are not judged
words, ref = ref_of(112, 1)
check("correct 112:1", (status(r := run(112, 1, ref.phonemes, ref.sifat)), found(r)), ("no_difference_detected", []))
words, ref = ref_of(112, 3)
vowel = ref.phonemes.replace("يَلِد", "يَلُد", 1)
check("vowel يَلِدْ→يَلُدْ: not judged", status(run(112, 3, vowel)), "no_difference_detected")
r = run(112, 3, ref.phonemes.replace("يَلِد", "يَلِت", 1))
check("letter د→ت: the word is wrong", (status(r), found(r)), ("possible_difference", [("يَلِدْ", "الكلمة")]))
r = run(112, 3, ref.phonemes.replace("يَلِد", "تَلِد", 1))
check("يلد→تلد: the word is wrong", (status(r), found(r)), ("possible_difference", [("يَلِدْ", "الكلمة")]))
check("close letters د→ذ: not judged", status(run(112, 3, ref.phonemes.replace("يَلِد", "يَلِذ", 1))), "no_difference_detected")
words, ref = ref_of(112, 2)
check("close letters ص→س: not judged", status(run(112, 2, ref.phonemes.replace("صصَ", "سسَ", 1))), "no_difference_detected")
check("shadda left out: not judged", status(run(112, 2, ref.phonemes.replace("صصَ", "صَ", 1))), "no_difference_detected")
words, ref = ref_of(112, 4)
check("tanween left out (كُفُوًا): not judged", status(run(112, 4, ref.phonemes.replace("كُفُوَن", "كُفُوَ", 1))), "no_difference_detected")
words, ref = ref_of(1, 2)
r = run(1, 2, ref.phonemes.replace("رَببِ", "", 1))
bare = lambda w: "".join(c for c in w if "\u0621" <= c <= "\u064a")
check("skipped رَبِّ: flagged", (status(r) != "no_difference_detected", [bare(w) for w, _ in found(r)][:1]), (True, ["رب"]))
# harakat=True: the full report as before
words, ref = ref_of(112, 3)
r = run(112, 3, vowel, harakat=True)
check("harakat on: vowel judged", (status(r), [(x["word"], x["kind"]) for x in r["results"][0]["findings"]]),
      ("possible_difference", [("يَلِدْ", "tashkeel")]))
# finding the ayah
words, ref = ref_of(67, 1)
r = run(67, 1, ref.phonemes, None, ayah=False)
check("auto identify 67:1", (r["results"][0].get("first_ayah"), status(r)), ("67:1", "no_difference_detected"))
words, ref = ref_of(55, 13)
r = run(55, 13, ref.phonemes, None, ayah=False)
check("55:13 repeated: ambiguous", (status(r), r["results"][0].get("candidate_count")), ("ambiguous", 31))
check("report says words only", r["judged"], "words")
print("all passed" if not failed else f"{len(failed)} failed")
sys.exit(1 if failed else 0)
