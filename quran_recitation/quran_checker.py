#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = [
#   "quran-muaalem==0.3.0", "quran-transcript==0.6.4", "transformers==4.55.4", "torch==2.7.1",
#   "numpy>=2.2.6,<3", "imageio-ffmpeg>=0.6,<0.7", "sounddevice>=0.5,<0.6",
# ]
# [tool.uv.sources]
# torch = [{index = "pytorch-cpu", marker = "(sys_platform == 'linux' and platform_machine == 'x86_64') or (sys_platform == 'win32' and platform_machine == 'AMD64')"}]
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
# ///
"""مراجعة تلاوة القرآن من الصوت، برواية حفص عن عاصم.

على الكمبيوتر، أسهل تشغيل مع uv (يدير بايثون والحزم تلقائياً):
    uv run quran_checker.py --demo
    uv run quran_checker.py my_voice.m4a
    uv run quran_checker.py my_voice.m4a --ayah 1:7
    uv run quran_checker.py --mic 12 --ayah 1:7
    uv run quran_checker.py --live
    uv run quran_checker.py --live --ayah 1:7
    uv run quran_checker.py passage.wav --ayah 2:255 --words 1-4
    uv run quran_checker.py passage.wav --ayah 1:2-4 --json report.json

للاستخدام من مشروع بايثون:
    from quran_checker import check_recitation
    report = check_recitation("my_voice.m4a", ayah="1:7")

يمكن التثبيت داخل بيئة بايثون 3.11 باستخدام pip عوضاً عن uv:
    python -m pip install quran-muaalem==0.3.0 quran-transcript==0.6.4 transformers==4.55.4 torch==2.7.1 imageio-ffmpeg sounddevice

أول تشغيل يحتاج اتصالاً لتنزيل نموذج Muaalem الأصلي (حوالي 2.42 GB).
مكان نموذج محلي محمّل مسبقاً: --model-dir PATH. التسجيل يعالَج على جهازك.
إعداد uv الافتراضي مهيأ للمعالج العادي على حواسيب x86-64. لتشغيل CUDA
استخدم بيئة بايثون فيها نسخة PyTorch ملائمة لبطاقتك ثم اختر --device cuda.
--demo ينزل ملفاً صوتياً عاماً للآية 1:2. وقت التسجيل بالميكروفون بالثواني.
للاختبار في الحفظ حدّد الآية المقصودة. البحث التلقائي يرجّح آية كاملة لكل
مقطع. عبارة مشتركة أو مطابقة ضعيفة قد تعطي عدة مرشحين. يمكنك تحديد عدة
آيات متتالية من نفس السورة أو أرقام الكلمات في آية واحدة (تبدأ من 1).
استخدم صوت قارئ واحد، واترك فاصلاً مسموعاً عند أخذ النفس. بعد انتهاء
التسجيل يعرض البرنامج الكلمة والفرق المحتمل وزمن مراجعته التقريبي.
وضع --live يراجع كل عبارة بعد سكتة تقارب ثانية ويستمر حتى Ctrl+C. إذا
حددت --ayah في هذا الوضع يكرر فحص المرجع نفسه في كل مرة. مع --json -
يخرج كائن JSON مستقلاً لكل عبارة. عند اختيار مسار ملف يُحفظ تقرير الجلسة عند الإيقاف.

المرجع الافتراضي يستعمل أربع حركات للمتصل والمنفصل ويجرب 2/4/6 للعارض
للسكون. يمكن تحديد أوجه حفص عبر --profile وإعطائه ملف JSON بخصائص
MoshafAttributes، مثال {"madd_aared_len": 2}. يعرض التقرير الوجه المستخدم.
درجات النموذج الخام والتوقيت عند انبعاث الرمز الصوتي تقديرات. لا تمثل
احتمال صحة التلاوة أو قياساً زمنياً لعدد الحركات. يُعرض نقص التغطية
عندما يتعذر ربط صفة بأحد الحروف. الإشمام المرئي، اختيار مواضع الوقف
والابتداء من جهة المعنى، والقراءات الأخرى خارج نطاق هذا الفحص الصوتي.
النموذج قد يسهو أو يخطّئ قراءة صحيحة. استمع للموضع مع معلّم متقن عند الشك.

Model: https://huggingface.co/obadx/muaalem-model-v3_2
Model implementation & notice: https://github.com/obadx/quran-muaalem
Reference, phonetics and rule engine: https://github.com/obadx/quran-transcript
Example audio: https://everyayah.com/data/Alafasy_128kbps/001002.mp3
"""
from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
from collections import defaultdict, deque
from dataclasses import dataclass
from functools import lru_cache
from itertools import groupby
import json
import math
import os
from pathlib import Path
import re
import shutil
from statistics import median
import subprocess
import sys
from tempfile import NamedTemporaryFile, TemporaryDirectory
import unicodedata
from urllib.request import urlopen

MODEL_ID = "obadx/muaalem-model-v3_2"
MODEL_REVISION = "01a1ef9fbe40d144ef845101e89ff924aed3fef5"
SAMPLE_RATE = 16_000
MAX_AUDIO = 300
MAX_BREATH = 60
DEMO_URL = "https://everyayah.com/data/Alafasy_128kbps/001002.mp3"
NOTICE = "هذا فحص بالذكاء الاصطناعي قد يخطئ. راجع الموضع مع معلّم متقن ومجاز عند الشك."
# Conservative abstention heuristics. These have not been calibrated on learners.
PHONE_FLOOR = 0.68
SIFA_FLOOR = 0.90
AUTO_MIN, AUTO_MARGIN = 0.82, 0.08
VOWELS = {"َ": "فتحة", "ُ": "ضمة", "ِ": "كسرة", "۪": "فتحة ممالة", "ؙ": "ضمة مختلسة"}
SIFA_NAMES = {
    "hams_or_jahr": "الهمس والجهر", "shidda_or_rakhawa": "الشدة والرخاوة",
    "tafkheem_or_taqeeq": "التفخيم والترقيق", "itbaq": "الإطباق والانفتاح",
    "safeer": "الصفير", "qalqla": "القلقلة", "tikraar": "التكرير",
    "tafashie": "التفشي", "istitala": "الاستطالة", "ghonna": "الغنة",
}
STATUS = {
    "no_difference_detected": "لم يظهر اختلاف في الأجزاء التي أمكن فحصها",
    "possible_difference": "ظهر اختلاف محتمل يستحق الاستماع",
    "review": "يحتاج إلى مراجعة",
    "ambiguous": "تعذّر تحديد الآية بثقة",
}
BASE_PROFILE = dict(rewaya="hafs", madd_monfasel_len=4, madd_mottasel_len=4,
                    madd_mottasel_waqf=4)


class CheckError(Exception):
    """An actionable input, reference, or inference problem."""


@dataclass(frozen=True)
class Word:
    sura: int
    ayah: int
    num: int
    text: str

    @property
    def ref(self):
        return f"{self.sura}:{self.ayah}"


@dataclass(frozen=True)
class Emission:
    text: str
    time: float  # peak CTC frame in seconds, relative to the original recording
    strength: float  # raw local softmax, never a correctness score


@dataclass
class Breath:
    start: float
    end: float
    heads: dict[str, list[Emission]]

    @property
    def phones(self):
        return self.heads["phonemes"]

    @property
    def heard(self):
        return "".join(x.text for x in self.phones)


def info(message):
    print(message, file=sys.stderr, flush=True)


def decode_file(path):
    import numpy as np
    path = Path(path).expanduser()
    if not path.is_file():
        raise CheckError(f"ملف الصوت غير موجود: {path}")
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    # Read one extra second so a long file is rejected rather than truncated.
    cmd = [ffmpeg, "-nostdin", "-v", "error", "-i", str(path.resolve()),
           "-map", "0:a:0", "-t", str(MAX_AUDIO + 1), "-ac", "1", "-ar",
           str(SAMPLE_RATE), "-f", "f32le", "pipe:1"]
    r = subprocess.run(cmd, capture_output=True, check=False)
    if r.returncode:
        raise CheckError("تعذرت قراءة مسار الصوت. تأكد أن الملف يفتح في مشغل الصوت.")
    return np.frombuffer(r.stdout, dtype="<f4").copy()


def record_mic(seconds):
    try:
        import sounddevice as sd
    except OSError as exc:
        raise CheckError("تعذّر فتح نظام التسجيل. على لينكس ثبّت PortAudio أو استخدم ملفاً صوتياً.") from exc
    info(f"بدأ التسجيل الآن لمدة {seconds:g} ثانية…")
    try:
        data = sd.rec(round(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                      channels=1, dtype="float32", blocking=True)
    except (sd.PortAudioError, ValueError) as exc:
        raise CheckError("تعذّر فتح الميكروفون. تحقق من إذن التسجيل أو مرّر ملفاً صوتياً.") from exc
    info("انتهى التسجيل.")
    return data[:, 0]


def split_audio(wave):
    """Conservative energy-based breath segmentation, keeping original offsets."""
    import numpy as np
    length = len(wave)
    if length < SAMPLE_RATE * 0.24:
        raise CheckError("التسجيل قصير جداً أو فارغ.")
    if length > MAX_AUDIO * SAMPLE_RATE:
        raise CheckError(f"قسّم الملف إلى تسجيلات لا تتجاوز {MAX_AUDIO} ثانية.")
    if not np.isfinite(wave).all():
        raise CheckError("يحتوي الصوت على عينات تالفة.")
    step = SAMPLE_RATE // 50
    frames = np.pad(wave, (0, (-length) % step)).reshape(-1, step)
    # Variance rejects a constant DC signal as well as digital silence.
    rms = np.std(frames, axis=1, dtype="float64")
    # Use the loudest 0.4 s, so leading silence does not drown out a short ayah.
    k = min(len(rms), 20)
    level = float(np.median(np.partition(rms, -k)[-k:]))
    if level < 0.00045:
        raise CheckError("التسجيل صامت أو منخفض جداً. قرّب الميكروفون وأعد القراءة.")
    active = np.flatnonzero(rms > max(0.00018, level * 0.043))
    if len(active) < 8:
        raise CheckError("لم يظهر مقطع صوتي كافٍ للتحليل.")
    groups = np.split(active, np.where(np.diff(active) >= 51)[0] + 1)
    spans = []
    for group in groups:
        if len(group) < 5:
            continue
        left = max(0, int(group[0]) * step - SAMPLE_RATE // 6)
        right = min(length, (int(group[-1]) + 1) * step + SAMPLE_RATE // 6)
        if right - left > MAX_BREATH * SAMPLE_RATE:
            raise CheckError(f"تجاوز مقطع متصل {MAX_BREATH} ثانية. قصّ التسجيل عند وقف واضح.")
        speech = wave[int(group[0]) * step:min(length, (int(group[-1]) + 1) * step)]
        if np.mean(np.abs(speech) >= 0.999) > 0.015:
            raise CheckError("الصوت متشبّع في جزء من التلاوة. خفّض مستوى التسجيل وأعد القراءة.")
        spans.append((left, right))
    if not spans:
        raise CheckError("لم يظهر مقطع صوتي كافٍ للتحليل.")
    return spans


def collapse_ctc(ids, strengths, vocabulary, blank, start, end):
    events = []
    # Uniform mapping to emitted frames, for replay cues only.
    frame_seconds = (end - start) / len(ids) if len(ids) else 0
    for token_id, run in groupby(range(len(ids)), key=ids.__getitem__):
        if token_id == blank:
            continue
        peak = max(run, key=strengths.__getitem__)
        events.append(Emission(vocabulary[token_id], start + (peak + 0.5) * frame_seconds,
                               float(strengths[peak])))
    return events


class AcousticModel:
    """Independent decoding of each CTC head. The audio pass sees no verse text."""
    def __init__(self, directory, device):
        import torch
        from transformers import SeamlessM4TFeatureExtractor
        from quran_muaalem.modeling.modeling_multi_level_ctc import Wav2Vec2BertForMultilevelCTC
        if directory:
            root = Path(directory).expanduser()
            if not root.is_dir():
                raise CheckError("مجلد النموذج المحلي غير موجود.")
        else:
            from huggingface_hub import snapshot_download
            info("تحميل النموذج الأصلي أو فتح نسخته المحفوظة على الجهاز…")
            root = Path(snapshot_download(MODEL_ID, revision=MODEL_REVISION,
                        allow_patterns=["config.json", "vocab.json", "preprocessor_config.json",
                                        "model.safetensors"]))
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        if self.device == "cuda" and not torch.cuda.is_available():
            raise CheckError("CUDA غير متاح. استخدم --device cpu.")
        vocab = json.loads((root / "vocab.json").read_text(encoding="utf-8"))
        self.lookup = {level: {i: txt for txt, i in values.items()} for level, values in vocab.items()}
        self.blank = {level: values["[PAD]"] for level, values in vocab.items()}
        if not all(v in vocab["phonemes"] for v in "َُِ"):
            raise CheckError("مفردات النموذج لا تتضمن الحركات الثلاث.")
        self.extractor = SeamlessM4TFeatureExtractor.from_pretrained(str(root), local_files_only=True)
        self.model = Wav2Vec2BertForMultilevelCTC.from_pretrained(
            str(root), local_files_only=True, use_safetensors=True).to(self.device).eval()

    def listen(self, wave, left, right, offset_s=0.0):
        import torch
        features = self.extractor(wave[left:right], sampling_rate=SAMPLE_RATE, return_tensors="pt")
        features = {k: v.to(self.device) for k, v in features.items()}
        start, end = offset_s + left / SAMPLE_RATE, offset_s + right / SAMPLE_RATE
        heads = {}
        with torch.inference_mode():
            outputs = self.model(**features, return_dict=False)[0]
            for level, logits in outputs.items():
                strength, ids = logits[0].float().softmax(dim=-1).max(dim=-1)
                scores = strength.cpu().tolist()
                if not all(math.isfinite(x) for x in scores):
                    raise CheckError("تعذّر استخراج قيم صوتية مستقرة لهذا التسجيل.")
                heads[level] = collapse_ctc(ids.cpu().tolist(), scores, self.lookup[level],
                                             self.blank[level], start, end)
        return Breath(start, end, heads)


def phone_key(text):
    """Ignore length only during retrieval and broad mismatch checks."""
    return "".join(char for char, _ in groupby(text))


SPELLING = str.maketrans({"إ": "ء", "أ": "ء", "ؤ": "ء", "ئ": "ء", "آ": "ء",
                          "ٲ": "ء", "ٱ": "", "ا": "", "ى": "", "ة": "ه",
                          "ں": "ن", "۾": "م", "ۦ": "ي", "ۥ": "و"})
CONSONANTS = set("ءبتثجحخدذرزسشصضطظعغفقكلمنهوي")


def retrieval_key(text):
    return phone_key("".join(c for c in text.translate(SPELLING) if c in CONSONANTS))


def similarity(a, b):
    from Levenshtein import distance  # dependency of quran-transcript
    return 1 - distance(a, b) / max(1, len(a), len(b))


def alignment(a, b):
    """Map reference phones to free acoustic emissions by minimum edit distance."""
    from Levenshtein import opcodes
    pairs = []
    for tag, i, k, j, m in opcodes(a, b):
        paired = min(k-i, m-j)
        pairs.extend(zip(range(i, i+paired), range(j, j+paired)))
        pairs.extend((n, None) for n in range(i+paired, k))
        pairs.extend((None, n) for n in range(j+paired, m))
    return pairs


def options(profile_path):
    from quran_transcript import MoshafAttributes
    custom = json.loads(Path(profile_path).read_text(encoding="utf-8")) if profile_path else {}
    if not isinstance(custom, dict):
        raise CheckError("ملف إعدادات الرواية يجب أن يحتوي على كائن JSON.")
    unknown = set(custom) - set(MoshafAttributes.model_fields)
    if unknown or custom.get("rewaya", "hafs") != "hafs":
        raise CheckError(f"تحقق من خصائص حفص في ملف الإعدادات. خصائص غير معروفة: {sorted(unknown)}")
    values = {**BASE_PROFILE, **custom}
    lengths = [values["madd_aared_len"]] if "madd_aared_len" in values else [2, 4, 6]
    profiles = []
    for n in lengths:
        try:
            profiles.append(MoshafAttributes(**{**values, "madd_aared_len": n}))
        except ValueError:
            if len(lengths) == 1:
                raise
    if not profiles:
        raise CheckError("تركيبة أوجه القراءة في ملف الإعدادات غير صالحة.")
    return profiles


class Quran:
    def __init__(self, profiles):
        from quran_transcript import Aya
        self.aya = Aya(1, 1)
        self.profiles = profiles
        self.keys = None

    def words(self, sura, first, last, word_range=None):
        if not 1 <= sura <= 114:
            raise CheckError("رقم السورة يجب أن يكون بين 1 و114.")
        self.aya.set(sura, 1)
        total = self.aya.get().num_ayat_in_sura
        if not 1 <= first <= last <= total:
            raise CheckError(f"الآية خارج النطاق. عدد آيات السورة {sura} هو {total}.")
        words = []
        for aya in range(first, last + 1):
            self.aya.set(sura, aya)
            words.extend(Word(sura, aya, i, t) for i, t in enumerate(self.aya.get().uthmani.split(), 1))
        if word_range:
            a, b = word_range
            if first != last or not 1 <= a <= b <= len(words):
                raise CheckError("--words يقبل نطاقاً موجوداً ضمن آية واحدة، بدءاً من الكلمة 1.")
            words = words[a-1:b]
        return words

    @lru_cache(maxsize=96)
    def reference(self, text, sura, profile_idx=0):
        from quran_transcript import quran_phonetizer
        return quran_phonetizer(text, self.profiles[profile_idx], remove_spaces=True, sura_idx=sura)

    def build_keys(self):
        if self.keys is not None:
            return
        keys = defaultdict(list)
        for s in range(1, 115):
            self.aya.set(s, 1)
            for a in range(1, self.aya.get().num_ayat_in_sura + 1):
                self.aya.set(s, a)
                text = self.aya.get().uthmani
                keys[retrieval_key(text)].append((s, a, text))
        self.keys = keys

    def identify(self, events):
        from heapq import nlargest
        heard = "".join(e.text for e in events)
        key = retrieval_key(heard)
        if not key:
            return None, [], 0
        self.build_keys()
        # All verses sharing a shortlisted key survive, including repeated ayat.
        shortlist = nlargest(32, self.keys, key=lambda k: similarity(key, k))
        acoustic = phone_key(heard)
        ranked = []
        for k in shortlist:
            for sura, ayah, text in self.keys[k]:
                canonical = phone_key(self.reference(text, sura).phonemes)
                ranked.append((similarity(acoustic, canonical), sura, ayah, canonical))
        ranked.sort(key=lambda item: (-item[0], item[1], item[2]))
        best = ranked[0][0]
        if best < 0.58:
            return None, [], 0
        nearby = [row for row in ranked if row[0] > best - AUTO_MARGIN + 1e-10]
        unambiguous = len(nearby) == 1
        # For a unique exact match, a clearly emitted discriminating consonant
        # can resolve close verses, e.g. al-hamdu vs. wal-hamdu. Absence alone
        # cannot resolve one, and identical refrains still yield all their IDs.
        if len(nearby) > 1 and best == 1.0 and nearby[1][0] < 1.0:
            acoustic_events = [max(run, key=lambda e: e.strength)
                               for _, run in groupby(events, key=lambda e: e.text)]
            unambiguous = all(positive_contrast(row[3], acoustic, acoustic_events)
                              for row in nearby[1:])
        picked = ranked[0][1:3] if best >= AUTO_MIN and len(key) >= 3 and unambiguous else None
        return picked, [f"{s}:{a}" for _, s, a, _ in nearby], len(nearby)


def positive_contrast(other, observed, events):
    return any(j is not None and (i is None or other[i] != observed[j])
               and observed[j] in CONSONANTS and events[j].strength >= 0.90
               for i, j in alignment(other, observed))


def joined(words):
    return " ".join(w.text for w in words)


def word_starts(words):
    starts, pos = [], 0
    for w in words:
        starts.append(pos)
        pos += len(w.text) + 1
    return starts


def phonetic_word_starts(ref, words):
    starts = word_starts(words)
    cuts = []
    for w, i in zip(words, starts):
        used = [m.pos[0] for m in ref.mappings[i:i+len(w.text)]
                if m is not None and not getattr(m, "deleted", False) and m.pos[1] > m.pos[0]]
        if not used:
            return []
        cuts.append(min(used))
    return cuts


def assign_breaths(quran, words, breaths):
    """Project a breath stop only when matching phones bracket one word boundary."""
    if len(breaths) == 1:
        return [words]
    ref = quran.reference(joined(words), words[0].sura)
    boundaries = phonetic_word_starts(ref, words)
    observed = "".join(b.heard for b in breaths)
    # A waqf vowel and the vowel on a fresh hamzat al-wasl may align to one
    # another across the silence. Consonant anchors avoid moving the word cut.
    matches = [(j, i) for i, j in alignment(ref.phonemes, observed)
               if i is not None and j is not None and ref.phonemes[i] == observed[j]
               and ref.phonemes[i] in CONSONANTS]
    heard_idx = [j for j, _ in matches]
    cuts, n = [0], 0
    for breath in breaths[:-1]:
        n += len(breath.phones)
        p = bisect_left(heard_idx, n)
        if p == 0 or p == len(matches):
            return None
        lj, li = matches[p-1]
        rj, ri = matches[p]
        if n - lj > 10 or rj - n > 7:
            return None
        candidates = [ix for ix, cut in enumerate(boundaries) if ix > cuts[-1] and li < cut <= ri]
        if len(candidates) != 1:
            return None
        cuts.append(candidates[0])
    cuts.append(len(words))
    return [words[a:b] for a, b in zip(cuts, cuts[1:])]


def locate_word(words, char_position):
    starts = word_starts(words)
    i = max(0, bisect_right(starts, char_position) - 1)
    if i + 1 < len(words) and char_position == starts[i] + len(words[i].text):
        i += 1
    w = words[i]
    return {"ayah": w.ref, "word_number": w.num, "word": w.text}


def vowel_description(phones):
    if not phones:
        return "لم يظهر صوت"
    marks = [VOWELS[ch] for ch in phones if ch in VOWELS]
    if len(marks) == 1:
        core = "".join(c for c in phones if c not in VOWELS)
        if not core:
            return marks[0]
        base = phone_key(core)
        vowel = next(c for c in phones if c in VOWELS)
        written = base + ("ّ" if len(base) == 1 and core == base * 2 else "") + vowel
        return f"{marks[0]} على {base} ({written})"
    if len(phones) == 1 and phones in CONSONANTS and not marks:
        return f"{phones} من دون حركة ظاهرة"
    return ("◌" if unicodedata.combining(phones[0]) else "") + phones


def human_phones(expected, heard, kind):
    if kind == "tashkeel":
        return vowel_description(expected), vowel_description(heard)
    return expected or "لا صوت متوقع", heard or "لم يظهر في التفريغ الصوتي"


def event_positions(pairs, start, end, heard_count):
    """For a deletion, return an acoustic replay anchor and no evidence tokens."""
    positions = []
    cursor = 0
    for r, p in pairs:
        if r is None:
            if p is not None and (cursor == start == end or start < cursor < end):
                positions.append(p)
        else:
            if start <= r < end and p is not None:
                positions.append(p)
            cursor = r + 1
    if positions:
        return positions, positions[0]
    anchor = next((p for r, p in pairs if r is not None and r >= start and p is not None), heard_count-1)
    return [], max(0, anchor)


def rule_names(error):
    names = []
    for field in ("ref_tajweed_rules", "missing_tajweed_rules", "replaced_tajweed_rules",
                  "inserted_tajweed_rules"):
        for rule in getattr(error, field, None) or []:
            if rule.name.ar not in names:
                names.append(rule.name.ar)
    return names


def check_attributes(ref, breath, pairs, words):
    """Use a sifat head only with a one-to-one, locally timed match of its groups."""
    from quran_muaalem.modeling.vocab import SIFAT_ATTR_TO_ARABIC
    from quran_transcript import chunck_phonemes
    groups = ref.sifat
    total = len(groups) * len(SIFA_NAMES)
    report = {"compared": 0, "possible": total}
    differences = []
    if not groups or "".join(g.phonemes for g in groups) != ref.phonemes:
        return differences, report
    emitted_groups = chunck_phonemes(breath.heard)
    if "".join(emitted_groups) != breath.heard:
        return differences, report
    group_at = {}
    first = 0
    for idx, chunk in enumerate(emitted_groups):
        end = first + len(chunk)
        group_at[first, end] = idx
        first = end
    # An insertion/deletion must not shift subsequent articulation labels.
    matched = {r: p for r, p in pairs if r is not None and p is not None
               and ref.phonemes[r] == breath.heard[p]}
    eligible = []
    first = 0
    for g in groups:
        stop = first + len(g.phonemes)
        a = matched.get(first)
        if a is not None and stop > first:
            b = a + stop - first
            group_idx = group_at.get((a, b))
            if group_idx is not None and all(matched.get(i) == a + i - first for i in range(first, stop)):
                eligible.append((g, first, group_idx, a, b))
        first = stop
    phone_positions = [None] * len(ref.phonemes)
    for pos, m in enumerate(ref.mappings):
        if m is not None and not getattr(m, "deleted", False):
            for i in range(max(0, m.pos[0]), min(len(phone_positions), m.pos[1])):
                if phone_positions[i] is None:
                    phone_positions[i] = pos
    for level, name in SIFA_NAMES.items():
        events = breath.heads.get(level, [])
        if len(events) != len(emitted_groups):
            continue
        for g, ref_pos, group_idx, a, b in eligible:
            e = events[group_idx]
            phones = breath.phones[a:b]
            if min(p.strength for p in phones) < PHONE_FLOOR or e.strength < SIFA_FLOOR:
                continue
            # Bound each group by midpoints between adjacent acoustic emissions.
            lo = (breath.phones[a-1].time + phones[0].time)/2 if a else breath.start
            hi = (phones[-1].time + breath.phones[b].time)/2 if b < len(breath.phones) else breath.end
            if not lo <= e.time <= hi:
                continue
            expected = SIFAT_ATTR_TO_ARABIC.get(getattr(g, level, None))
            if expected is None or phone_positions[ref_pos] is None:
                continue
            report["compared"] += 1
            if expected != e.text:
                differences.append({**locate_word(words, phone_positions[ref_pos]), "kind": "sifa",
                    "label": name, "expected": expected.strip("[]"), "heard": e.text.strip("[]"),
                    "near_s": round(e.time, 2), "status": "possible_difference"})
    return differences, report


def assess(quran, words, breath):
    from quran_transcript import explain_error
    text = joined(words)
    header = {"first_ayah": words[0].ref, "last_ayah": words[-1].ref,
              "first_word": words[0].num, "last_word": words[-1].num,
              "reference_text": text, "start_s": round(breath.start, 2),
              "end_s": round(breath.end, 2), "heard_phonemes": breath.heard}
    # True minimal edit distance; no stripping of vowels, shadda cues or lengths.
    variants = [quran.reference(text, words[0].sura, idx) for idx in range(len(quran.profiles))]
    scores = [similarity(ref.phonemes, breath.heard) for ref in variants]
    best_idx = max(range(len(scores)), key=scores.__getitem__)
    best_score, ref = scores[best_idx], variants[best_idx]
    header["reference_phonemes"] = ref.phonemes
    header["profile"] = quran.profiles[best_idx].model_dump(mode="json")
    if (not breath.phones or best_score < 0.58
            or median(x.strength for x in breath.phones) < PHONE_FLOOR):
        return {**header, "status": "review",
                "reason": "المقطع بعيد عن المرجع أو إشارات الصوت ضعيفة. تحقّق من الآية وأعد التسجيل."}
    pairs = alignment(ref.phonemes, breath.heard)
    findings = []
    labels = {"tashkeel": "حركة", "normal": "حرف أو مقطع", "tajweed": "حكم تجويد"}
    ops = {"replace": "تبديل", "insert": "زيادة", "delete": "حذف محتمل"}
    try:
        errors = explain_error(uthmani_text=text, ref_ph_text=ref.phonemes,
                               predicted_ph_text=breath.heard, mappings=ref.mappings)
    except (ValueError, IndexError, AssertionError):
        # Some unusual sound sequences have no rule explanation in the toolkit.
        # The unequal raw phonemes below still force a review result.
        errors = []
    for err in errors:
        detected = getattr(err, "predicted_ph", getattr(err, "preditected_ph", ""))
        expected, heard = human_phones(err.expected_ph, detected, err.error_type)
        indices, pos = event_positions(pairs, *err.ph_pos, len(breath.phones))
        evidence = [breath.phones[j].strength for j in indices]
        is_clear = bool(evidence) and min(evidence) >= PHONE_FLOOR and err.speech_error_type != "delete"
        names = rule_names(err)
        finding = {**locate_word(words, err.uthmani_pos[0]), "kind": err.error_type,
                   "label": "، ".join(names) or labels.get(err.error_type, err.error_type),
                   "operation": ops.get(err.speech_error_type, err.speech_error_type),
                   "expected": expected, "heard": heard, "expected_phonemes": err.expected_ph,
                   "heard_phonemes": detected, "near_s": round(breath.phones[pos].time, 2),
                   "status": "possible_difference" if is_clear else "review"}
        if err.expected_len is not None or err.predicted_len is not None:
            finding.update(model_expected_count=err.expected_len, model_heard_count=err.predicted_len)
        findings.append(finding)
    attributes, coverage = check_attributes(ref, breath, pairs, words)
    findings.extend(attributes)
    low = {}
    for r, p in pairs:
        if r is None or p is None or breath.phones[p].strength >= PHONE_FLOOR:
            continue
        low[p] = round(breath.phones[p].time, 2)
    status = "no_difference_detected"
    if any(f["status"] == "possible_difference" for f in findings):
        status = "possible_difference"
    elif findings or low or ref.phonemes != breath.heard:
        status = "review"
    result = {**header, "status": status, "findings": findings, "sifat_coverage": coverage,
              "weak_audio_times_s": list(low.values())}
    if ref.phonemes != breath.heard and not findings:
        result["reason"] = "ظهر اختلاف في الأصوات تعذّر إسناده إلى كلمة بثقة."
    return result


def weak_breath(b):
    return not b.phones or median(e.strength for e in b.phones) < PHONE_FLOOR


def manual_results(quran, words, breaths):
    if any(weak_breath(b) for b in breaths):
        return [{"status": "review", "first_ayah": words[0].ref, "last_ayah": words[-1].ref,
                 "reason": "يوجد مقطع فارغ أو ضعيف. تعذّرت مطابقة التسجيل كاملاً بالمرجع."}]
    groups = assign_breaths(quran, words, breaths)
    if groups is None:
        return [{"status": "review", "first_ayah": words[0].ref, "last_ayah": words[-1].ref,
                 "reason": "موضع الوقف داخل النص غير محسوم. افحص كل نَفَس على حدة وحدد كلماته بخيار --words."}]
    return [assess(quran, group, b) for group, b in zip(groups, breaths)]


def auto_results(quran, breaths):
    if len(breaths) > 1 and not any(weak_breath(b) for b in breaths):
        # A long ayah may require several breaths. Verify the internal word cuts.
        whole = [e for b in breaths for e in b.phones]
        if len(whole) <= 550:
            place, _, _ = quran.identify(whole)
            if place:
                s, a = place
                words = quran.words(s, a, a)
                groups = assign_breaths(quran, words, breaths)
                if groups:
                    return [assess(quran, g, b) for g, b in zip(groups, breaths)]
    results = []
    for b in breaths:
        span = {"start_s": round(b.start, 2), "end_s": round(b.end, 2)}
        if weak_breath(b):
            results.append({**span, "status": "review", "reason": "الإشارة الصوتية غير كافية لتعيين آية."})
            continue
        place, candidates, count = quran.identify(b.phones)
        if place is None:
            results.append({**span, "status": "ambiguous", "candidates": candidates,
                            "candidate_count": count, "heard_phonemes": b.heard,
                            "reason": "للتقييم حدّد المرجع بخيار --ayah، ولجزء من آية أضف --words."})
            continue
        s, a = place
        results.append(assess(quran, quran.words(s, a, a), b))
    return results


def select_range(value):
    m = re.fullmatch(r"(\d+)(?:[-–](\d+))?", value.strip())
    if not m:
        raise argparse.ArgumentTypeError("اكتب رقماً أو نطاقاً مثل 2-4")
    a, b = int(m[1]), int(m[2] or m[1])
    if a < 1 or b < a:
        raise argparse.ArgumentTypeError("اختر نطاقاً متصاعداً يبدأ من 1")
    return a, b


def select_ayah(value):
    parts = value.strip().split(":")
    if len(parts) != 2 or not parts[0].isdecimal():
        raise argparse.ArgumentTypeError("الصيغة المطلوبة 1:2 أو 1:2-4")
    sura = int(parts[0])
    if not 1 <= sura <= 114:
        raise argparse.ArgumentTypeError("رقم السورة بين 1 و114")
    return sura, *select_range(parts[1])


def write_report(path, report, protected):
    payload = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if path == "-":
        print(payload, end="")
        return
    target = Path(path).expanduser().resolve()
    if target in {Path(p).expanduser().resolve() for p in protected if p}:
        raise CheckError("اختر اسماً مستقلاً لملف التقرير.")
    tmp = None
    try:
        with NamedTemporaryFile("w", encoding="utf-8", dir=target.parent, delete=False) as out:
            tmp = Path(out.name)
            out.write(payload)
        os.replace(tmp, target)
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)


def show(report):
    for result in report["results"]:
        loc = result.get("first_ayah", "")
        if result.get("last_ayah", loc) != loc:
            loc += " – " + result["last_ayah"]
        stamp = f" | {result['start_s']:.1f}–{result['end_s']:.1f} ث" if "start_s" in result else ""
        print(f"\n{loc}{stamp}  {STATUS[result['status']]}")
        if result.get("reference_text"):
            print(result["reference_text"])
        if result.get("reason"):
            print(result["reason"])
        if result.get("candidates"):
            listed = result["candidates"][:10]
            suffix = f" … ({result['candidate_count']} موضعاً)" if result["candidate_count"] > len(listed) else ""
            print("مواضع مقترحة:", "، ".join(listed) + suffix)
        for f in result.get("findings", []):
            print(f"  • عند {f['near_s']:.2f} ث | {f['ayah']} كلمة {f['word_number']} «{f['word']}»"
                  f" | {f['label']} {f.get('operation', '')}")
            print(f"    المتوقع: {f['expected']}  |  ما رجّحه الصوت: {f['heard']}  |  {STATUS[f['status']]}")
            if f.get("model_expected_count") is not None and f.get("model_heard_count") is not None:
                print(f"    تقدير الطول بالعدّ الصوتي: المرجع {f['model_expected_count']}،"
                      f" تقدير النموذج {f['model_heard_count']}.")
        if coverage := result.get("sifat_coverage"):
            print(f"مقارنات صفات الحروف التي أمكن إسنادها: {coverage['compared']} من {coverage['possible']}.")
        if weak := result.get("weak_audio_times_s"):
            print("إشارات صوتية ضعيفة قرب الثواني:", "، ".join(str(t) for t in weak[:8]),
                  "…" if len(weak) > 8 else "")


def arguments(argv):
    p = argparse.ArgumentParser(description="فحص تلاوة برواية حفص، من صوتك إلى الكلمة والحركة.",
        epilog="مثال: python quran_checker.py my_voice.m4a --ayah 1:7  |  التشغيل الأول ينزّل النموذج.")
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("audio", nargs="?", help="ملف WAV أو MP3 أو M4A أو أي ملف صوتي يدعمه FFmpeg")
    source.add_argument("--mic", metavar="SECONDS", type=float, help="سجّل من الميكروفون لعدد محدد من الثواني")
    source.add_argument("--live", action="store_true", help="استمع باستمرار واعرض نتيجة كل عبارة عند الوقف، حتى Ctrl+C")
    source.add_argument("--demo", action="store_true", help="اختبر بملف تلاوة حقيقي للفاتحة 1:2 من EveryAyah")
    p.add_argument("--ayah", type=select_ayah, metavar="S:A[-B]", help="الآية المقصودة أو نطاق متصل من سورة واحدة")
    p.add_argument("--words", type=select_range, metavar="N[-M]", help="كلمات آية محددة، يبدأ العد من 1")
    p.add_argument("--profile", metavar="FILE", help="ملف JSON اختياري بأوجه القراءة في حفص")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="جهاز تشغيل النموذج")
    p.add_argument("--model-dir", metavar="DIR", help="نسخة النموذج المحلية بملفاتها الأربع")
    p.add_argument("--json", metavar="REPORT", help="حفظ تقرير JSON أو تمرير - لطباعته")
    a = p.parse_args(argv)
    if a.demo:
        if a.ayah not in (None, (1, 2, 2)) or a.words:
            p.error("--demo يستخدم الآية 1:2 كاملة.")
        a.ayah = (1, 2, 2)
    if a.words and (not a.ayah or a.ayah[1] != a.ayah[2]):
        p.error("--words يحتاج إلى --ayah لآية واحدة.")
    if a.mic is not None and not 0.5 <= a.mic <= MAX_AUDIO:
        p.error(f"--mic يقبل مدة بين 0.5 و{MAX_AUDIO} ثانية.")
    return a


def get_wave(args):
    if args.demo:
        info("تنزيل مثال التلاوة للفاتحة، الآية الثانية…")
        with TemporaryDirectory() as temp:
            path = Path(temp) / "001002.mp3"
            with urlopen(DEMO_URL, timeout=35) as response:
                data = response.read(8_000_001)
            if len(data) > 8_000_000:
                raise CheckError("حجم ملف المثال غير متوقع. استعمل تسجيلك الشخصي.")
            path.write_bytes(data)
            return decode_file(path)
    return record_mic(args.mic) if args.mic is not None else decode_file(args.audio)


def live_phrases(blocks):
    """Yield (session offset, waveform) from consecutive 100 ms microphone blocks."""
    import numpy as np
    lead = deque(maxlen=3)
    phrase = []
    noise, peak = 0.0004, 0.0
    origin = total = quiet = 0
    discard = False
    for block in blocks:
        level = float(np.std(block))
        gate = max(0.0010, noise * 3.2)
        total += len(block)
        if discard:
            quiet = quiet + 1 if level < gate else 0
            if quiet >= 10:
                discard, quiet = False, 0
            continue
        if not phrase:
            lead.append(block)
            if level < gate:
                noise = 0.94 * noise + 0.06 * level
                continue
            phrase = list(lead)
            lead.clear()
            origin, peak, quiet = total - sum(map(len, phrase)), level, 0
        else:
            phrase.append(block)
            peak = max(peak, level)
            quiet = quiet + 1 if level < max(gate * 0.8, peak * 0.035) else 0
        if quiet >= 10:
            yield origin / SAMPLE_RATE, np.concatenate(phrase)
            phrase, quiet = [], 0
        elif total - origin > (MAX_BREATH + 1) * SAMPLE_RATE:
            yield origin / SAMPLE_RATE, None
            phrase, discard, quiet = [], True, 0


def analyze_spans(wave, spans, quran, selected, model, offset_s=0.0):
    breaths = [model.listen(wave, a, b, offset_s) for a, b in spans]
    return manual_results(quran, selected, breaths) if selected else auto_results(quran, breaths)


def report_for(results, selected, model_dir):
    return {"riwaya": "حفص عن عاصم", "reference_mode": "selected" if selected else "automatic",
            "model": MODEL_ID, "model_revision": MODEL_REVISION if not model_dir else "local",
            "notice": NOTICE, "results": results}


def live_session(args):
    from queue import Empty, Full, Queue
    from threading import Event
    try:
        import sounddevice as sd
    except OSError as exc:
        raise CheckError("تعذّر فتح نظام الميكروفون. استخدم ملفاً أو ثبّت PortAudio.") from exc
    info(NOTICE)
    quran = Quran(options(args.profile))
    selected = quran.words(*args.ayah, word_range=args.words) if args.ayah else None
    model = AcousticModel(args.model_dir, args.device)
    backlog, lost = Queue(maxsize=450), Event()
    completed = []

    def receive(data, frames, clock, status):
        if status.input_overflow:
            lost.set()
        try:
            backlog.put_nowait(data[:, 0].copy())
        except Full:
            lost.set()

    def blocks():
        while not lost.is_set():
            try:
                yield backlog.get(timeout=0.25)
            except Empty:
                continue
        raise CheckError("انقطع جزء من صوت الميكروفون. أعد المحاولة أو افحص تسجيلاً محفوظاً.")

    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, blocksize=SAMPLE_RATE // 10, channels=1,
                            dtype="float32", callback=receive):
            info("الميكروفون مفتوح. اقرأ ثم قف لثانية حتى تظهر الملاحظات. للإيقاف اضغط Ctrl+C.")
            for offset, wave in live_phrases(blocks()):
                try:
                    if wave is None:
                        raise CheckError("طال المقطع المتصل. خذ نَفَساً أوضح بين العبارات.")
                    spans = split_audio(wave)
                    results = analyze_spans(wave, spans, quran, selected, model, offset)
                except CheckError as exc:
                    results = [{"status": "review", "reason": str(exc), "start_s": round(offset, 2),
                                "end_s": round(offset + (len(wave)/SAMPLE_RATE if wave is not None else MAX_BREATH), 2)}]
                report = report_for(results, selected, args.model_dir)
                if args.json == "-":
                    print(json.dumps(report, ensure_ascii=False, allow_nan=False), flush=True)
                else:
                    show(report)
                    sys.stdout.flush()
                    if args.json:
                        completed.extend(results)
    except KeyboardInterrupt:
        info("توقف الاستماع.")
    except sd.PortAudioError as exc:
        raise CheckError("تعذّر التسجيل من الميكروفون. تحقّق من الإذن أو افحص ملفاً صوتياً.") from exc
    finally:
        if completed:
            write_report(args.json, report_for(completed, selected, args.model_dir), [args.profile, __file__])
    return 0


def check_recitation(audio, *, ayah=None, words=None, profile=None, model_dir=None, device="auto"):
    """Return a JSON-compatible report. Input is a path or mono float samples at 16 kHz.

    ayah accepts e.g. "1:2" or "1:2-4", words accepts e.g. "3-4" (one ayah).
    """
    import numpy as np
    info(NOTICE)
    wave = decode_file(audio) if isinstance(audio, (str, os.PathLike)) else np.asarray(audio)
    if wave.ndim != 1:
        raise CheckError("مرّر ملف الصوت أو مصفوفة أحادية القناة بمعدل 16000 عينة في الثانية.")
    if isinstance(ayah, str):
        ayah = select_ayah(ayah)
    if isinstance(words, str):
        words = select_range(words)
    if words and not ayah:
        raise CheckError("حدّد الآية عند اختيار نطاق للكلمات.")
    spans = split_audio(wave)
    quran = Quran(options(profile))
    selected = quran.words(*ayah, word_range=words) if ayah else None
    model = AcousticModel(model_dir, device)
    results = analyze_spans(wave, spans, quran, selected, model)
    return report_for(results, selected, model_dir)


def main(argv=None):
    args = arguments(argv)
    if args.live:
        return live_session(args)
    audio = args.audio if args.audio else get_wave(args)
    report = check_recitation(audio, ayah=args.ayah, words=args.words, profile=args.profile,
                              model_dir=args.model_dir, device=args.device)
    if args.json:
        write_report(args.json, report, [args.audio, args.profile, __file__])
    if args.json != "-":
        show(report)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        info("توقف الفحص.")
        sys.exit(130)
    except ImportError as exc:
        info(f"تعذّر تحميل إحدى الحزم ({exc}). شغّل الملف باستخدام uv run كما في مقدّمته.")
        sys.exit(2)
    except (CheckError, OSError, ValueError, RuntimeError) as exc:
        info(f"تعذّر إكمال الفحص: {exc}")
        sys.exit(2)
