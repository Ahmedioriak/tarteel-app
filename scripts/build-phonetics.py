"""Build phonetics.json: for every ayah, what a correct Hafs recitation sounds like in the Quran Phonetic Script,
the span of each word in it, and the tajweed units a teacher listens for (madd, qalqalah, ghunnah, ikhfa, iqlab,
idgham, shadda).

The unit logic follows `expected()` in HisnLabs tajweed-v2 `imam.py` (Apache-2.0,
https://huggingface.co/HisnLabs/tajweed-v2), and the phonetic script comes from quran-transcript (MIT).

    pip install quran-transcript
    python3 scripts/build-phonetics.py
"""
import json
import sys
from pathlib import Path

from quran_transcript import Aya, MoshafAttributes, quran_phonetizer

ROOT = Path(__file__).resolve().parent.parent

LONG = {"ا": "alif", "ۥ": "waw", "ۦ": "yaa"}
HARAKAT = set("َُِ")
NASAL = set("نم")

# Hafs ʿan ʿAsim: the lengths a teacher accepts (the turuq of the Shatibiyya and the Tayyiba), as in imam.py
MADD_ALLOWED = {
    "NormalMaddRule": [2],
    "MonfaselMaddRule": [2, 4, 5],
    "MottaselMaddRule": [4, 5, 6],
    "MottaselMaddPauseRule": [4, 5, 6],
    "LazemMaddRule": [6],
    "AaredMaddRule": [2, 4, 6],
    "LeenMaddRule": [2, 4, 6],
}
RULES = list(MADD_ALLOWED)  # rule index used in the JSON
KINDS = ["madd", "qalqalah", "ikhfa", "iqlab", "ghunnah", "idgham_ghunnah", "idgham", "shadda"]


def moshaf():
    return MoshafAttributes(
        rewaya="hafs", madd_monfasel_len=4, madd_mottasel_len=4, madd_mottasel_waqf=4, madd_aared_len=4
    )


def expected(words, m):
    """Port of imam.expected (Hafs only): phonemes, per-word spans, tajweed units."""
    text = " ".join(words)
    o = quran_phonetizer(text, m, remove_spaces=True)
    ph, mp = o.phonemes, o.mappings
    spans, pos = [], 0
    for w in words:
        ps = [x.pos for x in mp[pos : pos + len(w)] if x.pos[1] > x.pos[0]]
        spans.append(
            (min(p[0] for p in ps), max(p[1] for p in ps)) if ps else (spans[-1][1] if spans else 0,) * 2
        )
        pos += len(w) + 1

    def word_at(i):
        return next((k for k, (a, b) in enumerate(spans) if a <= i < b), len(spans) - 1)

    starts = [sum(len(x) + 1 for x in words[:k]) for k in range(len(words))]
    vanishing = [
        any(x.deleted for x in mp[p0 : p0 + len(w)][-3:])
        and (w.rstrip("ْۡ").endswith("ن") or any(c in w for c in "ًࣰٌࣱٍࣲ"))
        for w, p0 in zip(words, starts)
    ]

    units, ruled = [], set()
    for x in mp:
        for r in x.tajweed_rules or []:
            a, b = x.pos
            name = type(r).__name__
            if name in MADD_ALLOWED:
                seg = ph[a:b]
                c = next((ch for ch in seg if ch in LONG), None) or (seg[0] if name == "LeenMaddRule" else None)
                if c is None:
                    continue
                j = seg.index(c)
                k = j
                while k < len(seg) and seg[k] == c:
                    k += 1
                golden = k - j + (name == "LeenMaddRule")
                units.append(
                    dict(s=a + j, e=a + k, kind="madd", word=word_at(a), golden=golden, rule=name, char=c)
                )
                ruled.update(range(a + j, a + k))
            elif name == "Qalqalah":
                q = ph.find("ڇ", a, b)
                if q >= 0:
                    units.append(dict(s=q, e=q + 1, kind="qalqalah", word=word_at(q), golden=0, rule="", char="ڇ"))
                    ruled.add(q)
    i = 0
    while i < len(ph):
        j = i
        while j < len(ph) and ph[j] == ph[i]:
            j += 1
        c, n, w = ph[i], j - i, word_at(i)
        if i not in ruled:
            if c == "ں":
                units.append(dict(s=i, e=j, kind="ikhfa", word=w, golden=n, rule="", char=c))
            elif c == "۾":
                units.append(dict(s=i, e=j, kind="iqlab", word=w, golden=n, rule="", char=c))
            elif c in NASAL and n >= 3:
                prev_w = word_at(i - 1) if i else w
                kind = "idgham_ghunnah" if (prev_w != w and vanishing[prev_w]) else "ghunnah"
                units.append(
                    dict(s=i, e=j, kind=kind, word=prev_w if kind == "idgham_ghunnah" else w, golden=n, rule="", char=c)
                )
            elif c in "يو" and n >= 3:
                units.append(dict(s=i, e=j, kind="idgham_ghunnah", word=w, golden=n, rule="", char=c))
            elif c not in LONG and c not in HARAKAT and n == 2:
                prev_w = word_at(i - 1) if i else w
                if i == spans[w][0] and prev_w != w and vanishing[prev_w]:
                    units.append(dict(s=i, e=j, kind="idgham", word=prev_w, golden=2, rule="", char=c))
                else:
                    units.append(dict(s=i, e=j, kind="shadda", word=w, golden=2, rule="", char=c))
        i = j
    units.sort(key=lambda u: u["s"])
    return ph, spans, units


def pack_units(units):
    out = []
    for u in units:
        out.append(
            [u["s"], u["e"], KINDS.index(u["kind"]), u["word"], u["golden"], RULES.index(u["rule"]) if u["rule"] else -1, u["char"]]
        )
    return out


def skeleton(w):
    import re
    w = re.sub(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED\u08D3-\u08FF\u0640\s]", "", w)
    for c in "ٱأإآ":
        w = w.replace(c, "ا")
    return w.replace("ى", "ي").replace("ی", "ي")


def word_map(ref_words, disp_words):
    """ref word index -> displayed word index, for the few ayat where the two texts split words differently
    (e.g. لَّوْ مَا vs لَّوۡمَا). Words are matched by their consonant skeletons, two pointers."""
    rs = [skeleton(w) for w in ref_words]
    ds = [skeleton(w) for w in disp_words]
    out, i, j = [], 0, 0
    acc_r, acc_d = "", ""
    while i < len(rs):
        if not acc_d:
            acc_d = ds[j] if j < len(ds) else ""
        acc_r += rs[i]
        out.append(min(j, len(ds) - 1))
        i += 1
        if len(acc_r) >= len(acc_d):
            # the display word is complete; if the reference word ran past it, the display had a split
            while len(acc_r) > len(acc_d) and j + 1 < len(ds):
                j += 1
                acc_d += ds[j]
            j += 1
            acc_r, acc_d = "", ""
    return out


def main():
    quran = json.loads((ROOT / "quran.json").read_text("utf8"))
    m = moshaf()
    ayat, mism = [], []
    for s, (name, n) in enumerate(quran["surahs"], start=1):
        for a in range(1, n + 1):
            g = Aya(s, a).get()
            words = g.uthmani.split(" ")
            ph, spans, units = expected(words, m)
            entry = [ph, [x for sp in spans for x in sp], pack_units(units), len(words)]
            disp = quran["ayahs"][len(ayat)][4].split()
            if len(disp) != len(words):
                entry.append(word_map(words, disp))
            ayat.append(entry)
        print(f"\r{s}/114", end="", file=sys.stderr)
    print(file=sys.stderr)

    # isti'adha and basmala as recited before a passage: breaths that match them are not judged
    g = Aya(2, 1).get()  # Aya(1, 1) is the basmala itself, so it carries no separate basmala text
    extras = {}
    for key, text in (("istiatha", g.istiaatha_uthmani), ("basmala", g.bismillah_uthmani)):
        ph, _, _ = expected(text.split(" "), m)
        extras[key] = ph

    out = {
        "src": "Phonetic script: quran-transcript (MIT). Tajweed units: port of HisnLabs tajweed-v2 imam.expected (Apache-2.0). Hafs, murattal, munfasil/muttasil 4.",
        "kinds": KINDS,
        "rules": RULES,
        "allowed": [MADD_ALLOWED[r] for r in RULES],
        "extras": extras,
        "ayat": ayat,
    }
    (ROOT / "phonetics.json").write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")), "utf8")
    print("ayat:", len(ayat), file=sys.stderr)


if __name__ == "__main__":
    main()
