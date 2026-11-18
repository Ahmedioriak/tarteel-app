/* رفيق الحفظ — محرك التسميع الدقيق (الحروف والحركات والمدود)
 *
 * Runs only on the app's own site (GitHub Pages): it needs the microphone and downloads a model.
 *
 *   microphone → 16 kHz → breaths (pauses) → Whisper log-mel (80 bins, as faster-whisper)
 *   → HisnLabs tajweed-v2 ONNX (CTC, Quran Phonetic Script) → what was heard, symbol by symbol
 *   → aligned with what a correct Hafs recitation sounds like (phonetics.json, built by scripts/build-phonetics.py)
 *   → per-word remarks: another letter, another vowel, a dropped / added sound, a skipped word (لحن جلي),
 *     madd length, ghunnah, ikhfa, iqlab, idgham, qalqalah, shadda (لحن خفي).
 *
 * The remark rules are a port of `listen()` in HisnLabs tajweed-v2 imam.py (Apache-2.0):
 * https://huggingface.co/HisnLabs/tajweed-v2 — model, rule engine and code by HisnLabs.
 */
(function (global) {
  "use strict";

  const MODEL_URL = "https://huggingface.co/HisnLabs/tajweed-v2/resolve/main/onnx/model-int8.onnx";
  const ORT_VER = "1.31.1";
  const ORT_BASE = "https://cdn.jsdelivr.net/npm/onnxruntime-web@" + ORT_VER + "/dist/";
  const PHONEMES = ["ء","ب","ت","ث","ج","ح","خ","د","ذ","ر","ز","س","ش","ص","ض","ط","ظ","ع","غ","ف","ق","ك","ل","م","ن","ه","و","ي","ا","ۦ","ۥ","َ","ُ","ِ","۪","ـ","ٲ","ڇ","ں","۾","ۜ","ؙ"];
  const SR = 16000, HOP = 160, NFFT = 400, NMEL = 80;
  const LEAD = Math.round(SR * 0.4); // the training clips start and end with silence

  /* ------------------------------------------------------------------ */
  /* 1. Log-mel features, numerically the same as faster-whisper's FeatureExtractor(feature_size=80)(x, padding=0) */
  /* ------------------------------------------------------------------ */
  const MEL = (function melFilters() {
    const nBins = NFFT / 2 + 1;
    const fftfreqs = new Float64Array(nBins);
    for (let i = 0; i < nBins; i++) fftfreqs[i] = i * SR / NFFT;
    const minMel = 0, maxMel = 45.245640471924965, n = NMEL + 2;
    const fSp = 200 / 3, minLogHz = 1000, minLogMel = minLogHz / fSp, logstep = Math.log(6.4) / 27;
    const freqs = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const m = minMel + (maxMel - minMel) * i / (n - 1);
      freqs[i] = m >= minLogMel ? minLogHz * Math.exp(logstep * (m - minLogMel)) : fSp * m;
    }
    const W = new Float32Array(NMEL * nBins);
    for (let m = 0; m < NMEL; m++) {
      const enorm = 2 / (freqs[m + 2] - freqs[m]);
      for (let k = 0; k < nBins; k++) {
        const lower = -(freqs[m] - fftfreqs[k]) / (freqs[m + 1] - freqs[m]);
        const upper = (freqs[m + 2] - fftfreqs[k]) / (freqs[m + 2] - freqs[m + 1]);
        W[m * nBins + k] = Math.max(0, Math.min(lower, upper)) * enorm;
      }
    }
    // sparse rows: [start, end) of the non-zero bins of each filter
    const rows = [];
    for (let m = 0; m < NMEL; m++) {
      let s = -1, e = -1;
      for (let k = 0; k < nBins; k++) if (W[m * nBins + k] > 0) { if (s < 0) s = k; e = k + 1; }
      rows.push([Math.max(0, s), Math.max(0, e)]);
    }
    return { W, rows, nBins };
  })();
  const HANN = (function () { const w = new Float32Array(NFFT); for (let i = 0; i < NFFT; i++) w[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / NFFT); return w; })();

  // Mixed-radix complex FFT for n = 400 (2^4 · 5^2), recursive decimation in time with precomputed twiddles.
  const FFT = (function makeFFT(n) {
    const factors = []; let m = n;
    for (const p of [4, 2, 5, 3]) while (m % p === 0) { factors.push(p); m /= p; }
    if (m !== 1) factors.push(m);
    const cos = new Float64Array(n), sin = new Float64Array(n);
    for (let i = 0; i < n; i++) { cos[i] = Math.cos(-2 * Math.PI * i / n); sin[i] = Math.sin(-2 * Math.PI * i / n); }
    const scratchR = [], scratchI = [];
    function rec(inR, inI, inOff, stride, len, outR, outI, outOff, depth) {
      if (len === 1) { outR[outOff] = inR[inOff]; outI[outOff] = inI[inOff]; return; }
      const p = factors[depth], sub = len / p;
      for (let r = 0; r < p; r++) rec(inR, inI, inOff + r * stride, stride * p, sub, outR, outI, outOff + r * sub, depth + 1);
      if (!scratchR[depth]) { scratchR[depth] = new Float64Array(n); scratchI[depth] = new Float64Array(n); }
      const tR = scratchR[depth], tI = scratchI[depth];
      const tw = n / len; // twiddle step for this length
      for (let k = 0; k < sub; k++) {
        for (let q = 0; q < p; q++) {
          const kk = k + q * sub; let sr = 0, si = 0;
          for (let r = 0; r < p; r++) {
            const xr = outR[outOff + r * sub + k], xi = outI[outOff + r * sub + k];
            const idx = ((r * kk) % len) * tw;
            const c = cos[idx], s = sin[idx];
            sr += xr * c - xi * s; si += xr * s + xi * c;
          }
          tR[kk] = sr; tI[kk] = si;
        }
      }
      for (let i = 0; i < len; i++) { outR[outOff + i] = tR[i]; outI[outOff + i] = tI[i]; }
    }
    const zeros = new Float64Array(n), oR = new Float64Array(n), oI = new Float64Array(n);
    return function (re) { rec(re, zeros, 0, 1, n, oR, oI, 0, 0); return [oR, oI]; };
  })(NFFT);

  /** audio: Float32Array at 16 kHz (the LEAD silence already added). Returns {data: Float32Array(80*T), frames: T} */
  function logMel(audio) {
    const L = audio.length, pad = NFFT / 2;
    const total = Math.floor(L / HOP); // 1 + L//HOP frames, the last one dropped
    const frames = Math.min(3000, total - (total % 2));
    const nb = MEL.nBins, buf = new Float64Array(NFFT), power = new Float64Array(nb);
    const out = new Float32Array(NMEL * frames);
    const reflect = (i) => { if (i < 0) i = -i; if (i >= L) i = 2 * (L - 1) - i; return audio[i] || 0; };
    let gmax = -Infinity;
    for (let t = 0; t < frames; t++) {
      const start = t * HOP - pad;
      for (let i = 0; i < NFFT; i++) buf[i] = reflect(start + i) * HANN[i];
      const [R, I] = FFT(buf);
      for (let k = 0; k < nb; k++) power[k] = R[k] * R[k] + I[k] * I[k];
      for (let m = 0; m < NMEL; m++) {
        const [s, e] = MEL.rows[m]; let acc = 0;
        for (let k = s; k < e; k++) acc += MEL.W[m * nb + k] * power[k];
        const v = Math.log10(Math.max(acc, 1e-10));
        out[m * frames + t] = v; if (v > gmax) gmax = v;
      }
    }
    const floor = gmax - 8;
    for (let i = 0; i < out.length; i++) out[i] = (Math.max(out[i], floor) + 4) / 4;
    return { data: out, frames };
  }

  /* ------------------------------------------------------------------ */
  /* 2. Model                                                             */
  /* ------------------------------------------------------------------ */
  let session = null, ortLoading = null;
  function loadScript(src) {
    return new Promise((res, rej) => { const s = document.createElement("script"); s.src = src; s.onload = res; s.onerror = () => rej(new Error("script " + src)); document.head.appendChild(s); });
  }
  async function fetchCached(url, onProgress) {
    let cache = null;
    try { cache = await caches.open("hifz-tasmee-v1"); const hit = await cache.match(url); if (hit) return new Uint8Array(await hit.arrayBuffer()); } catch (e) { cache = null; }
    const r = await fetch(url);
    if (!r.ok) throw new Error("download " + r.status);
    const total = +r.headers.get("content-length") || 0;
    const reader = r.body.getReader(); const parts = []; let got = 0;
    for (;;) { const { done, value } = await reader.read(); if (done) break; parts.push(value); got += value.length; if (onProgress) onProgress(got, total); }
    const buf = new Uint8Array(got); let o = 0; for (const p of parts) { buf.set(p, o); o += p.length; }
    if (cache) { try { await cache.put(url, new Response(buf, { headers: { "content-type": "application/octet-stream" } })); } catch (e) {} }
    return buf;
  }
  async function loadModel(onProgress) {
    if (session) return session;
    if (!ortLoading) ortLoading = (global.ort ? Promise.resolve() : loadScript(ORT_BASE + "ort.wasm.min.js"));
    await ortLoading;
    const ort = global.ort;
    ort.env.wasm.wasmPaths = ORT_BASE;
    ort.env.wasm.numThreads = global.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 2) : 1;
    const bytes = await fetchCached(MODEL_URL, onProgress);
    session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
    return session;
  }

  /** samples: Float32Array 16 kHz (one breath, < ~29 s). → {heard, probs, times} as tajweed-v2 inference.hear() */
  async function hear(samples) {
    const ort = global.ort;
    const audio = new Float32Array(samples.length + 2 * LEAD); audio.set(samples, LEAD);
    const { data, frames } = logMel(audio);
    if (frames < 2) return { heard: "", probs: [], times: [0] };
    const out = await session.run({
      mel: new ort.Tensor("float32", data, [1, NMEL, frames]),
      riwaya: new ort.Tensor("int64", BigInt64Array.from([0n]), [1]),
    });
    const lp = out.logprobs || out[Object.keys(out)[0]];
    const [, T, V] = lp.dims, d = lp.data;
    const step = audio.length / SR / T;
    let heard = "", prev = 0, last = -1; const probs = [], times = [];
    for (let t = 0; t < T; t++) {
      let best = 0, bv = -Infinity;
      for (let v = 0; v < V; v++) { const x = d[t * V + v]; if (x > bv) { bv = x; best = v; } }
      if (best !== 0) last = t;
      if (best !== prev && best !== 0) { heard += PHONEMES[best - 1]; probs.push(Math.exp(bv)); times.push((t + 0.5) * step - LEAD / SR); }
      prev = best;
    }
    times.push(last >= 0 ? (last + 1) * step - LEAD / SR : 0);
    return { heard, probs, times };
  }

  /* ------------------------------------------------------------------ */
  /* 3. Reference data                                                    */
  /* ------------------------------------------------------------------ */
  let PH = null; // phonetics.json
  async function loadPhonetics(url) {
    if (PH) return PH;
    const r = await fetch(url || "phonetics.json"); PH = await r.json();
    return PH;
  }
  const KINDS = () => PH.kinds, RULES = () => PH.rules;
  const MADD_NAME = {
    NormalMaddRule: "المد الطبيعي", MonfaselMaddRule: "المد المنفصل", MottaselMaddRule: "المد المتصل",
    MottaselMaddPauseRule: "المد المتصل وقفًا", LazemMaddRule: "المد اللازم", AaredMaddRule: "المد العارض للسكون", LeenMaddRule: "مد اللين",
  };

  /** A passage: consecutive ayat (global indices into quran.json order) as one expected script. */
  function unpackUnits(us, off, w0) {
    return us.map(u => { const rule = u[5] >= 0 ? PH.rules[u[5]] : ""; return { s: u[0] + off, e: u[1] + off, kind: PH.kinds[u[2]], word: u[3] + w0, golden: u[4], rule, allowed: rule ? PH.allowed[u[5]] : [], char: u[6] }; });
  }
  function buildPassage(ayahIdx) {
    let ph = ""; const spans = [], units = [], words = [], ayahStart = [], ayahEnd = [], forms = [], first = [], last = [];
    for (const g of ayahIdx) {
      const [p, sp, us, nw, wm, fm] = PH.ayat[g];
      const off = ph.length, w0 = words.length;
      ayahStart.push(off);
      for (let k = 0; k < nw; k++) {
        spans.push([sp[2 * k] + off, sp[2 * k + 1] + off]); words.push({ ayah: g, w: wm ? wm[k] : k });
        const f = fm && fm[k]; forms.push(f ? { W: f[0], Wu: f[1], I: f[2], Iu: f[3] } : null);
        first.push(k === 0); last.push(k === nw - 1);
      }
      for (const u of us) {
        const kind = PH.kinds[u[2]], rule = u[5] >= 0 ? PH.rules[u[5]] : "";
        units.push({ s: u[0] + off, e: u[1] + off, kind, word: u[3] + w0, golden: u[4], rule, allowed: rule ? PH.allowed[u[5]] : [], char: u[6] });
      }
      ph += p; ayahEnd.push(ph.length);
    }
    return { ayahIdx, ph, spans, units, words, ayahStart, ayahEnd, forms, first, last };
  }

  /** The expected script of a breath made of segments [a, b, startForm, endForm] (passage words a..b; 'I' = the
   * first word as restarted (ibtida), 'W' = the last word as stopped on (waqf), 'c' = as inside the ayah).
   * → {ph, spans, units, occ: passage word of each occurrence, segStart: [ph index], segEnd: [ph index]} */
  function compose(P, segs) {
    let ph = ""; const spans = [], units = [], occ = [], segStart = [], segEnd = [];
    const addForm = (k, f, fu) => { const o = ph.length; ph += f; spans.push([o, ph.length]); occ.push(k); for (const u of unpackUnits(fu, o, 0)) units.push(Object.assign(u, { word: occ.length - 1 })); };
    for (const [a, b, sf, ef] of segs) {
      segStart.push(ph.length);
      const useI = sf === "I" && P.forms[a] && P.forms[a].I, useW = ef === "W" && P.forms[b] && P.forms[b].W;
      if (a === b && (useW || useI)) { const f = P.forms[a]; useW ? addForm(a, f.W, f.Wu) : addForm(a, f.I, f.Iu); segEnd.push(ph.length); continue; }
      let x = a, y = b;
      if (useI) { addForm(a, P.forms[a].I, P.forms[a].Iu); x = a + 1; }
      if (useW) y = b - 1;
      if (x <= y) {
        const s0 = P.spans[x][0], e0 = P.spans[y][1], o = ph.length, w0 = occ.length;
        ph += P.ph.slice(s0, e0);
        for (let k = x; k <= y; k++) { spans.push([P.spans[k][0] - s0 + o, P.spans[k][1] - s0 + o]); occ.push(k); }
        for (const u of P.units) if (u.s >= s0 && u.e <= e0 && u.word >= x && u.word <= y) units.push(Object.assign({}, u, { s: u.s - s0 + o, e: u.e - s0 + o, word: u.word - x + w0 }));
      }
      if (useW) addForm(b, P.forms[b].W, P.forms[b].Wu);
      segEnd.push(ph.length);
    }
    units.sort((p, q) => p.s - q.s);
    return { ph, spans, units, occ, segStart, segEnd };
  }

  /** Port of imam.recited(): where inside one breath the reciter stopped (waqf) and from where they started again
   * (ibtida, often a few words back). Greedy: a split must explain at least `margin` more sounds. */
  function recited(P, ws, we, heard, startForm, endForm, isWaqf, back = 8, margin = 3) {
    let segs = [[ws, we, startForm, endForm]];
    let cur = compose(P, segs), best = levDistance(cur.ph, heard);
    if (best === 0 || we - ws > 120) return { segs, cost: best };
    for (let iter = 0; iter < 3; iter++) {
      // only stops near where the breath disagrees with the expected script are worth trying
      const hot = new Set();
      for (const [t, i1, i2] of opcodes(cur.ph, heard)) {
        if (t === "equal") continue;
        let o1 = 0; while (o1 < cur.spans.length - 1 && cur.spans[o1][1] <= i1) o1++;
        let o2 = o1; while (o2 < cur.spans.length - 1 && cur.spans[o2][1] < i2) o2++;
        for (let o = Math.max(0, o1 - 2); o <= Math.min(cur.occ.length - 1, o2 + 1); o++) hot.add(cur.occ[o]);
      }
      let cand = null, cc = best;
      for (let i = 0; i < segs.length; i++) {
        const [a, b, sf, ef] = segs[i];
        for (let k = a; k < b; k++) {          // stop on word k ...
          if (!hot.has(k) && !hot.has(k + 1)) continue;
          for (let j = Math.max(a, k - back + 1); j <= k + 1; j++) { // ... start again from word j (j <= k: a repeat)
            const ns = segs.slice(0, i).concat([[a, k, sf, "W"], [j, b, "I", ef]], segs.slice(i + 1));
            // a stop where the mushaf marks a pause (ۚ ۖ ۗ ...) needs less evidence than one anywhere else
            const need = j === k + 1 && isWaqf && isWaqf(k) ? 1 : margin;
            const c = levDistance(compose(P, ns).ph, heard) + need - margin; if (c < cc) { cc = c; cand = ns; }
          }
        }
      }
      if (!cand || cc > best - margin) break;
      best = cc; segs = cand; cur = compose(P, segs);
      if (best === 0) break;
    }
    return { segs, cost: best };
  }

  /* ------------------------------------------------------------------ */
  /* 4. Alignment                                                         */
  /* ------------------------------------------------------------------ */
  const HARAKAT = new Set(["َ", "ُ", "ِ"]);
  const LONG = new Set(["ا", "ۥ", "ۦ"]);

  /** Levenshtein opcodes, identical to python-Levenshtein's opcodes(): [tag, i1, i2, j1, j2], tags equal/replace/insert/delete.
   * Same as the C library: common prefix and suffix stripped, then the edit path traced back from the end preferring
   * delete > replace > insert > equal (checked equal on 400 random phonetic strings). */
  function opcodes(A, B) {
    let pre = 0; while (pre < A.length && pre < B.length && A[pre] === B[pre]) pre++;
    let suf = 0; while (suf < A.length - pre && suf < B.length - pre && A[A.length - 1 - suf] === B[B.length - 1 - suf]) suf++;
    const a = A.slice(pre, A.length - suf), b = B.slice(pre, B.length - suf);
    const n = a.length, m = b.length, W = m + 1;
    const D = new Uint32Array((n + 1) * W);
    for (let i = 0; i <= n; i++) D[i * W] = i;
    for (let j = 0; j <= m; j++) D[j] = j;
    for (let i = 1; i <= n; i++) {
      const ai = a[i - 1];
      for (let j = 1; j <= m; j++) {
        const sub = D[(i - 1) * W + j - 1] + (ai === b[j - 1] ? 0 : 1);
        const del = D[(i - 1) * W + j] + 1, ins = D[i * W + j - 1] + 1;
        D[i * W + j] = sub < del ? (sub < ins ? sub : ins) : (del < ins ? del : ins);
      }
    }
    const ops = []; let i = n, j = m;
    while (i > 0 || j > 0) {
      const cur = D[i * W + j];
      if (i > 0 && cur === D[(i - 1) * W + j] + 1) { ops.push(["delete", i - 1, j]); i--; }
      else if (i > 0 && j > 0 && a[i - 1] !== b[j - 1] && cur === D[(i - 1) * W + j - 1] + 1) { ops.push(["replace", i - 1, j - 1]); i--; j--; }
      else if (j > 0 && cur === D[i * W + j - 1] + 1) { ops.push(["insert", i, j - 1]); j--; }
      else { ops.push(["equal", i - 1, j - 1]); i--; j--; }
    }
    ops.reverse();
    const full = [];
    for (let k = 0; k < pre; k++) full.push(["equal", k, k]);
    for (const [t, x, y] of ops) full.push([t, x + pre, y + pre]);
    for (let k = 0; k < suf; k++) full.push(["equal", pre + n + k, pre + m + k]);
    const out = [];
    for (const [tag, x, y] of full) {
      const last = out[out.length - 1];
      if (last && last[0] === tag && last[2] === x && last[4] === y) { if (tag !== "insert") last[2]++; if (tag !== "delete") last[4]++; }
      else out.push([tag, x, tag === "insert" ? x : x + 1, y, tag === "delete" ? y : y + 1]);
    }
    return out;
  }
  function levDistance(a, b) { let d = 0; for (const o of opcodes(a, b)) if (o[0] !== "equal") d += Math.max(o[2] - o[1], o[4] - o[3]); return d; }

  /** Semi-global alignment of `heard` inside ref[from, to): free end in the reference; the start is free up to `ptr`
   * (going back to repeat is normal) and costs 0.8 per sound after it (skipping ahead is a mistake, not a shortcut).
   * Without `ptr` the start is free everywhere (searching). Cheap substitution between two vowels / two length marks.
   * Gaps in the reference are affine (open 2, then 0.35 a sound): a word left out is one gap, not a string of wrong
   * letters, so the breath still lines up with the words after it. → {s, e, cost} (ref slice) */
  function locate(ref, from, to, heard, ptr) {
    const n = heard.length, m = to - from, W = m + 1;
    if (!n || m <= 0) return { s: from, e: from, cost: n };
    const OPEN = 2, EXT = 0.35, INF = 1e9;
    const M = new Float32Array((n + 1) * W).fill(INF), G = new Float32Array((n + 1) * W).fill(INF), I = new Float32Array((n + 1) * W).fill(INF);
    const SM = new Int32Array((n + 1) * W), SG = new Int32Array((n + 1) * W), SI = new Int32Array((n + 1) * W);
    for (let j = 0; j <= m; j++) { M[j] = ptr == null || from + j <= ptr ? 0 : (from + j - ptr) * 0.8; SM[j] = j; }
    const best3 = (x) => { let v = M[x], st = SM[x]; if (G[x] < v) { v = G[x]; st = SG[x]; } if (I[x] < v) { v = I[x]; st = SI[x]; } return [v, st]; };
    for (let i = 0; i <= n; i++) {
      const h = i ? heard[i - 1] : "";
      for (let j = 0; j <= m; j++) {
        const x = i * W + j;
        if (i > 0 && j > 0) { // heard i against ref j
          const r = ref[from + j - 1];
          const c = h === r ? 0 : ((HARAKAT.has(h) && HARAKAT.has(r)) || (LONG.has(h) && LONG.has(r)) ? 0.5 : 1);
          const [v, st] = best3((i - 1) * W + j - 1); M[x] = v + c; SM[x] = st;
        }
        if (i > 0) { // heard i not in the reference
          const y = (i - 1) * W + j; let v = M[y] + 1, st = SM[y];
          if (I[y] + 1 < v) { v = I[y] + 1; st = SI[y]; } if (G[y] + 1 < v) { v = G[y] + 1; st = SG[y]; }
          I[x] = v; SI[x] = st;
        }
        if (j > 0 && i > 0) { // ref j left out (a gap inside the breath)
          const y = i * W + j - 1; let v = M[y] + OPEN, st = SM[y];
          if (G[y] + EXT < v) { v = G[y] + EXT; st = SG[y]; } if (I[y] + OPEN < v) { v = I[y] + OPEN; st = SI[y]; }
          G[x] = v; SG[x] = st;
        }
      }
    }
    const endC = new Float32Array(m + 1), endS = new Int32Array(m + 1);
    let bj = 0, bc = INF;
    for (let j = 0; j <= m; j++) { const x = n * W + j; let v = M[x], st = SM[x]; if (I[x] < v) { v = I[x]; st = SI[x]; } endC[j] = v; endS[j] = st; if (v < bc) { bc = v; bj = j; } }
    // a later end almost as good means the breath's last sounds matched real words further on (a word left out in
    // between), not that the reciter stopped early with a garbled word: prefer it
    for (let j = m; j > bj; j--) if (endC[j] <= bc + 2) { bj = j; break; }
    return { s: from + endS[bj], e: from + bj, cost: endC[bj] };
  }

  /* ------------------------------------------------------------------ */
  /* 5. Remarks — port of imam.listen() (Hafs), Arabic messages           */
  /* ------------------------------------------------------------------ */
  const ONSET = 3;
  const TIME_SIX = 8.3, TIME_FIVE = 7.0;
  const VOWEL_NAME = { "َ": "فتحة", "ُ": "ضمة", "ِ": "كسرة" };
  const COUNTS = { 2: "حركتان", 3: "ثلاث حركات", 4: "أربع حركات", 5: "خمس حركات", 6: "ست حركات" };
  const counts = (n) => n ? (COUNTS[n] || n + " حركات") : "بلا مد";
  const READ = { "ء": "همزة", "ا": "ألف", "ۥ": "واو مد", "ۦ": "ياء مد", "ڇ": "قلقلة", "ں": "نون مخفاة", "۾": "ميم مخفاة", "ٲ": "همزة مسهلة", "ـ": "ألف ممالة", "۪": "فتحة ممالة", "ۜ": "سكت", "ؙ": "ضمة مختلسة" };
  function show(ph) { // a phonetic snippet, readable: letters with their vowels, the special marks named
    let out = ""; for (const c of ph) { if (HARAKAT.has(c)) out += c; else if (READ[c] && c !== "ء") out += (out ? " " : "") + "(" + READ[c] + ")"; else out += c; }
    return out.trim();
  }
  function stripLen(s) { let out = ""; for (const c of s) { if (out && out[out.length - 1] === c) continue; out += c; } let r = ""; for (const c of out) if (!LONG.has(c) && c !== "ں" && c !== "ڇ" && c !== "۾") r += c; return r; }

  function boundaries(ref, heard) {
    const b = new Array(ref.length + 1).fill(0);
    for (const [tag, i1, i2, j1, j2] of opcodes(ref, heard)) {
      for (let k = i1; k <= i2; k++) b[k] = j1 + (i2 > i1 ? Math.round((k - i1) * (j2 - j1) / (i2 - i1)) : 0);
    }
    b[ref.length] = heard.length; return b;
  }
  function run(heard, a, b, c) {
    let k = -1; for (let x = a; x < b; x++) if (heard[x] === c) { k = x; break; }
    if (k < 0) return [0, a, a];
    let s = k, e = k; while (s > 0 && heard[s - 1] === c) s--; while (e < heard.length && heard[e] === c) e++;
    return [e - s, s, e];
  }
  function dur(heard, times, s, e) { const a = s && HARAKAT.has(heard[s - 1]) ? times[s - 1] : times[s]; return Math.max(0, times[Math.min(e, times.length - 1)] - a); }
  function tempoOf(ref, heard, b, times, tempo) {
    if (!times || times.length !== heard.length + 1) return null;
    const last = ref.spans.length - 1; const nat = [];
    for (const u of ref.units) if (u.kind === "madd" && u.rule === "NormalMaddRule" && u.word !== last) {
      const [n, s, e] = run(heard, Math.max(0, b[u.s] - 1), Math.min(heard.length, b[u.e] + 1), u.char);
      if (n >= 1 && n <= 3) nat.push(dur(heard, times, s, e));
    }
    const ok = nat.filter(d => d > 0.08 && d < 2.0);
    if (tempo) { tempo.push(...ok); if (tempo.length > 40) tempo.splice(0, tempo.length - 40); }
    const src = ok.length >= 2 ? ok : (tempo && tempo.length >= 3 ? tempo : []);
    if (!src.length) return null;
    const srt = [...src].sort((a, b) => a - b); return srt[Math.floor(srt.length / 2)] / 2;
  }

  /** ref: {ph, spans, units, words(any length)} for one breath. → notes per word: [{severity:'jali'|'khafi', rule, msg, pos}] */
  function listen(ref, heard, probs, times, opts) {
    opts = opts || {};
    const minProb = opts.minProb ?? 0.3, jaliThr = opts.minProbJali ?? 0.7;
    const R = ref.ph, nW = ref.spans.length;
    const b = boundaries(R, heard);
    const notes = Array.from({ length: nW }, () => []);
    const covered = new Uint8Array(R.length);
    let span = [0, 0], pos = 0;
    const unsure = (hs, he, thr) => { if (!probs || !probs.length) return false; const seg = probs.slice(Math.max(0, hs - 1), Math.min(probs.length, he + 1)); return seg.length > 0 && Math.min(...seg) < (thr ?? minProb); };
    const add = (w, rule, sev, msg) => { if (unsure(span[0], span[1])) return; notes[w].push({ rule, severity: sev, msg, pos }); };
    const moreNun = (u, hs, he) => (heard.slice(Math.max(0, hs - 2), he).split("ن").length - 1) > (R.slice(Math.max(0, u.s - 2), u.e).split("ن").length - 1);
    const unit = tempoOf(ref, heard, b, times, opts.tempo);

    for (const u of ref.units) {
      for (let x = u.s; x < u.e; x++) covered[x] = 1;
      const hs = b[u.s], he = b[u.e]; pos = u.s; span = [hs, he];
      if (u.kind === "madd") {
        let [n, rs, re] = run(heard, Math.max(0, hs - 1), Math.min(heard.length, he + 1), u.char);
        if (u.rule.startsWith("Leen") && n > 0) n += 1;
        if (unit && (n === 4 || n === 5) && times) { const tc = dur(heard, times, rs, re) / unit; n = tc >= TIME_SIX ? 6 : tc >= TIME_FIVE ? 5 : 4; }
        else if (opts.capLenient !== false && (n === 4 || n === 5) && u.allowed.some(a => a >= 6)) n = Math.max(n, Math.min(...u.allowed.filter(a => a >= n))); // no clock yet: 4 symbols may be a held 6
        const allowed = u.allowed; let near = allowed[0]; for (const a of allowed) if (Math.abs(a - n) < Math.abs(near - n)) near = a;
        const name = MADD_NAME[u.rule] || "المد";
        if (n === 0 && !u.rule.startsWith("Leen")) add(u.word, u.rule, "jali", name + ": المد محذوف");
        else if (Math.abs(n - near) > 1 && !unsure(hs, he)) {
          const want = allowed.includes(u.golden) ? u.golden : near;
          notes[u.word].push({ rule: u.rule, severity: "khafi", pos: u.s, msg: name + ": " + counts(n) + " بدل " + counts(want) });
        }
      } else if (u.kind === "qalqalah") {
        if (!heard.slice(Math.max(0, hs - 1), he + 1).includes("ڇ")) add(u.word, "qalqalah", "khafi", "القلقلة غير ظاهرة على " + (u.s ? R[u.s - 1] : ""));
      } else if (u.kind === "ikhfa") {
        const [n] = run(heard, hs, Math.max(he, hs + 1), "ں");
        if (n === 0 && heard.slice(hs, he + 1).includes("ن")) add(u.word, "ikhfa", "khafi", "إخفاء: أظهرتَ النون، والصواب إخفاؤها مع الغنة");
        else if (n && n < u.golden - 1) add(u.word, "ikhfa", "khafi", "إخفاء: الغنة قصيرة");
      } else if (u.kind === "iqlab") {
        if (!heard.slice(Math.max(0, hs - 1), he + 1).includes("۾")) add(u.word, "iqlab", "khafi", "إقلاب: تُقلب النون ميمًا مخفاة مع الغنة");
      } else if (u.kind === "ghunnah" || u.kind === "idgham_ghunnah") {
        const [n] = run(heard, Math.max(0, hs - 1), Math.min(heard.length, he + 1), u.char);
        if (u.kind === "idgham_ghunnah" && u.char !== "ن" && moreNun(u, hs, he)) add(u.word, "idgham", "khafi", "إدغام: أظهرتَ النون، والصواب إدغامها في " + u.char);
        else if (n < u.golden) add(u.word, u.kind, "khafi", u.kind === "ghunnah" ? "الغنة ناقصة: أطِل الغنة حركتين" : "إدغام بغنة: الغنة ناقصة");
      } else if (u.kind === "shadda" || u.kind === "idgham") {
        const [n] = run(heard, Math.max(0, hs - 1), Math.min(heard.length, he + 1), u.char);
        if (u.kind === "idgham" && moreNun(u, hs, he)) add(u.word, "idgham", "khafi", "إدغام: أظهرتَ النون، والصواب إدغامها في " + u.char);
        else if (n === 1) add(u.word, "shadda", "jali", "الشدة على «" + u.char + "» غير ظاهرة");
      }
    }

    const SPECIAL = new Set([...LONG, "ڇ", "ں", "۾", "ن", "م"]);
    for (const [tag, i1, i2, j1, j2] of opcodes(R, heard)) {
      if (tag === "equal") continue;
      const exp = R.slice(i1, i2), got = heard.slice(j1, j2);
      if (i2 > i1) {
        let allCov = true; for (let x = i1; x < i2; x++) if (!covered[x]) { allCov = false; break; }
        if (allCov) { const expSet = new Set(exp); let sub = true; for (const c of got) if (!expSet.has(c) && !SPECIAL.has(c)) { sub = false; break; } if (!got || sub) continue; }
      }
      if (tag === "insert" && ref.units.some(u => u.kind === "madd" && u.s <= i1 && i1 <= u.e && [...got].every(c => c === u.char))) continue;
      let w = nW - 1; for (let k = 0; k < nW; k++) if (ref.spans[k][0] <= i1 && i1 < ref.spans[k][1]) { w = k; break; }
      pos = i1; span = [j1, j2];
      if (i1 < ONSET) continue;
      if (opts.endGrace && i1 >= R.length - opts.endGrace) continue; // the end of a breath: its waqf form
      if (opts.skip && opts.skip(i1)) continue;
      const e = stripLen(exp), g = stripLen(got);
      if (e === g) continue;
      if (unsure(j1, j2, jaliThr)) continue;
      if (tag === "replace" && e.length === 1 && g.length === 1 && HARAKAT.has(e) && HARAKAT.has(g)) {
        const letter = consonantBefore(R, i1);
        notes[w].push({ rule: "vowel", severity: "jali", pos: i1, msg: "الحركة" + (letter ? " على «" + letter + "»" : "") + ": " + VOWEL_NAME[g] + " بدل " + VOWEL_NAME[e] });
      } else if (tag === "replace") {
        add(w, "letter", "jali", "نطقتَ «" + show(got) + "» بدل «" + show(exp) + "»");
      } else if (tag === "delete") {
        if (exp.length === 1 && (LONG.has(exp) || HARAKAT.has(exp))) continue;
        add(w, "dropped", "jali", "سقط: «" + show(exp) + "»");
      } else if (tag === "insert") {
        if (i1 === 0 || i1 === R.length) continue;
        if ([...got].every(c => LONG.has(c))) { if (got.length >= 2) add(w, "madd_added", "khafi", "مد زائد في غير موضعه"); continue; }
        if (got.length === 1 && HARAKAT.has(got)) continue;
        add(w, "added", "jali", "زيادة: «" + show(got) + "»");
      }
    }

    // a word whose sounds are (almost) all missing was skipped
    for (let k = 0; k < nW; k++) {
      const [a, z] = ref.spans[k];
      if (z > a && b[z] - b[a] <= Math.max(0, Math.floor((z - a) / 4))) notes[k] = [{ rule: "word", severity: "jali", pos: a, msg: "ما قريت هالكلمة" }];
    }
    return { notes, b };
  }
  function consonantBefore(R, i) { for (let k = i - 1; k >= 0; k--) { const c = R[k]; if (!HARAKAT.has(c) && !LONG.has(c) && c !== "ڇ") return c; } return ""; }

  /* ------------------------------------------------------------------ */
  /* 6. Finding where the reciter is: anywhere in the Quran               */
  /* ------------------------------------------------------------------ */
  function skeleton(ph) { let out = ""; for (const c of ph) { if (HARAKAT.has(c) || LONG.has(c) || c === "ڇ" || c === "ۜ" || c === "ؙ" || c === "۪") continue; if (out[out.length - 1] === c) continue; out += c; } return out; }
  let INDEX = null;
  function buildIndex() {
    if (INDEX) return INDEX;
    const grams = new Map();
    PH.ayat.forEach((a, g) => {
      const sk = skeleton(a[0]); const seen = new Set();
      for (let i = 0; i + 4 <= sk.length; i++) { const k = sk.slice(i, i + 4); if (seen.has(k)) continue; seen.add(k); let l = grams.get(k); if (!l) grams.set(k, l = []); l.push(g); }
    });
    INDEX = grams; return grams;
  }
  /** heard phonemes → best {ayah, score} or {ambiguous:true} or null */
  function findAyah(heard) {
    buildIndex();
    const sk = skeleton(heard); if (sk.length < 6) return { short: true };
    const votes = new Map();
    for (let i = 0; i + 4 <= sk.length; i++) { const l = INDEX.get(sk.slice(i, i + 4)); if (!l || l.length > 400) continue; for (const g of l) votes.set(g, (votes.get(g) || 0) + 1); }
    // a breath may run into the next ayah: give ayah g the votes of g+1 too (and try g-1 for every voted g: a very
    // short ayah such as «يسٓ» has no 4-grams of its own)
    const pool = new Set(); for (const g of votes.keys()) { pool.add(g); if (g > 0) pool.add(g - 1); }
    const cand = [...pool].map(g => [g, (votes.get(g) || 0) + 0.5 * (votes.get(g + 1) || 0)]).sort((a, b) => b[1] - a[1]).slice(0, 30);
    const scored = [];
    for (const [g] of cand) {
      const ref = PH.ayat[g][0] + (PH.ayat[g + 1] ? PH.ayat[g + 1][0] : "");
      const loc = locate(ref, 0, ref.length, heard);
      scored.push({ ayah: g, score: loc.cost / heard.length, s: loc.s, e: loc.e });
    }
    scored.sort((a, b) => a.score - b.score);
    if (!scored.length || scored[0].score > 0.4) return null;
    const best = scored[0], second = scored.find(x => x.ayah !== best.ayah && Math.abs(x.ayah - best.ayah) > 1);
    if (second && second.score - best.score < 0.06) return { ambiguous: true, best, second };
    return best;
  }
  function isExtra(heard) { // isti'adha or basmala
    for (const k of ["istiatha", "basmala"]) { const r = PH.extras[k]; const loc = locate(r, 0, r.length, heard); if (loc.cost / Math.max(1, heard.length) < 0.3 && (loc.e - loc.s) > r.length * 0.6) return k; }
    return null;
  }

  /* ------------------------------------------------------------------ */
  /* 7. Microphone and breaths                                            */
  /* ------------------------------------------------------------------ */
  const WORKLET = `class Cap extends AudioWorkletProcessor{process(i){const c=i[0]&&i[0][0];if(c)this.port.postMessage(c.slice(0));return true}}registerProcessor('hifz-cap',Cap);`;
  function Recorder(onBreath, onLevel, onInterim) {
    let ctx = null, stream = null, node = null, src = null, rate = SR, active = false;
    let cur = [], curLen = 0, speech = false, silence = 0, noise = 0.004, voiced = 0, pre = [], lastInterim = 0;
    const FR = 320; let acc = new Float32Array(0);
    function resample(x) {
      if (rate === SR) return x;
      const ratio = rate / SR, n = Math.floor(x.length / ratio), y = new Float32Array(n);
      for (let i = 0; i < n; i++) { // box filter over the source span, then pick: enough for speech
        const a = i * ratio, b = a + ratio; let s = 0, c = 0;
        for (let k = Math.floor(a); k < Math.min(x.length, Math.ceil(b)); k++) { s += x[k]; c++; }
        y[i] = c ? s / c : 0;
      }
      return y;
    }
    function frame(f) {
      let e = 0; for (let i = 0; i < f.length; i++) e += f[i] * f[i]; const rms = Math.sqrt(e / f.length);
      if (onLevel) onLevel(rms);
      const thr = Math.max(0.012, noise * 3.5), isV = rms > thr;
      if (!speech) {
        noise = isV ? noise : noise * 0.95 + rms * 0.05;
        pre.push(f); if (pre.length > 12) pre.shift(); // 240 ms before the voice starts
        voiced = isV ? voiced + 1 : 0;
        if (voiced >= 3) { speech = true; silence = 0; cur = pre.slice(); curLen = cur.reduce((s, x) => s + x.length, 0); pre = []; lastInterim = curLen; }
        return;
      }
      cur.push(f); curLen += f.length;
      silence = isV ? 0 : silence + 1;
      const secs = curLen / SR;
      if ((silence >= 23 && secs > 0.6) || secs >= 26) close(silence >= 23);
      else if (onInterim && curLen - lastInterim > SR * 2.5) { lastInterim = curLen; onInterim(join(cur, curLen)); }
    }
    function join(parts, len) { const y = new Float32Array(len); let o = 0; for (const p of parts) { y.set(p, o); o += p.length; } return y; }
    function close(byPause) {
      let y = join(cur, curLen);
      if (byPause) y = y.slice(0, Math.max(0, y.length - 320 * 18)); // keep ~100 ms of the pause
      speech = false; cur = []; curLen = 0; voiced = 0; pre = [];
      if (y.length > SR * 0.35) onBreath(y);
    }
    function push(chunk) {
      const x = resample(chunk);
      const merged = new Float32Array(acc.length + x.length); merged.set(acc); merged.set(x, acc.length);
      let o = 0; for (; o + FR <= merged.length; o += FR) frame(merged.slice(o, o + FR));
      acc = merged.slice(o);
    }
    return {
      async start() {
        stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true } });
        try { ctx = new AudioContext({ sampleRate: SR }); } catch (e) { ctx = new AudioContext(); }
        rate = ctx.sampleRate;
        src = ctx.createMediaStreamSource(stream);
        try {
          const url = URL.createObjectURL(new Blob([WORKLET], { type: "application/javascript" }));
          await ctx.audioWorklet.addModule(url);
          node = new AudioWorkletNode(ctx, "hifz-cap"); node.port.onmessage = (ev) => { if (active) push(ev.data); };
          src.connect(node);
        } catch (e) {
          node = ctx.createScriptProcessor(4096, 1, 1); node.onaudioprocess = (ev) => { if (active) push(ev.inputBuffer.getChannelData(0).slice(0)); };
          src.connect(node); node.connect(ctx.destination);
        }
        if (ctx.state === "suspended") await ctx.resume();
        active = true;
      },
      stop(flush) {
        if (flush && speech && curLen > SR * 0.35) close(false);
        active = false;
        try { src && src.disconnect(); node && node.disconnect(); } catch (e) {}
        try { stream && stream.getTracks().forEach(t => t.stop()); } catch (e) {}
        try { ctx && ctx.close(); } catch (e) {}
        speech = false; cur = []; curLen = 0;
      },
      get active() { return active; },
    };
  }

  /* ------------------------------------------------------------------ */
  /* 8. Session: follows the reciter through a passage                    */
  /* ------------------------------------------------------------------ */
  /** opts: {ayat: [global ayah idx] | null (free start), stay: bool, onEvent(ev)} */
  function Session(opts) {
    let P = null; // passage
    let ptr = 0;  // where the reciter is expected next (phoneme index in P.ph)
    const state = []; // per passage word: 0 pending, 1 ok, 2 error (jali), 3 khafi only, 4 fixed after an error, 5 skipped
    const wordNotes = [];
    const tempo = [];
    let pendingHeard = "", busy = false; const queue = [];
    const emit = (type, data) => opts.onEvent && opts.onEvent(Object.assign({ type }, data || {}));

    function setPassage(ayat, startPh) {
      P = buildPassage(ayat); ptr = startPh || 0;
      state.length = 0; wordNotes.length = 0;
      for (let k = 0; k < P.words.length; k++) { state.push(0); wordNotes.push([]); }
      emit("passage", { passage: P });
    }
    if (opts.ayat && opts.ayat.length) setPassage(opts.ayat, 0);

    function wordAt(i) { for (let k = 0; k < P.spans.length; k++) if (P.spans[k][0] <= i && i < P.spans[k][1]) return k; return P.spans.length - 1; }

    async function processBreath(samples, interim) {
      const t0 = performance.now();
      const { heard, probs, times } = await (opts.hear || hear)(samples);
      const ms = performance.now() - t0;
      if (!heard) return;
      if (!interim && isExtra(heard)) { emit("extra", { which: isExtra(heard) }); return; }
      if (!P) { // free start: find the ayah
        if (interim) return;
        const h = pendingHeard + heard;
        const f = findAyah(h);
        if (!f) { pendingHeard = ""; emit("notfound", { heard }); return; }
        if (f.short) { pendingHeard = h; emit("short", {}); return; }
        if (f.ambiguous) { pendingHeard = h.slice(-400); emit("ambiguous", {}); return; }
        pendingHeard = "";
        const g = f.ayah, list = []; // from this ayah to the end of its surah
        const sIdx = opts.surahOf(g); for (let k = g; k < PH.ayat.length && opts.surahOf(k) === sIdx; k++) list.push(k);
        setPassage(list, f.s);
        for (let k = 0; k < P.words.length && P.spans[k][1] <= f.s + 2; k++) { state[k] = 6; emit("word", { word: k, state: 6, notes: [] }); }
        emit("found", { ayah: g });
        return judge(heard, probs, times, false, ms, true);
      }
      return judge(heard, probs, times, interim, ms, false);
    }

    function judge(heard, probs, times, interim, ms, first) {
      const back = 80, from = Math.max(0, ptr - back), to = Math.min(P.ph.length, ptr + heard.length * 2 + 60);
      let loc = locate(P.ph, from, to, heard, ptr);
      let score = loc.cost / heard.length;
      if (score > 0.45 && !interim) { // not where we expected: the reciter may have jumped; look in the whole passage
        const wide = locate(P.ph, 0, P.ph.length, heard);
        if (wide.cost / heard.length < score - 0.1) { loc = wide; score = wide.cost / heard.length; }
      }
      const maxScore = heard.length < 15 ? 0.3 : 0.45; // a short breath must match closely
      if (score > maxScore) { if (!interim) emit("lost", { heard, score }); return; }
      // snap the slice to whole words
      let ws = wordAt(loc.s), we = wordAt(Math.max(loc.s, loc.e - 1));
      if (loc.e < P.spans[we][1] - Math.max(2, (P.spans[we][1] - P.spans[we][0]) * 0.4)) we--; // the last word was cut off
      if (we < ws) { if (!interim) emit("lost", { heard, score }); return; }

      if (interim) { // live: only show which words were heard, judge at the pause
        for (let k = ws; k <= we; k++) if (state[k] === 0) emit("heard", { word: k });
        return;
      }
      // words jumped over between where the reciter was expected and where this breath starts: not recited
      const pw = ptr < P.ph.length ? wordAt(ptr) : P.words.length;
      if (ws > pw) {
        for (let k = pw; k < ws; k++) if (state[k] !== 1 && state[k] !== 3 && state[k] !== 4 && state[k] !== 6) {
          state[k] = 5; wordNotes[k] = [{ rule: "word", severity: "jali", pos: P.spans[k][0], msg: "ما قريت هالكلمة" }];
          emit("word", { word: k, state: 5, notes: wordNotes[k] });
        }
        if (opts.stay) { ptr = P.spans[pw][0]; emit("breath", { heard, ms, from: ws, to: we, stopAt: pw, done: false }); return; }
      }
      // the breath as recited: restarted at ws (unless an ayah starts there), stopped on we (unless an ayah ends there),
      // plus any stop / restart inside it
      // the first / last word of the breath: restart / stop forms, or as joined if that matches the reciter better
      let sf = "c", ef = "c", bestC = Infinity;
      for (const a of P.first[ws] ? ["c"] : ["I", "c"]) for (const b of P.last[we] ? ["c"] : ["W", "c"]) {
        const c = levDistance(compose(P, [[ws, we, a, b]]).ph, heard); if (c < bestC) { bestC = c; sf = a; ef = b; }
      }
      const { segs } = recited(P, ws, we, heard, sf, ef, opts.isWaqf ? (k) => opts.isWaqf(P.words[k].ayah, P.words[k].w) : null);
      const ref = compose(P, segs);
      // around ayah boundaries and stops a reciter's join / pause changes a few sounds: not judged as clear mistakes
      const grace = new Set();
      const mark = (a, b) => { for (let x = a; x < b; x++) grace.add(x); };
      ref.occ.forEach((k, o) => { const [a, b] = ref.spans[o]; if (P.first[k]) mark(a, a + 3); if (P.last[k]) mark(b - 3, b); });
      for (const x of ref.segStart) mark(x, x + 3);
      for (const x of ref.segEnd) mark(x - 3, x);
      const { notes } = listen(ref, heard, probs, times, { tempo, endGrace: 3, skip: (i) => grace.has(i) });

      // apply, in recitation order (a repeated word: its later reading counts); with "stay", stop at the first clear
      // mistake that was not put right later in the same breath
      const lastOcc = new Map(); ref.occ.forEach((k, o) => lastOcc.set(k, o));
      let stopAt = -1;
      for (let o = 0; o < ref.occ.length; o++) {
        const k = ref.occ[o];
        if (lastOcc.get(k) !== o) { // an earlier reading of a repeated word
          const n0 = notes[o]; if (n0.some(x => x.severity === "jali") && state[k] !== 2) { state[k] = 2; wordNotes[k] = n0; }
          continue;
        }
        const n = notes[o], jali = n.some(x => x.severity === "jali"), skipped = n.some(x => x.rule === "word");
        if (opts.stay && stopAt >= 0) { if (state[k] === 0) emit("word", { word: k, state: 0, notes: [] }); continue; }
        const prev = state[k];
        let st = skipped ? 5 : jali ? 2 : n.length ? 3 : 1;
        if ((st === 1 || st === 3) && (prev === 2 || prev === 5)) st = 4; // corrected
        state[k] = st; wordNotes[k] = n;
        emit("word", { word: k, state: st, notes: n, prev });
        if (opts.stay && (st === 2 || st === 5)) stopAt = k;
      }
      ptr = stopAt >= 0 ? P.spans[stopAt][0] : P.spans[we][1];
      emit("breath", { heard, ms, from: ws, to: we, stopAt, done: ptr >= P.ph.length, segs: segs.length });
    }

    async function pump() {
      if (busy) return; busy = true;
      while (queue.length) { const [x, interim] = queue.shift(); if (interim && queue.length) continue; try { await processBreath(x, interim); } catch (e) { emit("error", { error: String(e && e.message || e) }); } }
      busy = false;
    }
    const rec = Recorder((y) => { queue.push([y, false]); pump(); }, (lvl) => emit("level", { level: lvl }), (y) => { if (!busy) { queue.push([y, true]); pump(); } });

    return {
      start: () => rec.start(),
      stop: () => rec.stop(true),
      get passage() { return P; },
      get state() { return state; },
      get notes() { return wordNotes; },
      get pointer() { return ptr; },
      /** jump the expected position to a word (e.g. the reciter wants to start again from there) */
      goTo(word) { if (P) ptr = P.spans[word][0]; },
      feed(samples) { queue.push([samples, false]); return pump(); }, // tests: a whole recording as one breath
    };
  }

  global.TasmeeEngine = { loadModel, loadPhonetics, hear, logMel, opcodes, levDistance, locate, listen, buildPassage, compose, recited, findAyah, skeleton, Session, Recorder, PHONEMES, get PH() { return PH; } };
})(typeof window !== "undefined" ? window : globalThis);
