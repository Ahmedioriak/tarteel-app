/* Load as a module, then insert <quran-recitation endpoint="/api/recitation/analyze">. */
const messages = {
  no_difference_detected: "لم يظهر اختلاف في المواضع المفحوصة",
  possible_difference: "ظهر اختلاف محتمل",
  review: "يحتاج إلى مراجعة",
  ambiguous: "تعذّر تعيين الآية"
};
const maxBytes = 24 * 1024 * 1024;

class QuranRecitation extends HTMLElement {
  constructor() {
    super();
    this.attachShadow({mode: "open"});
    this.shadowRoot.innerHTML = `
      <style>
        :host{display:block;color:inherit;font:inherit;direction:rtl}
        form,section{background:var(--quran-surface,#fff);border:1px solid var(--quran-border,#d6e2dc);border-radius:14px;padding:1.1rem}
        form{display:grid;gap:.85rem}label{display:grid;gap:.3rem}
        small{font-size:.88em}input,button,select{font:inherit;box-sizing:border-box;max-width:100%}
        input,select{border:1px solid #9caea4;background:white;color:#183c32;border-radius:7px;padding:.45rem}
        button{background:var(--quran-accent,#225949);color:#fff;border:0;border-radius:8px;padding:.48rem 1.05rem;cursor:pointer}
        button[data-mic],button[data-cue],button[data-place]{background:var(--quran-soft,#e7f1eb);color:#163d32}
        button:disabled{opacity:.55;cursor:default}button:focus-visible,input:focus-visible,select:focus-visible{outline:2px solid #27674d;outline-offset:2px}
        .row{display:flex;gap:.65rem;align-items:center;flex-wrap:wrap}audio{display:block;width:100%}
        output,p{margin:.25rem 0}section{margin-top:.9rem}h2{font-size:1.1em;margin:.2rem 0}
        .verse{font-size:1.22em;line-height:1.95}li{margin:.55rem 0}button[data-cue]{margin-inline-start:.4rem;padding:.05rem .5rem}
        audio[hidden],button[hidden],label[hidden]{display:none}
      </style>
      <form dir="rtl" lang="ar">
        <small>مراجعة صوتية برواية حفص عن عاصم. قد يخطئ الفحص الآلي. يُرسل التسجيل لخادم الموقع عند الضغط على «افحص التلاوة».</small>
        <label>السورة والآية (اختياري)<input data-ayah dir="ltr" placeholder="1:7" maxlength="24"></label>
        <label>ملف التلاوة<input data-file type="file" accept="audio/*,.mp3,.wav,.m4a,.webm,.ogg,.flac,.aac,.mp4"></label>
        <div class="row"><button data-mic type="button">سجّل بالميكروفون</button>
          <button data-stop type="button" hidden>إنهاء التسجيل</button>
          <button data-submit type="submit" disabled>افحص التلاوة</button></div>
        <audio controls preload="metadata" hidden></audio>
        <output aria-live="polite" role="status"></output>
      </form><div data-results dir="rtl" lang="ar"></div>`;
    this.el = (selector) => this.shadowRoot.querySelector(selector);
    this.form = this.el("form");
    this.file = this.el("[data-file]");
    this.player = this.el("audio");
    this.status = this.el("output");
    this.results = this.el("[data-results]");
    this.epoch = 0;
    this.mode = "ready";
    this.file.addEventListener("change", () => this.use(this.file.files[0] || null));
    this.el("[data-mic]").addEventListener("click", () => this.record());
    this.el("[data-stop]").addEventListener("click", () => this.stop());
    this.form.addEventListener("submit", (e) => { e.preventDefault(); this.send(); });
  }

  connectedCallback() { this.sync(); }

  disconnectedCallback() {
    this.epoch++;
    this.request?.abort();
    clearTimeout(this.timer);
    if (this.recorder?.state === "recording") this.recorder.stop();
    this.stream?.getTracks().forEach((t) => t.stop());
    this.mode = "ready";
    this.file.value = "";
    this.use(null);
  }

  sync() {
    const ready = this.mode === "ready";
    const micOK = Boolean(navigator.mediaDevices?.getUserMedia && window.MediaRecorder);
    this.el("[data-mic]").hidden = !micOK || this.mode === "recording";
    this.el("[data-mic]").disabled = !ready;
    this.el("[data-stop]").hidden = this.mode !== "recording";
    this.el("[data-submit]").disabled = !ready || !this.blob;
    this.file.disabled = !ready;
    this.el("[data-ayah]").disabled = !ready;
  }

  use(blob) {
    this.player.pause();
    this.player.removeAttribute("src");
    if (this.objectURL) URL.revokeObjectURL(this.objectURL);
    this.objectURL = null;
    this.blob = null;
    this.results.replaceChildren();
    this.status.textContent = "";
    if (blob && (blob.size === 0 || blob.size > maxBytes)) {
      this.status.textContent = "اختر تسجيلاً غير فارغ وحجمه أقل من 24 ميبيبايت.";
      this.file.value = "";
    } else if (blob) {
      this.blob = blob;
      this.objectURL = URL.createObjectURL(blob);
      this.player.src = this.objectURL;
    }
    this.player.hidden = !this.blob;
    this.sync();
  }

  async record() {
    if (this.mode !== "ready") return;
    const epoch = this.epoch;
    this.mode = "permission";
    this.status.textContent = "بانتظار إذن الميكروفون…";
    this.sync();
    try {
      const stream = await navigator.mediaDevices.getUserMedia({audio: true});
      if (!this.isConnected || epoch !== this.epoch) {
        stream.getTracks().forEach((t) => t.stop());
        return;
      }
      this.stream = stream;
      const mime = ["audio/webm;codecs=opus","audio/mp4","audio/ogg;codecs=opus"]
        .find((type) => MediaRecorder.isTypeSupported(type));
      const recorder = new MediaRecorder(stream, mime ? {mimeType: mime} : undefined);
      this.recorder = recorder;
      let failed = false;
      const chunks = [];
      recorder.ondataavailable = ({data}) => { if (data.size) chunks.push(data); };
      recorder.onerror = () => { failed = true; this.stop(); };
      recorder.onstop = () => {
        stream.getTracks().forEach((t) => t.stop());
        if (epoch !== this.epoch || !this.isConnected) return;
        clearTimeout(this.timer);
        this.stream = null;
        this.recorder = null;
        this.mode = "ready";
        if (failed || !chunks.length) {
          this.status.textContent = "تعذّر إكمال التسجيل. حاول مجدداً أو اختر ملفاً.";
          this.sync();
        } else {
          this.file.value = "";
          this.use(new Blob(chunks, {type: recorder.mimeType || chunks[0].type}));
        }
      };
      recorder.start(1000);
      this.mode = "recording";
      this.status.textContent = "جارٍ التسجيل…";
      this.timer = setTimeout(() => this.stop(), 5 * 60 * 1000);
      this.sync();
    } catch {
      this.stream?.getTracks().forEach((t) => t.stop());
      if (epoch !== this.epoch) return;
      this.mode = "ready";
      this.status.textContent = "تعذّر فتح الميكروفون. اسمح باستخدامه أو اختر ملفاً.";
      this.sync();
    }
  }

  stop() {
    if (this.recorder?.state === "recording") this.recorder.stop();
    clearTimeout(this.timer);
  }

  async send() {
    if (!this.blob || this.mode !== "ready") return;
    const epoch = this.epoch;
    this.mode = "analyzing";
    this.results.replaceChildren();
    this.status.textContent = "جارٍ فحص التلاوة…";
    this.sync();
    this.request = new AbortController();
    try {
      const endpoint = new URL(this.getAttribute("endpoint") || "/api/recitation/analyze", document.baseURI);
      const ayah = this.el("[data-ayah]").value.trim();
      if (ayah) endpoint.searchParams.set("ayah", ayah);
      const response = await fetch(endpoint, {method: "POST", credentials: "same-origin",
        headers: {"Content-Type": this.blob.type || "application/octet-stream"},
        body: this.blob, signal: this.request.signal});
      const report = await response.json().catch(() => null);
      if (!response.ok) throw new Error(typeof report?.detail === "string" ? report.detail : "تعذّر إرسال التسجيل لخادم الفحص.");
      if (!Array.isArray(report?.results)) throw new Error("تعذّر قراءة نتيجة الفحص من الخادم.");
      if (!this.isConnected || this.epoch !== epoch) return;
      for (const result of report.results) this.display(result);
      this.status.textContent = "اكتمل الفحص.";
    } catch (e) {
      if (epoch === this.epoch) this.status.textContent = e instanceof TypeError ? "تعذّر الاتصال بخادم الفحص." : e.message;
    } finally {
      if (epoch === this.epoch) { this.mode = "ready"; this.sync(); }
    }
  }

  display(r) {
    const section = document.createElement("section");
    const add = (tag, text, parent = section) => {
      const el = document.createElement(tag);
      el.textContent = text;
      parent.append(el);
      return el;
    };
    const cue = (parent, t) => {
      if (!Number.isFinite(t)) return;
      const button = add("button", "استمع عند " + t.toFixed(1) + " ث", parent);
      button.type = "button"; button.dataset.cue = "";
      button.onclick = () => {
        this.player.currentTime = Math.max(0, t - 0.35);
        this.player.play().catch(() => { this.status.textContent = "اضغط تشغيل لسماع التسجيل."; });
      };
    };
    const range = r.first_ayah && r.last_ayah && r.first_ayah !== r.last_ayah ?
      r.first_ayah + " – " + r.last_ayah : r.first_ayah;
    add("h2", [range, messages[r.status] || "نتيجة الفحص"].filter(Boolean).join(" · "));
    if (r.reference_text) add("p", r.reference_text).className = "verse";
    if (r.reason) add("p", r.reason);
    if (r.findings?.length) {
      const list = add("ul", "");
      for (const f of r.findings) {
        const li = add("li", "", list);
        add("strong", "«" + f.word + "» · " + f.label, li);
        add("div", "المتوقع «" + f.expected + "». رجّح النموذج «" + f.heard + "».", li);
        add("small", messages[f.status] || "موضع للمراجعة", li);
        cue(li, f.near_s);
      }
    }
    if (r.weak_audio_times_s?.length) {
      const p = add("p", "إشارة صوتية ضعيفة. ");
      cue(p, r.weak_audio_times_s[0]);
    }
    if (r.candidates?.length) {
      const field = add("label", "اختر موضع القراءة ثم أعد الفحص");
      const sel = add("select", "", field);
      add("option", "اختر الآية", sel).value = "";
      for (const id of r.candidates) add("option", id, sel).value = id;
      sel.onchange = () => {
        if (sel.value) { this.el("[data-ayah]").value = sel.value; this.el("[data-submit]").focus(); }
      };
    }
    this.results.append(section);
  }
}
if (!customElements.get("quran-recitation")) customElements.define("quran-recitation", QuranRecitation);
