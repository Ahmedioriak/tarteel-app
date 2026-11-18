# رفيق الحفظ

**التطبيق:** https://claude.ai/artifact/X7HUqdHuVeESWVjacXJzzT

تطبيق ويب يساعدك تحفظ القرآن الكريم وتراجعه كل يوم. يشتغل كصفحة وحدة (`index.html`) مع ملف بيانات المصحف (`quran.json`).

## المميزات

- **خطة الحفظ:** تختار الكمية اليومية (ربع صفحة، نص صفحة، صفحة، صفحتين)، وأيام الحفظ في الأسبوع، وترتيب الحفظ (من الناس للفاتحة أو العكس). التطبيق يحسب كم صفحة تحفظ في الشهر والسنة، ومتى تختم.
- **مهام اليوم:** الحفظ الجديد، والمراجعة القريبة (آخر ما حفظت)، والمراجعة البعيدة (دورة على المحفوظ القديم)، وتثبيت الصفحات اللي غلطت فيها.
- **الحفظ والتكرار:** المصحف بالرسم العثماني ومقسم على صفحات مصحف المدينة. فيه عداد تكرار لكل آية، ووضع "اختبر نفسك" يخفي الكلمات.
- **اختيار أي مقطع:** تختار السورة ومن آية إلى آية، وتسمعها على Quran.com، أو تكررها، أو تختبر نفسك فيها.
- **التسميع بالصوت:** تقرأ من حفظك وتطلع الكلمات وحدة وحدة كل ما قلتها صح، والكلمات اللي تنساها تطلع بالأحمر.
- **التسميع الدقيق (من موقعك):** يسمع الكلمات والحروف ويشرح كل غلط بالعربي، ويوقف عندك لين تصححه. الحركات والتجويد اختياريين. وفيه تسميع حر تبدأ فيه من أي آية، والبسملة تعني بداية سورة.
- **ثلاث ألوان** للتطبيق.
- **التقدم:** خريطة للـ604 صفحة، وعدد الأيام المتتالية، وكم حفظت هالشهر.

## تشغيله من موقعك (GitHub Pages)

1. في المستودع: Settings ← Pages.
2. Source: **Deploy from a branch**، والفرع `main`، والمجلد `/ (root)`، واضغط Save.
3. بعد دقيقة أو دقيقتين يصير الموقع على: https://ahmedioriak.github.io/tarteel-app/

من موقعك:
- **الاستماع داخل التطبيق:** تختار القارئ (الحصري، العفاسي، المنشاوي، عبد الباسط، الشاطري)، وتكرر الآية أو المقطع، وتغير السرعة. الصوت من [EveryAyah](https://everyayah.com) و[Islamic Network](https://alquran.cloud/cdn).
- **التسميع بالصوت:** يستخدم التعرف على الكلام في المتصفح. يشتغل في Chrome (كمبيوتر وأندرويد) وSafari (آيفون)، وما يشتغل في Firefox.
- **التقدم:** ينحفظ في المتصفح على نفس الجهاز.

## التشغيل محلياً

```bash
python3 -m http.server 8080
# وافتح http://localhost:8080
```

لازم يشتغل من خادم ويب (مو بفتح الملف مباشرة)، لأن الصفحة تحمّل `quran.json` بـ `fetch`.

## ملاحظات

- **داخل Claude:** التقدم ينحفظ في حسابك ويتزامن بين أجهزتك، والتسميع بالصوت يستخدم خاصية الإملاء في Claude، والاستماع يفتح المقطع على [Quran.com](https://quran.com) لأن صفحات Claude ما تقدر تشغّل صوت من مواقع ثانية.
- **من موقعك:** الاستماع والتسميع داخل التطبيق، والتقدم في المتصفح (`localStorage`).

## كيف يشتغل التسميع الدقيق

`tasmee-engine.js` يشتغل كله داخل المتصفح:

1. المايك ← 16 kHz ← تقسيم التلاوة لأنفاس عند الوقفات.
2. ميزات صوتية (log-mel، 80) مطابقة لـ faster-whisper (فرق أقل من 0.000002).
3. نموذج [HisnLabs tajweed-v2](https://huggingface.co/HisnLabs/tajweed-v2) (Apache-2.0) عبر onnxruntime-web: يكتب اللي سمعه بالرسم الصوتي للقرآن (حروف، حركات، مدود، غنة، إخفاء، إقلاب، قلقلة).
4. مقارنة مع النطق الصحيح لكل آية من `phonetics.json`، ومعه صيغ الوقف والابتداء لكل كلمة.
5. قواعد الملاحظات منقولة من `imam.py` في نفس النموذج، ومطابقة له في 2459 من 2459 حالة اختبار.

`phonetics.json` يُبنى بـ `scripts/build-phonetics.py` من [quran-transcript](https://pypi.org/project/quran-transcript/) (MIT).

اختبار على تلاوات حقيقية: افتح `test/engine-test.html` من موقعك.

## مصادر البيانات

- **نص المصحف:** الرسم العثماني من [موسوعة القرآن الكريم (QuranEnc)](https://quranenc.com)، عن طريق حزمة [`quran-json`](https://www.npmjs.com/package/quran-json).
- **الصفحات والأجزاء:** تقسيم مصحف المدينة (رواية حفص) من حزمة [`quran-meta`](https://www.npmjs.com/package/quran-meta).
- **مراجعة الأجزاء وعدد الآيات:** مقارنة مع نص [Tanzil](https://tanzil.net) من حزمة [`@muslims-community/quran`](https://www.npmjs.com/package/@muslims-community/quran).

لإعادة بناء `quran.json`:

```bash
npm install
npm run build-data
```

السكربت يتأكد إن عدد الآيات 6236، وعدد الصفحات 604، وإن الأجزاء تطابق المصدرين.

---

**Rafiq al-Hifz** is a single-page web app for daily Quran memorization: a configurable plan, daily new/near/far review tasks, repetition and self-test modes on the Uthmani text paged to the Madina mushaf, range selection with listening via Quran.com, voice recitation checking, and a 604-page progress map.
