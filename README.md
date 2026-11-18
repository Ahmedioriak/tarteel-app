# رفيق الحفظ

**التطبيق:** https://claude.ai/artifact/X7HUqdHuVeESWVjacXJzzT

تطبيق ويب يساعدك تحفظ القرآن الكريم وتراجعه كل يوم. يشتغل كصفحة وحدة (`index.html`) مع ملف بيانات المصحف (`quran.json`).

## المميزات

- **خطة الحفظ:** تختار الكمية اليومية (ربع صفحة، نص صفحة، صفحة، صفحتين)، وأيام الحفظ في الأسبوع، وترتيب الحفظ (من الناس للفاتحة أو العكس). التطبيق يحسب كم صفحة تحفظ في الشهر والسنة، ومتى تختم.
- **مهام اليوم:** الحفظ الجديد، والمراجعة القريبة (آخر ما حفظت)، والمراجعة البعيدة (دورة على المحفوظ القديم)، وتثبيت الصفحات اللي غلطت فيها.
- **الحفظ والتكرار:** المصحف بالرسم العثماني ومقسم على صفحات مصحف المدينة. فيه عداد تكرار لكل آية، ووضع "اختبر نفسك" يخفي الكلمات.
- **اختيار أي مقطع:** تختار السورة ومن آية إلى آية، وتسمعها على Quran.com، أو تكررها، أو تختبر نفسك فيها.
- **التسميع بالصوت:** تقرأ من حفظك وتطلع الكلمات وحدة وحدة كل ما قلتها صح، والكلمات اللي تنساها تطلع بالأحمر.
- **التقدم:** خريطة للـ604 صفحة، وعدد الأيام المتتالية، وكم حفظت هالشهر.

## التشغيل محلياً

```bash
python3 -m http.server 8080
# وافتح http://localhost:8080
```

لازم يشتغل من خادم ويب (مو بفتح الملف مباشرة)، لأن الصفحة تحمّل `quran.json` بـ `fetch`.

## ملاحظات

- التطبيق مبني يشتغل كـ artifact في Claude. هناك ينحفظ التقدم في حساب المستخدم، والتسميع بالصوت يستخدم خاصية الإملاء في Claude.
- لما يشتغل برا Claude، التقدم ينحفظ في المتصفح (`localStorage`)، والتسميع بالصوت ما يكون متوفر.
- الاستماع يفتح المقطع على [Quran.com](https://quran.com).

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
