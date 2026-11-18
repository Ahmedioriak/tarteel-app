import { findPage, findJuz, getSurahMeta, meta } from "quran-meta/hafs";
import fs from "fs";
const q = JSON.parse(fs.readFileSync("node_modules/quran-json/dist/quran.json","utf8"));
const tz = JSON.parse(fs.readFileSync("node_modules/@muslims-community/quran/dist/quran.json","utf8"));
const surahs=[], ayahs=[];
let errs=0;
for (const s of q) {
  const m = getSurahMeta(s.id);
  if (m.ayahCount !== s.verses.length || tz.surahs[s.id-1].ayat.length!==s.verses.length) { console.log("count mismatch", s.id); errs++; }
  surahs.push([s.name, s.verses.length]);
  for (const v of s.verses) {
    const p = findPage(s.id, v.id), j = findJuz(s.id, v.id);
    if (j !== tz.surahs[s.id-1].ayat[v.id-1].juz) { console.log("juz mismatch", s.id, v.id); errs++; }
    ayahs.push([s.id, v.id, p, j, v.text.trim()]);
  }
}
const pages = new Set(ayahs.map(a=>a[2]));
console.log("ayahs", ayahs.length, "meta", meta.numAyahs, "pages", pages.size, "min", Math.min(...pages), "max", Math.max(...pages), "errs", errs);
// monotonic page check
for (let i=1;i<ayahs.length;i++) if (ayahs[i][2]<ayahs[i-1][2] || ayahs[i][2]-ayahs[i-1][2]>1) console.log("page jump", ayahs[i]);
const show = p => ayahs.filter(a=>a[2]===p).map(a=>a[0]+":"+a[1]);
console.log("p1", show(1).join(" "), "| p2 first", show(2)[0], "| p582 first", show(582)[0], "| p604", show(604).join(" "));
console.log("p50 first/last", show(50)[0], show(50).at(-1));
const juzStart = []; for (let j=1;j<=30;j++){ juzStart.push(ayahs.find(a=>a[3]===j)[2]); }
console.log("juz start pages", juzStart.join(","));
fs.writeFileSync(new URL("../quran.json", import.meta.url), JSON.stringify({src:"Uthmani text: The Noble Qur'an Encyclopedia (quranenc.com) via quran-json; pages: quran-meta (Hafs, Madina mushaf)", surahs, ayahs}));
