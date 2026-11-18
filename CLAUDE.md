# رفيق الحفظ: GitHub and the live app stay in sync

This repo is the source of truth for the "رفيق الحفظ" app. The live app is a Claude artifact:

- Artifact slug: `f3d2c74f-ebf2-4b98-8f0c-55370c1464b0`
- Link: https://claude.ai/artifact/X7HUqdHuVeESWVjacXJzzT
- Repo: `Ahmedioriak/tarteel-app` (public), branch `main`, files at the repo root
- The artifact is shared by public link, so republishing needs `shared_live: "public"` and `artifact_title: "رفيق الحفظ"`.

## Every change, every time

1. Edit the files here (`index.html`, and `quran.json` only through `npm run build-data`).
2. Bump `APP_VERSION` in `index.html` and add a line to `CHANGELOG.md`.
3. Test locally: `python3 -m http.server 8080` and open the app on a phone-sized viewport.
4. Commit and push to `main`.
5. Publish the same files to the artifact with `publish_files` (mode `patch`): `index.html` as `index.html`, and `quran.json` as `quran.json` if it changed.
6. If the page starts using a new capability, call `declare_capabilities` first, then publish. Current declaration: `db` (rules: root read `view` / write `owner`; `data/users/{self}` write `interact`), `user`, `dictation`.

The version shown at the bottom of the plan screen must match the latest commit's `APP_VERSION`.

## Two places, one file

`index.html` runs in two places and picks its behaviour at load:
- **Inside Claude (artifact):** `window.claude` exists. Progress in `db`, recitation checking through Claude's `dictation`, listening by links to Quran.com (artifact pages can't load remote audio).
- **Own site (GitHub Pages, https://ahmedioriak.github.io/tarteel-app/):** `STANDALONE` is true. Progress in `localStorage`, in-app audio player (EveryAyah per-ayah MP3s, Islamic Network CDN as fallback), recitation checking through the browser's Web Speech API (`webSpeechAdapter`, `ar-SA`).
Test both: a change to audio or speech must keep the artifact path working, and the reverse.

## Accurate recitation checking (own site only)

- `tasmee-engine.js` + `phonetics.json` load only when `STANDALONE` (needs the mic and the model from huggingface.co, both blocked inside Claude).
- Model: `HisnLabs/tajweed-v2` `onnx/model-int8.onnx`, onnxruntime-web from jsDelivr, cached in Cache Storage.
- `phonetics.json` is built by `scripts/build-phonetics.py` (≈8 min). Rebuild it whenever `quran.json` changes.
- Checks before shipping a change to the engine: `listen()` must stay identical to the Python reference on the random cases (`capLenient:false`), and `test/engine-test.html` on real EveryAyah audio must keep clear mistakes on correct recitations near zero.

## Things to keep

- User progress lives in the artifact's `db` at `data/users/<id>/hifz`. Never change that path or the state shape without a migration, or people lose their progress.
- Quran text must come only from `quran.json`, built and validated by `scripts/build-data.mjs` (6236 ayahs, 604 pages). Never type or edit Quran text by hand.
