# PERA website

The PERA project website is maintained in this repository alongside the HoloMotion implementation.

## Development

Requires Node.js 22 and npm.

```bash
cd website
npm ci
npm run dev
```

```bash
PAGES_BASE_PATH=/PERA/ npm run build
npm run preview
```

Edit `src/content.js`, `src/ExpertGallery.jsx` and `src/styles.css` to update the page. Gallery data are in `public/data/expert-catalog.json`; videos, posters and source metadata are in `public/media/`.

The homepage uses the existing independent native-scene recordings and single-task gallery. Shared-world validation contact sheets are excluded from the website and README.

## Validation

```bash
node website/scripts/check_catalog.mjs
PAGES_BASE_PATH=/PERA/ npm --prefix website run build
```

The catalogue check validates all 40 ON/OFF gallery records, media hashes, policy identity and server rendering. `--archival-sources` additionally checks original research logs, which are not distributed. Historical recording/rendering scripts require those original logs; they are separate from runnable student scene reproduction in [SCENE_REPRODUCTION.md](../docs/SCENE_REPRODUCTION.md).

## Hosting

Canonical repository: https://github.com/shangchaohu718/PERA

Deployment base: `/PERA/`. The manual Pages workflow template is at `docs/workflows/pages.yml.example` in the repository root. Installing it as `.github/workflows/pages.yml` requires workflow-authorized credentials. The current publishing token has `repo` access but no `workflow` scope. It publishes only `website/dist`, never models, scene archives or training data.

The repository remains **private**. On 2026-11-16 GitHub rejected Pages creation with HTTP 422: “Your current plan does not support GitHub Pages for this repository.” The new Pages URL is therefore not live. The existing `pera-project` site is retained until hosting is resolved; it is not the canonical source repository. No account-plan or repository-visibility changes are made by this migration.

## Attribution

Recordings use the official frozen HoloMotion G1 v1.4 Actor and matching task experts. Keep the original recording metadata and third-party notices. See [LICENSE](../LICENSE), [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) and `licenses/` for applicable terms. A selected recording is not a new success-rate estimate.
