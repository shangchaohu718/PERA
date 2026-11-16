# Website verification

Earlier sections are an audit history. The 2026-11-06 section at the end describes the current website.

## Build and isolation

- `npm run build`: passed on Node 22.23.2 / Vite 7.2.2. Both default relative-base and explicit `/PERA-g1/` builds were checked.
- Production preview loaded successfully at a `/PERA-g1/` project subpath, including local videos, CSS, JS, evidence and SVG downloads.
- Website dependencies: React, React DOM and Vite; no animation, chart, icon or web-font runtime dependencies.
- Training core, existing checkpoints, source trajectories and experiment reports were not changed. Existing unrelated workspace edits remain untouched.
- GitHub Pages activation was attempted through the API and rejected with HTTP 422 because the current account plan does not support Pages for this private repository. The workflow is prepared; no public deployment is claimed.

## Browser checks

Real in-app Chromium checks, not mock screenshots:

| Check | Result |
| --- | --- |
| 320 px narrow viewport | No document-level horizontal overflow |
| 390 × 844 phone | No overflow; responsive navigation, complete hero robot, skill tabs and provenance visible |
| 768 × 1024 tablet | No overflow; skill selection and five skill labels usable |
| 1280 × 720 laptop | Compact hero keeps the complete G1 in view |
| 1440 × 1000 desktop | Hero, method diagram and results visually inspected |
| Phone navigation | Menu expanded and collapsed; section navigation worked |
| Skills | Dance, martial-arts and climbing videos selected; readyState 4, playback active, no media error |
| Keyboard skills | ArrowRight changed focus and selected skill |
| Planned recovery | Displays pending state without a fake video or success claim |
| Method tabs | Inference/training content switched; Home key restored inference |
| Full results | Expanded all seven rows, including worse expert-on results |
| Training curve | Keyboard Home selected 2,000 updates / 0.02353 MSE |
| Video dialog | Opens real clip, Escape closes, focus returns to opener |
| Comparison control | Selects actual `roundhouse-base.mp4`; paused until playback is requested |
| Concept switch | `aria-pressed` and descriptive text update; explicitly labeled conceptual |
| Lazy media | Fresh production page loaded only the hero MP4; two offscreen videos had null sources and readyState 0 |
| Console | No application warnings or errors observed in successful page sessions |

Reduced-motion behavior is implemented with `matchMedia('(prefers-reduced-motion: reduce)')`, no automatic playback under that preference, paused playback on preference changes, and CSS removal of scroll/reveal transitions. The OS preference itself was not toggled during this verification. Autoplay still depends on each visitor's browser rules; manual playback and direct MP4 fallbacks are available. Safari/iOS hardware was not available for direct testing.

## Research and media integrity

`python website/scripts/verify_assets.py` verifies:

- All **5** exported MP4s are H.264 / yuv420p / 1920×1080 / 30 fps.
- MP4 `moov` precedes `mdat` (faststart), output frame counts match manifests, and posters exist.
- Video hashes match manifests and source trajectory hashes match original files.
- All **21** training points equal the original round mean losses exactly; all **7** motion comparisons equal the source summary exactly.
- Exported evidence includes source hashes and no machine-local absolute paths.
- Motion speed remains 1×; no interpolated robot poses.

Independent forward-kinematics spot-checks reconstructed all 30 named training body positions from archived root quaternions and joint positions using the selected G1 model. Samples every 25 frames:

| Clip | Samples | Maximum body position difference |
| --- | ---: | ---: |
| Public Charleston | 41 | 0.980 µm |
| Expert roundhouse | 11 | 0.799 µm |
| Baseline roundhouse | 11 | 0.697 µm |
| Public horse-stance punch | 13 | 0.715 µm |

This validates pose/coordinate/joint-order conversion for sampled states. It does not establish contact forces or dynamic equivalence between Isaac Lab and MuJoCo. Terrain uses the exact saved native scene and qpos, with terrain geometry retained.

Contact sheets from exported videos were inspected for complete hands/feet, robot identity, lighting and real obstacle geometry. Masters and the first rendering pass are preserved in `.render-work/`. Full episodes loop, but an end-to-start pose match is not guaranteed.

## Screenshots

Actual screenshot files are in the ignored `website/qa/` directory:

- `desktop.jpg` — final desktop hero
- `laptop.jpg` — compact laptop hero
- `mobile.jpg` — phone hero
- `desktop-method.jpg` — method diagram
- `desktop-results.jpg` — paired result and true metrics
- `tablet-skills.jpg`, `mobile-skills.jpg` — skill gallery checks
- `contact-sheets/` — sampled actual video frames

The terrain and hero camera were refined after the first gallery captures. Final hero captures reflect the closer hero framing. Source hashes are enforced so future experiment changes require deliberate evidence re-export.

## Six-expert highlights revision

The current homepage replaces the public-policy dance with a 30-second montage of six actual expert executions. Six short excerpts (5.0 / 4.8 / 5.2 / 5.0 / 5.0 / 5.0 seconds), their complete recordings/task segments, and exact source timing manifests were added. See `HIGHLIGHTS.md` for selection and scope. Previous assets are retained; the media checker now validates 18 videos, including all montage chapters and short-to-complete-recording links.

Brighter lighting preserves original robot albedo and actual poses. All terrain and the skateboard state remain intact. Both older quantitative comparison videos were re-rendered with matching new lighting; their data were not changed. The refined kick highlight is clearly separated from the older quantitative snapshot.

Verified in the live browser: all six short clips resolve and play at their stated durations; the complete forward-roll segment is 8.333 seconds, the complete skateboard recording is 30 seconds, the short skateboard expansion is 5 seconds. Native dialog Escape restores focus. The six-tab phone grid has no document-level horizontal overflow. The hero loads the actual 30-second file and identifies it as selected excerpts. Screenshots are in `qa/highlights/`. All masters and previous rendering revisions remain local in `.render-work/`.

## White studio, skateboard selection and tennis — current revision

- Seven selected skill clips now feed a 35.0-second montage. The media checker passes all 20 MP4s, their source/video hashes, frame counts and faststart layout. Vite production build passes.
- Roundhouse, roll, vault, skating, gaps, steps and both older quantitative comparison videos have been re-rendered. The flat studio floor is white (kick preview dominant RGB 255/255/255); raised terrain retains visible contours.
- Skateboard selection is now source 0.0–5.0 s, viewed from azimuth 70°, showing push-off, recovery to the deck and two-foot gliding. The original native initialization already has a foot on the board; this is not an off-board mounting demonstration.
- Tennis uses a fresh frozen selected-student rollout: 40 s, seed 97001, 2,000 student control steps, zero teacher queries/executed steps, zero training updates. Its 5 s excerpt shows a selected return. The full recording includes misses.
- Both hand meshes use exact upstream wrist mounts and are added only in the replay model with zero density and no collisions. Body masses/inertias remain equal. Sampled body and site positions match exactly (maximum difference 0 m), including wrist and racket attachment sites.
- Live in-app browser: new tennis clip loads at duration 5 s / readyState 4; full tennis loads at 40 s / readyState 4. The hero loads the 35 s file. The skateboard tab exposes the new push/return/glide description and plays. The new seven-tab phone layout has no document horizontal overflow at 320 and 390 px.
- Current screenshots: `qa/highlights/white-tennis-desktop.jpg` (expanded tennis), `white-skate-desktop.jpg`, `white-mobile.jpg`. Preview contact sheets show the revised poses and terrain contours.

## Scene context, continuous tennis and recorded ON/OFF switch — current revision

- Terrain clips use contrasting gray-blue surfaces with visible recesses and a subtle world texture. The original hfields, gaps, steps and obstacle meshes remain intact. White sky/background is retained.
- Skating retains source 0–5 s (push-off, recovery to the deck, glide), with a world-anchored ground grid and contact shadows. The camera follows the robot; texture motion provides a displacement cue.
- Tennis retains the source court colors, markings, net and yellow ball. The previously invisible collision-only terrain plane remains invisible, avoiding overlap with the actual court. Both restored visual hands remain present.
- The tennis gallery now plays the first 20 continuous seconds, containing four incoming-ball cycles with relaunches around 5.02, 10.02 and 15.02 s. The second return is unsuccessful and is retained. A fixed court view and synchronized robot inset show the same recorded pose at each instant. Full 40 s remains accessible.
- Homepage montage is 50 s / 1,500 frames. All 20 MP4s pass source/video hash, frame-count, codec and faststart checks. Frontend build passes.
- The accessible switch at `#expert-comparison` changes between actual earlier matched `roundhouse.mp4` (expert ON) and `roundhouse-base.mp4` (expert OFF/public only) recordings. Switching restarts playback; motion preferences remain respected. This pair is distinct from the refined kick gallery checkpoint.
- Live browser checks: OFF changes aria-checked to false and loads `roundhouse-base.mp4` with readyState 4; ON changes it to true and loads `roundhouse.mp4` with readyState 4. Keyboard Enter also toggles the switch. Tennis loads at 20 s / readyState 4; its expanded player shows the colored court and inset. At 390 px there is no horizontal document overflow, and the switch/real OFF playback are visible and usable.
- Current screenshots: `qa/highlights/expert-off.jpg`, `expert-toggle-mobile.jpg`, `court-tennis-v4.jpg`. Original recordings and earlier exports remain archived locally; no research source, checkpoint, training data or original report was changed.

## Eight-skill comparisons, close-up tennis and moonwalk — current revision

- All eight gallery skills have a real expert ON/OFF switch. OFF is the frozen HoloMotion G1 v1.4 public path. The setting persists when browsing skills, so the whole gallery can be inspected in OFF mode.
- New frozen paired evaluations were recorded for roll, vault, gaps, steps and skate. Initial qpos are identical for each pair (maximum difference 0). No optimizer/training or teacher actions were used; public hashes remain unchanged. Native failure/end times are preserved.
- Roll ON completes 8.34 s; OFF terminates at 3.62 s. Vault ON completes 10.46 s; OFF terminates at 4.02 s. First recorded gaps/steps ON episodes complete their native horizons; OFF terminates at 4.60 / 5.96 s. Gap and step source reports retain all four evaluation seeds, not just the recorded first seed.
- Skate ON and OFF both pass this particular basic test: 40.66 m versus 25.01 m displacement in 30 s. No failure is fabricated for OFF. First 5 s are displayed; full 30 s are linked.
- Tennis is now a close following camera, distance 3.2 m, without a wide-view inset. ON and OFF each play 20 s; the complete 40 s student source is retained. Both visual hands remain restored with zero physics impact.
- Moonwalk uses latest model_1999 (2,000 PPO iterations / 48,000 Actor Adam updates), a six-second excerpt, actual public control, and complete 17.8-second recordings. Later root drift remains visible. Exact single-reference/seed metrics are exported in `public/data/moonwalk.json`.
- The hero is 56 s / 1,680 frames. `verify_assets.py` passes all 46 H.264 / 1080p / 30 fps videos, source/video hashes, frame counts, faststart boxes, eight ON/OFF links, all 21 earlier curve points and seven earlier quantitative comparisons. Vite production build passes.
- Live browser checks selected all eight skills in OFF mode and all eight in ON mode: every displayed video had readyState 4, correct source ID/duration and no media error. The Moonwalk switch changes actual sources `moonwalk.mp4` / `moonwalk-off.mp4`. The real tennis close-up and gallery switch were visually inspected. Native ON/OFF durations match their actual recording ends.
- Current screenshots are in `qa/highlights/`: `tennis-close-v5.jpg`, `moonwalk-v5.jpg`, `eight-skills-mobile-v5.jpg`. Earlier revisions remain archived locally. The original research files and existing unrelated workspace edits were not changed.

## Single expert switch, 500-iteration moonwalk and no-reset aftermath — 2026-11-06

- Moonwalk now uses stage_500/model_499, 500 PPO iterations / 12,000 Actor Adam updates, with full-run mean body error 7.17 cm versus public 50.19 cm. Short and full ON/OFF media, source hashes, evaluation data and hero montage were regenerated. The hero remains 56 seconds.
- The same frozen public Actor is explained before the sole gallery expert switch. The conceptual control is now static; the redundant earlier result player/switch is replaced by a still and link. Earlier quantitative measurements and downloads remain intact.
- Fresh OFF recordings retain native scoring, then continue the same controller for four seconds without reset or control substitution. Roll stops natively at 3.64 s, vault 4.04 s and gaps 5.18 s; high steps reaches the 10-second horizon without completing the target and falls during the explicitly labeled post-horizon visualization. This distinction is visible in the outcome text; the later fall is not misreported as an in-horizon termination. Prior source recordings and four-seed reports remain archived.
- Gap and step cameras pitch down as the body falls so terrain edges no longer conceal it. Actual terrain geometry and recorded poses are retained. Preview contact sheets were visually checked for the full fallen body.
- Comparison players play once and hold their last frame. Live Chromium verified the Moonwalk ended/paused state, `loop=false`, explicit Replay restarting from time zero, and exactly one switch in the document. The 390 px layout has eight skill tabs and no horizontal document overflow.
- `verify_assets.py` passes all 46 MP4 codec/frame-count/faststart/source/video hash checks, all eight comparison mappings, four continuation controller/reset guards, selected moonwalk stage, 21 exact training points and seven earlier motion comparisons. Vite production build passes.

## Hero cleanup and research entry points — 2026-11-06

- Paper/Code entry points moved beneath the hero title and description. Paper is a non-link Coming soon status by user choice. Code uses the verified private repository URL and displays Private repository. No local draft PDF is exposed.
- Removed all hero-side specification/action badges, elapsed-duration text and bottom micro-links. The hero montage is regenerated without any burned-in action labels from the original lossless masters, retaining all 1,680 original-speed frames and source chapter provenance. Gallery videos retain their own labels.
- Live mobile Chromium at 375 × 773: Paper/Code visible in the initial viewport, no document horizontal overflow, zero hero annotation nodes, hero duration 56 seconds and readyState 4.
- A genuine single-shot continuous simulation is under discussion. Existing verified course data include a 279.7-second zero-reset, one-Actor ten-task run with explicit expert scheduling and 0.6-second action handover. This is evidence for planning, not a claim that all eight gallery skills have already been joined in one world.


## Six-station scene and camera preview — 2026-11-06

- Current hero: 47.7667 s / 1,433 frames, H.264/yuv420p, 1920×1080/30, faststart. One scene, one G1, zero camera cuts. Seven native motion stages form six stations; six joins are explicitly authored animation. The 56 s experimental montage is retained independently.
- `verify_assets.py` passes all 47 videos, source/video hashes, native gallery checks and design-preview provenance. It explicitly permits pose interpolation only for this flagged preview.
- Actual exported frames were inspected for roll, boarding, racket pickup, terrain and tennis. Source scene and timeline remain archived.
- Kinematic continuity checks: maximum camera angular speed 16.875°/s; maximum root displacement at a join 0.04923 m/frame; maximum joint change at a join 0.24055 rad/frame; pickup wrist target error 0.000259 m. These measurements do not validate physics, contacts or a learned transition controller.
- Hero and expanded-player text explicitly state scene preview, recorded motions with animated transitions, and no continuous policy rollout. Experimental ON/OFF gallery and research links remain available.
- Final live browser: hero and expanded player resolve `scene-preview.mp4`, duration 47.766667 s, readyState 4, loop false; exactly one expert switch; no warn/error console entries. At a 490 px CSS viewport, document width is 475 px with no horizontal overflow. Expanded player visibly shows the design-preview scope. Screenshot: `qa/scene-preview/browser-preview.png`. Vite production build passes.


## Current four-scene highlights and racket visibility — 2026-11-06

- Replaced the animated six-scene homepage previs with 26.4 s / 792 frames of four real policy excerpts: skateboard, rough ground + raised steps, upstairs, tennis. Three direct cuts, original speed, no pose interpolation or animated walking.
- Kept the complete 45.1667 s physical source separately linked. The shared frozen public Actor is unchanged; robot resets/state writes after initialization are zero; three temporal-memory resets are disclosed.
- Independent replay of the entire accepted course: 2,257 control steps / 9,028 physics steps, zero inserted robot poses, maximum root and joint errors both 0. Source/video/report/audit hashes are checked by `verify_assets.py`.
- Tennis full source retains three incoming-ball cycles: two actual racket contacts, one miss. The highlight selects the first two.
- Non-tennis highlight frames hide the racket mesh and shadow. The source robot trajectory, mounted racket inertia and collisions remain unchanged; full original source keeps its original prop visibility. A contact sheet confirms bare hands for skate, rough ground and stairs, with racket present for tennis.
- All 48 H.264 / 1080p / 30 fps videos, faststart layout, source/video hashes, original ON/OFF links, 21 earlier training points and seven earlier quantitative comparisons pass. Vite production build passes.
- Final live browser: updated hero and expanded player load at duration 26.4 s / readyState 4 / loop false, with one expert switch and no console warnings/errors. At 495 px CSS width, document width is 480 px with no horizontal overflow. The refreshed first skateboard frame visibly has no racket. Screenshot: `qa/policy-highlights/browser-racket-fix.png`; all-stage contact sheet: `qa/policy-highlights/racket-visibility-contact-sheet.jpg`.


## Current four independent native scenes — 2026-11-06

- Homepage: 31 s / 930 frames, four separate native source recordings and four distinct scene hashes, three direct cuts, no pose interpolation, scene relocation, artificial joins or prop hiding.
- Skate and high-step terrain run 5 s each; native MimicX stairs run 6 s; close-up native ATHLETE tennis runs 15 s. Only tennis contains a racket. Complete sources are linked separately in the player.
- Native MuJoCo 3.5 stair model vs portable visual replay: all body positions match exactly at source frames 0, 140, 290 and 440; maximum error 0. `native-fk-check.json` is retained in `.render-work/native-four-scenes-v1/`.
- Source/video/report/scene hashes, chapter durations, native-scene uniqueness and complete-source links are checked in `verify_assets.py`. Prior eight-skill ON/OFF media, 21 training points and seven quantitative comparisons remain intact.
- Final live browser: hero resolves the new MP4 at 31 s, readyState 4, loop false, with no console warnings/errors. The player exposes four separate full-recording links. One expert switch remains. At 518 px CSS width, document width is 503 px with no horizontal overflow. Screenshot: `qa/native-scenes/browser-native-scenes.png`; actual export contact sheet: `qa/native-scenes/four-native-scenes.jpg`. Production build and all 48 media checks pass.
