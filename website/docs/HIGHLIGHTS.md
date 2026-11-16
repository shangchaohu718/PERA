# Eight skills: highlights, close-up tennis and real expert ON/OFF

The experimental gallery replays recorded robot/object states at original speed, without pose animation or fabricated failure footage. The current hero is a clearly edited 31-second film of four separate task recordings, each in its own native scene, with hard cuts and no animated joins. Full recordings are linked per chapter. The gallery independently switches real task-expert ON and frozen-public-policy OFF replays.

The table below describes the retained experimental montage `showreel.mp4`, not the current homepage preview.

| Skill | Montage source interval | Montage duration | Gallery comparison |
| --- | --- | ---: | --- |
| Refined roundhouse | 0–5 s | 5 s | Refined PPO replay / original public-policy control on the same joint-motion reference, spatially aligned camera |
| Moonwalk | 0–6 s | 6 s | 500-iteration model_499 / recorded stage_500 public control |
| Forward roll | 1.9–6.7 s | 4.8 s | Fresh matched standalone native ON/OFF evaluations |
| Obstacle vault | 2.4–7.6 s | 5.2 s | Fresh matched standalone native ON/OFF evaluations |
| Skateboard | 0–5 s | 5 s | Fresh matched ON/OFF, first 5 s shown; complete 30 s available |
| Gap crossing | 0.5–5.5 s | 5 s | Fresh matched ON/OFF, including actual public failure termination |
| High steps | 0–5 s | 5 s | Fresh matched ON/OFF, including actual public failure termination |
| Tennis return | 0–20 s | 20 s | Close-up student ON / frozen-public OFF in the same ATHLETE task protocol |

The archived montage is 56.0 s / 1,680 frames at 30 fps. Chapters end at 5, 11, 15.8, 21, 26, 31, 36 and 56 seconds. `showreel.json` records all source hashes and exact chapter intervals. The earlier body-error quantitative snapshot remains distinct from the newer refined kick.

## Real ON/OFF data

Existing public-policy tests were found for roll, vault, gaps and high steps, but most did not record qpos/video. Skate had only a 0.2-second OFF smoke recording. `record_comparisons.py` therefore records fresh bounded evaluations with the existing frozen public Actor and trained task plug-ins, without optimizers or training. The public weights are hashed before and after. No teacher actions are executed. The scored evaluation retains native physics and termination rules. The separate post-termination visualization described below clears only the manual step guard, with auto-reset disabled.

Roll, vault, skate, gaps and steps have exactly equal initial qpos between their ON/OFF source recordings (maximum absolute difference 0). Their common task/reset/seed is preserved; switching restarts each recorded clip. ON/OFF can end at different times because actual native terminations are retained. The gap and step source evaluations also retain their original four-seed reporting procedure; only the first seed is recorded for the website.

The skateboard public policy passed this particular basic test as well: 25.01 m in 30 s, compared with the expert's 40.66 m. Both real successful outcomes are retained. A visual switch is not evidence that the public policy must fail on every task.

`comparison_clips.json` contains the render inputs. `public/data/comparisons.json` maps all eight gallery skills to actual ON/OFF clips, complete source recordings, durations, outcomes and source hashes. Each video has its own JSON provenance. Full raw recordings and earlier renders remain locally archived; original `reports/` files are untouched.

## Tennis framing and hands

The main camera follows the robot at distance 3.2 m, azimuth 125°, elevation −24°. There is no wide main view or inset. The colored original court, white markings, racket and yellow ball remain. ON plays the first 20 continuous seconds of seed 97001 (four incoming-ball cycles, including an unsuccessful return); the complete 40-second source is linked. OFF is a fresh 20-second run of the same frozen public Actor with the tennis plug-in bypassed, same seed and launch protocol. Incoming balls are relaunched between cycles; there is no opponent policy and no robot reset between balls.

Both upstream rubber-hand visual meshes are restored at their exact commented wrist mounts in a separate replay model. They have zero density and no collisions. Body masses/inertias and sampled body/racket-site positions match the original replay exactly. Debug target markers and collision overlays are hidden. The originally invisible collision-only plane stays invisible, preserving the real colored court surface.

## Moonwalk

The selected source is `reports/moonwalk_expert_20261105/stage_500/`, model_499, 500 PPO iterations / 12,000 cumulative Actor Adam updates, seed 42. Full-run mean body error is 7.17 cm versus the public control’s 50.19 cm; bilateral heel-phase RMSE is 0.588 cm versus 5.145 cm. Both complete recordings are 17.8 seconds. The stage_2000 result remains archived; the website deliberately selects stage_500 at the user’s request. This single repeated reference and single seed do not establish generalization or old-skill retention.

## Scene rendering and reproduction

White sky/background is retained. Terrain uses contrasting gray-blue surfaces and visible recesses; skating and moonwalk have world-anchored grids. Contact shadows and actual terrain geometry remain. Rendering only changes camera, lighting and presentation materials.

`highlights.json` controls the experimental highlights and archived montage. Lossless FFV1 masters are in `.render-work/highlights-v6/`. New media are first rendered to `.render-work/site-v6/media/`, verified, then copied into `public/media/` before UI updates. Prior exports are retained in `.render-work/`. Existing masters/MP4s are never silently overwritten.

Use the existing robotics Python and `scripts/run_flex_supermimic.sh` to run `website/scripts/record_comparisons.py` for roll, vault, gaps, steps and skate. `record_tennis.py --expert off --seconds 20 --output <new-directory>` records tennis OFF; run `tennis_hands.py --directory <directory>` afterwards. Archive previous generated output before regenerating. `prepare_comparisons.py` validates starts and prepares the comparison source index; `render_comparisons.py` renders it. `verify_assets.py` checks all 48 video/source hashes, frame counts, codecs, faststart layout, comparison links and the unchanged earlier experimental data. `npm --prefix website run build` builds the static site.

## One switch and visible failure aftermath — 2026-11-06

The gallery is the sole expert ON/OFF module. The overview uses a static architecture concept; the Results section keeps the earlier metrics and a still image. The explanation before the gallery identifies the common frozen HoloMotion G1 v1.4 public Actor, task-specific experts, and task-specific observation adapters/native simulation setups. These prerecorded tasks are not claimed to be one uninterrupted mixed-skill evaluation.

`record_failure_continuations.py` records fresh OFF runs for roll, vault, gaps and steps, then continues the same frozen public controller for four seconds with no reset. The official failure is scored at its first native termination; the continuation is visualization only. Native termination terms continue to run, while the environment's manual step guard is explicitly cleared after termination solely to permit these extra diagnostic steps. Auto-reset stays disabled. No control substitution, passive ragdoll, pose animation, teacher action or training is used. The public SHA remains `898d1a02829f901fbad4ed07178df2b06778620b54e8bccf5e07ffcb4c31d32d`.

New recordings are under `.render-work/comparisons-v6/`; original v5 trajectories and reports are retained. Each fresh run records its own native evaluation end time, initial-pose match and previous-trajectory difference, since repeated GPU physics may diverge slightly. Prior four-seed parkour results are archived separately in each new report rather than attributed to the newly recorded episode.

Comparison videos play once and hold the final frame. Replay is explicit. A visible badge identifies the post-termination interval; provenance includes the native end time, four-second continuation and zero-reset/frozen-controller checks. Successful OFF behaviors, including skate and upright tennis/moonwalk footage, remain unchanged. The archived montage remains available independently; the current homepage highlights play once and hold their final frame.

High steps in the new recording reaches the 10-second native horizon without completing its goal; its physical fall occurs during the four post-horizon seconds. It is explicitly reported as a timeout/target failure, not an earlier native fall. Gap and step cameras tilt down during the fall to reveal the body inside the original terrain recesses.


## Superseded six-station scene preview — 2026-11-06

An earlier homepage revision presented an explicitly labeled design preview. It is now archived locally and replaced by recorded policy highlights. Route: moonwalk (stage_500/model_499) → forward roll → skateboard → rough ground + high steps → upstairs (final MimicX student_4004) → tennis. Duration is 47.7667 seconds, 1,433 frames, 1080p/30. This is one rendered scene with one G1 and zero camera cuts, not a concatenation of finished videos.

The native motion intervals are 4.0, 4.8, 4.8, 2.0, 3.8, 8.98 and 8.6 seconds respectively. XYZ/joints are interpolated and free-joint rotations use Slerp. Source motions are spatially retargeted into the planned route. Walking, boarding/dismount, racket pickup and settling are authored animation; all scene geometry is presentation geometry. No policy executes and no contact or stability validation is claimed. The tennis section contains two incoming-ball cycles without an opponent.

`render_scene_preview.py` generates the scene, timeline, camera path and MP4. The archived JSON records source hashes, seven motion stages, six authored joins, the lack of continuous-policy execution, interpolation and presentation scope. The current `public/media/scene-preview.json` instead describes the replacement highlight film. Experimental gallery files are unchanged. QA contact sheets and continuity measurements live in `qa/scene-preview/`; these are visual/kinematic checks only.


## Superseded shared-course highlights — 2026-11-06

The user chose edited excerpts. The hero is 26.4 s / 792 frames, with three direct cuts and no per-action labels. Moonwalk and forward roll are removed from the hero; their real gallery comparisons remain available.

| Scene | Highlight interval | Complete-source interval |
| --- | --- | --- |
| Skateboard | 0–4.8 s | 0–4.8 s |
| Rough ground and first raised steps | 4.8–10 s | 12.2667–17.4667 s |
| Upstairs and platform entry | 10–16.4 s | 19–25.4 s |
| Tennis | 16.4–26.4 s | 30.1667–40.1667 s |

All excerpts originate from `reports/continuous_four_scene_v1/route_full_04/`, one actual MuJoCo physical rollout with one frozen HoloMotion public Actor and four explicitly selected existing experts. Simulator localization supplies waypoint commands; no robot root/joint state is rewritten after initial reset, and no teacher actions or training updates occur. Three expert temporal-memory resets are explicit in provenance; executed-action feedback is retained.

`audit_policy_course.py` independently integrates the saved executed joint targets over all 2,257 control steps, including ball-only relaunches/rebound during tennis. Maximum robot root and joint errors are zero. This checks the specific trajectory, not a native-task score, broad robustness or autonomous scene recognition.

The complete source is linked as `policy-course-full.mp4` (45.1667 s). It keeps all three incoming-ball cycles, two with racket contact and a third miss. There is no opponent. The highlight selects the first two cycles and makes no rally or success-rate claim.

The original physical robot carries the native mounted racket throughout. In the edited highlight only, `render_policy_course.py --highlight-props` hides the racket mesh and shadow in non-tennis excerpts. Robot/object poses, inertia and collision simulation are unchanged. The hard cut introduces the tennis scene holding the racket; no pickup animation is fabricated. The complete source retains original prop visibility. `scene-preview.json` flags this presentation change.


## Current native-scene highlights — 2026-11-06

The user removed the shared-course requirement. The homepage now contains four independent native recordings, 31 s / 930 frames, with direct cuts and no action labels. No robot/scene relocation, artificial walking transitions, pose interpolation or racket visibility edits are used.

| Action | Homepage interval | Original recording interval | Native scene |
| --- | --- | --- | --- |
| Skateboard | 0–5 s | 0–5 s | Original HUSKY dynamic skateboard and ground |
| Rough high steps | 5–10 s | 0–5 s | Original Instinct uneven high-step terrain |
| Stair ascent | 10–16 s | 2.8–8.8 s | Original MimicX fifteen-step staircase |
| Tennis | 16–31 s | 0–15 s | Original ATHLETE court, close following camera |

The first three scenes natively contain no racket; only tennis includes it. The first three non-tennis shots run 5–6 seconds each. Tennis retains three continuous incoming-ball cycles, including an unsuccessful return; no opponent exists. The shared frozen public Actor is unchanged across each task’s expert/adapters. Separate recordings are not described as one episode.

`native_scene_highlights.py` reads the original lossless masters for skate/high steps/tennis and replays the actual accepted MimicX student trajectory for stairs. A portable export of the native MuJoCo 3.5 stair scene is used only for visual replay; sampled body positions match the original model exactly (maximum error 0). The complete stair source is 449 physical frames at 50 Hz, rendered to 270 nearest-state frames at 30 fps. `native-stairs-full.json` records the small endpoint time quantization.

All four chapter sources/scenes/reports and full recording links are in `public/media/scene-preview.json` and `public/data/hero-highlights.json`. Earlier shared-course public assets were moved into `.render-work/native-four-scenes-v2/retired-unified-course-public/`; the original simulation, audits and source snapshots remain intact.
