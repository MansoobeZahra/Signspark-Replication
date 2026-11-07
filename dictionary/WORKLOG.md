# ASL dictionary worklog

Running record of what was tried on the ASL Citizen / WLASL / 3D-LEX SMPL-X dictionary, including failures.
Newest entries at the bottom. Numbers are from the scripts named; "unseen signers" = ASL Citizen test split.

## 2026-10-29 / 30 — before this log (reconstructed)

- **Extraction**: WiLoR + NLF (+ FAST) on ASL Citizen via `dictionary/aslc_extract.py`, sharded. Long jobs killed
  at session end even under setsid/nohup (00:16, 2026-10-30); h5s modified after 00:15:50 deleted as possibly
  truncated (25 wilor, 19 nlf); relaunched as `systemd-run --user` units `aslc_full_{wilor,nlf0,nlf1}`.
- **ASL-LEX 2.0** downloaded (OSF zpha4) to `data/asl_lex/`. ASL Citizen CSV "ASL-LEX Code" joins to ASL-LEX
  "Code" for 2725/2731 glosses.
- **Phonology check** (`dictionary/phonology.py`): logistic-regression attribute classifiers on 14 pose features,
  trained on train-split clips, tested on unseen signers. Agreement: handshape 0.86, location 0.65, one/two hands
  0.93 (majority baselines 0.43 / 0.35 / 0.53). `--apply` arbitration: a replacement for the consensus sign is kept
  only if it agrees with ASL-LEX at least as well. Reverted WHO and WHERE from 3D-LEX to consensus.
- **Visual review of the 27 signs**: PLEASE, SORRY, LEARN, GOOD, THANKYOU, TOMORROW, HELLO correct. Systematic
  fault: hands too high and elbows too bent; YOU points up at the chin instead of at the viewer; MEET, NAME, NICE,
  SLOW, WHAT1/2, LANGUAGE, SIGNLANGUAGE, NEED too high; ME handshape mushy. 3D-LEX signs untrimmed (SEE 141 frames).
- **Diagnosis**: NLF median elbow 61° (ASL Citizen) / 59° (WLASL) vs 74–77° on 3D-LEX mocap; NLF wrist depth
  nearly constant across signs, WiLoR's hand depth (apparent size) varies 0.07–0.29 m in front of the shoulder.
- **Bugs hit**: NLF keypoints in mm (WiLoR in m); h5 needs `import hdf5plugin`; WiLoR left-hand 3D keypoints are
  x-mirrored (first IK version collapsed left elbows to 5–20° until x was un-flipped).
- **Arm-depth fix** (`aslc_build.fix_arm_depth`, A/B in `data/asl_citizen/dict_armfix`): wrist moved along its
  camera ray to WiLoR depth, two-link IK, wrist global orientation kept.
  Elbow median 61.1 → 68.1° (mocap 74.2); forward reach 0.249 → 0.267 m (mocap 0.286); wrist height relative to
  shoulder unchanged (+0.013 vs mocap −0.048 m). 3D-LEX top-1 0.13 → 0.27, WLASL top-3 0.74 → 0.77.
  Visually small: NAME's hands now touch, YOU/MEET still at chin/mouth height. Video:
  `outputs/dictionary/signs_before_after.mp4`.

## 2026-10-30 — user decision: keep the arm fix, work on hand height

### Hand-height diagnosis (`dictionary/height_probe.py`, log `logs/height_probe.log`)

- Earlier "+6 cm" was measured on different sign sets (27 dictionary glosses vs all of 3D-LEX). Redone on
  **matched glosses** (825 ASL Citizen glosses also in 3D-LEX; probe uses 200 x up to 3 clips), same frames
  definition (word-metric trim 0.35).
- 5-clip smoke test: wrist is +0.20–0.30 m higher relative to the shoulder than in 3D-LEX; elbow still ~16° more
  bent after the arm fix.
- **Cause found (hypothesis 1)**: NLF puts a large forward bend in spine1-3 (median +24° relative to the pelvis,
  smoke test). SignSparK streams keep only joints 12-21, so the spine is rendered straight and the arms, which are
  attached to spine3, swing up with it. Folding the spine rotation into neck + collars ("fold") cut the height
  error from +0.23 to +0.04 m in the smoke test.
- Variant "cam" (fold only the spine3 forward pitch relative to the camera vertical) corrected less (+0.14 m):
  NLF tilts the pelvis back in the camera (upper-body-only webcam framing leaves the pelvis unconstrained), so
  camera-referenced lean is smaller and also depends on webcam tilt.
- Side finding: 3D-LEX collars droop ±33° and its neck pitches +22°, ours ~0 (shoulder 9 cm vs 2 cm below the
  neck). Possibly an artifact of the 3D-LEX → SMPL-X conversion, so 3D-LEX is not trusted for shoulder height;
  only wrist-relative-to-shoulder and retrieval are used to decide.
- Speed: open-set retrieval eval on CPU took 70 s/clip (4 variants x 735 DTW); moved the DTW only to the GPU
  (8 s for 3 clips, identical ranks). Full eval runs as systemd unit `aslc_height_eval`.

### 3D-LEX rest-frame trim (`consensus.trim_rest`)

- When a flagged gloss takes the 3D-LEX sign, final/ now gets only the span where a wrist is above 20 % of its
  rise (pelvis-relative), ±4 frames; meta records `trim_rest: [a, b, T]`. Examples: SEE 141 → 47 frames
  (40–87), WHO 130 → 47, BATHROOM 147 → 56, WHERE 116 → 60, YOUR 62 → 49, PLEASE 101 → 65.

### Height probe, full run (600 clips, 200 matched glosses; `outputs/dictionary/height_probe.json`)

Wrist relative to shoulder vs the 3D-LEX sign of the same gloss (median of per-clip differences, m):

| variant | height diff | forward diff | elbow diff |
|---|---|---|---|
| cur (arm fix) | **+0.111** | −0.021 | −11.1° |
| fold (spine → collars) | **−0.007** | −0.050 | −13.4° |
| cam (camera-vertical pitch) | +0.070 | −0.022 | −11.9° |

- NLF spine1-3 pitch relative to pelvis: median +29.6° (IQR 19.7–36.1); pelvis in camera −17.7° (tilted back);
  spine3 in camera +10.8°. So NLF models webcam signers as slouched pelvis + strongly bent spine; dropping the spine
  is what puts the hands at the chin.
- Per ASL-LEX location (height diff cur → fold): Neutral +0.166 → +0.029, Hand +0.148 → +0.005, Arm +0.126 → +0.010,
  Head +0.052 → −0.028, Body +0.026 → −0.046. Fold slightly over-corrects body/head-located signs and moves the
  hand back (Body: wrist 0.17 m in front of shoulder vs 0.25 in 3D-LEX), so chest contact must be checked visually.
- Elbow stays ~11–13° more bent than 3D-LEX in all variants (the arm-depth fix closed only part of it).
- Added `aslc_build.fold_spine` + `--spine-fold` flag (off until the retrieval eval and visual check are in).
- A/B build `data/asl_citizen/dict_fold` (arm fix + spine fold, 27 demo glosses) → consensus → phonology, as
  systemd unit `aslc_fold_ab` (log `logs/fold_ab.log`). `ASLC_SPINE_FOLD=1` env var so consensus.py builds the
  WLASL clips the same way. Failed twice first: `systemd-run --setenv=ASLC_DICT=$PWD/...` unquoted splits on the
  space in "Speech to sign" ("Failed to find executable to:"); quote the whole `--setenv=...` argument.

### Open-set retrieval eval (`dictionary/height_eval.py`, `outputs/dictionary/height_eval.json`)

800 ASL Citizen clips (400 matched glosses x 2) ranked against all 735 3D-LEX signs:

| variant | top-1 | top-5 | top-20 | median rank |
|---|---|---|---|---|
| noarm (old) | 0.016 | 0.049 | 0.131 | 197 |
| cur (arm fix) | 0.019 | 0.049 | 0.133 | 201 |
| **fold** | **0.033** | **0.106** | **0.223** | **110** |
| cam | 0.019 | 0.056 | 0.156 | 172 |

- Spine fold roughly doubles cross-source agreement (top-5 0.049 → 0.106, median rank 201 → 110).
- Honest note: on this open 735-way test the arm-depth fix alone is neutral (top-5 equal, median rank 197 → 201);
  its earlier gain was on the 27-gloss closed set. Kept as decided; it is not hurting.
- Camera-referenced pitch ("cam") is clearly worse than pelvis-referenced fold → rejected.
- Absolute level is low (real SignAvatars recordings reach ~0.55 top-5 on the same bank), so a large
  cross-source gap remains (elbow still ~12° too bent, signer/variant differences).

### Fold A/B on the 27 demo signs (`data/asl_citizen/dict_fold`, now promoted to `dict`)

Closed-set cross-check (27 glosses):

| build | 3D-LEX top-1 | 3D-LEX top-3 | 3D-LEX mean rank | WLASL top-1 | WLASL top-3 |
|---|---|---|---|---|---|
| noarm | 0.13 | 0.53 | 3.93 | 0.52 | 0.74 |
| armfix | 0.27 | 0.53 | 3.27 | 0.54 | 0.77 |
| **armfix + fold** | **0.33** | **0.67** | **3.13** | **0.58** | 0.75 |

- ASL-LEX check (unseen signers): handshape 0.87, location 0.64, two-handed 0.94 — same as before (classifiers are
  refit per build, so this mainly shows nothing broke).
- Flagged: BATHROOM, GOOD, NEED, THANKYOU, WHERE, YOUR. Arbitration (`phonology.py --apply`): THANKYOU and WHERE
  back to consensus; BATHROOM, YOUR keep 3D-LEX (now rest-trimmed); GOOD, NEED keep the clip closest to WLASL.
- **Visual check** (contact sheets armfix vs fold): MEET index fingers now meet at chest height (was mouth) ✔;
  NAME H-hands tap in front of the chest (was mouth) ✔; YOU points forward at chest height (was up at the chin) ✔;
  SORRY, ME stay on the chest ✔. PLEASE sits a little low (lower chest instead of mid-chest), matching the probe's
  −4.6 cm on Body-located signs. Shoulders look slightly hunched (collars carry NLF's forward lean).
  YOU's consensus is only 22 frames (short).
- **Decision**: spine fold on by default (`aslc_build.SPINE_FOLD`, env `ASLC_SPINE_FOLD=0` or `--no-spine-fold`
  to disable). Folder renames: `dict` → `dict_noarm` (old baseline), `dict_fold` → `dict`, same for
  `data/wlasl/` and `outputs/dictionary/signs/`; `phonology_check_dict.json` is now the fold build.
- Sentences re-stitched with the new dictionary: old ones kept in `outputs/dictionary/sentences_noarm/` and
  `all_sentences_noarm.mp4`. Videos: `outputs/dictionary/all_sentences.mp4`, `sentences_before_after.mp4`
  (old top / new bottom), `signs_before_after_fold.mp4` (27 signs, old vs new).
- Hit on the way: a safety check blocked `rm -f $S/*` in a scratch script (unguarded variable); wrote to a fresh
  scratch folder instead.

### Still open
- Elbows ~12° more bent than 3D-LEX after both fixes; body-contact signs ~4 cm low after fold.
- Open-set match to 3D-LEX is 0.106 top-5 vs ~0.55 for real recordings: a big gap remains.
- Full-vocabulary build + recognizer training wait for NLF extraction (≈60k / 83k videos at 11:00).

## 2026-10-30 — user: fix the elbow bend and PLEASE height too

- Reference problem: measuring "PLEASE too low" relative to 3D-LEX's shoulder is confounded by 3D-LEX's drooped
  collars; and 3D-LEX is one signer. New, linguistic ground truth: **ASL-LEX 2.0 Contact + MinorLocation**
  (PLEASE: Body / TorsoTop / contact). For a contact sign the avatar's hand must reach the body surface (gap ≈ 0;
  too-bent elbow = hand inside the body, over-corrected depth = hand floating) at the coded height.
- New knobs in `aslc_build.py`: `DEPTH_SCALE` (env `ASLC_DEPTH_SCALE`, × WiLoR hand depth) and `FOLD_ALPHA`
  (env `ASLC_FOLD_ALPHA`, fraction of the spine bend folded in). Defaults 1.0 / 1.0 = current dictionary.
- `dictionary/contact_eval.py`: signed closest approach of the right-hand surface vertices to the torso/neck/head
  surface (vertex normals), contact height below the neck joint, elbow angle, open-set 3D-LEX rank; grid
  k ∈ {1.1, 1.0, 0.9, 0.8, 0.7} × a ∈ {0.6, 0.8, 1.0}; 3D-LEX signs of the same glosses measured as reference.
  Rest SMPL-X landmarks below the neck joint: spine3 −0.16, spine2 −0.22, spine1 −0.34, pelvis −0.45 m.
- Smoke test (4 clips): k = 1.0 already gives gap +0.014 m; k 0.9 / 0.8 / 0.7 open the elbow (44 → 53 → 68 → 86°)
  but lift the hand off the body (gap +0.06 / +0.11 / +0.15). First sign that the elbow can't simply be opened
  by depth without breaking contact. Full run: systemd unit `aslc_contact_eval`, log `logs/contact_eval.log`.
- Hypothesis "elbow gap = 3D-LEX's lower shoulders" **rejected**: shoulder rel. neck ours (−0.136, −0.032, +0.047)
  vs 3D-LEX (−0.150, −0.090, +0.025); re-solving 3D-LEX's elbow with our shoulder position and its wrists fixed
  changes its median elbow only 84.4 → 85.2° (ours 72.2°). So our hands really are closer to the shoulder.
- But per location (height probe) the elbow gap is as large on contact signs (Body −14°, Head −15°) as elsewhere,
  and for Body signs 3D-LEX's wrist is 0.25 m in front of the shoulder while the SMPL-X chest surface is only
  ~0.14 m in front of it → suspect 3D-LEX hands hover in front of the body. The contact eval's 3D-LEX reference
  row tests this. (A 2D re-check of shoulder height would need YOLO-pose weights, which are not on disk; not
  downloaded.)
- Contact eval, partial (22 body clips): **3D-LEX's own contact signs hover** (gap +4.4 cm median, 75 % > 3 cm) —
  3D-LEX is not an elbow reference for contact signs. Ours at k = 1.0 hover +2.7 cm; k = 1.1 reaches contact
  (gap ≈ 0) with an even more bent elbow (46°); k < 1 opens the elbow but lifts the hand off the body. So on
  contact signs our elbow is not over-bent.
- PLEASE in the dictionary: gap +3.3 cm (hovering), contact point 16.9 cm below the neck joint, palm centre
  22.9 cm (noarm 7.6, armfix 7.9, 3D-LEX 21.6).
- **Contact snap v1** (`contact_snap.py`, nearest body point anywhere): all coded contact signs end within ±4 mm
  of the surface, but GOOD's hand was pressed onto the chest instead of the chin → **v2 restricts the target to
  the ASL-LEX region** (Head → neck/head surface, Body → torso surface).
- **Torso anchor** (pull hands that touch the torso in the unfolded pose back to that spot): **rejected**. In NLF's
  own unfolded pose PLEASE's hand is 8.7 cm in front of the chest at collarbone height, so NLF's hand–chest
  relation is not reliable; and the video check below says unfolded heights are wrong. Code kept,
  `TORSO_ANCHOR` default off (env `ASLC_TORSO_ANCHOR=1`). Bug fixed on the way: Gaussian-smoothed weight could
  exceed 1 by rounding → Slerp ValueError.
- **Video 2D check** (`dictionary/video2d_check.py`, all 27 glosses × 4 clips, `outputs/dictionary/video2d_check.json`):
  dominant wrist height above the shoulder line in shoulder widths, render (front view) − video (NLF 2D fit):
  fold 0.0 / 0.5 / 0.75 / 1.0 / 1.25 → +0.380 / +0.205 / +0.128 / **+0.048** / −0.020 (median |.| 0.380 → 0.092).
  Fold = 1.0 is right (slightly under). Body signs at fold 1.0: ME −0.06, SORRY +0.05, **PLEASE +0.11 (render
  still higher than the signers' video)**. With 3D-LEX's PLEASE at the same height (palm 21.6 cm below neck), two
  independent sources say PLEASE's height is what signers do; my visual "a little low" was wrong. PLEASE's real
  defect is the hover → contact snap. Still too high vs video even at fold 1.0: YOUR +0.26, WHAT2 +0.21,
  LANGUAGE +0.22, THANKYOU +0.19, WHAT1 +0.18 (forward-held signs; possibly webcam perspective/tilt).
- Contact eval stopped after 42 body-contact clips (it ran ~4 h for 280; trend was clear). Summary in
  `logs/contact_eval_partial_summary.txt`. 3D-LEX ref gap +5.4 cm (88 % > 3 cm). Ours, fold 1.0:
  k = 1.1 gap −0.3 cm (24 % > 3 cm), elbow 47°, 3D-LEX median rank 70 · k = 1.0 +2.7 cm, 59°, rank 75 ·
  k = 0.9 +7.7 cm, 73°, rank 80 · k = 0.8 +11.6 cm, 89°, rank 98 · k = 0.7 +17.5 cm, 110°, rank 108.
  → on contact signs, opening the elbow (k < 1) lifts the hand off the body AND matches 3D-LEX worse.
  Fold fraction barely moves contact height (−0.13 to −0.17 m for all a).
- `dictionary/depth_eval.py`: same k grid on 150 non-contact Neutral/Hand glosses (3D-LEX matched), fold on.
  systemd unit `aslc_depth_eval`.

### Depth scale for signs in the air (`depth_eval.py`, 150 non-contact Neutral/Hand glosses, `outputs/dictionary/depth_eval.json`)

| k | top-5 | median rank | elbow vs 3D-LEX | reach (3D-LEX 0.349) | forward (3D-LEX 0.300) |
|---|---|---|---|---|---|
| 1.2 | 0.113 | 94 | −33.7° | 0.220 | 0.152 |
| 1.1 | 0.133 | 98 | −24.5° | 0.264 | 0.208 |
| 1.0 | 0.120 | 108 | −11.6° | 0.308 | 0.262 |
| **0.9** | 0.120 | 112 | **+1.9°** | **0.360** | **0.313** |
| 0.8 | 0.093 | 120 | +18.5° | 0.412 | 0.367 |

- k = 0.9 matches mocap elbow, reach and forward distance; retrieval is flat (top-5 equal, rank within noise).
- Conflict: contact signs need k ≈ 1.1 (k = 0.9 lifts them +7.7 cm). Resolution: **k = 0.9 default + ASL-LEX
  contact snap applied in the build itself** (`aslc_build.apply_contact`, every clip whose gloss is coded
  body/head contact; also the WLASL cross-check clips in consensus.py) and again on final/ after consensus
  averaging. Env: `ASLC_DEPTH_SCALE`, `ASLC_CONTACT_SNAP`.
- A/B build `data/asl_citizen/dict_v3` (build → consensus → phonology --apply → contact_snap), unit `aslc_v3_ab`,
  log `logs/v3_ab.log`.
- v3 A/B result (k 0.9 + fold + contact snap, max gap 12 cm), 27 glosses:
  3D-LEX top-1 0.33 / top-3 0.67 → **0.73** / mean rank 3.13 → **3.00**; WLASL top-1 0.58 → **0.61**, top-3
  0.75 → **0.80**; ASL-LEX check location 0.64 → **0.66**, two-handed 0.94 → 0.95, handshape 0.87 = 0.87.
  Final contact gaps after snap: PLEASE +4.8 → +0.5 cm, SORRY +3.5 → −0.5, ME +0.9 → +0.1, WHO +3.4 → +0.2,
  SEE +6.5 → +1.2. Arbitration now keeps 3D-LEX WHO (ASL-LEX agrees with it in this build).
- **Problem**: head-contact signs hover 11–16 cm in front of the face at k 0.9 (clips: GOOD median 13.7 cm, 22/31
  skipped; THANKYOU 13.5, SEE 11.4, HELLO 8.0) and GOOD's final (18.5 cm) was skipped by the 12 cm limit.
  Raised `MAX_GAP` to 0.20 m: GOOD's fingers then touch the chin (visual check ✔). Rebuilding v3 with it
  (unit `aslc_v3b_ab`, `logs/v3b_ab.log`).
- Found on the way: GOOD's v3 consensus is only 6 frames (0.25 s, starts mid-air) — too-aggressive trim of the
  medoid clip; same family as YOU (22 frames). Not fixed yet.
- Safety check blocked an `rm -rf "$PWD/$D/final_t"` (variable path); not needed, the folder did not exist.
  Leftover test folders: `data/asl_citizen/dict/final_snaptest`, `data/asl_citizen/dict_v3/final_t` (+ their
  renders) — safe to delete by hand.
- **Short signs cause**: the medoid/consensus metric resamples every clip to 64 frames, so a truncated clip
  (GOOD clips trimmed down to 0.26 s, PLEASE 0.33 s) can win as "most typical". Fix: `aslc_build.medoids`
  excludes clips shorter than 0.6 × the gloss's median duration from medoid choice (`index.json` → `short_clips`),
  `consensus.py` excludes them as neighbours. v3b stopped; rebuilding as `aslc_v3c_ab` (`logs/v3c_ab.log`).
- v3c (short-clip fix) result: all body/head contact finals within 0.4 cm of the surface (PLEASE, SORRY, ME,
  GOOD, THANKYOU, WHO, SEE, HELLO); shortest final 25 frames (GOOD 15 → 29). Cross-check vs `dict`: 3D-LEX top-1
  0.33 → 0.40 (top-3 0.67 =), WLASL top-1 0.58 → 0.54, top-3 0.75 → 0.77 (≈ noise at 69 clips).
- **Side-view renders** (`dictionary/render_side.py`; the front view cannot show elbow/depth): v3 MEET arms reach
  further out with a straighter elbow, close to 3D-LEX; PLEASE's hand rests on the chest. New artifact seen only
  from the side: in fold and v3 the **head is bowed** (looking at the chest, +32°).
- Head fix try 1 (counter-rotate neck by NLF's pelvis pitch in the camera): fixed MEET/PLEASE but made HELLO worse
  (its pelvis tilts forward: 14.8 → 28.6°). **Rejected.**
- Head fix try 2 (kept): per clip, neck counter-rotated so the rendered head pitch equals NLF's head pitch in the
  camera as filmed (`aslc_build.HEAD_LEVEL`, meta `head_cam_pitch_deg`/`head_fix_deg`). MEET +32.2 → +11.6
  (filmed +11.6), PLEASE +32.0 → +5.2, HELLO +14.8 → +22.3, YOU +31.6 → +19.9, THANKYOU +24.6 → +9.7.
- Final A/B rebuild with everything: unit `aslc_v3d_ab`, `logs/v3d_ab.log`.
- v3d (all fixes) on the 27-sign closed set, metric as is: 3D-LEX top-1 0.27 / top-3 0.67 / mean 3.27, WLASL top-1
  0.52 / top-3 0.75 — below v3c (0.40 / 0.54). Found why the metric reacts to the head fix: ver_04's 12 body
  points include **5 face landmarks** (SMPL-X 55-59), and 3D-LEX's neck is pitched +22°.
  `dictionary/crosscheck_nohead.py` (face points removed): noarm 3D-LEX top-1 0.33 / top-3 0.53 / mean 3.53,
  WLASL top-1 0.50 / top-3 0.73 · dict(fold) 0.33 / 0.67 / 3.07, 0.59 / 0.73 · v3d 0.27 / 0.67 / 3.20,
  0.50 / 0.76 / mean 2.98 (dict 2.59). → on this small set v3 is level on 3D-LEX and slightly behind on WLASL
  top-1. Confounded: the short-clip rule also changed several medoids.
- Decisive open-set A/B of the clip pipelines on identical clips (`dictionary/pipeline_eval.py`, 200 matched
  glosses × 2, all 735 3D-LEX signs, metric with and without face points): unit `aslc_pipeline_eval`.
- **Open-set pipeline A/B** (`pipeline_eval.py`, 400 clips = 200 matched glosses × 2, identical clips, 735 signs):
  metric as is: fold top-5 0.080 / median rank 112 → v3 0.080 / 108; without face points: 0.083 / 102 → 0.075 /
  103 (v3 better on 34 % of clips, worse on 52 %, mostly by a few ranks). **Elbow median 70.1° → 82.6°** (3D-LEX
  ≈ 84°). → v3 delivers the geometry (elbow, contact, head) with recognisability against 3D-LEX unchanged.
- **Decision: v3 is the default.** Renames: `dict` → `dict_fold`, `dict_v3` → `dict` (also `data/wlasl/`,
  `outputs/dictionary/signs/`; `phonology_check_dict.json` = v3, previous = `_dict_fold`); sentences re-stitched
  (previous in `sentences_fold/`, `all_sentences_fold.mp4`).
- Side views (`outputs/dictionary/side_views/*.png`, before / now / 3D-LEX): MEET arms reach forward with an open
  elbow like the mocap, head upright; GOOD fingertips touch the chin; PLEASE hand on the chest.
- Videos: `outputs/dictionary/sentences_before_after_v3.mp4` (original top / now bottom), `all_sentences.mp4`.
- Remaining: GOOD's final is still 15 frames (the "closest to WLASL" replacement path does not apply the
  short-clip rule yet); open-set match to 3D-LEX still ~0.08–0.12 top-5 vs ~0.55 for real recordings.

## 2026-10-31 — user: "fix GOOD too and continue with the full build"

- GOOD's final was still 15 frames: GOOD is flagged by the cross-check, so its final comes from the
  "ASL Citizen clip closest to the WLASL clips" path, which did not apply the short-clip rule (it picked
  `7988103105967095-GOOD`, one of the 3 clips in `short_clips`). Fix: `consensus.py` excludes `short_clips` from
  those candidates too. Re-ran consensus → phonology --apply → contact_snap on `dict` (unit `aslc_good_fix`,
  `logs/good_fix.log`).
- NLF + WiLoR extraction finished: 83 399 clips each. `fast.pt` only covered the 833 demo clips, so FAST is
  re-run on all WiLoR features → `feats/fast_full.pt` (unit `aslc_fast_full`, `logs/fast_full.log`).
- Full build: one process over 83k clips would take more than a day (and medoids over 2.7k glosses means about
  2.4M CPU DTWs). The recognizer only needs `clips/*/*`, so `aslc_build.py` gained `--shard k/n` (clips only: no
  medoids, no index.json, no LMDB; writes `shard_kofn.json` with clips and failures) and `--skip-done`. Shards
  write into `dict/clips` (the 27 demo glosses are already v3 clips and get skipped).
- GOOD rerun result: final 15 → 29 frames, but the picked clip (`18340714145664716-GOOD`) is one whose hand
  never comes within 20 cm of the chin, so its contact snap was skipped (6 of 31 GOOD clips are like that).
  Also found a bogus snap: `06635586886652112-seedGOOD` had gap −0.56 m (hand "inside" by more than half a
  metre = a bad mesh pose) and was moved 56 cm. Fixes: `contact_snap.snap` skips when |gap| > 20 cm (both
  directions); `aslc_build.medoids` records `no_contact_clips` (contact sign, snap skipped) and excludes them from
  medoid choice; `consensus.py` excludes them as neighbours and as closest-to-WLASL candidates. Rebuilding the
  27 demo glosses with the whole chain (unit `aslc_good_fix2`, `logs/good_fix2.log`).
- User asked how long the medoid + consensus step would take for all glosses. Timed: CPU DTW 31×31 = 3.0 s,
  GPU 0.22 s; one query vs 2 731 signs on the GPU 8 ms. → about 4–5 h on the CPU, 10–20 min on the GPU, plus
  1–2 h of SMPL-X joint FK for 83k clips. My earlier "days" estimate was wrong. Plan: run it after the shards.
- FAST on all WiLoR features: 83 399 clips, 137 389 sign onsets (≈ 25 min on cuda:0). The 833 demo clips'
  segments are identical to the old file. `feats/fast.pt` is now the full file; the old one is kept as
  `feats/fast_demo833.pt`.
- Shard timing: 41 clips in 9.9 s including start-up (≈ 0.2 s/clip). Full build launched as 8 shards × 3 threads,
  units `aslc_full_s0..7`, logs `logs/full_s*.log`, `--skip-done`.
- New `dictionary/full_merge.py` for the all-gloss step: `--shard k/n` runs medoid (`aslc_build.medoids`, its DTW
  swapped for a GPU version of the same DTW) + consensus (`consensus.one_consensus`, refactored out of
  `consensus.main` with no change in behaviour, plus a single-usable-clip case) + `final/<GLOSS>` + ASL-LEX contact
  snap for every non-demo gloss; `--gather` → `index_full.json`. The 27 demo glosses keep their cross-checked
  finals, because ranking them against a 2.7k-sign bank would change their flags. `aslc_build.medoids` now loads
  the SMPL-X joint model once instead of once per call.
- Test on scratch copies of GOOD and NICE (renamed so they count as non-demo): works. Profile with NICE (31 clips):
  4.2 s, of which SMPL-X FK is 2.3 s. The first test took 2.5 min for 2 glosses because the 8 build shards were
  running and the test had no thread limit. → about 3 h for all glosses in one process, < 1 h in 4–6 shards.
- Chain after the shards: `dictionary/run_full_chain.sh` (unit `aslc_full_chain`, `logs/full_chain.log`). It runs recognizer train (GPU) alongside 4 full_merge shards, then gather, then recognizer check.
- `aslc_good_fix2` result: GOOD final = `008087702450223277-GOOD` (closest to WLASL among clips that are neither
  short nor no-contact), **28 frames, chin contact gap +0.17 cm** (snap moved it 1.0 cm). All other contact finals
  stay within 0.4 cm (HELLO −0.2, ME +0.4, PLEASE −0.1, SEE −0.3, SORRY +0.3, THANKYOU −0.1, WHO +0.2 cm).
  Arbitration: GOOD / NEED / THANKYOU keep closest-to-WLASL, WHERE → consensus, WHO and YOUR keep 3D-LEX.
- Shards 1–7 finished in about 50 min: 72 232 clips, 22 failed. **Shard 0 hung** at 100% CPU for 50 min on
  `029351578687442537-BAKE 1`: NLF keypoints at depth 0 gave `inf` rows in the pinhole fit
  (`fix_arm_depth`, divide-by-zero warning), and `np.linalg.lstsq` never returns on non-finite input. Fix:
  the fit drops non-finite or depth ≤ 0.1 m keypoints (skips arm-depth if fewer than 10 are left), and the per-frame
  ray rescale skips wrists with depth ≤ 0.1 m. The clip now builds in 0.02 s (21 frames). Shard 0's remaining
  ≈ 9 970 clips restarted as 8 sub-shards (`--shard k/64`, k = 0, 8 … 56, units `aslc_full_s0b_*`); the chain
  script now waits for every `aslc_full_s*` unit and was relaunched.
- 07:48 all clips built: **83 287** clip folders (sub-shards: 1 more failure). Chain started recognizer train and
  4 full_merge shards, and **was OOM-killed at 07:51** (whole unit; 180 glosses done; the recognizer cache
  `recognizer_cache.npz` was complete: 83 284 clips, 4 240 650 frames, train 40 084 / val 10 294 / test 32 906).
  I only noticed 25 min later because my log filter did not cover the unit itself dying. Relaunched as separate
  units with memory caps so an OOM hits only the offending job: `aslc_merge_s0..3` (MemoryMax 5G, resume
  skips done glosses), `aslc_recog_train` (MemoryMax 28G).
- Recognizer OOM cause: `recognizer.load_cache` sliced `z["x"]` inside the list comprehension; an npz re-reads (4 GB) the array on every access → > 28 GB within 25 s. Fixed (read once); relaunched `aslc_recog_train`. full_merge shards use about 1.2 GB each (fine).
- Recognizer trained (80 epochs, 2 728 glosses, train 40 084 clips, about 17 min on cuda:0). Best val top-1 0.769.
  The end-of-train reload crashed: the checkpoint stored the class names as numpy str, which torch 2.6's
  `weights_only` load rejects. Fix: save plain str, load own checkpoint with `weights_only=False`, new mode
  `recognizer.py test` (scores the saved checkpoint).
  **Result: val top-1 0.770 / top-5 0.942 / top-10 0.965 (10 294 clips); test, unseen signers, top-1 0.673 /
  top-5 0.883 / top-10 0.919 (32 906 clips)** → `outputs/dictionary/recognizer_train.json`. ASL Citizen paper
  baselines on the same split, as I recall them: I3D (RGB) about 0.63 / 0.86, ST-GCN (2D pose) about 0.60 / 0.83.
  So the WiLoR+NLF → SMPL-X extraction keeps sign identity at least as well as the video.
- `recognizer.py check` launched (unit `aslc_recog_check`, `logs/recognizer_check.log`).
- `recognizer.py check` (→ `outputs/dictionary/recognizer_check.json`): sentence sign spans top-1 0.80 / top-5 0.90
  (n = 30); WLASL clips 0.56 / 0.88 (n = 66, other signers and recordings, same pipeline); final dictionary signs
  built so far 0.97 / 1.00 (n = 1 677 during the merge, not independent: consensus of mostly train clips);
  **3D-LEX signs 0 / 0 (n = 14)**. The sentence misses are exactly the two finals taken from 3D-LEX: YOUR
  (rank 131) and WHO (rank 493); PLEASE and BATHROOM are rank 2 (MY, TOILET are top-1: similar signs).
  ASL Citizen consensus of the same glosses: YOUR rank 1, WHO rank 1. So the recognizer does not transfer to
  3D-LEX mocap at all (a domain gap: different skeleton conversion, no NLF / arm-depth / fold artefacts); that is not
  evidence against those two particular signs. Not changed: left for the user to decide (YOUR/WHO → consensus?).
- full_merge done: 2 728 glosses, HURDLE and THIS failed (`KeyError: None`). Cause: 3 ASL Citizen glosses contain
  "/" (HURDLE/TRIP1, HURDLE/TRIP2, THIS/IT), so their clips sit one folder deeper (`clips/HURDLE/TRIP1/<stem>`).
  Fix: full_merge finds glosses from `clips/**/meta.json` (medoid file name with "/" → "__"). Rerun → **2 731
  glosses, 83 376 clips, 23 failed clips, 0 gloss errors** (`index_full.json`). Contact snap on non-demo finals:
  720 snapped, 0 skipped, |gap after| median 0.2 cm, max 2.7 cm.
- The recognizer cache had the same blind spot (`clips/*/*/meta.json`: 83 284 clips, 2 728 classes). Fixed
  (`clips/**/meta.json`); first run kept as `recognizer_cache_2728.npz`, `aslc_smplx_2728.pt`,
  `recognizer_{train,check}_2728.json`. Retrain + check: unit `aslc_recog2`, `logs/recognizer2.log`.
- Retrain with all 2 731 glosses (train 40 133 / val 10 304 / test 32 939): **val top-1 0.768 / top-5 0.945;
  test (unseen signers) top-1 0.670 / top-5 0.881 / top-10 0.919** (2 728-gloss run: 0.673 / 0.883 / 0.919;
  same within noise). Check: sentence signs top-1 0.90 / top-5 0.90 (PLEASE and BATHROOM now rank 1; misses only
  the 3D-LEX finals YOUR rank 76, WHO rank 1648); WLASL 0.53 / 0.82; 3D-LEX 0 / 0; final dictionary
  (2 422 words) 0.96 / 1.00, not independent.
- State at end of this run: the full dictionary is in `data/asl_citizen/dict/` (`clips/` 83 376,
  `consensus/` + `final/` for 2 731 glosses, `index_full.json`; the demo `index.json` is unchanged), and the
  recognizer is in `checkpoints/recognizer/aslc_smplx.pt`. LMDB export for the full set was not built (shard mode).
  Open question for the user: replace the 3D-LEX finals YOUR and WHO with their ASL Citizen consensus?
