# Model tests: were SignSparK, MaDiS and SOKE tested correctly?

**Date:** 2026-10-20. **Scope:** text-to-sign models considered as the source for our Speech → sign → robot pipeline.

For each model this report gives the exact inputs, exact outputs, every conversion step and the evidence that
each step is correct. It then gives the results of each model's **official** evaluation code against its paper.
Everything here was produced in `model_tests/`; no existing project folder was modified.

---

## 0. Summary

| Model | Official code run on its own benchmark | Paper reproduced? | Better than standing still? | Different sentences → identifiable signs? |
|---|---|---|---|---|
| **SignSparK** text-only | `sample.py`, How2Sign test, 2,183 clips | **No.** Body DTW-JPE 8.75 vs paper 6.30; hands 12.07 vs 11.43. The authors now state that the released checkpoint/data differ from the paper's | **No** (standing still: 6.41 / 10.48) | **No** (retrieval at chance; outputs nearly identical across sentences) |
| SignSparK keyframe mode | `sample.py`, How2Sign test, 2,183 clips | (not in the paper's T2P table) | **Yes, by far** (1.18 / 3.03), because it is given the real signer's keyframes | yes, but the keyframes come from the real recording |
| **MaDiS** | `test.py --cfg configs/madis_how2sign.yaml`, SOKE's How2Sign test fittings, 2,308 clips | **Yes.** 6.575 / 9.414 vs paper 6.59 / 9.45 (SiBLEU 4.317 / 3.432 vs 4.41 / 3.46) | **No** on body (standing still 5.66 vs 6.58); hands tied (9.63 vs 9.41) | **No** (retrieval 1% top-1 vs 0.5% chance) |
| **SOKE** | cannot be run: the generator checkpoint was **never released** | - | - | - |

**Plain conclusion:** our pipelines are set up correctly, and we can show it:
- Every conversion is verified to ≤ 0.0003 mm.
- The official MaDiS evaluation reproduces its paper to the second decimal.
- Real ground truth passes through the same conversion and renders correctly.

**But matching the benchmark does not mean producing correct signs.** On both benchmarks a motionless mean pose
scores as well as or better than the text-to-sign models. Neither model's output can be matched to its own
sentence above chance level. In the side-by-side renders the real signing is visibly correct, and the
text-only outputs are not.

---

## 1. Shared method and environment

### 1.1 The metric (identical for all models)

**Definition**, as coded by the authors (SOKE `mGPT/metrics/t2m.py` + `dtw.py`; MaDiS `mGPT/metrics/generation.py` + `dtw.py`):
- SMPL-X neutral with face contour (144 joints): root, legs and spine zero; SOKE's fixed betas.
- **Body** = 12 joints: neck, shoulders, elbows, wrists, ears, eyes, nose.
- **Hand** = 21 joints per hand (wrist, 15 regressed joints, 5 fingertip vertices). "Hand" = (left + right) / 2.
- Per-frame cost = mean joint L2 distance, after aligning the prediction to the GT.
  - **JPE:** translation only (body: pelvis; hands: wrist).
  - **PA-JPE:** similarity Procrustes over all joints passed.
- **DTW score = accumulated cost along the path**: a *sum* of per-frame metre errors, not divided by length,
  averaged over clips. The score therefore grows with clip length and frame rate.

**Three implementations, cross-checked** (`official_metric.py`):
1. **MaDiS's released torch DTW** (`madis/mGPT/metrics/dtw.py`), loaded from the authors' file and used for
   DTW-JPE on full test sets.
2. **SOKE's original pure-Python `dtw` + `l2_dist_align`** (`third_party/SOKE/mGPT/metrics/dtw.py` @ 5cbc55d),
   loaded *unmodified*. Its one import (`rigid_align`) is taken verbatim from SOKE's `human_models.py` source.
3. Our vectorised port (`../ver_03_gf_t2p/soke_metric.py`), used for DTW-PA-JPE (its Procrustes is too slow in
   SOKE's own code for full test sets).

**Evidence:** on real model outputs (first 3 clips of each SignSparK run, in each result JSON under
`soke_original_crosscheck`), all three agree to 6 decimals. Example, keyframe mode clip 0:

| | Body JPE | Body PA-JPE | R-hand JPE | R-hand PA-JPE |
|---|---|---|---|---|
| SOKE original | 1.755068 | 2.398637 | 4.370125 | 1.174159 |
| our port | 1.755068 | 2.398637 | 4.370125 | 1.174159 |
| MaDiS official | 1.755068 | - | 4.370125 | - |

On random data the agreement is to 1e-10 (SOKE vs port) and 1e-8 (MaDiS, float32).

**PA subset:** DTW-PA-JPE is computed on the first 300 clips of each test set (the body Procrustes costs ~1 s per
clip). JPE uses all clips. Standing-still PA numbers use the same 300 clips.

### 1.2 Standing still (baseline)

The **mean ground-truth pose of the whole test set, held motionless** for each clip's real length, then scored
with the same code. It contains no information about the sentence. A text-to-sign model that does not beat it is
not producing measurable sentence content.

### 1.3 Environment notes (things that went wrong, and how they were handled)

- **GPU driver.** An NVIDIA driver update on 2026-10-19 09:13 was not followed by a reboot (kernel 595.91,
  userspace 595.99). CUDA works, so all model runs are unaffected. NVML/`nvidia-smi` and **EGL (OpenGL) fail**.
  The side-by-side renders therefore use a CPU fallback renderer (§5) on the same meshes. After a reboot,
  `render_side_by_side.py` uses ver_02's GPU renderer automatically.
- **Download.** `Neural-Sign-Actors.zip` (SOKE's How2Sign fittings, 7.25 GB, approved) dropped at 99%
  ("connection reset", 1.6 MB short). All 2,318 test clips lie in the first 0.77 GB and were extracted from their
  local zip headers (`extract_nsa_test.py`). Verification:
  - 375,380 frame files; every clip complete (frames 0..N−1, no gaps).
  - Every deflate stream ended cleanly.
  - The train/val parts are not needed.
- **MaDiS `--device`.** The first attempt failed because `--device` takes several values and swallowed the
  config overrides. It was rerun with the overrides before `--device` (`run_madis_official.sh`).

---

## 2. SignSparK

**Code:** github.com/JianHe0628/SignSparK at **a08b0d6**, unmodified (`git status` clean).
- Upstream is now at 916402d. Its 4 newer commits change only docs, data download, `split.yaml` comments and a
  new `fast/` folder; `sample.py`, the model and the configs are unchanged.
- **Checkpoints:** `hand/` and `body/ema_0.9999_200000.pt` (Hugging Face `LionelLow/SignSparK`).
- **Data:** `How2Sign_reopt_test.lmdb` (HF `LionelLow/SignSparK_data`, 0.59 GB), 2,183 clips.

### 2.1 Exact input

**Per test clip, one LMDB record** (`np.savez` blob). Example `G1GUMky8kWc_20-2-rgb_front`:

| field | shape / value |
|---|---|
| `language` | `American` |
| `translation` | `One and two and three and four.` |
| `gloss` | `""` (How2Sign has no glosses) |
| `segment` | (112,) int32 in {0, 1, 2}: non-sign / sign / sign start. This clip has 4 sign segments |
| `left_features`, `right_features` | (112, 90) float32: 15 hand joints × rot6D |
| `body_features` | (112, 126) float32: 21 body joints × rot6D (the loader keeps joints 12-21 → 60) |
| `face_features` | (112, 56) (not used) |

**Frame rate = the source video's own rate.** This clip is 3.74 s at 30 fps = 112 frames; the set mixes 24, 30
and 50 fps clips.

**Model conditioning** (built by `signspark/pose_datasets_lmdb.py` + `sample.py`):
- Text `"<American> " + translation`, encoded by M-CLIP (XLM-R Large ViT-B-16+) into **one pooled 640-d vector**.
- A length mask over 304 frames.
- **Keyframe mode:** `obs_x0` = the clip itself; `obs_mask` = first/middle/last frame of every sign segment
  (mode 3; 26.0 keyframes per clip on average, 17% of frames).
- **Text-only:** `eval.keyframes_mask_all=True` (no keyframes).

### 2.2 Exact output

`sample.py` writes one `.npy` dict per stream, next to the checkpoint. Example:
`signspark/ckpt/hand/debug0_eulerstepsize50_can1_anchor0_samples/seed102_clampstep0_kf2p.npy`

| key | shape | content |
|---|---|---|
| `pred_poses` | (1, 4366, 90, 304) float32 | hand stream: 2,183 clips × 2 hands **interleaved L0, R0, L1, R1**; rot6D (first two rotation-matrix rows) |
| `gt_poses` | (1, 4366, 90, 304) | the loader's input (padded with the last frame) |
| `lengths` | (1, 4366) | real length (≤ 304; longer clips truncated); median 140 |
| `text`, `video_names` | (1, 4366) | `"<American> …"`, clip id |
| `keyframes` | (1, 4366, 90, 304) bool | observed frames |

The body stream is the same with B = 2,183 and 60 dims (SMPL-X joints 12-21).

### 2.3 Conversion steps and evidence

| Step | Code | Evidence |
|---|---|---|
| rot6D → rotation → axis-angle; zero legs/spine/root | SignSparK's own `tools/visualize.py:build_smplx_input` | our earlier conversion (`ver_03_gf_t2p/eval_t2p.py:smplx_in`) gives the same SMPL-X joints: **max 0.0003 mm** on 20 GT clips |
| Left hand: flip by diag(1,−1,−1) **only if the array is in the model's right-hand frame** | `score_signspark.left_convention` measures it per array: a true SMPL-X left hand has mean ≈ mirror(mean right) | measured on all frames, see table below |
| SMPL-X joints for the metric | `soke_metric.SoKeJoints` (SOKE's joint set, betas, face contour; `flat_hand_mean=True` = SignSparK's convention) | identical joint definitions to MaDiS's `human_models.py` (indices and regressor vertex ids checked line by line) |
| Metric | `official_metric.py` | §1.1 |

**Left-hand convention, measured** (mean |Δ axis-angle| in rad; the smaller column decides):

| Array | vs mirror(right) | vs right | Convention |
|---|---|---|---|
| GT, README config (`flip_left_hand=False`) | **0.021** | 0.414 | true SMPL-X left |
| Text-only prediction | 0.415 | **0.0015** | model's right-hand frame |
| Keyframe prediction, README config | **0.131** | 0.307 | mostly true-left (copies true-left keyframes), but impure |
| GT and prediction, `flip_left_hand=True` (training value) | 0.427 / 0.425 | **0.024 / 0.023** | right-hand frame, clean |

**Two setup traps, found and handled:**
1. **`tools/visualize.py` flips the left hand of GT and prediction by default.** With the README's inference
   config the GT left hand is *not* in the right-hand frame, so that default mis-converts the GT:
   - text-only left-hand JPE becomes 14.26 instead of 12.32 (hand average 13.04 vs 12.07);
   - keyframe mode: 3.11 vs 3.03.

   Our scorer applies the flip per array. The "visualize default" numbers are in each result JSON for comparison.
2. **The README's inference config (`flip_left_hand: False`) contradicts training (`True`).** In keyframe mode
   the model receives true-left keyframes it never saw in training. The left hand is worse than the right (3.67
   vs 2.39); with `flip_left_hand=True` (512 clips) the left hand is 2.09. The text-only mode is unaffected (no
   keyframes).

### 2.4 Official results (How2Sign test, 2,183 clips; `results/signspark/*.json`)

| Run (all: README command, 50 Euler steps) | Body DTW-JPE | Hand DTW-JPE | Body DTW-PA-JPE | Hand DTW-PA-JPE |
|---|---|---|---|---|
| **Paper, SignSparK (no KF)** (arXiv 2603.10446 Tab. 3) | **6.30** | **11.43** | **7.26** | **2.72** |
| Text-only, no guidance (README default) | 8.75 ± 0.11 | 12.07 ± 0.14 | 10.94 | 2.69 |
| Text-only, guidance 2.5 (2,176 clips, `drop_last`) | 9.57 | 12.43 | 11.87 | 2.83 |
| Keyframe mode (README config) | 1.18 ± 0.02 | 3.03 ± 0.04 | 1.74 | 1.40 |
| Keyframe mode, `flip_left_hand=True` (first 512 clips) | 1.38 | 2.43 | - | - |
| **Standing still** (same GT) | **6.41** | **10.48** | **8.01** | **2.26** |
| Standing still, first 512 clips | 6.64 | 11.86 | - | - |

(± = standard error over clips.)

- **Hands reproduce** (12.07 vs 11.43 JPE; 2.69 vs 2.72 PA). **The body does not** (8.75 vs 6.30).
- The paper says its zero-keyframe row uses classifier-free guidance, at an unstated scale. Guidance 2.5 made
  things *worse*. The earlier sweep (`../ver_03_gf_t2p`) found no tag, length, step-count or flip setting that
  closes the gap.
- **Explanation from the authors** (upstream README, commit 916402d, 2026-10-01): *"The released datasets and
  checkpoints are higher quality than those used in the paper. Datasets have been reoptimized for improved contact
  points, while checkpoints have been trained on 15x more data, so reproduced results may exceed the originally
  reported numbers."* So the released checkpoint and LMDB are **not** the paper's, and an exact reproduction is
  not possible. Here the released model is *worse* on the body, not better.

### 2.5 Earlier SignSparK tests (same conversion code; see the source folders for full details)

| Test | Where | Result |
|---|---|---|
| **Language tag** | `ver_02/PROJECT_REPORT.md` §2.18 | `<ASL>` was never seen in training; the correct tag is `<American>`, verified by output change. Fixed before all tests here |
| 5 test clips, positions in mm (frame-aligned, both hands) | `ver_02/outputs/how2sign_eval/`, `PROJECT_REPORT.md` §2.20 | text-only (our setup): wrist error 128 mm, fingertips 185 mm. **Keyframe mode: 26 mm / 41 mm. Linear interpolation of the same keyframes: 26 mm / 43 mm** (the keyframes carry the motion) |
| "Which sentence is it?" 5×5 DTW matrix | same | text-only matches its own clip 1/5 (chance 1/5) |
| 200-clip subset, our fast setup (CFG 2.5, 10 steps) | `ver_03_gf_t2p/REPORT.md` | 9.55 / 12.28 / 10.59 / 2.54; standing still 6.74 / 11.01 / 7.79 / 2.16; worse than standing still on 94.5% of clips (body) |
| Seed vs text | first diagnosis | the same sentence with different seeds differs 3× more (0.142 rot6D) than different sentences with the same seed (0.047) |
| Spread between sentences (rot6D, right hand) | `ver_03_gf_t2p`, `ver_03_madis` | 0.047 (6 demo) / 0.067 (5 test) vs **0.183 between real clips** |
| 200-way sentence retrieval (length-normalised DTW) | `ver_03_madis/outputs/h2s200/scores.json` | top-1 0.5%, top-5 6%, median rank 88 (chance: 0.5%, 2.5%, 100) |

---

## 3. MaDiS

**Code:** github.com/2000ZRL/madis at **071828e** (cloned unmodified into `model_tests/madis/`).
- **Weights:** HF `2000zrl/MaDiS` @ ceaed59. Used: `sft_how2sign.ckpt` (6.30 GB), `Qwen3-0.6B-Base-en-zh-de`,
  `mean.pt` / `std.pt`, `TOKENS_h2s_csl_phoenix` (test tokens).
- **Data:** SOKE's How2Sign fittings (Neural-Sign-Actors.zip, §1.3) + SOKE's split file
  `how2sign_realigned_test_preprocessed_fps.csv`. MaDiS's own loader keeps **2,308** test clips.
- **Venv:** `ver_03_madis/.venv-madis`, with MaDiS's pinned packages. Torch 2.10+cu128 is shared by hard link,
  because the README's torch 2.5 lacks RTX 5090 support.

### 3.1 Exact input

- **Text:** the plain English sentence (no language tag), e.g. `One and two and three and four.`, filled into
  MaDiS's default template (`<Caption_Placeholder>` → the sentence) and tokenised by the Qwen3 tokenizer.
- **Length:** not given; the model predicts EOS, up to 100 tokens = 400 frames.
- **GT pose files**, one pkl per frame: `data/How2Sign/poses_video_level/test/poses/<clip>/<clip>_<frame>_3D.pkl`.
  Keys: `smplx_root_pose` (3), `smplx_body_pose` (63), `smplx_lhand_pose` (45), `smplx_rhand_pose` (45),
  `smplx_jaw_pose` (3), `smplx_shape` (10), `smplx_expr` (10), `cam_trans` (3).
- The loader keeps the upper body, hands, jaw and expression (133 dims), resamples sources above 24 fps to
  **24 fps**, and normalises with mean/std.
- Example: `G1GUMky8kWc_20-2-rgb_front` has 113 frame files (30 fps) → 88 frames at 24 fps.

### 3.2 Exact output

- Three token streams (body, left hand, right hand) of ≤ 100 tokens each, decoded by MaDiS's VQ-VAEs
  (4 frames per token).
- `test.py` with `TEST.SAVE_PREDICTIONS=True` writes one pkl per clip:
  `madis/results/madis/madis_how2sign/test_rank_0/<clip>.pkl`.
  - `feats_rst` (T_pred, 133) float32, **normalised**
  - `feats_ref` (T_gt, 133)
  - `text`
- Layout after de-normalising: `[0:30]` SMPL-X body joints 12-21 · `[30:75]` left hand · `[75:120]` right
  hand · `[120:123]` jaw · `[123:133]` expression. Axis-angle, `flat_hand_mean=False`, left hand true SMPL-X-left.
- Example: for `G1GUMky8kWc_20-2-rgb_front`, `feats_rst` is (32, 133) while GT `feats_ref` is (88, 133).
  Median predicted / GT length = 0.71.

### 3.3 Conversion steps and evidence

| Step | Code | Evidence |
|---|---|---|
| Official metric path: de-normalise → zero lower body → SOKE betas → SMPL-X (face contour, `flat_hand_mean=False`, with jaw/expression) | MaDiS's own `H2SDataModule.feats2joints`, called by `score_madis_official.py` | re-scoring the saved predictions reproduces `test.py`'s printed numbers **exactly**: 6.575 / 9.414 |
| To SignSparK's stream format (for renders and our robot tools): add `hands_mean`; left hand conjugated into the right-hand frame; axis-angle → rot6D rows | `ver_03_madis/convert_madis.py` | SMPL-X vertices native vs converted: **max 0.0003 mm** on 6 sentences. Control (no left flip): 138-148 mm |
| Left-hand convention | measured on 18,836 generated frames | mean(left) vs mirror(mean right) 0.043 rad, against 0.466 unmirrored: **true SMPL-X left** |

### 3.4 Official results (SOKE's How2Sign test fittings, 2,308 clips)

`results/logs/madis_test.log` (the official table), `results/madis/official_test.json` (re-score + baselines):

| | Body DTW-JPE | Hand DTW-JPE | SiBLEU body | SiBLEU hands | Body DTW-PA-JPE* | Hand DTW-PA-JPE* |
|---|---|---|---|---|---|---|
| **Paper, MaDiS** (arXiv 2601.19577 Tab. 1) | **6.59** | **9.45** | 4.41 | 3.46 | - | - |
| **Official `test.py`, our run** | **6.575** | **9.414** | 4.317 | 3.432 | 6.88 | 2.04 |
| **Standing still** (GT mean pose, same data, same code) | **5.66** | **9.63** | - | - | 6.12 | 1.95 |

(*PA: not reported by MaDiS; first 300 clips.)

- **Reproduced:** body within 0.2% and hands within 0.4% of the paper.
- **Deviations from the README command, all harmless:**
  - `--device 0` (one GPU instead of 4);
  - `TRAIN.PRETRAINED_VAE=` (the tokenizer weights are inside `sft_how2sign.ckpt` and load strictly; the 1.77 GB
    `tokenizer.ckpt` was not downloaded);
  - `TEST.SAVE_PREDICTIONS=True`.
- **Against standing still:** worse on the body (6.58 vs 5.66); MaDiS is better on only **24%** of clips. On the
  hands it is marginally better (9.41 vs 9.63), on **50%** of clips. Procrustes-aligned, standing still is
  better on both.
- **Sentence retrieval** (200 random test clips, each prediction against all 200 GTs, length-normalised DTW):
  top-1 **1.0%**, top-5 6.0%, median rank 86.5 (chance 0.5% / 2.5% / 100).

### 3.5 Earlier MaDiS tests on SignSparK's ground truth (`ver_03_madis/README.md`)

- **Same 200 clips as SignSparK:**

  | | Body JPE | Hand JPE | Body PA | Hand PA |
  |---|---|---|---|---|
  | MaDiS (resampled to real length) | 11.60 | 13.27 | 10.90 | 2.70 |
  | MaDiS (own length) | 10.99 | 11.57 | 10.12 | 2.33 |

  Part of the body error is a **fitting-domain offset**: MaDiS's own mean pose, held still, scores 10.69 against
  6.74 for the GT mean pose (its arms sit ~85 mm lower). That is why §3.4, on MaDiS's own data, is the fair test.
- **Sentence-dependence:** MaDiS's outputs vary between sentences as much as real signing does (spread
  0.15-0.20 vs 0.18; SignSparK 0.05-0.07). But the variation is not identifiable: 200-way retrieval is at
  chance, as in §3.4.

---

## 4. SOKE: the generator was never released

- **Code:** github.com/2000ZRL/SOKE @ 5cbc55d; the repository was last pushed 2026-02-21. **Licence:** CC BY-NC-ND 4.0.
- **Released:** only the tokenizer (`tokenizer.ckpt`, 1.6 GB, a VQ-VAE that cannot generate from text), mean/std,
  and SMPL-X fittings of How2Sign / CSL-Daily / Phoenix.
- **Not released:** the text-to-sign generator (mBART-large, ~610 M params; `test.py` expects
  `experiments/mgpt/SOKE/checkpoints/last.ckpt`), and the word-level retrieval dictionary.
- **GitHub issue #2 "Last.ckpt missing":** opened 2025-10-24, **still open on 2026-10-20**. Six requests
  (2025-10-27 … 2026-10-13); none answered by the repository owner (every commenter's association: `NONE`).
- **Indirect evidence:** MaDiS (same authors) uses SOKE's tokenizer and data. Its paper reports a *retrained*
  SOKE (Qwen3) at 7.18 / 10.09 DTW-JPE on How2Sign, the same benchmark as §3.4, where standing still scores
  5.66 / 9.63.

---

## 5. Side by side: real ground truth vs model output, through the same conversion

`renders/<model>/<clip>/sheet.png` (6 time points) and `video.mp4`. Every row in a sheet is built by the
**same** code path and the same renderer layout (front view | right-hand close-up). The row header states the
measured left-hand convention and the frame count.
- **SignSparK:** GT | official text-only | official keyframe mode.
- **MaDiS:** GT (SOKE fittings) | official `test.py` output.

The CPU fallback renderer (§1.3) shows the same meshes; videos use every 3rd frame.

| Clip | SignSparK (`renders/signspark/`) | MaDiS (`renders/madis/`) |
|---|---|---|
| `G1GUMky8kWc_20-2` "One and two and three and four." | **GT counts on the fingers at the chin: one, two (V), three, four.** Keyframe mode reproduces the counting. Text-only holds a hand at the chin/mouth and never counts | GT (88 frames) counts at the chin. MaDiS (32 frames) keeps the hand at the chest with a pointing hand; no counting |
| `G095RWKQ39g_13-1` "All telescopes should come with a finder." | GT: two-handed signing at chest height (184 frames, 30 fps). Keyframe mode follows it; text-only shows different handshapes and locations | GT (64 frames at 24 fps in SOKE's fittings) vs MaDiS (52 frames): different handshapes and timing |
| `G3FhmHz_7hs_6-10` "Now it's your responsibility to provide the healthy food." | same pattern | same pattern |

**What the renders prove:**
- **The conversion works:** the real recordings render as plausible, correctly-handed signing, and SignSparK's
  keyframe mode reproduces them.
- **The text-only outputs do not show the sentence's signs.**

Note that the same sentence differs between the two benchmarks: "All telescopes…" is 184 frames at 30 fps
(6.1 s) in SignSparK's data but 64 frames at 24 fps (2.7 s) in SOKE's fittings. That is one reason the two
benchmarks' numbers are **not** comparable with each other, only within each benchmark.

---

## 6. Conclusions

**Did we reproduce each paper's numbers?**
- **MaDiS: yes.** The official code on its own test data gives 6.575 / 9.414 vs 6.59 / 9.45 DTW-JPE, and
  SiBLEU 4.317 / 3.432 vs 4.41 / 3.46.
- **SignSparK: hands yes, body no.** Text-only, official `sample.py`, full test set: hands 12.07 / 2.69 vs
  11.43 / 2.72, body 8.75 / 10.94 vs 6.30 / 7.26. What differs:
  1. The authors now state that the released checkpoint (trained on 15× more data) and the re-optimised LMDB are
     not the ones used in the paper. This is a documented difference we cannot remove.
  2. The paper's guidance scale for the zero-keyframe row is not given. None tried (none, 2.5) closes the gap.
  3. Two setup traps in the release would make a careless reproduction worse. We avoided both and documented
     them (§2.3): the left-hand flip default of `visualize.py`, and the `flip_left_hand` mismatch between the
     inference and training configs.
- **SOKE:** cannot be reproduced; the generator was never released.

**Does matching the benchmark mean correct signs? No.** Three independent checks:
1. **Standing still.** A motionless mean pose scores better than SignSparK text-only on body and hands
   (6.41 / 10.48 vs 8.75 / 12.07). It also beats MaDiS on the body and nearly ties it on the hands (5.66 / 9.63 vs
   6.58 / 9.41), on MaDiS's own benchmark where MaDiS matches its paper. The DTW-JPE benchmark is therefore
   almost insensitive to whether any sentence-specific signing is produced.
2. **Sentence retrieval.** If an output contained its sentence's signs, it should be closest to its own real
   clip. Among 200 clips, the right one is found 0.5-1% of the time (chance 0.5%) for both models.
3. **Visual review.** In the side-by-side renders the real signing (e.g. counting 1-2-3-4) is visible and
   correct after the same conversion. Neither text-only model produces it. SignSparK's keyframe mode does, but
   only because it is given the real signer's keyframes; at that keyframe density, plain interpolation is
   just as accurate.

**For our project:**
- Our test set-up is correct. The negative result is a property of the models, not of our pipeline.
- No released text-only model tested here produces identifiable signs.
- Correct signs require motion that comes from real signing (keyframes or retrieved sign clips). That is the
  direction of `ver_03_sr_t2p`.

---

## 7. Reproduction

```bash
cd model_tests
# SignSparK official runs (README command; outputs next to the symlinked checkpoints in signspark/ckpt/<stream>/)
signspark/run_official.sh          # hand/body x keyframe/text-only, + hand keyframe with flip_left_hand=True (512)
signspark/run_official_cfg.sh      # text-only with guidance 2.5
PY=../ver_02/.venv-signspark/bin/python
$PY score_signspark.py --name t2p --hand <hand.npy> --body <body.npy>      # likewise kf2p, t2p_cfg2.5, ...
# MaDiS official test on SOKE's fittings
python3 extract_nsa_test.py        # test clips from Neural-Sign-Actors.zip -> data/How2Sign/poses_video_level/test/poses
./run_madis_official.sh            # test.py --cfg configs/madis_how2sign.yaml --nodebug (+ overrides, §3.4)
cd madis && ../../ver_03_madis/.venv-madis/bin/python ../score_madis_official.py results/madis/madis_how2sign/test_rank_0 official_test
# renders
$PY render_side_by_side.py signspark --t2p <hand.npy> <body.npy> --kf2p <hand.npy> <body.npy>
$PY render_side_by_side.py madis --pred-dir madis/results/madis/madis_how2sign/test_rank_0
```

**Files:**

| Path | Content |
|---|---|
| `official_metric.py` | the three metric implementations, loaded from the authors' files + cross-check |
| `score_signspark.py`, `score_madis_official.py` | scoring, conventions measured, static baseline, retrieval |
| `render_side_by_side.py` | GT vs output renders (ver_02 renderer, CPU fallback) |
| `extract_nsa_test.py` | test-split extraction from the (partial) Neural-Sign-Actors archive |
| `results/signspark/{t2p,t2p_cfg2.5,kf2p,kf2p_flipTrue_512}.json` | per-clip and mean scores, convention evidence, cross-checks |
| `results/madis/official_test.json`, `results/logs/madis_test.log` | MaDiS re-score + baselines; the official printed table |
| `signspark/logs/*.log`, `results/logs/*.log` | full logs of every run |
| `madis/`, `third_party/SOKE/` | upstream code at the commits above, unmodified |
| `data/` (git-ignored) | LMDB link, SOKE split files, MaDiS tokens, extracted test fittings, partial zip |
