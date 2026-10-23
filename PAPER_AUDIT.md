# SignSparK replication: paper vs released code vs our runs

Paper: arXiv 2603.10446v4 (ECCV 2026). Code: github.com/JianHe0628/SignSparK @ 916402d, vendored unmodified in
`upstream/` (commit 2f969e7); every later change to `upstream/` is a separate commit. Data: HF
`LionelLow/SignSparK_data` @ 9fe7e04 (How2Sign, CSL-Daily, PHOENIX-2014T; train/dev/test). Released weights: HF
`LionelLow/SignSparK` @ b907f25 (`{hand,body,face}/ema_0.9999_200000.pt`). No supplementary material is public
(arXiv v1-v4 have none), so anything the paper defers to it is unknown.

## 1. Method, point by point

| # | Paper says | Released code does | Match? |
|---|---|---|---|
| 1 | SMPL-X upper body 10 joints, MANO 15 joints/hand, rot6D (Sec. 3.2) | body 60-d (joints 12-21), hand 90-d, rot6D | yes |
| 2 | Separate channels B, (H_L, H_R), F | 3 separate models (hand/body/face); the hand model is shared by both hands (left flipped into the right frame) | yes (one-hand model is an implementation detail) |
| 3 | Keyframes = onset, mid, offset of each FAST segment (Sec. 3.1) | `keyframe_selection_mode: 3` = first/middle/last frame of each `segment` run | yes |
| 4 | Control signal C = M⊙x1 + (1-M)⊙x_t (Eq. 2) | `_interp_xt_and_mask` | yes |
| 5 | Target u_t = x1 - C_t (Eq. 3) | `vf_gt = x1 - keyframed_x0` = x1 - x0 at free frames (0 at keyframes) | **differs in form**: the code uses the standard rectified-flow target x1 - x0. The paper's x1 - C_t would be (1-t)(x1 - x0). The code's form is the one consistent with Eq. 4 (x1_est = x_t + (1-t)v), so Eq. 3 is a typo in the paper |
| 6 | Recon loss on x1_est = x_t + (1-t)v (Eq. 4-5) | same | yes |
| 7 | Velocity loss (Eq. 6) | same | yes |
| 8 | Final loss = λ_CFM L_CFM + λ_recon L_recon + λ_vel L_vel, best weighting 2:1:1 (Sec. 4.3) | `lambda_flow 0.5, lambda_recon 1.0, lambda_velocity 0.5` (= 1:2:1) **plus `lambda_acceleration 0.1`**, an acceleration loss the paper never mentions | **no**. Our retrain uses the paper's 2:1:1 and no acceleration loss |
| 9 | CFG: guidance inputs dropped with p = 0.1, replaced by null (Sec. 3.2) | text and keyframes each dropped with p = 0.1, independently; null text = zero vector | yes |
| 10 | v_uncond = f(t, ∅, ∅) | `ClassifierFreeSampleModel` nulls only the text; keyframes stay in the uncond branch | minor (in T2P there are no keyframes, so identical) |
| 11 | Text: Multilingual-CLIP (Fig. 3) | M-CLIP XLM-R-Large-ViT-B-16+, **mean-pooled to one 640-d vector**, linear → added to the timestep embedding (AdaGN) | yes, but see §3: one pooled vector, no token-level attention |
| 12 | Language token, e.g. `<ASL>` (Tab. 7) | `"<{language}> "`, language field = `American`, `Chinese`, `German` | yes (the literal tag is `<American>`, not `<ASL>`) |
| 13 | Data: Ph14T + CSL + H2S + 10% BOBSL | BOBSL not released | closest = Tab. 7 row 3 (paper: no measurable difference from adding BOBSL) |
| 14 | 10-step sampling, 0.01 s/vid | Euler ODE, `ode_stepnum` free; README uses 50 | steps of the T2P row unknown (see §2) |
| 15 | Released vs paper weights | README (916402d): released data "reoptimized", checkpoints "trained on 15x more data" | **not the paper's model** |
| — | (not in paper) | sinusoidal timestep embedding fed t ∈ [0,1] with max_period 10⁴ (no ×1000): most frequencies are ~constant | weak time signal; not a bug in the paper's sense |
| — | (not in paper) | inference config `flip_left_hand: False` while training uses `True` | keyframe-mode trap; our evaluator uses the training convention |

## 2. Results on the released checkpoints (`eval/evaluate.py`, 200 random test clips, our harness)

DTW metrics in SOKE units (sum of metres along the DTW path). "Static" = the training set's mean pose, motionless.

**KF2P (Tab. 4), 10 steps: reproduced, better than the paper.**

| | body PA | hand PA | body JPE | hand JPE |
|---|---|---|---|---|
| How2Sign paper / released | 2.11 / **1.47** | 0.86 / **0.51** | 1.68 / **1.21** | 2.80 / **2.10** |
| CSL-Daily paper / released | 2.50 / **1.20** | 0.63 / **0.31** | 2.02 / **1.01** | 2.71 / **1.65** |

**GF-T2P (Tab. 3, zero keyframes), How2Sign: body not reproduced, and the score depends on the step count.**

| ODE steps | body PA | hand PA | body JPE | hand JPE | beats static (body JPE) |
|---|---|---|---|---|---|
| paper | 7.26 | 2.72 | 6.30 | 11.43 | |
| 1 | 7.97 | 2.19 | 7.04 | 10.99 | 34% |
| 2 | 8.43 | 2.22 | 7.41 | 10.88 | 23% |
| 5 | 9.35 | 2.31 | 8.23 | 11.53 | 12% |
| 10 | 9.89 | 2.41 | 8.71 | 12.05 | 10% |
| 50 | 10.49 | 2.52 | 9.26 | 12.61 | 4% |
| **static pose** | **7.88** | **2.16** | **6.87** | **10.89** | |

CSL-Daily, 10 steps: 9.42 / 2.01 / 8.14 / 10.75 vs paper 7.26 / 2.00 / 6.27 / 10.63; static 8.20 / 1.79 / 6.94 / 10.11.

**Text-vs-noise control** (`--controls`, 10 steps): swap the sentence while keeping the noise, or keep the sentence and
swap the noise. Mean per-frame output change in rot6D:

| | text swapped | noise swapped | ratio | own text closer to GT than other text (body JPE) |
|---|---|---|---|---|
| How2Sign | 0.57 | 3.48 | **0.16** | 54% of clips (p = 0.04) |
| CSL-Daily | 0.74 | 3.66 | **0.20** | 62% (p < 0.001) |

D1 (`diagnostics/d1_text_conditioning.py`): pooled M-CLIP embeddings of different sentences are distinct (mean
cosine 0.54). After the conditioning MLP, the sentence-to-sentence spread of the AdaGN vector is only ~10% of its
norm, and 3-4x smaller than its change across flow time.

## 3. Diagnosis so far

1. **Why text-only output gives "wrong signs":** the output is decided ~85% by the noise sample and ~15% by the
   sentence. The sentence enters only as one mean-pooled vector added to the timestep embedding, and only ~9% of
   training samples are text-only (keyframes dropped with p = 0.1). The model therefore learns a good *motion
   prior*, a good *keyframe in-betweener* (KF2P beats the paper), and a weak text→motion mapping.
2. **Why the paper's T2P body number looks better than ours:** DTW-JPE rewards outputs near the mean pose. The
   static pose scores 6.87, and the model approaches it as the step count drops (1 step: 7.04). The paper's 6.30 is
   at the static level. It is consistent with a near-mean (few-step / conditional-mean-like) output, not with
   correct signing. The paper reports no static baseline and no B-T for T2P.
3. **What the paper actually relies on for correct signs:** keyframes (KF2P / SR-T2P, Tab. 2-4). That part
   reproduces. For English→ASL no released gloss dictionary or text-to-gloss model exists (SR-T2P release covers
   CSL and Phoenix only).

The retrain (§4) answers whether the paper-exact recipe changes this: it does not.

## 4. Retrain (paper config)

`train/run.sh` + `train/queue_paper.sh`: upstream `train.py`, unchanged except one patch (commit "Gradient
accumulation...") so that batch 64 = 2 × 32 microbatches gives the full-batch mean gradient (fp32 batch 64 does not
fit on one 32 GB RTX 5090). 100k steps per stream (lr 1e-5, linear anneal to 0, EMA 0.9999), ~13 h per stream.

Finished 2026-10-23 (body 12.0 h, hand 12.0 h). `paper_*` = both retrained streams; `paperbody_*` = retrained body
+ released hand. All rows are from the same harness, the same 200 clips per dataset, and the same seeds.

**Body DTW-JPE, text-only (T2P)** (the metric that did not reproduce):

| | How2Sign | CSL-Daily | PHOENIX-2014T |
|---|---|---|---|
| paper | 6.30 | 6.27 | 4.40 |
| static pose | 7.04 | 6.94 | 4.50 |
| released, 1 / 10 / 50 steps | 7.04 / 8.71 / 9.26 | 6.82 / 8.14 / 8.52 | 5.07 / 5.84 / 6.16 |
| **retrained (paper config), 1 / 10 / 50 steps** | **7.17 / 8.73 / 9.21** | **6.69 / 8.05 / 8.36** | **4.41 / 5.34 / 5.60** |

**Full rows, retrained, 10 steps** (body PA / hand PA / body JPE / hand JPE):
- How2Sign: T2P 9.88 / 2.38 / 8.73 / 11.99 (paper 7.26 / 2.72 / 6.30 / 11.43); KF2P 1.62 / 0.52 / 1.35 / 2.21 (paper 2.11 / 0.86 / 1.68 / 2.80).
- CSL-Daily: T2P 9.42 / 1.94 / 8.05 / 10.63 (paper 7.26 / 2.00 / 6.27 / 10.63); KF2P 1.33 / 0.31 / 1.15 / 1.70 (paper 2.50 / 0.63 / 2.02 / 2.71).
- PHOENIX: T2P 6.12 / 1.35 / 5.34 / 6.77 (paper 5.24 / 1.52 / 4.40 / 7.10); KF2P 1.47 / 0.35 / 1.23 / 1.82 (paper 2.39 / 0.71 / 1.92 / 2.25).

**Text-vs-noise ratio (10 steps):** retrained 0.12 / 0.18 / 0.22 (H2S / CSL / Ph) vs released 0.16 / 0.20 / 0.17.
The own sentence beats another clip's sentence on body JPE in 52 / 66 / 62% of clips (released 54 / 62 / 54%).
At 1 step the ratio is not comparable: a single Euler step adds the noise sample x0 to the output directly.

### Conclusions

1. **The paper's training recipe reproduces the released model's behaviour.** Correct loss weights, no
   acceleration loss and the paper's data quantity give T2P within ±0.1 of the released model on How2Sign and
   slightly better on CSL/PHOENIX. KF2P beats the paper on every metric and dataset, as the released model does.
   **The 1:2:1 + acceleration config is not the cause** of the text-only failure.
2. **The paper's T2P table matches few-step sampling, not real signing.** Our 1-step retrained body on PHOENIX is
   4.41 vs the paper's 4.40, and the hands (6.26 vs 7.10) are better than the paper. On CSL/How2Sign, 1 step is
   0.4-0.9 off. Every T2P number, ours and the paper's, sits at or near the motionless-pose score.
   More ODE steps (= more real motion) always worsen DTW-JPE. The paper reports neither a static baseline nor
   B-T for T2P, so its Tab. 3 cannot show that text-only output carries the sentence.
3. **The specific issue behind "wrong signs": the text pathway, not training noise or a bug.**
   - Swapping the sentence moves the output 5-8x less than swapping the noise, in both the released and our
     retrained model.
   - The sentence is 1 mean-pooled M-CLIP vector added to the timestep embedding (no word-level attention, no
     notion of word timing).
   - Only ~9% of training samples are text-only (keyframe dropout 0.1).
   The model learns a strong motion prior and a strong in-betweener (KF2P), but almost no text→sign mapping.
   That is the paper's design, so retraining it as published cannot fix it.
4. **What would fix it** (each needs new work, none is in the release):
   - (a) Keyframes from real signs: SR-T2P = English→ASL gloss + an ASL sign dictionary (3D-LEX/WLASL in
     `ver_03_sr_t2p`), with SignSparK as the in-betweener. That is the regime the paper actually relies on.
   - (b) Retraining with token-level text conditioning (cross-attention over the XLM-R token states) and a much
     higher keyframe-drop rate, then measuring with sentence retrieval / B-T, not DTW-JPE alone.

Checkpoints: `outputs/train/paper_{body,hand}_100k/checkpoints/*/` (167 GB each; EMA at every 10k steps, plus
model/opt states).
