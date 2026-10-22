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

Open question the retrain answers: does a model trained **exactly as the paper describes** (2:1:1, no acceleration
loss, the paper's data size) reach the paper's T2P numbers, and does its text pathway behave differently from the
released one? See §4.

## 4. Retrain (paper config)

`train/run.sh` + `train/queue_paper.sh`: upstream `train.py`, unchanged except one patch (commit "Gradient
accumulation...") so that batch 64 = 2 × 32 microbatches gives the full-batch mean gradient (fp32 batch 64 does not
fit on one 32 GB RTX 5090). 100k steps per stream (lr 1e-5, linear anneal to 0, EMA 0.9999), ~13 h per stream.

Results: *pending (queue running; see `logs/queue_paper.log`, `outputs/eval/paper*_*.json`).*
