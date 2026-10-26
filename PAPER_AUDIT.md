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

## 5. Do the generated signs resemble real ASL signs? (sign-level judge, 2026-10-26)

DTW-JPE cannot answer this (it rewards a motionless pose), so a judge that knows what real ASL signs look like was
built and **calibrated on real signing first**. Code: `judge/` (CPU only; the GPU belonged to the MaDiS run).

**Judge.** A small temporal transformer on joint positions (12 upper-body joints relative to the neck, both hands
relative to their wrists, scaled by shoulder width).
- **Training data:** the 1,000 real isolated-sign WLASL clips we have as SMPL-X fits (SignAvatars release; 124
  signs, "a".."available", 53 signers).
- **Augmentation:** speed, crop, mirror, rotation.
- **Calibration, 5 folds split by signer:** on signers never seen in training, **top-1 56.7%, top-5 82.1%**
  (chance 0.8% / 4.0%). This is the final model: a 3-net ensemble trained on all clips.

**Test** (`judge/spot.py`, `judge/significance.py`):
- Every How2Sign test sentence is scanned with windows of 6 lengths. Each of the 124 signs gets its best window
  score, minus the clip's mean over signs, so that longer clips do not score higher.
- For each word that is a judge class and appears in ≥ 10 test sentences ("a" and "and" excluded; 17 words), the
  AUC asks whether that word's sign scores higher in sentences containing the word than in those that don't.
- p-values come from 2,000 permutations of sentences across clips.
- The words are chosen on the **real** signing only, then the same words are tested on the model.
- Each model is compared with the real signing in its own data: SignSparK with SignSparK's fits of How2Sign,
  MaDiS with SOKE's fits (the fits MaDiS was trained and evaluated on).

| Source | 17 words: mean AUC (p) | words the judge finds in real signing: mean AUC (p) |
|---|---|---|
| Real How2Sign (SignSparK fits) | 0.571 (0.0005) | 0.655 (0.0005), 7 words |
| **SignSparK text-only** (official, 50 steps), same sentences | **0.482 (0.90)** | **0.507 (0.36)** |
| Real How2Sign (SOKE fits) | 0.579 (0.0005) | 0.637 (0.0005), 8 words |
| **MaDiS** (official test.py), same sentences | **0.517 (0.11)** | **0.569 (0.0015)** |

(0.5 = no trace of the sign; chance 95% band ≈ 0.52-0.54.)

Per word, real → model:
- SignSparK: add 0.73 → 0.48, again 0.73 → 0.53, already 0.71 → 0.44, ahead 0.64 → 0.45, against 0.66 → 0.55.
- MaDiS: all 0.55 → 0.60 (p < 0.001), after 0.61 → 0.64 (p = 0.007), already 0.70 → 0.61, add 0.66 → 0.58,
  again 0.65 → 0.52, ahead 0.63 → 0.45.

Renders (`outputs/judge/words/{all,after,again,add}.png`; rows: real WLASL sign | real How2Sign signer | SignSparK |
MaDiS, same sentence, the judge's best window): the real How2Sign signer visibly makes the WLASL sign. Neither
model does at that point.

**Answer:**
- **SignSparK text-only:** its output contains **no detectable real ASL signs**, at chance on every word set.
- **MaDiS:** its output contains a **weak but real trace** of a few frequent signs ("all", "after"), about half
  the effect of real signing on the detectable words. It is not word-by-word correct signing: it is not visible
  in individual renders, and it adds nothing on most words.
- **Limits:**
  - The judge covers only 124 signs (17 testable words).
  - Even real signing scores only 0.57-0.58 mean AUC, because an English word is often not signed with that
    WLASL sign.
  - A bigger WLASL fit set (the ASL dictionary session is extracting more) would make this test stronger.

## 6. Dictionary-based production (SR-T2P): are the signs correct, and does SignSparK beat interpolation? (2026-10-26)

Everything below ran on the CPU (the GPU belongs to the MaDiS queue). Same 200 test clips/sentences as §2 unless
stated. SignSparK = released checkpoints, 10 ODE steps. Code: `eval/evaluate.py --baseline slerp / --keyframes
dense`, `eval/rescore_pasteback.py`, `eval/bt_kf2p.py`, `eval/sr_t2p.py`, `dictionary/`.

**SLERP baseline** = per-joint quaternion SLERP of rot6D between the *same* keyframes SignSparK gets, ends held
constant, no model. (Note: the released body rot6D is not orthonormal, |a1·a2| ≈ 0.05-0.1. Both GT and outputs
go through the same Gram-Schmidt in scoring, so SLERP is exact at keyframes in rotation space.)

### 6.1 Tab. 4 (KF2P, onset/mid/offset keyframes, 15-26 per clip): SignSparK vs SLERP

DTW body PA / hand PA / body JPE / hand JPE. B-T = SignSparK-BT (authors' models). BLEU-4 on 200 clips is noisy, so
chrF is the main number (CSL: per character), with a paired Wilcoxon test on sentence chrF.

| | paper SLERP | paper SignSparK | **our SLERP** | **released SignSparK** | SSK better than SLERP, per clip (body / hand JPE) |
|---|---|---|---|---|---|
| How2Sign | 4.18 / 0.98 / 4.36 / 3.76 | 2.11 / 0.86 / 1.68 / 2.80 | **1.66 / 0.51 / 1.24 / 2.39** | 1.47 / 0.51 / 1.21 / 2.10 | 43% (n.s.) / 60% (p = 2e-5) |
| CSL-Daily | 4.39 / 0.85 / 4.05 / 4.35 | 2.50 / 0.63 / 2.02 / 2.71 | **1.69 / 0.32 / 1.32 / 2.08** | 1.20 / 0.31 / 1.01 / 1.65 | 92% / 89% (p < 1e-28) |
| PHOENIX-2014T | 4.10 / 0.89 / 3.71 / 4.12 | 2.39 / 0.71 / 1.92 / 2.25 | **1.54 / 0.34 / 1.22 / 1.90** | 1.50 / 0.37 / 1.23 / 1.94 | 46% / 46% (n.s.) |

| B-T (BLEU-4 / chrF) | GT | static pose | SLERP | released SignSparK | retrained (§4) | SSK vs SLERP, sentence chrF |
|---|---|---|---|---|---|---|
| How2Sign | 3.26 / 19.67 | 0.49 / 13.06 | 2.23 / 18.35 | 2.48 / 18.61 | 2.67 / 18.66 | p = 0.43 |
| CSL-Daily | 6.23 / 6.95 | 0.00 / 1.90 | 5.80 / 6.68 | 6.23 / 6.95 | 6.29 / 6.96 | p = 0.30 |
| PHOENIX-2014T | 17.48 / 39.83 | 0.98 / 20.63 | 15.11 / 36.21 | 17.18 / 37.55 | 16.57 / 37.76 | p = 0.012 |

- **The paper's SLERP row is not reproduced.** Our SLERP on the same keyframe policy is **2.5-3.5x better** than
  the paper's SLERP. It is better than the paper's own SignSparK on every DTW metric and dataset. Paper B-T
  drops for SLERP are 18 / 29 / 38%; ours are 32 / 7 / 14% (BLEU-4, noisy) and 7 / 4 / 9% (chrF). The paper does
  not say how its SLERP was built. Whatever it was, it is not interpolation between these keyframes.
- **SignSparK still beats SLERP, but by much less than claimed:**
  - CSL-Daily: clearly better on DTW (and the B-T drop is 0 vs 7% BLEU-4).
  - How2Sign: hands only.
  - PHOENIX: tie on DTW, small B-T win (chrF 37.6 vs 36.2, p = 0.012).
  - The released checkpoint was not trained on PHOENIX, according to the SignSparK_Retrieval README.

### 6.2 Stitching regime on real clips (every sign frame kept, only transitions generated; 100 clips)

Interior transitions are short: median 6 frames, p90 11-13. SignSparK does **not** reproduce its keyframes
exactly: mean drift is 0.014-0.017 in rot6D, which costs more than the in-betweening gains. Raw SSK is *worse* than
SLERP (PHOENIX body JPE 1.71 vs 0.84). With the keyframes **pasted back** (as SignSparK_Retrieval's `merge` does):

| body PA / hand PA / body JPE / hand JPE | SLERP | SignSparK + paste-back | SSK better (body / hand JPE) |
|---|---|---|---|
| How2Sign | 0.90 / 0.32 / 0.68 / 1.38 | 0.84 / 0.26 / 0.68 / 1.27 | 50% / 70% (p = 8e-5) |
| CSL-Daily | 1.41 / 0.27 / 1.10 / 1.71 | 0.87 / 0.23 / 0.76 / 1.26 | 94% / 89% (p < 1e-14) |
| PHOENIX-2014T | 1.08 / 0.23 / 0.84 / 1.37 | 0.92 / 0.21 / 0.77 / 1.24 | 72% / 67% (p < 1e-4) |

When the gaps sit between *real, continuous* signing, SignSparK's transitions are closer to the real ones than
SLERP's, as long as the kept frames are pasted back.

### 6.3 SR-T2P (Tab. 3 retrieval rows), released pipeline, back-translated

Upstream `SignSparK_Retrieval` @ b1ed927, unmodified:
- `stitch --mode train`, dictionary = isolated signs from the train split;
- the paper's own T2G gloss predictions (`predictions/<ds>/test_pres.txt`);
- `--no-embed` (skips the 8 GB OOV embedding model): 26 / 45 OOV glosses dropped (PHOENIX / CSL).

Then our CPU generation (`eval/sr_t2p.py`, 10 steps; upstream `generate` uses `sample.py` at 50 steps on CUDA)
with the upstream keyframe policies and paste-back. 200 test sentences per dataset, scored against the real
recording (DTW) and by B-T.

| | DTW body PA / hand PA / body JPE / hand JPE | B-T BLEU-4 / chrF / ROUGE |
|---|---|---|
| **PHOENIX-2014T** GT | | 17.48 / 39.83 / 41.47 |
| paper SR-T2P / README (train, sparse) | 3.68 / 1.18 / 3.51 / 4.82 · 5.52 / 1.08 / 4.87 / 5.14 | (not reported) |
| stitched (upstream eased placeholders, no model) | 5.30 / 1.06 / 4.58 / 5.53 | **18.89 / 39.41 / 42.81** |
| SLERP dense / sparse | 5.26 / 1.06 / 4.54 / 5.49 · 5.23 / 1.05 / 4.52 / 5.49 | 18.81 / 39.60 / 42.54 · 18.65 / 39.79 / 43.44 |
| SignSparK dense / sparse | 5.31 / 1.07 / 4.58 / 5.57 · 5.31 / 1.07 / 4.59 / 5.58 | 18.26 / 38.02 / 42.01 · 17.73 / 38.23 / 41.88 |
| **CSL-Daily** GT | | 6.23 / 6.95 / 25.98 |
| paper SR-T2P / README (train, sparse) | 4.87 / 1.37 / 4.89 / 6.56 · 6.77 / 1.22 / 6.09 / 6.24 | (not reported) |
| stitched | 7.10 / 1.30 / 6.20 / 7.84 | **7.67 / 8.08 / 28.10** |
| SLERP dense / sparse | 7.05 / 1.29 / 6.15 / 7.78 · 7.02 / 1.28 / 6.13 / 7.79 | 7.62 / 8.05 / 28.08 · 7.78 / 8.17 / 28.45 |
| SignSparK dense / sparse | 7.10 / 1.31 / 6.20 / 7.87 · 7.09 / 1.30 / 6.20 / 7.88 | 7.43 / 7.84 / 27.70 · 6.93 / 7.41 / 26.55 |

Paired sentence chrF, SignSparK vs SLERP with the same keyframes:
- PHOENIX: dense 39.5 vs 40.9 (p = 0.018), sparse 40.2 vs 41.5 (p = 0.042).
- CSL: dense 8.87 vs 9.12 (p = 0.77), sparse 8.51 vs 9.35 (p = 0.035).

- **The SR-T2P DTW row reproduces the README to within about 0.3 on the body, not the paper.** Hands on CSL are
  1.6 worse than the README; likely causes are 10 vs 50 steps and the dropped OOV glosses. Every variant scores the
  same within 0.1 DTW, the raw stitch included. **DTW cannot tell the in-betweeners apart here.**
- **Dictionary retrieval gives sentence-specific, readable signing.** Stitched dictionary signs back-translate
  **as well as or better than the real recordings** (PHOENIX BLEU-4 18.9 vs 17.5; CSL 7.7 vs 6.2). Compare
  text-only T2P, which sits at the "someone else's sentence" floor (§3, ver_03_gf_t2p). Caveat: the dictionary
  is cut from the train split, which the BT model was also trained on. So "above GT" is partly
  in-distribution bias, but the signs are clearly the right ones.
- **SignSparK as the in-betweener does not help on stitched isolated signs.** It is equal (CSL dense) or
  significantly worse (PHOENIX both policies, CSL sparse) than SLERP on B-T. Sparse keyframes (regenerating each
  sign from 3 frames) lose sign content. Dense keeps it, but adds nothing over interpolation for 7-frame gaps.

### 6.4 ASL dictionary prototype (WLASL video → NLF SMPL-X), compared with 3D-LEX

- **FAST** (`upstream/fast/`, `signlangtk` 0.3.1) needs **WiLoR hand features** (`.h5`, from `sltk extract
  wilor`). Raw video works only through the WiLoR extractor, which needs the license-gated **`MANO_RIGHT.pkl`**
  (+ `mano_mean_params.npz`). An optional NLF `.h5` assigns hands to signers in multi-person video.
- **The authors' LMDB pipeline** (DATA.md §3): segment (FAST) → extract (WiLoR hands, NLF body, and in `sltk
  smplfx` also TEASER face + RTMPose + optional depth) → **re-optimise** (`smplfx`) → export rot6D. TEASER needs
  FLAME (gated).
- **Prototype, done:**
  - 23 glosses from the robot demo sentences (`ver_03_sr_t2p/sentences.json` + ver_02's six demo sentences),
    3 WLASL clips each.
  - 69/69 fetched (`dictionary/wlasl_fetch.py`).
  - NLF-L SMPL-X fits on the CPU (`dictionary/extract_nlf.py`). The released TorchScript casts to fp16 and pins
    its detector to `cuda:0`, so `checkpoints/nlf/PATCH.md` makes an fp32 CPU copy; boxes come from
    YOLOv8x).
  - Output: SignSparK streams (`data/wlasl/streams/<gloss>/<id>/`).
- **Not yet:** WiLoR hands, FAST segments and `smplfx` re-optimisation. Waiting on MANO (user) and, for WiLoR, a
  2.57 GB checkpoint. Hands are NLF's for now.
- **Comparison** (`dictionary/compare_proto.py`; ver_04's word-level metric unchanged: trim 0.35, DTW,
  handedness-invariant):

| | WLASL → NLF (ours) | SignAvatars-WLASL fits (ver_04 calibration) |
|---|---|---|
| clip → its 3D-LEX sign, 735-way: top-1 / top-5 / median rank | **7.1% / 23.8% / 27** (42 clips, 14 glosses) | 5.3% / 13.2% / 48 (303 clips) |
| clip → 3D-LEX, closed set of the 14 prototype glosses: top-1 / top-5 | **50% / 90%** (chance 7% / 36%) | — |
| leave-one-out nearest other WLASL clip has the same gloss (69 clips) | **41%** (chance 2.9%) | — |

Renders: `outputs/dictionary/renders/<gloss>.png` (rows: WLASL video | NLF fit, 3 signers, + 3D-LEX if animated).

What the renders show:
- Arm trajectories and locations are right: HELLO salutes from the forehead, BATHROOM shakes at the shoulder.
- NLF handshapes are approximate: the T and A shapes come out as loose curls. WiLoR is needed for the hands.
- Some WLASL signers are left-dominant (e.g. HELLO 27180), so clips must be mirrored to right-dominant, as
  `ver_03_sr_t2p/lex3d_convert.MIRROR` does.
- **9 of the 23 glosses** (hello, me, good, need, nice, learn, language, thank you, name) are frozen in 3D-LEX, and
  WLASL covers them all.

### 6.5 Recommendation

1. **Stitching: use SLERP / eased interpolation, not SignSparK, to join dictionary signs.**
   - On stitched isolated signs (the robot's case), SignSparK never beats interpolation on B-T and is
     significantly worse in 3 of 4 settings. DTW cannot separate them.
   - It also costs 1.4 B params per stream, and its kept frames drift unless pasted back.
   - SignSparK + paste-back is better only between real continuous signing (§6.2), which a dictionary
     pipeline does not have.
   - Revisit only with a stitched-sign B-T or recognition test on ASL, once an ASL BT/judge covers the vocabulary.
2. **Dictionary source to scale: WLASL video → (WiLoR + NLF → FAST → `smplfx`) in the LMDB format.**
   - **Coverage:** WLASL has 2,000 glosses, ~10 videos each from different signers (often dead links); 3D-LEX has 735 animated signs, 203 frozen
     (including 9 of our 23).
   - **Agreement:** our NLF-only fits already agree with 3D-LEX better than the SignAvatars fits do.
   - **Selection:** several signers per gloss allow picking the medoid, as SR-T2P's `mode train` does.
   - **Keep 3D-LEX as the reference / tie-breaker** for the glosses it animates: mocap hands are better than any
     video fit.
   - **Next steps:**
     - Get MANO, then install WiLoR (2.57 GB, ask first) for the hands.
     - Add FAST segments, so dictionary entries are trimmed to the sign. That also replaces the wrist-height trim.
     - Mirror left-dominant clips.
     - Pick the medoid clip per gloss.
     - Stitch with SLERP.
     - Score with the ver_04 word metric and B-T once an ASL-capable BT/judge covers the vocabulary.
