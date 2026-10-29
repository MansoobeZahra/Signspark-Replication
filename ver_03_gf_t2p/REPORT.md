# ver_03_gf_t2p: improving text-only SignSparK (GF-T2P)

ver_02 is untouched. Shared assets are symlinks: `SignSparK`, `.venv-signspark` -> ver-01; `smplx` and the
How2Sign test LMDB -> ver_02 (read-only, `lock=False`).

| File | Purpose |
|---|---|
| `soke_metric.py` | SOKE's DTW-JPE / DTW-PA-JPE, ported; `python soke_metric.py` checks it against verbatim copies of SOKE's `dtw`/`l2_dist_align`/`rigid_align` (identical to 1e-15) |
| `eval_t2p.py` | text-only generation on a How2Sign test subset + scoring; `--scale --steps --seed --tag --length --static --gt-baseline ...` |
| `diag_body.py` | step-1 diagnosis: body error split into arm and head joints |
| `outputs/t2p_eval/<name>.json` | per-clip and mean scores, args; `<name>_pred.npz` the generated streams |

## Step 1: reproduce the paper (2026-09-28)

**Metric, as SOKE codes it** (github.com/2000ZRL/SOKE @ 5cbc55d, `mGPT/metrics/t2m.py`, `dtw.py`,
`utils/human_models.py`, `data/H2S.py`):
- SMPL-X neutral with face contour (144 joints); root, legs and spine zero; betas fixed (H2S.py `shape_param`).
- **Body** = neck, shoulders, elbows, wrists, ears, eyes, nose (12 joints). **Hands** = 21 joints each
  (wrist, 15 regressed, 5 fingertip vertices); "hand" = mean of left and right.
- Per-frame distance: align the prediction to GT, then take the mean joint L2 distance.
  - **JPE**: translation only; joint 0 is matched (body: pelvis; hand: wrist).
  - **PA-JPE**: similarity Procrustes over *all* joints passed (body: all 144), then distance on the part's joints.
- **DTW score = the accumulated cost at the end of the path**: a *sum* over path steps of per-frame errors in
  metres, not divided by length. It is averaged over clips. The score therefore grows with clip length.
- Caveat: SOKE's released `TM2TMetrics` computes only the translation-aligned variant and names it "PA".
  Its code comment says `align_idx=None` is DTW-PA-JPE. The SignSparK paper reports both, so its authors changed
  something. Both are implemented here, each named for what it is.

**Setup.** 200 random How2Sign test clips (`default_rng(0)`; mean 165 frames vs 158 on the full test set),
truncated to 304 frames as the loader does; generated at the real length. GT and prediction go through the same
SMPL-X path (SignSparK's `visualize.py`: `flat_hand_mean=True`, legs/spine zero). The LMDB left hand is true
SMPL-X-left (left mean = mirror of right mean: 0.028 rad residual vs 0.427 unmirrored), so GT left is used
as is. Only the model's left item (right-hand frame, as trained) is flipped.

| Metric (SOKE units) | Paper, SignSparK no KF | **Ours** (`<American>`, CFG 2.5, 10 steps) | sample.py T2P (no CFG, 50 steps) | Motionless mean pose (no model) | A different clip's real motion |
|---|---|---|---|---|---|
| Body DTW-PA-JPE | 7.26 | **10.59** ± 0.45 | 10.49 | 7.79 | 10.67 |
| Hand DTW-PA-JPE | 2.72 | **2.54** | 2.55 | 2.16 | 2.82 |
| Body DTW-JPE | 6.30 | **9.55** ± 0.40 | 9.26 | 6.74 | 8.96 |
| Hand DTW-JPE | 11.43 | **12.28** | 12.67 | 11.01 | 13.68 |

(± = standard error over the 200 clips.) Readable scale: ours is 58 mm (body JPE) and 76 mm (hand JPE) per path step.

**Hands reproduce the paper** (within 7%). **The body does not:** ours is 45-52% worse, far outside the sampling error.

**Diagnosis of the body gap**:

| Factor | Result |
|---|---|
| Tag | `<American>` is right: no tag gives 10.86 / 9.67 (worse) |
| Length | real length is right: 304 frames gives 17.14 / 15.62. The score is a sum, so longer is worse |
| Guidance / steps | no CFG at 50 steps (sample.py) gives 10.49 / 9.26, the same as ours |
| Left-hand flip | ours is right: without the flip, hand PA is 3.51 instead of 2.54. The flip does not affect the body |
| Hand-mean convention | SOKE's smplx default (`flat_hand_mean=False`) moves hands to 2.33 / 10.33. Body PA is 10.37, JPE unchanged |
| Units / normalisation | consistent: the same code reproduces the paper's hands, and the static pose lands at the paper's scale |
| Subset | 165 vs 158 mean frames: accounts for ~4%, not ~50% |
| Which joints | arms 11.76 vs 8.47 static, head 6.11 vs 4.32 static; wrists worst (21.1 vs 15.3). The whole upper body wanders, not one joint |

**What remains unexplained:** the released checkpoint and the paper's evaluation details. The HF model card says
the released checkpoints were trained on CSL-Daily + How2Sign + BOBSL. It does not say whether they are the
checkpoints behind the paper's How2Sign table. The guidance, steps and number of candidates used for the paper's
T2P row are also not given. Nothing on our side (tag, length, guidance, steps, flip, units) closes the gap.

**The more important finding: at this scale the metric barely separates signing from standing still.** A motionless
mean pose scores 7.79 / 2.16 / 6.74 / 11.01. That beats our text-only model on every metric (on 94.5% of clips
for body JPE) and beats the paper's own hand numbers. The paper's body numbers are only ~7% better than the
static pose. For text-only generation, DTW-JPE rewards staying near the average pose, so step 2's sweep will
favour settings that move less. It needs a second check, such as motion energy or a comparison with the static
baseline, so that "better score" does not simply mean "less signing".

---

## Step 2 (2026-10-20/21): can noise/seed or generation-time "residuals" make text-only SignSparK sign the sentence?

Supervisor's ideas: (a) change the noise/seed, (b) add residuals in generation. **Short answer: no.**
- Picking the best seed by back-translation drives the BT score above real signing, and it rises almost as much
  when the candidates were generated from the wrong sentence.
- Pose-side checks show only a weak sentence signal, which neither selection nor any residual variant turns into
  identifiable signs (200-way retrieval top-1 ≤ 1%, chance 0.5%).

New files (this folder only; nothing outside modified):

| File | Purpose |
|---|---|
| `bt_common.py` | SignSparK-BT scorer (authors' package, unmodified), BT feature assembly, 200-way retrieval (as `ver_03_madis`, batched DTW, identical to `dtw_fast` to 0.0), motion energy |
| `bt_validate.py` | 2.1: BT on GT, baselines and earlier outputs → `outputs/bt/validate.json` |
| `gen_common.py` | generation with hooks: cached text embeddings, negative-prompt guidance, per-chunk composite conditioning, noise temperature. Reproduces `eval_t2p.generate` to ≤ 0.005 (rot6D; seed-to-seed spread is 0.14) |
| `bt_bestofn.py` | 2.2: best-of-N, `gen` / `gen_ctrl` / `score` → `outputs/bestofn/` |
| `residuals.py` | 2.3: residual variants, `gen <variants>` / `score` → `outputs/residuals/results.json` |
| `render_bestofn.py`, `render_residuals.py` | side-by-side sheets → `renders/bestofn/`, `renders/residuals/` |
| `third_party/SignSparK_BT` @ 8dd7cdf, `third_party/SignSparK_Retrieval` @ b1ed927 | upstream code, cloned for reading/importing, not installed |
| `checkpoints/bt/How2Sign` | HF `LionelLow/SignSparK_BT` @ 0ced459, How2Sign model only (0.49 GB). `How2Sign_nopg/` = same files, config without `pseudo_gloss_dir` (only needed to train the gloss head) |

Logs of every run are in `logs/`.

### 2.0 What upstream released (not installed)

| | **SignSparK_Retrieval** (SR-T2P) | **SignSparK_BT** (back-translation) |
|---|---|---|
| Code | github.com/JianHe0628/SignSparK_Retrieval @ b1ed927 (2026-10-01). Note: underscore, not `-Retrieval` | github.com/JianHe0628/SignSparK_BT @ 8dd7cdf (2026-09-21) |
| What it does | sentence → **glosses** (fine-tuned mBART T2G) → isolated sign clips from a dictionary → stitched LMDB → SignSparK regenerates transitions (keyframe mode, sparse or dense) | poses (240-d rot6D per frame: body 60 + left 90 + right 90) → English text (JoeyNMT/SLT transformer, 3+3 layers, beam 3); scored with BLEU-1..4, chrF, ROUGE |
| Languages | **CSL-Daily and PHOENIX-2014T only.** `datasets.py`: "Only datasets with gloss annotations are supported." **No How2Sign/ASL**: How2Sign has no glosses and there is no ASL isolated-sign dictionary | PHOENIX-2014T (de), CSL-Daily (zh), **How2Sign (en)** |
| Dictionary | built from Spoken2Sign isolated-sign spans (`annotations/{csl,phoenix}_iso.*`) cut out of the SignSparK **train** LMDBs (`train` mode) or the test LMDB (`best_case` = upper bound) | none |
| Downloads | HF `LionelLow/SignSparK_Retrieval`: T2G 1.44 GB (CSL) + 1.43 GB (PHOENIX), annotations 17 MB, optional trimmed mBART bases 1.5 GB each (training only): **2.9 GB minimum, 5.8 GB all**. Plus SignSparK train LMDBs: CSL-Daily 3.33 GB, PHOENIX 1.28 GB. Plus the SignSparK face checkpoint (`SIGNSPARK_CKPT_DIR/{hand,body,face}`) | model dirs: How2Sign 0.49 GB, CSL 0.33 GB, PHOENIX 0.25 GB. Scoring our own features needs only the model dir. Split scoring needs the test LMDB (we have it) |
| Deps | torch 2.10+cu128, transformers 4.56.1, sentencepiece, sentence-transformers, peft, sacrebleu, accelerate | torch, numpy, lmdb, PyYAML, tqdm (all in our SignSparK venv). jieba (zh) and spaCy (How2Sign pseudo-glosses, training only) are optional |
| Licence | code Apache-2.0. **Models, annotations and predictions: non-commercial research only** (HF card `license: other`), under the CSL-Daily/PHOENIX terms | code Apache-2.0; HF card Apache-2.0. The README notes the datasets' own licences govern models trained on them |
| Published | CSL `train`/sparse: DTW-PA 6.77 / 1.22, JPE 6.09 / 6.24 (body/hand) | How2Sign GT (2,183 clips): BLEU-1 23.07, BLEU-4 3.53, chrF 19.81, ROUGE 23.58; KF2P: 22.26 / 3.34 / 19.36 |

**For us:** SR-T2P cannot be used for English → ASL as released. It would need an ASL gloss dictionary and an
English → ASL-gloss model, neither of which exists in the release. It only reproduces CSL/PHOENIX. Hence not installed.
BT is the scorer used below.

### 2.1 Scorer: SignSparK-BT validated

Authors' CLI, `score --split test`, on the How2Sign test LMDB: **BLEU-1 23.03, BLEU-2 10.25, BLEU-3 5.68,
BLEU-4 3.48, chrF 19.77, ROUGE 23.57** vs published 23.07 / 10.33 / 5.73 / 3.53 / 19.81 / 23.58. Reproduced.
Our own feature path (`bt_common`) gives identical numbers.

| How2Sign test (2,183 clips unless noted) | BLEU-1 | BLEU-4 | chrF | ROUGE | distinct outputs |
|---|---|---|---|---|---|
| **Real GT** | **23.03** | **3.48** | **19.77** | 23.57 | 2,125 |
| SignSparK keyframe mode (official dump, README config) | 20.74 | 2.82 | 18.16 | 21.77 | 2,108 |
| GT with the left hand wrongly flipped | 15.69 | 0.60 | 14.07 | 17.67 | 937 |
| Text-only, official `sample.py`, no CFG | 17.18 | 0.69 | 16.46 | 16.03 | 2,124 |
| Text-only, official, CFG 2.5 (2,176) | 17.03 | 0.68 | 16.14 | 16.51 | 2,092 |
| A **different** clip's real motion | 16.48 | 1.10 | 15.08 | 15.84 | 2,125 |
| **Standing still** (GT mean pose) | 15.86 | 0.32 | 15.92 | 15.25 | 28 |
| 200-subset: GT / our CFG 2.5 run (`ours`) / standing still | 22.61 / 16.32 / 13.98 | 3.26 / 0.00 / 0.69 | 19.67 / 15.87 / 14.67 | | |

- **BT does read the poses.** It drops sharply if the left hand is mis-flipped, and keyframe mode stays close to GT.
- **Its floor is high.** A fluent but unrelated sentence (other clip, standing still) already gets BLEU-1 ~16 and
  chrF ~15-16, because How2Sign sentences share many words ("you", "going to", "the").
- Text-only SignSparK sits **at that floor**: 17.2 BLEU-1 vs 16.5 for someone else's real signing.
- BLEU-4 on 50-200 clips is mostly 0 and too noisy to use. Below, chrF is the main number.
- Trap for anyone using the CLI's `--body-npy` mode: `load_sample_dump` flips the left hand of **both** `gt_poses` and
  `pred_poses`. With the README inference config the dump's GT left hand is already true-left, so its GT row is
  then mis-flipped (cf. the 15.69 row). The authors' published dumps must come from `flip_left_hand=True`.

### 2.2 Seed / noise: best-of-N by back-translation

**Setup.**
- **Sentences:** 50 How2Sign test sentences (the 200-subset's indices 0-48, plus 72 = `G3FhmHz_7hs_6-10`,
  a model_tests render clip) and the 6 demo sentences.
- **Candidates:** 32 seeds each (seed 1000·i + j, shared by the hand and body models), CFG 2.5, 10 steps, real
  length. Demo length = words × 9.9 frames/word, the subset's median.
- **Selection:** the candidate whose BT output has the highest sentence-level chrF against the input sentence.
  At inference this is possible: it needs only the text.

**Controls.**
- **Matched control (main):** the same 50 × 32 generation, same lengths, **same noise seeds**, but the text of
  sentence i+25. Only the text differs, and the same selector is applied with sentence i as the reference.
  - The shift is +25 because a +1 shift (first attempt) paired 18/50 clips with a neighbour from the same video,
    i.e. a similar sentence by the same signer.
  - Without length matching the comparison is biased: BT chrF depends on length. Standing still at sentence i's
    length scores 15.8 against sentence i but 15.2 against other sentences.
- **Pose check:** 200-way retrieval of the selected candidate among the subset's 200 real clips
  (length-normalised DTW, as in model_tests). Chance: top-1 0.5%, top-5 2.5%, median rank 100.5.

| 50 test sentences | BLEU-1 | BLEU-4 | chrF (corpus) | mean sentence chrF | retrieval top-1 / top-5 / median rank |
|---|---|---|---|---|---|
| Real GT | 20.92 | 2.55 | 18.77 | | 1 (by construction) |
| Standing still | 15.70 | 0.00 | 15.77 | | 0 / 0 / 123 |
| Other clip's real motion | 15.55 | 0.00 | 14.78 | | 0 / 4% / 96 |
| **N = 1** | 15.41 | 0.00 | 15.55 | 15.73 | 0 / 2% / 88 |
| N = 1, wrong text (matched) | 15.44 | 0.00 | 15.64 | 16.19 (p = 0.86) | 0 / 0 / 122 |
| **Best of 8** | 25.18 | 2.87 | 22.34 | 23.08 | 0 / 4% / 82 |
| Best of 8, wrong text | 22.69 | 1.84 | 20.85 | 22.04 (p = 0.037) | 0 / 0 / 123 |
| **Best of 32** | **29.08** | **4.50** | **25.41** | 26.86 | 0 / 2% / 70 |
| Best of 32, wrong text | 26.48 | 4.47 | 24.31 | 25.71 (p = 0.012) | 0 / 0 / 113 |

(p = one-sided Wilcoxon over the 50 sentences, own vs wrong text.)

- **Best-of-N "beats" real signing on BT** (best of 32: BLEU-1 29.1, chrF 25.4 vs GT 20.9 / 18.8). This is the
  selector exploiting the metric: with 32 different BT sentences per clip, one of them shares words with the
  reference by chance.
  - **Wrong-text candidates get almost the same gain** (26.5 / 24.3). Of the +13.7 BLEU-1 gained from N=1 to 32,
    about 11 is pure selection and ~2.6 can be credited to the text.
  - That small text effect is real (p = 0.01-0.04 at N = 8 and 32; over all 32 candidates own 16.37 vs 15.80,
    p = 0.02), but it is a fraction of the gap between real signing and the floor.
- **Selection does not move the motion towards the real signing.**
  - Within a sentence, the BT score of a candidate and its closeness to the real clip are uncorrelated
    (Spearman 0.006 ± 0.026).
  - The BT-selected candidate never retrieves its own clip (top-1 0% at every N).
  - Its median rank (88 → 82 → 70) stays within the spread of the all-candidates average (mean rank 92.6).
- **Weak pose signal independent of selection.** Over all 1,600 candidates, own text vs wrong text with the same
  noise: mean rank 92.6 vs 109.8 (p = 0.0008), top-5 3.75% vs 2.56%. The text does shift the motion slightly
  towards the right clip, but nowhere near identifiability.
- **Upper bound for seed choice** ("pose oracle": pick the seed whose output is closest to the real recording,
  which needs the recording): best of 32 gives top-1 8%, top-5 28%. With wrong-text candidates the same oracle
  reaches 6% / 20%. **Most of what seed choice can buy is seed diversity, not sentence content.**
- **Demo sentences** (BT output, N = 1 → best of 32):

  | Sentence | N = 1 | Best of 32 |
  |---|---|---|
  | "Hello, how are you?" | "this is called spotting ." | "there are multiple ways to do this ." |
  | "What is your name?" | "that's the kick ." | "and you can see that it is a long time ." |
  | "Where is the bathroom?" | "this is dr paul ." | "you can see that there is different colors ." |

  For "Thank you very much." and "I am learning sign language." candidate 0 is already the best. None of the
  32 BT outputs of any demo sentence is related to its meaning.
- **Renders** (`renders/bestofn/<clip>/sheet.png`; rows: real | N = 1 | BT best of 32 | wrong text, BT best of 32):
  - The real signer works at the face and chin with specific handshapes.
  - All generated rows hold generic hand shapes in front of the chest. They are indistinguishable between own
    and wrong text, and between N = 1 and best of 32.
  - Demo sheets: `renders/bestofn/demo0`, `demo5`.

### 2.3 "Add residuals in generation": what it can mean, and tests

SignSparK text-only = rectified flow, UNet over (joints × time). The text enters as **one pooled 640-d M-CLIP
vector**, added to the timestep embedding (`unet_model.forward_core`). Guidance: v = v(null) + s·(v(e) − v(null)),
where null is a zero vector. Keyframe mode overwrites observed frames (`obs_x0`, `obs_mask`).
Possible meanings of "residual":

| # | Meaning | Retraining? | Tested |
|---|---|---|---|
| R1 | **Guidance strength** (the CFG residual v(e) − v(null), scaled) | no | cfg 1 / 2.5 / 5 / 7.5 |
| R2 | **Negative-prompt residual against the average sentence**: v(e_mean) + s·(v(e) − v(e_mean)), so only what differs from the average sentence is amplified | no | s 2.5, 5 |
| R3 | **Embedding residual**: condition on e_mean + λ(e − e_mean) | no | λ 2, 4 |
| R4 | **Output residual from an average sign**: x_mean + λ(x − x_mean). x_mean = output for e_mean with the same noise; rot6D re-orthonormalised | no | λ 2, 3 (+ x_mean itself) |
| R5 | **Noise residual / temperature**: x_0 scaled by τ | no | τ 0.7, 1.3 |
| R6 | **Per-word (per-chunk) text conditioning**: each time window gets its own text. Training-free version: the model runs once per chunk at every ODE step, and velocities are blended with soft time masks (3 word chunks, span ∝ word count, 9-frame cross-fades) | no (composite); a real per-token cross-attention model would need retraining | chunks3 |
| R7 | **Keyframe / residual injection from real signs** (SR-T2P: dictionary clips as keyframes) | no retraining, but needs an **ASL sign dictionary**, which does not exist in any release (2.0). Keyframes from an "average sign" or from the model's own output carry no sentence information | not testable |
| R8 | Token-level text encoder / cross-attention, residual adapters (ControlNet-style), or training on residuals to a mean motion | **yes** | no |

Same 50 sentences × 8 seeds (seeds identical to step-2 candidates 0-7, so every variant is paired with the
baseline). Measures:
- **BT specificity** = sentence chrF against the own sentence minus that against the other 49 sentences. It cancels
  generic "fluent sentence" gains and length.
- **Retrieval** over all 400 candidates.
- **Paired p** = one-sided Wilcoxon over sentences vs the baseline.

| Variant | BT specificity | p better | retrieval mean rank | top-1 / top-5 | p better | best-of-8 chrF | motion |
|---|---|---|---|---|---|---|---|
| **baseline** (CFG 2.5) | +1.40 | | 92.1 | 0.5% / 4.5% | | 22.34 | 0.018 |
| control: wrong text | +0.77 | (worse, p = 0.03) | 108.5 | 0.25% / 2.3% | (worse, p = 0.002) | 20.85 | 0.017 |
| R4 x_mean (average sign alone) | +0.70 | (worse, p = 0.01) | 106.8 | 0.75% / 2.3% | (worse, p = 0.0001) | 20.74 | 0.018 |
| R1 cfg 1 (no guidance) | +1.26 | 0.69 | 103.1 | 0.5% / 2.8% | (worse, p = 0.001) | 21.60 | 0.018 |
| R1 cfg 5 | +1.15 | 0.89 | 86.6 | 0.25% / 4.0% | 0.04 | 20.88 | 0.018 |
| R1 cfg 7.5 | +1.10 | 0.90 | 86.2 | 0.5% / 4.0% | 0.09 | 20.52 | 0.020 |
| R2 neg-mean s 2.5 | +1.47 | 0.32 | 93.2 | 1.0% / 3.5% | 0.61 | 22.44 | 0.017 |
| R2 neg-mean s 5 | +1.57 | 0.12 | 85.1 | 1.0% / 5.0% | 0.03 | 21.44 | 0.017 |
| R3 emb-res λ 2 | +1.52 | 0.22 | 84.2 | 0.75% / 5.8% | 0.02 | 22.04 | 0.016 |
| R3 emb-res λ 4 | +1.45 | 0.49 | 88.7 | 0.5% / 4.5% | 0.30 | 20.00 | 0.024 |
| R4 out-res λ 2 | +1.44 | 0.62 | 86.2 | 1.0% / 4.3% | **0.002** | 21.68 | 0.020 |
| R4 out-res λ 3 | +1.41 | 0.63 | 84.2 | 1.0% / 5.3% | 0.01 | 21.35 | 0.023 |
| R5 τ 0.7 | +0.95 | 0.91 | 97.3 | 0.5% / 3.0% | (worse, p = 0.03) | 21.27 | 0.010 |
| R5 τ 1.3 | +1.17 | 0.63 | 91.2 | 1.0% / 4.8% | 0.39 | 22.24 | 0.038 |
| R6 chunks3 | +1.04 | 0.91 | 100.1 | 0.25% / 2.8% | (worse, p = 0.0001) | 20.74 | 0.016 |

(Chance: top-1 0.5%, top-5 2.5%, mean rank 100.5. Real GT motion energy 0.016.)

- **Sanity:** wrong text and the average sign alone are significantly worse on both measures, so the measures
  respond to sentence content.
- **BT:** no variant improves specificity (best p = 0.12).
- **Pose:** the variants that push along the sentence-specific direction (R1 cfg 5, R2 s 5, R3 λ 2, R4 λ 2-3) improve
  retrieval rank by ~6-8 of 200. Only R4 λ 2 survives a Bonferroni correction for 14 tests (0.002 × 14 = 0.03).
  Top-1 stays ≤ 1% (chance 0.5%). This amplifies the weak signal from 2.2. It does not create signs.
- Lower noise (τ 0.7) halves the motion and gets worse. Higher noise (τ 1.3) doubles the motion, with no gain.
  **Per-chunk conditioning is worse**: the model has no notion of word timing, so chunk texts only perturb the
  same generic motion.
- **Renders** (`renders/residuals/<clip>/sheet.png`; rows: real | baseline | R4 λ 2 | R3 λ 2 | R6): all variants
  keep the baseline's motion for the same seed with small amplitude changes. None reproduces the real signer's
  locations or handshapes.

### 2.4 Conclusion for the supervisor's two ideas

1. **Changing the seed/noise cannot make text-only SignSparK sign the sentence.**
   - Selecting by back-translation inflates BT beyond real signing, and nearly as much for candidates made from the
     wrong sentence.
   - Even an oracle that picks the seed closest to the real recording mostly profits from seed diversity
     (top-1 8% vs 6% with wrong text).
2. **Generation-time residuals** (guidance, negative-prompt, embedding, output, noise, per-chunk) **do not either.**
   - The best amplify a weak, statistically real sentence signal in the motion (retrieval rank ~92 → ~85 of 200).
   - BT does not see any improvement, and the renders show no sentence-specific signs.
   - The limiting factor is the model: one pooled sentence vector gives a motion that is mostly determined by the
     noise. As found earlier (model_tests §2.5), the same sentence with different seeds differs 3× more than
     different sentences with the same seed.
3. What would change this needs either sign content from real signing (an ASL dictionary + text-to-gloss, as
   SR-T2P does for CSL/PHOENIX, R7) or retraining with token-level text conditioning (R8). Neither is available
   for ASL in the current releases.

Side effect to note: `git fetch` was run in the `SignSparK` symlink (i.e. `ver-01/SignSparK/.git`) to confirm
upstream is at 916402d. That updated only the remote-tracking ref; the working tree is unchanged.
