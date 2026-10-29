# ver_04_finetune: can fine-tuning give a text-to-sign model that produces correct, identifiable ASL signs?

**Date:** 2026-10-22. **Scope:** report only (Prompt 4). Nothing was trained and nothing was downloaded.
No file outside `ver_04_finetune/` was changed. Other folders' code is imported unmodified.

Every number below was **measured** on our RTX 5090 or on our data; none is estimated blind. The GPU benchmarks ran
from 12:08 to 12:11, after the other session's MaDiS step-4 sweep had finished. The GPU was otherwise idle
(< 0.9 GB in use; `logs/run_bench.log`). Raw results are in `outputs/*.json`; scripts are in `scripts/`.

---

## 0. Summary

| | **A. SignSparK + token cross-attention** | **B. MaDiS fine-tuned** | **C. Word pairs added** (3D-LEX / WLASL) |
|---|---|---|---|
| Trainable params | **33.6 M** per stream (8 bottleneck adapters), or 67.2 M (+7 level adapters). Base 1,395 M frozen | **497.0 M** (Qwen3 LM 482.8 M + MoP mappers 14.2 M), full fine-tune. VAEs 56 M frozen | Same as the host model |
| GPU, per stream | B=64: **10.9 GB, 0.30 s/step** (15.6 GB, 0.38 s with level adapters) | fp32 B=16: **23.2 GB, 0.27 s/step**; B=32 OOM. bf16 B=32: 25.4 GB, 0.34 s/step | Words are short: MaDiS word batch B=64 fp32 23.9 GB, 0.34 s |
| Time per epoch | **2.3 min** per stream on How2Sign train (≈29k clips), ×2 streams (hand, body) | **8.7 min** (fp32 B16) / **5.4 min** (bf16 B32) on 30,684 clips. The paper's recipe (150 epochs) ≈ 13-22 h | Word data adds 1,735 clips: +6% per epoch |
| Data missing | How2Sign **train LMDB 7.9 GB** (download needs your OK). The 452 MB dev LMDB is enough for a pilot | Nothing more: the released **train** tokens are correct (§1.4), and raw poses + re-tokenised val/test for all three datasets are now in `ver_03_madis/data/{H2S,CSL,Phoenix}_full` (the partial zip held all 35,007 How2Sign clips) | Nothing for SignSparK: 735 3D-LEX signs are now converted (§1.5). For MaDiS: **its tokenizer cannot represent 3D-LEX signs (§1.6)** |
| Blocker | none found | none for B on its own (released val/test tokens are broken, but re-tokenised copies exist) | MaDiS: the VQ-VAE would have to be retrained, i.e. effectively retrain MaDiS |
| Verdict | **Feasible and cheap** | Feasible but expensive. On How2Sign alone it is likely to change little (it is already trained 150 epochs on exactly this data) | **C on A: the pilot (§3).** C on B: not without retraining the tokenizer |

Three findings matter beyond the cost table:
1. **The word-level test needs a ceiling.** A real WLASL recording of a word finds the same word's 3D-LEX sign only
   **5%** of the time (top-1 of 735; chance 0.14%; median rank 48 after trimming rest poses). Recognising a sign by 3D pose
   distance across signers and capture methods is hard even for real signing. "Correct = closest of all 735" therefore
   can't be the pass mark on its own. The plan scores every model against this measured ceiling and against
   wrong-word controls (§2.2).
2. **Corrected 2026-10-22: MaDiS's released *val/test* tokens are broken, its *train* tokens are fine.** The first
   version of this point (test clips only) suspected that MaDiS was trained on tokens its decoder can't read. The MaDiS
   session's full check rules that out. The released train tokens match the tokenizer's encoding at 99.1-99.98% in
   How2Sign, CSL-Daily and Phoenix; val/test are at chance. **Tokens don't explain MaDiS's wrong signs**, but
   SiBLEU computed against the released test tokens is meaningless (§1.4).
3. **MaDiS's tokenizer can't carry 3D-LEX signs.** After a 3D-LEX round trip through it, the wrist moves by
   **258 mm** (How2Sign: 64 mm), and the signs no longer identify as themselves (top-1 1.5%) (§1.6).

**Recommendation:** pilot C on A, i.e. SignSparK + token cross-attention adapters, trained on 3D-LEX + WLASL word
pairs mixed with How2Sign dev sentences. No large download is needed, and it takes about 2-3 GPU hours (§3).

---

## 1. Candidates in detail

### 1.1 A: SignSparK + a residual per-token text branch

**Why:** text-only SignSparK sees the sentence as one mean-pooled 640-d M-CLIP vector, added to the timestep embedding
(`SignSparK/signspark/unet_model.py:146-149, 206-207`). Seed and generation-time residuals can't fix that
(`ver_03_gf_t2p/REPORT.md` step 2).

**Architecture facts** (read from the code and checkpoints; ver-01/SignSparK @ 916402d):
- Each stream (hand 90-d, body 60-d) is a separate model: M-CLIP text encoder 560.5 M + UNet ≈ 834.8 M = **1,395 M**.
  The checkpoint file is the fp32 EMA weights only (5.58 GB).
- The UNet has **no self-attention** at the temporal levels (`attention=False`). The only attention is in **8
  bottleneck transformer blocks** (1024 ch, T=38 at 304 frames). Their cross-attention attends to *one* token, the
  pooled condition (`unet_model_large.py:250, 267-279`).
- Per-token states exist: `text_enc_model.transformer(**tok)[0]` is (B, L, 1024), mean-pooled at `unet_model.py:148`.

**Code changes** (all in `ver_04_finetune`, nothing edited in SignSparK; prototype in `scripts/bench_a_signspark_xattn.py`):
- `TokenXAttn`: LayerNorm(x) → Q; LayerNorm(token states) → K, V; 16 heads × 64; the output projection is
  **zero-initialised** and added as a residual.
- Installed with `register_forward_hook` after each bottleneck block (`mid`, 8 adapters). Option `mid+lev` adds one after
  each down/up level (7 more, 512-2048 ch, T=38-304).
- **Identity at init is verified:** hooked vs unhooked output differs by **0.0** (max abs) for both streams and both placements.
- Text-only training: `obs_mask` all False (the released model saw keyframes in 90% of training steps, which is why its
  text path is weak). Loss: SignSparK's own `flow.training_losses_poses` (flow + recon + velocity + acceleration).
- Still to write for a real run:
  - a training loop: AdamW on the adapters only; EMA of the adapters; the CFG drop must also empty the token
    context (p = 0.1, as `mask_cond`);
  - a dataset class mixing the LMDB and word folders;
  - saving the adapters only (~135 MB per stream at 33.6 M).

**Measured** (`outputs/bench_a_{hand,body}.json`; real How2Sign test clips standing in for train batches (same LMDB
format), padded to 304; one step = XLM-R token encoding + forward + backward + AdamW):

| Placement | Adapter params | B | s/step (of which text enc.) | Peak GPU | min/epoch, How2Sign train 29k | + 1,735 word clips |
|---|---|---|---|---|---|---|
| mid (8) | 33.6 M | 16 | 0.102 (0.021) | 7.3 GB | 3.1 | 3.3 |
| mid (8) | 33.6 M | 32 | 0.173-0.178 | 8.7 GB | 2.6-2.7 | 2.8 |
| mid (8) | 33.6 M | 64 | 0.299-0.304 (0.059) | 10.9 GB | **2.3** | 2.4 |
| mid+lev (15) | 67.2 M | 64 | 0.375-0.377 | 15.6 GB | 2.8 | 3.0 |

Hand and body differ by < 2%. Both streams must be trained, so double the times. Excluded: the LMDB dataloader
(4 workers in SignSparK's loader) and evaluation.

### 1.2 B: MaDiS fine-tuned

**Facts** (ver_03_madis/madis @ 071828e):
- The How2Sign SFT recipe is `configs/madis_how2sign.yaml`: stage `lm_instruct`, 4 GPUs × batch 64, 150 epochs,
  lr 5e-4, AdamW, `32-true`. It starts from `pretrained.ckpt`; we only have `sft_how2sign.ckpt`, which is the *result*
  of that recipe and can be the starting point (`TRAIN.PRETRAINED=…sft_how2sign.ckpt`, `PRETRAINED_VAE=''`).
- **Full fine-tune** of the whole Qwen3 LM; there is no LoRA path in the code. The VAEs are frozen.
- A batch needs **tokens and raw poses**: the reconstruction loss compares against `batch["motion"]`
  (`mGPT/models/madis.py:156`), and the dataset drops clips without poses (`mGPT/data/sign_dataset.py:117-133`).
  - Tokens: all **34,709** How2Sign token files are on disk (train 30,684 / val 1,717 / test 2,308 after the loader's filters).
  - Raw train/val poses: **not** on disk. `Neural-Sign-Actors.zip` (7.25 GB) is truncated; only the test split was extracted.
- `ver_03_madis/ft_benchmark.py` never ran successfully: `generate_madis.load()` returns the model with
  `requires_grad=False`, so `backward()` fails. `scripts/bench_b_madis.py` is a fixed copy. It also takes a real
  AdamW step on a copy of the weights, restores them after each run, and saves nothing.

**Measured** (`outputs/bench_b_madis.json`; How2Sign batches = official test dump tokens + reference motion):

| Precision | B | s/step | Peak GPU | min/epoch (30,684 clips) | 150 epochs (paper recipe) |
|---|---|---|---|---|---|
| 32-true (config) | 8 | 0.140 | 14.7 GB | 8.9 | 22 h |
| 32-true | 16 | 0.271 | 23.2 GB | **8.7** | 22 h |
| 32-true | 32 | OOM | | | |
| bf16-mixed | 16 | 0.190 | 19.4 GB | 6.1 | 15 h |
| bf16-mixed | 32 | 0.338 | 25.4 GB | **5.4** | 13.5 h |
| bf16-mixed | 64 | OOM | | | |

The paper's effective batch is 256. On one GPU, matching it means gradient accumulation (same time per epoch, 8-16×
fewer optimizer steps).

**Expected value of B alone:** low. `sft_how2sign.ckpt` is already 150 epochs of exactly this objective on exactly
this data. The step-4 sampler sweep (`ver_03_madis/logs/step4_sweep.log`, finished 12:08 today) found that no sampler
setting gets retrieval much above chance: median rank 80-97 of 200, vs 56 for an oracle that picks the candidate
closest to the real clip. More epochs of the same data don't address what the evaluations show is missing:
sentence-specific signs.

### 1.3 C: word-level pairs added

**Data** (`outputs/vocab_overlap.json`):

| Source | Signs | Format | Signers / capture | Licence |
|---|---|---|---|---|
| 3D-LEX v1.0 ASL | **735** animated glosses (938 total; 203 frozen FBX), 1 recording each, 24 fps, median 81 frames | converted to SignSparK streams in `data/3dlex/` (§1.5) | mocap (Vicon + gloves), native signers | **CC BY 4.0** |
| SignAvatars WLASL subset | **1,000 clips, 124 glosses** ("a" … "available"), median 8 clips per gloss, 24 fps, median 58 frames | SignSparK streams in `ver_03_sr_t2p/data/converted/` (verified ≤ 0.00025 mm) | SMPL-X fitted to video, many signers | SignAvatars: non-commercial research (request form); WLASL: C-UDA, non-commercial |
| Overlap | **33 glosses** in both | | | |

- How2Sign train contains **682** of the 735 3D-LEX words; 501 of them ≥ 10 times, 157 ≥ 100 times. 105 of the 124 WLASL glosses appear.
- Of the demo-sentence words, **14 are animated in 3D-LEX**: how, you, where, bathroom, sign, meet, please, help, school,
  sorry, tomorrow, see, who, your. Not available (frozen or absent): hello, thank, name, what, learn, language, nice, me.
- Only **13** of the 735 signs have been reviewed by you (`ver_03_sr_t2p/outputs/sign_check`). The other 722 pass
  through the same verified converter but have not been checked by eye. They are untrimmed (rest → sign → rest).

**Text for a word pair:** `"<American> where"` for SignSparK (same tag as How2Sign); the bare word for MaDiS.

### 1.4 Finding (corrected): MaDiS's released val/test tokens are broken; train tokens are fine

**First version (test clips only, `scripts/vq_debug.py`, `vq_debug2.py`, 50 official test clips):**
- The released test tokens, decoded with the checkpoint's own VAEs, land further from the reference motion than
  holding the mean pose (feature L1 0.93 / 1.00 / 0.95 for body / left hand / right hand, vs 0.75 / 0.78 / 0.77).
- Re-encoding the same poses reconstructs them (0.43 / 0.36 / 0.43). Token agreement was 0.1-0.9%.
- From this I suspected MaDiS had been trained on tokens its decoder can't read. **That was wrong**, because only
  test clips were sampled.

**Full check by the MaDiS session** (`ver_03_madis/README.md`, "Diagnosis" point 4; `vq_tokenizer_check{,2,3,4}.py`):
- `tokenizer.ckpt` is bit-identical to the VAEs inside `sft_how2sign.ckpt`.
- Token agreement of the released files with our encoding (body / lh / rh):

  | | How2Sign | CSL-Daily | Phoenix-2014T |
  |---|---|---|---|
  | **train** | **99.5 / 99.3 / 99.1%** | **99.98 / 99.98 / 99.97%** | **99.96 / 99.97 / 99.97%** |
  | val | 0.8 / 0.2 / 0.8% | 1.1 / 0.5 / 1.6% | 0.7 / 2.0 / 0.7% |
  | test | 0.9 / 0.2 / 0.6% | 0.8 / 0.4 / 0.8% | 1.0 / 2.6 / 0.6% |

  Chance is about 1.0 / 0.5 / 0.5%.

**Consequences:**
- MaDiS was trained on correct tokens, so the tokens don't explain its wrong signs.
- SiBLEU against the released test tokens (part of the paper's metric, reproduced in `model_tests`) compares against noise.
- For B, the released train tokens are usable as they are. Re-tokenised val/test and the raw poses of all three datasets
  are in `ver_03_madis/data/{H2S_full,CSL_full,Phoenix_full}`. No further download is needed.

### 1.5 3D-LEX: all 735 animated signs converted

`scripts/lex_convert_all.py` imports `ver_03_sr_t2p/lex3d_convert.py` (unmodified) and writes `data/3dlex/<gloss>/`.
- It took **40 s for 735 signs, 0 failures**. The three already-reviewed signs I compared (where, school, you) are **bit-identical**
  (max diff 0.0) to `ver_03_sr_t2p/data/3dlex/converted/`.
- **Handedness:** the signer is left-dominant in some signs. The automatic rule (mirror if the left wrist travels > 1.2×
  the right) reproduces the manual decision for YOU and YOUR. It doesn't mirror HELP, which is two-handed with similar
  wrist paths. The word metric is mirror-invariant, so this doesn't affect scoring. For training, the plan uses the rule's
  output, with `mirrored_auto` recorded in `meta.json`.
- **Domain gap** (`scripts/domain_gap.py`; mean |z| against SignSparK's How2Sign statistics, rot6D, std floored at 0.05):

  | | body | right hand | left hand |
  |---|---|---|---|
  | How2Sign test itself | 0.75 | 0.64 | 0.65 |
  | WLASL (video fits) | 1.11 | 1.49 | 2.08 |
  | 3D-LEX (mocap) | 2.78 | 1.53 | 1.78 |

  The 3D-LEX body offset is a **collar/shoulder convention**: 3D-LEX raises the arm in the collar (±0.65 rad), while the
  How2Sign fittings raise it in the shoulder. SignSparK is continuous and can learn either. But mixed training would teach
  two conventions for the same arm pose, so the pilot re-expresses the arms in the How2Sign split before training (§3).

### 1.6 Finding: MaDiS's tokenizer can't represent 3D-LEX signs

`scripts/bench_b_madis.py` (round trip) and `scripts/vq_eval.py`:
- **Conversion:** 3D-LEX streams → MaDiS 133-d features. The inverse of `convert_madis.convert` round-trips exactly
  (6e-8). Then the loader's length rule, normalisation, encode and decode.
- **Scoring:** compared in SOKE joints. Identity check: is the decoded sign still closest to its own original among the 735?

| Mean joint error | wrist | body 12 joints | hand joints | identity top-1 / median rank (of 735) |
|---|---|---|---|---|
| How2Sign test references (200) | 64 mm | 25 mm | 41 mm | |
| 3D-LEX (735) | **258 mm** | 110 mm | 102 mm | **1.5% / 232** |
| 3D-LEX, collars re-split to How2Sign's convention (`vq_collarfix.py`) | 228 mm | 86 mm | 103 mm | 1.0% / 256 |

The 3D-LEX features sit 1.7-2.3 std from MaDiS's training distribution (How2Sign: 0.7-0.8). The tokenizer
snaps them to How2Sign-like codes and loses the sign. **C on MaDiS therefore needs the three VQ-VAEs retrained or
fine-tuned on mixed data.** That changes the codebooks, which invalidates the LM, so it amounts to retraining MaDiS
(tokenizer + pretraining + SFT). Not a pilot.

### 1.7 Licences

| Item | Licence | Consequence |
|---|---|---|
| How2Sign (videos, text) | CC BY-NC 4.0 | non-commercial |
| SignSparK LMDBs (How2Sign fittings) | research use under the source datasets' terms (`SignSparK/DATA.md` §4); HF `license: other` | non-commercial |
| SOKE/Neural-Sign-Actors How2Sign fittings | not stated in anything on disk; **unverified** (assume How2Sign's NC terms) | check before use |
| 3D-LEX v1.0 | CC BY 4.0 | attribution only |
| SignAvatars (WLASL subset) | non-commercial research, request form | non-commercial |
| WLASL | C-UDA (computational use, non-commercial) | non-commercial |
| SignSparK code / checkpoints | Apache-2.0 / non-commercial research | fine-tuned adapters inherit NC |
| MaDiS code + weights | CC BY-NC-ND 4.0 | fine-tuning for internal use is OK; **fine-tuned weights may not be shared** |
| Qwen3-0.6B | Apache-2.0 | |
| M-CLIP (XLM-R Large ViT-B/16+) | not checked here | |
| SMPL-X model | MPI licence, no redistribution | |

Everything we would train is **non-commercial research only**.

---

## 2. Evaluation plan (the same for every candidate)

Each candidate gets its own baseline (the unmodified released model) and the same controls. "Better" always means
better than the baseline **and** than the wrong-text control, tested per item (one-sided Wilcoxon, Bonferroni over the
metrics reported).

### 2.1 Sentences: held-out How2Sign test

- **Set:** the fixed 200-clip subsets already used:
  - SignSparK: `ver_03_gf_t2p` (`default_rng(0)` over its test LMDB);
  - MaDiS: `ver_03_madis/step4_lab.py`'s subset of the official 2,308.
  - The two benchmarks use different fittings, so absolute DTW numbers are only comparable *within* a model.
- **Metrics** (existing code, imported):
  1. Official DTW-JPE / DTW-PA-JPE (body, hands): `ver_03_gf_t2p/soke_metric.py` (SignSparK);
     MaDiS's `feats2joints` + batched DTW (`step4_lab.py`).
  2. **Standing still** (GT mean pose) and the fraction of clips that beat it.
  3. **200-way retrieval**, pose only (top-1, top-5, median rank; chance 0.5% / 2.5% / 100.5): `bt_common.dist_matrix`.
  4. **Back-translation** (SignSparK-BT, validated: BLEU-4 3.48 vs 3.53 published): corpus chrF/BLEU and
     **specificity** = own-sentence chrF − other-sentences chrF (`ver_03_gf_t2p/residuals.py`).
  5. Motion energy (no "better" by moving less) and seed-vs-sentence spread.
- **Controls:** wrong text (sentence i+25, same seed and length); standing still; another clip's real motion.

### 2.2 Words: held-out 3D-LEX signs

**Metric** (`scripts/word_metric.py`, `word_metric_variants.py`):
- The generated sign → SOKE joints → rest phases trimmed (frames where both wrists are below 35% of their height range
  are dropped) → body 12 joints pelvis-relative + both hands wrist-relative, 64 frames.
- Length-normalised DTW to all 735 3D-LEX signs, mirror-invariant. Rank of its own gloss; correct = rank 1.

**Calibration on real recordings** (`outputs/word_metric_calibration.json`, `word_metric_variants.json`). Query = each
of the 303 WLASL clips whose gloss is one of the 33 shared glosses. Target = the 3D-LEX sign of that gloss.

| Variant | top-1 | top-5 | median rank | glosses whose best clip is top-1 / top-5 |
|---|---|---|---|---|
| no trimming | 1.3% | 7.6% | 152 | 9% / 36% |
| trim 20% | 4.3% | 14.2% | 53 | 21% / 55% |
| **trim 35% (chosen)** | **5.3%** | 13.2% | **48** | **30% / 55%** |
| trim 35%, hands only | 5.0% | 15.2% | 57 | 33% / 61% |
| wrists only | 0% | ≤ 2% | ≥ 146 | |
| chance | 0.14% | 0.68% | 368 | |
| self-check (3D-LEX vs itself) | 100% | | 1 | |

This is the **ceiling for cross-signer identification**: real signing of the right word by another person, captured
differently. A model whose word output reaches a median rank near 48 has done as well as real signing.

**Word splits** (fixed once with `default_rng(0)` over the 735; demo words are always in training):

| Split | Words | In training? | What it shows | Ceiling / expectation |
|---|---|---|---|---|
| W-train | the rest of the 735 (602); scored on a 100-word sample | yes (3D-LEX recording) | whether the model can reproduce a word's real sign at all. **Gate 0** | self-check = 100% |
| W-cross | the 33 shared glosses: 3D-LEX recording **held out**, WLASL clips in training | another signer's recording only | whether a sign learned from one source generalises to the same word from another | real-WLASL ceiling: top-1 5%, median rank 48 |
| W-h2s | 100 words seen ≥ 100× in How2Sign train, not used as word pairs (150 available) | only inside How2Sign sentences | the question you asked: does fine-tuning teach signs from sentences? | same ceiling at best. Prior evidence (sentence retrieval at chance) predicts chance |
| demo | 14 words of the demo sentences | yes | used by §2.3 | |

A held-out 3D-LEX word that is in **no** training source (not in WLASL, rare in How2Sign) can't be learned from
3D-LEX, so splitting the 735 at random would measure nothing. Hence the three groups.

**Generation:**
- Real length of the 3D-LEX recording (as the How2Sign test uses the real length); CFG 2.5, 10 steps, seeds 0-3.
- Score per seed and the mean over seeds; never best-of-N by the metric itself.
- **Controls:** the same seed with a different word (wrong-word); the base model (no fine-tuning); standing still.

### 2.3 Words inside sentences, and your review

- **Demo sentences** (the 6 + the 8 approved 3D-LEX sentences) are rendered with the existing renderer
  (`ver_02/render_signspark_reference.py`, `model_tests/render_side_by_side.py`). Rows: the 3D-LEX dictionary signs in
  gloss order | base model | fine-tuned. Plus 1× and 0.5× videos and a Lifeprint/Handspeak link per word.
- **Word-in-sentence probe** (automatic, for the pilot): for each trained word in a generated sentence, use
  subsequence DTW to find the best-matching window of the sentence motion. Then rank that match among the 735 signs.
  Compared with the same window in the wrong-sentence control.

---

## 3. Pilot: C on A (SignSparK adapters + word pairs), no large download

**Why this one:**
- It is the only candidate with no blocker: SignSparK has no tokenizer to retrain, and the word data is already in its stream format.
- It is the cheapest: 33.6 M trainable parameters, < 11 GB, 0.3 s/step.
- It tests the part How2Sign can't: whether the model can be made to produce a real sign when given its word.

**Data (all on disk except one 452 MB file):**
- Word pairs:
  - 3D-LEX W-train (602 words) plus the WLASL clips (1,000, of which the 33 W-cross glosses are kept);
  - W-cross 3D-LEX recordings and all W-h2s words are held out;
  - 3D-LEX arms re-split into How2Sign's collar convention (the same transform as `vq_collarfix.py`). Acceptance:
    the body |z| falls from 2.8 to ≤ 1.2, and the word-metric self-check stays 100%.
- Sentences: the **How2Sign dev LMDB** (`LionelLow/SignSparK_data`, 451.6 MB, 1.7k clips). It keeps the sentence
  behaviour from drifting and needs no approval under the 2 GB rule. **I have not downloaded it.** Say if you want it
  fetched as part of the pilot.
- Mix: 50% word pairs / 50% sentences per batch.

**Training:**
- Adapters `mid` (33.6 M per stream), base frozen, text-only (no keyframes), AdamW lr 1e-4, B=64, EMA 0.999.
- CFG drop 0.1 (pooled vector and token context together). Hand and body streams trained one after the other.
- **Budget:** about 3.5k pairs → 55 steps per epoch → 17 s per epoch per stream. 150 epochs ≈ 45 min per stream,
  **≈ 1.5 h for both**, plus ≈ 1 h of evaluation and renders. Every run is checkpointed and capped at 10 min per process.

**Success** (all measured with §2 against the base model and the wrong-word control):
1. **Gate 0 (it can learn words):** W-train top-1 ≥ 50% and median rank ≤ 5 of 735. Wrong-word control near chance.
2. **Generalisation:** W-cross median rank significantly better than wrong-word (p < 0.05), ideally near the
   real-WLASL ceiling (48).
3. **In sentences:** the probe finds the trained words in the demo sentences, better than the wrong-sentence control,
   and your review of the renders recognises at least half of the in-vocabulary words.
4. **No damage:** How2Sign test 200: retrieval and BT specificity not below the base model; DTW not more than 5% worse.

**Stop if:**
- **Gate 0 fails** (W-train top-1 < 20% after 150 epochs): a frozen base with bottleneck adapters can't carry word
  content. A bigger change (unfreezing the decoder levels, or keyframe-style conditioning) would be a new proposal.
- **Gate 0 passes but criterion 3 fails:** the model reproduces a word only when the word is the whole prompt. That is
  a lookup table, and the 3D-LEX dictionary (`ver_03_sr_t2p`) already does it better with real motion. The
  recommendation would then be the dictionary route plus text-to-gloss, not fine-tuning.
- W-h2s is reported but is not a stop criterion for the pilot: with only the dev set, How2Sign can't teach new words.
  Testing it needs the full run with the 7.9 GB train LMDB, after the pilot passes.

**After a successful pilot (needs your approval):**
- Download the How2Sign train LMDB (7.9 GB) and train both streams. About 2.4 min per epoch per stream; 30 epochs ≈ 2.5 h.
- Evaluate W-h2s and the full §2.1.

---

## 4. Files

| File | What |
|---|---|
| `scripts/vocab_overlap.py` → `outputs/vocab_overlap.json` | 3D-LEX / WLASL / How2Sign vocabulary overlap |
| `scripts/lex_convert_all.py` → `data/3dlex/<gloss>/` | all 735 animated 3D-LEX signs, SignSparK streams (imports `ver_03_sr_t2p/lex3d_convert.py`) |
| `scripts/word_metric.py` → `outputs/word_metric_calibration.json`, `lex_bank_feats.npz` | word-level metric; WLASL → 3D-LEX calibration |
| `scripts/word_metric_variants.py` → `outputs/word_metric_variants.json` | trimming / feature variants |
| `scripts/bench_a_signspark_xattn.py` → `outputs/bench_a_{hand,body}.json` | A: adapter prototype, identity check, cost |
| `scripts/bench_b_madis.py` → `outputs/bench_b_madis.json`, `vq_roundtrip.npz` | B/C: MaDiS SFT step cost (fixed `ft_benchmark.py`), 3D-LEX tokenisation |
| `scripts/vq_eval.py` → `outputs/vq_eval*.json` | VQ round-trip errors and identity retrieval |
| `scripts/vq_debug.py`, `vq_debug2.py` | §1.4 first released-token check (test clips only; superseded by ver_03_madis's full check) |
| `scripts/vq_collarfix.py` → `outputs/vq_roundtrip_collarfix.npz` | collar re-split test |
| `scripts/domain_gap.py` → `outputs/domain_gap.json` | data distance to How2Sign |
| `scripts/run_bench.sh`, `logs/` | benchmark runner (waited for the GPU to be free) and logs |

Venvs: `../ver-01/.venv-signspark` (SignSparK side), `../ver_03_madis/.venv-madis` (MaDiS side). Sizes: `data/` 74 MB, `outputs/` 167 MB (mostly the two VQ round-trip npz files, regenerable).
