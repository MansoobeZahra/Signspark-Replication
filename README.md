# SignSparK replication

Replication of **SignSparK** (ECCV 2026, arXiv 2603.10446) for an English speech → ASL → humanoid-robot project.
Goal: check every point of the paper against the released code, reproduce its numbers, retrain with the paper's
settings, and find out why text-only output gives wrong signs.

**Short answer:** keyframe mode (KF2P) reproduces and beats the paper. Text-only mode (T2P) does not produce
recognisable ASL, for the released model or for our retrain with the paper's exact settings. Its paper score sits
at the level of a motionless "standing still" pose. The cause is the text pathway in the method's design, not
the checkpoint, the data or a bug.

Full write-up with every number: [`PAPER_AUDIT.md`](PAPER_AUDIT.md). Bird's-eye reports with images:
[`report/index.html`](report/index.html), [`report/overview.html`](report/overview.html).

## Layout

| Path | What |
|---|---|
| `upstream/` | Authors' code at commit `916402d`, unmodified except one patch (`flow_train_util.py`: gradient accumulation so batch 64 = 2 × 32 gives the full-batch mean gradient on one 32 GB GPU) |
| `train/run.sh`, `train/queue_paper.sh` | Retrain with the paper's config: loss weights flow 1.0 / recon 0.5 / velocity 0.5, no acceleration loss, How2Sign + CSL-Daily + PHOENIX-2014T, 100k steps per stream |
| `eval/evaluate.py`, `eval/soke_metric.py` | Paper Tab. 3/4 protocol (SOKE DTW-JPE / DTW-PA-JPE), plus controls: static mean pose, other sentence, other noise seed; `--baseline slerp` |
| `diagnostics/d1_text_conditioning.py` | How much the sentence actually moves the conditioning vector |
| `judge/` | Sign-level check: WLASL isolated-sign recogniser (124 signs) + sign spotting in generated sentences, CPU only |
| `dictionary/` | Prototype real-sign dictionary (WLASL / ASL Citizen → NLF + WiLoR → SMPL-X) and SLERP stitching |
| `report/` | HTML reports and figures |

Not in git (see `.gitignore`): data (HF `LionelLow/SignSparK_data` @9fe7e04), checkpoints (HF `LionelLow/SignSparK`
@b907f25 and our 2 × 167 GB retrain runs), `smplx/` model files, outputs.

## Results

Metric: SOKE DTW-JPE (summed metres along the DTW path, lower is better), 200 random test clips per dataset, same
clips and seeds for every row. "Static" = the training set's mean pose held for the whole clip, no signing at all.

### Keyframe mode (KF2P, Tab. 4): reproduced, better than the paper

| Body / hand DTW-JPE | Paper | Released | Our retrain |
|---|---|---|---|
| How2Sign | 1.68 / 2.80 | **1.21 / 2.10** | 1.35 / 2.21 |
| CSL-Daily | 2.02 / 2.71 | **1.01 / 1.65** | 1.15 / 1.70 |
| PHOENIX-2014T | 1.92 / 2.25 | 1.23 / 1.94 | **1.23 / 1.82** |

Plain SLERP between the same keyframes scores almost as well (How2Sign body 1.24, hands 2.39). The paper's SLERP
baseline is 2.5-3.5× worse than a correct SLERP.

### Text-only mode (T2P, Tab. 3): not reproduced, and loses to standing still

| Body DTW-JPE | How2Sign | CSL-Daily | PHOENIX-2014T |
|---|---|---|---|
| Paper | 6.30 | 6.27 | 4.40 |
| **Static pose** | **7.04** | **6.94** | **4.50** |
| Released, 1 / 10 / 50 steps | 7.04 / 8.71 / 9.26 | 6.82 / 8.14 / 8.52 | 5.07 / 5.84 / 6.16 |
| Our retrain, 1 / 10 / 50 steps | 7.17 / 8.73 / 9.21 | 6.69 / 8.05 / 8.36 | 4.41 / 5.34 / 5.60 |

- Our retrain matches the released model (How2Sign 8.73 vs 8.71 at 10 steps).
- The paper's numbers are reached only with 1 sampling step, where the output stays near the mean pose
  (PHOENIX 4.41 vs paper 4.40). More steps give more motion and a worse score.

### Does the sentence control the output?

Swap the sentence and keep the noise, vs keep the sentence and swap the noise (mean per-frame rot6D change):

| Text / noise ratio | How2Sign | CSL-Daily | PHOENIX |
|---|---|---|---|
| Released | 0.16 | 0.20 | 0.17 |
| Our retrain | 0.12 | 0.18 | 0.22 |

The output is decided mostly by the random noise, not the words.

### Are there real ASL signs in the output? (`judge/`)

WLASL recogniser: 56.7% top-1 / 82.1% top-5 on held-out signers (124 signs, chance 0.8%). Sign-spotting AUC
(0.5 = no trace of the word's sign) on words it can find in real signing:

| Source | AUC | p |
|---|---|---|
| Real How2Sign signing | 0.655 | 0.0005 |
| SignSparK text-only | 0.507 | 0.36 (no signal) |
| Real signing, SOKE fittings | 0.637 | |
| MaDiS (released) | 0.569 | 0.0015 |

## Diagnosis

1. **The sentence barely reaches the model.** It enters as one mean-pooled M-CLIP vector added to the timestep
   embedding; no word-level attention.
2. **Text-only is rarely trained.** Keyframes are dropped with p = 0.1, so only ~9% of samples are text-only.
3. **The benchmark cannot see the failure.** DTW-JPE scores a motionless pose better than wrong signing; the paper
   reports no static baseline and no back-translation for T2P.
4. **Paper vs code differences** (loss weights 2:1:1 vs 1:2:1, an extra acceleration loss, 15× more data in the
   release) do not explain it: the retrain with the paper's settings behaves the same.

SignSparK is a strong keyframe in-betweener, but it cannot turn English into correct ASL signs as published.
The project therefore moved to a real-sign dictionary (gloss → recorded signs → smooth joins).

## Earlier SignSparK work in this repo (separate folders, not part of the replication above)

These folders were written before this replication, in other project folders, and are copied in unchanged
(code, docs, logs and small JSON results only). Each keeps its own report; none of them overwrites or is
overwritten by the replication files.

| Folder | What it is | Main finding |
|---|---|---|
| [`model_tests/`](model_tests/) | Official SignSparK evaluation with the authors' `sample.py` on all 2,183 How2Sign test clips (the MaDiS half of this work is in the MaDiS repo). Shared files: `REPORT.md`, `official_metric.py`, `render_side_by_side.py`, `extract_nsa_test.py` | Text-only body DTW-JPE 8.75 vs paper 6.30; standing still scores 6.41. Keyframe mode 1.18 / 3.03. Outputs cannot be matched to their sentence above chance |
| [`ver_03_gf_t2p/`](ver_03_gf_t2p/) | Text-only (GF-T2P) tests: SOKE's DTW metric ported and verified against SOKE's code (to 1e-15), then two attempts to make text-only SignSparK sign the sentence: best-of-N seed selection by back-translation, and generation-time "residuals" (negative-prompt guidance, per-chunk conditioning, noise temperature). See `REPORT.md` | Neither works. Best-of-N raises back-translation almost as much with the *wrong* sentence; 200-way sentence retrieval stays ≤ 1% (chance 0.5%). `eval/soke_metric.py` in the replication comes from here |
| [`ver_04_finetune/`](ver_04_finetune/) | Feasibility study (`PLAN.md`), not a training run: cost and blockers of (A) SignSparK + token cross-attention adapters, (B) MaDiS full fine-tune, (C) adding 3D-LEX / WLASL word pairs. GPU benchmarks, word-metric calibration, MaDiS tokenizer round-trip | Nothing was trained, so nothing failed. A was feasible and cheap (33.6 M trainable params, 2.3 min/epoch); MaDiS's tokenizer cannot carry 3D-LEX signs (wrist error 258 mm vs 64 mm on How2Sign). The pilot was dropped when the project moved to a real-sign dictionary |

## Licence

`upstream/` is the SignSparK authors' code under Apache 2.0 (`upstream/LICENSE`).
