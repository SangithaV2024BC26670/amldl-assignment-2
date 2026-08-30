# Bike-Buyer Propensity Modelling — Adventure Works Cycles

**Course:** MBA ZG582 — Applied Machine Learning and Deep Learning
**Problem type:** Supervised binary classification
**Dataset:** Microsoft AdventureWorks customer data (EdX DAT275x extract) — 16,404 unique customers, 23 features
**Target:** `BikeBuyer` — 33.25% positive

---

## Deliverables

| File | What it is |
|---|---|
| `Concept_Note_AdventureWorks_Bike_Buyer_Propensity.docx` | Phase 1 concept note — problem statement, dataset, EDA, methodology |
| `AMLDL_Assignment2_AdventureWorks_Progress_Report.docx` | Machine-learning phase report — 15 tables, 16 figures |
| `AMLDL_DeepLearning_Report.docx` | Deep-learning report — entity-embedding network vs the benchmark |
| `AMLDL_Assignment2_AdventureWorks.ipynb` | Colab-ready notebook, 26 cells — downloads its own data |

## Scripts

| Script | Purpose |
|---|---|
| `aw_pipeline.py` | Main ML pipeline — 10 steps, writes to `outputs/` |
| `aw_deep_learning.py` | Entity-embedding neural network (PyTorch), writes to `outputs_dl/` |
| `build_aw_notebook.py` | Generates the notebook from `aw_pipeline.py` |
| `build_aw_phase1.py` | Generates the concept note |
| `build_aw_report.py` | Generates the ML progress report |
| `build_aw_dl_report.py` | Generates the deep-learning report |

Every `build_*` script reads its numbers from `outputs/tables/` at build time, so a
document can never quote a stale result.

## Layout

```
data/adventureworks/    3 source CSVs
outputs/                ML pipeline — 16 figures, 15 tables, run_log.txt
outputs_dl/             Deep learning — 3 figures, 6 tables, saved model, run_log.txt
```

## Reproducing

Run from **inside this folder** — all paths are relative to it.

```bash
python aw_pipeline.py && python aw_deep_learning.py
```

Then regenerate the documents:

```bash
python build_aw_phase1.py && python build_aw_report.py && python build_aw_dl_report.py && python build_aw_notebook.py
```

`aw_pipeline.py` takes roughly 7 minutes (hyperparameter search dominates);
`aw_deep_learning.py` about 2 minutes. Seed is fixed at 42 throughout.

## Headline results

| Model | ROC-AUC | F1 | Accuracy |
|---|---|---|---|
| **HistGradientBoosting (tuned)** | **0.8606** | 0.6703 | 0.7998 |
| Entity-embedding neural network | 0.8548 | 0.6887 | 0.7729 |
| Majority-class baseline | 0.5000 | 0.0000 | 0.6675 |

Gradient boosting is the recommended production model. Top propensity decile buys
at 2.78× the base rate; the top 30% of the ranked list captures 64.6% of all buyers.

**Two findings reported as measured, not as hoped:**

- PCA does not help — 15 components for 95% variance, every variant scores below
  its full-feature equivalent. Roughly half the matrix is orthogonal dummies.
- Targeting adds only 0.53% profit at the assumed economics (margin 120, contact
  cost 2), because break-even sits at p > 1.67% and 92.5% of customers clear it.
  A sensitivity sweep shows targeting becomes decisive as contact cost rises.

## Before submitting

- Paste your Colab / Git link into Appendix A of each document.
- Add your name and student ID to the concept note title block.
- The campaign economics (margin 120, cost 2) are **stated assumptions**, not
  values derived from the data.
