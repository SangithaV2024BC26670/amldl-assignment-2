# Bike-Buyer Propensity Modelling — Adventure Works Cycles

**Course:** MBA ZG582 — Applied Machine Learning and Deep Learning
**Assignment:** 2 — Machine Learning Phase
**Problem type:** Supervised binary classification
**Dataset:** Microsoft AdventureWorks customer data (EdX DAT275x extract) — 16,404 unique customers, 23 features
**Target:** `BikeBuyer` — 33.25% positive

---

## Submission

| Name | BITS ID | Contribution |
|---|---|---|
| Sangitha V | 2024BC26670 | 100% |

**Colab notebook:** https://colab.research.google.com/drive/13-BGEgvCadYBMlL7__BtckakJjmhy-rI?usp=sharing

## Contents

| File | What it is |
|---|---|
| `2024BC26670_Sangitha_V_Concept_Note.docx` | Concept note — problem statement, dataset, EDA, methodology |
| `2024BC26670_Sangitha_V.docx` | Progress report — 15 tables, 16 figures |
| `2024BC26670_Sangitha_V_Deep_Learning_Report.docx` | Deep-learning report — entity-embedding network vs the benchmark |
| `2024BC26670_Sangitha_V.ipynb` | Full notebook — all code, executed, with outputs and plots inline |

```
data/adventureworks/    3 source CSVs
outputs/                16 figures, 15 result tables, run log
outputs_dl/             3 figures, 7 tables, trained model, run log
```

## Running the notebook

`2024BC26670_Sangitha_V.ipynb` is self-contained — it downloads the
three source CSVs on first run, so it works in Google Colab with no setup.

Open in Colab and **Runtime → Run all**, or locally:

```bash
jupyter notebook 2024BC26670_Sangitha_V.ipynb
```

It runs the whole pipeline end to end: data preparation, feature selection,
model training, hyperparameter tuning, PCA, ensembles, diagnostics and the
business analysis. Roughly 7 minutes; the randomised hyperparameter search
dominates. Seed fixed at 42, so results reproduce exactly.

The committed notebook already carries every output and plot, so the results can
be read without running anything.

## Method

| Stage | What was done |
|---|---|
| Data preparation | Duplicate resolution (3 distinct conflict types), PII removal, whitespace defect fix, ordinal + one-hot encoding |
| Feature selection | Leakage exclusion, zero-variance filter, \|r\| > 0.95 redundancy filter, mutual information |
| Split | Stratified 80/20, class ratio preserved |
| Models | 2 baselines, 6 candidates, 2 tuned, voting and stacking ensembles |
| Tuning | RandomizedSearchCV (25 draws, StratifiedKFold) + validation curves |
| Dimensionality reduction | PCA, models retrained on components |
| Evaluation | Accuracy, balanced accuracy, precision, recall, F1, ROC-AUC, PR-AUC |
| Business analysis | Threshold optimisation, decile lift and gains, campaign economics, sensitivity |

## Results

| Model | ROC-AUC | F1 | Accuracy |
|---|---|---|---|
| **HistGradientBoosting (tuned)** | **0.8606** | 0.6703 | 0.7998 |
| Stacking ensemble | 0.8601 | 0.6650 | 0.7973 |
| Entity-embedding neural network | 0.8548 | 0.6887 | 0.7729 |
| Majority-class baseline | 0.5000 | 0.0000 | 0.6675 |

Gradient boosting is the recommended model. The top propensity decile buys at
**2.78× the base rate**, and the top 30% of the ranked list captures **64.6%** of
all buyers.

### Findings reported as measured

- **The majority-class baseline scores 0.6675 accuracy with F1 = 0.000.** This is
  why accuracy alone is not used to judge any model here.
- **PCA does not help.** 15 components retain 95% of the variance, but every PCA
  variant scores below its full-feature equivalent — roughly half the matrix is
  orthogonal dummy columns, which PCA cannot compress. Implemented in full and
  reported as measured rather than assumed.
- **Targeting adds only 0.53% profit** at the assumed economics (margin 120,
  contact cost 2), because break-even sits at p > 1.67% and 92.5% of customers
  clear it. A sensitivity sweep shows targeting becomes decisive as contact cost
  rises — at cost 60, it is worth 91,440 more than mailing everyone.
- **The neural network does not beat the boosted trees** on ranking quality
  (0.8548 vs 0.8606), which is the expected outcome on 16k rows with categorical
  fields of at most six levels. Its value is interpretive: the learned embeddings
  recovered the ordinal structure of `Education` unprompted.

## Note on assumptions

The campaign economics used in the business analysis — margin of 120 per sale and
contact cost of 2 — are **stated assumptions**, not values derived from the
dataset. The sensitivity analysis exists precisely because the conclusion depends
on them.
