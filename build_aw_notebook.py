"""
Builds AMLDL_Assignment2_AdventureWorks.ipynb from aw_pipeline.py.

Generated from the pipeline source so the two cannot drift apart: every
'# ====' banner in the script becomes a markdown heading plus a code cell.
"""

import re
import nbformat as nbf

SRC = "aw_pipeline.py"
OUT = "AMLDL_Assignment2_AdventureWorks.ipynb"

NOTES = {
    "STEP 1": (
        "### 1. Dataset preparation\n\n"
        "Three files are joined on `CustomerID`: demographics, the `BikeBuyer` "
        "target, and average monthly spend.\n\n"
        "**Deduplication needs care.** The raw extract repeats some CustomerIDs, "
        "and a blanket `drop_duplicates()` would be wrong because three distinct "
        "cases exist:\n\n"
        "| Case | Count | Treatment |\n"
        "|---|---|---|\n"
        "| Exact duplicate rows | 98 | safe to drop |\n"
        "| Same ID, conflicting attributes | 17 | keep the last record |\n"
        "| Same ID, conflicting target | 4 | keep the last record |\n\n"
        "\"Keep last\" is applied consistently across all three tables so the "
        "customer, target and spend records stay aligned.\n\n"
        "**Other preparation steps:**\n\n"
        "- **PII and identifiers dropped** - names, phone, street address carry no "
        "generalisable signal; they are unique labels, not features.\n"
        "- **Whitespace stripped** - `'Bachelors '` ships with a trailing space and "
        "would otherwise encode as a separate level from `'Bachelors'`.\n"
        "- **Age derived against a fixed 1998 reference date.** AdventureWorks is a "
        "1998-vintage sample database; using today's date would add ~28 years to "
        "every customer and destroy the age signal.\n"
        "- **`Education` is ordinal**, so it is rank-encoded rather than one-hot - "
        "this preserves the ordering for the linear models and gives the trees a "
        "single splittable column."
    ),
    "STEP 2": (
        "### 2. Feature selection\n\n"
        "> **`AveMonthSpend` is deliberately excluded.**\n\n"
        "It records what a customer spends with Adventure Works, so it is measured "
        "contemporaneously with - and is partly *caused by* - the bike purchase "
        "itself. Including it would leak the outcome into the features, and it is "
        "unavailable at scoring time for a prospect who has not bought anything "
        "yet.\n\n"
        "Three screens are then applied: a zero-variance filter, an `|r| > 0.95` "
        "redundancy filter, and a mutual-information ranking that captures "
        "non-linear dependence on the target."
    ),
    "STEP 3": (
        "### 3. Stratified train / test split\n\n"
        "This dataset is a **customer cross-section, not a time series**, so a "
        "random split is the correct choice here - there is no temporal ordering to "
        "violate. The split is **stratified** so that both partitions carry the "
        "same 33% buyer proportion, which matters when the classes are imbalanced."
    ),
    "STEP 4": (
        "### 4. Model training and evaluation metrics\n\n"
        "**Why accuracy alone is the wrong metric.** 66.8% of customers do not buy, "
        "so a model that predicts \"nobody buys\" scores 66.8% accuracy while being "
        "completely useless - its F1 is 0.000 and its ROC-AUC is 0.500. Both "
        "baselines are scored explicitly to make this concrete.\n\n"
        "| Metric | What it tells us |\n"
        "|---|---|\n"
        "| Accuracy | overall hit rate - reported, but never used alone |\n"
        "| Balanced accuracy | mean recall across both classes |\n"
        "| Precision | of those we mail, what share actually buy - drives campaign waste |\n"
        "| Recall | of all buyers, what share we reach - drives missed revenue |\n"
        "| **F1** | harmonic mean of the two |\n"
        "| **ROC-AUC** | quality of the *ranking*, independent of any cut-off |\n"
        "| **PR-AUC** | ranking quality focused on the minority class |"
    ),
    "STEP 5": (
        "### 5. Hyperparameter tuning\n\n"
        "- **`RandomizedSearchCV`** - 25 sampled configurations per model, far more "
        "efficient than an exhaustive grid over this space.\n"
        "- **Validation curves** - sweep one parameter at a time, plotting training "
        "against cross-validation AUC so the onset of over-fitting is visible.\n\n"
        "**`StratifiedKFold`** preserves the 33/67 class ratio in every fold.\n\n"
        "> **Why tune on ROC-AUC rather than accuracy?** The deliverable is a "
        "*ranking* of customers by propensity, which the campaign then cuts at "
        "whatever depth the budget allows. AUC scores that ranking directly and is "
        "independent of any single threshold."
    ),
    "STEP 6": (
        "### 6. Dimensionality reduction (PCA)\n\n"
        "PCA is fitted on the standardised training matrix. The scree and cumulative "
        "variance plots show how many components retain 90% and 95% of the variance, "
        "and a two-component projection shows how separable the classes are in the "
        "reduced space. Logistic Regression and Random Forest are then retrained on "
        "the components, so PCA is kept or rejected on measured AUC rather than "
        "assumption."
    ),
    "STEP 7": (
        "### 7. Ensemble techniques\n\n"
        "| Family | Model | Mechanism |\n"
        "|---|---|---|\n"
        "| Bagging | Random Forest | parallel trees on bootstrap samples |\n"
        "| Boosting | Gradient Boosting / HistGradientBoosting | sequential trees, each correcting the last |\n"
        "| Voting | `VotingClassifier` (soft) | averages predicted *probabilities*, preserving ranking information |\n"
        "| Stacking | `StackingClassifier` | a logistic meta-model learns how to weight the base models |\n\n"
        "> Note the contrast with a time-series problem: `StratifiedKFold` **does** "
        "partition the data, so scikit-learn's `StackingClassifier` can build its "
        "meta-features directly via `cross_val_predict`. A `TimeSeriesSplit` cannot, "
        "and would require a hand-built temporal stack."
    ),
    "STEP 8": (
        "### 8. Model comparison and final selection\n\n"
        "Every model is ranked on hold-out ROC-AUC. Selection is justified on "
        "ranking quality first, then on F1, calibration and operational cost."
    ),
    "STEP 9": (
        "### 9. Diagnostics and threshold optimisation\n\n"
        "Four diagnostics: the confusion matrix, the ROC curve, the "
        "precision-recall curve (more informative than ROC under imbalance), and a "
        "**calibration curve** - which matters because the campaign uses predicted "
        "*probabilities*, not just labels, so those probabilities need to mean what "
        "they say.\n\n"
        "> **The 0.50 cut-off is an arbitrary default, not a business decision.** "
        "Two alternatives are evaluated: the F1-optimal threshold, and the threshold "
        "that maximises expected campaign profit."
    ),
    "STEP 10": (
        "### 10. Business interpretation\n\n"
        "The model output is converted into a campaign decision through a **decile "
        "lift and gains analysis**: customers are ranked by predicted propensity, "
        "split into ten equal groups, and the buy rate in each is compared with the "
        "base rate.\n\n"
        "A **sensitivity analysis** then tests how the value of targeting changes "
        "with campaign economics - because whether targeting pays turns out to be a "
        "property of the cost-to-margin ratio, not of the model."
    ),
}

HEADER_MD = """# AMLDL (MBA ZG582) - Assignment 2
## Phase 2: Machine Learning Phase

**Project:** Bike-Buyer Propensity Modelling for Adventure Works Cycles
**Dataset:** Microsoft AdventureWorks customer data (EdX DAT275x lab extract)
**Files:** `AdvWorksCusts.csv` + `AW_BikeBuyer.csv` + `AW_AveMonthSpend.csv`
**Scope:** 16,404 unique customers across 6 countries, 23 modelling features

---

### Problem statement

Adventure Works Cycles runs direct marketing campaigns to sell bicycles. Mailing
the entire customer base wastes budget on people who will never buy; mailing too
narrowly leaves revenue on the table. The business needs to know **which
customers to contact**.

**ML framing - supervised binary classification.** Predict `BikeBuyer`, whether a
customer has purchased a bike, from demographic and household attributes. The
model's real output is not a yes/no label but a **ranked propensity score**, which
the campaign cuts at whatever depth the budget allows.

| | |
|---|---|
| **Target** | `BikeBuyer` - binary, 33.25% positive |
| **Unit of analysis** | one customer |
| **Class balance** | 1 : 2.01 (moderate imbalance) |
| **Primary metric** | ROC-AUC - scores the ranking, independent of cut-off |
| **Secondary metrics** | F1, PR-AUC, precision, recall, balanced accuracy |
| **Baselines** | majority-class and stratified-random classifiers |

---

### Assignment-2 task coverage

| # | Task from the brief | Where |
|---|---|---|
| 1 | Prepare dataset (feature selection, encoding) | Steps 1-2 |
| 2 | Train selected models | Step 4 |
| 3 | Evaluate with appropriate metrics | Steps 4, 8, 9 |
| 4 | Hyperparameter tuning (random search / validation curves) | Step 5 |
| 5 | Compare performance, justify final selection | Step 8 |
| 6 | Dimensionality reduction (PCA) and ensembles | Steps 6-7 |
| 7 | Methodology, results, plots, business interpretation | throughout, Step 10 |

---

### Why classification, and why this dataset

The business decision is a **contact / do-not-contact** choice over a fixed
customer list, which is a classification problem by construction. The dataset
also carries genuine preparation work - duplicate records with conflicting
values, a whitespace defect in a category level, an ordinal variable that should
not be one-hot encoded, and a leakage trap in `AveMonthSpend` - so Step 1 is real
analysis rather than a formality.
"""

COLAB_SETUP = '''# --- Colab / local data setup -----------------------------------------------
# Downloads the three AdventureWorks CSVs if they are not already present.
import os
import urllib.request

DATA_DIR = os.path.join("data", "adventureworks")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs("outputs/figures", exist_ok=True)
os.makedirs("outputs/tables", exist_ok=True)

BASE = "https://raw.githubusercontent.com/kdavenpo/AdventureWorks/master/"
for fn in ["AdvWorksCusts.csv", "AW_BikeBuyer.csv", "AW_AveMonthSpend.csv"]:
    dest = os.path.join(DATA_DIR, fn)
    if not os.path.exists(dest):
        print("downloading", fn)
        urllib.request.urlretrieve(BASE + fn, dest)
    print(f"{fn:24s} ready ({os.path.getsize(dest):,} bytes)")'''


def split_sections(source: str):
    lines = source.splitlines()
    marks = []
    for i, ln in enumerate(lines):
        if (ln.startswith("# ====") and i + 2 < len(lines)
                and lines[i + 1].startswith("# STEP")
                and lines[i + 2].startswith("# ====")):
            marks.append((i, lines[i + 1][2:].strip()))

    blocks = [("PREAMBLE", "Setup, imports and configuration",
               "\n".join(lines[:marks[0][0]]).rstrip())]
    for idx, (start, title) in enumerate(marks):
        end = marks[idx + 1][0] if idx + 1 < len(marks) else len(lines)
        key = re.match(r"(STEP \d+)", title)
        blocks.append((key.group(1) if key else title, title,
                       "\n".join(lines[start:end]).rstrip()))
    return blocks


def main():
    with open(SRC, encoding="utf-8") as fh:
        source = fh.read()

    nb = nbf.v4.new_notebook()
    cells = [nbf.v4.new_markdown_cell(HEADER_MD),
             nbf.v4.new_markdown_cell(
                 "---\n## Environment setup\n\n"
                 "Run this first - it fetches the three CSVs straight from the "
                 "public GitHub mirror, so the notebook is self-contained on "
                 "Colab."),
             nbf.v4.new_code_cell(COLAB_SETUP)]

    for key, title, code in split_sections(source):
        if key == "PREAMBLE":
            cells.append(nbf.v4.new_markdown_cell(
                "---\n### 0. Imports, configuration and helpers\n\n"
                "A fixed seed (`SEED = 42`) makes every result reproducible. The "
                "campaign economics (`MARGIN_PER_SALE`, `COST_PER_CONTACT`) are "
                "**stated assumptions**, not values derived from the data."))
        else:
            cells.append(nbf.v4.new_markdown_cell(
                "---\n" + NOTES.get(key, "### " + title)))
        cells.append(nbf.v4.new_code_cell(code))

    cells.append(nbf.v4.new_markdown_cell(
        "---\n## Outputs\n\n"
        "Figures are written to `outputs/figures/` and result tables to "
        "`outputs/tables/`. `aw_t15_run_summary.json` holds the headline "
        "numbers quoted in the progress report."))

    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python", "version": "3.13"},
        "colab": {"provenance": [], "toc_visible": True},
    }

    with open(OUT, "w", encoding="utf-8") as fh:
        nbf.write(nb, fh)
    print("Wrote " + OUT + " with " + str(len(cells)) + " cells")


if __name__ == "__main__":
    main()
