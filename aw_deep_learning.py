"""
=============================================================================
AMLDL (MBA ZG582) - DEEP LEARNING PHASE
Project: Bike-Buyer Propensity Modelling for Adventure Works Cycles
Model:   Entity-embedding neural network (PyTorch)
=============================================================================
The benchmark for this project is a tuned gradient-boosting classifier at
ROC-AUC 0.8606. This script asks whether a neural network with learned entity
embeddings can beat it, and - just as important - what the embeddings reveal
that a tree ensemble cannot express.

Why entity embeddings for this dataset:
  One-hot encoding forces every categorical level to be equidistant from every
  other, so the model cannot know that 'Professional' and 'Management' are
  more alike than 'Professional' and 'Manual'. An embedding layer instead
  learns a dense vector per level, positioning similar levels near each other
  in the embedding space. Those vectors are inspectable, which turns the
  network from a black box into something a marketing team can read.

Framework note: the concept note nominated TensorFlow/Keras. PyTorch is used
instead because it is the framework available in this environment; the
architecture and training procedure are unchanged by that choice.
=============================================================================
"""

import os
import json
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score,
                             roc_curve, confusion_matrix)

warnings.filterwarnings("ignore")

SEED = 42
DATA_DIR = os.path.join("data", "adventureworks")
FIG_DIR = os.path.join("outputs_dl", "figures")
TAB_DIR = os.path.join("outputs_dl", "tables")
BM_TAB = os.path.join("outputs", "tables")
TARGET = "BikeBuyer"
AGE_REFERENCE = pd.Timestamp("1998-01-01")
EDUCATION_ORDER = ["Partial High School", "High School", "Partial College",
                   "Bachelors", "Graduate Degree"]

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 150,
                     "savefig.bbox": "tight", "font.size": 10})

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def banner(txt):
    print("\n" + "=" * 78)
    print(txt)
    print("=" * 78)


def save_fig(name):
    path = os.path.join(FIG_DIR, name)
    plt.savefig(path)
    plt.close()
    print("   [figure saved] " + path)


def save_table(df, name, index=False):
    path = os.path.join(TAB_DIR, name)
    df.to_csv(path, index=index)
    print("   [table saved ] " + path)


# ============================================================================
# STEP 1 - DATA PREPARATION (mirrors the benchmark pipeline)
# ============================================================================
banner("STEP 1 | DATA PREPARATION")


def dedupe(frame):
    return (frame.drop_duplicates()
            .drop_duplicates(subset="CustomerID", keep="last"))


custs = dedupe(pd.read_csv(os.path.join(DATA_DIR, "AdvWorksCusts.csv")))
buyer = dedupe(pd.read_csv(os.path.join(DATA_DIR, "AW_BikeBuyer.csv")))
df = custs.merge(buyer, on="CustomerID", how="inner")

DROP_PII = ["Title", "FirstName", "MiddleName", "LastName", "Suffix",
            "AddressLine1", "AddressLine2", "PhoneNumber", "City",
            "PostalCode", "StateProvinceName"]
df = df.drop(columns=DROP_PII)
for col in ["Education", "Occupation", "Gender", "MaritalStatus",
            "CountryRegionName"]:
    df[col] = df[col].astype(str).str.strip()

df["BirthDate"] = pd.to_datetime(df["BirthDate"])
df["Age"] = ((AGE_REFERENCE - df["BirthDate"]).dt.days / 365.25).round(1)
df["ChildrenAwayFromHome"] = (df["TotalChildren"]
                              - df["NumberChildrenAtHome"]).clip(lower=0)
df["IncomePerChild"] = df["YearlyIncome"] / (df["TotalChildren"] + 1)
df["CarsPerChild"] = df["NumberCarsOwned"] / (df["TotalChildren"] + 1)
df["LogIncome"] = np.log1p(df["YearlyIncome"])

# Guard against silent drift from the benchmark preparation
assert len(df) == 16404, "row count diverged from benchmark (expected 16404)"
print("Customers after dedup : " + str(len(df)))
print("Buy rate              : " + format(df[TARGET].mean() * 100, ".2f") + "%")

# --- 1.1 Split features into categorical (embedded) and continuous ----------
# Every low-cardinality field becomes an embedding, including the integer
# count columns: the network can then learn that "3 children" sits between
# "2" and "4" without being forced into a linear assumption.
CAT_COLS = ["Occupation", "Gender", "MaritalStatus", "CountryRegionName",
            "Education", "HomeOwnerFlag", "NumberCarsOwned",
            "NumberChildrenAtHome", "TotalChildren"]
CONT_COLS = ["Age", "YearlyIncome", "LogIncome", "IncomePerChild",
             "CarsPerChild", "ChildrenAwayFromHome"]

# Education keeps its natural ordering in the index so that, if the network
# learns an ordinal structure, it is visible in the embedding plot.
cat_levels, cat_maps = {}, {}
for col in CAT_COLS:
    levels = (EDUCATION_ORDER if col == "Education"
              else sorted(df[col].unique(), key=lambda v: (str(type(v)), v)))
    cat_maps[col] = {lvl: i for i, lvl in enumerate(levels)}
    cat_levels[col] = levels
    df[col + "_idx"] = df[col].map(cat_maps[col])
    assert df[col + "_idx"].notna().all(), "unmapped level in " + col

emb_spec = []
for col in CAT_COLS:
    card = len(cat_levels[col])
    dim = int(min(50, (card + 1) // 2))       # standard heuristic
    emb_spec.append((col, card, dim))
    print("  " + col.ljust(22) + " cardinality " + str(card).rjust(2)
          + " -> embedding dim " + str(dim))

print("Continuous features   : " + str(len(CONT_COLS)))
print("Total embedding params: "
      + str(sum(c * d for _, c, d in emb_spec)))

# --- 1.2 Identical split to the benchmark -----------------------------------------
X_cat = df[[c + "_idx" for c in CAT_COLS]].values.astype(np.int64)
X_con = df[CONT_COLS].values.astype(np.float32)
y = df[TARGET].values.astype(np.float32)

idx = np.arange(len(df))
idx_train, idx_test = train_test_split(idx, test_size=0.20, random_state=SEED,
                                       stratify=y)
# Carve a validation slice out of TRAIN only, for early stopping. The test set
# is never seen during training or model selection.
idx_tr, idx_val = train_test_split(idx_train, test_size=0.15,
                                   random_state=SEED, stratify=y[idx_train])

scaler = StandardScaler().fit(X_con[idx_tr])
X_con_s = scaler.transform(X_con).astype(np.float32)

print("\nTrain / val / test    : " + str(len(idx_tr)) + " / " + str(len(idx_val))
      + " / " + str(len(idx_test)))
print("Buy rate per split    : " + format(y[idx_tr].mean(), ".4f") + " / "
      + format(y[idx_val].mean(), ".4f") + " / " + format(y[idx_test].mean(), ".4f"))
print("Device                : " + str(DEVICE))


def make_loader(indices, batch=256, shuffle=False):
    ds = TensorDataset(torch.from_numpy(X_cat[indices]),
                       torch.from_numpy(X_con_s[indices]),
                       torch.from_numpy(y[indices]))
    return DataLoader(ds, batch_size=batch, shuffle=shuffle)


train_loader = make_loader(idx_tr, shuffle=True)
val_loader = make_loader(idx_val)
test_loader = make_loader(idx_test)


# ============================================================================
# STEP 2 - MODEL ARCHITECTURE
# ============================================================================
banner("STEP 2 | ENTITY-EMBEDDING NETWORK ARCHITECTURE")


class EntityEmbeddingNet(nn.Module):
    def __init__(self, emb_spec, n_cont, hidden=(128, 64), dropout=(0.35, 0.20)):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(card, dim) for _, card, dim in emb_spec])
        emb_total = sum(dim for _, _, dim in emb_spec)
        self.emb_drop = nn.Dropout(0.05)
        self.cont_bn = nn.BatchNorm1d(n_cont)

        layers, in_dim = [], emb_total + n_cont
        for h, p in zip(hidden, dropout):
            layers += [nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.ReLU(),
                       nn.Dropout(p)]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x_cat, x_cont):
        emb = [e(x_cat[:, i]) for i, e in enumerate(self.embeddings)]
        x = torch.cat(emb, dim=1)
        x = self.emb_drop(x)
        x = torch.cat([x, self.cont_bn(x_cont)], dim=1)
        return self.mlp(x).squeeze(1)


model = EntityEmbeddingNet(emb_spec, len(CONT_COLS)).to(DEVICE)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(model)
print("\nTrainable parameters  : " + format(n_params, ","))


# ============================================================================
# STEP 3 - TRAINING
# ============================================================================
banner("STEP 3 | TRAINING")

EPOCHS, PATIENCE = 120, 15
# pos_weight rebalances the 1:2 class ratio inside the loss itself, which is
# the neural-network equivalent of class_weight='balanced'.
pos_weight = torch.tensor([(y[idx_tr] == 0).sum() / (y[idx_tr] == 1).sum()],
                          device=DEVICE)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimiser = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimiser, mode="max", factor=0.5, patience=5)


def epoch_scores(loader):
    model.eval()
    ps, ys, losses = [], [], []
    with torch.no_grad():
        for xc, xn, yy in loader:
            xc, xn, yy = xc.to(DEVICE), xn.to(DEVICE), yy.to(DEVICE)
            logit = model(xc, xn)
            losses.append(criterion(logit, yy).item() * len(yy))
            ps.append(torch.sigmoid(logit).cpu().numpy())
            ys.append(yy.cpu().numpy())
    p, t = np.concatenate(ps), np.concatenate(ys)
    return float(np.sum(losses) / len(t)), roc_auc_score(t, p), p, t


history = []
best_auc, best_state, best_epoch, stale = -1.0, None, 0, 0
t0 = time.time()

for epoch in range(1, EPOCHS + 1):
    model.train()
    running = 0.0
    for xc, xn, yy in train_loader:
        xc, xn, yy = xc.to(DEVICE), xn.to(DEVICE), yy.to(DEVICE)
        optimiser.zero_grad()
        loss = criterion(model(xc, xn), yy)
        loss.backward()
        optimiser.step()
        running += loss.item() * len(yy)
    tr_loss = running / len(idx_tr)
    _, tr_auc, _, _ = epoch_scores(train_loader)
    va_loss, va_auc, _, _ = epoch_scores(val_loader)
    scheduler.step(va_auc)
    history.append({"Epoch": epoch, "TrainLoss": tr_loss, "ValLoss": va_loss,
                    "TrainAUC": tr_auc, "ValAUC": va_auc,
                    "LR": optimiser.param_groups[0]["lr"]})

    if va_auc > best_auc:
        best_auc, best_epoch, stale = va_auc, epoch, 0
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    else:
        stale += 1

    if epoch % 10 == 0 or epoch == 1:
        print("  epoch " + str(epoch).rjust(3)
              + " | train loss " + format(tr_loss, ".4f")
              + " | val loss " + format(va_loss, ".4f")
              + " | train AUC " + format(tr_auc, ".4f")
              + " | val AUC " + format(va_auc, ".4f"))
    if stale >= PATIENCE:
        print("  early stopping at epoch " + str(epoch)
              + " (no val-AUC gain for " + str(PATIENCE) + " epochs)")
        break

model.load_state_dict(best_state)
print("\nTrained in " + format(time.time() - t0, ".1f") + "s; best epoch "
      + str(best_epoch) + " with val AUC " + format(best_auc, ".4f"))

hist = pd.DataFrame(history)
save_table(hist.round(5), "dl_t01_training_history.csv")

fig, axes = plt.subplots(1, 2, figsize=(13, 4.3))
axes[0].plot(hist["Epoch"], hist["TrainLoss"], label="Train")
axes[0].plot(hist["Epoch"], hist["ValLoss"], label="Validation")
axes[0].axvline(best_epoch, ls=":", c="red", label="best epoch " + str(best_epoch))
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("BCE loss")
axes[0].set_title("Learning curve - loss"); axes[0].legend(fontsize=8)
axes[1].plot(hist["Epoch"], hist["TrainAUC"], label="Train")
axes[1].plot(hist["Epoch"], hist["ValAUC"], label="Validation")
axes[1].axvline(best_epoch, ls=":", c="red", label="best epoch " + str(best_epoch))
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("ROC-AUC")
axes[1].set_title("Learning curve - ROC-AUC"); axes[1].legend(fontsize=8)
plt.tight_layout()
save_fig("dl_fig01_learning_curves.png")


# ============================================================================
# STEP 4 - EVALUATION AND COMPARISON WITH THE BENCHMARK
# ============================================================================
banner("STEP 4 | EVALUATION ON THE HELD-OUT TEST SET")

_, test_auc, proba, y_true = epoch_scores(test_loader)
pred = (proba >= 0.5).astype(int)

dl_metrics = {
    "Accuracy": accuracy_score(y_true, pred),
    "BalancedAcc": balanced_accuracy_score(y_true, pred),
    "Precision": precision_score(y_true, pred, zero_division=0),
    "Recall": recall_score(y_true, pred, zero_division=0),
    "F1": f1_score(y_true, pred, zero_division=0),
    "ROC_AUC": roc_auc_score(y_true, proba),
    "PR_AUC": average_precision_score(y_true, proba),
}
for k, v in dl_metrics.items():
    print("  " + k.ljust(12) + format(v, ".4f"))

# --- 4.1 Head-to-head against the benchmark ---------------------------------------
bm = pd.read_csv(os.path.join(BM_TAB, "aw_t08_final_model_comparison.csv"))
bm_summary = json.load(open(os.path.join(BM_TAB, "aw_t15_run_summary.json")))
CHAMP = bm_summary["selected_model"]
champ_row = bm[bm["Model"] == CHAMP].iloc[0]

compare = pd.DataFrame([
    {"Model": CHAMP + " (benchmark)", **{k: round(float(champ_row[k]), 4)
                                       for k in dl_metrics}},
    {"Model": "Entity-Embedding Neural Network",
     **{k: round(v, 4) for k, v in dl_metrics.items()}},
])
compare["Delta_ROC_AUC"] = (compare["ROC_AUC"]
                              - float(champ_row["ROC_AUC"])).round(4)
print("\nHead-to-head on the identical hold-out split:")
print(compare.to_string(index=False))
save_table(compare, "dl_t02_benchmark_vs_network.csv")

delta = dl_metrics["ROC_AUC"] - float(champ_row["ROC_AUC"])
verdict = ("The neural network improves on the benchmark model."
           if delta > 0.002 else
           "The neural network does NOT beat the benchmark model."
           if delta < -0.002 else
           "The two models are statistically indistinguishable on this split.")
print("\nVerdict: " + verdict)
print("  ROC-AUC " + format(float(champ_row["ROC_AUC"]), ".4f") + " (boosting) vs "
      + format(dl_metrics["ROC_AUC"], ".4f") + " (network), delta "
      + format(delta, "+.4f"))

# --- 4.2 ROC overlay ---------------------------------------------------------
bm_scores = pd.read_csv(os.path.join(BM_TAB, "aw_t14_holdout_scores.csv"))
fpr_dl, tpr_dl, _ = roc_curve(y_true, proba)
fpr_p2, tpr_p2, _ = roc_curve(bm_scores["actual"], bm_scores["proba"])

fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
axes[0].plot(fpr_p2, tpr_p2, lw=1.7,
             label=CHAMP + " (AUC " + format(float(champ_row["ROC_AUC"]), ".4f") + ")")
axes[0].plot(fpr_dl, tpr_dl, lw=1.7,
             label="Entity-Embedding NN (AUC " + format(dl_metrics["ROC_AUC"], ".4f") + ")")
axes[0].plot([0, 1], [0, 1], "r--", lw=1)
axes[0].set_xlabel("False positive rate"); axes[0].set_ylabel("True positive rate")
axes[0].set_title("ROC - benchmark vs neural network"); axes[0].legend(loc="lower right", fontsize=8)

cm = confusion_matrix(y_true, pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[1],
            xticklabels=["Pred 0", "Pred 1"], yticklabels=["True 0", "True 1"])
axes[1].set_title("Confusion matrix - network (cut-off 0.50)")

metrics_plot = ["Accuracy", "F1", "ROC_AUC", "PR_AUC"]
xpos = np.arange(len(metrics_plot))
axes[2].bar(xpos - 0.2, [float(champ_row[m]) for m in metrics_plot], width=0.4,
            label="Gradient boosting (benchmark)")
axes[2].bar(xpos + 0.2, [dl_metrics[m] for m in metrics_plot], width=0.4,
            label="Embedding network")
axes[2].set_xticks(xpos); axes[2].set_xticklabels(metrics_plot, fontsize=8)
axes[2].set_ylim(0.5, 0.95)
axes[2].set_title("Metric comparison"); axes[2].legend(fontsize=8)
plt.tight_layout()
save_fig("dl_fig02_benchmark_vs_network.png")


# ============================================================================
# STEP 5 - WHAT THE EMBEDDINGS LEARNED
# ============================================================================
banner("STEP 5 | INTERPRETING THE LEARNED EMBEDDINGS")

# This is the part a tree ensemble cannot provide. Each categorical level now
# has a learned coordinate; levels the network treats as behaviourally similar
# end up close together, which can be read directly by a marketing team.
buy_rate_by_level = {c: df.groupby(c)[TARGET].mean() for c in CAT_COLS}

plot_cols = ["Occupation", "CountryRegionName", "Education",
             "NumberChildrenAtHome", "TotalChildren", "NumberCarsOwned"]
fig, axes = plt.subplots(2, 3, figsize=(16.5, 9))
emb_records = []

for ax, col in zip(axes.ravel(), plot_cols):
    i = CAT_COLS.index(col)
    W = model.embeddings[i].weight.detach().cpu().numpy()
    levels = cat_levels[col]
    coords = PCA(n_components=2, random_state=SEED).fit_transform(W) \
        if W.shape[1] > 2 else (W if W.shape[1] == 2
                                else np.column_stack([W[:, 0], np.zeros(len(W))]))
    rates = np.array([buy_rate_by_level[col].get(l, np.nan) for l in levels])
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=rates * 100, s=180,
                    cmap="RdYlGn", edgecolor="black", linewidth=.6, zorder=3)
    for j, lvl in enumerate(levels):
        ax.annotate(str(lvl), (coords[j, 0], coords[j, 1]),
                    fontsize=8, xytext=(6, 5), textcoords="offset points")
    ax.set_title(col + " (dim " + str(W.shape[1]) + ")", fontsize=10)
    ax.set_xlabel("embedding axis 1", fontsize=8)
    ax.set_ylabel("embedding axis 2", fontsize=8)
    plt.colorbar(sc, ax=ax, label="buy rate (%)")
    for j, lvl in enumerate(levels):
        emb_records.append({"Feature": col, "Level": str(lvl),
                            "EmbX": round(float(coords[j, 0]), 4),
                            "EmbY": round(float(coords[j, 1]), 4),
                            "BuyRate_%": round(float(rates[j] * 100), 2)})

plt.suptitle("Learned entity embeddings - levels the network treats as similar "
             "sit close together\n(colour = actual purchase rate)", y=1.01)
plt.tight_layout()
save_fig("dl_fig03_learned_embeddings.png")
save_table(pd.DataFrame(emb_records), "dl_t03_embedding_coordinates.csv")

# --- 5.1 Does embedding geometry track purchase behaviour? ------------------
# If the first embedding axis correlates with the observed buy rate, the
# network has organised each categorical field along a propensity dimension
# without ever being told to.
print("Correlation between embedding axis 1 and observed purchase rate:")
corr_rows = []
for col in plot_cols:
    sub = pd.DataFrame([r for r in emb_records if r["Feature"] == col])
    if len(sub) > 2 and sub["BuyRate_%"].std() > 0:
        r = float(np.corrcoef(sub["EmbX"], sub["BuyRate_%"])[0, 1])
        corr_rows.append({"Feature": col, "Levels": len(sub),
                          "Corr_axis1_vs_buyrate": round(r, 3),
                          "AbsCorr": round(abs(r), 3)})
        print("  " + col.ljust(22) + format(r, "+.3f"))
corr_tbl = pd.DataFrame(corr_rows).sort_values("AbsCorr", ascending=False)
save_table(corr_tbl, "dl_t04_embedding_vs_buyrate_correlation.csv")

# --- 5.2 Nearest-neighbour pairs in embedding space -------------------------
print("\nClosest level pairs in each embedding (what the network considers alike):")
near_rows = []
for col in ["Occupation", "CountryRegionName", "Education"]:
    i = CAT_COLS.index(col)
    W = model.embeddings[i].weight.detach().cpu().numpy()
    levels = cat_levels[col]
    d = np.linalg.norm(W[:, None, :] - W[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    a, b = np.unravel_index(np.argmin(d), d.shape)
    far = np.unravel_index(np.argmax(np.where(np.isinf(d), -1, d)), d.shape)
    print("  " + col.ljust(20) + " closest: " + str(levels[a]) + " <-> "
          + str(levels[b]) + "   |   most distant: " + str(levels[far[0]])
          + " <-> " + str(levels[far[1]]))
    near_rows.append({"Feature": col,
                      "ClosestPair": str(levels[a]) + " <-> " + str(levels[b]),
                      "MostDistantPair": str(levels[far[0]]) + " <-> "
                                         + str(levels[far[1]])})
save_table(pd.DataFrame(near_rows), "dl_t05_embedding_neighbours.csv")


# ============================================================================
# STEP 6 - SUMMARY
# ============================================================================
banner("STEP 6 | SUMMARY")

summary = {
    "framework": "PyTorch " + torch.__version__,
    "device": str(DEVICE),
    "architecture": "Entity embeddings -> BN -> 128 -> 64 -> 1",
    "trainable_parameters": int(n_params),
    "embedded_features": len(CAT_COLS),
    "continuous_features": len(CONT_COLS),
    "train_rows": int(len(idx_tr)), "val_rows": int(len(idx_val)),
    "test_rows": int(len(idx_test)),
    "epochs_run": int(hist["Epoch"].max()), "best_epoch": int(best_epoch),
    "best_val_auc": round(float(best_auc), 4),
    "network_test_ROC_AUC": round(dl_metrics["ROC_AUC"], 4),
    "network_test_F1": round(dl_metrics["F1"], 4),
    "network_test_Accuracy": round(dl_metrics["Accuracy"], 4),
    "network_test_PR_AUC": round(dl_metrics["PR_AUC"], 4),
    "benchmark_model": CHAMP,
    "benchmark_test_ROC_AUC": round(float(champ_row["ROC_AUC"]), 4),
    "delta_ROC_AUC": round(delta, 4),
    "verdict": verdict,
}
with open(os.path.join(TAB_DIR, "dl_t06_run_summary.json"), "w") as fh:
    json.dump(summary, fh, indent=2)
print(json.dumps(summary, indent=2))

torch.save({"state_dict": model.state_dict(), "emb_spec": emb_spec,
            "cat_cols": CAT_COLS, "cont_cols": CONT_COLS,
            "cat_maps": cat_maps}, os.path.join(TAB_DIR, "dl_model.pt"))
print("\n   [model saved] " + os.path.join(TAB_DIR, "dl_model.pt"))

banner("DEEP LEARNING PIPELINE COMPLETE")
print("Figures : " + FIG_DIR)
print("Tables  : " + TAB_DIR)
