#!/usr/bin/env python3
"""
Reproducible analysis script for:
Are undergraduate accounting students increasingly treating the MAcc as optional rather than necessary?

This script uses only Python standard-library modules so it runs in constrained environments.
"""

import csv
import math
import os
import statistics
from collections import Counter, defaultdict

# ------------------------------
# 1) Load raw Qualtrics CSV data
# ------------------------------
INPUT_FILE = "Alternative CPA Pathways Survey_December 31, 2025_09.45.csv"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(INPUT_FILE, encoding="utf-8-sig", newline="") as f:
    reader = csv.reader(f)
    qualtrics_codes = next(reader)   # first row in file (e.g., Q52)
    question_text = next(reader)     # second row in file (human-readable text)
    _import_meta = next(reader)      # third row metadata row from Qualtrics
    raw_rows = list(reader)

# Build a metadata dictionary that keeps both question code and question text.
columns = []
for i, code in enumerate(qualtrics_codes):
    columns.append({"index": i, "code": code, "text": question_text[i]})

idx_by_code = {c["code"]: c["index"] for c in columns}

# -------------------------------------------------------------------------
# 2) Restrict to undergraduates only (student status item from survey: Q27)
# -------------------------------------------------------------------------
status_col = "Q27"  # Are you currently an undergraduate student or graduate student?
undergrad_rows = [r for r in raw_rows if r[idx_by_code[status_col]].strip() == "Undergraduate"]

# ---------------------------------------------------------------------
# 3) Build perceived MAcc necessity measure from available undergrad items
# ---------------------------------------------------------------------
# We use two ordered items that exist in the undergraduate branch:
# - Q52: change in desire to pursue graduate degree given alternative pathway
# - Q55: expected lifetime earnings return from graduate degree
# Higher score = perceiving graduate degree as more necessary/valuable.

macc_item_1 = "Q52"
macc_item_2 = "Q55"

q52_map = {
    "Significantly decreased desire": 1,
    "Decreased desire": 2,
    "No change in desire": 3,
    "Increased desire": 4,
    "Significantly increased desire": 5,
}

q55_map = {
    "Definitely not": 1,
    "Probably not": 2,
    "Might or might not": 3,
    "Probably yes": 4,
    "Definitely yes": 5,
}

# -------------------------------------------------------------
# 4) Awareness of alternative pathway (Q53: Yes/No awareness)
# -------------------------------------------------------------
awareness_col = "Q53"
awareness_map = {"No": 0, "Yes": 1}

# ---------------------------------------------------------------
# Controls for adjusted models if available in undergrad responses
# ---------------------------------------------------------------
# Q29 CPA intent, Q46 accounting work >20h/week, Q65 age
control_q29 = "Q29"
control_q46 = "Q46"
control_q65 = "Q65"

q29_map = {
    "Very unlikely": 1,
    "Somewhat unlikely": 2,
    "Neither likely nor unlikely": 3,
    "Somewhat likely": 4,
    "Very likely": 5,
}

q46_map = {"No": 0, "Yes": 1}


def parse_age(x):
    x = x.strip()
    if x == "":
        return None
    try:
        return float(x)
    except ValueError:
        return None


def as_num(value, mapping):
    value = value.strip()
    if value == "":
        return None
    return mapping.get(value)


# -----------------------------------------------------------------
# 5) Clean data (drop missing on key variables for each model/table)
# -----------------------------------------------------------------
analysis_rows = []
for r in undergrad_rows:
    need_1 = as_num(r[idx_by_code[macc_item_1]], q52_map)
    need_2 = as_num(r[idx_by_code[macc_item_2]], q55_map)
    aware = as_num(r[idx_by_code[awareness_col]], awareness_map)
    cpa_intent = as_num(r[idx_by_code[control_q29]], q29_map)
    work20 = as_num(r[idx_by_code[control_q46]], q46_map)
    age = parse_age(r[idx_by_code[control_q65]])

    if need_1 is None or need_2 is None or aware is None:
        continue

    necessity_index = (need_1 + need_2) / 2.0

    # Robustness binary outcome: 1 = leans necessary, 0 = neutral/optional
    necessity_binary = 1 if necessity_index > 3 else 0

    analysis_rows.append(
        {
            "aware": aware,
            "necessity_q52": need_1,
            "necessity_q55": need_2,
            "necessity_index": necessity_index,
            "necessity_binary": necessity_binary,
            "cpa_intent": cpa_intent,
            "work20": work20,
            "age": age,
        }
    )

# Model rows with controls available
analysis_rows_controls = [
    d
    for d in analysis_rows
    if d["cpa_intent"] is not None and d["work20"] is not None and d["age"] is not None
]


# ---------------------------
# Utilities for statistics
# ---------------------------
def mean(x):
    return sum(x) / len(x) if x else float("nan")


def variance(x):
    if len(x) < 2:
        return float("nan")
    m = mean(x)
    return sum((v - m) ** 2 for v in x) / (len(x) - 1)


def norm_cdf(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def two_sided_p_from_z(z):
    return 2.0 * (1.0 - norm_cdf(abs(z)))


def welch_t_test(x, y):
    n1, n2 = len(x), len(y)
    m1, m2 = mean(x), mean(y)
    v1, v2 = variance(x), variance(y)
    if n1 < 2 or n2 < 2:
        return {"t": float("nan"), "p_approx": float("nan")}
    se = math.sqrt(v1 / n1 + v2 / n2)
    t = (m1 - m2) / se if se > 0 else float("nan")
    # Normal approximation for p-value (large-sample robust in this context)
    p = two_sided_p_from_z(t)
    return {"t": t, "p_approx": p}


def mann_whitney_u_test(x, y):
    # Average-rank ties handling
    pooled = [(v, 0) for v in x] + [(v, 1) for v in y]
    pooled.sort(key=lambda z: z[0])

    ranks = [0.0] * len(pooled)
    i = 0
    while i < len(pooled):
        j = i
        while j + 1 < len(pooled) and pooled[j + 1][0] == pooled[i][0]:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[k] = avg_rank
        i = j + 1

    rank_sum_x = sum(ranks[i] for i in range(len(pooled)) if pooled[i][1] == 0)
    n1, n2 = len(x), len(y)
    u1 = rank_sum_x - n1 * (n1 + 1) / 2.0

    mu = n1 * n2 / 2.0
    sigma = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
    z = (u1 - mu) / sigma if sigma > 0 else float("nan")
    p = two_sided_p_from_z(z)
    return {"u": u1, "z": z, "p_approx": p}


def two_proportion_z_test(success1, n1, success2, n2):
    p_pool = (success1 + success2) / (n1 + n2)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    z = ((success1 / n1) - (success2 / n2)) / se if se > 0 else float("nan")
    p = two_sided_p_from_z(z)
    return {"z": z, "p_approx": p}


def transpose(m):
    return [list(row) for row in zip(*m)]


def matmul(a, b):
    bt = transpose(b)
    out = []
    for row in a:
        out.append([sum(row[k] * col[k] for k in range(len(row))) for col in bt])
    return out


def invert_matrix(a):
    # Gauss-Jordan inversion
    n = len(a)
    aug = [row[:] + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(a)]

    for i in range(n):
        # pivot
        pivot_row = max(range(i, n), key=lambda r: abs(aug[r][i]))
        if abs(aug[pivot_row][i]) < 1e-12:
            raise ValueError("Singular matrix in OLS inversion.")
        aug[i], aug[pivot_row] = aug[pivot_row], aug[i]

        pivot = aug[i][i]
        aug[i] = [v / pivot for v in aug[i]]

        for r in range(n):
            if r == i:
                continue
            factor = aug[r][i]
            aug[r] = [aug[r][c] - factor * aug[i][c] for c in range(2 * n)]

    return [row[n:] for row in aug]


def ols(y, x, names):
    # y: list[float], x: list[list[float]] includes intercept
    n = len(y)
    k = len(x[0])

    y_col = [[v] for v in y]
    xt = transpose(x)
    xtx = matmul(xt, x)
    xtx_inv = invert_matrix(xtx)
    xty = matmul(xt, y_col)
    beta_col = matmul(xtx_inv, xty)
    beta = [b[0] for b in beta_col]

    # residual variance
    y_hat = [sum(x[i][j] * beta[j] for j in range(k)) for i in range(n)]
    resid = [y[i] - y_hat[i] for i in range(n)]
    sse = sum(e * e for e in resid)
    sigma2 = sse / (n - k)

    vcov = [[sigma2 * xtx_inv[i][j] for j in range(k)] for i in range(k)]
    se = [math.sqrt(vcov[i][i]) for i in range(k)]
    tvals = [beta[i] / se[i] if se[i] > 0 else float("nan") for i in range(k)]
    pvals = [two_sided_p_from_z(t) for t in tvals]  # normal approx

    # R-squared
    y_mean = mean(y)
    sst = sum((v - y_mean) ** 2 for v in y)
    r2 = 1.0 - (sse / sst) if sst > 0 else float("nan")

    rows = []
    for i in range(k):
        rows.append(
            {
                "term": names[i],
                "coef": beta[i],
                "std_err": se[i],
                "z_approx": tvals[i],
                "p_approx": pvals[i],
            }
        )

    return {"rows": rows, "n": n, "r2": r2}


# -------------------------------------
# 6) Descriptive statistics by awareness
# -------------------------------------
aware_grp = [d for d in analysis_rows if d["aware"] == 1]
unaware_grp = [d for d in analysis_rows if d["aware"] == 0]

aware_idx = [d["necessity_index"] for d in aware_grp]
unaware_idx = [d["necessity_index"] for d in unaware_grp]

# Ordinal distribution of the composite outcome (to 0.5 increments)
dist = Counter(d["necessity_index"] for d in analysis_rows)

with open(os.path.join(OUTPUT_DIR, "descriptive_statistics.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["group", "n", "mean_necessity_index", "median_necessity_index"])
    w.writerow(["Aware of alternative pathway", len(aware_idx), round(mean(aware_idx), 4), round(statistics.median(aware_idx), 4)])
    w.writerow(["Not aware of alternative pathway", len(unaware_idx), round(mean(unaware_idx), 4), round(statistics.median(unaware_idx), 4)])
    w.writerow(["Overall", len(analysis_rows), round(mean([d["necessity_index"] for d in analysis_rows]), 4), round(statistics.median([d["necessity_index"] for d in analysis_rows]), 4)])

with open(os.path.join(OUTPUT_DIR, "necessity_index_distribution.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["necessity_index", "count"])
    for k in sorted(dist.keys()):
        w.writerow([k, dist[k]])

# ---------------------------------------------
# 7) Group comparison tests (aware vs unaware)
# ---------------------------------------------
welch_res = welch_t_test(aware_idx, unaware_idx)
mw_res = mann_whitney_u_test(aware_idx, unaware_idx)

# ------------------------------
# 8) Regression (continuous scale)
# ------------------------------
# Unadjusted model: necessity_index ~ aware
x1 = [[1.0, d["aware"]] for d in analysis_rows]
y1 = [d["necessity_index"] for d in analysis_rows]
ols_unadjusted = ols(y1, x1, ["Intercept", "Aware alternative pathway"])

# Adjusted model with controls: + cpa_intent + work20 + age
x2 = [[1.0, d["aware"], d["cpa_intent"], d["work20"], d["age"]] for d in analysis_rows_controls]
y2 = [d["necessity_index"] for d in analysis_rows_controls]
ols_adjusted = ols(y2, x2, ["Intercept", "Aware alternative pathway", "CPA intent (1-5)", "Work >20h in accounting", "Age"])

# --------------------------------------------
# 9) Robustness: binary necessity specification
# --------------------------------------------
aware_bin = [d["necessity_binary"] for d in aware_grp]
unaware_bin = [d["necessity_binary"] for d in unaware_grp]

# Descriptive for binary outcome
with open(os.path.join(OUTPUT_DIR, "descriptive_statistics_binary.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["group", "n", "share_necessary"])
    w.writerow(["Aware of alternative pathway", len(aware_bin), round(mean(aware_bin), 4)])
    w.writerow(["Not aware of alternative pathway", len(unaware_bin), round(mean(unaware_bin), 4)])
    w.writerow(["Overall", len(analysis_rows), round(mean([d["necessity_binary"] for d in analysis_rows]), 4)])

# Group comparison for binary variable: two-proportion z-test
prop_res = two_proportion_z_test(sum(aware_bin), len(aware_bin), sum(unaware_bin), len(unaware_bin))

# Binary regression (linear probability model for transparency with stdlib)
x3 = [[1.0, d["aware"]] for d in analysis_rows]
y3 = [float(d["necessity_binary"]) for d in analysis_rows]
lpm_unadjusted = ols(y3, x3, ["Intercept", "Aware alternative pathway"])

x4 = [[1.0, d["aware"], d["cpa_intent"], d["work20"], d["age"]] for d in analysis_rows_controls]
y4 = [float(d["necessity_binary"]) for d in analysis_rows_controls]
lpm_adjusted = ols(y4, x4, ["Intercept", "Aware alternative pathway", "CPA intent (1-5)", "Work >20h in accounting", "Age"])

# -------------------------------------------------
# 10) Export regression + test results to one table
# -------------------------------------------------
reg_out = os.path.join(OUTPUT_DIR, "regression_results.csv")
with open(reg_out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["model", "n", "r_squared", "term", "coef", "std_err", "z_approx", "p_approx"])

    for model_name, res in [
        ("OLS necessity index (unadjusted)", ols_unadjusted),
        ("OLS necessity index (adjusted)", ols_adjusted),
        ("LPM necessity binary (unadjusted)", lpm_unadjusted),
        ("LPM necessity binary (adjusted)", lpm_adjusted),
    ]:
        for row in res["rows"]:
            w.writerow([
                model_name,
                res["n"],
                round(res["r2"], 5),
                row["term"],
                round(row["coef"], 5),
                round(row["std_err"], 5),
                round(row["z_approx"], 5),
                round(row["p_approx"], 5),
            ])

# Save test results too
with open(os.path.join(OUTPUT_DIR, "group_comparison_tests.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["test", "statistic_1", "statistic_2", "p_value_approx"])
    w.writerow(["Welch t-test (index, aware-unaware)", round(welch_res["t"], 5), "", round(welch_res["p_approx"], 5)])
    w.writerow(["Mann-Whitney U (index)", round(mw_res["u"], 5), round(mw_res["z"], 5), round(mw_res["p_approx"], 5)])
    w.writerow(["Two-proportion z-test (binary necessary)", round(prop_res["z"], 5), "", round(prop_res["p_approx"], 5)])

# -------------------------------------------
# 10) Visualization: group mean bar chart (SVG)
# -------------------------------------------
# We create a simple SVG directly to avoid non-stdlib plotting dependencies.
svg_path = os.path.join(OUTPUT_DIR, "awareness_vs_necessity.svg")
max_y = 5.0
bar_w = 160
gap = 90
left = 110
bottom = 300
plot_h = 220

means = [mean(unaware_idx), mean(aware_idx)]
labels = ["Not aware", "Aware"]
colors = ["#7A9CC6", "#D4835A"]

svg_lines = [
    '<svg xmlns="http://www.w3.org/2000/svg" width="650" height="380">',
    '<rect x="0" y="0" width="650" height="380" fill="white"/>',
    '<text x="30" y="30" font-size="18" font-family="Arial">Perceived MAcc Necessity by Awareness of Alternative Pathway</text>',
    f'<line x1="{left}" y1="{bottom}" x2="{left + 2*bar_w + gap}" y2="{bottom}" stroke="black"/>',
    f'<line x1="{left}" y1="{bottom}" x2="{left}" y2="{bottom - plot_h}" stroke="black"/>',
]

# y-axis ticks
for yval in [1, 2, 3, 4, 5]:
    ypix = bottom - (yval / max_y) * plot_h
    svg_lines.append(f'<line x1="{left-5}" y1="{ypix}" x2="{left}" y2="{ypix}" stroke="black"/>')
    svg_lines.append(f'<text x="{left-28}" y="{ypix+4}" font-size="11" font-family="Arial">{yval}</text>')

for i, m in enumerate(means):
    x = left + 30 + i * (bar_w + gap)
    h = (m / max_y) * plot_h
    y = bottom - h
    svg_lines.append(f'<rect x="{x}" y="{y}" width="{bar_w}" height="{h}" fill="{colors[i]}"/>')
    svg_lines.append(f'<text x="{x + bar_w/2 - 18}" y="{y - 8}" font-size="12" font-family="Arial">{m:.2f}</text>')
    svg_lines.append(f'<text x="{x + bar_w/2 - 35}" y="{bottom + 20}" font-size="12" font-family="Arial">{labels[i]}</text>')

svg_lines.append('</svg>')

with open(svg_path, "w", encoding="utf-8") as f:
    f.write("\n".join(svg_lines))

# -------------------------------------------------------------
# 11) Save compact text interpretation (associational, not causal)
# -------------------------------------------------------------
summary_txt = os.path.join(OUTPUT_DIR, "analysis_summary.txt")
with open(summary_txt, "w", encoding="utf-8") as f:
    f.write("Cross-sectional associational analysis\n")
    f.write("===================================\n\n")
    f.write(f"Undergraduate sample size after key-variable cleaning: n={len(analysis_rows)}\n")
    f.write(f"Undergraduate sample size with controls: n={len(analysis_rows_controls)}\n\n")

    f.write("Perceived MAcc necessity index (1-5; higher = more necessary)\n")
    f.write(f"- Mean (aware): {mean(aware_idx):.3f}\n")
    f.write(f"- Mean (not aware): {mean(unaware_idx):.3f}\n")
    f.write(f"- Welch t-test p (approx): {welch_res['p_approx']:.4f}\n")
    f.write(f"- Mann-Whitney p (approx): {mw_res['p_approx']:.4f}\n\n")

    aware_coef = [r for r in ols_adjusted["rows"] if r["term"] == "Aware alternative pathway"][0]
    f.write("Adjusted OLS association (index outcome):\n")
    f.write(
        f"- Aware coefficient: {aware_coef['coef']:.3f} (p≈{aware_coef['p_approx']:.4f})\n"
    )
    f.write("  Interpretation: awareness is associated with higher/lower index depending on sign;\n")
    f.write("  this is not a causal estimate due to cross-sectional observational design.\n\n")

    f.write("Binary robustness outcome (1=leans necessary, 0=neutral/optional)\n")
    f.write(f"- Share necessary (aware): {mean(aware_bin):.3f}\n")
    f.write(f"- Share necessary (not aware): {mean(unaware_bin):.3f}\n")
    f.write(f"- Two-proportion z-test p (approx): {prop_res['p_approx']:.4f}\n")

print("Analysis complete.")
print(f"Undergrad observations retained (key vars): {len(analysis_rows)}")
print(f"Outputs written to: {OUTPUT_DIR}")
