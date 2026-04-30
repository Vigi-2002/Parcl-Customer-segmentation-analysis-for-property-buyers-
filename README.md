<div align="center">

<img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/scikit--learn-1.x-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
<img src="https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
<img src="https://img.shields.io/badge/Plotly-5.x-3F4F75?style=for-the-badge&logo=plotly&logoColor=white"/>
<img src="https://img.shields.io/badge/Pandas-2.x-150458?style=for-the-badge&logo=pandas&logoColor=white"/>

<br/><br/>

# 🏙️ Parcl — Property Buyer Segmentation

### *Turning 2,000 client profiles into four actionable buyer personas using unsupervised machine learning*

<br/>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://opensource.org/licenses/MIT)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)
![Internship](https://img.shields.io/badge/Type-Internship%20Project-blueviolet?style=flat-square)
![Clusters](https://img.shields.io/badge/Clusters-K--Means%20k%3D4-orange?style=flat-square)

</div>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [The Problem](#-the-problem)
- [Dataset](#-dataset)
- [Methodology](#-methodology)
- [Buyer Segments Discovered](#-buyer-segments-discovered)
- [Key Insights](#-key-insights)
- [Project Structure](#-project-structure)
- [Streamlit Dashboard](#-streamlit-dashboard)
- [Results & Metrics](#-results--metrics)
- [Future Improvements](#-future-improvements)
- [Tech Stack](#-tech-stack)

---

## 🔍 Overview

This project applies **K-Means** and **Hierarchical Clustering** to segment Parcl's property buyer base into four distinct personas. The pipeline covers the full data science lifecycle — from raw messy CSVs to an interactive web dashboard — enabling Parcl's product, marketing, and customer success teams to make data-driven decisions about their users.

> **TL;DR:** Clean data → encode features → scale → cluster → interpret → dashboard.
> Four buyer types emerge, each with distinct investment behaviour, geography, and financing patterns.

---

The app opens at: https://s4or2hnl4bhqc96mqxbjbd.streamlit.app/

---

## 🎯 The Problem

Parcl had four knowledge gaps about their buyer base:

| Gap | Description |
|-----|-------------|
| 🧩 **Buyer Classification** | No systematic taxonomy of buyer types existed |
| 💡 **Investment Motivations** | Unknown why different demographics buy |
| 🌍 **Geographic Differences** | Behavioural variation by region unexplored |
| 💳 **Financing Patterns** | Loan vs cash-buyer dynamics unmapped |

---

## 📊 Dataset

Two CSV files provided for analysis:

```
📁 data/
├── clients.csv       # 2,000 rows × 12 columns — buyer demographics & behaviour
└── properties.csv    # 10,000 rows × 9 columns — property transactions
```

**clients.csv** — key columns:

| Column | Type | Description |
|--------|------|-------------|
| `client_id` | string | Unique buyer identifier |
| `date_of_birth` | string | Parsed → derived `age` feature |
| `country` / `region` | string | Geographic location (10 countries, 57 regions) |
| `acquisition_purpose` | string | `Home` or `Investment` |
| `loan_applied` | string | `Yes` or `No` |
| `satisfaction_score` | int | 1–5 platform rating |
| `referral_channel` | string | `Website`, `Agency`, or `Client` |

**properties.csv** — key columns:

| Column | Type | Description |
|--------|------|-------------|
| `client_ref` | string | Join key → links to `client_id` |
| `sale_price` | string | `$1,250,000.00` format → cleaned to float |
| `floor_area_sqft` | float | Unit size in square feet |
| `listing_status` | string | `Sold` or `Listed` — only Sold records used |

---

## 🔬 Methodology

```
┌─────────────┐    ┌──────────────────┐    ┌────────────────┐
│  Raw CSVs   │───▶│  Data Cleaning   │───▶│    Feature     │
│ 2K clients  │    │  • Deduplication │    │  Engineering   │
│ 10K props   │    │  • Label norm.   │    │  • Aggregation │
└─────────────┘    │  • Date parsing  │    │  • Encoding    │
                   │  • Price cleaning│    │  • Scaling     │
                   └──────────────────┘    └───────┬────────┘
                                                   │
                   ┌──────────────────┐    ┌───────▼────────┐
                   │  Interpretation  │◀───│   Clustering   │
                   │  • Segment names │    │  • K-Means k=4 │
                   │  • Profile cards │    │  • Hier. valid.│
                   │  • Insights      │    │  • PCA viz     │
                   └──────────────────┘    └────────────────┘
```

### Step 1 — Data Cleaning
```python
# Remove duplicates
clients = clients.drop_duplicates(subset='client_id')

# Normalise labels
for col in cat_cols:
    clients[col] = clients[col].str.strip().str.title()

# Parse age from date of birth
clients['age'] = ((pd.Timestamp('2024-01-01') - clients['dob']).dt.days / 365.25)

# Clean sale_price strings  ($1,250,000 → 1250000.0)
props['sale_price'] = props['sale_price'].str.replace(r'[\$,]', '', regex=True).astype(float)
```

### Step 2 — Feature Engineering
```python
# Aggregate property data per client
prop_agg = props[props['listing_status'] == 'Sold'].groupby('client_id').agg(
    num_properties   = ('listing_id',     'count'),
    total_investment = ('sale_price',      'sum'),
    avg_unit_area    = ('floor_area_sqft', 'mean'),
)

# Label encode binary categoricals
df['loan_enc']    = (df['loan_applied'] == 'Yes').astype(int)
df['purpose_enc'] = (df['acquisition_purpose'] == 'Investment').astype(int)

# One-hot encode multi-class categoricals (country, referral_channel)
ohe = pd.get_dummies(df[['referral_channel', 'country']], drop_first=False)
```

### Step 3 — Scaling
```python
from sklearn.preprocessing import StandardScaler
X = StandardScaler().fit_transform(X_raw)
# Result: mean ≈ 0, std ≈ 1 across all 21 features
```

### Step 4 — Optimal Cluster Selection

Three methods evaluated simultaneously:

```
k=2  →  Silhouette: 0.141  |  Inertia: 12,840  |  DB: 2.18
k=3  →  Silhouette: 0.148  |  Inertia: 11,203  |  DB: 2.05
k=4  →  Silhouette: 0.156  |  Inertia: 10,021  |  DB: 1.97  ✅ CHOSEN
k=5  →  Silhouette: 0.149  |  Inertia:  9,440  |  DB: 2.11
```

### Step 5 — K-Means Clustering
```python
from sklearn.cluster import KMeans

km = KMeans(n_clusters=4, init='k-means++', n_init=15, random_state=42)
df['cluster'] = km.fit_predict(X)
```

### Step 6 — Hierarchical Validation
```python
from scipy.cluster.hierarchy import linkage
from sklearn.cluster import AgglomerativeClustering

# Ward linkage dendrogram on 300-point sample → confirms 4-cluster structure
linked = linkage(X_sample, method='ward')
hc     = AgglomerativeClustering(n_clusters=4, linkage='ward').fit(X)
```

---

## 👥 Buyer Segments Discovered

| # | Segment | Clients | Share | Avg Investment | Avg Age | Satisfaction | Loan Rate |
|---|---------|---------|-------|---------------|---------|--------------|-----------|
| 💎 | **Luxury Investors** | 7 | 0.8% | $1,568,590 | 48.6 yrs | 2.57 / 5 | 29% |
| 🌍 | **Global Investors** | 74 | 8.7% | $1,272,131 | 49.8 yrs | 2.88 / 5 | 38% |
| 🏢 | **Corporate Buyers** | 458 | 53.6% | $1,265,070 | 52.8 yrs | 3.02 / 5 | 38% |
| 🏠 | **First-Time Buyers** | 316 | 37.0% | $1,260,165 | 53.6 yrs | 3.09 / 5 | 36% |

<details>
<summary>💎 <strong>Luxury Investors</strong> — expand for full profile</summary>

- Smallest but **highest-value** segment — $1.57M average investment
- Largest unit sizes: **1,369 sqft** average
- Lowest loan dependency: **29%** — predominantly cash buyers
- ⚠️ **Lowest satisfaction score: 2.57/5** — most urgent business risk
- 86% home-purpose purchases, 14% investment
- 100% individual buyers (no corporate entities)

</details>

<details>
<summary>🌍 <strong>Global Investors</strong> — expand for full profile</summary>

- Internationally diverse across all 10 countries in the dataset
- Highest investment-purpose rate: **34%**
- 97% individual buyers
- Likely driven by portfolio diversification and cross-border real estate arbitrage

</details>

<details>
<summary>🏢 <strong>Corporate Buyers</strong> — expand for full profile</summary>

- **Dominant segment** — 53.6% of the entire user base
- Highest proportion of company-type buyers (4%)
- Older average age: **52.8 years**
- 31% investment-purpose acquisition rate
- High-volume, repeat-purchase behaviour expected

</details>

<details>
<summary>🏠 <strong>First-Time Buyers</strong> — expand for full profile</summary>

- **Highest satisfaction score: 3.09/5** — most content segment
- Overwhelmingly home-focused: **68% home** vs 32% investment
- Highest average age: **53.6 years** (first purchase on Parcl platform)
- Smallest average unit sizes: **1,151 sqft**
- Most loan-reliant relative to their investment level

</details>

---

## 💡 Key Insights

> 🚨 **Most Urgent Finding:** Luxury Investors — the *highest-value* segment — have the *lowest satisfaction score* (2.57/5). This inverse relationship between client value and happiness is a major churn risk requiring immediate action.

```
Insight 1 │ 🇺🇸  76.1% of clients are US-based — significant single-market concentration risk
Insight 2 │ 💳  62.7% are cash buyers — Parcl attracts an affluent, self-financed audience
Insight 3 │ 💻  54.2% acquired via Website — digital channel is the primary growth engine
Insight 4 │ 🤝  Agency channel (37.2%) is underinvested relative to its acquisition volume
Insight 5 │ ⭐  Overall satisfaction of 3.01/5 — significant CX improvement headroom exists
```

---

## 📁 Project Structure

```
parcl-buyer-segmentation/
│
├── 📊 data/
│   ├── clients.csv
│   └── properties.csv
│
├── 🐍 parcl_segmentation.py      # Full ML pipeline (cleaning → EDA → clustering)
├── 🌐 app.py                     # Streamlit interactive dashboard
│
├── 📈 outputs/
│   ├── 01_eda_overview.png       # 8-panel EDA dashboard
│   ├── 02_eda_geo_crosstabs.png  # Geographic heatmaps
│   ├── 03_eda_correlation.png    # Pearson correlation matrix
│   ├── 04_cluster_selection.png  # Elbow + Silhouette + Davies-Bouldin
│   ├── 05_dendrogram.png         # Hierarchical clustering dendrogram
│   ├── 06_pca_clusters.png       # PCA 2D cluster scatter
│   ├── 07_radar_profiles.png     # Per-segment radar charts
│   ├── 08_segment_breakdown.png  # Distribution breakdowns by segment
│   ├── 09_financial_profiles.png # Violin charts — investment & unit area
│   └── 10_segment_cards.png      # Summary profile cards
│
└── 📄 README.md
```

---

## 🌐 Streamlit Dashboard

Four interactive tabs with real-time sidebar filtering by **Country**, **Region**, **Acquisition Purpose**, and **Client Type**:

| Tab | Contents |
|-----|----------|
| 📊 **Segmentation Overview** | Donut chart, bar chart, PCA 2D scatter, segment profile cards |
| 💰 **Investor Behavior** | Box plots, violin charts, bubble scatter, stacked bars, referral mix |
| 🌍 **Geographic Analysis** | Choropleth world map, treemap, country × purpose heatmap |
| 🔬 **Segment Insights** | Per-segment stats table, histograms, pie charts, raw data + CSV export |

---

## 📏 Results & Metrics

| Metric | Value | Notes |
|--------|-------|-------|
| **Algorithm** | K-Means (k=4) | Validated with Hierarchical Clustering (Ward) |
| **Silhouette Score** | 0.156 | Moderate — consistent with real-world human behaviour data |
| **Davies-Bouldin Score** | 1.97 | Best result across k = 2 through 10 |
| **PCA Variance (PC1+PC2)** | ~24% | Remaining variance spread across 19 other dimensions |
| **Clients (post-cleaning)** | 855 unique | After deduplication from 2,000 raw records |
| **Feature count** | 21 | 8 numeric/binary + 13 one-hot encoded columns |

---

## 🚀 Future Improvements

- [ ] **Behavioural features** — session frequency, search patterns, time-to-purchase
- [ ] **DBSCAN** — density-based clustering for automatic outlier detection
- [ ] **Automated retraining pipeline** — quarterly refresh with MLflow experiment tracking
- [ ] **Pre-clustering PCA** — dimensionality reduction before (not just after) clustering
- [ ] **Bootstrap stability testing** — validate segment consistency across random subsamples
- [ ] **NLP enrichment** — sentiment extraction from support tickets and survey responses

---

## 🛠 Tech Stack

| Category | Tools |
|----------|-------|
| **Language** | Python 3.11+ |
| **Data** | Pandas, NumPy |
| **Machine Learning** | scikit-learn, SciPy |
| **Visualisation** | Matplotlib, Seaborn, Plotly |
| **Dashboard** | Streamlit |
| **Clustering** | K-Means, Agglomerative (Ward linkage) |
| **Dimensionality Reduction** | PCA |
| **Evaluation** | Silhouette Score, Davies-Bouldin Score, Elbow Method |

---
---
Internship project — Parcl · 2026
