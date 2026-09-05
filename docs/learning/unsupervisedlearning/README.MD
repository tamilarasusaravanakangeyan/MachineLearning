# Unsupervised Learning

An unsupervised learning model aims to **identify meaningful patterns in a dataset** — without being given any correct answers (labels).

> **Analogy:** Imagine being handed a box of mixed LEGO bricks with no instructions. Nobody tells you what the categories are, but you naturally start grouping them — by color, by size, by shape. Unsupervised learning does the same thing with data: it finds the natural groupings and structure on its own.

```
Raw data (no labels)  ──►  Model finds structure  ──►  Groups / patterns / anomalies
```

The key difference from supervised learning: **no human provides the correct answers.** The model must discover the structure by itself.

---

## 1. Clustering

Many unsupervised learning models rely on a technique called **clustering** to organize similar data into groups (**"clusters"**).

### How clustering differs from classification

Clustering differs from classification because **the categories aren't defined by you.**

**Example:** An unsupervised model might cluster a weather dataset based on temperature, revealing segmentations that define the **seasons**. You might then attempt to *name* those clusters based on your understanding of the dataset:

```
Cluster 1: avg temp  2°C  → you name it "Winter"
Cluster 2: avg temp 15°C  → you name it "Spring / Autumn"
Cluster 3: avg temp 28°C  → you name it "Summer"
```

The model never knew the word "season" — it just noticed that the temperature readings naturally fall into groups.

| | Classification (supervised) | Clustering (unsupervised) |
|---|---|---|
| **Categories** | Defined by you in advance (`spam` / `not spam`) | Discovered by the model |
| **Training data** | Labeled | Unlabeled |
| **You provide** | Questions **and** answers | Questions only |
| **Naming the groups** | Done before training | Done *after*, by inspecting the clusters |

### More examples of clustering

| Scenario | Input data | Clusters the model might discover |
|---|---|---|
| **Customer segmentation** | Purchase history, visit frequency, cart size | "Bargain hunters", "loyal big spenders", "one-time buyers" |
| **News grouping** | Article text | Stories about the same event, grouped automatically |
| **Genetics** | DNA sequences | Population groups with shared ancestry |
| **Image organization** | Photo pixels | Photos of beaches vs. mountains vs. cities |

### Common clustering algorithms

- k-Means
- Hierarchical Clustering
- DBSCAN

---

## 2. Other common unsupervised techniques

### 2.1 Dimensionality reduction

Compresses data with many features into fewer features while keeping the important structure.

**Example:** A dataset with 100 measurements per customer can be reduced to 2–3 "summary" dimensions, making it possible to visualize the customers on a simple 2D chart.

*Common algorithms:* PCA, t-SNE, UMAP

### 2.2 Anomaly detection

Learns what "normal" data looks like, then flags anything unusual.

**Example:** A credit card system learns your typical spending pattern (small local purchases). A sudden $3,000 purchase from another country doesn't fit any learned pattern — it gets flagged as a possible fraud.

### 2.3 Association rule mining

Finds items that frequently occur together.

**Example:** A supermarket discovers that customers who buy bread and butter often also buy jam — so the three get placed near each other (*"market basket analysis"*).

---

## Supervised vs. Unsupervised at a glance

| | Supervised learning | Unsupervised learning |
|---|---|---|
| **Training data** | Labeled (features + correct answers) | Unlabeled (features only) |
| **Goal** | Predict a known target | Discover hidden structure |
| **Example task** | "Is this email spam?" | "What natural groups exist in my customers?" |
| **Main techniques** | Regression, classification | Clustering, dimensionality reduction, anomaly detection |
| **Human role** | Provides the answers up front | Interprets/names the patterns afterward |

> **Tip:** Unsupervised learning is often used *before* supervised learning — e.g., cluster your customers first to discover segments, then train a supervised model to assign new customers to those segments.
