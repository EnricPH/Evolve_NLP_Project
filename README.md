# Trustpilot Review Analysis — NLP & Deep Learning Case Study

## Overview

This project presents a full end-to-end Natural Language Processing
pipeline applied to Trustpilot customer reviews. The analysis was
conducted from the perspective of a Data Scientist at
**electricalworld.com**, operating in the **Electronics & Technology**
sector, with the goal of extracting actionable business intelligence
from unstructured customer feedback.

The pipeline covers data cleaning, exploratory data analysis,
competitive benchmarking, topic modelling with BERTopic, and
sentiment analysis — culminating in a structured comparison between
the target company and all sector competitors.

---

## Business Questions Answered

1. Are the majority of reviews positive or negative — for our company
   and for competitors?
2. What topics do reviews discuss — for our company and across the
   sector?
3. For each topic, is the predominant sentiment positive or negative —
   and where do we outperform or underperform competitors?
4. What are the priority areas for improvement?

---

## Dataset

- **Source**: Trustpilot Reviews (provided as `trustpilot-reviews-123k.csv`)
- **Total reviews**: 123,181
- **Companies**: 1,680 across 22 sectors
- **Target subset**: Companies with exactly 100 reviews (558 companies)
- **Selected company**: electricalworld.com
- **Selected sector**: Electronics & Technology
- **Sector reviews used**: 5,729

### Dataset columns

| Column | Description |
|--------|-------------|
| `category` | Business sector |
| `company` | Company domain name |
| `description` | Company description text |
| `title` | Review title |
| `review` | Full review text |
| `stars` | Star rating (1–5) |

---

## Project Structure
├── datasets/
      ├──trustpilot-reviews-123k.csv        # Full dataset (124 MB)
      ├──Emp_100_reviews.xlsx               # 558 companies with 100 reviews
├── notebooks/
      ├──nlp_analysis.ipynb   # Main analysis notebook
├── recursos/
├── src/
      ├── clean.py   # File containing cleaning functions
      ├── eda.py     # File containing eda and visualization functions
      ├── nlp.py     # File containing NLP model training and visualizations functions
├── requirements.txt    # All required libraries
└── README.md                          # This file

---

## Pipeline

The analysis is structured in the following sequential stages:

### Stage 1 — Data Cleaning
- Unicode normalisation and ASCII conversion
- HTML tag removal and entity decoding
- URL and email address removal
- Escaped newline and whitespace normalisation
- Special character and emoji removal
- Applied to `title`, `review`, and `description` columns
- Designed to work with any of the 22 available categories

### Stage 2 — Exploratory Data Analysis (EDA)
- Star rating distribution across the full sector
- Per-company aggregated statistics (avg stars, % positive,
  % negative, avg review word count)
- Review length analysis split by sentiment tier
- Top and bottom company rankings by average rating
- Styled summary tables with conditional colour formatting

### Stage 3 — Competitive Positioning
- KDE plot of average star rating with target company percentile
- Stacked sentiment split: target vs sector average, best, and worst
- Percentile dashboard across three KPIs:
  % positive reviews, % negative reviews, avg star rating
- Word count by sentiment tier with sector IQR benchmarking

### Stage 4 — Topic Modelling (BERTopic)
- Sentence embeddings via `all-MiniLM-L6-v2` (SentenceTransformer)
- Dimensionality reduction with UMAP
  (n_neighbors=30, min_dist=0.1, cosine metric)
- Density-based clustering with HDBSCAN
  (min_cluster_size=60, min_samples=10)
- Topic representation via KeyBERTInspired keyword extraction
- Post-fit topic reduction to ≤12 general topics
- Model fitted on competitor corpus (larger, more stable)
- Target reviews transformed using the same fitted model
  to ensure a directly comparable topic space

### Stage 5 — Sentiment Analysis
- Sentiment derived from star ratings (fast, ground-truth proxy):
  4–5★ → positive, 3★ → neutral, 1–2★ → negative
- Optional: transformer-based inference using
  `cardiffnlp/twitter-roberta-base-sentiment-latest`

### Stage 6 — Topic × Sentiment Analysis
- Per-topic sentiment aggregation (% positive, neutral, negative)
- Net sentiment score: % positive − % negative (range −100 to +100)
- Dominant sentiment classification per topic
- Head-to-head net sentiment comparison (target vs competitors)
- Gap analysis to identify strengths and weaknesses
- Automated ranking of improvement areas

### Stage 7 — Visualizations
| Plot | Purpose |
|------|---------|
| Star rating distribution | Sector-wide polarity overview |
| Top/bottom companies by avg stars | Competitive landscape |
| Word count by sentiment | Length–polarity relationship |
| Avg word count per company | Engagement vs satisfaction |
| Company summary heatmap | Multi-KPI overview |
| KDE positioning plot | Target percentile in sector |
| Sentiment split stacked bar | Benchmark comparison |
| Percentile dashboard | 3-KPI strip chart |
| Word count box plot | Verbosity positioning |
| Topic distribution bars | Topic volume and dominant sentiment |
| Topic × sentiment heatmap | Granular topic-level polarity |
| Head-to-head net sentiment | Strength/weakness identification |
| Top words + avg sentiment | Most discussed terms and their tone |
| PCA embedding scatter | Semantic cluster structure |

---

## Key Findings

### Competitive Position
electricalworld.com ranks at the **4th percentile** of its sector,
with an average of 3.00 stars and a perfectly polarised 40% positive /
40% negative review split — matching the sector's worst-performing
companies.

### Topic Insights
Reviews concentrate around two operational modes:

**When the order goes right:**
Customer service interactions and same-day delivery generate net
sentiment scores of +73 to +100, *outperforming* sector competitors
by gaps of up to +117 points.

**When the order goes wrong:**
Wrong items sent, failed deliveries, and order cancellations generate
100% negative reviews, underperforming competitors by up to −93 points.

### Priority Improvement Areas
1. Order picking accuracy and item verification (net −100, gap −93)
2. Cancellation and refund process speed and communication (net −100)
3. Failed delivery resolution and courier SLA management (net −100)

---

## Technologies Used

| Library | Purpose |
|---------|---------|
| `pandas` / `numpy` | Data manipulation |
| `matplotlib` / `seaborn` | Static visualizations |
| `scikit-learn` | PCA, CountVectorizer, TF-IDF |
| `sentence-transformers` | SBERT review embeddings |
| `umap-learn` | Dimensionality reduction |
| `hdbscan` | Density-based clustering |
| `bertopic` | Topic modelling |
| `transformers` | Optional transformer sentiment model |
| `plotly` | Interactive BERTopic visualizations |

### Installation

```bash
pip install --no-deps bertopic
pip install --upgrade numpy hdbscan umap-learn pandas scikit-learn \
            tqdm plotly pyyaml sentence-transformers transformers \
            matplotlib seaborn nbformat
```