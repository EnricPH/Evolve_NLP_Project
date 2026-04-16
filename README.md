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