import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
from transformers import pipeline as hf_pipeline


# ══════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════

SBERT_MODEL     = "all-MiniLM-L6-v2"
SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
RANDOM_STATE    = 42

# ── Similarity threshold ───────────────────────────────────────────────
# A review is assigned to its closest macro topic if its cosine
# similarity to that topic's seed centroid is >= SIMILARITY_THRESHOLD.
# If below threshold it lands in MACRO_OTHER.
#
# Set very low (0.05) so virtually every review gets assigned.
# Only raise it if you want to enforce a minimum confidence of match.
SIMILARITY_THRESHOLD = 0.05

MACRO_SEED_TOPICS = [
    # 0
    ["delivery", "shipping", "dispatch", "courier", "parcel",
     "package", "arrived", "tracked", "transit", "postage",
     "next day", "same day", "delivered", "shipment",
     "not arrived", "missing", "late", "delayed", "waiting",
     "still waiting", "never arrived", "weeks", "days"],
    # 1
    ["order", "purchase", "checkout", "bought", "buying",
     "placed order", "online order", "transaction", "ordered",
     "easy purchase", "ordering process", "confirmation",
     "order placed", "add to cart", "basket", "payment",
     "invoice", "receipt", "process smooth"],
    # 2
    ["refund", "return", "cancellation", "cancelled", "money back",
     "reimbursement", "reversed", "chargeback", "exchange",
     "send back", "returned item", "refunded", "dispute",
     "compensation", "credit", "void", "rejected return",
     "refused refund", "waiting refund"],
    # 3
    ["customer service", "support", "helpdesk", "agent", "staff",
     "representative", "call centre", "response", "replied",
     "helpful team", "contact", "communicate", "assistance",
     "rude", "unhelpful", "ignored", "no reply", "email",
     "phone call", "live chat", "waiting time", "hold",
     "spoke", "told", "advised", "promised"],
    # 4
    ["quality", "condition", "build", "durable", "reliable",
     "product quality", "well made", "broken", "faulty",
     "defective", "damaged", "poor quality", "excellent condition",
     "refurbished", "second hand", "scratched", "incomplete",
     "missing parts", "wrong item", "not as described",
     "like new", "grade", "tested"],
    # 5
    ["price", "value", "affordable", "expensive", "cost",
     "worth", "cheap", "overpriced", "discount", "deal",
     "money", "competitive price", "good value", "pricing",
     "fee", "charge", "quoted", "paid", "saving", "budget"],
    # 6
    ["website", "app", "online", "interface", "navigation",
     "checkout page", "user experience", "loading", "error",
     "login", "account", "platform", "digital", "easy to use",
     "page", "link", "browser", "mobile", "desktop",
     "slow", "crashed", "glitch", "update", "password"],
    # 7
    ["repair", "fix", "technician", "engineer", "diagnostic",
     "fault", "broken device", "service centre", "technical",
     "warranty repair", "maintenance", "replaced part",
     "booked", "appointment", "turnaround", "quote",
     "assessed", "inspected", "under warranty", "out of warranty"],
]

MACRO_TOPIC_NAMES = [
    "Delivery & Shipping",
    "Order & Purchase Process",
    "Returns, Refunds & Cancellations",
    "Customer Service & Support",
    "Product Quality & Condition",
    "Price & Value",
    "Website & Online Experience",
    "Repair & Technical Service",
]

MACRO_OTHER = "General / Other"


# ══════════════════════════════════════════════════════════════════════
# STAGE 1 — SPLIT
# ══════════════════════════════════════════════════════════════════════

def split_target_competitors(df: pd.DataFrame, target: str) -> tuple:
    """
    Split the cleaned category DataFrame into target and competitor
    subsets. Empty reviews are dropped from both.

    Parameters
    ----------
    df     : pd.DataFrame   cleaned category DataFrame
    target : str            company name matching df['company']

    Returns
    -------
    tuple (df_target, df_competitors)
    """
    if target not in df['company'].values:
        raise ValueError(f"'{target}' not found. Check TARGET variable.")

    mask           = df['company'] == target
    df_target      = df[mask].copy().reset_index(drop=True)
    df_competitors = df[~mask].copy().reset_index(drop=True)

    for d, name in [(df_target, 'target'), (df_competitors, 'competitors')]:
        empty = d['review'].fillna('').str.strip() == ''
        if empty.sum():
            print(f"  ⚠ Dropping {empty.sum()} empty reviews from {name}")

    df_target      = df_target[
        df_target['review'].fillna('').str.strip() != ''
    ].reset_index(drop=True)
    df_competitors = df_competitors[
        df_competitors['review'].fillna('').str.strip() != ''
    ].reset_index(drop=True)

    print(f"  Target '{target}'       : {len(df_target)} reviews")
    print(f"  Competitors             : {len(df_competitors)} reviews "
          f"({df_competitors['company'].nunique()} companies)")

    return df_target, df_competitors


# ══════════════════════════════════════════════════════════════════════
# STAGE 2 — SEED CENTROIDS
# ══════════════════════════════════════════════════════════════════════

def build_seed_centroids(sbert_instance) -> np.ndarray:
    """
    Encode every seed word list and compute its centroid embedding.

    Each macro topic becomes a single vector = mean of all its seed
    word embeddings. This centroid represents the semantic centre of
    that topic in the SBERT embedding space.

    Called once and reused for both competitor and target assignment,
    so the same semantic space is used for both — making results
    directly comparable.

    Parameters
    ----------
    sbert_instance : loaded SentenceTransformer

    Returns
    -------
    np.ndarray of shape (n_macro_topics, embedding_dim)
        Row i = centroid for MACRO_TOPIC_NAMES[i]
    """
    print("  Building seed centroids ...")
    centroids = []
    for i, seed_list in enumerate(MACRO_SEED_TOPICS):
        seed_embs = sbert_instance.encode(
            seed_list, convert_to_numpy=True, show_progress_bar=False
        )
        centroid = seed_embs.mean(axis=0)
        centroids.append(centroid)
        print(f"    [{i}] {MACRO_TOPIC_NAMES[i]:<40} "
              f"({len(seed_list)} seeds)")

    return np.array(centroids)


# ══════════════════════════════════════════════════════════════════════
# STAGE 3 — DIRECT COSINE SIMILARITY ASSIGNMENT
# ══════════════════════════════════════════════════════════════════════

def assign_macro_topics(
    embeddings: np.ndarray,
    seed_centroids: np.ndarray,
    threshold: float = SIMILARITY_THRESHOLD,
) -> tuple:
    """
    Assign each review to its closest macro topic using cosine similarity
    against the pre-computed seed centroids.

    This completely replaces BERTopic's HDBSCAN clustering step.
    Every review gets a deterministic, interpretable assignment:
    the macro topic whose seed centroid is semantically closest
    in the SBERT embedding space wins.

    The threshold is intentionally set very low (default 0.05) so
    General/Other is virtually never used. Only truly off-topic reviews
    (e.g. reviews written in a foreign language or containing only
    punctuation) will fall below it.

    Parameters
    ----------
    embeddings     : np.ndarray   shape (n_reviews, embedding_dim)
                                  pre-computed SBERT embeddings
    seed_centroids : np.ndarray   shape (n_topics, embedding_dim)
                                  output of build_seed_centroids()
    threshold      : float        minimum cosine similarity to assign
                                  a topic — reviews below this for ALL
                                  topics go to MACRO_OTHER

    Returns
    -------
    tuple (topic_ids, macro_labels, similarity_scores)
        topic_ids        : list of int   index into MACRO_TOPIC_NAMES, -1 = other
        macro_labels     : list of str   human-readable macro topic name
        similarity_scores: np.ndarray    shape (n_reviews,) best sim per review
    """
    # (n_reviews, n_topics) similarity matrix
    sims   = cosine_similarity(embeddings, seed_centroids)

    best_ids   = sims.argmax(axis=1)          # index of closest topic
    best_scores = sims.max(axis=1)            # similarity of that topic

    topic_ids    = []
    macro_labels = []

    for idx, score in zip(best_ids, best_scores):
        if score >= threshold:
            topic_ids.append(int(idx))
            macro_labels.append(MACRO_TOPIC_NAMES[int(idx)])
        else:
            topic_ids.append(-1)
            macro_labels.append(MACRO_OTHER)

    n_other = macro_labels.count(MACRO_OTHER)
    print(f"  ✓ Assigned          : {len(macro_labels) - n_other} reviews")
    print(f"  ✓ General / Other   : {n_other} reviews "
          f"({n_other / len(macro_labels) * 100:.1f}%)")

    dist = pd.Series(macro_labels).value_counts()
    print("\n  Distribution:")
    for name, cnt in dist.items():
        bar = '█' * int(cnt / len(macro_labels) * 40)
        print(f"    {name:<45} {cnt:>5}  {bar}")

    return topic_ids, macro_labels, best_scores


# ══════════════════════════════════════════════════════════════════════
# STAGE 4 — SENTIMENT
# ══════════════════════════════════════════════════════════════════════

def sentiment_from_stars(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive sentiment label directly from star ratings.

        4–5 stars → positive  (score 0.95)
        3 stars   → neutral   (score 0.70)
        1–2 stars → negative  (score 0.95)

    Parameters
    ----------
    df : pd.DataFrame   must contain a 'stars' column (int 1–5)

    Returns
    -------
    pd.DataFrame columns: sentiment, sentiment_score, sentiment_num
    """
    def _map(s):
        if s >= 4:   return ('positive', 0.95,  1)
        elif s == 3: return ('neutral',  0.70,  0)
        else:        return ('negative', 0.95, -1)

    mapped = df['stars'].apply(_map)
    return pd.DataFrame(
        mapped.tolist(),
        columns=['sentiment', 'sentiment_score', 'sentiment_num']
    )


def run_sentiment(
    texts: list,
    model_name: str = SENTIMENT_MODEL,
    batch_size: int = 32,
    max_length: int = 512,
) -> pd.DataFrame:
    """
    Run transformer sentiment classifier (positive / neutral / negative).
    Use sentiment_from_stars() instead for speed on CPU.

    Parameters
    ----------
    texts      : list of str
    model_name : str
    batch_size : int
    max_length : int

    Returns
    -------
    pd.DataFrame columns: sentiment, sentiment_score, sentiment_num
    """
    print(f"  Loading sentiment model '{model_name}' ...")
    classifier = hf_pipeline(
        "sentiment-analysis",
        model=model_name, tokenizer=model_name,
        truncation=True, max_length=max_length,
        batch_size=batch_size, device=-1
    )
    print(f"  Running sentiment on {len(texts):,} reviews ...")
    results = []
    for i in range(0, len(texts), batch_size):
        batch = [t if t.strip() else "neutral"
                 for t in texts[i: i + batch_size]]
        results.extend(classifier(batch))

    label_map = {'positive': 1, 'neutral': 0, 'negative': -1}
    df_sent = pd.DataFrame({
        'sentiment'      : [r['label'].lower() for r in results],
        'sentiment_score': [r['score']         for r in results],
    })
    df_sent['sentiment_num'] = df_sent['sentiment'].map(label_map)
    return df_sent


# ══════════════════════════════════════════════════════════════════════
# STAGE 5 — ASSEMBLE
# ══════════════════════════════════════════════════════════════════════

def assemble_results(
    df_source: pd.DataFrame,
    topic_ids: list,
    macro_labels: list,
    similarity_scores: np.ndarray,
    df_sentiment: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge original reviews with macro topic assignments and sentiment.

    Parameters
    ----------
    df_source         : pd.DataFrame   original review subset
    topic_ids         : list of int    macro topic index (-1 = other)
    macro_labels      : list of str    macro topic names
    similarity_scores : np.ndarray     best cosine sim per review
    df_sentiment      : pd.DataFrame   output of sentiment function

    Returns
    -------
    pd.DataFrame with original columns plus:
        macro_topic       : str    human-readable macro topic name
        topic_similarity  : float  cosine similarity to assigned topic
        sentiment         : str
        sentiment_score   : float
        sentiment_num     : int
    """
    df = df_source.copy()
    df['macro_topic']      = macro_labels
    df['topic_similarity'] = similarity_scores
    df['sentiment']        = df_sentiment['sentiment'].values
    df['sentiment_score']  = df_sentiment['sentiment_score'].values
    df['sentiment_num']    = df_sentiment['sentiment_num'].values
    return df


# ══════════════════════════════════════════════════════════════════════
# STAGE 6 — AGGREGATION
# ══════════════════════════════════════════════════════════════════════

def macro_topic_summary(
    df: pd.DataFrame,
    exclude_other: bool = False,
) -> pd.DataFrame:
    """
    Aggregate sentiment metrics at the macro topic level.

    Parameters
    ----------
    df            : pd.DataFrame   output of assemble_results()
    exclude_other : bool           drop MACRO_OTHER rows if True

    Returns
    -------
    pd.DataFrame sorted by net_sentiment descending, columns:
        macro_topic, n_reviews, pct_positive, pct_neutral,
        pct_negative, net_sentiment, dominant_sentiment,
        avg_similarity
    """
    if exclude_other:
        df = df[df['macro_topic'] != MACRO_OTHER].copy()

    agg = df.groupby('macro_topic').apply(lambda g: pd.Series({
        'n_reviews'     : len(g),
        'pct_positive'  : (g['sentiment'] == 'positive').mean() * 100,
        'pct_neutral'   : (g['sentiment'] == 'neutral').mean()  * 100,
        'pct_negative'  : (g['sentiment'] == 'negative').mean() * 100,
        'avg_similarity': g['topic_similarity'].mean(),
    })).reset_index()

    agg['net_sentiment']      = agg['pct_positive'] - agg['pct_negative']
    agg['dominant_sentiment'] = (
        agg[['pct_positive', 'pct_neutral', 'pct_negative']]
        .idxmax(axis=1)
        .str.replace('pct_', '')
    )

    return agg.sort_values('net_sentiment', ascending=False).reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════
# STAGE 7 — VISUALIZATIONS
# ══════════════════════════════════════════════════════════════════════

def plot_macro_distribution(df: pd.DataFrame, title: str) -> None:
    """
    Horizontal bar chart of macro topic review counts, colored by
    dominant sentiment. Net sentiment and count annotated per bar.

    Parameters
    ----------
    df    : pd.DataFrame   output of assemble_results()
    title : str
    """
    summary = macro_topic_summary(df, exclude_other=False)\
                  .sort_values('n_reviews')

    color_map = {'positive': '#2ca02c', 'neutral': '#bcbd22',
                 'negative': '#d62728'}
    colors = [color_map.get(s, 'gray') for s in summary['dominant_sentiment']]

    fig, ax = plt.subplots(figsize=(11, max(4, len(summary) * 0.6)))
    bars = ax.barh(summary['macro_topic'], summary['n_reviews'],
                   color=colors, edgecolor='white')

    for bar, row in zip(bars, summary.itertuples()):
        ax.text(
            bar.get_width() + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"net={row.net_sentiment:+.0f}  ({row.n_reviews} reviews)",
            va='center', fontsize=8.5
        )

    ax.legend(handles=[
        mpatches.Patch(color='#2ca02c', label='Dominant: positive'),
        mpatches.Patch(color='#bcbd22', label='Dominant: neutral'),
        mpatches.Patch(color='#d62728', label='Dominant: negative'),
    ], fontsize=9)
    ax.set_xlabel("Number of Reviews")
    ax.set_title(f"Macro Topic Distribution — {title}",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_macro_heatmap(
    df_target: pd.DataFrame,
    df_comp: pd.DataFrame,
    target_name: str,
) -> None:
    """
    Side-by-side heatmaps: pos / neutral / neg % per macro topic.
    Shared row order (target net sentiment desc) for easy comparison.

    Parameters
    ----------
    df_target   : pd.DataFrame
    df_comp     : pd.DataFrame
    target_name : str
    """
    s_t = macro_topic_summary(df_target, exclude_other=False)\
              .set_index('macro_topic')
    s_c = macro_topic_summary(df_comp,   exclude_other=False)\
              .set_index('macro_topic')

    row_order = s_t.sort_values('net_sentiment', ascending=False)\
                   .index.tolist()
    for t in s_c.index:
        if t not in row_order:
            row_order.append(t)

    cols       = ['pct_positive', 'pct_neutral', 'pct_negative']
    col_labels = ['Positive %',   'Neutral %',   'Negative %']

    def _heat(s, order):
        return s.reindex(order)[cols]\
                .rename(columns=dict(zip(cols, col_labels)))\
                .fillna(0)

    heat_t = _heat(s_t, row_order)
    heat_c = _heat(s_c, row_order)

    fig, axes = plt.subplots(
        1, 2, figsize=(16, max(4, len(row_order) * 0.6))
    )
    for ax, heat, title in zip(
        axes, [heat_t, heat_c], [target_name, "Competitors avg"]
    ):
        sns.heatmap(
            heat, annot=True, fmt='.0f', cmap='RdYlGn',
            vmin=0, vmax=100, linewidths=0.5, ax=ax,
            cbar_kws={'label': '% of reviews', 'shrink': 0.6}
        )
        ax.set_title(f"Macro Topic × Sentiment — {title}",
                     fontsize=12, fontweight='bold')
        ax.set_ylabel("")

    plt.tight_layout()
    plt.show()


def plot_macro_head_to_head(
    df_target: pd.DataFrame,
    df_comp: pd.DataFrame,
    target_name: str,
) -> pd.DataFrame:
    """
    Grouped bar chart: net_sentiment per macro topic, target vs
    competitors. Gap Δ annotated and color-coded green/red.

    Parameters
    ----------
    df_target   : pd.DataFrame
    df_comp     : pd.DataFrame
    target_name : str

    Returns
    -------
    pd.DataFrame   merged comparison with gap column
    """
    s_t = macro_topic_summary(df_target, exclude_other=True)\
              [['macro_topic', 'net_sentiment', 'n_reviews']]
    s_c = macro_topic_summary(df_comp,   exclude_other=True)\
              [['macro_topic', 'net_sentiment', 'n_reviews']]

    merged = s_t.merge(s_c, on='macro_topic',
                       suffixes=('_target', '_comp'))
    merged['gap'] = merged['net_sentiment_target'] - merged['net_sentiment_comp']
    merged = merged.sort_values('gap')

    fig, ax = plt.subplots(figsize=(12, max(5, len(merged) * 0.7)))
    y, h = np.arange(len(merged)), 0.35

    ax.barh(y + h/2, merged['net_sentiment_target'], h,
            color='#1f77b4', label=target_name, edgecolor='white')
    ax.barh(y - h/2, merged['net_sentiment_comp'],   h,
            color='#aec7e8', label='Competitors avg', edgecolor='white')

    for i, row in enumerate(merged.itertuples()):
        x_pos = max(row.net_sentiment_target, row.net_sentiment_comp) + 1
        color = '#2ca02c' if row.gap > 0 else '#d62728'
        ax.text(x_pos, i, f"Δ{row.gap:+.0f}",
                va='center', fontsize=8, color=color, fontweight='bold')

    ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_yticks(y)
    ax.set_yticklabels(merged['macro_topic'], fontsize=10)
    ax.set_xlabel("Net Sentiment (% Positive − % Negative)")
    ax.set_title(
        f"Macro Topic Net Sentiment — {target_name} vs Competitors\n"
        "green Δ = we outperform  |  red Δ = we underperform",
        fontsize=12, fontweight='bold'
    )
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.show()

    return merged


def print_macro_strengths_weaknesses(
    merged: pd.DataFrame,
    target_name: str,
) -> None:
    """
    Print ranked strengths, weaknesses, and improvement priorities.

    Parameters
    ----------
    merged      : pd.DataFrame   output of plot_macro_head_to_head()
    target_name : str
    """
    strengths  = merged[merged['gap'] > 0].sort_values('gap', ascending=False)
    weaknesses = merged[merged['gap'] < 0].sort_values('gap')

    print(f"\n{'='*65}")
    print(f"  MACRO STRENGTHS — {target_name} outperforms competitors")
    print(f"{'='*65}")
    for _, row in strengths.iterrows():
        print(f"  ✅  {row['macro_topic']:<40}  "
              f"net={row['net_sentiment_target']:+.0f}  gap=+{row['gap']:.0f}")

    print(f"\n{'='*65}")
    print(f"  MACRO WEAKNESSES — {target_name} underperforms competitors")
    print(f"{'='*65}")
    for _, row in weaknesses.iterrows():
        print(f"  ⚠️   {row['macro_topic']:<40}  "
              f"net={row['net_sentiment_target']:+.0f}  gap={row['gap']:.0f}")

    print(f"\n{'='*65}")
    print("  PRIORITY IMPROVEMENT AREAS (lowest net sentiment in target)")
    print(f"{'='*65}")
    for _, row in merged.sort_values('net_sentiment_target').iterrows():
        flag = ('🔴' if row['net_sentiment_target'] < -25 else
                '🟡' if row['net_sentiment_target'] < 0   else '🟢')
        print(f"  {flag}  {row['macro_topic']:<40}  "
              f"net={row['net_sentiment_target']:+.0f}")


# ══════════════════════════════════════════════════════════════════════
# STAGE 8 — ROOT CAUSE
# ══════════════════════════════════════════════════════════════════════

def get_weak_macro_topics(
    merged: pd.DataFrame,
    min_gap: float = -5.0,
) -> list:
    """
    Return macro topics where target underperforms competitors by
    more than min_gap points, sorted worst first.

    Parameters
    ----------
    merged  : pd.DataFrame
    min_gap : float

    Returns
    -------
    list of str
    """
    return (merged[merged['gap'] < min_gap]
            .sort_values('gap')['macro_topic']
            .tolist())


def extract_negative_reviews(
    df_target: pd.DataFrame,
    macro_topic: str,
    sentiment_filter: str = 'negative',
) -> pd.DataFrame:
    """
    Extract raw reviews for a macro topic filtered by sentiment,
    sorted by stars ascending (worst first).

    Parameters
    ----------
    df_target        : pd.DataFrame
    macro_topic      : str
    sentiment_filter : str   'negative' | 'neutral' | 'positive' | 'all'

    Returns
    -------
    pd.DataFrame
    """
    mask = df_target['macro_topic'] == macro_topic
    if sentiment_filter != 'all':
        mask = mask & (df_target['sentiment'] == sentiment_filter)

    cols      = ['company', 'stars', 'macro_topic', 'topic_similarity',
                 'sentiment', 'sentiment_score', 'title', 'review']
    available = [c for c in cols if c in df_target.columns]

    return (df_target[mask][available]
            .sort_values('stars', ascending=True)
            .reset_index(drop=True))


def root_cause_report(
    df_target: pd.DataFrame,
    df_comp: pd.DataFrame,
    merged: pd.DataFrame,
    target_name: str,
    min_gap: float = -5.0,
    max_reviews_shown: int = 5,
) -> dict:
    """
    Root cause analysis for all weak macro topics.

    For each topic prints:
        1. Performance gap vs competitors
        2. Raw negative reviews verbatim
        3. Word frequency comparison: target vs competitors

    Parameters
    ----------
    df_target         : pd.DataFrame
    df_comp           : pd.DataFrame
    merged            : pd.DataFrame
    target_name       : str
    min_gap           : float
    max_reviews_shown : int

    Returns
    -------
    dict {macro_topic: DataFrame of negative reviews}
    """
    weak_topics = get_weak_macro_topics(merged, min_gap=min_gap)

    if not weak_topics:
        print("✅ No macro topics underperform below the threshold.")
        return {}

    all_negative_reviews = {}

    for macro in weak_topics:
        if len(df_target[df_target['macro_topic']==macro])<10:
            continue
        row = merged[merged['macro_topic'] == macro].iloc[0]

        print(f"\n{'█'*65}")
        print(f"  ROOT CAUSE — {macro}")
        print(f"{'█'*65}")
        print(f"  {target_name} net sentiment : {row['net_sentiment_target']:+.0f}")
        print(f"  Competitors net sentiment   : {row['net_sentiment_comp']:+.0f}")
        print(f"  Gap                         : {row['gap']:+.0f}")

        neg_reviews = extract_negative_reviews(
            df_target, macro, sentiment_filter='negative'
        )
        all_negative_reviews[macro] = neg_reviews

        # print(f"\n  Negative reviews: {len(neg_reviews)} total — "
        #       f"showing {min(max_reviews_shown, len(neg_reviews))}")

        # for i, rev in neg_reviews.head(max_reviews_shown).iterrows():
        #     if rev.get('topic_similarity', 0) < 0.3:
        #         continue
        #     stars_str = '★' * int(rev['stars']) + '☆' * (5 - int(rev['stars']))
        #     sim_str   = f"(sim={rev.get('topic_similarity', 0):.2f})"
        #     print(f"\n  [{i+1}] {stars_str}  {sim_str}  |  "
        #           f"{rev.get('title', '')}")
        #     print(f"  Review : {str(rev['review'])[:500]}"
        #           f"{'...' if len(str(rev['review'])) > 500 else ''}")

        _plot_word_gap(df_target, df_comp, macro, target_name)

    return all_negative_reviews


def _plot_word_gap(
    df_target: pd.DataFrame,
    df_comp: pd.DataFrame,
    macro_topic: str,
    target_name: str,
    top_n: int = 15,
) -> None:
    """
    Side-by-side word frequency in negative reviews: target vs
    competitors. Reveals the specific language driving dissatisfaction.

    Parameters
    ----------
    df_target   : pd.DataFrame
    df_comp     : pd.DataFrame
    macro_topic : str
    target_name : str
    top_n       : int
    """
    stop = {
        'the','and','for','was','are','has','had','have','been','with',
        'this','that','they','from','not','but','its','our','your',
        'their','all','just','very','also','more','than','did','one',
        'got','get','can','it','is','in','to','of','a','an','on','at',
        'be','my','we','i','me','he','she','us','you','so','if','or',
        'as','by','do','no','up','out','his','her','him','who','what',
        'how','would','could','should','still','even','then','when',
    }

    def _freq(df, macro):
        neg = df[(df['macro_topic'] == macro) &
                 (df['sentiment'] == 'negative')]['review'].fillna('')
        counts = {}
        for text in neg:
            for w in str(text).lower().split():
                w = w.strip('.,!?:;"\'')
                if w.isalpha() and len(w) > 3 and w not in stop:
                    counts[w] = counts.get(w, 0) + 1
        return pd.Series(counts).sort_values(ascending=False).head(top_n)

    freq_t = _freq(df_target, macro_topic)
    freq_c = _freq(df_comp,   macro_topic)

    if freq_t.empty and freq_c.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, freq, label, color in zip(
        axes,
        [freq_t, freq_c],
        [f"{target_name}\nnegative reviews", "Competitors\nnegative reviews"],
        ['#d62728', '#ff7f7f']
    ):
        if freq.empty:
            ax.text(0.5, 0.5, 'No negative reviews', ha='center',
                    va='center', transform=ax.transAxes)
        else:
            freq_s = freq.sort_values()
            ax.barh(freq_s.index, freq_s.values,
                    color=color, edgecolor='white', alpha=0.85)
        ax.set_title(label, fontsize=11, fontweight='bold')
        ax.set_xlabel("Word frequency in negative reviews")

    fig.suptitle(
        f"Most Common Words in Negative Reviews — {macro_topic}",
        fontsize=12, fontweight='bold'
    )
    plt.tight_layout()
    plt.show()


# ══════════════════════════════════════════════════════════════════════
# STAGE 9 — FULL PIPELINE
# ══════════════════════════════════════════════════════════════════════

def run_macro_pipeline(
    df: pd.DataFrame,
    target: str,
    use_star_sentiment: bool = True,
    min_gap: float = -5.0,
    max_reviews_shown: int = 5,
    threshold: float = SIMILARITY_THRESHOLD,
) -> dict:
    """
    End-to-end macro topic + sentiment pipeline.

    BERTopic is no longer used. Assignment is done via direct cosine
    similarity between each review embedding and the macro topic seed
    centroids. This guarantees every review is assigned to a meaningful
    topic — General/Other only appears when similarity is below
    `threshold` for ALL topics (virtually never at threshold=0.05).

    Steps
    -----
    1.  Split into target / competitor subsets
    2.  Load SBERT; embed both corpora + build seed centroids
    3.  Assign macro topics via cosine similarity (competitors)
    4.  Assign macro topics via cosine similarity (target)
    5.  Sentiment from stars or transformer
    6.  Assemble DataFrames
    7.  Plots: distribution, heatmap, head-to-head
    8.  Root cause: negative reviews + word frequency

    Parameters
    ----------
    df                 : pd.DataFrame
    target             : str
    use_star_sentiment : bool
    min_gap            : float   root cause threshold
    max_reviews_shown  : int
    threshold          : float   min cosine sim to assign a topic
                                 (default 0.05 → near-zero General/Other)

    Returns
    -------
    dict with keys:
        df_target, df_comp, summary_target, summary_comp,
        seed_centroids, merged, negative_reviews
    """

    # ── Step 1 — Split ────────────────────────────────────────────────
    print("\n[1/6] Splitting dataset ...")
    df_target, df_comp = split_target_competitors(df, target)
    texts_target = df_target['review'].tolist()
    texts_comp   = df_comp['review'].tolist()

    # ── Step 2 — Embed + seed centroids ──────────────────────────────
    print("\n[2/6] Loading SBERT, embedding reviews & building centroids ...")
    sbert = SentenceTransformer(SBERT_MODEL)

    emb_comp   = sbert.encode(texts_comp,   batch_size=64,
                               show_progress_bar=True, convert_to_numpy=True)
    emb_target = sbert.encode(texts_target, batch_size=64,
                               show_progress_bar=True, convert_to_numpy=True)

    seed_centroids = build_seed_centroids(sbert)

    # ── Step 3 — Assign macro topics: competitors ─────────────────────
    print("\n[3/6] Assigning macro topics to competitor reviews ...")
    ids_comp, labels_comp, scores_comp = assign_macro_topics(
        emb_comp, seed_centroids, threshold=threshold
    )

    # ── Step 4 — Assign macro topics: target ─────────────────────────
    print("\n[4/6] Assigning macro topics to target reviews ...")
    ids_target, labels_target, scores_target = assign_macro_topics(
        emb_target, seed_centroids, threshold=threshold
    )

    # ── Step 5 — Sentiment ────────────────────────────────────────────
    print("\n[5/6] Computing sentiment ...")
    if use_star_sentiment:
        sent_target = sentiment_from_stars(df_target)
        sent_comp   = sentiment_from_stars(df_comp)
    else:
        sent_target = run_sentiment(texts_target)
        sent_comp   = run_sentiment(texts_comp)

    # ── Step 6 — Assemble ─────────────────────────────────────────────
    print("\n[6/6] Assembling result DataFrames ...")
    df_t = assemble_results(df_target, ids_target, labels_target,
                             scores_target, sent_target)
    df_c = assemble_results(df_comp,   ids_comp,   labels_comp,
                             scores_comp,   sent_comp)

    # ── Visualizations ────────────────────────────────────────────────
    print(f"\n{'─'*55}\n  VISUALIZATIONS\n{'─'*55}")

    # plot_macro_distribution(df_t, title=target)
    # plot_macro_distribution(df_c, title="Competitors")
    # plot_macro_heatmap(df_t, df_c, target_name=target)
    # merged = plot_macro_head_to_head(df_t, df_c, target_name=target)
    # print_macro_strengths_weaknesses(merged, target_name=target)

    # # ── Root cause ────────────────────────────────────────────────────
    # neg_reviews = root_cause_report(
    #     df_target=df_t, df_comp=df_c,
    #     merged=merged, target_name=target,
    #     min_gap=min_gap, max_reviews_shown=max_reviews_shown
    # )

    return {
        'df_target'       : df_t,
        'df_comp'         : df_c,
        'summary_target'  : macro_topic_summary(df_t),
        'summary_comp'    : macro_topic_summary(df_c),
        'seed_centroids'  : seed_centroids,   
        # 'merged'          : merged,
        # 'negative_reviews': neg_reviews,
    }
