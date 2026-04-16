import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from transformers import pipeline as hf_pipeline


SBERT_MODEL    = "all-MiniLM-L6-v2"          # fast, good quality
SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
MIN_TOPIC_SIZE  = 35                          # min reviews per topic
N_TOP_WORDS     = 10                           # words shown per topic
RANDOM_STATE    = 42


# ══════════════════════════════════════════════════════════════════════
# STAGE 2 — PREP: split target vs competitors
# ══════════════════════════════════════════════════════════════════════

def split_target_competitors(df: pd.DataFrame, target: str) -> tuple:
    """
    Split the cleaned category DataFrame into the target company subset
    and the competitor subset.

    Both subsets keep all original columns. Reviews that are empty after
    cleaning are dropped from both.

    Parameters
    ----------
    df     : pd.DataFrame  cleaned category DataFrame
    target : str           company name to analyse as 'ours'

    Returns
    -------
    tuple (df_target, df_competitors)
        df_target     : reviews belonging to the target company
        df_competitors: reviews from all other companies in the category
    """
    if target not in df['company'].values:
        raise ValueError(f"'{target}' not found. Check TARGET variable.")

    mask          = df['company'] == target
    df_target     = df[mask].copy().reset_index(drop=True)
    df_competitors = df[~mask].copy().reset_index(drop=True)

    # Drop empty reviews
    for d, name in [(df_target, 'target'), (df_competitors, 'competitors')]:
        empty = d['review'].fillna('').str.strip() == ''
        if empty.sum():
            print(f"  ⚠ Dropping {empty.sum()} empty reviews from {name}")

    df_target      = df_target[df_target['review'].fillna('').str.strip() != ''].reset_index(drop=True)
    df_competitors = df_competitors[df_competitors['review'].fillna('').str.strip() != ''].reset_index(drop=True)

    print(f"  Target '{target}': {len(df_target)} reviews")
    print(f"  Competitors ({df_competitors['company'].nunique()} companies): {len(df_competitors)} reviews")

    return df_target, df_competitors


# ══════════════════════════════════════════════════════════════════════
# STAGE 3 — EMBED: sentence embeddings with SBERT
# ══════════════════════════════════════════════════════════════════════

def embed_reviews(texts: list, model_name: str = SBERT_MODEL,
                  batch_size: int = 64, show_progress: bool = True) -> np.ndarray:
    """
    Generate sentence embeddings for a list of review texts using a
    SentenceTransformer model.

    The embeddings are used as input to BERTopic's dimensionality
    reduction + clustering pipeline.

    Parameters
    ----------
    texts        : list of str   cleaned review texts
    model_name   : str           HuggingFace SentenceTransformer model id
    batch_size   : int           number of texts encoded per GPU/CPU batch
    show_progress: bool          show tqdm progress bar

    Returns
    -------
    np.ndarray of shape (n_reviews, embedding_dim)
    """
    print(f"  Loading embedding model '{model_name}' ...")
    model = SentenceTransformer(model_name)
    print(f"  Embedding {len(texts):,} reviews ...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True
    )
    print(f"  ✓ Embeddings shape: {embeddings.shape}")
    return embeddings


# ══════════════════════════════════════════════════════════════════════
# STAGE 4 — TOPIC MODELLING with BERTopic
# ══════════════════════════════════════════════════════════════════════

def build_topic_model(
    sbert_model_name: str = SBERT_MODEL,
    min_topic_size: int   = MIN_TOPIC_SIZE,
    n_top_words: int      = N_TOP_WORDS,
    random_state: int     = RANDOM_STATE,
) -> BERTopic:
    """
    Build a configured BERTopic model with the SBERT embedding model
    passed explicitly.

    Passing the embedding model to BERTopic is required when using
    KeyBERTInspired as the representation model, because KeyBERT needs
    to re-embed vocabulary words internally during topic refinement.
    Without it, topic_model.embedding_model is None and the call to
    embed_documents() raises AttributeError.

    When we call fit_transform() we still pass our pre-computed
    embeddings so documents are NOT re-embedded — SBERT is only invoked
    by KeyBERT for the small vocabulary embedding step.

    Parameters
    ----------
    sbert_model_name : str   HuggingFace SentenceTransformer model id
    min_topic_size   : int   minimum reviews to form a topic
    n_top_words      : int   representative words shown per topic
    random_state     : int   reproducibility seed

    Returns
    -------
    BERTopic model (not yet fitted)
    """
    from bertopic.backend import BaseEmbedder
    from sentence_transformers import SentenceTransformer

    # BERTopic wraps SBERT automatically when you pass a string model name
    # but we build it explicitly so we can reuse the already-loaded model
    sbert = SentenceTransformer(sbert_model_name)

    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric='cosine',
        random_state=random_state
    )

    hdbscan_model = HDBSCAN(
        min_cluster_size=min_topic_size,
        min_samples=1,
        metric='euclidean',
        cluster_selection_method='eom',
        prediction_data=True
    )

    vectorizer_model = CountVectorizer(
        ngram_range=(1, 2),
        stop_words='english',
        min_df=2
    )

    representation_model = KeyBERTInspired()

    topic_model = BERTopic(
        embedding_model=sbert,      
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        representation_model=representation_model,
        top_n_words=n_top_words,
        verbose=True
    )

    return topic_model



def fit_topics(
    texts: list,
    embeddings: np.ndarray,
    min_topic_size: int = MIN_TOPIC_SIZE,
) -> tuple:
    """
    Fit BERTopic on a corpus using pre-computed embeddings.

    Passing embeddings to fit_transform() prevents BERTopic from
    re-embedding documents (expensive), while still giving KeyBERT
    access to the internal SBERT model for vocabulary embedding.

    Parameters
    ----------
    texts          : list         review strings (same order as embeddings)
    embeddings     : np.ndarray   pre-computed SBERT embeddings
    min_topic_size : int          passed to build_topic_model()

    Returns
    -------
    tuple (topic_model, topics, probs, topic_info)
    """
    topic_model = build_topic_model(min_topic_size=min_topic_size)

    # Pass pre-computed embeddings → documents are not re-embedded
    # BERTopic uses them for UMAP + HDBSCAN only
    topics, probs = topic_model.fit_transform(
        documents=texts,
        embeddings=embeddings   # ← pre-computed, skips internal embedding step
    )

    topic_info = topic_model.get_topic_info()
    n_topics   = len(topic_info[topic_info['Topic'] != -1])
    n_outliers = sum(1 for t in topics if t == -1)

    print(f"\n  ✓ Topics found      : {n_topics}")
    print(f"  ✓ Outlier reviews   : {n_outliers} ({n_outliers/len(texts)*100:.1f}%)")

    return topic_model, topics, probs, topic_info


def get_topic_labels(topic_model: BERTopic) -> dict:
    """
    Build a human-readable label for each topic from its top keywords.

    Label format: "Topic N: word1, word2, word3"
    Topic -1 gets the label "Outliers / uncategorised".

    Parameters
    ----------
    topic_model : fitted BERTopic

    Returns
    -------
    dict {topic_id (int): label (str)}
    """
    labels = {-1: "Outliers / uncategorised"}
    for topic_id in topic_model.get_topic_freq()['Topic']:
        if topic_id == -1:
            continue
        words = [w for w, _ in topic_model.get_topic(topic_id)]
        labels[topic_id] = f"T{topic_id}: {', '.join(words[:4])}"
    return labels


# ══════════════════════════════════════════════════════════════════════
# STAGE 5 — SENTIMENT ANALYSIS per review
# ══════════════════════════════════════════════════════════════════════

def run_sentiment(
    texts: list,
    model_name: str = SENTIMENT_MODEL,
    batch_size: int = 32,
    max_length: int = 512,
) -> pd.DataFrame:
    """
    Run a pre-trained transformer sentiment classifier on a list of reviews.

    Model used by default: cardiffnlp/twitter-roberta-base-sentiment-latest
    Labels returned: 'positive', 'neutral', 'negative'

    Each text is truncated to max_length tokens before inference.
    Empty strings receive label='neutral', score=0.0 as a safe default.

    Parameters
    ----------
    texts      : list of str   cleaned review texts
    model_name : str           HuggingFace model id for sentiment
    batch_size : int           inference batch size
    max_length : int           max token length before truncation

    Returns
    -------
    pd.DataFrame with columns:
        sentiment       : str  'positive' | 'neutral' | 'negative'
        sentiment_score : float confidence score (0–1)
        sentiment_num   : int   1 (positive), 0 (neutral), -1 (negative)
    """
    print(f"  Loading sentiment model '{model_name}' ...")
    classifier = hf_pipeline(
        "sentiment-analysis",
        model=model_name,
        tokenizer=model_name,
        truncation=True,
        max_length=max_length,
        batch_size=batch_size,
        device=-1           # CPU; set to 0 for GPU
    )

    print(f"  Running sentiment on {len(texts):,} reviews ...")
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        # Replace empty strings to avoid tokenizer errors
        batch = [t if t.strip() else "neutral" for t in batch]
        preds = classifier(batch)
        results.extend(preds)

    label_map = {'positive': 1, 'neutral': 0, 'negative': -1}

    df_sent = pd.DataFrame({
        'sentiment'      : [r['label'].lower() for r in results],
        'sentiment_score': [r['score']         for r in results],
    })
    df_sent['sentiment_num'] = df_sent['sentiment'].map(label_map)

    dist = df_sent['sentiment'].value_counts(normalize=True) * 100
    print("  ✓ Sentiment distribution:")
    for lbl, pct in dist.items():
        print(f"      {lbl:<10}: {pct:.1f}%")

    return df_sent


# ══════════════════════════════════════════════════════════════════════
# STAGE 6 — ASSEMBLE: merge topics + sentiment into one DataFrame
# ══════════════════════════════════════════════════════════════════════

def assemble_results(
    df_source: pd.DataFrame,
    topics: list,
    df_sentiment: pd.DataFrame,
    topic_labels: dict,
) -> pd.DataFrame:
    """
    Merge the original review DataFrame with topic assignments and
    sentiment predictions into a single analysis DataFrame.

    Parameters
    ----------
    df_source    : pd.DataFrame   original reviews (target or competitor subset)
    topics       : list of int    BERTopic topic IDs (same order as df_source)
    df_sentiment : pd.DataFrame   output of run_sentiment()
    topic_labels : dict           {topic_id: label_str}

    Returns
    -------
    pd.DataFrame with all original columns plus:
        topic_id      : int    BERTopic topic assignment
        topic_label   : str    human-readable topic label
        sentiment     : str    'positive' | 'neutral' | 'negative'
        sentiment_score: float confidence (0–1)
        sentiment_num : int    1 / 0 / −1
    """
    df = df_source.copy()
    df['topic_id']       = topics
    df['topic_label']    = [topic_labels.get(t, f"T{t}") for t in topics]
    df['sentiment']      = df_sentiment['sentiment'].values
    df['sentiment_score'] = df_sentiment['sentiment_score'].values
    df['sentiment_num']  = df_sentiment['sentiment_num'].values
    return df


# ══════════════════════════════════════════════════════════════════════
# STAGE 7 — TOPIC-SENTIMENT AGGREGATION
# ══════════════════════════════════════════════════════════════════════

def topic_sentiment_summary(df: pd.DataFrame, exclude_outliers: bool = True) -> pd.DataFrame:
    """
    Aggregate sentiment counts and percentages per topic.

    Returns one row per topic with:
        topic_id      : int
        topic_label   : str
        n_reviews     : total reviews in topic
        pct_positive  : % positive
        pct_neutral   : % neutral
        pct_negative  : % negative
        dominant_sentiment : the sentiment with the highest %
        net_sentiment : pct_positive − pct_negative  (−100 to +100)

    Parameters
    ----------
    df               : pd.DataFrame   output of assemble_results()
    exclude_outliers : bool           if True, drop topic_id == -1

    Returns
    -------
    pd.DataFrame sorted by net_sentiment descending
    """
    if exclude_outliers:
        df = df[df['topic_id'] != -1].copy()

    agg = df.groupby(['topic_id', 'topic_label']).apply(lambda g: pd.Series({
        'n_reviews'   : len(g),
        'pct_positive': (g['sentiment'] == 'positive').mean() * 100,
        'pct_neutral' : (g['sentiment'] == 'neutral').mean()  * 100,
        'pct_negative': (g['sentiment'] == 'negative').mean() * 100,
    })).reset_index()

    agg['net_sentiment']      = agg['pct_positive'] - agg['pct_negative']
    agg['dominant_sentiment'] = agg[['pct_positive', 'pct_neutral', 'pct_negative']].idxmax(axis=1)\
                                    .str.replace('pct_', '')

    return agg.sort_values('net_sentiment', ascending=False).reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════
# STAGE 8 — VISUALIZATIONS
# ══════════════════════════════════════════════════════════════════════

# ── 8.1 Topic distribution bar ─────────────────────────────────────────

def plot_topic_distribution(df: pd.DataFrame, title: str, top_n: int = 15) -> None:
    """
    Horizontal bar chart of the most common topics by review count,
    colored by dominant sentiment (green = positive, red = negative).

    Parameters
    ----------
    df    : pd.DataFrame   output of assemble_results()
    title : str            chart title (e.g. company name)
    top_n : int            max topics to show
    """
    summary = topic_sentiment_summary(df)
    summary = summary.head(top_n).sort_values('n_reviews')

    color_map = {'positive': '#2ca02c', 'neutral': '#bcbd22', 'negative': '#d62728'}
    colors = [color_map.get(s, 'gray') for s in summary['dominant_sentiment']]

    fig, ax = plt.subplots(figsize=(11, max(4, len(summary) * 0.45)))
    bars = ax.barh(summary['topic_label'], summary['n_reviews'], color=colors, edgecolor='white')

    for bar, row in zip(bars, summary.itertuples()):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"net={row.net_sentiment:+.0f}", va='center', fontsize=8)

    legend_handles = [
        mpatches.Patch(color='#2ca02c', label='Dominant: positive'),
        mpatches.Patch(color='#bcbd22', label='Dominant: neutral'),
        mpatches.Patch(color='#d62728', label='Dominant: negative'),
    ]
    ax.legend(handles=legend_handles, fontsize=9)
    ax.set_xlabel("Number of Reviews")
    ax.set_title(f"Topic Distribution — {title}", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()


# ── 8.2 Sentiment heatmap per topic ───────────────────────────────────

def plot_sentiment_heatmap(summary: pd.DataFrame, title: str) -> None:
    """
    Heatmap where rows = topics, columns = positive/neutral/negative %,
    sorted by net_sentiment so the best topics are at the top.

    Parameters
    ----------
    summary : pd.DataFrame   output of topic_sentiment_summary()
    title   : str            chart title
    """
    heat_df = summary.set_index('topic_label')[['pct_positive', 'pct_neutral', 'pct_negative']]
    heat_df.columns = ['Positive %', 'Neutral %', 'Negative %']

    fig, ax = plt.subplots(figsize=(9, max(4, len(heat_df) * 0.42)))
    sns.heatmap(
        heat_df, annot=True, fmt='.0f', cmap='RdYlGn',
        vmin=0, vmax=100, linewidths=0.5, ax=ax,
        cbar_kws={'label': '% of reviews', 'shrink': 0.7}
    )
    ax.set_title(f"Topic × Sentiment Heatmap — {title}", fontsize=13, fontweight='bold')
    ax.set_ylabel("")
    plt.tight_layout()
    plt.show()


# ── 8.3 Head-to-head: TARGET vs COMPETITORS (shared topics) ──────────

def plot_head_to_head(
    summary_target: pd.DataFrame,
    summary_comp: pd.DataFrame,
    target_name: str,
    top_n: int = 12,
) -> pd.DataFrame:
    """
    Side-by-side grouped bar chart comparing net_sentiment per topic
    between the target company and competitors.

    Only topics present in BOTH summaries are shown (inner join on
    topic_label). Sorted by the gap (target − competitor) to immediately
    surface strengths and weaknesses.

    Parameters
    ----------
    summary_target : pd.DataFrame   topic_sentiment_summary() for target
    summary_comp   : pd.DataFrame   topic_sentiment_summary() for competitors
    target_name    : str            company display name
    top_n          : int            number of topics to show

    Returns
    -------
    pd.DataFrame   merged comparison table (also printed)
    """
    merged = summary_target[['topic_label', 'net_sentiment', 'n_reviews']]\
        .merge(
            summary_comp[['topic_label', 'net_sentiment', 'n_reviews']],
            on='topic_label', suffixes=('_target', '_comp')
        )

    merged['gap'] = merged['net_sentiment_target'] - merged['net_sentiment_comp']
    merged = merged.reindex(merged['gap'].abs().nlargest(top_n).index)
    merged = merged.sort_values('gap')

    fig, ax = plt.subplots(figsize=(12, max(5, len(merged) * 0.55)))
    y = np.arange(len(merged))
    bar_h = 0.35

    ax.barh(y + bar_h / 2, merged['net_sentiment_target'], bar_h,
            color='#1f77b4', label=target_name, edgecolor='white')
    ax.barh(y - bar_h / 2, merged['net_sentiment_comp'], bar_h,
            color='#aec7e8', label='Competitors avg', edgecolor='white')

    ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_yticks(y)
    ax.set_yticklabels(merged['topic_label'], fontsize=9)
    ax.set_xlabel("Net Sentiment (% Positive − % Negative)")
    ax.set_title(
        f"Net Sentiment per Topic — {target_name} vs Competitors\n"
        "(topics sorted by largest gap)",
        fontsize=12, fontweight='bold'
    )
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.show()

    return merged


# ── 8.4 Strengths & weaknesses summary table ──────────────────────────

def print_strengths_weaknesses(merged: pd.DataFrame, target_name: str, top_n: int = 5) -> None:
    """
    Print a plain-text summary of the top strengths (topics where target
    outperforms competitors) and top weaknesses (topics where target
    underperforms).

    Parameters
    ----------
    merged      : pd.DataFrame   output of plot_head_to_head()
    target_name : str
    top_n       : int            number of topics in each list
    """
    strengths  = merged[merged['gap'] > 0].nlargest(top_n, 'gap')
    weaknesses = merged[merged['gap'] < 0].nsmallest(top_n, 'gap')

    print(f"\n{'='*60}")
    print(f"  STRENGTHS — {target_name} outperforms competitors")
    print(f"{'='*60}")
    for _, row in strengths.iterrows():
        print(f"  ✅  {row['topic_label'][:55]:<55}  gap = +{row['gap']:.0f}")

    print(f"\n{'='*60}")
    print(f"  WEAKNESSES — {target_name} underperforms competitors")
    print(f"{'='*60}")
    for _, row in weaknesses.iterrows():
        print(f"  ⚠️   {row['topic_label'][:55]:<55}  gap = {row['gap']:.0f}")

    print(f"\n{'='*60}")
    print("  IMPROVEMENT AREAS (most negative net sentiment in target)")
    print(f"{'='*60}")
    improvement = merged.nsmallest(top_n, 'net_sentiment_target')
    for _, row in improvement.iterrows():
        print(f"  🔧  {row['topic_label'][:55]:<55}  net = {row['net_sentiment_target']:.0f}")


# ══════════════════════════════════════════════════════════════════════
# STAGE 9 — FINAL PIPELINE
# ══════════════════════════════════════════════════════════════════════

def run_nlp_pipeline(df: pd.DataFrame, target: str) -> dict:
    """
    Execute the complete NLP topic + sentiment pipeline.

    Steps
    -----
    1.  Split df into target and competitor subsets
    2.  Embed both corpora with SBERT
    3.  Fit BERTopic on COMPETITORS (larger corpus = better topics)
    4.  Transform TARGET reviews using the same fitted model
        (so topics are comparable across both groups)
    5.  Run sentiment analysis on both subsets
    6.  Assemble results DataFrames

    Parameters
    ----------
    df     : pd.DataFrame   cleaned category DataFrame (all companies)
    target : str            company to analyse as 'ours'

    Returns
    -------
    dict with keys:
        'df_target'        : reviews + topics + sentiment for target
        'df_competitors'   : reviews + topics + sentiment for competitors
        'summary_target'   : topic-sentiment summary for target
        'summary_comp'     : topic-sentiment summary for competitors
        'topic_model'      : fitted BERTopic instance
        'topic_labels'     : dict {id: label}
    """
    # ── Step 1 — Split ────────────────────────────────────────────────
    print("\n[1/7] Splitting dataset ...")
    df_target, df_comp = split_target_competitors(df, target)

    texts_target = df_target['review'].tolist()
    texts_comp   = df_comp['review'].tolist()

    # ── Step 2 — Load SBERT once, embed both corpora ─────────────────
    print("\n[2/7] Loading SBERT and embedding reviews ...")
    from sentence_transformers import SentenceTransformer
    sbert = SentenceTransformer(SBERT_MODEL)

    emb_comp   = sbert.encode(texts_comp,   batch_size=64,
                               show_progress_bar=True, convert_to_numpy=True)
    emb_target = sbert.encode(texts_target, batch_size=64,
                               show_progress_bar=True, convert_to_numpy=True)
    print(f"  ✓ Competitor embeddings : {emb_comp.shape}")
    print(f"  ✓ Target embeddings     : {emb_target.shape}")

    # ── Step 3 — Fit topics on competitors (reuse loaded sbert) ──────
    print("\n[3/7] Fitting BERTopic on competitors ...")

    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0,
                      metric='cosine', random_state=RANDOM_STATE)
    hdbscan_model = HDBSCAN(min_cluster_size=MIN_TOPIC_SIZE, min_samples=1,
                             metric='euclidean', cluster_selection_method='eom',
                             prediction_data=True)
    vectorizer_model = CountVectorizer(ngram_range=(1, 2), stop_words='english', min_df=2)

    topic_model = BERTopic(
        embedding_model=sbert,                      
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        representation_model=KeyBERTInspired(),
        top_n_words=N_TOP_WORDS,
        verbose=True
    )

    topics_comp, _ = topic_model.fit_transform(documents=texts_comp, embeddings=emb_comp)

    topic_info  = topic_model.get_topic_info()
    topic_labels = get_topic_labels(topic_model)

    print("\n  Top topics discovered:")
    print(topic_info[topic_info['Topic'] != -1][['Topic','Count','Name']].head(15).to_string(index=False))

    # ── Step 4 — Transform target with same model ─────────────────────
    print("\n[4/7] Assigning topics to target reviews ...")
    topics_target, _ = topic_model.transform(texts_target, emb_target)

    # ── Step 5 — Sentiment ────────────────────────────────────────────
    print("\n[5/7] Running sentiment on target ...")
    sent_target = run_sentiment(texts_target)

    print("\n  Running sentiment on competitors ...")
    sent_comp = run_sentiment(texts_comp)

    # ── Step 6 — Assemble ─────────────────────────────────────────────
    print("\n[6/7] Assembling result DataFrames ...")
    df_target_full = assemble_results(df_target, topics_target, sent_target, topic_labels)
    df_comp_full   = assemble_results(df_comp,   topics_comp,   sent_comp,   topic_labels)

    # ── Step 7 — Summaries ────────────────────────────────────────────
    print("\n[7/7] Building topic-sentiment summaries ...")
    summary_target = topic_sentiment_summary(df_target_full)
    summary_comp   = topic_sentiment_summary(df_comp_full)


    return df_target_full, df_comp_full, summary_target, summary_comp, topic_model