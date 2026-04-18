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
MIN_TOPIC_SIZE  = 10                       # min reviews per topic
N_TOP_WORDS     = 15                           # words shown per topic
RANDOM_STATE    = 42


# ══════════════════════════════════════════════════════════════════════
# STAGE 1 — PREP: split target vs competitors
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
# STAGE 2 — EMBED: sentence embeddings with SBERT
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
# STAGE 3 — TOPIC MODELLING with BERTopic
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
# STAGE 4 — SENTIMENT ANALYSIS per review
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
# STAGE 5 — ASSEMBLE: merge topics + sentiment into one DataFrame
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
# STAGE 6 — PREPROCESSING PIPELINE
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
        'topic_model'      : fitted BERTopic instance
        'topic_labels'     : dict {id: label}
    """
    # ── Step 1 — Split ────────────────────────────────────────────────
    print("\n[1/6] Splitting dataset ...")
    df_target, df_comp = split_target_competitors(df, target)

    texts_target = df_target['review'].tolist()
    texts_comp   = df_comp['review'].tolist()

    # ── Step 2 — Load SBERT once, embed both corpora ─────────────────
    print("\n[2/6] Loading SBERT and embedding reviews ...")
    from sentence_transformers import SentenceTransformer
    sbert = SentenceTransformer(SBERT_MODEL)

    emb_comp   = sbert.encode(texts_comp,   batch_size=64,
                               show_progress_bar=True, convert_to_numpy=True)
    emb_target = sbert.encode(texts_target, batch_size=64,
                               show_progress_bar=True, convert_to_numpy=True)
    print(f"  ✓ Competitor embeddings : {emb_comp.shape}")
    print(f"  ✓ Target embeddings     : {emb_target.shape}")

    # ── Step 3 — Fit topics on competitors (reuse loaded sbert) ──────
    print("\n[3/6] Fitting BERTopic on competitors ...")

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
    print("\n[4/6] Assigning topics to target reviews ...")
    topics_target, _ = topic_model.transform(texts_target, emb_target)

    # ── Step 5 — Sentiment ────────────────────────────────────────────
    print("\n[5/6] Running sentiment on target ...")
    sent_target = run_sentiment(texts_target)

    print("\n  Running sentiment on competitors ...")
    sent_comp = run_sentiment(texts_comp)

    # ── Step 6 — Assemble ─────────────────────────────────────────────
    print("\n[6/6] Assembling result DataFrames ...")
    df_target_full = assemble_results(df_target, topics_target, sent_target, topic_labels)
    df_comp_full   = assemble_results(df_comp,   topics_comp,   sent_comp,   topic_labels)


    return df_target_full, df_comp_full, topic_model



# ══════════════════════════════════════════════════════════════════════
# STAGE 7 — MACRO TOPIC MAPPING
# Map granular BERTopic clusters (T0, T1 ...) into a small set of
# business-meaningful macro categories relevant to Electronics & Tech
# ══════════════════════════════════════════════════════════════════════

# ── Define macro topic keyword groups ─────────────────────────────────
# Each macro topic is defined by keywords that should appear in the
# BERTopic label. The matching is done on the full topic label string
# (which contains the top 4 keywords), case-insensitive.
# Order matters — first match wins, so put specific terms before generic.

MACRO_TOPIC_KEYWORDS = {
    "Delivery & Shipping": [
        "deliver", "shipping", "dispatch", "courier", "arrived",
        "arrival", "parcel", "package", "tracked", "next day",
        "day delivery", "quick delivery", "fast delivery",
        "tried deliver", "delivering"
    ],
    "Order & Purchase Process": [
        "order", "purchase", "bought", "checkout", "buy",
        "ordered", "buying", "item purchased", "ordered item",
        "ordered wrong", "ordered online", "easy purchase",
        "went smoothly", "process"
    ],
    "Returns, Refunds & Cancellations": [
        "refund", "return", "cancel", "cancelled", "refunded",
        "replacement", "exchange", "money back", "service order",
        "order cancelled"
    ],
    "Customer Service & Support": [
        "customer service", "customer services", "support",
        "helpful", "staff", "team", "agent", "contact",
        "response", "replied", "communicate", "pleasant helpful",
        "service friendly", "service helpful", "service quick",
        "service excellent", "great service", "service received"
    ],
    "Product Quality & Condition": [
        "quality", "condition", "refurbished", "excellent condition",
        "product", "item", "received phone", "reliable",
        "performance", "features", "functionality", "basic",
        "repaired", "repairs", "micro repairs", "dryer", "washer",
        "appliance"
    ],
    "Price & Value": [
        "price", "value", "affordable", "cheap", "expensive",
        "cost", "discount", "deal", "worth", "good discount",
        "overall", "improvements"
    ],
    "Website & Online Experience": [
        "website", "online", "tool", "easy use", "app",
        "scam", "checkout", "billing", "contact", "review",
        "easy use", "process", "use"
    ],
    "Repair & Technical Service": [
        "repair", "repaired", "fix", "technician", "engineer",
        "micro repairs", "repaired returned", "letting know",
        "repair quickly"
    ],
}

MACRO_OTHER = "General / Other"


def assign_macro_topic(topic_label: str,
                       keyword_map: dict = MACRO_TOPIC_KEYWORDS) -> str:
    """
    Map a granular BERTopic label to a macro business topic by keyword
    matching on the label string.

    The label is expected to follow the format produced by
    get_topic_labels(): "TN: word1, word2, word3, word4".
    Matching is case-insensitive and first-match wins, so the order of
    keys in MACRO_TOPIC_KEYWORDS controls priority.

    Parameters
    ----------
    topic_label  : str    BERTopic topic label string
    keyword_map  : dict   {macro_topic_name: [keywords]}

    Returns
    -------
    str   macro topic name, or MACRO_OTHER if no keyword matches
    """
    label_lower = topic_label.lower()
    for macro, keywords in keyword_map.items():
        if any(kw.lower() in label_lower for kw in keywords):
            return macro
    return MACRO_OTHER


def add_macro_topics(df: pd.DataFrame,
                     keyword_map: dict = MACRO_TOPIC_KEYWORDS) -> pd.DataFrame:
    """
    Add a 'macro_topic' column to an assembled results DataFrame by
    applying assign_macro_topic() to every row's topic_label.

    Outlier reviews (topic_id == -1) are labelled as MACRO_OTHER.

    Parameters
    ----------
    df          : pd.DataFrame   output of assemble_results()
    keyword_map : dict           macro topic keyword map

    Returns
    -------
    pd.DataFrame   copy of df with new 'macro_topic' column
    """
    df = df.copy()
    df['macro_topic'] = df['topic_label'].apply(
        lambda lbl: MACRO_OTHER if lbl == "Outliers / uncategorised"
        else assign_macro_topic(lbl, keyword_map)
    )
    return df


def macro_topic_summary(df: pd.DataFrame,
                        exclude_other: bool = False) -> pd.DataFrame:
    """
    Aggregate sentiment metrics at the macro topic level.

    Parameters
    ----------
    df            : pd.DataFrame   output of add_macro_topics()
    exclude_other : bool           drop 'General / Other' rows

    Returns
    -------
    pd.DataFrame with columns:
        macro_topic, n_reviews, pct_positive, pct_neutral,
        pct_negative, net_sentiment, dominant_sentiment
    Sorted by net_sentiment descending.
    """
    if exclude_other:
        df = df[df['macro_topic'] != MACRO_OTHER].copy()

    agg = df.groupby('macro_topic').apply(lambda g: pd.Series({
        'n_reviews'   : len(g),
        'pct_positive': (g['sentiment'] == 'positive').mean() * 100,
        'pct_neutral' : (g['sentiment'] == 'neutral').mean()  * 100,
        'pct_negative': (g['sentiment'] == 'negative').mean() * 100,
    })).reset_index()

    agg['net_sentiment']      = agg['pct_positive'] - agg['pct_negative']
    agg['dominant_sentiment'] = agg[['pct_positive',
                                      'pct_neutral',
                                      'pct_negative']].idxmax(axis=1)\
                                     .str.replace('pct_', '')
    return agg.sort_values('net_sentiment', ascending=False).reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════
# STAGE 8 — MACRO TOPIC VISUALIZATIONS
# ══════════════════════════════════════════════════════════════════════

def plot_macro_distribution(df: pd.DataFrame, title: str) -> None:
    """
    Horizontal bar chart of macro topic review counts, colored by
    dominant sentiment. Net sentiment annotated on each bar.

    Parameters
    ----------
    df    : pd.DataFrame   output of add_macro_topics()
    title : str            chart title
    """
    summary = macro_topic_summary(df, exclude_other=False)\
                .sort_values('n_reviews')

    color_map = {
        'positive': '#2ca02c',
        'neutral' : '#bcbd22',
        'negative': '#d62728'
    }
    colors = [color_map.get(s, 'gray') for s in summary['dominant_sentiment']]

    fig, ax = plt.subplots(figsize=(11, max(4, len(summary) * 0.55)))
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


def plot_macro_heatmap(df_target: pd.DataFrame,
                       df_comp: pd.DataFrame,
                       target_name: str) -> None:
    """
    Side-by-side heatmaps showing positive / neutral / negative % per
    macro topic for the target company and competitors.

    Both heatmaps share the same row order (sorted by target net
    sentiment) so differences are immediately visible.

    Parameters
    ----------
    df_target   : pd.DataFrame   add_macro_topics() output for target
    df_comp     : pd.DataFrame   add_macro_topics() output for competitors
    target_name : str
    """
    s_target = macro_topic_summary(df_target, exclude_other=False)\
                   .set_index('macro_topic')
    s_comp   = macro_topic_summary(df_comp,   exclude_other=False)\
                   .set_index('macro_topic')

    # Shared row order = target net sentiment descending
    row_order = s_target.sort_values('net_sentiment',
                                     ascending=False).index.tolist()
    # Add any macro topics that appear in competitors but not target
    for t in s_comp.index:
        if t not in row_order:
            row_order.append(t)

    cols = ['pct_positive', 'pct_neutral', 'pct_negative']
    col_labels = ['Positive %', 'Neutral %', 'Negative %']

    def _heat(s, order):
        return s.reindex(order)[cols]\
                .rename(columns=dict(zip(cols, col_labels)))\
                .fillna(0)

    heat_t = _heat(s_target, row_order)
    heat_c = _heat(s_comp,   row_order)

    fig, axes = plt.subplots(1, 2, figsize=(16, max(4, len(row_order) * 0.55)))

    for ax, heat, title in zip(
        axes,
        [heat_t, heat_c],
        [target_name, "Competitors avg"]
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


def plot_macro_head_to_head(df_target: pd.DataFrame,
                             df_comp: pd.DataFrame,
                             target_name: str) -> pd.DataFrame:
    """
    Grouped bar chart comparing net_sentiment per macro topic between
    target and competitors. Sorted by gap (target − competitors) to
    surface strengths and weaknesses immediately.

    Parameters
    ----------
    df_target   : pd.DataFrame   add_macro_topics() output for target
    df_comp     : pd.DataFrame   add_macro_topics() output for competitors
    target_name : str

    Returns
    -------
    pd.DataFrame   merged comparison table with gap column
    """
    s_t = macro_topic_summary(df_target, exclude_other=True)\
              [['macro_topic', 'net_sentiment', 'n_reviews']]
    s_c = macro_topic_summary(df_comp,   exclude_other=True)\
              [['macro_topic', 'net_sentiment', 'n_reviews']]

    merged = s_t.merge(s_c, on='macro_topic',
                       suffixes=('_target', '_comp'))
    merged['gap'] = merged['net_sentiment_target'] - merged['net_sentiment_comp']
    merged = merged.sort_values('gap')

    fig, ax = plt.subplots(figsize=(12, max(5, len(merged) * 0.65)))
    y, h = np.arange(len(merged)), 0.35

    ax.barh(y + h/2, merged['net_sentiment_target'], h,
            color='#1f77b4', label=target_name, edgecolor='white')
    ax.barh(y - h/2, merged['net_sentiment_comp'],   h,
            color='#aec7e8', label='Competitors avg', edgecolor='white')

    # Annotate gap
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
        "(sorted by gap: green Δ = we outperform, red Δ = we underperform)",
        fontsize=12, fontweight='bold'
    )
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.show()

    return merged


def print_macro_strengths_weaknesses(merged: pd.DataFrame,
                                      target_name: str) -> None:
    """
    Print macro-level strengths, weaknesses, and improvement priorities.

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
    worst = merged.sort_values('net_sentiment_target')
    for _, row in worst.iterrows():
        flag = '🔴' if row['net_sentiment_target'] < -25 else \
               '🟡' if row['net_sentiment_target'] < 0   else '🟢'
        print(f"  {flag}  {row['macro_topic']:<40}  "
              f"net={row['net_sentiment_target']:+.0f}")


# ══════════════════════════════════════════════════════════════════════
# STAGE 9 — ROOT CAUSE EXTRACTION
# For macro topics where we underperform, extract the exact reviews
# and summarise the specific complaints driving negative sentiment
# ══════════════════════════════════════════════════════════════════════

def get_weak_macro_topics(merged: pd.DataFrame,
                           min_gap: float = -5.0) -> list:
    """
    Return the list of macro topics where the target company
    underperforms competitors by at least min_gap points.

    Parameters
    ----------
    merged  : pd.DataFrame   output of plot_macro_head_to_head()
    min_gap : float          gap threshold (negative = underperforming)
                             e.g. -5.0 means gap < -5

    Returns
    -------
    list of str   macro topic names sorted by gap ascending (worst first)
    """
    weak = merged[merged['gap'] < min_gap].sort_values('gap')
    return weak['macro_topic'].tolist()


def extract_negative_reviews(df_target: pd.DataFrame,
                              macro_topic: str,
                              sentiment_filter: str = 'negative') -> pd.DataFrame:
    """
    Extract raw reviews from the target company for a given macro topic,
    filtered to a specific sentiment.

    Parameters
    ----------
    df_target       : pd.DataFrame   add_macro_topics() output for target
    macro_topic     : str            one of the keys in MACRO_TOPIC_KEYWORDS
    sentiment_filter: str            'negative' | 'neutral' | 'positive' | 'all'

    Returns
    -------
    pd.DataFrame   subset of df_target with columns:
        company, stars, topic_label, macro_topic,
        sentiment, sentiment_score, title, review
    Sorted by stars ascending (worst reviews first).
    """
    mask = df_target['macro_topic'] == macro_topic
    if sentiment_filter != 'all':
        mask = mask & (df_target['sentiment'] == sentiment_filter)

    cols = ['company', 'stars', 'topic_label', 'macro_topic',
            'sentiment', 'sentiment_score', 'title', 'review']
    available = [c for c in cols if c in df_target.columns]

    return df_target[mask][available]\
               .sort_values('stars', ascending=True)\
               .reset_index(drop=True)


def root_cause_report(df_target: pd.DataFrame,
                       df_comp: pd.DataFrame,
                       merged: pd.DataFrame,
                       target_name: str,
                       min_gap: float = -5.0,
                       max_reviews_shown: int = 5) -> dict:
    """
    Full root cause analysis for all macro topics where the target
    company underperforms competitors.

    For each weak macro topic:
        1. Prints the performance gap vs competitors
        2. Shows the granular BERTopic sub-topics that map to it
        3. Prints the most negative raw reviews verbatim (up to
           max_reviews_shown), so the exact customer language is visible
        4. Compares the word distribution of negative reviews between
           target and competitors in that macro topic

    Parameters
    ----------
    df_target         : pd.DataFrame   add_macro_topics() output for target
    df_comp           : pd.DataFrame   add_macro_topics() output for competitors
    merged            : pd.DataFrame   output of plot_macro_head_to_head()
    target_name       : str
    min_gap           : float          gap threshold to flag as weak
    max_reviews_shown : int            max raw reviews printed per topic

    Returns
    -------
    dict  {macro_topic: DataFrame of negative reviews}
          Use this to inspect or export the full review text per topic.
    """
    weak_topics = get_weak_macro_topics(merged, min_gap=min_gap)

    if not weak_topics:
        print("✅ No macro topics found where gap is below threshold. "
              "No clear underperformance vs competitors.")
        return {}

    all_negative_reviews = {}

    for macro in weak_topics:
        row = merged[merged['macro_topic'] == macro].iloc[0]

        print(f"\n{'█'*65}")
        print(f"  ROOT CAUSE — {macro}")
        print(f"{'█'*65}")
        print(f"  {target_name} net sentiment : {row['net_sentiment_target']:+.0f}")
        print(f"  Competitors net sentiment   : {row['net_sentiment_comp']:+.0f}")
        print(f"  Gap                         : {row['gap']:+.0f}  "
              f"({'we underperform' if row['gap'] < 0 else 'we outperform'})")

        # ── Sub-topics breakdown ───────────────────────────────────────
        sub_mask   = df_target['macro_topic'] == macro
        sub_topics = df_target[sub_mask]['topic_label'].value_counts()

        print(f"\n  Granular sub-topics in this macro category:")
        for topic_lbl, cnt in sub_topics.items():
            sub_neg_pct = (
                df_target[sub_mask & (df_target['topic_label'] == topic_lbl)]
                ['sentiment'].eq('negative').mean() * 100
            )
            flag = '🔴' if sub_neg_pct > 60 else '🟡' if sub_neg_pct > 30 else '🟢'
            print(f"    {flag}  {topic_lbl:<55}  "
                  f"n={cnt:>3}   neg={sub_neg_pct:.0f}%")

        # ── Raw negative reviews ───────────────────────────────────────
        neg_reviews = extract_negative_reviews(df_target, macro,
                                               sentiment_filter='negative')
        all_negative_reviews[macro] = neg_reviews

        print(f"\n  Most negative reviews ({len(neg_reviews)} total "
              f"negative, showing {min(max_reviews_shown, len(neg_reviews))}):")

        for i, rev in neg_reviews.head(max_reviews_shown).iterrows():
            stars_str = '★' * int(rev['stars']) + '☆' * (5 - int(rev['stars']))
            title_str = rev.get('title', '')
            print(f"\n  [{i+1}] {stars_str}  |  {title_str}")
            print(f"  Sub-topic : {rev['topic_label']}")
            print(f"  Review    : {rev['review'][:500]}"
                  f"{'...' if len(str(rev['review'])) > 500 else ''}")

        # ── Word frequency comparison: target vs competitors ──────────
        _plot_word_gap(df_target, df_comp, macro, target_name)

    return all_negative_reviews


def _plot_word_gap(df_target: pd.DataFrame,
                   df_comp: pd.DataFrame,
                   macro_topic: str,
                   target_name: str,
                   top_n: int = 15) -> None:
    """
    Internal helper: side-by-side word frequency bar charts for
    NEGATIVE reviews in a given macro topic, comparing target vs
    competitors. Reveals the specific language driving dissatisfaction.

    Parameters
    ----------
    df_target   : pd.DataFrame   add_macro_topics() output for target
    df_comp     : pd.DataFrame   add_macro_topics() output for competitors
    macro_topic : str
    target_name : str
    top_n       : int            words to display per panel
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
            freq_sorted = freq.sort_values()
            ax.barh(freq_sorted.index, freq_sorted.values,
                    color=color, edgecolor='white', alpha=0.85)
        ax.set_title(label, fontsize=11, fontweight='bold')
        ax.set_xlabel("Word frequency in negative reviews")

    fig.suptitle(f"Most Common Words in Negative Reviews — {macro_topic}",
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()


# ══════════════════════════════════════════════════════════════════════
# STAGE 10 — FULL MACRO PIPELINE
# ══════════════════════════════════════════════════════════════════════

def run_macro_analysis(df_target_full: pd.DataFrame,
                        df_comp_full: pd.DataFrame,
                        target_name: str,
                        min_gap: float = -5.0,
                        max_reviews_shown: int = 5) -> dict:
    """
    Full macro topic analysis pipeline. Call this after run_nlp_pipeline().

    Steps
    -----
    1.  Assign macro topics to target and competitor DataFrames
    2.  Print macro topic distribution for target
    3.  Plot macro topic distribution bar (target)
    4.  Plot macro topic distribution bar (competitors)
    5.  Plot side-by-side sentiment heatmaps
    6.  Plot head-to-head net sentiment comparison
    7.  Print macro strengths, weaknesses, and improvement priorities
    8.  Run root cause analysis on all underperforming macro topics:
        - Sub-topic breakdown
        - Raw negative reviews printed verbatim
        - Word frequency comparison (target vs competitors)

    Parameters
    ----------
    df_target_full    : pd.DataFrame   output of assemble_results() for target
    df_comp_full      : pd.DataFrame   output of assemble_results() for competitors
    target_name       : str
    min_gap           : float          threshold for flagging weak topics (default −5)
    max_reviews_shown : int            raw reviews shown per weak topic (default 5)

    Returns
    -------
    dict with keys:
        'df_target_macro'  : target DataFrame with macro_topic column
        'df_comp_macro'    : competitor DataFrame with macro_topic column
        'summary_target'   : macro_topic_summary() for target
        'summary_comp'     : macro_topic_summary() for competitors
        'merged'           : head-to-head comparison DataFrame
        'negative_reviews' : {macro_topic: DataFrame of negative reviews}
    """

    # ── Step 1 — Assign macro topics ──────────────────────────────────
    print("\n[1/5] Assigning macro topics ...")
    df_t = add_macro_topics(df_target_full)
    df_c = add_macro_topics(df_comp_full)

    macro_counts_t = df_t['macro_topic'].value_counts()
    print(f"\n  Macro topic distribution in {target_name}:")
    print(macro_counts_t.to_string())

    # ── Step 2 — Distribution plots ───────────────────────────────────
    print(f"\n[2/5] Macro topic distribution plots ...")
    plot_macro_distribution(df_t, title=target_name)
    plot_macro_distribution(df_c, title="Competitors")

    # ── Step 3 — Side-by-side heatmaps ────────────────────────────────
    print("\n[3/5] Macro sentiment heatmaps ...")
    plot_macro_heatmap(df_t, df_c, target_name=target_name)

    # ── Step 4 — Head-to-head ─────────────────────────────────────────
    print("\n[4/5] Head-to-head net sentiment ...")
    merged = plot_macro_head_to_head(df_t, df_c, target_name=target_name)
    print_macro_strengths_weaknesses(merged, target_name=target_name)

    # ── Step 5 — Root cause ───────────────────────────────────────────
    print(f"\n[5/5] Root cause extraction (gap threshold = {min_gap}) ...")
    neg_reviews = root_cause_report(
        df_target=df_t,
        df_comp=df_c,
        merged=merged,
        target_name=target_name,
        min_gap=min_gap,
        max_reviews_shown=max_reviews_shown
    )

    return {
        'df_target_macro' : df_t,
        'df_comp_macro'   : df_c,
        'summary_target'  : macro_topic_summary(df_t),
        'summary_comp'    : macro_topic_summary(df_c),
        'merged'          : merged,
        'negative_reviews': neg_reviews,
    }
