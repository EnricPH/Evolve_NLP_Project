import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# ── Style ──────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'figure.dpi': 130, 'figure.figsize': (12, 5)})

ACCENT = "#4C72B0"   # primary color used across single-color plots


# ══════════════════════════════════════════════════════════════════════
# 1. BASIC DATASET OVERVIEW
# ══════════════════════════════════════════════════════════════════════

def dataset_overview(df: pd.DataFrame) -> pd.DataFrame:
    """
    Print a quick structural summary of the cleaned category DataFrame.

    Shows shape, number of companies, star distribution, and basic
    null/empty counts for the three key columns.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame for a single category.

    Returns
    -------
    pd.DataFrame
        Summary table (also printed to stdout).
    """
    print("=" * 55)
    print("  DATASET OVERVIEW")
    print("=" * 55)
    print(f"  Total reviews      : {len(df):,}")
    print(f"  Unique companies   : {df['company'].nunique():,}")
    print(f"  Star range         : {int(df['stars'].min())} – {int(df['stars'].max())}")
    print(f"  Avg stars (global) : {df['stars'].mean():.2f}")
    print()

    # Null / empty check
    summary_rows = []
    for col in ['title', 'review', 'stars']:
        nulls = df[col].isna().sum()
        if col != 'stars':
            empty = (df[col].str.strip() == '').sum()
        else:
            empty = 0
        summary_rows.append({'column': col, 'nulls': nulls, 'empty_strings': empty})

    summary = pd.DataFrame(summary_rows).set_index('column')
    print(summary.to_string())
    print("=" * 55)
    return summary


# ══════════════════════════════════════════════════════════════════════
# 2. FEATURE ENGINEERING — text lengths
# ══════════════════════════════════════════════════════════════════════

def add_length_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add character-count and word-count columns for title and review.

    New columns added (in-place on a copy):
        - title_char_len   : number of characters in title
        - title_word_count : number of words in title
        - review_char_len  : number of characters in review
        - review_word_count: number of words in review

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame with 'title' and 'review' columns.

    Returns
    -------
    pd.DataFrame
        Copy of df with four new numeric columns appended.
    """
    df = df.copy()

    for col in ['title', 'review']:
        df[f'{col}_char_len']   = df[col].fillna('').str.len()
        df[f'{col}_word_count'] = df[col].fillna('').str.split().str.len()

    return df


# ══════════════════════════════════════════════════════════════════════
# 3. COMPANY-LEVEL AGGREGATION
# ══════════════════════════════════════════════════════════════════════

def company_stats(df: pd.DataFrame) -> pd.DataFrame:

    """
    Compute per-company statistics needed for positioning charts.

    Metrics computed:
        avg_stars        : mean star rating
        pct_positive     : % reviews with stars >= 4
        pct_neutral      : % reviews with stars == 3
        pct_negative     : % reviews with stars <= 2
        avg_review_words : mean word count of reviews
        avg_title_words  : mean word count of titles
        review_word_count: mean review word count (all reviews)
        wc_positive      : mean word count of positive reviews (stars>=4)
        wc_neutral       : mean word count of neutral reviews (stars==3)
        wc_negative      : mean word count of negative reviews (stars<=2)
        review_count     : total number of reviews

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned category DataFrame with 'review', 'stars', 'company'.

    Returns
    -------
    pd.DataFrame
        One row per company.
    """
    df = df.copy()
    df['word_count'] = df['review'].fillna('').str.split().str.len()

    stats = df.groupby('company').agg(
        avg_stars        = ('stars', 'mean'),
        pct_positive     = ('stars', lambda x: (x >= 4).mean() * 100),
        pct_neutral      = ('stars', lambda x: (x == 3).mean() * 100),
        pct_negative     = ('stars', lambda x: (x <= 2).mean() * 100),
        avg_review_words = ('review_word_count', 'mean'),
        avg_title_words  = ('title_word_count', 'mean'),
        review_word_count = ('word_count', 'mean'),
        wc_positive      = ('word_count', lambda x: x[df.loc[x.index, 'stars'] >= 4].mean() if any(df.loc[x.index, 'stars'] >= 4) else 0),
        wc_neutral       = ('word_count', lambda x: x[df.loc[x.index, 'stars'] == 3].mean() if any(df.loc[x.index, 'stars'] == 3) else 0),
        wc_negative      = ('word_count', lambda x: x[df.loc[x.index, 'stars'] <= 2].mean() if any(df.loc[x.index, 'stars'] <= 2) else 0),
        review_count     = ('review', 'count')
    ).round(2).sort_values('avg_stars', ascending=False)
    #.reset_index()
    return stats


# ══════════════════════════════════════════════════════════════════════
# 4. VISUALIZATIONS
# ══════════════════════════════════════════════════════════════════════

# ── 4.1  Star distribution ─────────────────────────────────────────────

def plot_star_distribution(df: pd.DataFrame) -> None:
    """
    Bar chart of review counts per star rating (1–5) for the whole category.

    Each bar is colored on a red→green gradient to make polarity
    immediately readable.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame with a 'stars' column.
    """
    counts = df['stars'].value_counts().sort_index()
    colors = ['#d62728', '#ff7f0e', '#bcbd22', '#2ca02c', '#1a7f4b']

    fig, ax = plt.subplots()
    bars = ax.bar(counts.index.astype(str), counts.values, color=colors, edgecolor='white')

    # Annotate bars with count + percentage
    total = counts.sum()
    for bar, count in zip(bars, counts.values):
        pct = count / total * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + total * 0.003,
            f"{count:,}\n({pct:.1f}%)",
            ha='center', va='bottom', fontsize=9
        )

    ax.set_title("Star Rating Distribution", fontsize=14, fontweight='bold')
    ax.set_xlabel("Stars")
    ax.set_ylabel("Number of Reviews")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    plt.tight_layout()
    plt.show()


# ── 4.2  Avg stars per company (top & bottom N) ────────────────────────

def plot_avg_stars_per_company(stats: pd.DataFrame, top_n: int = 10) -> None:
    """
    Horizontal bar chart showing the top-N and bottom-N companies by
    average star rating side by side.

    Parameters
    ----------
    stats : pd.DataFrame
        Output of company_stats().
    top_n : int
        Number of companies to show at each end. Default 10.
    """
    top    = stats.head(top_n)[['avg_stars']].copy()
    bottom = stats.tail(top_n)[['avg_stars']].copy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, data, title, color in zip(
        axes,
        [top, bottom],
        [f"Top {top_n} Companies by Avg Stars", f"Bottom {top_n} Companies by Avg Stars"],
        ['#2ca02c', '#d62728']
    ):
        data = data.sort_values('avg_stars')
        ax.barh(data.index, data['avg_stars'], color=color, edgecolor='white')
        ax.set_xlim(0, 5.5)
        ax.set_xlabel("Average Stars")
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axvline(data['avg_stars'].mean(), color='gray', linestyle='--', linewidth=1, label='Mean')

        for i, (company, row) in enumerate(data.iterrows()):
            ax.text(row['avg_stars'] + 0.05, i, f"{row['avg_stars']:.2f}", va='center', fontsize=8)

    plt.tight_layout()
    plt.show()


# ── 4.3  Review word count distribution ───────────────────────────────

def plot_review_length_distribution(df: pd.DataFrame) -> None:
    """
    Overlapping histogram + KDE of review word counts, split by sentiment
    (positive = stars >= 4, negative = stars <= 2, neutral = 3).

    Helps understand whether positive and negative reviewers write
    differently in terms of length.

    Parameters
    ----------
    df : pd.DataFrame
        Output of add_length_features(), must have 'review_word_count'
        and 'stars' columns.
    """
    df = df.copy()
    df['sentiment_label'] = df['stars'].apply(
        lambda s: 'Positive (4-5)' if s >= 4 else ('Negative (1-2)' if s <= 2 else 'Neutral (3)')
    )

    palette = {
        'Positive (4-5)': '#2ca02c',
        'Neutral (3)':    '#bcbd22',
        'Negative (1-2)': '#d62728',
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    for label, color in palette.items():
        subset = df[df['sentiment_label'] == label]['review_word_count']
        axes[0].hist(subset, bins=40, alpha=0.6, color=color, label=label, density=True)
    axes[0].set_title("Review Word Count Distribution by Sentiment", fontweight='bold')
    axes[0].set_xlabel("Word Count")
    axes[0].set_ylabel("Density")
    axes[0].legend()
    axes[0].set_xlim(0, df['review_word_count'].quantile(0.99))  # trim extreme outliers

    # Box plot
    order = ['Negative (1-2)', 'Neutral (3)', 'Positive (4-5)']
    sns.boxplot(
        data=df[df['sentiment_label'].isin(order)],
        x='sentiment_label', y='review_word_count',
        order=order, palette=palette, ax=axes[1],
        showfliers=False     # hide extreme outliers for readability
    )
    axes[1].set_title("Word Count Spread by Sentiment", fontweight='bold')
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Word Count")
    axes[1].set_ylim(0, df['review_word_count'].quantile(0.97))

    plt.tight_layout()
    plt.show()


# ── 4.4  Avg review length per company ────────────────────────────────

def plot_avg_review_length_per_company(stats: pd.DataFrame, top_n: int = 15) -> None:
    """
    Horizontal bar chart of average review word count per company,
    colored by their average star rating.

    Useful for spotting whether companies with more engaged reviewers
    also tend to have higher / lower ratings.

    Parameters
    ----------
    stats : pd.DataFrame
        Output of company_stats().
    top_n : int
        Number of companies to display. Default 15.
    """
    data = stats.nlargest(top_n, 'avg_review_words').sort_values('avg_review_words')

    # Color map: star rating → green/red gradient
    norm  = plt.Normalize(vmin=1, vmax=5)
    cmap  = plt.cm.RdYlGn
    colors = [cmap(norm(v)) for v in data['avg_stars']]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(data.index, data['avg_review_words'], color=colors, edgecolor='white')

    # Colorbar legend
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='Avg Star Rating', shrink=0.7)

    ax.set_title(f"Top {top_n} Companies by Avg Review Word Count", fontsize=13, fontweight='bold')
    ax.set_xlabel("Avg Word Count per Review")
    plt.tight_layout()
    plt.show()

# ── 4.5  Summary stats table per company ──────────────────────────────

def display_company_summary(stats: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """
    Pretty-print the top-N companies by average stars as a formatted table.

    Parameters
    ----------
    stats : pd.DataFrame
        Output of company_stats().
    top_n : int
        Rows to display.

    Returns
    -------
    pd.DataFrame
        Formatted slice of stats.
    """
    display_cols = ['review_count', 'avg_stars', 'pct_positive', 'pct_negative',
                    'avg_review_words', 'avg_title_words']
    return stats[display_cols].head(top_n).style\
        .background_gradient(subset=['avg_stars'],    cmap='RdYlGn', vmin=1, vmax=5)\
        .background_gradient(subset=['pct_positive'], cmap='Greens')\
        .background_gradient(subset=['pct_negative'], cmap='Reds')\
        .format({'avg_stars': '{:.2f}', 'pct_positive': '{:.1f}%',
                 'pct_negative': '{:.1f}%', 'avg_review_words': '{:.0f}',
                 'avg_title_words': '{:.0f}'})\
        .set_caption("Company Summary — Sorted by Avg Stars")


# ══════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════
# OUR COMPANY VS THE OTHERS
# ══════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════


def _percentile_rank(series: pd.Series, value: float) -> float:
    """Return the percentile rank (0–100) of *value* within *series*."""
    return (series < value).mean() * 100


def _add_percentile_band(ax, series, value, color, label):
    """
    Draw a vertical dashed line for TARGET and shade the percentile band.
    Annotates with the percentile rank.
    """
    pct = _percentile_rank(series, value)
    ax.axvline(value, color=color, linewidth=2, linestyle='--', zorder=5, label=label)
    ax.axvspan(series.min(), value, alpha=0.08, color=color, zorder=0)
    ax.text(
        value, ax.get_ylim()[1] * 0.88,
        f" P{pct:.0f}",
        color=color, fontsize=10, fontweight='bold', va='top'
    )
    return pct


# ══════════════════════════════════════════════════════════════════════
# CHART 1 — Avg Stars: where does TARGET sit in the distribution?
# ══════════════════════════════════════════════════════════════════════

def plot_stars_positioning(stats: pd.DataFrame, CAT: str, target: str) -> None:
    """
    KDE + histogram of avg_stars across all companies in the category,
    with a vertical marker showing where TARGET sits and its percentile.

    Parameters
    ----------
    stats  : pd.DataFrame   output of company_stats()
    target : str            company name (must match stats index/column)
    """
    target_val = stats.loc[stats['company'] == target, 'avg_stars'].values[0]
    others     = stats.loc[stats['company'] != target, 'avg_stars']

    fig, ax = plt.subplots(figsize=(11, 4))

    # Distribution of competitors
    sns.histplot(others, bins=10, stat='density', color='#4C72B0',
                 alpha=0.35, edgecolor='white', ax=ax, label='Competitors')
    #sns.kdeplot(others, color='#4C72B0', linewidth=1.8, ax=ax)

    ax.set_xlim(2.5, 5.25)
    ax.set_ylim(bottom=0)

    pct = _add_percentile_band(ax, others, target_val, '#d62728', f'{target} (target)')

    # Sector median line
    median = others.median()
    ax.axvline(median, color='gray', linewidth=1.2, linestyle=':', label=f'Sector median ({median:.2f})')

    ax.set_title(
        f"Avg Star Rating — {target} vs {CAT} sector\n"
        f"Target is at the {pct:.0f}th percentile",
        fontsize=13, fontweight='bold'
    )
    ax.set_xlabel("Average Stars per Company")
    ax.set_ylabel("Density")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.show()
    print(f"  ► {target} avg stars: {target_val:.2f}  |  Sector median: {median:.2f}  |  Percentile: {pct:.0f}th")


# ══════════════════════════════════════════════════════════════════════
# CHART 2 — Positive / Neutral / Negative % stacked bar (target vs sector avg)
# ══════════════════════════════════════════════════════════════════════

def plot_sentiment_pct_comparison(stats: pd.DataFrame, target: str) -> None:
    """
    Stacked horizontal bar chart comparing the target company's
    positive / neutral / negative review split against the sector average
    and the sector's best and worst performers.

    Parameters
    ----------
    stats  : pd.DataFrame   output of company_stats()
    target : str            company name
    """
    trow    = stats[stats['company'] == target].iloc[0]
    sector  = stats[stats['company'] != target]

    # Build comparison rows
    rows = {
        target                        : trow,
        'Sector median'              : sector.median(numeric_only=True),
        f"Best (★ {sector['avg_stars'].max():.2f})" : sector.loc[sector['avg_stars'].idxmax()],
        f"Worst (★ {sector['avg_stars'].min():.2f})": sector.loc[sector['avg_stars'].idxmin()],
    }

    labels   = list(rows.keys())
    pos_vals = [rows[k]['pct_positive'] for k in labels]
    neu_vals = [rows[k]['pct_neutral']  for k in labels]
    neg_vals = [rows[k]['pct_negative'] for k in labels]

    colors = {'positive': '#2ca02c', 'neutral': '#bcbd22', 'negative': '#d62728'}

    fig, ax = plt.subplots(figsize=(11, 3.8))
    y = np.arange(len(labels))

    b1 = ax.barh(y, pos_vals, color=colors['positive'], label='Positive (4-5★)', edgecolor='white')
    b2 = ax.barh(y, neu_vals, left=pos_vals, color=colors['neutral'],  label='Neutral (3★)',    edgecolor='white')
    b3 = ax.barh(y, neg_vals,
                 left=[p + n for p, n in zip(pos_vals, neu_vals)],
                 color=colors['negative'], label='Negative (1-2★)', edgecolor='white')

    # Annotate values inside bars
    for i, (p, nu, ng) in enumerate(zip(pos_vals, neu_vals, neg_vals)):
        if p > 6:   ax.text(p / 2,              i, f"{p:.0f}%",  ha='center', va='center', fontsize=8.5, color='white', fontweight='bold')
        if nu > 6:  ax.text(p + nu / 2,         i, f"{nu:.0f}%", ha='center', va='center', fontsize=8.5, color='white', fontweight='bold')
        if ng > 6:  ax.text(p + nu + ng / 2,    i, f"{ng:.0f}%", ha='center', va='center', fontsize=8.5, color='white', fontweight='bold')

    # Highlight target row
    ax.axhspan(len(labels) - 1 - 0.4, len(labels) - 1 + 0.4, color='#d62728', alpha=0.06, zorder=0)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlim(0, 105)
    ax.set_xlabel("% of Reviews")
    ax.set_title(f"Review Sentiment Split — {target} vs Benchmarks", fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    plt.tight_layout()
    plt.show()


# ══════════════════════════════════════════════════════════════════════
# CHART 3 — Star percentile dashboard (3 KPIs in one figure)
# ══════════════════════════════════════════════════════════════════════

def plot_percentile_dashboard(stats: pd.DataFrame, CAT: str, target: str) -> None:
    """
    Three-panel strip chart showing where TARGET sits (as a percentile)
    across three key metrics:
        - % Positive reviews
        - % Negative reviews
        - Average star rating

    Each panel shows the full distribution of competitors as grey dots,
    the sector median as a dashed line, and the target as a coloured dot.

    Parameters
    ----------
    stats  : pd.DataFrame   output of company_stats()
    target : str            company name
    """
    metrics = [
        ('pct_positive', '% Positive reviews',  '#2ca02c'),
        ('pct_negative', '% Negative reviews',  '#d62728'),
        ('avg_stars',    'Avg star rating',      '#1f77b4'),
    ]

    trow    = stats[stats['company'] == target].iloc[0]
    others  = stats[stats['company'] != target]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    for ax, (col, label, color) in zip(axes, metrics):
        comp_vals = others[col].dropna().values
        target_val = trow[col]
        pct = _percentile_rank(pd.Series(comp_vals), target_val)

        # Jittered strip chart of competitors
        jitter = np.random.uniform(-0.15, 0.15, size=len(comp_vals))
        ax.scatter(comp_vals, jitter, color='gray', alpha=0.35, s=18, zorder=2)

        # Median reference
        med = np.median(comp_vals)
        ax.axvline(med, color='gray', linewidth=1.2, linestyle=':', zorder=3, label=f'Median: {med:.1f}')

        # Target dot
        ax.scatter([target_val], [0], color=color, s=140, zorder=5,
                   edgecolors='white', linewidths=1.5, label=f'Target: {target_val:.1f}')

        # Percentile annotation
        ax.set_ylim(-0.6, 0.6)
        ax.set_yticks([])
        ax.set_xlabel(label, fontsize=10)
        ax.set_title(f"P{pct:.0f}", fontsize=18, fontweight='bold', color=color)
        ax.legend(fontsize=8, loc='upper left')

    fig.suptitle(
        f"Percentile Positioning — {target} within {CAT}",
        fontsize=13, fontweight='bold', y=1.02
    )
    plt.tight_layout()
    plt.show()


# ══════════════════════════════════════════════════════════════════════
# CHART 4 — Word count by sentiment: TARGET vs sector percentiles
# ══════════════════════════════════════════════════════════════════════

def plot_word_count_by_sentiment(stats: pd.DataFrame, target: str) -> None:
    """
    Box plot of per-company average word count split by sentiment tier
    (positive / neutral / negative reviews), with the target company
    overlaid as a coloured marker showing its percentile within each tier.

    Answers: "Do our unhappy customers write more or less than competitors?"

    Parameters
    ----------
    stats  : pd.DataFrame   output of company_stats()
    target : str            company name
    """
    trow   = stats[stats['company'] == target].iloc[0]
    others = stats[stats['company'] != target]

    tiers = [
        ('wc_negative', 'Negative\n(1-2)', '#d62728'),
        ('wc_neutral',  'Neutral\n(3)',    '#bcbd22'),
        ('wc_positive', 'Positive\n(4-5)', '#2ca02c'),
    ]

    fig, ax = plt.subplots(figsize=(12, 5))

    positions = [1, 2, 3]
    bp_data   = [others[col].dropna().values for col, *_ in tiers]

    bp = ax.boxplot(
        bp_data,
        positions=positions,
        widths=0.45,
        patch_artist=True,
        medianprops=dict(color='black', linewidth=2),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        flierprops=dict(marker='o', markersize=3, alpha=0.4, markeredgewidth=0),
    )

    # Color each box
    box_colors = ['#d62728', '#bcbd22', '#2ca02c']
    for patch, c in zip(bp['boxes'], box_colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.25)

    # Overlay target markers
    for pos, (col, tier_label, color) in zip(positions, tiers):
        val = trow[col]
        if pd.isna(val):
            continue
        comp_vals = others[col].dropna()
        pct = _percentile_rank(comp_vals, val)

        ax.scatter(pos, val, color=color, s=160, zorder=5,
                   edgecolors='black', linewidths=1.2)
        ax.annotate(
            f"P{pct:.0f}\n{val:.0f}w",
            xy=(pos, val), xytext=(pos + 0.25, val),
            fontsize=9, fontweight='bold', color=color, va='center',
            arrowprops=dict(arrowstyle='->', color=color, lw=1.2)
        )

    ax.set_ylim(bottom=0, top=150)
    ax.set_xticks(positions)
    ax.set_xticklabels([t for _, t, _ in tiers], fontsize=11)
    ax.set_ylabel("Avg Words per Review")
    ax.set_title(
        f"Avg Review Word Count by Sentiment — {target} (●) vs Sector\n"
        "Box = sector distribution (IQR), whiskers = 1.5×IQR",
        fontsize=12, fontweight='bold'
    )

    # Legend
    target_patch = mpatches.Patch(color='gray', label=f'{target} marker (with percentile)')
    sector_patch = mpatches.Patch(facecolor='gray', alpha=0.25, label='Sector IQR')
    ax.legend(handles=[target_patch, sector_patch], fontsize=9)

    plt.tight_layout()
    plt.show()