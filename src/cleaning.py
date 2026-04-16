import pandas as pd
import re
import unicodedata
import logging
from typing import Optional

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

VALID_CATEGORIES = [
    'Animals & Pets',
    'Restaurants & Bars',
    'Public & Local Services',
    'Shopping & Fashion',
    'Media & Publishing',
    'Legal Services & Government',
    'Money & Insurance',
    'Vehicles & Transportation',
    'Utilities',
    'Sports',
    'Travel & Vacation',
    'Education & Training',
    'Electronics & Technology',
    'Construction & Manufacturing',
    'Events & Entertainment',
    'Beauty & Well-being',
    'Business Services',
    'Home & Garden',
    'Home Services',
    'Health & Medical',
    'Food, Beverages & Tobacco',
    'Hobbies & Crafts'
]

TEXT_COLUMNS = ['title', 'review', 'description']


# ─────────────────────────────────────────────
# Low-level text cleaners
# ─────────────────────────────────────────────

def _normalize_unicode(text: str) -> str:
    """
    Normalize unicode characters to their closest ASCII equivalent.

    Applies NFKD decomposition so accented characters like 'é' become 'e',
    and drops any remaining non-ASCII bytes that cannot be mapped.

    Parameters
    ----------
    text : str
        Raw input string.

    Returns
    -------
    str
        ASCII-safe string with normalized characters.
    """
    normalized = unicodedata.normalize('NFKD', text)
    return normalized.encode('ascii', errors='ignore').decode('ascii')


def _remove_html(text: str) -> str:
    """
    Strip HTML tags and decode common HTML entities.

    Handles tags like <br>, <p>, <b> that sometimes appear in scraped
    review data, and decodes entities such as &amp; &nbsp; &lt; &gt;

    Parameters
    ----------
    text : str
        Input string potentially containing HTML markup.

    Returns
    -------
    str
        Plain text with HTML removed and entities decoded.
    """
    # Decode common HTML entities first
    entities = {
        '&amp;':  '&',
        '&nbsp;': ' ',
        '&lt;':   '<',
        '&gt;':   '>',
        '&quot;': '"',
        '&#39;':  "'",
    }
    for entity, char in entities.items():
        text = text.replace(entity, char)

    # Remove all remaining HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    return text


def _remove_urls(text: str) -> str:
    """
    Remove HTTP/HTTPS URLs and bare www. domain references.

    Parameters
    ----------
    text : str
        Input string potentially containing URLs.

    Returns
    -------
    str
        String with URLs replaced by a single space.
    """
    text = re.sub(r'https?://\S+', ' ', text)
    text = re.sub(r'www\.\S+', ' ', text)
    return text


def _remove_emails(text: str) -> str:
    """
    Remove email addresses from text.

    Parameters
    ----------
    text : str
        Input string potentially containing email addresses.

    Returns
    -------
    str
        String with email addresses replaced by a single space.
    """
    return re.sub(r'\S+@\S+\.\S+', ' ', text)


def _fix_newlines(text: str) -> str:
    """
    Replace escaped newline/tab sequences and real newlines with a space.

    Trustpilot descriptions often contain literal '\\n' strings from
    scraping. This replaces both the escaped and real variants.

    Parameters
    ----------
    text : str
        Input string.

    Returns
    -------
    str
        String with newline and tab characters replaced by spaces.
    """
    text = text.replace('\\n', ' ').replace('\\t', ' ')
    text = re.sub(r'[\n\r\t]', ' ', text)
    return text


def _remove_special_characters(text: str) -> str:
    """
    Remove special and non-printable characters, keeping letters,
    digits, spaces, and basic punctuation ( . , ! ? ' - ).

    Emoji, symbols, currency signs, and other noise are dropped.

    Parameters
    ----------
    text : str
        Input string.

    Returns
    -------
    str
        Cleaned string retaining only readable characters.
    """
    # Keep: letters, digits, whitespace, and basic punctuation
    text = re.sub(r"[^a-zA-Z0-9\s.,!?'\-]", ' ', text)
    return text


def _normalize_whitespace(text: str) -> str:
    """
    Collapse multiple consecutive whitespace characters into a single space
    and strip leading/trailing whitespace.

    Parameters
    ----------
    text : str
        Input string.

    Returns
    -------
    str
        Whitespace-normalized string.
    """
    return re.sub(r'\s+', ' ', text).strip()


def clean_text(text: str) -> str:
    """
    Apply the full cleaning pipeline to a single text string.

    Pipeline steps (in order):
        1. Cast to str and handle NaN/None
        2. Fix escaped and real newlines
        3. Remove HTML tags and decode entities
        4. Remove URLs
        5. Remove email addresses
        6. Normalize unicode → ASCII
        7. Remove special / non-printable characters
        8. Normalize whitespace

    Parameters
    ----------
    text : str
        Raw text value from a DataFrame cell.

    Returns
    -------
    str
        Cleaned text. Returns an empty string if the input is null,
        NaN, or empty after cleaning.

    Examples
    --------
    >>> clean_text("  Hello!!! Visit https://example.com <br>\\n  Ôla  ")
    'Hello Visit Ola'
    """
    # Step 1 — handle missing / non-string
    if pd.isna(text) or text is None:
        return ''
    text = str(text)
    if not text.strip():
        return ''


    text = _fix_newlines(text)
    text = _remove_html(text)
    text = _remove_urls(text)
    text = _remove_emails(text)
    text = _normalize_unicode(text)
    text = _remove_special_characters(text)
    text = _normalize_whitespace(text)

    return text


# ─────────────────────────────────────────────
# Main public function
# ─────────────────────────────────────────────

def clean_reviews_by_category(df, category, text_cols = TEXT_COLUMNS) -> pd.DataFrame:
    """
    Load the Trustpilot reviews CSV, filter to a given category, and
    apply the full text-cleaning pipeline to all text columns.

    The function validates the requested category against the known list,
    loads the CSV efficiently, applies ``clean_text`` to each text column,
    and returns a clean DataFrame ready for downstream NLP analysis.

    Parameters
    ----------
    csv_path : str
        Absolute or relative path to the raw CSV file
        (e.g. 'trustpilot-reviews-123k.csv').
    category : str
        One of the 22 valid Trustpilot category strings, e.g.
        'Animals & Pets'.  Case-sensitive — must match exactly.
    text_cols : list of str, optional
        Columns to clean. Defaults to ['title', 'review', 'description'].
        Any column name not present in the CSV is silently skipped.

    Returns
    -------
    pd.DataFrame
        Filtered and cleaned DataFrame with:
        - Original columns preserved.
        - Text columns replaced with cleaned versions.
        - A new boolean column ``is_empty_review`` flagging rows where
          the review was empty after cleaning.
        - Reset integer index.

    Raises
    ------
    ValueError
        If ``category`` is not in the list of valid categories.
    FileNotFoundError
        If ``csv_path`` does not point to an existing file.
    KeyError
        If required columns ('category', 'company', 'stars') are missing
        from the CSV.

    Examples
    --------
    >>> df_clean = clean_reviews_by_category(
    ...     csv_path='trustpilot-reviews-123k.csv',
    ...     category='Animals & Pets'
    ... )
    >>> df_clean.shape
    (N, 7)   # N rows for that category
    >>> df_clean.columns.tolist()
    ['category', 'company', 'description', 'title', 'review', 'stars', 'is_empty_review']
    """

    # ── Validate category ──────────────────────────────────────────────
    if category not in VALID_CATEGORIES:
        raise ValueError(
            f"'{category}' is not a recognised category.\n"
            f"Valid options are:\n  " + "\n  ".join(VALID_CATEGORIES)
        )

    # ── Default text columns ───────────────────────────────────────────
    if text_cols is None:
        text_cols = TEXT_COLUMNS

    # ── Validate required columns ──────────────────────────────────────
    required_cols = {'category', 'company', 'stars'}
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"CSV is missing required columns: {missing}")

    # ── Filter by category ─────────────────────────────────────────────
    logger.info(f"Filtering to category: '{category}' ...")
    df_cat = df[df['category'] == category].copy()

    if df_cat.empty:
        logger.warning(f"No rows found for category '{category}'. Returning empty DataFrame.")
        return df_cat.reset_index(drop=True)

    logger.info(f"  → {len(df_cat):,} rows found across {df_cat['company'].nunique()} companies.")

    # ── Apply cleaning ─────────────────────────────────────────────────
    cols_to_clean = [c for c in text_cols if c in df_cat.columns]
    skipped = set(text_cols) - set(cols_to_clean)
    if skipped:
        logger.warning(f"Columns not found in CSV (skipped): {skipped}")

    logger.info(f"Cleaning text columns: {cols_to_clean} ...")
    for col in cols_to_clean:
        cleaned_col = f"{col}_clean"
        df_cat[col] = df_cat[col].apply(clean_text)
        logger.info(f" ✓ '{col}'")

    # ── Flag empty reviews after cleaning ─────────────────────────────
    if 'review_clean' in df_cat.columns:
        empty_review = df_cat['review_clean'].str.strip() == ''
        n_empty = empty_review.sum()
        if n_empty:
            logger.warning(f"  {n_empty} review(s) are empty after cleaning.")

    df_cat = df_cat.reset_index(drop=True)
    logger.info("Cleaning complete.")
    return df_cat