"""
features.py

Builds four feature matrices from raw CSVs.
All transformers (scaler, TF-IDF vectorizer) are fit on the training set only,
then applied to validation — no leakage.

Feature sets:
    X_financial  (4)    — debt/equity, ROA, current ratio, log market cap
    X_sentiment  (8)    — Loughran-McDonald word-list ratios + Fog index + log word count
    X_tfidf      (500)  — TF-IDF unigrams, top 500 by document frequency
    X_finbert    (768)  — mean-pooled FinBERT embeddings (chunked for long docs)
"""

import os
import re

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR     = os.path.join(BASE_DIR, "Data")
RAW_DIR      = os.path.join(DATA_DIR, "raw")
LM_PATH      = os.path.join(DATA_DIR, "loughran_mcdonald.csv")
FEATURES_DIR = os.path.join(DATA_DIR, "features")

os.makedirs(FEATURES_DIR, exist_ok=True)

TRAIN_YEARS    = list(range(2010, 2018))
FINBERT_MODEL  = "yiyanghkust/finbert-pretrain"
MAX_TOKENS     = 512       # FinBERT hard limit


# ── Load raw data ──────────────────────────────────────────────────────────────

def load_raw():
    """
    Merge filings, targets, and financials; split by year into train / val.
    Returns (df_train, df_val).
    """
    df_filings    = pd.read_csv(os.path.join(RAW_DIR, "filings.csv"))
    df_targets    = pd.read_csv(os.path.join(RAW_DIR, "targets.csv"))
    df_financials = pd.read_csv(os.path.join(RAW_DIR, "financials.csv"))

    df = (df_filings
          .merge(df_targets[["ticker", "year", "volatility", "high_volatility"]],
                 on=["ticker", "year"])
          .merge(df_financials, on=["ticker", "year"]))

    train_mask = df["year"].isin(TRAIN_YEARS)
    return df[train_mask].reset_index(drop=True), df[~train_mask].reset_index(drop=True)


# ── Feature Set 1: Financial ratios ───────────────────────────────────────────

def build_financial_features(df_train: pd.DataFrame, df_val: pd.DataFrame):
    """
    Standardize the four financial ratio columns.
    Missing values are imputed with per-column training medians before scaling.
    Returns (X_train, X_val, fitted_scaler).
    """
    cols = ["debt_equity", "roa", "current_ratio", "log_mktcap"]

    X_train = df_train[cols].values.astype(float)
    X_val   = df_val[cols].values.astype(float)

    # Median imputation (fit on train); fall back to 0 if entire column is NaN
    medians = np.nanmedian(X_train, axis=0)
    medians = np.where(np.isnan(medians), 0.0, medians)
    for i in range(X_train.shape[1]):
        X_train[np.isnan(X_train[:, i]), i] = medians[i]
        X_val[np.isnan(X_val[:, i]), i]     = medians[i]

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)

    return X_train, X_val, scaler


# ── Feature Set 2: Sentiment (Loughran-McDonald) — implemented from scratch ───

def load_lm_wordlists(path: str = LM_PATH) -> dict:
    """
    Parse the LM master dictionary CSV.
    Returns {category_name: set_of_uppercase_words}.
    Non-zero value in the category column = word belongs to that category.
    """
    df = pd.read_csv(path)
    categories = ["Negative", "Positive", "Uncertainty",
                  "Litigious", "Strong_Modal", "Weak_Modal"]
    return {
        cat: set(df.loc[df[cat] != 0, "Word"].str.upper())
        for cat in categories
        if cat in df.columns
    }


def _count_syllables(word: str) -> int:
    """Approximate syllable count by counting vowel groups (fast heuristic)."""
    word = word.lower().rstrip("e")          # silent trailing 'e'
    return max(1, len(re.findall(r"[aeiou]+", word)))


def _gunning_fog(tokens: list, raw_text: str) -> float:
    """
    Gunning Fog readability index.
    = 0.4 * (words/sentences  +  100 * complex_words/words)
    complex word = 3+ syllables.
    """
    n_words     = len(tokens)
    n_sentences = max(1, len(re.split(r"[.!?]+", raw_text)))
    if n_words == 0:
        return 0.0
    n_complex = sum(1 for w in tokens if _count_syllables(w) >= 3)
    return 0.4 * (n_words / n_sentences + 100 * n_complex / n_words)


def extract_sentiment_features(text: str, wordlists: dict) -> list:
    """
    8 features per document:
        neg%, pos%, uncertainty%, litigious%, strong_modal%, weak_modal%
        Fog index
        log(word_count + 1)
    All word-list ratios are raw_count / total_words.
    """
    tokens    = re.findall(r"[A-Za-z']+", text.upper())
    n_words   = len(tokens)

    if n_words == 0:
        return [0.0] * 8

    token_list = tokens   # already uppercase

    features = []
    for cat in ["Negative", "Positive", "Uncertainty",
                "Litigious", "Strong_Modal", "Weak_Modal"]:
        wl = wordlists.get(cat, set())
        features.append(sum(1 for t in token_list if t in wl) / n_words)

    features.append(_gunning_fog(tokens, text))
    features.append(np.log1p(n_words))

    return features


def build_sentiment_features(df_train: pd.DataFrame, df_val: pd.DataFrame):
    """
    Extract 8 sentiment features per document from scratch using LM word lists.
    Returns (X_train, X_val, fitted_scaler).
    """
    wordlists = load_lm_wordlists()

    def _batch(df):
        return np.array([
            extract_sentiment_features(row["item1a_text"], wordlists)
            for _, row in tqdm(df.iterrows(), total=len(df),
                               desc="  Sentiment", leave=False)
        ])

    X_train = _batch(df_train)
    X_val   = _batch(df_val)

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)

    return X_train, X_val, scaler


# ── Feature Set 3: TF-IDF ─────────────────────────────────────────────────────

def build_tfidf_features(df_train: pd.DataFrame, df_val: pd.DataFrame,
                         max_features: int = 500):
    """
    TF-IDF over unigrams.  Vectorizer is fit on the training corpus only.
    sublinear_tf=True applies log(1+tf) to dampen frequency effects.
    Returns (X_train, X_val, fitted_vectorizer).
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words="english",
        min_df=5,
        sublinear_tf=True,
    )
    X_train = vectorizer.fit_transform(df_train["item1a_text"]).toarray()
    X_val   = vectorizer.transform(df_val["item1a_text"]).toarray()

    return X_train, X_val, vectorizer


# ── Feature Set 4: FinBERT embeddings ─────────────────────────────────────────

def _embed_single(text: str, tokenizer, model, device) -> np.ndarray:
    """
    Embed one document via FinBERT.

    Long documents are split into non-overlapping chunks of (MAX_TOKENS - 2)
    tokens (leaving room for [CLS] and [SEP]).  Each chunk's token embeddings
    are mean-pooled to a 768-dim vector; chunk vectors are then averaged to
    produce a single document embedding.
    """
    enc    = tokenizer(text, return_tensors="pt", truncation=False,
                       add_special_tokens=False)
    ids    = enc["input_ids"][0]              # (total_tokens,)

    chunk_size = MAX_TOKENS - 2
    chunks     = [ids[i: i + chunk_size] for i in range(0, len(ids), chunk_size)]

    chunk_vecs = []
    with torch.no_grad():
        for chunk in chunks:
            input_ids = torch.cat([
                torch.tensor([tokenizer.cls_token_id]),
                chunk,
                torch.tensor([tokenizer.sep_token_id]),
            ]).unsqueeze(0).to(device)

            attn_mask = torch.ones_like(input_ids)
            out       = model(input_ids=input_ids, attention_mask=attn_mask)

            # Mean-pool token embeddings, excluding [CLS] and [SEP]
            hidden = out.last_hidden_state[0, 1:-1, :]    # (seq, 768)
            chunk_vecs.append(hidden.mean(dim=0).cpu().numpy())

    return np.mean(chunk_vecs, axis=0)   # (768,)


def build_finbert_features(df_train: pd.DataFrame, df_val: pd.DataFrame,
                            cache_dir: str = FEATURES_DIR):
    """
    Compute FinBERT embeddings with chunk-based mean pooling.
    Results are cached to disk so this only runs once.
    Returns (X_train_scaled, X_val_scaled).
    """
    cache_tr  = os.path.join(cache_dir, "X_finbert_train.npy")
    cache_val = os.path.join(cache_dir, "X_finbert_val.npy")

    if os.path.exists(cache_tr) and os.path.exists(cache_val):
        print("  Loading FinBERT embeddings from cache...")
        X_train = np.load(cache_tr)
        X_val   = np.load(cache_val)
    else:
        device = (
            torch.device("cuda") if torch.cuda.is_available()
            else torch.device("mps") if torch.backends.mps.is_available()
            else torch.device("cpu")
        )
        print(f"  Computing FinBERT embeddings on {device}  (cached after first run)...")

        tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
        model     = AutoModel.from_pretrained(FINBERT_MODEL).to(device).eval()

        def _embed_batch(df):
            return np.array([
                _embed_single(row["item1a_text"], tokenizer, model, device)
                for _, row in tqdm(df.iterrows(), total=len(df),
                                   desc="  FinBERT", leave=False)
            ])

        X_train = _embed_batch(df_train)
        X_val   = _embed_batch(df_val)

        np.save(cache_tr,  X_train)
        np.save(cache_val, X_val)
        print(f"  Cached to {cache_dir}")

    # Normalize (fit scaler on train only)
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)

    return X_train, X_val


# ── Build all feature sets ─────────────────────────────────────────────────────

def build_all_features():
    """
    Orchestrates all feature engineering.
    Returns:
        feature_sets : dict  {name: (X_train, X_val)}
        targets      : dict  {task: (y_train, y_val)}
        df_train, df_val
    """
    print("Loading raw data...")
    df_train, df_val = load_raw()
    print(f"  Train: {len(df_train)} | Val: {len(df_val)}\n")

    print("Building X_financial...")
    X_fin_tr, X_fin_val, _   = build_financial_features(df_train, df_val)

    print("Building X_sentiment...")
    X_sent_tr, X_sent_val, _ = build_sentiment_features(df_train, df_val)

    print("Building X_tfidf...")
    X_tfidf_tr, X_tfidf_val, _ = build_tfidf_features(df_train, df_val)

    print("Building X_finbert...")
    X_fb_tr, X_fb_val = build_finbert_features(df_train, df_val)

    # Combined: stack all feature sets
    X_all_tr  = np.hstack([X_fin_tr,  X_sent_tr,  X_tfidf_tr,  X_fb_tr])
    X_all_val = np.hstack([X_fin_val, X_sent_val, X_tfidf_val, X_fb_val])

    feature_sets = {
        "financial": (X_fin_tr,   X_fin_val),
        "sentiment": (X_sent_tr,  X_sent_val),
        "tfidf":     (X_tfidf_tr, X_tfidf_val),
        "finbert":   (X_fb_tr,    X_fb_val),
        "all":       (X_all_tr,   X_all_val),
    }

    targets = {
        "regression":     (df_train["volatility"].values.astype(float),
                           df_val["volatility"].values.astype(float)),
        "classification": (df_train["high_volatility"].values.astype(int),
                           df_val["high_volatility"].values.astype(int)),
    }

    print("\nFeature shapes (train):")
    for name, (X_tr, _) in feature_sets.items():
        print(f"  {name:12s}: {X_tr.shape}")

    return feature_sets, targets, df_train, df_val


if __name__ == "__main__":
    build_all_features()
