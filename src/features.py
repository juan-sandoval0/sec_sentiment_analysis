"""
features.py - builds four feature matrices from raw CSVs.
scalers/vectorizers fit on train only.

feature sets:
    financial (4):  debt/equity, ROA, current ratio, log market cap
    sentiment (8):  Loughran-McDonald word ratios + fog index + log word count
    tfidf     (500): TF-IDF unigrams
    finbert   (768): mean-pooled FinBERT chunk embeddings
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

TRAIN_YEARS    = list(range(2013, 2018))
VAL_YEARS      = list(range(2018, 2020))
TEST_YEARS     = list(range(2020, 2024))   # held-out
FINBERT_MODEL  = "yiyanghkust/finbert-pretrain"
MAX_TOKENS     = 512       # FinBERT hard limit


def load_raw():
    df_filings    = pd.read_csv(os.path.join(RAW_DIR, "filings.csv"))
    df_targets    = pd.read_csv(os.path.join(RAW_DIR, "targets.csv"))
    df_financials = pd.read_csv(os.path.join(RAW_DIR, "financials.csv"))

    df = (df_filings
          .merge(df_targets[["ticker", "year", "volatility",
                              "high_volatility", "vol_threshold"]],
                 on=["ticker", "year"])
          .merge(df_financials, on=["ticker", "year"]))

    train_mask = df["year"].isin(TRAIN_YEARS)
    val_mask   = df["year"].isin(VAL_YEARS)
    test_mask  = df["year"].isin(TEST_YEARS)

    return (df[train_mask].reset_index(drop=True),
            df[val_mask].reset_index(drop=True),
            df[test_mask].reset_index(drop=True))


def build_financial_features(df_train, df_val, df_test):
    cols = ["debt_equity", "roa", "current_ratio", "log_mktcap"]

    X_train = df_train[cols].values.astype(float)
    X_val   = df_val[cols].values.astype(float)
    X_test  = df_test[cols].values.astype(float) if len(df_test) > 0 else np.zeros((0, len(cols)))

    # impute with train medians only, fall back to 0 if whole column is NaN
    medians = np.nanmedian(X_train, axis=0)
    medians = np.where(np.isnan(medians), 0.0, medians)
    for i in range(X_train.shape[1]):
        X_train[np.isnan(X_train[:, i]), i] = medians[i]
        X_val[np.isnan(X_val[:, i]), i]     = medians[i]
        if len(X_test) > 0:
            X_test[np.isnan(X_test[:, i]), i] = medians[i]

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test) if len(X_test) > 0 else X_test

    return X_train, X_val, X_test, scaler


def load_lm_wordlists(path: str = LM_PATH) -> dict:
    df = pd.read_csv(path)
    categories = ["Negative", "Positive", "Uncertainty",
                  "Litigious", "Strong_Modal", "Weak_Modal"]
    return {
        cat: set(df.loc[df[cat] != 0, "Word"].str.upper())
        for cat in categories
        if cat in df.columns
    }


def _count_syllables(word: str) -> int:
    """rough syllable count, good enough for fog index"""
    word = word.lower().rstrip("e")   # silent trailing 'e'
    return max(1, len(re.findall(r"[aeiou]+", word)))


def _gunning_fog(tokens: list, raw_text: str) -> float:
    """0.4 * (words/sentences + 100 * complex_words/words), complex = 3+ syllables"""
    n_words     = len(tokens)
    n_sentences = max(1, len(re.split(r"[.!?]+", raw_text)))
    if n_words == 0:
        return 0.0
    n_complex = sum(1 for w in tokens if _count_syllables(w) >= 3)
    return 0.4 * (n_words / n_sentences + 100 * n_complex / n_words)


def extract_sentiment_features(text: str, wordlists: dict) -> list:
    """neg%, pos%, uncertainty%, litigious%, strong_modal%, weak_modal%, fog, log(word_count+1)"""
    tokens  = re.findall(r"[A-Za-z']+", text.upper())
    n_words = len(tokens)

    if n_words == 0:
        return [0.0] * 8

    features = []
    for cat in ["Negative", "Positive", "Uncertainty",
                "Litigious", "Strong_Modal", "Weak_Modal"]:
        wl = wordlists.get(cat, set())
        features.append(sum(1 for t in tokens if t in wl) / n_words)

    features.append(_gunning_fog(tokens, text))
    features.append(np.log1p(n_words))

    return features


def build_sentiment_features(df_train, df_val, df_test):
    wordlists = load_lm_wordlists()

    def _batch(df, label=""):
        if len(df) == 0:
            return np.zeros((0, 8))
        return np.array([
            extract_sentiment_features(row["item1a_text"], wordlists)
            for _, row in tqdm(df.iterrows(), total=len(df),
                               desc=f"  Sentiment ({label})", leave=False)
        ])

    X_train = _batch(df_train, "train")
    X_val   = _batch(df_val,   "val")
    X_test  = _batch(df_test,  "test")

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test) if len(X_test) > 0 else X_test

    return X_train, X_val, X_test, scaler


def build_tfidf_features(df_train, df_val, df_test, max_features: int = 500):
    """sublinear_tf=True uses log(1+tf) to dampen high-frequency terms"""
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words="english",
        min_df=5,
        sublinear_tf=True,
    )
    X_train = vectorizer.fit_transform(df_train["item1a_text"]).toarray()
    X_val   = vectorizer.transform(df_val["item1a_text"]).toarray()
    X_test  = (vectorizer.transform(df_test["item1a_text"]).toarray()
               if len(df_test) > 0 else np.zeros((0, max_features)))

    return X_train, X_val, X_test, vectorizer


def _embed_single(text: str, tokenizer, model, device) -> np.ndarray:
    """10-K filings are too long for one forward pass, so chunk and mean-pool"""
    enc    = tokenizer(text, return_tensors="pt", truncation=False,
                       add_special_tokens=False)
    ids    = enc["input_ids"][0]

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

            # skip [CLS] and [SEP]
            hidden = out.last_hidden_state[0, 1:-1, :]
            chunk_vecs.append(hidden.mean(dim=0).cpu().numpy())

    return np.mean(chunk_vecs, axis=0)


def build_finbert_features(df_train, df_val, df_test, cache_dir: str = FEATURES_DIR):
    """FinBERT embeddings with chunked mean pooling, cached per split"""
    cache_tr   = os.path.join(cache_dir, "X_finbert_train.npy")
    cache_val  = os.path.join(cache_dir, "X_finbert_val.npy")
    cache_test = os.path.join(cache_dir, "X_finbert_test.npy")

    need_model = (not os.path.exists(cache_tr) or
                  not os.path.exists(cache_val) or
                  (len(df_test) > 0 and not os.path.exists(cache_test)))

    tokenizer = model_fb = None
    if need_model:
        device = (
            torch.device("cuda") if torch.cuda.is_available()
            else torch.device("mps") if torch.backends.mps.is_available()
            else torch.device("cpu")
        )
        print(f"  Computing FinBERT embeddings on {device}...")
        tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
        model_fb  = AutoModel.from_pretrained(FINBERT_MODEL).to(device).eval()
    else:
        print("  Loading FinBERT from cache...")

    def _embed_batch(df, label):
        return np.array([
            _embed_single(row["item1a_text"], tokenizer, model_fb, device)
            for _, row in tqdm(df.iterrows(), total=len(df),
                               desc=f"  FinBERT ({label})", leave=False)
        ])

    if os.path.exists(cache_tr):
        X_train = np.load(cache_tr)
    else:
        X_train = _embed_batch(df_train, "train")
        np.save(cache_tr, X_train)

    if os.path.exists(cache_val):
        X_val = np.load(cache_val)
    else:
        X_val = _embed_batch(df_val, "val")
        np.save(cache_val, X_val)

    if len(df_test) == 0:
        X_test = np.zeros((0, 768))
    elif os.path.exists(cache_test):
        X_test = np.load(cache_test)
    else:
        X_test = _embed_batch(df_test, "test")
        np.save(cache_test, X_test)

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test) if len(X_test) > 0 else X_test

    return X_train, X_val, X_test


def build_all_features():
    print("Loading raw data...")
    df_train, df_val, df_test = load_raw()
    print(f"  Train: {len(df_train)} | Val: {len(df_val)} | Test: {len(df_test)}\n")

    print("Building X_financial...")
    X_fin_tr, X_fin_val, X_fin_te, _     = build_financial_features(df_train, df_val, df_test)

    print("Building X_sentiment...")
    X_sent_tr, X_sent_val, X_sent_te, _  = build_sentiment_features(df_train, df_val, df_test)

    print("Building X_tfidf...")
    X_tfidf_tr, X_tfidf_val, X_tfidf_te, _ = build_tfidf_features(df_train, df_val, df_test)

    print("Building X_finbert...")
    X_fb_tr, X_fb_val, X_fb_te = build_finbert_features(df_train, df_val, df_test)

    X_all_tr  = np.hstack([X_fin_tr,  X_sent_tr,  X_tfidf_tr,  X_fb_tr])
    X_all_val = np.hstack([X_fin_val, X_sent_val, X_tfidf_val, X_fb_val])
    X_all_te  = (np.hstack([X_fin_te, X_sent_te, X_tfidf_te, X_fb_te])
                 if len(df_test) > 0 else np.zeros((0, X_all_tr.shape[1])))

    feature_sets = {
        "financial": (X_fin_tr,   X_fin_val,   X_fin_te),
        "sentiment": (X_sent_tr,  X_sent_val,  X_sent_te),
        "tfidf":     (X_tfidf_tr, X_tfidf_val, X_tfidf_te),
        "finbert":   (X_fb_tr,    X_fb_val,    X_fb_te),
        "all":       (X_all_tr,   X_all_val,   X_all_te),
    }

    targets = {
        "regression": (
            df_train["volatility"].values.astype(float),
            df_val["volatility"].values.astype(float),
            df_test["volatility"].values.astype(float) if len(df_test) > 0 else np.array([]),
        ),
        "classification": (
            df_train["high_volatility"].values.astype(int),
            df_val["high_volatility"].values.astype(int),
            df_test["high_volatility"].values.astype(int) if len(df_test) > 0 else np.array([], dtype=int),
        ),
    }

    print("\nFeature shapes (train):")
    for name, (X_tr, _, _) in feature_sets.items():
        print(f"  {name:12s}: {X_tr.shape}")

    return feature_sets, targets, df_train, df_val, df_test


if __name__ == "__main__":
    build_all_features()
