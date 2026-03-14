# SEC Sentiment Analysis — Predicting Stock Volatility from 10-K Filings

Claude Code was used to help with data pipelines, results analysis, and structuring code from our ideas. However, model selection and every choice for training was made by the team. We also selected the best models using our best judgement after validation results. Obviously, all analysis from the results themselves in the paper is also done by the team.

Predicts forward realized volatility from SEC 10-K Item 1A (Risk Factors) text using four feature representations and four model families.

## Overview

- **Data:** 304 firm-year observations, 50 S&P 500 companies, 2013–2019
- **Target:** Annualized realized volatility over 252 trading days following each filing (regression), binarized at the training-set 75th percentile (classification)
- **Train:** 2013–2017 (216 obs) | **Val:** 2018–2019 (88 obs)

## Feature Sets

| Name | Dim | Description |
|---|---|---|
| `financial` | 4 | Debt/equity, ROA, current ratio, log market cap |
| `sentiment` | 8 | Loughran-McDonald word-list ratios, Gunning Fog, log word count |
| `tfidf` | 500 | TF-IDF unigrams, fit on train only |
| `finbert` | 768 | Mean-pooled FinBERT chunk embeddings |
| `all` | 1280 | Concatenation of all above |

## Models

- Ridge Regression (volatility prediction)
- Logistic Regression (high/low volatility classification)
- Random Forest (classification)
- Custom PyTorch MLP with BatchNorm, Dropout, early stopping, and ReduceLROnPlateau

## Results (Validation AUC)

| Model | Feature Set | AUC | F1 |
|---|---|---|---|
| MLP | tfidf | 0.723 | 0.344 |
| LogReg | tfidf | 0.697 | 0.582 |
| MLP | finbert | 0.694 | 0.681 |
| RF | tfidf | 0.680 | 0.295 |
| Any | financial | ~0.50 | — |

Financial-only features perform at chance. TF-IDF text features yield a +0.20 AUC gain over the financial baseline. TF-IDF outperforms FinBERT (0.723 vs 0.694), likely due to limited training data.

## Setup

```bash
pip install -r requirements.txt
```

Set your SEC EDGAR identity before running the data pipeline:

```bash
export EDGAR_IDENTITY="Your Name your@email.com"
```

## Usage

Run cells in order in `notebooks/experiments.ipynb`:

1. **Data pipeline** — fetches 10-K text, volatility, and financial ratios (runs once, cached to `Data/raw/`)
2. **Feature engineering** — builds all feature matrices (FinBERT cached to `Data/features/`)
3. **Model training** — grid search for Ridge, LogReg, RF, and MLP
4. **Evaluation** — ROC curves, feature importance, comparison heatmap (saved to `results/figures/`)

## Structure

```
src/
  data_pipeline.py   — EDGAR + Yahoo Finance data collection
  features.py        — all feature engineering
  models.py          — all model training and grid search
  evaluate.py        — metrics and figures
notebooks/
  experiments.ipynb  — main run notebook
results/
  val_summary.csv    — validation metrics for all model/feature combinations
  figures/           — all generated plots
```
