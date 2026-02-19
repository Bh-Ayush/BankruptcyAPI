"""
Bankruptcy Prediction API — FastAPI Backend
Trains a LightGBM model on startup from the combined 128K-row dataset
(US + Taiwan + Poland). No pickle files required.
"""

import os
import sys
import json
import logging
import warnings
import re
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

warnings.filterwarnings("ignore")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    precision_score, recall_score
)

import lightgbm as lgb
from imblearn.over_sampling import SMOTE

# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bankruptcy-api")

# ─── App ─────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Bankruptcy Risk Prediction API",
    description="Real predictions from LightGBM trained on 128,906 companies (US + Taiwan + Poland)",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Global State ────────────────────────────────────────────────────────────
SEED = 42
trained_model = None
preproc = None
feature_columns = None
numeric_cols = None
cat_cols = None
model_metrics = None

# ─── Data Paths ──────────────────────────────────────────────────────────────
POSSIBLE_PATHS = [
    "combined_raw.csv",
    "data/combined_raw.csv",
    "../data/combined_raw.csv",
    os.path.join(os.path.dirname(__file__), "combined_raw.csv"),
    os.path.join(os.path.dirname(__file__), "data", "combined_raw.csv"),
]


def find_data():
    for p in POSSIBLE_PATHS:
        if os.path.exists(p):
            return p
    return None


# ─── Training ────────────────────────────────────────────────────────────────
def train_model():
    global trained_model, preproc, feature_columns, numeric_cols, cat_cols, model_metrics

    data_path = find_data()
    if data_path is None:
        logger.warning("combined_raw.csv not found! Place it next to main.py or in data/")
        return False

    logger.info(f"Loading {data_path}...")
    df = pd.read_csv(data_path)
    logger.info(f"Loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    # ── Find target ──
    if "y" not in df.columns:
        logger.error("No 'y' column found in combined_raw.csv")
        return False

    # ── Drop leaky columns ──
    drop_cols = ["status_label"]
    for c in drop_cols:
        if c in df.columns:
            df = df.drop(columns=[c])
            logger.info(f"Dropped leaky column: {c}")

    # ── Drop columns with >95% correlation (same logic as your notebook) ──
    num_only = df.select_dtypes(include=["number"]).drop(columns=["y"], errors="ignore")
    corr_matrix = num_only.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr_cols = [col for col in upper.columns if any(upper[col] > 0.95)]
    if high_corr_cols:
        df = df.drop(columns=high_corr_cols, errors="ignore")
        logger.info(f"Dropped {len(high_corr_cols)} highly correlated columns")

    # ── Separate X / y ──
    y = df["y"].astype(int)
    X = df.drop(columns=["y"])

    # ── Identify column types ──
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in numeric_cols and c != "dataset_source"]
    ohe_cols = ["dataset_source"] if "dataset_source" in X.columns else []

    # Store feature columns for mapping later
    feature_columns = list(X.columns)
    logger.info(f"Features: {len(numeric_cols)} numeric, {len(cat_cols)} categorical, {len(ohe_cols)} OHE")

    # ── Preprocessing pipeline (matches your notebook exactly) ──
    num_tf = Pipeline([("imputer", SimpleImputer(strategy="median"))])

    ohe_tf = Pipeline([("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])

    from sklearn.preprocessing import FunctionTransformer

    def to_str(X):
        return X.astype(str)

    cat_tf = Pipeline([
        ("to_str", FunctionTransformer(to_str)),
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])

    transformers = [("num", num_tf, numeric_cols)]
    if ohe_cols:
        transformers.append(("ohe_src", ohe_tf, ohe_cols))
    if cat_cols:
        transformers.append(("cat", cat_tf, cat_cols))

    preproc = ColumnTransformer(transformers, remainder="drop")

    # ── Stratified split per dataset source ──
    train_parts, test_parts = [], []
    for src in X["dataset_source"].unique() if "dataset_source" in X.columns else ["all"]:
        if src == "all":
            mask = pd.Series(True, index=X.index)
        else:
            mask = X["dataset_source"] == src
        X_src = X[mask]
        y_src = y[mask]
        if len(X_src) < 10:
            train_parts.append((X_src, y_src))
            continue
        Xtr, Xte, ytr, yte = train_test_split(
            X_src, y_src, test_size=0.2, stratify=y_src, random_state=SEED
        )
        train_parts.append((Xtr, ytr))
        test_parts.append((Xte, yte))

    X_train = pd.concat([p[0] for p in train_parts], ignore_index=True)
    y_train = pd.concat([p[1] for p in train_parts], ignore_index=True)
    X_test = pd.concat([p[0] for p in test_parts], ignore_index=True)
    y_test = pd.concat([p[1] for p in test_parts], ignore_index=True)

    logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}, Positive rate: {y_train.mean():.4f}")

    # ── Fit preprocessor and transform ──
    preproc.fit(X_train)
    X_train_t = preproc.transform(X_train)
    X_test_t = preproc.transform(X_test)

    # ── SMOTE ──
    sm = SMOTE(random_state=SEED)
    X_train_res, y_train_res = sm.fit_resample(X_train_t, y_train)
    logger.info(f"After SMOTE: {len(X_train_res)} training samples")

    # ── Internal validation split for early stopping ──
    X_tr_f, X_val_f, y_tr_f, y_val_f = train_test_split(
        X_train_res, y_train_res, test_size=0.15, random_state=SEED, stratify=y_train_res
    )

    # ── Train LightGBM ──
    trained_model = lgb.LGBMClassifier(
        objective="binary",
        random_state=SEED,
        n_estimators=3000,
        learning_rate=0.03,
        num_leaves=31,
        n_jobs=-1,
        verbose=-1,
    )
    trained_model.fit(
        X_tr_f, y_tr_f,
        eval_set=[(X_val_f, y_val_f)],
        eval_metric="auc",
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    # ── Evaluate on held-out test ──
    proba = trained_model.predict_proba(X_test_t)[:, 1]
    preds = (proba >= 0.5).astype(int)

    model_metrics = {
        "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
        "accuracy": round(float(accuracy_score(y_test, preds)), 4),
        "precision": round(float(precision_score(y_test, preds, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, preds)), 4),
        "f1": round(float(f1_score(y_test, preds)), 4),
        "training_samples": int(len(X_train_res)),
        "test_samples": int(len(X_test)),
        "features": int(X_train_t.shape[1]),
        "datasets": "US + Taiwan + Poland (128,906 companies)",
        "best_iteration": int(trained_model.best_iteration_),
    }
    logger.info(f"Model trained! {json.dumps(model_metrics, indent=2)}")
    return True


# ─── Map user financial inputs → model features ─────────────────────────────
def map_user_inputs_to_features(data: dict) -> tuple:
    """
    Convert financial statement inputs to a DataFrame matching
    the combined dataset's columns. Unmapped features stay NaN
    and get median-imputed by the preprocessor.
    """
    if feature_columns is None:
        raise RuntimeError("Model not trained")

    row = {col: np.nan for col in feature_columns}

    # Metadata
    row["dataset_source"] = "us"
    if "horizon_years" in row:
        row["horizon_years"] = 1

    # Raw inputs
    ta = data.get("totalAssets", 0) or 1
    tl = data.get("totalLiabilities", 0) or 1
    ca = data.get("currentAssets", 0) or 0
    cl = data.get("currentLiabilities", 0) or 0
    te = data.get("totalEquity") or (ta - tl)
    td = data.get("totalDebt", 0) or 0
    rev = data.get("revenue", 0) or 1
    ni = data.get("netIncome", 0) or 0
    ebit = data.get("ebit", 0) or 0
    ie = data.get("interestExpense", 0) or 0
    cfo = data.get("cashFromOperations", 0) or 0
    re_val = data.get("retainedEarnings", 0) or 0
    dep = data.get("depreciation", 0) or 0
    cash = data.get("cash") or (ca * 0.3)
    wc = ca - cl
    gp = rev * 0.4  # estimate without COGS

    def safe(n, d):
        return n / d if d and d != 0 else np.nan

    # ── US features (X1-X18) ──
    us_map = {
        "X1": safe(ni, ta),
        "X2": safe(ni, rev),
        "X3": safe(ca, cl),
        "X4": safe(tl, ta),
        "X5": safe(td, te),
        "X6": safe(ebit, ta),
        "X7": safe(rev, ta),
        "X8": safe(wc, ta),
        "X10": safe(ebit, ie),
        "X11": safe(cfo, td),
        "X13": safe(ni, te),
        "X14": safe(ca, ta),
        "X15": safe(cl, tl),
        "X17": safe(gp, rev),
        "X18": safe(ebit, rev),
    }

    # ── Poland features (Attr1-64) ──
    poland_map = {
        "Attr1": safe(ni, ta),
        "Attr2": safe(tl, ta),
        "Attr3": safe(wc, ta),
        "Attr4": safe(ca, cl),
        "Attr5": safe(re_val, ta),
        "Attr6": safe(ebit, ta),
        "Attr7": safe(te, tl),
        "Attr8": safe(rev, ta),
        "Attr9": safe(ni, rev),
        "Attr10": safe(te, ta),
        "Attr11": safe(cfo, rev),
        "Attr12": safe(gp, rev),
        "Attr13": safe(ebit, ie),
        "Attr14": safe(ni, te),
        "Attr15": safe(td, te),
        "Attr16": safe(cfo, tl),
        "Attr17": safe(ca, ta),
        "Attr18": safe(cl, ta),
        "Attr19": safe(ebit + dep, tl),
        "Attr20": safe(rev, ca),
        "Attr21": safe(cfo, ta),
        "Attr22": safe(ni + dep, td),
        "Attr23": safe(cl, tl),
        "Attr24": safe(wc, rev),
        "Attr27": safe(ni, ebit),
        "Attr34": safe(cfo, cl),
        "Attr37": safe(ta - ca, ta),
        "Attr38": safe(cl, te),
        "Attr39": safe(ni, rev),
        "Attr44": safe(ca - cl, rev - gp) if (rev - gp) != 0 else np.nan,
        "Attr55": safe(wc, ta),
        "Attr58": safe(td, ta),
        "Attr59": safe(cfo, rev),
        "Attr60": safe(rev, ta),
    }

    # ── Taiwan features (named columns) ──
    # Match by checking if these substrings appear in the actual column names
    taiwan_ratio_map = {
        "current ratio": safe(ca, cl),
        "quick ratio": safe(ca * 0.7, cl),
        "debt ratio": safe(tl, ta),
        "total debt/total net worth": safe(td, te),
        "interest expense ratio": safe(ie, rev),
        "working capital to total assets": safe(wc, ta),
        "current assets/total assets": safe(ca, ta),
        "cash / total assets": safe(cash, ta),
        "Retained Earnings/Total assets": safe(re_val, ta),
        "current liability to assets": safe(cl, ta),
        "net income to total assets": safe(ni, ta),
        "Cash flow to Sales": safe(cfo, rev),
        "Cash flow to total assets": safe(cfo, ta),
        "cash flow to liability": safe(cfo, tl),
        "CFO to ASSETS": safe(cfo, ta),
        "total asset turnover": safe(rev, ta),
        "operating gross margin": safe(gp, rev),
        "Net income to stockholder": safe(ni, te),
        "equity to liability": safe(te, tl),
        "Interest coverage ratio": safe(ebit, ie),
        "current liability / liability": safe(cl, tl),
        "current liability/equity": safe(cl, te),
        "Gross profit to Sales": safe(gp, rev),
        "liability to equity": safe(tl, te),
        "ROA(C) before interest and depreciation before interest": safe(ni + ie + dep, ta),
        "ROA(A) before interest and % after tax": safe(ni + ie, ta),
        "net worth/assets": safe(te, ta),
        "borrowing dependency": safe(td, ta),
        "fix assets to assets": safe(ta - ca, ta),
        "cash reinvestment": safe(cfo, ta),
        "cash / current liability": safe(cash, cl),
        "Quick asset /current liabilities": safe(ca * 0.7, cl),
        "Quick asset/Total asset": safe(ca * 0.7, ta),
        "operating funds to liability": safe(cfo, tl),
        "cash flow to equity": safe(cfo, te),
        "current liabilities to current assets": safe(cl, ca),
        "working capital/equity": safe(wc, te),
        "long-term liability to current assets": safe(tl - cl, ca),
        "equity to long-term liability": safe(te, tl - cl) if (tl - cl) != 0 else np.nan,
        "Degree of financial leverage": safe(ebit, ebit - ie) if (ebit - ie) != 0 else np.nan,
        "total expense /assets": safe(rev - ni, ta),
        "total income / total expense": safe(rev, rev - ni) if (rev - ni) != 0 else np.nan,
        "total asset growth rate": 0.0,
        "one if total liabilities exceeds total assets": 1.0 if tl > ta else 0.0,
        "one if net income was negative": 1.0 if ni < 0 else 0.0,
    }

    # ── Apply all mappings ──
    mapped = 0

    # Direct column name matches (US, Poland)
    for col_name, val in {**us_map, **poland_map}.items():
        if col_name in row and val is not None and not (isinstance(val, float) and np.isnan(val)):
            row[col_name] = val
            mapped += 1

    # Fuzzy match for Taiwan columns (column names are messy with spaces/special chars)
    for feature_col in feature_columns:
        if not (isinstance(row[feature_col], float) and np.isnan(row[feature_col])):
            continue  # already mapped
        col_lower = feature_col.strip().lower()
        for pattern, val in taiwan_ratio_map.items():
            if val is None or (isinstance(val, float) and np.isnan(val)):
                continue
            if pattern.lower() in col_lower or col_lower in pattern.lower():
                row[feature_col] = val
                mapped += 1
                break

    df = pd.DataFrame([row])[feature_columns]
    logger.info(f"Mapped {mapped} features from user inputs")
    return df, mapped


# ─── Altman Z-Score ──────────────────────────────────────────────────────────
def compute_altman_z(data: dict) -> dict:
    ta = data.get("totalAssets", 0) or 1
    tl = data.get("totalLiabilities", 0) or 1
    ca = data.get("currentAssets", 0) or 0
    cl = data.get("currentLiabilities", 0) or 0
    re_val = data.get("retainedEarnings", 0) or 0
    ebit = data.get("ebit", 0) or 0
    rev = data.get("revenue", 0) or 0
    te = data.get("totalEquity") or (ta - tl)
    wc = ca - cl

    wc_ta = wc / ta
    re_ta = re_val / ta
    ebit_ta = ebit / ta
    eq_tl = te / tl if tl else 0
    sales_ta = rev / ta

    z = 0.717 * wc_ta + 0.847 * re_ta + 3.107 * ebit_ta + 0.42 * eq_tl + 0.998 * sales_ta

    if z > 2.9:
        zone, label = "safe", "Safe Zone"
    elif z > 1.23:
        zone, label = "grey", "Grey Zone"
    else:
        zone, label = "distress", "Distress Zone"

    return {
        "z_score": round(z, 4),
        "zone": zone,
        "label": label,
        "components": {
            "wc_ta": round(wc_ta, 4),
            "re_ta": round(re_ta, 4),
            "ebit_ta": round(ebit_ta, 4),
            "equity_tl": round(eq_tl, 4),
            "sales_ta": round(sales_ta, 4),
        },
    }


# ─── Request / Response ─────────────────────────────────────────────────────
class FinancialInput(BaseModel):
    companyName: str = "Unknown Company"
    totalAssets: float = 0
    totalLiabilities: float = 0
    currentAssets: float = 0
    currentLiabilities: float = 0
    totalEquity: Optional[float] = None
    totalDebt: float = 0
    revenue: float = 0
    netIncome: float = 0
    ebit: float = 0
    interestExpense: float = 0
    cashFromOperations: float = 0
    retainedEarnings: float = 0
    depreciation: Optional[float] = 0
    marketValueEquity: Optional[float] = None
    cash: Optional[float] = None


class PredictionResponse(BaseModel):
    company_name: str
    model_prediction: Optional[dict] = None
    altman_z: dict
    features_mapped: int
    features_total: int
    model_loaded: bool


# ─── Endpoints ───────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    success = train_model()
    if success:
        logger.info("Model ready!")
    else:
        logger.warning("Training failed — Altman Z-Score still works.")


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": trained_model is not None,
        "features": model_metrics.get("features", 0) if model_metrics else 0,
        "metrics": model_metrics,
    }


@app.post("/api/predict", response_model=PredictionResponse)
async def predict(inputs: FinancialInput):
    data = inputs.dict()
    if data["totalEquity"] is None:
        data["totalEquity"] = data["totalAssets"] - data["totalLiabilities"]
    if data["marketValueEquity"] is None:
        data["marketValueEquity"] = data["totalEquity"]

    altman = compute_altman_z(data)

    model_pred = None
    features_mapped = 0
    features_total = model_metrics.get("features", 0) if model_metrics else 0

    if trained_model is not None and preproc is not None:
        try:
            df_raw, features_mapped = map_user_inputs_to_features(data)
            df_transformed = preproc.transform(df_raw)
            proba = trained_model.predict_proba(df_transformed)
            pred = trained_model.predict(df_transformed)

            model_pred = {
                "probability_bankrupt": round(float(proba[0][1]), 4),
                "probability_healthy": round(float(proba[0][0]), 4),
                "predicted_class": int(pred[0]),
                "predicted_label": "Bankrupt" if int(pred[0]) == 1 else "Healthy",
                "threshold": 0.5,
                "model_roc_auc": model_metrics["roc_auc"] if model_metrics else None,
                "model_accuracy": model_metrics["accuracy"] if model_metrics else None,
            }
        except Exception as e:
            logger.error(f"Prediction failed: {e}", exc_info=True)
            model_pred = {"error": str(e)}

    return PredictionResponse(
        company_name=inputs.companyName,
        model_prediction=model_pred,
        altman_z=altman,
        features_mapped=features_mapped,
        features_total=features_total,
        model_loaded=trained_model is not None,
    )


@app.get("/api/model-info")
async def model_info_endpoint():
    return {
        "model_loaded": trained_model is not None,
        "metrics": model_metrics,
        "training_approach": "LightGBM + SMOTE + early stopping, trained on startup",
        "dataset": "Combined: US + Taiwan + Poland (128,906 companies)",
    }


# Frontend served separately (Vercel/Netlify/etc.)
