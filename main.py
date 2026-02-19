"""
Bankruptcy Prediction API — FastAPI Backend
Trains a LightGBM model on startup from taiwan_bankruptcy.csv
"""

import os
import sys
import json
import logging
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

warnings.filterwarnings("ignore")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    precision_score, recall_score
)

import lightgbm as lgb
from imblearn.over_sampling import SMOTE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bankruptcy-api")

app = FastAPI(title="Bankruptcy Risk Prediction API", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SEED = 42
trained_model = None
scaler = None
feature_columns = None
model_metrics = None
norm_to_original = {}


def find_data():
    candidates = [
        "taiwan_bankruptcy.csv",
        "data/taiwan_bankruptcy.csv",
        "combined_raw.csv",
        "data/combined_raw.csv",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def normalize_col_name(col):
    import re
    c = col.strip().lower()
    c = c.replace("%", "").replace("(", "").replace(")", "")
    c = c.replace("/", "_").replace("'", "_").replace("'", "_")
    c = re.sub(r"[^a-z0-9]+", "_", c)
    c = c.strip("_")
    return c


def train_model():
    global trained_model, scaler, feature_columns, model_metrics, norm_to_original

    data_path = find_data()
    if data_path is None:
        logger.warning("No CSV found! Place taiwan_bankruptcy.csv next to main.py")
        return False

    logger.info(f"Loading {data_path}...")
    df = pd.read_csv(data_path)
    logger.info(f"Loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    # ── Find target column ──
    target_col = None

    # Check for 'y' first (combined dataset)
    if "y" in df.columns:
        target_col = "y"
    else:
        # Search for bankrupt-style column
        for col in df.columns:
            col_clean = col.strip().lower().replace("?", "")
            if "bankrupt" in col_clean:
                target_col = col
                break

    if target_col is None:
        # Last resort: try first column
        logger.warning(f"No target column found. Columns: {list(df.columns[:10])}")
        return False

    logger.info(f"Using target column: '{target_col}'")

    # ── Drop leaky columns ──
    for c in ["status_label", "dataset_source", "horizon_years"]:
        if c in df.columns and c != target_col:
            df = df.drop(columns=[c])

    # ── Separate X / y ──
    y = df[target_col].astype(int)
    X = df.drop(columns=[target_col])

    # Drop any remaining non-numeric columns
    non_numeric = X.select_dtypes(exclude=["number"]).columns.tolist()
    if non_numeric:
        logger.info(f"Dropping non-numeric columns: {non_numeric}")
        X = X.drop(columns=non_numeric)

    X.columns = X.columns.str.strip()
    feature_columns = list(X.columns)

    # Build normalized name mapping
    norm_to_original = {}
    for col in feature_columns:
        norm = normalize_col_name(col)
        norm_to_original[norm] = col

    logger.info(f"Features: {len(feature_columns)}, Positive rate: {y.mean():.4f}")

    # ── Split ──
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )

    # ── Scale ──
    scaler = StandardScaler()
    X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_columns, index=X_train.index)
    X_test_s = pd.DataFrame(scaler.transform(X_test), columns=feature_columns, index=X_test.index)

    # ── SMOTE ──
    sm = SMOTE(random_state=SEED)
    X_train_res, y_train_res = sm.fit_resample(X_train_s, y_train)
    logger.info(f"After SMOTE: {len(X_train_res)} samples")

    # ── Early stopping split ──
    X_tr_f, X_val_f, y_tr_f, y_val_f = train_test_split(
        X_train_res, y_train_res, test_size=0.15, random_state=SEED, stratify=y_train_res
    )

    # ── Train ──
    trained_model = lgb.LGBMClassifier(
        objective="binary", random_state=SEED,
        n_estimators=2000, learning_rate=0.03, num_leaves=31,
        n_jobs=-1, verbose=-1,
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

    # ── Evaluate ──
    proba = trained_model.predict_proba(X_test_s)[:, 1]
    preds = (proba >= 0.5).astype(int)

    model_metrics = {
        "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
        "accuracy": round(float(accuracy_score(y_test, preds)), 4),
        "precision": round(float(precision_score(y_test, preds, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, preds)), 4),
        "f1": round(float(f1_score(y_test, preds)), 4),
        "training_samples": int(len(X_train_res)),
        "test_samples": int(len(X_test)),
        "features": int(len(feature_columns)),
        "best_iteration": int(trained_model.best_iteration_),
    }
    logger.info(f"Model trained! {json.dumps(model_metrics, indent=2)}")
    return True


def map_user_inputs_to_features(data: dict) -> tuple:
    if feature_columns is None:
        raise RuntimeError("Model not trained")

    row = {col: 0.0 for col in feature_columns}

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
    gp = rev * 0.4

    def safe(n, d):
        return n / d if d and d != 0 else 0.0

    # Map computed ratios to column name patterns (fuzzy match)
    ratio_map = {
        "roa_c_before_interest_and_depreciation_before_interest": safe(ni + ie + dep, ta),
        "roa_a_before_interest_and_after_tax": safe(ni + ie, ta),
        "roa_b_before_interest_and_depreciation_after_tax": safe(ni + dep, ta),
        "operating_gross_margin": safe(gp, rev),
        "realized_sales_gross_margin": safe(gp, rev),
        "operating_profit_rate": safe(ebit, rev),
        "tax_pre_net_interest_rate": safe(ni, te),
        "after_tax_net_interest_rate": safe(ni, ta),
        "cash_flow_rate": safe(cfo, rev),
        "interest_bearing_debt_interest_rate": safe(ie, td) if td else 0,
        "tax_rate_a": 0.21,
        "per_net_share_value_b": te,
        "net_value_per_share_a": te,
        "net_value_per_share_c": te,
        "persistent_eps_in_the_last_four_seasons": ni,
        "cash_flow_per_share": cfo,
        "revenue_per_share_yuan": rev,
        "operating_profit_per_share_yuan": ebit,
        "per_share_net_profit_before_tax_yuan": ebit,
        "realized_sales_gross_profit_growth_rate": 0.0,
        "operating_profit_growth_rate": 0.0,
        "after_tax_net_profit_growth_rate": 0.0,
        "regular_net_profit_growth_rate": 0.0,
        "continuous_net_profit_growth_rate": 0.0,
        "total_asset_growth_rate": 0.0,
        "net_value_growth_rate": 0.0,
        "total_asset_return_growth_rate_ratio": 0.0,
        "cash_reinvestment": safe(cfo, ta),
        "current_ratio": safe(ca, cl),
        "quick_ratio": safe(ca * 0.7, cl),
        "interest_expense_ratio": safe(ie, rev),
        "total_debt_total_net_worth": safe(td, te),
        "debt_ratio": safe(tl, ta),
        "net_worth_assets": safe(te, ta),
        "long_term_fund_suitability_ratio_a": safe(te + (tl - cl), ta - ca) if (ta - ca) != 0 else 1.0,
        "borrowing_dependency": safe(td, ta),
        "contingent_liabilities_net_worth": 0.0,
        "operating_profit_paid_in_capital": safe(ebit, te),
        "net_profit_before_tax_paid_in_capital": safe(ebit, te),
        "inventory_and_accounts_receivable_net_value": safe(ca * 0.5, te),
        "total_asset_turnover": safe(rev, ta),
        "accounts_receivable_turnover": safe(rev, ca * 0.2) if ca else 0,
        "average_collection_days": safe(365 * ca * 0.2, rev) if rev else 0,
        "inventory_turnover_rate_times": safe(rev * 0.6, ca * 0.3) if ca else 0,
        "fixed_assets_turnover_frequency": safe(rev, ta - ca) if (ta - ca) != 0 else 0,
        "net_worth_turnover_rate_times": safe(rev, te),
        "revenue_per_person": rev,
        "operating_profit_per_person": ebit,
        "allocation_rate_per_person": rev,
        "working_capital_to_total_assets": safe(wc, ta),
        "quick_asset_total_asset": safe(ca * 0.7, ta),
        "current_assets_total_assets": safe(ca, ta),
        "cash_total_assets": safe(cash, ta),
        "quick_asset_current_liabilities": safe(ca * 0.7, cl),
        "cash_current_liability": safe(cash, cl),
        "current_liability_to_assets": safe(cl, ta),
        "operating_funds_to_liability": safe(cfo, tl),
        "inventory_working_capital": safe(ca * 0.3, wc) if wc != 0 else 0,
        "inventory_current_liability": safe(ca * 0.3, cl),
        "current_liability_liability": safe(cl, tl),
        "working_capital_equity": safe(wc, te),
        "current_liability_equity": safe(cl, te),
        "long_term_liability_to_current_assets": safe(tl - cl, ca),
        "retained_earnings_total_assets": safe(re_val, ta),
        "total_income_total_expense": safe(rev, rev - ni) if (rev - ni) != 0 else 1.0,
        "total_expense_assets": safe(rev - ni, ta),
        "current_asset_turnover_rate": safe(rev, ca),
        "quick_asset_turnover_rate": safe(rev, ca * 0.7),
        "working_capitcal_turnover_rate": safe(rev, wc) if wc != 0 else 0,
        "cash_turnover_rate": safe(rev, cash),
        "cash_flow_to_sales": safe(cfo, rev),
        "fix_assets_to_assets": safe(ta - ca, ta),
        "current_liability_to_liability": safe(cl, tl),
        "current_liability_to_equity": safe(cl, te),
        "equity_to_long_term_liability": safe(te, tl - cl) if (tl - cl) != 0 else 0,
        "cash_flow_to_total_assets": safe(cfo, ta),
        "cash_flow_to_liability": safe(cfo, tl),
        "cfo_to_assets": safe(cfo, ta),
        "cash_flow_to_equity": safe(cfo, te),
        "current_liabilities_to_current_assets": safe(cl, ca),
        "one_if_total_liabilities_exceeds_total_assets_zero_otherwise": 1 if tl > ta else 0,
        "net_income_to_total_assets": safe(ni, ta),
        "total_assets_to_gnp_price": ta / 1e9,
        "no_credit_interval": safe(ca - cl, (rev - ni) / 365) if (rev - ni) != 0 else 0,
        "gross_profit_to_sales": safe(gp, rev),
        "net_income_to_stockholder_s_equity": safe(ni, te),
        "liability_to_equity": safe(tl, te),
        "degree_of_financial_leverage_dfl": safe(ebit, ebit - ie) if (ebit - ie) != 0 else 1.0,
        "interest_coverage_ratio_interest_expense_to_ebit": safe(ebit, ie),
        "one_if_net_income_was_negative_for_the_last_two_year_zero_otherwise": 1 if ni < 0 else 0,
        "equity_to_liability": safe(te, tl),
        "non_industry_income_and_expenditure_revenue": 0.0,
        "continuous_interest_rate_after_tax": safe(ni, ta),
        "operating_expense_rate": safe(rev - gp, rev),
        "research_and_development_expense_rate": 0.0,
    }

    mapped = 0
    for norm_key, value in ratio_map.items():
        # Direct match
        if norm_key in norm_to_original:
            row[norm_to_original[norm_key]] = value
            mapped += 1
            continue
        # Fuzzy: check if key is substring of any column or vice versa
        for norm_orig, orig_col in norm_to_original.items():
            if len(norm_key) > 5 and len(norm_orig) > 5:
                if norm_key in norm_orig or norm_orig in norm_key:
                    row[orig_col] = value
                    mapped += 1
                    break

    df = pd.DataFrame([row])[feature_columns]
    if scaler is not None:
        df = pd.DataFrame(scaler.transform(df), columns=feature_columns)

    logger.info(f"Mapped {mapped} features")
    return df, mapped


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

    if trained_model is not None:
        try:
            df, features_mapped = map_user_inputs_to_features(data)
            proba = trained_model.predict_proba(df)
            pred = trained_model.predict(df)

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
        "training_approach": "LightGBM + SMOTE + StandardScaler, trained on startup",
        "dataset": "Taiwan Bankruptcy Prediction (6,819 companies, 95 financial ratios)",
    }
