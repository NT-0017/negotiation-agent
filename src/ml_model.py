import os
import sqlite3

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

DB_PATH = "database/negotiation.db"
MODEL_PATH = "data/negotiation_model.pkl"
PRICE_MODEL_PATH = "data/price_model.pkl"
FEATURES = [
    "product_id", "buyer_budget", "seller_min_price", "start_price",
    "buyer_personality_enc", "seller_personality_enc",
]


def load_negotiation_data():
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query("SELECT * FROM negotiations", conn)


def _prepare_features(df):
    df = df.copy()
    df["buyer_personality"] = df["buyer_personality"].fillna("balanced").astype(str).str.lower()
    df["seller_personality"] = df["seller_personality"].fillna("balanced").astype(str).str.lower()
    buyer_encoder = LabelEncoder()
    seller_encoder = LabelEncoder()
    df["buyer_personality_enc"] = buyer_encoder.fit_transform(df["buyer_personality"])
    df["seller_personality_enc"] = seller_encoder.fit_transform(df["seller_personality"])
    return df, buyer_encoder, seller_encoder


def train_model():
    df = load_negotiation_data()
    required = ["status", "product_id", "buyer_budget", "seller_min_price", "start_price",
                "buyer_personality", "seller_personality"]
    if df.empty or len(df) < 50 or any(c not in df.columns for c in required):
        return None, "Not enough complete negotiation records to train the success model."

    df = df.dropna(subset=["status", "product_id", "buyer_budget", "seller_min_price", "start_price"])
    df, buyer_encoder, seller_encoder = _prepare_features(df)
    df["status_label"] = (df["status"].astype(str).str.upper() == "SUCCESS").astype(int)
    if df["status_label"].nunique() < 2 or df["status_label"].value_counts().min() < 2:
        return None, "Training requires at least two examples from each outcome class."

    X = df[FEATURES]
    y = df["status_label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    model = RandomForestClassifier(
        n_estimators=300, max_depth=12, min_samples_leaf=2,
        class_weight="balanced", random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    report = (
        f"Accuracy: {acc:.4f}\n\n"
        f"Classification report:\n{classification_report(y_test, y_pred, zero_division=0)}\n"
        f"Confusion matrix (rows=actual [Failed, Successful], columns=predicted [Failed, Successful]):\n{cm}"
    )
    os.makedirs("data", exist_ok=True)
    joblib.dump({
        "model": model,
        "buyer_encoder": buyer_encoder,
        "seller_encoder": seller_encoder,
    }, MODEL_PATH)
    return acc, report


def train_price_model():
    df = load_negotiation_data()
    if df.empty:
        return None, "No negotiation data is available."
    df = df[(df["status"] == "SUCCESS") & (df["final_price"] > 0) & (df["start_price"] > 0)]
    df = df.dropna(subset=["product_id", "buyer_budget", "seller_min_price", "start_price", "final_price"])
    if len(df) < 50:
        return None, f"Not enough valid successful transactions. Found {len(df)}; at least 50 are required."

    df, buyer_encoder, seller_encoder = _prepare_features(df)
    X = df[FEATURES]
    y = df["final_price"].astype(float)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    model = RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_leaf=2,
                                  random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = {
        "MAE": round(mean_absolute_error(y_test, y_pred), 2),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 2),
        "R2": round(r2_score(y_test, y_pred), 4),
        "Test records": int(len(y_test)),
    }
    os.makedirs("data", exist_ok=True)
    joblib.dump({"model": model, "buyer_encoder": buyer_encoder,
                 "seller_encoder": seller_encoder}, PRICE_MODEL_PATH)
    return metrics, "Price model trained successfully."


def _make_prediction_frame(product_id, buyer_budget, seller_min_price, start_price,
                           buyer_personality, seller_personality, saved):
    buyer_encoder = saved["buyer_encoder"]
    seller_encoder = saved["seller_encoder"]
    try:
        buyer_enc = int(buyer_encoder.transform([str(buyer_personality).lower()])[0])
        seller_enc = int(seller_encoder.transform([str(seller_personality).lower()])[0])
    except ValueError as exc:
        raise ValueError("Unknown buyer or seller personality. Retrain the model with the current personality options.") from exc
    return pd.DataFrame([[
        int(product_id), float(buyer_budget), float(seller_min_price), float(start_price),
        buyer_enc, seller_enc,
    ]], columns=FEATURES)


def predict_outcome(product_id, buyer_budget, seller_min_price, start_price,
                    buyer_personality, seller_personality):
    if not os.path.exists(MODEL_PATH):
        return None, None, "Success model not found. Train the model first."
    saved = joblib.load(MODEL_PATH)
    X_new = _make_prediction_frame(product_id, buyer_budget, seller_min_price, start_price,
                                   buyer_personality, seller_personality, saved)
    model = saved["model"]
    probabilities = model.predict_proba(X_new)[0]
    classes = list(model.classes_)
    success_index = classes.index(1) if 1 in classes else None
    if success_index is None:
        return None, None, "The trained model does not contain a successful outcome class."
    prob_success = float(probabilities[success_index])
    return prob_success, 1.0 - prob_success, None


def predict_final_price(product_id, buyer_budget, seller_min_price, start_price,
                        buyer_personality, seller_personality):
    if not os.path.exists(PRICE_MODEL_PATH):
        return None, "Price model not found. Train the model first."
    saved = joblib.load(PRICE_MODEL_PATH)
    X_new = _make_prediction_frame(product_id, buyer_budget, seller_min_price, start_price,
                                   buyer_personality, seller_personality, saved)
    predicted = float(saved["model"].predict(X_new)[0])
    # Keep the estimate within economically valid negotiation boundaries.
    lower = float(seller_min_price)
    upper = float(buyer_budget)
    if upper < lower:
        return None, "The buyer budget is below the seller's minimum price; no feasible deal exists."
    predicted = min(max(predicted, lower), upper)
    return round(predicted, 2), None


def recommend_best_strategy(product_row, buyer_budget):
    """Compare all personality pairs while keeping the buyer's budget fixed."""
    buyer_budget = float(buyer_budget)
    min_price = float(product_row["min_price"])
    if buyer_budget < min_price:
        return None
    results = []
    for buyer_type in ["aggressive", "balanced", "flexible"]:
        for seller_type in ["strict", "balanced", "flexible"]:
            probability, _, error = predict_outcome(
                product_row["product_id"], buyer_budget, min_price,
                product_row["market_price"], buyer_type, seller_type
            )
            if error or probability is None:
                continue
            expected_price, price_error = predict_final_price(
                product_row["product_id"], buyer_budget, min_price,
                product_row["market_price"], buyer_type, seller_type
            )
            if price_error or expected_price is None:
                continue
            results.append({
                "Buyer Personality": buyer_type,
                "Seller Personality": seller_type,
                "Buyer Budget (₹)": round(buyer_budget, 2),
                "Predicted Success (%)": round(probability * 100, 2),
                "Expected Final Price (₹)": expected_price,
            })
    if not results:
        return None
    return pd.DataFrame(results).sort_values(
        ["Predicted Success (%)", "Expected Final Price (₹)"],
        ascending=[False, True]
    ).reset_index(drop=True)
