import os
import random
import sqlite3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.agents import (
    BuyerAgent,
    SellerAgent,
    adjust_strategy_from_history,
    generate_message,
)
from src.db import init_db, save_negotiation_summary, save_round, clear_database
from src.simulator import run_negotiation
from src.ml_model import (
    predict_outcome,
    train_model,
    recommend_best_strategy,
    train_price_model,
    predict_final_price,
)

# -----------------------------------------------------------------------------
# APP CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Autonomous Negotiation Agent", page_icon="🤝", layout="wide")
os.makedirs("database", exist_ok=True)
os.makedirs("data", exist_ok=True)
DB_PATH = "database/negotiation.db"
MODEL_PATH = "data/negotiation_model.pkl"
PRICE_MODEL_PATH = "data/price_model.pkl"
init_db()


# -----------------------------------------------------------------------------
# SHARED HELPERS
# -----------------------------------------------------------------------------
def load_table(table_name):
    allowed = {"negotiations", "negotiation_rounds"}
    if table_name not in allowed:
        raise ValueError("Unsupported database table")
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(f"SELECT * FROM {table_name}", conn)


def save_completed_negotiation(product, budget, buyer_type, seller_type, status, final_price, log):
    start_price = float(log[0][1]) if log else 0.0
    negotiation_id = save_negotiation_summary(
        product_id=int(product["product_id"]),
        product_name=str(product["product_name"]),
        buyer_budget=float(budget),
        seller_min_price=float(product["min_price"]),
        start_price=start_price,
        final_price=float(final_price or 0),
        status=status,
        total_rounds=len(log),
        buyer_personality=buyer_type,
        seller_personality=seller_type,
    )
    for index, (speaker, offer, message) in enumerate(log, start=1):
        save_round(negotiation_id, index, speaker, float(offer or 0), str(message))
    return negotiation_id


def style_axis(ax, title, xlabel=None, ylabel=None):
    fig = ax.figure
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")
    ax.set_title(title, color="white", fontsize=12, fontweight="bold", pad=12)
    if xlabel:
        ax.set_xlabel(xlabel, color="white")
    if ylabel:
        ax.set_ylabel(ylabel, color="white")
    ax.tick_params(colors="#d8dee9", labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#697386")
    ax.grid(axis="y", alpha=0.22, linestyle="--")
    ax.set_axisbelow(True)


def add_bar_labels(ax, bars, formatter=None):
    for bar in bars:
        height = bar.get_height()
        if pd.isna(height):
            continue
        label = formatter(height) if formatter else f"{height:,.0f}"
        ax.annotate(
            label,
            (bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4), textcoords="offset points",
            ha="center", va="bottom", color="white", fontsize=8,
        )


def reset_manual_session():
    st.session_state.pop("manual_negotiation", None)


# -----------------------------------------------------------------------------
# LOAD PRODUCT AND DATABASE DATA
# -----------------------------------------------------------------------------
products = pd.read_csv("data/products.csv")
df_all = load_table("negotiations")

# -----------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# -----------------------------------------------------------------------------
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Go to", ["🤝 Simulator", "📊 Analytics", "🗂 History"])
st.sidebar.divider()
st.sidebar.subheader("🗄️ Database")
st.sidebar.metric("Total Negotiations", len(df_all))
with st.sidebar.expander("Database maintenance"):
    st.warning("Database deletion is permanent. Export your data before clearing it.")
    confirm_clear = st.checkbox("I understand and want to clear all records", key="confirm_clear_db")
    if st.button("Clear Database", disabled=not confirm_clear, key="clear_database_button"):
        clear_database()
        reset_manual_session()
        st.success("Database cleared.")
        st.rerun()


# =============================================================================
# SIMULATOR PAGE
# =============================================================================
if page == "🤝 Simulator":
    st.title("🤝 Autonomous Negotiation Agent")
    st.caption("Configure buyer and seller agents, review machine-learning estimates, and simulate negotiations.")

    col1, col2 = st.columns(2)
    with col1:
        product_name = st.selectbox("Select Product", products["product_name"].tolist())
        product_row = products.loc[products["product_name"] == product_name].iloc[0].to_dict()
    with col2:
        buyer_budget = st.slider(
            "Buyer Maximum Budget (₹)",
            min_value=int(product_row["min_price"] * 0.6),
            max_value=int(product_row["market_price"] * 1.2),
            value=int(product_row["market_price"] * 0.85),
            step=50,
        )

    col3, col4, col5 = st.columns([1, 1, 1])
    with col3:
        buyer_personality = st.selectbox("Buyer Personality", ["aggressive", "balanced", "flexible"], index=1)
    with col4:
        seller_personality = st.selectbox("Seller Personality", ["strict", "balanced", "flexible"], index=1)
    with col5:
        max_rounds = st.selectbox("Maximum Negotiation Rounds", [3, 4, 5, 6, 8, 10], index=3)

    # Display clean product details instead of raw JSON/debug output.
    st.subheader("📦 Product Details")
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Category", str(product_row["category"]))
    p2.metric("Market Price", f"₹{float(product_row['market_price']):,.2f}")
    p3.metric("Seller Minimum", f"₹{float(product_row['min_price']):,.2f}")
    p4.metric("Maximum Budget", f"₹{buyer_budget:,.2f}")

    effective_buyer_personality = adjust_strategy_from_history(df_all, buyer_personality)
    if effective_buyer_personality != buyer_personality:
        st.info(f"Adaptive strategy suggests **{effective_buyer_personality}** buyer behaviour based on historical outcomes.")

    # ML predictions
    st.subheader("🧠 Machine Learning Insights")
    pred_col1, pred_col2, pred_col3 = st.columns(3)
    prob_success = prob_fail = None
    prediction_error = None
    if buyer_budget < float(product_row["min_price"]):
        prediction_error = "The buyer budget is below the seller's minimum price. A feasible deal is not possible."
    else:
        prob_success, prob_fail, prediction_error = predict_outcome(
            int(product_row["product_id"]), buyer_budget,
            float(product_row["min_price"]), float(product_row["market_price"]),
            effective_buyer_personality, seller_personality,
        )
    if prediction_error:
        st.warning(prediction_error)
    else:
        pred_col1.metric("Predicted Success", f"{prob_success * 100:.1f}%")
        pred_col2.metric("Predicted Failure", f"{prob_fail * 100:.1f}%")

    predicted_price, price_error = predict_final_price(
        int(product_row["product_id"]), buyer_budget,
        float(product_row["min_price"]), float(product_row["market_price"]),
        effective_buyer_personality, seller_personality,
    ) if buyer_budget >= float(product_row["min_price"]) else (None, "No feasible deal: budget is below seller minimum.")
    if price_error:
        pred_col3.metric("Expected Final Price", "Unavailable")
    else:
        pred_col3.metric("Expected Final Price", f"₹{predicted_price:,.2f}")

    with st.expander("🛠️ Model training and evaluation"):
        model_col1, model_col2 = st.columns(2)
        with model_col1:
            st.write("**Negotiation success model**")
            st.caption("Predicts the probability of a successful agreement.")
            if st.button("Train Success Prediction Model"):
                with st.spinner("Training success model..."):
                    accuracy, report = train_model()
                if accuracy is None:
                    st.error(report)
                else:
                    st.success(f"Success model trained. Holdout accuracy: {accuracy:.2%}")
                    st.code(report)
        with model_col2:
            st.write("**Final-price model**")
            st.caption("Estimates the final transaction price for successful deals.")
            if st.button("Train Final-Price Model"):
                with st.spinner("Training final-price model..."):
                    metrics, message = train_price_model()
                if metrics is None:
                    st.error(message)
                else:
                    st.success(message)
                    st.json(metrics)

        if os.path.exists(MODEL_PATH):
            try:
                success_model = __import__("joblib").load(MODEL_PATH)["model"]
                importances = pd.Series(success_model.feature_importances_, index=success_model.feature_names_in_).sort_values()
                fig, ax = plt.subplots(figsize=(8, 3.5))
                bars = ax.barh(importances.index, importances.values, color="#65a9e8")
                style_axis(ax, "Success Model – Feature Importance", "Relative importance", "Input feature")
                for bar in bars:
                    ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                            f"{bar.get_width():.3f}", va="center", color="white", fontsize=8)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            except Exception as exc:
                st.caption(f"Feature importance is unavailable: {exc}")

    # Strategy recommendation: no budget values above the user's maximum are evaluated.
    st.subheader("💡 Smart Strategy Recommendation")
    if st.button("Recommend Best Strategy", disabled=buyer_budget < float(product_row["min_price"]), key="recommend_strategy"):
        strategy_df = recommend_best_strategy(product_row, buyer_budget)
        if strategy_df is None or strategy_df.empty:
            st.warning("No recommendation could be generated. Check that both ML models are available.")
        else:
            st.success("Strategies are ranked using predicted success probability. All recommendations respect your entered budget.")
            st.dataframe(strategy_df, use_container_width=True, hide_index=True)

    # Automatic simulations
    st.subheader("⚡ Automatic Simulation")
    auto_col1, auto_col2 = st.columns([1, 3])
    with auto_col1:
        runs = st.number_input("Number of simulation runs", min_value=1, max_value=200, value=25, step=1)
    with auto_col2:
        st.caption("The autonomous buyer and seller agents negotiate independently for each run.")
    if st.button("Run Automatic Simulation", type="primary"):
        progress = st.progress(0)
        completed = 0
        success_count = 0
        for run_index in range(int(runs)):
            selected_product = products.sample(1).iloc[0].to_dict()
            budget_low = int(selected_product["min_price"] * 0.6)
            budget_high = max(budget_low, int(selected_product["market_price"] * 1.1))
            budget = random.randint(budget_low, budget_high)
            buyer_type = random.choice(["aggressive", "balanced", "flexible"])
            seller_type = random.choice(["strict", "balanced", "flexible"])
            status, final_price, log = run_negotiation(
                selected_product, budget, max_rounds=max_rounds,
                buyer_personality=buyer_type, seller_personality=seller_type,
            )
            save_completed_negotiation(selected_product, budget, buyer_type, seller_type,
                                       status, final_price, log)
            success_count += int(status == "SUCCESS")
            completed += 1
            progress.progress(completed / int(runs))
        st.success(f"Completed {completed} simulations. Successful deals: {success_count}; failed deals: {completed - success_count}.")
        st.rerun()

    # Interactive manual negotiation – user plays buyer, AI plays seller.
    st.subheader("🧪 Manual Negotiation")
    st.caption("You act as the buyer. The AI seller responds to each offer. The completed conversation is saved to History.")
    current_config = (
        int(product_row["product_id"]), float(buyer_budget), effective_buyer_personality,
        seller_personality, int(max_rounds),
    )
    manual_state = st.session_state.get("manual_negotiation")
    if manual_state and manual_state.get("config") != current_config:
        reset_manual_session()
        manual_state = None

    start_col, info_col = st.columns([1, 3])
    with start_col:
        if st.button("Start / Restart Manual Negotiation", key="start_manual"):
            if buyer_budget < float(product_row["min_price"]):
                st.error("Your budget is below the seller's minimum price. Increase your budget to start.")
            else:
                seller_agent = SellerAgent(float(product_row["min_price"]), seller_personality, max_rounds)
                opening_offer = seller_agent.first_offer(float(product_row["market_price"]))
                opening_log = [("Seller", opening_offer, f"Hello! My opening offer is ₹{opening_offer:,.2f}.")]
                st.session_state.manual_negotiation = {
                    "config": current_config,
                    "seller_agent": seller_agent,
                    "seller_offer": opening_offer,
                    "round_no": 1,
                    "log": opening_log,
                    "status": "IN_PROGRESS",
                    "saved": False,
                }
                st.rerun()
    with info_col:
        st.caption(f"Budget: ₹{buyer_budget:,.2f} · Seller minimum: ₹{float(product_row['min_price']):,.2f} · Maximum rounds: {max_rounds}")

    manual_state = st.session_state.get("manual_negotiation")
    if manual_state:
        st.markdown(f"**AI seller's current offer: ₹{manual_state['seller_offer']:,.2f}**")
        for speaker, offer, message in manual_state["log"]:
            st.write(f"**{speaker} — ₹{float(offer):,.2f}:** {message}")

        if manual_state["status"] == "IN_PROGRESS":
            if manual_state["seller_offer"] <= buyer_budget:
                if st.button("Accept Seller's Current Offer", key="accept_seller_offer"):
                    manual_state["status"] = "SUCCESS"
                    manual_state["final_price"] = float(manual_state["seller_offer"])
                    manual_state["log"].append(("Buyer", manual_state["seller_offer"],
                                                 f"I accept your offer of ₹{manual_state['seller_offer']:,.2f}. Deal!"))
                    st.rerun()
            with st.form("manual_offer_form", clear_on_submit=True):
                buyer_offer = st.number_input(
                    "Enter your counter-offer (₹)", min_value=1.0,
                    max_value=float(buyer_budget), value=float(min(buyer_budget, max(float(product_row["min_price"]) * 0.8, 1))),
                    step=50.0,
                )
                submit_offer = st.form_submit_button("Submit Counter-Offer")
            if submit_offer:
                manual_state["log"].append(("Buyer", float(buyer_offer), generate_message("Buyer", float(buyer_offer), manual_state["round_no"])))
                seller_agent = manual_state["seller_agent"]
                round_no = manual_state["round_no"]
                if seller_agent.accept(float(buyer_offer), round_no, max_rounds):
                    manual_state["status"] = "SUCCESS"
                    manual_state["final_price"] = float(buyer_offer)
                    manual_state["log"].append(("Seller", float(buyer_offer), f"Accepted your offer of ₹{buyer_offer:,.2f}. Deal!"))
                elif round_no >= max_rounds:
                    manual_state["status"] = "FAILED"
                    manual_state["final_price"] = None
                    manual_state["log"].append(("Seller", float(buyer_offer), "I cannot accept this offer. Negotiation limit reached."))
                else:
                    new_seller_offer = seller_agent.counter_offer(
                        manual_state["seller_offer"], float(buyer_offer), round_no, max_rounds
                    )
                    manual_state["seller_offer"] = float(new_seller_offer)
                    manual_state["round_no"] = round_no + 1
                    manual_state["log"].append(("Seller", float(new_seller_offer), generate_message("Seller", float(new_seller_offer), round_no)))
                st.rerun()

        if manual_state["status"] in {"SUCCESS", "FAILED"} and not manual_state["saved"]:
            final_price = manual_state.get("final_price")
            negotiation_id = save_completed_negotiation(
                product_row, buyer_budget, effective_buyer_personality, seller_personality,
                manual_state["status"], final_price, manual_state["log"],
            )
            manual_state["saved"] = True
            manual_state["negotiation_id"] = negotiation_id
            st.rerun()
        elif manual_state["status"] == "SUCCESS":
            st.success(f"Negotiation successful! Final agreed price: ₹{manual_state['final_price']:,.2f}. Saved as negotiation #{manual_state['negotiation_id']}.")
        elif manual_state["status"] == "FAILED":
            st.error(f"Negotiation ended without an agreement. Saved as negotiation #{manual_state['negotiation_id']}.")


# =============================================================================
# ANALYTICS PAGE
# =============================================================================
elif page == "📊 Analytics":
    st.title("📊 Negotiation Analytics Dashboard")
    df = load_table("negotiations")
    if df.empty:
        st.warning("No negotiation data is available yet. Run a simulation first.")
    else:
        df["status"] = df["status"].astype(str).str.upper()
        total = len(df)
        success = int((df["status"] == "SUCCESS").sum())
        failed = int((df["status"] == "FAILED").sum())
        success_rate = success / total * 100 if total else 0
        avg_rounds = pd.to_numeric(df["total_rounds"], errors="coerce").mean()

        # Only successful deals with valid positive opening/final prices count in discount calculations.
        price_df = df[(df["status"] == "SUCCESS") & (df["start_price"] > 0) & (df["final_price"] > 0)].copy()
        price_df["discount_percent"] = ((price_df["start_price"] - price_df["final_price"]) / price_df["start_price"]) * 100
        avg_discount = price_df["discount_percent"].mean() if not price_df.empty else 0

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Negotiations", f"{total:,}")
        k2.metric("Successful Deal Rate", f"{success_rate:.2f}%")
        k3.metric("Average Rounds", f"{avg_rounds:.2f}" if pd.notna(avg_rounds) else "—")
        k4.metric("Average Discount", f"{avg_discount:.2f}%", help=f"Calculated from {len(price_df)} successful deals with valid positive prices.")
        st.caption(f"Price and discount analysis uses {len(price_df)} successful transactions with valid positive prices. All {total} negotiations remain included in outcome analysis.")
        st.divider()

        # 1. Success and failure counts
        left, right = st.columns(2)
        with left:
            fig, ax = plt.subplots(figsize=(7, 4))
            bars = ax.bar(["Successful", "Failed"], [success, failed], color=["#2ca879", "#d65b63"], width=0.58)
            style_axis(ax, "Negotiation Outcomes", "Outcome", "Number of negotiations")
            add_bar_labels(ax, bars)
            ax.set_ylim(0, max(success, failed, 1) * 1.18)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        with right:
            fig, ax = plt.subplots(figsize=(7, 4))
            if not price_df.empty:
                counts, bins, patches = ax.hist(price_df["final_price"], bins=min(15, max(5, int(np.sqrt(len(price_df))))), color="#65a9e8", edgecolor="#0e1117")
                for count, left_edge, right_edge in zip(counts, bins[:-1], bins[1:]):
                    if count > 0:
                        ax.text((left_edge + right_edge) / 2, count + 0.3, f"{int(count)}", ha="center", color="white", fontsize=7)
            else:
                ax.text(0.5, 0.5, "No valid successful transaction prices", transform=ax.transAxes, ha="center", color="white")
            style_axis(ax, "Final Transaction Price Distribution", "Final transaction price (₹)", "Number of successful deals")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        # Product analysis
        left, right = st.columns(2)
        with left:
            product_prices = price_df.groupby("product_name")["final_price"].mean().sort_values(ascending=False) if not price_df.empty else pd.Series(dtype=float)
            fig, ax = plt.subplots(figsize=(7, 4))
            if not product_prices.empty:
                bars = ax.bar(product_prices.index, product_prices.values, color="#65a9e8")
                add_bar_labels(ax, bars, lambda v: f"₹{v:,.0f}")
                ax.tick_params(axis="x", rotation=25)
                ax.set_ylim(0, max(product_prices.values) * 1.18)
            else:
                ax.text(0.5, 0.5, "No valid transaction prices", transform=ax.transAxes, ha="center", color="white")
            style_axis(ax, "Average Final Price by Product", "Product", "Average final price (₹)")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        with right:
            product_success = (df.assign(is_success=(df["status"] == "SUCCESS").astype(int))
                               .groupby("product_name")["is_success"].mean().mul(100).sort_values(ascending=False))
            fig, ax = plt.subplots(figsize=(7, 4))
            bars = ax.bar(product_success.index, product_success.values, color="#2ca879")
            add_bar_labels(ax, bars, lambda v: f"{v:.1f}%")
            ax.tick_params(axis="x", rotation=25)
            ax.set_ylim(0, 110)
            style_axis(ax, "Success Rate by Product", "Product", "Successful negotiations (%)")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        # Personality analysis
        left, right = st.columns(2)
        with left:
            buyer_success = (df.assign(is_success=(df["status"] == "SUCCESS").astype(int))
                             .groupby("buyer_personality")["is_success"].mean().mul(100).sort_values(ascending=False))
            fig, ax = plt.subplots(figsize=(7, 4))
            bars = ax.bar(buyer_success.index, buyer_success.values, color="#a78bfa")
            add_bar_labels(ax, bars, lambda v: f"{v:.1f}%")
            ax.set_ylim(0, 110)
            style_axis(ax, "Success Rate by Buyer Personality", "Buyer personality", "Successful negotiations (%)")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        with right:
            seller_success = (df.assign(is_success=(df["status"] == "SUCCESS").astype(int))
                              .groupby("seller_personality")["is_success"].mean().mul(100).sort_values(ascending=False))
            fig, ax = plt.subplots(figsize=(7, 4))
            bars = ax.bar(seller_success.index, seller_success.values, color="#e5a44d")
            add_bar_labels(ax, bars, lambda v: f"{v:.1f}%")
            ax.set_ylim(0, 110)
            style_axis(ax, "Success Rate by Seller Personality", "Seller personality", "Successful negotiations (%)")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        st.subheader("🤝 Buyer–Seller Personality Combination Analysis")
        combo = (df.assign(is_success=(df["status"] == "SUCCESS").astype(int))
                 .groupby(["buyer_personality", "seller_personality"])
                 .agg(negotiations=("is_success", "size"), success_rate=("is_success", "mean"))
                 .reset_index())
        combo["success_rate"] = (combo["success_rate"] * 100).round(2)
        combo = combo.rename(columns={"buyer_personality": "Buyer Personality", "seller_personality": "Seller Personality", "negotiations": "Negotiations", "success_rate": "Success Rate (%)"})
        st.dataframe(combo.sort_values("Success Rate (%)", ascending=False), use_container_width=True, hide_index=True)


# =============================================================================
# HISTORY PAGE
# =============================================================================
elif page == "🗂 History":
    st.title("🗂 Negotiation History")
    df_neg = load_table("negotiations")
    df_rounds = load_table("negotiation_rounds")
    if df_neg.empty:
        st.warning("No negotiation history is available yet.")
    else:
        filter_col1, filter_col2 = st.columns(2)
        with filter_col1:
            status_filter = st.selectbox("Filter by outcome", ["All", "SUCCESS", "FAILED"])
        with filter_col2:
            product_filter = st.selectbox("Filter by product", ["All"] + sorted(df_neg["product_name"].dropna().unique().tolist()))
        filtered = df_neg.copy()
        if status_filter != "All":
            filtered = filtered[filtered["status"] == status_filter]
        if product_filter != "All":
            filtered = filtered[filtered["product_name"] == product_filter]
        st.caption(f"Showing {len(filtered)} of {len(df_neg)} negotiations.")
        st.dataframe(filtered.sort_values("negotiation_id", ascending=False), use_container_width=True, hide_index=True)

        csv_neg = filtered.to_csv(index=False).encode("utf-8")
        csv_rounds = df_rounds[df_rounds["negotiation_id"].isin(filtered["negotiation_id"])].to_csv(index=False).encode("utf-8")
        download1, download2 = st.columns(2)
        download1.download_button("Download Negotiations CSV", csv_neg, "negotiations_history.csv", "text/csv")
        download2.download_button("Download Conversation Rounds CSV", csv_rounds, "negotiation_rounds.csv", "text/csv")

        if not filtered.empty:
            selected_id = st.selectbox("Select a negotiation to inspect", filtered["negotiation_id"].tolist())
            selected_record = filtered[filtered["negotiation_id"] == selected_id].iloc[0]
            st.subheader(f"Negotiation #{selected_id} – {selected_record['product_name']}")
            detail1, detail2, detail3, detail4 = st.columns(4)
            detail1.metric("Outcome", selected_record["status"])
            detail2.metric("Buyer Budget", f"₹{selected_record['buyer_budget']:,.2f}")
            detail3.metric("Final Price", f"₹{selected_record['final_price']:,.2f}" if selected_record["final_price"] else "No deal")
            detail4.metric("Rounds", int(selected_record["total_rounds"]))
            conversation = df_rounds[df_rounds["negotiation_id"] == selected_id].sort_values("round_no")
            st.markdown("**Conversation transcript**")
            if conversation.empty:
                st.info("No conversation rounds were stored for this negotiation.")
            else:
                for _, row in conversation.iterrows():
                    st.write(f"**{row['speaker']} — ₹{float(row['offer_price']):,.2f}:** {row['message']}")
                fig, ax = plt.subplots(figsize=(9, 4))
                for speaker, group in conversation.groupby("speaker"):
                    ax.plot(group["round_no"], group["offer_price"], marker="o", linewidth=2, label=speaker)
                    for _, point in group.iterrows():
                        ax.annotate(f"₹{point['offer_price']:,.0f}", (point["round_no"], point["offer_price"]),
                                    textcoords="offset points", xytext=(0, 7), ha="center", color="white", fontsize=8)
                style_axis(ax, f"Offer Progression – Negotiation #{selected_id}", "Conversation entry", "Offer price (₹)")
                ax.legend(facecolor="#1b1f2a", labelcolor="white")
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
