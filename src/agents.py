import random


# ==========================================================
# 🔥 ADAPTIVE STRATEGY (FIXED)
# ==========================================================
def adjust_strategy_from_history(df_history, personality):
    try:
        if df_history is None or len(df_history) < 10:
            return personality

        success_rate = (df_history["status"] == "SUCCESS").mean()

        if success_rate < 0.4:
            return "flexible"
        elif success_rate > 0.7:
            return "aggressive"
        else:
            return personality

    except:
        return personality


# ==========================================================
# 💬 HUMAN-LIKE MESSAGE GENERATOR
# ==========================================================
def generate_message(role, offer, round_no):
    if role == "Buyer":
        messages = [
            f"Hmm... I can stretch to ₹{offer}.",
            f"That's a bit high. I can offer ₹{offer}.",
            f"I'm trying my best — ₹{offer}.",
            f"Can we close at ₹{offer}?",
        ]
    else:
        messages = [
            f"I can reduce to ₹{offer}.",
            f"Best I can do is ₹{offer}.",
            f"Final offer ₹{offer}.",
            f"I’ve already reduced a lot — ₹{offer}.",
        ]

    return random.choice(messages)


# ==========================================================
# 🛒 BUYER AGENT
# ==========================================================
class BuyerAgent:
    def __init__(self, budget, personality="balanced", patience=6):
        self.budget = float(budget)
        self.personality = personality.lower()
        self.patience = patience

    # -----------------------------
    # ✅ ACCEPT LOGIC (FIXED 🔥)
    # -----------------------------
    def accept(self, seller_offer, market_price, round_no, max_rounds):
        seller_offer = float(seller_offer)
        market_price = float(market_price)

        min_accept_price = market_price * 0.65

        # Normal condition
        if seller_offer <= self.budget and seller_offer >= min_accept_price:
            return True

        # 🔥 FINAL ROUND FLEXIBILITY
        if round_no >= max_rounds - 1:
            if seller_offer <= self.budget:
                return True

        return False

    # -----------------------------
    # 🔁 COUNTER OFFER (SMART 🔥)
    # -----------------------------
    def counter_offer(self, last_seller_offer, round_no, max_rounds):
        last_seller_offer = float(last_seller_offer)

        urgency = round_no / max_rounds

        # Budget-aware progression
        budget_factor = 0.6 + (urgency * 0.4)

        offer = (last_seller_offer * 0.5) + (self.budget * budget_factor * 0.5)

        # Personality adjustment
        if self.personality == "aggressive":
            offer *= 0.95
        elif self.personality == "flexible":
            offer *= 1.05

        # Slight hesitation
        if random.random() < 0.2:
            offer *= 0.98

        return round(min(offer, self.budget), 2)


# ==========================================================
# 🏪 SELLER AGENT
# ==========================================================
class SellerAgent:
    def __init__(self, min_price, personality="balanced", patience=6):
        self.min_price = float(min_price)
        self.personality = personality.lower()
        self.patience = patience

    # -----------------------------
    # 🟢 FIRST OFFER
    # -----------------------------
    def first_offer(self, market_price):
        market_price = float(market_price)

        if self.personality == "strict":
            factor = 1.15
        elif self.personality == "flexible":
            factor = 1.05
        else:
            factor = 1.10

        return round(max(market_price * factor, self.min_price), 2)

    # -----------------------------
    # 🔁 COUNTER OFFER
    # -----------------------------
    def counter_offer(self, last_seller_offer, last_buyer_offer, round_no, max_rounds):
        last_seller_offer = float(last_seller_offer)
        last_buyer_offer = float(last_buyer_offer)

        urgency = round_no / max_rounds

        # If buyer is too low → gradual reduction
        if last_buyer_offer < self.min_price:
            reduction = (last_seller_offer - self.min_price) * 0.3
            new_offer = last_seller_offer - reduction
            return round(max(new_offer, self.min_price), 2)

        gap = last_seller_offer - last_buyer_offer

        step = 0.15 + (urgency * 0.2)

        new_offer = last_seller_offer - (gap * step)

        return round(max(new_offer, self.min_price), 2)

    # -----------------------------
    # ✅ ACCEPT LOGIC
    # -----------------------------
    def accept(self, buyer_offer, round_no, max_rounds):
        buyer_offer = float(buyer_offer)

        if buyer_offer < self.min_price:
            return False

        fair_price = self.min_price + (0.15 * self.min_price)

        if buyer_offer >= fair_price:
            return True

        # 🔥 FINAL ROUND ACCEPT
        if round_no >= max_rounds - 1:
            if buyer_offer >= self.min_price:
                return True

        return False