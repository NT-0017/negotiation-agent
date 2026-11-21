from src.agents import BuyerAgent, SellerAgent, generate_message


def run_negotiation(
    product,
    buyer_budget,
    max_rounds=6,
    buyer_personality="balanced",
    seller_personality="balanced",
):

    market_price = float(product["market_price"])
    min_price = float(product["min_price"])

    # 🚫 EARLY EXIT
    if buyer_budget < min_price:
        return "FAILED", None, [
            ("System", 0, "❌ Budget too low. No deal possible.")
        ]

    buyer = BuyerAgent(buyer_budget, buyer_personality, max_rounds)
    seller = SellerAgent(min_price, seller_personality, max_rounds)

    rounds_log = []

    seller_offer = seller.first_offer(market_price)
    rounds_log.append(("Seller", seller_offer, f"My offer is ₹{seller_offer}"))

    for r in range(1, max_rounds + 1):

        # Buyer accept
        if buyer.accept(seller_offer, market_price, r, max_rounds):
            rounds_log.append(("Buyer", seller_offer, f"Deal at ₹{seller_offer}"))
            return "SUCCESS", seller_offer, rounds_log

        # Buyer counter
        buyer_offer = buyer.counter_offer(seller_offer, r, max_rounds)
        rounds_log.append(("Buyer", buyer_offer, generate_message("Buyer", buyer_offer, r)))

        # Seller accept
        if seller.accept(buyer_offer, r, max_rounds):
            rounds_log.append(("Seller", buyer_offer, f"Accepted at ₹{buyer_offer}"))
            return "SUCCESS", buyer_offer, rounds_log

        # Seller counter
        seller_offer = seller.counter_offer(seller_offer, buyer_offer, r, max_rounds)
        rounds_log.append(("Seller", seller_offer, generate_message("Seller", seller_offer, r)))

    return "FAILED", None, rounds_log