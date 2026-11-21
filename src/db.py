import sqlite3
from datetime import datetime

DB_PATH = "database/negotiation.db"


def get_connection():
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def init_db():
    conn = get_connection()
    cur = conn.cursor()

    # Negotiations table (now includes personality columns)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS negotiations (
        negotiation_id INTEGER PRIMARY KEY AUTOINCREMENT,
        product_id INTEGER,
        product_name TEXT,
        buyer_budget REAL,
        seller_min_price REAL,
        start_price REAL,
        final_price REAL,
        status TEXT,
        total_rounds INTEGER,
        buyer_personality TEXT,
        seller_personality TEXT,
        created_at TEXT
    )
    """)

    # Rounds table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS negotiation_rounds (
        round_id INTEGER PRIMARY KEY AUTOINCREMENT,
        negotiation_id INTEGER,
        round_no INTEGER,
        speaker TEXT,
        offer_price REAL,
        message TEXT,
        timestamp TEXT,
        FOREIGN KEY(negotiation_id) REFERENCES negotiations(negotiation_id)
    )
    """)

    # ---- Add missing columns safely (for old databases) ----
    try:
        cur.execute("ALTER TABLE negotiations ADD COLUMN buyer_personality TEXT;")
    except:
        pass

    try:
        cur.execute("ALTER TABLE negotiations ADD COLUMN seller_personality TEXT;")
    except:
        pass

    conn.commit()
    conn.close()


def save_negotiation_summary(product_id, product_name, buyer_budget, seller_min_price,
                            start_price, final_price, status, total_rounds,
                            buyer_personality="balanced",
                            seller_personality="balanced"):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
    INSERT INTO negotiations (
        product_id, product_name, buyer_budget, seller_min_price,
        start_price, final_price, status, total_rounds,
        buyer_personality, seller_personality,
        created_at
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        product_id, product_name, buyer_budget, seller_min_price,
        start_price, final_price, status, total_rounds,
        buyer_personality, seller_personality,
        datetime.now().isoformat()
    ))

    negotiation_id = cur.lastrowid
    conn.commit()
    conn.close()
    return negotiation_id


def save_round(negotiation_id, round_no, speaker, offer_price, message):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
    INSERT INTO negotiation_rounds (
        negotiation_id, round_no, speaker, offer_price, message, timestamp
    )
    VALUES (?, ?, ?, ?, ?, ?)
    """, (
        negotiation_id, round_no, speaker, offer_price, message,
        datetime.now().isoformat()
    ))

    conn.commit()
    conn.close()


# ✅ Clear all DB data
def clear_database():
    conn = get_connection()
    cur = conn.cursor()

    # Delete child table first (foreign key safe)
    cur.execute("DELETE FROM negotiation_rounds;")
    cur.execute("DELETE FROM negotiations;")

    conn.commit()
    conn.close()
