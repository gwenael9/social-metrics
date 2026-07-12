"""Load data/seed_tweets.csv into the tweets table."""
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.db import get_connection

CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "seed_tweets.csv")


def seed():
    conn = get_connection()
    cur = conn.cursor()
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [(r["text"], int(r["positive"]), int(r["negative"])) for r in reader]

    cur.executemany(
        "INSERT INTO tweets (text, positive, negative) VALUES (%s, %s, %s)",
        rows,
    )
    conn.commit()
    print(f"Inserted {len(rows)} rows into tweets.")
    cur.close()
    conn.close()


if __name__ == "__main__":
    seed()
