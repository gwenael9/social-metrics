import os

import mysql.connector
from dotenv import load_dotenv

load_dotenv()


def get_connection():
    return mysql.connector.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "3306")),
        user=os.getenv("DB_USER", "root"),
        password=os.getenv("DB_PASSWORD", ""),
        database=os.getenv("DB_NAME", "social_metrics"),
    )


def fetch_all_tweets():
    conn = get_connection()
    cur = conn.cursor(dictionary=True)
    cur.execute("SELECT id, text, positive, negative FROM tweets")
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows
