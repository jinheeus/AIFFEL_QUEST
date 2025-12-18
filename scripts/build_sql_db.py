import json
import sqlite3
import re
from datetime import datetime

# Paths
AUDIT_FILE = "00_data/audit_data.json"
DATA_V2_FILE = "00_data/raw_data/data_v2.json"
DB_PATH = "audit_metadata.db"


def parse_date(date_str):
    """Convert 'YYYY.MM.DD' to 'YYYY-MM-DD'."""
    try:
        return datetime.strptime(str(date_str).strip(), "%Y.%m.%d").strftime("%Y-%m-%d")
    except:
        return None


def main():
    print(f"🔹 Loading data from {AUDIT_FILE} and {DATA_V2_FILE}...")

    with open(AUDIT_FILE, "r", encoding="utf-8") as f:
        audit_data = json.load(f)

    with open(DATA_V2_FILE, "r", encoding="utf-8") as f:
        data_v2 = json.load(f)

    # Index data_v2 by idx
    v2_map = {item["idx"]: item for item in data_v2}

    print(f"🔹 Connecting to SQLite: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Create Table
    cursor.execute("DROP TABLE IF EXISTS audits")
    cursor.execute("""
        CREATE TABLE audits (
            id INTEGER PRIMARY KEY,
            idx INTEGER,
            date TEXT,
            title TEXT,
            site TEXT,
            company TEXT,
            company_code TEXT,
            category TEXT,
            cat TEXT,
            sub_cat TEXT,
            file_path TEXT,
            download_url TEXT,
            problem TEXT,
            action TEXT
        )
    """)

    inserted_count = 0

    for item in audit_data:
        idx = item.get("idx")
        v2_item = v2_map.get(idx, {})

        # Base fields
        site = item.get("site", "").strip()
        raw_date = item.get("date", "")
        formatted_date = parse_date(raw_date)
        title = item.get("title", "")
        category = item.get("category", "")
        file_path = item.get("file_path", "")
        download_url = item.get("download_url", "")

        # V2 fields
        cat = v2_item.get("cat", "")
        sub_cat = v2_item.get("sub_cat", "")

        # Content fields (for reference in retrieval, though search might be text-to-sql on metadata)
        # item['contents'] is summary, v2 might have problems/action too.
        # User mapped: audit_data has site, date, category...
        # v2 seems to have similar structure.
        # Let's take 'problem' and 'action' from audit_data (it has content summary keys like 'problem')
        # Check audit_data structure again: lines 13, 14 show item["problem"], item["action"] exist.
        problem = item.get("problem", "")
        action = item.get("action", "")

        # ALIO Company Parsing
        company = ""
        company_code = ""

        if site == "ALIO 공공기관 경영정보 공개시스템":
            # category format: "Company, Code" or "Company | Code"
            if category:
                # Split by comma or pipe
                parts = re.split(r"[,|]", category)
                if len(parts) >= 2:
                    company = parts[0].strip()
                    company_code = parts[1].strip()
        else:
            # For non-ALIO, fallback?
            # '감사원' usually targets a specific agency in title or content, but it's hard to parse reliably.
            # We'll leave company blank for now, or maybe use site?
            # User instruction focused on ALIO logic.
            # But if a user asks "Incheon Airport cases", and it's from '감사원', we might miss it.
            # However, text-to-sql can SEARCH 'title' and 'problem' for "Incheon Airport".
            pass

        # Insert
        cursor.execute(
            """
            INSERT INTO audits (
                idx, date, title, site, company, company_code, category,
                cat, sub_cat, file_path, download_url, problem, action
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                idx,
                formatted_date,
                title,
                site,
                company,
                company_code,
                category,
                cat,
                sub_cat,
                file_path,
                download_url,
                problem,
                action,
            ),
        )

        inserted_count += 1

    conn.commit()

    # Create Indices
    print("🔹 Creating Indices...")
    cursor.execute("CREATE INDEX idx_date ON audits(date)")
    cursor.execute("CREATE INDEX idx_company ON audits(company)")
    cursor.execute("CREATE INDEX idx_site ON audits(site)")

    conn.close()
    print(f"✅ Database built successfully! {inserted_count} records inserted.")


if __name__ == "__main__":
    main()
