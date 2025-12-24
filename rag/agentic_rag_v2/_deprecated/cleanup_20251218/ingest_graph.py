import os
import json
import time
from neo4j import GraphDatabase
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "00_data", "data_v2.json"
)


class GraphIngestor:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def create_constraints(self):
        """Create unique constraints to prevent duplicates and speed up lookup."""
        queries = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:AuditCase) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (o:Organization) REQUIRE o.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (cat:Category) REQUIRE cat.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (sub:SubCategory) REQUIRE sub.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (r:Regulation) REQUIRE r.name IS UNIQUE",
        ]
        with self.driver.session() as session:
            for q in queries:
                session.run(q)
        print("[GraphIngestor] Constraints created.")

    def ingest_data(self, data):
        """Ingest list of audit cases into Neo4j."""
        query = """
        UNWIND $batch AS row
        
        // 1. Create Audit Case Node
        MERGE (c:AuditCase {id: row.idx})
        SET c.title = row.title,
            c.date = row.date,
            c.summary = row.problems,
            c.action = row.action

        // 2. Organization (Site/Source)
        MERGE (o:Organization {name: row.site})
        MERGE (o)-[:PUBLISHED]->(c)

        // 3. Category & SubCategory
        MERGE (cat:Category {name: row.cat})
        MERGE (c)-[:BELONGS_TO]->(cat)
        
        MERGE (sub:SubCategory {name: row.sub_cat})
        MERGE (c)-[:BELONGS_TO]->(sub)
        MERGE (sub)-[:CHILD_OF]->(cat)

        // 4. Regulations (FOREACH hack for list)
        FOREACH (reg_name IN row.regulations | 
            MERGE (r:Regulation {name: reg_name})
            MERGE (c)-[:VIOLATED]->(r)
        )
        """

        # Pre-process data to split standards into a list
        processed_batch = []
        for item in data:
            # Split standards by comma and clean whitespace
            standards_text = item.get("standards", "")
            regulations = [r.strip() for r in standards_text.split(",") if r.strip()]

            # Add to batch
            item_copy = item.copy()
            item_copy["regulations"] = regulations
            # Handle missing fields
            if "cat" not in item_copy:
                item_copy["cat"] = "Unknown"
            if "sub_cat" not in item_copy:
                item_copy["sub_cat"] = "Unknown"
            if "site" not in item_copy:
                item_copy["site"] = "Unknown"

            processed_batch.append(item_copy)

        # Execute in chunks
        batch_size = 100
        total = len(processed_batch)

        with self.driver.session() as session:
            for i in range(0, total, batch_size):
                batch = processed_batch[i : i + batch_size]
                print(
                    f"[GraphIngestor] Processing batch {i} to {min(i + batch_size, total)}..."
                )
                session.run(query, batch=batch)

        print(f"[GraphIngestor] Ingestion complete. Processed {total} records.")


def main():
    print(f"Loading data from {DATA_PATH}...")
    if not os.path.exists(DATA_PATH):
        print(f"Error: File not found at {DATA_PATH}")
        return

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    max_retries = 3
    retry_delay = 5

    for attempt in range(max_retries):
        try:
            ingestor = GraphIngestor(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)
            ingestor.create_constraints()
            ingestor.ingest_data(data)
            ingestor.close()
            break
        except Exception as e:
            print(f"Connection failed (Attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                print(
                    "Failed to connect to Neo4j. Please check your credentials and ensure the server is running."
                )


if __name__ == "__main__":
    main()
