import sys
import os
# Add root directory to sys.path to import config
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import json
from tqdm import tqdm
from neo4j import GraphDatabase
from config import Config

class GraphIngestor:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            Config.NEO4J_URI, 
            auth=(Config.NEO4J_USERNAME, Config.NEO4J_PASSWORD)
        )
        self.verify_connection()

    def verify_connection(self):
        try:
            self.driver.verify_connectivity()
            print("Connected to Neo4j successfully.")
        except Exception as e:
            print(f"Failed to connect to Neo4j: {e}")
            raise

    def close(self):
        self.driver.close()

    def load_data(self, filepath):
        print(f"Loading data from {filepath}...")
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)

    def clean_graph(self):
        print("Cleaning existing graph data...")
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        print("Graph cleared.")

    def create_constraints(self):
        print("Creating constraints...")
        queries = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (d:Document) REQUIRE d.doc_code IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Category) REQUIRE c.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (s:SubCategory) REQUIRE s.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (o:Organization) REQUIRE o.name IS UNIQUE"
        ]
        with self.driver.session() as session:
            for q in queries:
                session.run(q)

    def ingest_data(self, data, batch_size=1000):
        print(f"Starting ingestion of {len(data)} records with batch size {batch_size}...")
        
        batch = []
        with self.driver.session() as session:
            for item in tqdm(data):
                processed = self._prepare_item(item)
                if processed:
                    batch.append(processed)
                
                if len(batch) >= batch_size:
                    self._ingest_batch(session, batch)
                    batch = []
            
            if batch:
                self._ingest_batch(session, batch)
        
        print("Ingestion complete!")

    def _prepare_item(self, item):
        # 1. Parse contents_summary
        summary_raw = item.get('contents_summary')
        if not summary_raw:
            return None

        summary = {}
        if isinstance(summary_raw, str):
            try:
                summary = json.loads(summary_raw)
            except:
                return None
        elif isinstance(summary_raw, dict):
            summary = summary_raw
        else:
            return None

        # 2. Extract Fields
        doc_code = str(summary.get('idx', item.get('doc_code', '')))
        title = summary.get('title', '')
        date = summary.get('date', '')
        problem_summary = summary.get('problems', '')
        
        cat_l1 = summary.get('cat', '').split(',')[0].strip() if summary.get('cat') else "Uncategorized"
        cat_l2 = summary.get('sub_cat', '').split(',')[0].strip() if summary.get('sub_cat') else None
        
        org_raw = summary.get('category', '')
        org_name = org_raw.split('|')[0].strip() if '|' in org_raw else org_raw.strip()
        if not org_name:
            org_name = "Unknown Org"

        return {
            "doc_code": doc_code,
            "title": title,
            "date": date,
            "problem_summary": problem_summary,
            "cat_l1": cat_l1,
            "cat_l2": cat_l2,
            "org_name": org_name
        }

    def _ingest_batch(self, session, batch):
        cypher = """
        UNWIND $batch AS row
        
        MERGE (c:Category {name: row.cat_l1})
        MERGE (o:Organization {name: row.org_name})
        
        MERGE (d:Document {doc_code: row.doc_code})
        SET d.title = row.title,
            d.date = row.date,
            d.problem_summary = row.problem_summary
        
        MERGE (d)-[:BELONGS_TO]->(c)
        MERGE (d)-[:ISSUED_BY]->(o)
        
        FOREACH (_ IN CASE WHEN row.cat_l2 IS NOT NULL THEN [1] ELSE [] END |
            MERGE (s:SubCategory {name: row.cat_l2})
            MERGE (d)-[:HAS_SUB_CATEGORY]->(s)
            MERGE (c)-[:PARENT_OF]->(s)
        )
        """
        session.run(cypher, {"batch": batch})

if __name__ == "__main__":
    ingestor = GraphIngestor()
    try:
        data = ingestor.load_data(Config.DATA_PATH)
        ingestor.clean_graph() # Optional: Clear graph before ingest
        ingestor.create_constraints()
        ingestor.ingest_data(data)
    except Exception as e:
        print(f"Error: {e}")
    finally:
        ingestor.close()
