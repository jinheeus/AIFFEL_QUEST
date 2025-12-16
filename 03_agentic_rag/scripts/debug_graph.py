import os
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")


def debug_graph():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

    query_regs = """
    MATCH (r:Regulation)
    RETURN r.name as name
    ORDER BY rand()
    LIMIT 10
    """

    query_sample = """
    MATCH (c:AuditCase)-[rel:VIOLATED]->(r:Regulation)
    RETURN c.title, r.name
    LIMIT 5
    """

    with driver.session() as session:
        print("--- Regulations (Random 10) ---")
        result = session.run(query_regs)
        for record in result:
            print(record["name"])

        print("\n--- Regulation Connections ---")
        result = session.run(query_sample)
        found = False
        for record in result:
            found = True
            print(
                f"Case: {record['c.title'][:30]}... -> Violated -> {record['r.name']}"
            )

        if not found:
            print("No connections found between AuditCase and Regulation!")

    driver.close()


if __name__ == "__main__":
    debug_graph()
