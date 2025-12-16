import os
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")


def verify_graph():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

    query_stats = """
    MATCH (n)
    RETURN count(n) as node_count
    """

    query_edges = """
    MATCH ()-[r]->()
    RETURN count(r) as edge_count
    """

    query_schema = """
    CALL db.schema.visualization()
    """

    with driver.session() as session:
        # Check Nodes
        result_nodes = session.run(query_stats).single()
        node_count = result_nodes["node_count"]
        print(f"Total Nodes: {node_count}")

        # Check Edges
        result_edges = session.run(query_edges).single()
        edge_count = result_edges["edge_count"]
        print(f"Total Edges: {edge_count}")

        # Check Schema (Labels)
        result_schema = session.run("CALL db.labels()")
        labels = [r[0] for r in result_schema]
        print(f"Node Labels: {labels}")

    driver.close()


if __name__ == "__main__":
    verify_graph()
