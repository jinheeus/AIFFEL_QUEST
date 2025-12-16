from pymilvus import connections, utility
from config import Config


def test():
    print(f"Connecting to {Config.MILVUS_URI}...")
    connections.connect("default", uri=Config.MILVUS_URI, token=Config.MILVUS_TOKEN)
    print("Connected!")
    print(f"Collections: {utility.list_collections()}")
    print("Connection Test Passed.")


if __name__ == "__main__":
    test()
