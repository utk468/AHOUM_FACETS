import sys
from pymongo import MongoClient
from backend.app.config import settings

class Database:
    client: MongoClient = None
    db = None

    @classmethod
    def connect_db(cls):
        try:
            print(f"[*] Connecting to MongoDB at {settings.MONGO_URI}...")
            cls.client = MongoClient(settings.MONGO_URI, serverSelectionTimeoutMS=5000)
            # Force a connection check
            cls.client.server_info()
            cls.db = cls.client[settings.MONGO_DB]
            print(f"[+] Connected to MongoDB database '{settings.MONGO_DB}' successfully!")
        except Exception as e:
            print(f"[x] Could not connect to MongoDB: {e}")
            print("[!] Running without MongoDB. Database operations will be mocked or raise exceptions.")
            cls.client = None
            cls.db = None

    @classmethod
    def get_db(cls):
        if cls.db is None:
            # Try to connect if not already connected
            cls.connect_db()
        return cls.db

    @classmethod
    def close_db(cls):
        if cls.client is not None:
            cls.client.close()
            print("[-] Closed connection to MongoDB.")

def get_db_client():
    return Database.get_db()
