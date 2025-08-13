from pymongo import MongoClient
import logging

# MongoDB configuration
MONGO_URI = "mongodb://localhost:27017/"

try:
    client = MongoClient(MONGO_URI)
    # Test the connection
    client.server_info()
    print("Successfully connected to MongoDB")
    
    db = client['eye_disease_db']
    users_collection = db['users']
    history_collection = db['history']
    
    # Create indexes if they don't exist
    users_collection.create_index('email', unique=True)
    history_collection.create_index('user_id')
    
    print("Database collections initialized")
except Exception as e:
    print(f"Error connecting to MongoDB: {e}")
    raise
