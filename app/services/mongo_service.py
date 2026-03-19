import os
import uuid
import logging
from pymongo import MongoClient
import gridfs
from bson.objectid import ObjectId
from app.config import settings
from fastapi import HTTPException

logger = logging.getLogger(__name__)

# Initialize MongoDB client
_client = None
_db = None
_fs = None

def get_mongo_fs():
    """Get or initialize GridFS instance."""
    global _client, _db, _fs
    if _fs is None:
        try:
            _client = MongoClient(settings.MONGO_URI, serverSelectionTimeoutMS=5000)
            # Verify connection
            _client.admin.command('ping')
            _db = _client[settings.MONGO_DB_NAME]
            _fs = gridfs.GridFS(_db)
            logger.info("MongoDB GridFS initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise Exception("Database connection failed")
    return _fs

def close_mongo_connection():
    """Close the MongoDB connection pool."""
    global _client
    if _client is not None:
        _client.close()
        logger.info("MongoDB connection closed.")

def upload_file_to_mongo(file_path: str, original_filename: str, content_type: str = "application/pdf") -> str:
    """
    Uploads a file to MongoDB GridFS and returns a reference URL.
    Returns something like '/api/content/files/{file_id}'
    """
    fs = get_mongo_fs()
    file_id = None
    
    try:
        with open(file_path, 'rb') as f:
            file_id = fs.put(f, filename=original_filename, content_type=content_type)
        
        url = f"/api/content/files/{str(file_id)}"
        logger.info(f"File uploaded to MongoDB GridFS. ID: {file_id}")
        return url
    except Exception as e:
        logger.error(f"Error uploading file to MongoDB: {e}")
        # Cleanup in case of partial upload if possible? put handles it mostly.
        raise

def get_file_from_mongo(file_id: str):
    """
    Retrieve a file from MongoDB GridFS by ID.
    """
    fs = get_mongo_fs()
    try:
        return fs.get(ObjectId(file_id))
    except gridfs.errors.NoFile:
        return None
    except Exception as e:
        logger.error(f"Error retrieving file {file_id} from MongoDB: {e}")
        raise

def delete_file_from_mongo(file_url: str):
    """
    Deletes a file from MongoDB GridFS given its reference URL.
    """
    if not file_url or not file_url.startswith("/api/content/files/"):
        return
        
    fs = get_mongo_fs()
    try:
        file_id_str = file_url.split('/')[-1]
        file_id = ObjectId(file_id_str)
        if fs.exists(file_id):
            fs.delete(file_id)
            logger.info(f"Deleted file {file_id_str} from MongoDB Storage.")
    except Exception as e:
        logger.error(f"Error deleting file from MongoDB: {e}")
