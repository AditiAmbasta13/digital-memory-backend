from neo4j import GraphDatabase
from app.config import settings
import logging

logger = logging.getLogger(__name__)


class Neo4jDriver:
    _instance = None
    _driver = None

    @classmethod
    def get_driver(cls):
        if cls._driver is None:
            try:
                cls._driver = GraphDatabase.driver(
                    settings.NEO4J_URI,
                    auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
                )
                cls._driver.verify_connectivity()
                logger.info("Connected to Neo4j successfully")
            except Exception as e:
                logger.warning(f"Neo4j connection failed: {e}. Running in fallback mode.")
                cls._driver = None
        return cls._driver

    @classmethod
    def close(cls):
        if cls._driver:
            cls._driver.close()
            cls._driver = None

    @classmethod
    def is_connected(cls) -> bool:
        if cls._driver is None:
            return False
        try:
            cls._driver.verify_connectivity()
            return True
        except Exception:
            return False


def get_neo4j_session():
    """Get a Neo4j session. Returns None if Neo4j is not available."""
    driver = Neo4jDriver.get_driver()
    if driver is None:
        return None
    return driver.session()
