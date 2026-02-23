import asyncio
import os
from dotenv import load_dotenv
from neo4j import AsyncGraphDatabase

load_dotenv()
uri = os.getenv("NEO4J_URI")
user = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")

async def main():
    driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
    pages = list(range(109, 118)) + list(range(193, 202))
    
    async with driver.session() as session:
        print("\n--- Images from specific pages in Neo4j ---")
        query = f"MATCH (i:Imagen) WHERE i.pagina IN {pages} RETURN i.id, substring(i.ocr_text, 0, 150) AS ocr_preview, size(i.embedding_uni) AS uni_size, size(i.embedding_plip) AS plip_size ORDER BY i.pagina"
        res = await session.run(query)
        for record in await res.data():
            print(record)

    await driver.close()

if __name__ == "__main__":
    asyncio.run(main())
