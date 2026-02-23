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
    async with driver.session() as session:
        # Find images on the same page as chunks with 'nervioso' or 'neurona'
        query = """
        MATCH (c:Chunk)
        WHERE toLower(c.texto) CONTAINS 'nervioso' OR toLower(c.texto) CONTAINS 'neurona'
        MATCH (c)-[:PERTENECE_A]->(pdf:PDF)<-[:PERTENECE_A]-(i:Imagen)
        // Note: Chunk and Image might not directly link to page unless we use chunk text to guess page? 
        // Or wait, images have `pagina` property. Does Chunk have `chunk_id` index?
        RETURN c.id, i.id, i.pagina, i.path LIMIT 5
        """
        # Let's just find the images where their ocr_text contains 'neurona' or 'cerebral'
        print("\n--- Images with 'neurona' or 'cerebr' in OCR ---")
        res = await session.run("MATCH (i:Imagen) WHERE toLower(i.ocr_text) CONTAINS 'neurona' OR toLower(i.ocr_text) CONTAINS 'cerebr' RETURN i.id, substring(i.ocr_text, 0, 100) AS ocr_preview, i.path, i.pagina LIMIT 10")
        for record in await res.data():
            print(record)

    await driver.close()

asyncio.run(main())
