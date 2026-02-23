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
        print("\n--- Page mapping for chunk 249 and 250 ---")
        query = """
        MATCH (c:Chunk)
        WHERE toLower(c.texto) CONTAINS 'nervioso' OR toLower(c.texto) CONTAINS 'neurona'
        RETURN c.id, c.texto LIMIT 10
        """
        # wait, the chunks in neo4j don't have a direct "pagina" property? Let's check all properties of those chunks.
        query = """
        MATCH (c:Chunk)
        WHERE c.id IN ['chunk_arch2.pdf_249', 'chunk_arch2.pdf_250']
        RETURN properties(c)
        """
        res = await session.run(query)
        for record in await res.data():
            print(record)
            
        print("\n--- Try to find Image linked to same PDF with similar text in OCR ---")
        query2 = """
        MATCH (i:Imagen)
        WHERE toLower(i.ocr_text) CONTAINS 'pirami' OR toLower(i.ocr_text) CONTAINS 'cerebel'
        RETURN i.id, i.pagina, substring(i.ocr_text, 0, 100)
        """
        res = await session.run(query2)
        for record in await res.data():
            print(record)

    await driver.close()

if __name__ == "__main__":
    asyncio.run(main())
