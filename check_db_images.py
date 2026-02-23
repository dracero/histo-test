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
        print("\n--- Images with 'neurona' or 'cerebr' or 'cerebel' or 'gris' or 'nervi' in OCR ---")
        res = await session.run("""
        MATCH (i:Imagen) 
        WHERE toLower(i.ocr_text) CONTAINS 'neurona' 
           OR toLower(i.ocr_text) CONTAINS 'cerebr' 
           OR toLower(i.ocr_text) CONTAINS 'cerebel' 
           OR toLower(i.ocr_text) CONTAINS 'nervio'
           OR toLower(i.ocr_text) CONTAINS 'médula espinal'
        RETURN i.id, substring(i.ocr_text, 0, 100) AS ocr_preview, i.path, i.pagina 
        LIMIT 20
        """)
        for record in await res.data():
            print(record)

    await driver.close()

if __name__ == "__main__":
    asyncio.run(main())
