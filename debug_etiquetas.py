"""Debug: List all image labels in Neo4j to understand the mapping"""
import asyncio
import os
from dotenv import load_dotenv
from neo4j import AsyncGraphDatabase

load_dotenv()

async def main():
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USERNAME") 
    pwd = os.getenv("NEO4J_PASSWORD")
    
    driver = AsyncGraphDatabase.driver(uri, auth=(user, pwd))
    
    async with driver.session() as s:
        # All images with their labels
        r = await s.run("""
            MATCH (i:Imagen)
            RETURN i.nombre_archivo AS archivo, 
                   i.etiqueta AS etiqueta, 
                   i.pagina AS pagina,
                   substring(coalesce(i.caption,''), 0, 100) AS caption,
                   i.path AS path
            ORDER BY i.pagina
        """)
        imgs = [dict(rec) async for rec in r]
        
        print(f"\n{'='*80}")
        print(f"Total imágenes en Neo4j: {len(imgs)}")
        print(f"{'='*80}")
        for img in imgs:
            print(f"  pag={img['pagina']:>3} | archivo={img['archivo']:<25} | etiqueta='{img.get('etiqueta', '')}' | caption='{img.get('caption', '')[:60]}'")
        
        # Specifically look for sarcomera
        print(f"\n{'='*80}")
        print("Buscando 'sarco' o '13.4' en etiqueta/caption:")
        print(f"{'='*80}")
        r2 = await s.run("""
            MATCH (i:Imagen)
            WHERE toLower(coalesce(i.etiqueta,'')) CONTAINS 'sarco' 
               OR toLower(coalesce(i.caption,'')) CONTAINS 'sarco'
               OR toLower(coalesce(i.etiqueta,'')) CONTAINS '13.4'
               OR toLower(coalesce(i.caption,'')) CONTAINS '13.4'
            RETURN i.nombre_archivo AS archivo, i.etiqueta AS etiqueta, 
                   i.pagina AS pagina, substring(coalesce(i.caption,''), 0, 200) AS caption
        """)
        sarco = [dict(rec) async for rec in r2]
        if sarco:
            for s2 in sarco:
                print(f"  ✅ pag={s2['pagina']} | archivo={s2['archivo']} | etiqueta='{s2.get('etiqueta', '')}' | caption='{s2.get('caption', '')}'")
        else:
            print("  ❌ No se encontró ninguna imagen con 'sarco' o '13.4'")
    
    await driver.close()

asyncio.run(main())
