import asyncio
import os
import pytesseract
from PIL import Image
from dotenv import load_dotenv
from neo4j import AsyncGraphDatabase

load_dotenv()

uri = os.getenv("NEO4J_URI")
user = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")

async def main():
    driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
    async with driver.session() as session:
        print("🔍 Buscando todas las imágenes en Neo4j...")
        # Obtenemos todas las imágenes. Aunque tengan texto, lo regeneramos porque estaba fallando silenciosamente.
        res = await session.run("MATCH (i:Imagen) RETURN i.id, i.path")
        
        updates = []
        records = await res.data()
        print(f"Total imágenes a procesar: {len(records)}")
        
        for idx, record in enumerate(records, 1):
            img_id = record["i.id"]
            img_path = record["i.path"]
            
            if os.path.exists(img_path):
                try:
                    pil_img = Image.open(img_path)
                    ocr_text = pytesseract.image_to_string(pil_img).strip()[:300]
                    if ocr_text:
                        updates.append({"id": img_id, "ocr_text": ocr_text})
                except Exception as e:
                    print(f"⚠️ Error procesando {img_path}: {e}")
            else:
                print(f"⚠️ Imagen no encontrada en disco: {img_path}")
                
            if idx % 50 == 0:
                print(f"Progreso OCR local: {idx}/{len(records)} imágenes...")
        
        if updates:
            print(f"⚙️ Actualizando OCR de {len(updates)} imágenes en Neo4j...")
            for upd in updates:
                await session.run("MATCH (i:Imagen {id: $id}) SET i.ocr_text = $ocr", 
                                  {"id": upd["id"], "ocr": upd["ocr_text"]})
            print("✅ Actualización completada.")
        else:
            print("✅ No hay imágenes para actualizar o Tesseract falló.")

    await driver.close()

if __name__ == "__main__":
    asyncio.run(main())
