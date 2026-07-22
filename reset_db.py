import os
import shutil
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

# Configuración de Neo4j
uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
user = os.getenv("NEO4J_USERNAME", "neo4j")
password = os.getenv("NEO4J_PASSWORD", "password123")

print("=== Reseteando Base de Datos local ===")

# 1. Vaciar Neo4j
try:
    print(f"Conectando a Neo4j en {uri}...")
    with GraphDatabase.driver(uri, auth=(user, password)) as driver:
        with driver.session() as session:
            print("Borrando todos los nodos y relaciones en Neo4j...")
            result = session.run("MATCH (n) DETACH DELETE n")
            summary = result.consume()
            print(f"✅ Neo4j vaciado con éxito (nodos eliminados: {summary.counters.nodes_deleted})")
except Exception as e:
    print(f"❌ Error al vaciar Neo4j: {e}")

# 2. Vaciar Qdrant (borrando la carpeta de base de datos local)
qdrant_dir = "./qdrant_memoria"
if os.path.exists(qdrant_dir):
    try:
        print(f"Eliminando base de datos local de Qdrant en '{qdrant_dir}'...")
        shutil.rmtree(qdrant_dir)
        print("✅ Qdrant vaciado con éxito")
    except Exception as e:
        print(f"❌ Error al vaciar Qdrant: {e}")
else:
    print("Qdrant ya estaba vacío (directorio no encontrado).")

# 3. Limpiar carpeta de imágenes extraídas si existiera
imagenes_dir = "./imagenes_extraidas"
if os.path.exists(imagenes_dir):
    try:
        print(f"Limpiando imágenes extraídas en '{imagenes_dir}'...")
        shutil.rmtree(imagenes_dir)
        print("✅ Carpeta de imágenes extraídas limpiada")
    except Exception as e:
        print(f"❌ Error al limpiar imágenes extraídas: {e}")

print("=== Base de datos y almacenamiento reseteados ===")
