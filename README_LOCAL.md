# Guía de Inicio Rápido: Neo4j Local en Espacio de Usuario

Para evitar requerir permisos de administrador (`sudo`) o depender de un demonio de Docker (que no está instalado en el sistema), hemos configurado un entorno **standalone local** (User Space) para **Neo4j Community Edition 5.20.0** y **Java JDK 21** dentro de este proyecto.

Toda la base de datos y su Java Runtime están aislados en el directorio `neo4j-local/` (el cual está correctamente ignorado en `.gitignore` para no subirlo al repositorio).

---

## 🛠️ Herramientas de Control

Hemos creado tres scripts muy sencillos en la raíz del proyecto para controlar el ciclo de vida de la base de datos:

*   **Iniciar:** `./start-neo4j.sh` (Inicia el servidor en segundo plano)
*   **Estado:** `./status-neo4j.sh` (Muestra si está corriendo y en qué PID)
*   **Detener:** `./stop-neo4j.sh` (Detiene el servidor)
*   **Resetear Bases de Datos:** `npm run reset` (Vacía Neo4j, Qdrant y limpia las imágenes para re-indexar desde cero)

---

## 📝 Detalles de la Configuración Realizada

1. **Java Runtime:** Descargamos e instalamos **Eclipse Temurin JDK 21 (x64)** en `neo4j-local/jdk`.
2. **Neo4j Standalone:** Descargamos **Neo4j Community 5.20.0** en `neo4j-local/neo4j`.
3. **APOC Plugin Habilitado:**
   - Copiamos el JAR de APOC Core a `plugins/`.
   - Configuramos `neo4j.conf` para habilitar `apoc.*` sin restricciones.
   - Creamos `apoc.conf` para permitir importaciones/exportaciones de archivos.
4. **Contraseña Inicial:** Configurada a `password123`.
5. **Configuración de Variables de Entorno:**
   - Editamos el archivo `.env` en la raíz del proyecto para apuntar a la instancia local (`bolt://localhost:7687` con credenciales `neo4j` / `password123`).

---

## 🚀 Servidor RAG Histología Neo4j

El servidor FastAPI ya está corriendo localmente con el nuevo entorno:
*   Ha conectado con éxito a la base de datos Neo4j local.
*   Inicializó el esquema de Neo4j (3 índices).
*   Leyó e indexó los PDFs (`arch3.pdf` y `arch4.pdf`) con sus textos, imágenes y relaciones visuales.
*   El servidor está listo para usar en:
    👉 **`http://localhost:10005`**
