# RAG Histología Multimodal v4.2 (Neo4j + LangGraph)

Sistema de Generación Aumentada por Recuperación (RAG) multimodal especializado en histología. Utiliza una base de datos de grafos Neo4j para almacenar y recuperar conocimiento complejo (texto, imágenes, entidades anatómicas) y un flujo de trabajo orquestado con LangGraph para procesar consultas de texto e imagen.

## Características Principales

- **Búsqueda híbrida**: Combina búsqueda semántica por texto, búsqueda visual (UNI + PLIP) y búsqueda por entidades en grafo Neo4j.
- **Extracción inteligente de imágenes**: Las imágenes se muestran al usuario basándose en las referencias que el LLM hace en su respuesta (ej: "Imagen 15.1"), no por similitud visual.
- **Memoria conversacional**: Qdrant mantiene historial de conversación y contexto visual entre turnos.
- **Arquitectura LangGraph**: Flujo de trabajo como grafo de estados con nodos especializados.
- **Modelos especializados**: UNI (Mahmood Lab) y PLIP para embeddings histológicos, Groq (Llama 4) para razonamiento, Gemini para embeddings de texto.

## Arquitectura del Sistema

```
Usuario → Inicializar → Procesar Imagen → Clasificar Dominio
                                              ↓
                                    Generar Consulta Híbrida
                                              ↓
                                    Búsqueda Neo4j (5 estrategias)
                                              ↓
                                    Filtrar y Reranquear Contexto
                                              ↓
                                    Análisis Comparativo Visual
                                              ↓
                                    Generar Respuesta (LLM)
                                              ↓
                                    Finalizar (extraer imágenes referenciadas)
                                              ↓
                                    Respuesta + Imágenes al Frontend
```

### Flujo de Imágenes (v4.2)

El sistema sigue un flujo de dos fases para mostrar imágenes relevantes:

1. **Fase de búsqueda**: `busqueda_hibrida()` recupera chunks de texto e imágenes del grafo mediante 5 estrategias (vectorial, UNI, PLIP, entidades, vecindad).
2. **Fase de generación**: El LLM genera la respuesta usando solo el texto de los chunks. El LLM referencia imágenes por etiqueta (ej: "Imagen 15.1: tejido nervioso neuroglia") porque esa información está en el texto del manual.
3. **Fase de post-procesamiento** (`_nodo_finalizar`): Se parsean las referencias a imágenes de la respuesta del LLM, se buscan los nodos `:Imagen` correspondientes en Neo4j por etiqueta, y se envían al frontend con su path en disco.

Este enfoque garantiza que las imágenes mostradas sean exactamente las que el LLM referencia en su respuesta, no imágenes aleatorias del mismo PDF.

## Esquema de Grafo (Neo4j)

### Nodos

| Nodo | Descripción | Propiedades clave |
|------|-------------|-------------------|
| `PDF` | Documento fuente | `nombre` |
| `Chunk` | Fragmento de texto | `id`, `texto`, `fuente`, `embedding` |
| `Imagen` | Imagen extraída del PDF | `id`, `path`, `fuente`, `pagina`, `etiqueta`, `caption`, `embedding_uni`, `embedding_plip` |
| `Pagina` | Página del PDF | `numero`, `pdf_nombre` |
| `Tejido` | Entidad anatómica | `nombre` |
| `Estructura` | Estructura celular | `nombre` |
| `Tincion` | Técnica de tinción | `nombre` |

### Relaciones

| Relación | Descripción |
|----------|-------------|
| `(Chunk/Imagen)-[:PERTENECE_A]->(PDF)` | Trazabilidad de origen |
| `(Imagen)-[:EN_PAGINA]->(Pagina)` | Ubicación en el documento |
| `(Chunk)-[:MENCIONA]->(Tejido/Estructura/Tincion)` | Vinculación semántica |
| `(Tejido)-[:CONTIENE]->(Estructura)` | Jerarquía anatómica |
| `(Tejido/Estructura)-[:TENIDA_CON]->(Tincion)` | Técnicas de tinción |
| `(Imagen)-[:SIMILAR_A]->(Imagen)` | Similitud visual por embedding UNI |

### Índices Vectoriales

| Índice | Nodo | Propiedad | Dimensiones | Uso |
|--------|------|-----------|-------------|-----|
| `histo_text` | `Chunk` | `embedding` | 384 | Búsqueda semántica de texto |
| `histo_img_uni` | `Imagen` | `embedding_uni` | 1024 | Búsqueda visual UNI |
| `histo_img_plip` | `Imagen` | `embedding_plip` | 512 | Búsqueda visual PLIP |

## Búsqueda Híbrida (5 estrategias)

`busqueda_hibrida()` combina múltiples fuentes de resultados con pesos dinámicos:

| Estrategia | Descripción | Peso (texto) | Peso (imagen) |
|------------|-------------|:------------:|:--------------:|
| Vectorial texto | Búsqueda por embedding de texto en chunks | 0.80 | 0.40 |
| UNI | Búsqueda visual con modelo UNI | 0.20 | 0.70 |
| PLIP | Búsqueda visual con modelo PLIP | 0.20 | 0.70 |
| Entidades | Búsqueda por entidades en el grafo | 0.60 | 0.60 |
| Vecindad | Expansión de vecindad en el grafo | 0.20 | 0.20 |

### Expansión de Vecindad (`expandir_vecindad`)

Dado un conjunto de nodos iniciales, expande el grafo en 4 direcciones:

1. **Expansión 1a**: Chunks de texto del mismo PDF (límite 3)
2. **Expansión 1b**: Imágenes del mismo PDF (límite 5, separadas para garantizar inclusión)
3. **Expansión 2**: Chunks que comparten entidades vía `:MENCIONA` (límite 5)
4. **Expansión 3**: Imágenes similares por embedding vía `:SIMILAR_A` (límite 5)
5. **Expansión 4**: Imágenes de la misma página vía `:EN_PAGINA` (límite 5)

Los resultados de vecindad se marcan con `origen: "vecindad"` para que el filtrado posterior los preserve independientemente de su score de similitud.

## Filtrado de Contexto (`_nodo_filtrar_contexto`)

Aplica umbrales de similitud diferenciados:

| Tipo | Modo texto | Modo imagen |
|------|:----------:|:-----------:|
| Chunks de texto | ≥ 0.45 | ≥ 0.60 |
| Imágenes (búsqueda semántica) | ≥ 0.70 | ≥ 0.70 |
| Imágenes (vecindad, `origen: "vecindad"`) | Sin umbral | Sin umbral |
| Imágenes con path inválido | Rechazadas siempre | Rechazadas siempre |

## Extracción de Imágenes por Referencia (`_nodo_finalizar`)

Después de que el LLM genera la respuesta:

1. **Parseo**: Regex extrae etiquetas como "Imagen 15.1", "Fig 3A", "Figura 5.1" del texto de la respuesta.
2. **Búsqueda en Neo4j**: Busca nodos `:Imagen` cuya `etiqueta`, `caption` o `nombre_archivo` contenga el patrón.
3. **Ordenamiento**: Las imágenes se ordenan según el orden de mención en la respuesta.
4. **Validación**: Se verifica que el archivo existe en disco antes de enviarlo al frontend.
5. **Envío**: Las imágenes se sirven como archivos estáticos desde `/imagenes_extraidas/`.

## Estructura del Proyecto

```
├── ne4j-histo.py              # Módulo principal: Neo4jClient, AsistenteHistologiaNeo4j, LangGraph
├── server.py                  # Servidor FastAPI (endpoints REST)
├── client/                    # Frontend (HTML/JS/CSS)
│   ├── index.html
│   ├── app.js
│   └── style.css
├── pdf/                       # PDFs fuente (manual de histología)
├── imagenes_extraidas/        # Imágenes extraídas de los PDFs (servidas al frontend)
├── imagenes_chat/             # Imágenes subidas por usuarios
├── qdrant_memoria/            # Base de datos Qdrant (memoria conversacional)
├── temario_histologia.json    # Temario extraído del manual
├── trayectoria_neo4j.json     # Última trayectoria de ejecución (debug)
├── .env                       # Variables de entorno (no versionado)
├── .env.example               # Plantilla de variables de entorno
├── pyproject.toml             # Dependencias Python (uv)
└── package.json               # Scripts de ejecución (npm)
```

## Requisitos e Instalación

### Dependencias

```bash
uv sync
```

### Variables de Entorno

Copiar `.env.example` a `.env` y completar:

| Variable | Descripción | Requerida |
|----------|-------------|:---------:|
| `GOOGLE_API_KEY` | API key de Gemini (embeddings de texto) | Sí |
| `GROQ_API_KEY` | API key de Groq (LLM Llama 4) | Sí |
| `HF_TOKEN` | Token de Hugging Face (modelos UNI y PLIP) | Sí |
| `NEO4J_URI` | URI de la base de datos Neo4j | Sí |
| `NEO4J_USERNAME` | Usuario Neo4j | Sí |
| `NEO4J_PASSWORD` | Contraseña Neo4j | Sí |
| `LANGSMITH_API_KEY` | API key de LangSmith (observabilidad) | No |

### Base de Datos Neo4j

Se recomienda [Neo4j AuraDB](https://neo4j.com/cloud/aura/) (tier gratuito disponible). El sistema crea el esquema automáticamente al iniciar si la base está vacía.

## Ejecución

```bash
# Desarrollo (con hot-reload)
npm run dev

# Producción
npm start
```

El servidor escucha en `http://localhost:10005`.

## Cambios v4.2

### Extracción de imágenes por referencia en respuesta

- Las imágenes mostradas al usuario ahora se determinan parseando las referencias del LLM ("Imagen 15.1") y buscando los nodos correspondientes en Neo4j por etiqueta.
- En modo texto, no se envían imágenes de vecindad al LLM para evitar que describa imágenes irrelevantes.
- Regex ampliada para capturar "Imagen X.X", "Fig XA", "Figura X.X".

### Expansión de vecindad mejorada

- Expansión 1 dividida en dos consultas separadas: chunks de texto (límite 3) e imágenes (límite 5).
- Las imágenes del mismo PDF ahora tienen slots dedicados y no compiten con los chunks de texto.
- Todos los resultados de vecindad se marcan con `origen: "vecindad"`.

### Filtrado de imágenes de vecindad

- Imágenes con `origen: "vecindad"` no se filtran por umbral de similitud semántica.
- La validación de path en disco se mantiene para todas las imágenes.
- Imágenes de búsqueda semántica directa siguen usando el umbral de 0.70.

### Boost de imágenes de vecindad

- Las imágenes de vecindad reciben un boost mínimo de similitud (0.50) en `busqueda_hibrida()` para garantizar que entren en el top 15 de resultados.
