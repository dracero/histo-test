# Requirements Document

## Introduction

El sistema RAG multimodal de histología actualmente realiza una búsqueda semántica adicional e independiente cuando el usuario solicita explícitamente ver imágenes (ej: "mostrame una imagen de..."). Esta búsqueda adicional retorna las top-3 imágenes más similares a la consulta mediante el método `busqueda_imagenes_semantica()`, sin aprovechar las imágenes que ya están presentes en `resultados_busqueda`.

El sistema ya recupera imágenes vinculadas al contexto textual a través de la **expansión de vecindad del grafo** (`expandir_vecindad()`), que retorna nodos `:Imagen` conectados a los chunks de texto más relevantes. Estas imágenes están garantizadas a estar relacionadas con el contenido textual porque provienen de la vecindad en el grafo de los chunks recuperados por similitud semántica.

Esta funcionalidad simplificará el sistema para que, cuando el usuario solicite ver imágenes, simplemente **extraiga y muestre las imágenes que ya están en `resultados_busqueda`** (provenientes de la expansión de vecindad), eliminando la necesidad de una búsqueda adicional y garantizando coherencia entre texto e imágenes mostradas.

## Glossary

- **Sistema_RAG**: El sistema de Generación Aumentada por Recuperación (Retrieval-Augmented Generation) multimodal de histología
- **Neo4j**: Base de datos de grafos utilizada para almacenar y recuperar conocimiento histológico
- **Chunk**: Nodo de tipo `:Chunk` en Neo4j que representa un fragmento de texto del manual con propiedades `fuente` (nombre del PDF) y `chunk_id` (índice)
- **Imagen**: Nodo de tipo `:Imagen` en Neo4j que representa una imagen extraída del manual con propiedades `fuente` (nombre del PDF), `pagina` (número de página), `path` (ruta en disco), `caption`, `nombre_archivo` y `etiqueta`
- **Resultados_Busqueda**: Lista de resultados recuperados por búsqueda híbrida en Neo4j que incluye chunks de texto e imágenes relevantes. Las imágenes provienen de la expansión de vecindad del grafo
- **Imagenes_Para_Mostrar**: Lista de imágenes que se envían al frontend para mostrar al usuario cuando solicita explícitamente ver imágenes
- **Solicitud_Imagen_Explicita**: Consulta del usuario que solicita ver, mostrar o buscar imágenes de la base de datos (detectada por `_detectar_solicitud_imagen()`)
- **Busqueda_Hibrida**: Método que combina búsqueda vectorial de texto, búsqueda por entidades, y expansión de vecindad del grafo para recuperar contexto relevante
- **Expansion_Vecindad**: Método `expandir_vecindad()` que retorna nodos conectados en el grafo a los chunks más relevantes, incluyendo nodos `:Imagen` del mismo PDF, misma página, o relacionados por entidades
- **Filtro_Imagenes_Vecindad**: Nuevo mecanismo que extrae imágenes de `resultados_busqueda` que ya fueron recuperadas por expansión de vecindad
- **Nodo_Vecino**: Nodo en el grafo conectado a un chunk relevante mediante relaciones como `:PERTENECE_A`, `:MENCIONA`, `:SIMILAR_A`, o `:EN_PAGINA`

## Requirements

### Requirement 1: Extracción de Imágenes de Resultados de Búsqueda

**User Story:** Como sistema procesando una solicitud de imágenes, necesito extraer las imágenes que ya están en los resultados de búsqueda híbrida, para mostrarlas al usuario sin realizar búsquedas adicionales.

#### Acceptance Criteria

1. WHEN el Sistema_RAG ejecuta filtrado de imágenes, THEN THE Sistema_RAG SHALL filtrar Resultados_Busqueda para obtener solo resultados donde `tipo == "imagen"`
2. THE Sistema_RAG SHALL extraer de cada resultado de tipo imagen las propiedades: `id`, `path`, `texto` (caption), `nombre_archivo`, `etiqueta`, `fuente`, `similitud`
3. WHEN Resultados_Busqueda está vacío, THEN THE Sistema_RAG SHALL retornar lista vacía de imágenes
4. WHEN Resultados_Busqueda no contiene resultados de tipo "imagen", THEN THE Sistema_RAG SHALL retornar lista vacía de imágenes
5. THE Sistema_RAG SHALL preservar el orden de similitud de los resultados originales

### Requirement 2: Validación de Integridad de Imágenes

**User Story:** Como sistema procesando una solicitud de imágenes, necesito validar que las imágenes extraídas tengan datos completos y archivos existentes, para evitar errores en el frontend.

#### Acceptance Criteria

1. THE Sistema_RAG SHALL verificar que cada imagen extraída tenga propiedad `path` no nula
2. THE Sistema_RAG SHALL verificar que el archivo en `path` existe en el sistema de archivos usando `os.path.exists()`
3. WHEN el archivo no existe, THEN THE Sistema_RAG SHALL omitir esa imagen y registrar advertencia con el path faltante
4. THE Sistema_RAG SHALL verificar que cada imagen tenga propiedad `nombre_archivo` no vacía
5. WHEN `nombre_archivo` está vacío, THEN THE Sistema_RAG SHALL usar `os.path.basename(path)` como fallback
6. THE Sistema_RAG SHALL garantizar que cada imagen retornada tenga todas las propiedades requeridas: `id`, `path`, `caption`, `nombre_archivo`, `etiqueta`, `fuente`, `similitud_semantica`

### Requirement 3: Renombrado de Propiedad de Similitud

**User Story:** Como sistema enviando imágenes al frontend, necesito renombrar la propiedad `similitud` a `similitud_semantica`, para mantener consistencia con la interfaz esperada.

#### Acceptance Criteria

1. THE Sistema_RAG SHALL renombrar la propiedad `similitud` a `similitud_semantica` en cada imagen extraída
2. THE valor de `similitud_semantica` SHALL ser el mismo que el valor original de `similitud`
3. THE Sistema_RAG SHALL eliminar la propiedad `similitud` original después del renombrado
4. WHEN una imagen no tiene propiedad `similitud`, THEN THE Sistema_RAG SHALL asignar `similitud_semantica = 0.0` como valor por defecto

### Requirement 4: Renombrado de Propiedad de Caption

**User Story:** Como sistema enviando imágenes al frontend, necesito renombrar la propiedad `texto` a `caption`, para mantener consistencia con la interfaz esperada.

#### Acceptance Criteria

1. THE Sistema_RAG SHALL renombrar la propiedad `texto` a `caption` en cada imagen extraída
2. THE valor de `caption` SHALL ser el mismo que el valor original de `texto`
3. THE Sistema_RAG SHALL eliminar la propiedad `texto` original después del renombrado
4. WHEN una imagen no tiene propiedad `texto`, THEN THE Sistema_RAG SHALL asignar `caption = ""` como valor por defecto

### Requirement 5: Limitación de Resultados

**User Story:** Como usuario solicitando imágenes, quiero recibir un número limitado de imágenes relevantes, para evitar sobrecarga visual y mantener foco en las más importantes.

#### Acceptance Criteria

1. THE Sistema_RAG SHALL limitar el número de imágenes retornadas al parámetro `top_k` (por defecto 3)
2. THE Sistema_RAG SHALL retornar las primeras `top_k` imágenes después de filtrado y validación
3. WHEN el número de imágenes válidas es menor que `top_k`, THEN THE Sistema_RAG SHALL retornar todas las imágenes válidas disponibles
4. THE Sistema_RAG SHALL preservar el orden de similitud original al aplicar el límite

### Requirement 6: Modificación de _nodo_buscar_neo4j

**User Story:** Como sistema procesando una solicitud de imágenes, necesito extraer imágenes de los resultados de búsqueda en lugar de realizar búsqueda adicional, para simplificar el flujo y garantizar coherencia.

#### Acceptance Criteria

1. WHEN `state["mostrar_imagenes"]` es verdadero, THEN THE nodo `_nodo_buscar_neo4j` SHALL invocar nueva función `extraer_imagenes_de_resultados()` con parámetro `resultados=state["resultados_busqueda"]`
2. THE nodo SHALL invocar la función después de que `resultados_busqueda` haya sido poblado por búsqueda híbrida
3. THE nodo SHALL almacenar el resultado en `state["imagenes_para_mostrar"]`
4. THE nodo SHALL NO invocar `busqueda_imagenes_semantica()` cuando `mostrar_imagenes` es verdadero
5. WHEN la función retorna lista vacía, THEN THE nodo SHALL registrar advertencia "No se encontraron imágenes en los resultados recuperados"
6. WHEN la función retorna imágenes, THEN THE nodo SHALL registrar número de imágenes encontradas y establecer `state["contexto_suficiente"] = True`

### Requirement 7: Logging y Trazabilidad

**User Story:** Como desarrollador del sistema, necesito registrar información detallada sobre la extracción de imágenes, para depuración y análisis de rendimiento.

#### Acceptance Criteria

1. THE Sistema_RAG SHALL registrar en consola el número total de resultados en Resultados_Busqueda
2. THE Sistema_RAG SHALL registrar en consola el número de resultados de tipo "imagen" encontrados
3. THE Sistema_RAG SHALL registrar en consola el número de imágenes válidas después de validación
4. THE Sistema_RAG SHALL registrar en consola las top-3 imágenes con formato: `nombre_archivo | sim={similitud:.3f} | {fuente}`
5. WHEN no se encuentran imágenes en los resultados, THEN THE Sistema_RAG SHALL registrar advertencia específica "No hay imágenes en los resultados de búsqueda"
6. THE Sistema_RAG SHALL incluir en trayectoria el campo `imagenes_extraidas_de_vecindad` con valor booleano verdadero

### Requirement 8: Manejo de Casos Especiales

**User Story:** Como sistema procesando una solicitud de imágenes, necesito manejar casos especiales donde no hay imágenes disponibles, para proporcionar feedback apropiado.

#### Acceptance Criteria

1. WHEN Resultados_Busqueda está vacío Y `mostrar_imagenes` es verdadero, THEN THE Sistema_RAG SHALL retornar lista vacía y registrar advertencia
2. WHEN todas las imágenes son filtradas por validación de integridad, THEN THE Sistema_RAG SHALL retornar lista vacía
3. WHEN la propiedad `path` de una imagen no existe en disco, THEN THE Sistema_RAG SHALL omitir esa imagen de los resultados finales
4. WHEN Resultados_Busqueda contiene solo resultados de tipo "texto", THEN THE Sistema_RAG SHALL retornar lista vacía con mensaje informativo
5. THE Sistema_RAG SHALL continuar procesando todas las imágenes disponibles incluso si algunas fallan validación

### Requirement 9: Eliminación de Búsqueda Adicional

**User Story:** Como desarrollador del sistema, necesito eliminar la invocación de `busqueda_imagenes_semantica()` cuando el usuario solicita imágenes, para simplificar el código y evitar búsquedas redundantes.

#### Acceptance Criteria

1. WHEN `state["mostrar_imagenes"]` es verdadero, THEN THE nodo `_nodo_buscar_neo4j` SHALL NOT invocar `busqueda_imagenes_semantica()`
2. THE método `busqueda_imagenes_semantica()` SHALL permanecer en el código sin modificaciones (para uso futuro o legacy)
3. THE Sistema_RAG SHALL usar únicamente `extraer_imagenes_de_resultados()` para obtener imágenes cuando el usuario las solicita
4. THE cambio SHALL ser transparente para el usuario (mismo comportamiento observable)

### Requirement 10: Formato de Salida Consistente

**User Story:** Como frontend consumiendo imágenes, necesito recibir imágenes en el formato esperado, para renderizarlas correctamente sin cambios en el código del cliente.

#### Acceptance Criteria

1. THE Sistema_RAG SHALL retornar lista de diccionarios con estructura idéntica a la actual
2. EACH imagen en la lista SHALL contener claves: `id`, `path`, `caption`, `nombre_archivo`, `etiqueta`, `fuente`, `similitud_semantica`
3. THE formato de valores SHALL ser: `id` (string), `path` (string), `caption` (string), `nombre_archivo` (string), `etiqueta` (string), `fuente` (string), `similitud_semantica` (float)
4. THE Sistema_RAG SHALL garantizar compatibilidad con el código del frontend existente sin modificaciones
5. THE orden de las claves en el diccionario SHALL ser irrelevante (JSON no garantiza orden)

