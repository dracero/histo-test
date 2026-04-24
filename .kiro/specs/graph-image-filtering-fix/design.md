# Graph Image Filtering Fix - Bugfix Design

## Overview

This bugfix addresses incorrect filtering of contextually relevant images retrieved via graph neighborhood expansion. The system correctly retrieves images through Neo4j graph relationships (with similitud scores of 0.3 or 0.95), but the filtering logic in `_nodo_filtrar_contexto()` removes images with similitud < 0.70 before they reach the user. The fix adds an `origen: "vecindad"` marker to images retrieved by `expandir_vecindad()` and modifies the filtering logic to skip similarity threshold checks for these marked images, preserving them based on graph connectivity rather than semantic similarity.

**Impact**: Users querying topics like "tejido muscular" will see relevant images from the knowledge graph instead of text-only responses.

**Files Modified**: `ne4j-histo.py` (methods `expandir_vecindad()` and `_nodo_filtrar_contexto()`)

## Glossary

- **Bug_Condition (C)**: The condition that triggers the bug - when images retrieved via `expandir_vecindad()` with similitud < 0.70 are filtered out despite being contextually relevant through graph relationships
- **Property (P)**: The desired behavior - images from graph neighborhood expansion should be preserved regardless of similarity score because graph connectivity guarantees contextual relevance
- **Preservation**: Existing filtering behavior for non-graph images (semantic search results) and text chunks must remain unchanged
- **expandir_vecindad()**: Method in `Neo4jClient` (line 701) that retrieves neighboring nodes via graph relationships (shared PDF, entities, similar embeddings, same page)
- **_nodo_filtrar_contexto()**: Method in `AsistenteHistologiaNeo4j` (line 2686) that filters search results based on similarity thresholds
- **SIMILARITY_THRESHOLD**: Class constant (0.70) used to filter images based on semantic similarity
- **origen**: New marker field to distinguish graph-expanded images from semantic search images
- **resultados_busqueda**: State field containing combined results from semantic search and graph expansion

## Bug Details

### Bug Condition

The bug manifests when images retrieved via graph neighborhood expansion are filtered out before reaching the user. The `expandir_vecindad()` method correctly retrieves images with similitud=0.3 (from graph relationships) or similitud=0.95 (same page), but `_nodo_filtrar_contexto()` applies a blanket similarity threshold (0.70) that removes the 0.3-scored images, even though they are contextually relevant by virtue of graph connectivity.

**Formal Specification:**
```
FUNCTION isBugCondition(input)
  INPUT: input of type Dict (result dictionary with keys: tipo, similitud, origen)
  OUTPUT: boolean
  
  RETURN input["tipo"] == "imagen"
         AND input["similitud"] < 0.70
         AND input was retrieved by expandir_vecindad()
         AND NOT input.get("origen") == "vecindad"  // Bug: no marker exists
END FUNCTION
```

**Explanation**: The bug occurs when:
1. An image node is retrieved via `expandir_vecindad()` through graph relationships
2. The image receives similitud=0.3 (standard score for graph neighbors)
3. The image lacks an `origen` marker to identify its source
4. `_nodo_filtrar_contexto()` filters it out because 0.3 < 0.70

### Examples

**Example 1: Shared PDF Relationship**
- User queries: "tejido muscular"
- `expandir_vecindad()` retrieves: "Imagen 13.1: Tejido Muscular Estriado Voluntario" (similitud=0.3, from shared PDF relationship)
- **Current behavior**: Image filtered out (0.3 < 0.70)
- **Expected behavior**: Image preserved (graph relationship guarantees relevance)

**Example 2: Shared Entity Relationship**
- User queries: "epitelio estratificado"
- `expandir_vecindad()` retrieves: Image of stratified epithelium (similitud=0.3, from shared entity "Epitelio")
- **Current behavior**: Image filtered out (0.3 < 0.70)
- **Expected behavior**: Image preserved (entity relationship guarantees relevance)

**Example 3: Same Page Relationship (Already Works)**
- User queries: "tejido conectivo"
- `expandir_vecindad()` retrieves: Image from same page as matched chunk (similitud=0.95)
- **Current behavior**: Image preserved (0.95 > 0.70) ✓
- **Expected behavior**: Image preserved (no change needed)

**Example 4: Edge Case - Invalid Path**
- `expandir_vecindad()` retrieves: Image with similitud=0.3 but imagen_path is null or file doesn't exist
- **Current behavior**: Image filtered out (invalid path check)
- **Expected behavior**: Image filtered out (invalid images should always be removed, regardless of origen)

## Expected Behavior

### Preservation Requirements

**Unchanged Behaviors:**
- Images from semantic search (not graph expansion) with similitud < 0.70 must continue to be filtered out
- Text chunks with similitud below their thresholds (0.45 or 0.6) must continue to be filtered out
- Images with invalid paths (null or non-existent files) must continue to be filtered out regardless of origen
- The weighting and combination logic in `busqueda_hibrida()` must remain unchanged
- The sorting and display logic for valid results must remain unchanged

**Scope:**
All inputs that do NOT originate from `expandir_vecindad()` should be completely unaffected by this fix. This includes:
- Images from semantic search (`busqueda_imagenes_uni()`, `busqueda_imagenes_plip()`)
- Images from text-based search (`busqueda_imagenes_por_texto()`)
- Text chunks from any search method
- Results when graph expansion is not invoked

## Hypothesized Root Cause

Based on the bug description and code analysis, the root cause is:

1. **Missing Origin Tracking**: The `expandir_vecindad()` method does not add any marker field to distinguish graph-expanded images from semantic search images. All images look identical to downstream filtering logic.

2. **Blanket Similarity Threshold**: The `_nodo_filtrar_contexto()` method applies the same similarity threshold (0.70) to all images, regardless of their retrieval method. This is appropriate for semantic search but incorrect for graph expansion.

3. **Semantic vs. Structural Relevance Confusion**: The code conflates two different types of relevance:
   - **Semantic relevance**: Measured by embedding similarity (appropriate for semantic search)
   - **Structural relevance**: Guaranteed by graph relationships (appropriate for graph expansion)

4. **Data Flow Gap**: Results from `expandir_vecindad()` are merged with semantic search results in `busqueda_hibrida()` using the `agregar()` function, which preserves the similitud score but doesn't track the origin. By the time results reach `_nodo_filtrar_contexto()`, there's no way to distinguish graph-expanded images.

## Correctness Properties

Property 1: Bug Condition - Graph-Expanded Images Are Preserved

_For any_ image result where the bug condition holds (image retrieved via expandir_vecindad with similitud < 0.70), the fixed _nodo_filtrar_contexto function SHALL preserve the image in resultados_validos, bypassing the similarity threshold check because graph connectivity guarantees contextual relevance.

**Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5**

Property 2: Preservation - Non-Graph Images Filtered Identically

_For any_ image result that does NOT originate from expandir_vecindad (semantic search images, text chunks, invalid images), the fixed _nodo_filtrar_contexto function SHALL produce exactly the same filtering decision as the original function, preserving all existing threshold-based filtering behavior.

**Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5**

## Fix Implementation

### Changes Required

Assuming our root cause analysis is correct:

**File**: `ne4j-histo.py`

**Change 1: Add origen marker in expandir_vecindad()**

**Location**: Line 701, method `expandir_vecindad()` in class `Neo4jClient`

**Specific Changes**:
Modify the Cypher query's RETURN clause to add an `origen` field:

```python
# BEFORE (line ~760):
RETURN DISTINCT
    v.id AS id,
    CASE 
        WHEN v:Imagen THEN 
            CASE 
                WHEN v.caption IS NOT NULL AND v.caption <> '' THEN v.caption
                WHEN v.texto_pagina IS NOT NULL AND v.texto_pagina <> '' THEN v.texto_pagina
                WHEN v.ocr_text IS NOT NULL AND v.ocr_text <> '' THEN v.ocr_text
                ELSE ''
            END
        ELSE coalesce(v.texto, '') 
    END AS texto,
    v.fuente AS fuente,
    CASE WHEN v:Imagen THEN 'imagen' ELSE 'texto' END AS tipo,
    CASE WHEN v:Imagen THEN v.path ELSE null END AS imagen_path,
    CASE 
        WHEN (n:Imagen AND v:Imagen AND n.pagina = v.pagina) OR
             (n:Chunk AND v:Imagen AND n.fuente = v.fuente)
        THEN 0.95 
        ELSE 0.3 
    END AS similitud,
    CASE WHEN v:Imagen THEN coalesce(v.nombre_archivo, '') ELSE '' END AS nombre_archivo,
    CASE WHEN v:Imagen THEN coalesce(v.etiqueta, '') ELSE '' END AS etiqueta
LIMIT 15

# AFTER (add origen field):
RETURN DISTINCT
    v.id AS id,
    CASE 
        WHEN v:Imagen THEN 
            CASE 
                WHEN v.caption IS NOT NULL AND v.caption <> '' THEN v.caption
                WHEN v.texto_pagina IS NOT NULL AND v.texto_pagina <> '' THEN v.texto_pagina
                WHEN v.ocr_text IS NOT NULL AND v.ocr_text <> '' THEN v.ocr_text
                ELSE ''
            END
        ELSE coalesce(v.texto, '') 
    END AS texto,
    v.fuente AS fuente,
    CASE WHEN v:Imagen THEN 'imagen' ELSE 'texto' END AS tipo,
    CASE WHEN v:Imagen THEN v.path ELSE null END AS imagen_path,
    CASE 
        WHEN (n:Imagen AND v:Imagen AND n.pagina = v.pagina) OR
             (n:Chunk AND v:Imagen AND n.fuente = v.fuente)
        THEN 0.95 
        ELSE 0.3 
    END AS similitud,
    CASE WHEN v:Imagen THEN coalesce(v.nombre_archivo, '') ELSE '' END AS nombre_archivo,
    CASE WHEN v:Imagen THEN coalesce(v.etiqueta, '') ELSE '' END AS etiqueta,
    'vecindad' AS origen
LIMIT 15
```

**Rationale**: Adding `'vecindad' AS origen` to the RETURN clause marks all results from graph expansion with a distinctive field that downstream filtering logic can check.

**Change 2: Modify filtering logic in _nodo_filtrar_contexto()**

**Location**: Line 2686, method `_nodo_filtrar_contexto()` in class `AsistenteHistologiaNeo4j`

**Specific Changes**:
Modify the image filtering logic to skip similarity threshold for graph-expanded images:

```python
# BEFORE (lines ~2695-2698):
for r in state["resultados_busqueda"]:
    current_sim = r.get("similitud", 0)
    if r.get("tipo") == "texto" and current_sim < umbral_texto:
        continue
    if r.get("tipo") == "imagen" and current_sim < umbral_imagen:
        continue

    # Si es imagen pero no existe el archivo en disco, lo rechazamos
    if r.get("tipo") == "imagen":
        img_p = r.get("imagen_path")
        if not img_p or not os.path.exists(img_p):
            continue
    validos.append(r)

# AFTER (add origen check):
for r in state["resultados_busqueda"]:
    current_sim = r.get("similitud", 0)
    
    # Filter text chunks by threshold
    if r.get("tipo") == "texto" and current_sim < umbral_texto:
        continue
    
    # Filter images by threshold, UNLESS they come from graph expansion
    if r.get("tipo") == "imagen":
        es_vecindad = r.get("origen") == "vecindad"
        if not es_vecindad and current_sim < umbral_imagen:
            continue

    # Si es imagen pero no existe el archivo en disco, lo rechazamos
    # (applies to ALL images, regardless of origen)
    if r.get("tipo") == "imagen":
        img_p = r.get("imagen_path")
        if not img_p or not os.path.exists(img_p):
            continue
    
    validos.append(r)
```

**Rationale**: 
- Check if `origen == "vecindad"` before applying similarity threshold
- Graph-expanded images bypass the threshold check entirely
- Invalid path check still applies to ALL images (preservation requirement)
- Text filtering logic remains unchanged

**Change 3: No changes needed in busqueda_hibrida()**

The `agregar()` function in `busqueda_hibrida()` (line ~1160) already preserves all fields from result dictionaries when combining results. The `origen` field will automatically flow through to `resultados_busqueda` without any code changes.

## Testing Strategy

### Validation Approach

The testing strategy follows a two-phase approach: first, surface counterexamples that demonstrate the bug on unfixed code, then verify the fix works correctly and preserves existing behavior.

### Exploratory Bug Condition Checking

**Goal**: Surface counterexamples that demonstrate the bug BEFORE implementing the fix. Confirm or refute the root cause analysis. If we refute, we will need to re-hypothesize.

**Test Plan**: Write tests that simulate queries triggering graph expansion, inspect the `resultados_busqueda` before and after filtering, and verify that images with similitud=0.3 are incorrectly removed. Run these tests on the UNFIXED code to observe failures and understand the root cause.

**Test Cases**:
1. **Graph Expansion Filtering Test**: Query "tejido muscular", verify that `expandir_vecindad()` returns images with similitud=0.3, then verify that `_nodo_filtrar_contexto()` removes them (will fail on unfixed code - images should be preserved)
2. **Same Page Preservation Test**: Query that triggers same-page expansion (similitud=0.95), verify images are preserved (should pass on unfixed code - already works)
3. **Semantic Search Filtering Test**: Query with direct semantic search images (similitud=0.5), verify they are correctly filtered out (should pass on unfixed code - existing behavior)
4. **Invalid Path Test**: Simulate graph-expanded image with invalid path, verify it's filtered out (should pass on unfixed code - existing behavior)

**Expected Counterexamples**:
- Images with similitud=0.3 from `expandir_vecindad()` are removed from `resultados_validos`
- Possible causes: missing `origen` marker, blanket similarity threshold application

### Fix Checking

**Goal**: Verify that for all inputs where the bug condition holds, the fixed function produces the expected behavior.

**Pseudocode:**
```
FOR ALL input WHERE isBugCondition(input) DO
  result := _nodo_filtrar_contexto'(state_containing_input)
  ASSERT input IN result["resultados_validos"]
  ASSERT "Graph-expanded images are preserved regardless of similitud score"
END FOR
```

**Test Implementation**:
```python
def test_fix_checking_graph_images_preserved():
    # Setup: Create state with graph-expanded image (similitud=0.3)
    state = {
        "resultados_busqueda": [
            {
                "id": "img_001",
                "tipo": "imagen",
                "similitud": 0.3,
                "origen": "vecindad",  # Marker added by fix
                "imagen_path": "/valid/path.png",
                "texto": "Tejido muscular",
                "fuente": "arch2.pdf"
            }
        ],
        "tiene_imagen": False
    }
    
    # Execute fixed function
    result = await asistente._nodo_filtrar_contexto(state)
    
    # Assert: Image should be in resultados_validos
    assert len(result["resultados_validos"]) == 1
    assert result["resultados_validos"][0]["id"] == "img_001"
    assert result["resultados_validos"][0]["similitud"] == 0.3
```

### Preservation Checking

**Goal**: Verify that for all inputs where the bug condition does NOT hold, the fixed function produces the same result as the original function.

**Pseudocode:**
```
FOR ALL input WHERE NOT isBugCondition(input) DO
  ASSERT _nodo_filtrar_contexto_original(state) = _nodo_filtrar_contexto_fixed(state)
  ASSERT "Non-graph-expanded images are filtered identically before and after fix"
END FOR
```

**Testing Approach**: Property-based testing is recommended for preservation checking because:
- It generates many test cases automatically across the input domain
- It catches edge cases that manual unit tests might miss
- It provides strong guarantees that behavior is unchanged for all non-buggy inputs

**Test Plan**: Observe behavior on UNFIXED code first for semantic search images and text chunks, then write property-based tests capturing that behavior.

**Test Cases**:
1. **Semantic Search Image Preservation**: Observe that images with similitud < 0.70 from semantic search are filtered out on unfixed code, then write test to verify this continues after fix
2. **Text Chunk Preservation**: Observe that text chunks with similitud < threshold are filtered out on unfixed code, then write test to verify this continues after fix
3. **Invalid Path Preservation**: Observe that images with invalid paths are filtered out on unfixed code, then write test to verify this continues after fix (even for graph-expanded images)
4. **High-Similarity Image Preservation**: Observe that images with similitud > 0.70 are preserved on unfixed code, then write test to verify this continues after fix

**Test Implementation**:
```python
def test_preservation_semantic_search_images():
    # Setup: Create state with semantic search image (no origen marker)
    state = {
        "resultados_busqueda": [
            {
                "id": "img_002",
                "tipo": "imagen",
                "similitud": 0.5,
                # No origen field - semantic search image
                "imagen_path": "/valid/path.png",
                "texto": "Some image",
                "fuente": "arch2.pdf"
            }
        ],
        "tiene_imagen": False
    }
    
    # Execute fixed function
    result = await asistente._nodo_filtrar_contexto(state)
    
    # Assert: Image should be filtered out (0.5 < 0.70)
    assert len(result["resultados_validos"]) == 0
```

### Unit Tests

- Test that `expandir_vecindad()` adds `origen: "vecindad"` to all returned results
- Test that `_nodo_filtrar_contexto()` preserves images with `origen: "vecindad"` regardless of similitud
- Test that `_nodo_filtrar_contexto()` filters images without `origen` marker when similitud < 0.70
- Test that invalid path check applies to all images (with and without origen marker)
- Test text chunk filtering remains unchanged
- Test edge cases: empty resultados_busqueda, missing fields, null values

### Property-Based Tests

- Generate random similitud scores (0.0 to 1.0) for graph-expanded images and verify all are preserved
- Generate random similitud scores for semantic search images and verify filtering matches threshold
- Generate random combinations of graph-expanded and semantic search images and verify correct filtering
- Generate random invalid paths and verify all are filtered out regardless of origen

### Integration Tests

- Test full query flow: user query → semantic search → graph expansion → filtering → display
- Test that images from graph expansion appear in final response
- Test that semantic search images below threshold do not appear in final response
- Test mixed scenarios: some images from graph, some from semantic search
- Test that the fix works across different query types (text-only, image+text)
