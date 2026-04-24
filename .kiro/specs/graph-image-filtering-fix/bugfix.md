# Bugfix Requirements Document

## Introduction

This document specifies the requirements for fixing a bug where images retrieved via graph neighborhood expansion are incorrectly filtered out before being displayed to users. The system correctly retrieves images through graph relationships (with similitud scores of 0.3 or 0.95), but the filtering logic in `_nodo_filtrar_contexto()` removes images with similitud < 0.70, preventing users from seeing contextually relevant images that exist in the knowledge graph.

**Impact**: Users querying topics like "tejido muscular" receive text responses but no images, despite relevant images being retrieved from the graph and existing in the database.

**Affected Component**: `ne4j-histo.py`, method `_nodo_filtrar_contexto()` (lines 2686-2710)

## Bug Analysis

### Current Behavior (Defect)

1.1 WHEN `expandir_vecindad()` retrieves images with similitud=0.3 (from graph relationships like shared PDF, entities, or similar embeddings) THEN the system filters out these images in `_nodo_filtrar_contexto()` because 0.3 < SIMILARITY_THRESHOLD (0.70)

1.2 WHEN images are assigned similitud=0.95 (same page as query chunk) by `expandir_vecindad()` THEN the system correctly preserves these images through the filtering stage

1.3 WHEN `extraer_imagenes_de_resultados()` attempts to extract images from `resultados_busqueda` THEN the system finds no images because they were already removed by `_nodo_filtrar_contexto()`

1.4 WHEN users query topics with relevant images in the graph (e.g., "tejido muscular") THEN the system displays text responses but no images, despite images being initially retrieved

### Expected Behavior (Correct)

2.1 WHEN `expandir_vecindad()` retrieves images with similitud=0.3 from graph relationships THEN the system SHALL preserve these images through the filtering stage because graph connectivity guarantees contextual relevance

2.2 WHEN `expandir_vecindad()` retrieves images with similitud=0.95 from same-page relationships THEN the system SHALL continue to preserve these images through the filtering stage

2.3 WHEN `extraer_imagenes_de_resultados()` attempts to extract images from `resultados_busqueda` THEN the system SHALL find and extract images that originated from graph neighborhood expansion

2.4 WHEN users query topics with relevant images in the graph (e.g., "tejido muscular") THEN the system SHALL display both text responses and the contextually relevant images retrieved via graph expansion

2.5 WHEN images originate from graph neighborhood expansion (marked with `origen: "vecindad"` or similar marker) THEN the system SHALL skip semantic similarity threshold filtering for these images

### Unchanged Behavior (Regression Prevention)

3.1 WHEN images are retrieved via direct semantic search (not from graph expansion) AND have similitud < 0.70 THEN the system SHALL CONTINUE TO filter out these images using the SIMILARITY_THRESHOLD

3.2 WHEN text chunks have similitud below their respective thresholds (0.45 or 0.6 depending on mode) THEN the system SHALL CONTINUE TO filter out these chunks as before

3.3 WHEN images have invalid paths (path is null or file does not exist on disk) THEN the system SHALL CONTINUE TO filter out these images regardless of their origin

3.4 WHEN `expandir_vecindad()` is not invoked (no graph expansion occurs) THEN the system SHALL CONTINUE TO apply standard similarity threshold filtering to all images

3.5 WHEN the filtering logic processes non-image results (tipo != "imagen") THEN the system SHALL CONTINUE TO apply existing filtering rules without modification


## Bug Condition and Property Specification

### Bug Condition Function

The bug condition identifies images that are incorrectly filtered out:

```pascal
FUNCTION isBugCondition(X)
  INPUT: X of type ResultadoBusqueda (dictionary with tipo, similitud, origen fields)
  OUTPUT: boolean
  
  // Returns true when an image from graph expansion is incorrectly filtered
  RETURN (X.tipo = "imagen") 
         AND (X.similitud < 0.70) 
         AND (X.origen = "vecindad" OR X was retrieved by expandir_vecindad())
END FUNCTION
```

**Explanation**: The bug occurs when an image (tipo="imagen") has a low similarity score (< 0.70) but originates from graph neighborhood expansion. These images are contextually relevant by virtue of graph connectivity, but the current code filters them out based solely on semantic similarity threshold.

### Property Specification: Fix Checking

The fixed code must preserve images from graph expansion regardless of similarity score:

```pascal
// Property: Fix Checking - Preserve Graph-Expanded Images
FOR ALL X WHERE isBugCondition(X) DO
  result ← _nodo_filtrar_contexto'(state_containing_X)
  ASSERT X IN result.resultados_validos
  ASSERT "Images from graph expansion are preserved regardless of similitud score"
END FOR
```

**Key Definitions:**
- **F**: `_nodo_filtrar_contexto()` - Original filtering function that removes images with similitud < 0.70
- **F'**: `_nodo_filtrar_contexto'()` - Fixed filtering function that preserves graph-expanded images

### Property Specification: Preservation Checking

For all images NOT from graph expansion, filtering behavior must remain unchanged:

```pascal
// Property: Preservation Checking - Maintain Existing Filtering
FOR ALL X WHERE NOT isBugCondition(X) DO
  result_original ← F(state_containing_X)
  result_fixed ← F'(state_containing_X)
  
  ASSERT (X IN result_original.resultados_validos) 
         ⟺ 
         (X IN result_fixed.resultados_validos)
  
  ASSERT "Non-graph-expanded images are filtered identically before and after fix"
END FOR
```

**Preservation Goal**: Images from direct semantic search, text chunks, and invalid images (missing paths) must be filtered exactly as before.

### Concrete Counterexample

**Input**: Query "tejido muscular"

**Buggy Execution (F)**:
1. `expandir_vecindad()` retrieves "Imagen 13.1: Tejido Muscular Estriado Voluntario" with similitud=0.3
2. `_nodo_filtrar_contexto()` filters out image because 0.3 < 0.70
3. `extraer_imagenes_de_resultados()` finds no images
4. User sees text response but no images

**Expected Execution (F')**:
1. `expandir_vecindad()` retrieves "Imagen 13.1: Tejido Muscular Estriado Voluntario" with similitud=0.3 and origen="vecindad"
2. `_nodo_filtrar_contexto'()` preserves image because origen="vecindad" (skip similarity threshold)
3. `extraer_imagenes_de_resultados()` extracts image successfully
4. User sees text response AND relevant image

### Implementation Approach

To distinguish between graph-expanded images and semantic search images:

**Option A: Add marker field during graph expansion**
```python
# In expandir_vecindad(), add origen marker:
return {
    "id": v.id,
    "tipo": "imagen",
    "similitud": 0.3,
    "origen": "vecindad",  # NEW: Marker field
    # ... other fields
}
```

**Option B: Track graph-expanded IDs separately**
```python
# In _nodo_buscar_neo4j(), track IDs:
vecindad_image_ids = {r["id"] for r in vecindad_results if r.get("tipo") == "imagen"}
state["vecindad_image_ids"] = vecindad_image_ids

# In _nodo_filtrar_contexto(), check membership:
if r.get("tipo") == "imagen":
    if r["id"] in state.get("vecindad_image_ids", set()):
        # Skip similarity threshold for graph-expanded images
        pass
    elif current_sim < umbral_imagen:
        continue  # Apply threshold for semantic search images
```

**Recommended**: Option A (marker field) is simpler and more explicit.
