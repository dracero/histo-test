# Expandir Vecindad Image Priority Bugfix Design

## Overview

The `expandir_vecindad()` method in `ne4j-histo.py` has a bug in Expansion 1 where text chunks and images from the same PDF are collected into a single list (`list_pdf`) with a shared limit of 5. Since text chunks are far more numerous than images in a typical PDF, they fill the limit before any images can be included. The fix splits Expansion 1 into two separate OPTIONAL MATCH clauses: one for text chunks (`list_pdf_chunks`, limit 3) and one for images (`list_pdf_images`, limit 5), then combines both lists in the final aggregation. This ensures images always have dedicated slots in the neighborhood expansion results.

## Glossary

- **Bug_Condition (C)**: The condition where Expansion 1 collects both text chunks and images into a single `list_pdf` with a shared limit of 5, causing images to be crowded out by text chunks
- **Property (P)**: The desired behavior where text chunks and images from the same PDF are collected separately with independent limits (3 for chunks, 5 for images)
- **Preservation**: Expansions 2-4 (entity-sharing chunks, similar images by embedding, same-page images), the final LIMIT 15, duplicate filtering, and similarity scoring must remain unchanged
- **expandir_vecindad()**: The async method in `ne4j-histo.py` (~line 700) that expands neighborhood around query result nodes using a Cypher query with 4 expansion strategies
- **list_pdf**: The current single collection variable that mixes text chunks and images from the same PDF (the buggy variable)
- **list_pdf_chunks**: The new collection for text-only chunks from the same PDF (limit 3)
- **list_pdf_images**: The new collection for images from the same PDF (limit 5)
- **Expansion 1**: The first OPTIONAL MATCH clause that retrieves nodes from the same PDF via `:PERTENECE_A` relationships

## Bug Details

### Bug Condition

The bug manifests when `expandir_vecindad()` executes Expansion 1 to retrieve neighbor nodes from the same PDF. The Cypher query collects ALL nodes (both text chunks and images) connected to the same PDF into a single `list_pdf` variable limited to 5 results. Since Neo4j's `collect()` does not guarantee ordering by node type, and text chunks vastly outnumber images in a typical PDF, text chunks fill the 5-slot limit before any images can be included.

**Formal Specification:**
```
FUNCTION isBugCondition(input)
  INPUT: input of type { node_ids: List[str], graph_state: GraphState }
  OUTPUT: boolean

  FOR EACH nid IN input.node_ids DO
    pdf_neighbors := getAllNodesConnectedToSamePDF(nid)
    text_chunks := filter(pdf_neighbors, node -> NOT node:Imagen AND node.id <> nid)
    images := filter(pdf_neighbors, node -> node:Imagen AND node.id <> nid)

    RETURN text_chunks.length > 0
           AND images.length > 0
           AND text_chunks.length + images.length > 5
           // Bug: images get crowded out because collect()[..5] fills with text chunks first
  END FOR
END FUNCTION
```

### Examples

- **Example 1**: Node `arch2_pag5_chunk1` belongs to `arch2.pdf`. The PDF has 38 text chunks and 20 images. Current query returns 5 text chunks and 0 images in `list_pdf`. Expected: 3 text chunks + up to 5 images.
- **Example 2**: Query "neuroglia" returns 15 text chunks and 0 images. Expected: text chunks from Expansion 1 limited to 3, with up to 5 images from the same PDF also included.
- **Example 3**: A PDF with only 2 text chunks and 10 images. Current: returns 2 chunks + 3 images (fills to 5). Expected: returns 2 chunks (list_pdf_chunks) + 5 images (list_pdf_images).
- **Edge case**: A PDF with 0 images. Current and expected behavior are the same — only text chunks returned (up to 3 in fixed version vs up to 5 in current).

## Expected Behavior

### Preservation Requirements

**Unchanged Behaviors:**
- Expansion 2 (chunks sharing entities via `:MENCIONA`) must continue to retrieve up to 5 entity-sharing chunks in `list_ent`
- Expansion 3 (similar images via `:SIMILAR_A`) must continue to retrieve up to 5 similar images in `list_sim`
- Expansion 4 (images from the same page via `:EN_PAGINA`) must continue to retrieve up to 5 same-page images in `list_pag`
- The final `LIMIT 15` on total results must remain unchanged
- Duplicate filtering (`WHERE NOT v.id IN ids_originales`) must remain unchanged
- Similarity scoring (0.95 for same-page/same-source, 0.3 for other neighbors) must remain unchanged
- The RETURN clause field mapping (id, texto, fuente, tipo, imagen_path, similitud, nombre_archivo, etiqueta, origen) must remain unchanged

**Scope:**
All inputs that do NOT involve Expansion 1 (same-PDF neighbor collection) should be completely unaffected by this fix. This includes:
- Expansion 2 entity-based neighbor retrieval
- Expansion 3 embedding-similarity neighbor retrieval
- Expansion 4 same-page image retrieval
- Final result aggregation, deduplication, and scoring
- The method signature and return type

## Hypothesized Root Cause

Based on the bug description, the most likely issue is:

1. **Single Collection with Shared Limit**: The Cypher query uses a single `OPTIONAL MATCH` for Expansion 1 that collects ALL neighbor nodes (both `:Chunk` and `:Imagen`) into one `list_pdf` variable with `collect(DISTINCT vecino_pdf)[..5]`. There is no type-based filtering or partitioning.

2. **Text Chunk Dominance**: In a typical PDF like `arch2.pdf`, there are far more text chunks (~38) than images (~20). When `collect()` gathers nodes, text chunks are encountered first or more frequently, filling the 5-slot limit before images get a chance.

3. **No Label Filtering in WHERE Clause**: The current WHERE clause only filters by `vecino_pdf.id <> nid` but does not distinguish between `:Chunk` and `:Imagen` labels. Both node types match the pattern `(pdf:PDF)<-[:PERTENECE_A]-(vecino_pdf)`.

4. **Downstream Impact**: Since `list_pdf` feeds into `vecinos_raw` which is the final aggregation, the absence of images from `list_pdf` means Expansion 1 contributes zero images. Images can only enter results through Expansions 3 and 4, which depend on different relationship types (`:SIMILAR_A` and `:EN_PAGINA`).

## Correctness Properties

Property 1: Bug Condition - Images from same PDF are collected separately with dedicated limit

_For any_ input where the bug condition holds (a node belongs to a PDF that contains both text chunks and images), the fixed `expandir_vecindad()` function SHALL collect images into a separate `list_pdf_images` collection with its own limit of 5, ensuring images are not crowded out by text chunks.

**Validates: Requirements 2.1, 2.2, 2.3**

Property 2: Preservation - Non-Expansion-1 behavior unchanged

_For any_ input processed by `expandir_vecindad()`, the fixed function SHALL produce identical results for Expansions 2, 3, and 4 (entity-sharing chunks, similar images, same-page images), and SHALL preserve the final LIMIT 15, duplicate filtering, and similarity scoring logic exactly as the original function.

**Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6**

## Fix Implementation

### Changes Required

Assuming our root cause analysis is correct:

**File**: `ne4j-histo.py`

**Function**: `expandir_vecindad()` (async method, ~line 700)

**Specific Changes**:

1. **Split Expansion 1 into two OPTIONAL MATCH clauses**: Replace the single `OPTIONAL MATCH` that collects all PDF neighbors into `list_pdf` with two separate clauses:
   - First clause: collect text chunks (non-`:Imagen` nodes) into `list_pdf_chunks[..3]`
   - Second clause: collect images (`:Imagen` nodes) into `list_pdf_images[..5]`

2. **Add label filter to first clause**: Add `AND NOT vecino_pdf:Imagen` to the WHERE clause of the first OPTIONAL MATCH to exclude images from the text chunk collection.

3. **Add new OPTIONAL MATCH for images**: Add a second OPTIONAL MATCH that specifically targets `:Imagen` nodes from the same PDF:
   ```
   OPTIONAL MATCH (n)-[:PERTENECE_A]->(pdf2:PDF)<-[:PERTENECE_A]-(img_pdf:Imagen)
   WHERE img_pdf.id <> nid
   ```

4. **Thread new variable through WITH clauses**: Update all subsequent WITH clauses to carry both `list_pdf_chunks` and `list_pdf_images` instead of the single `list_pdf`.

5. **Update final aggregation**: Change the `vecinos_raw` concatenation from:
   ```
   list_pdf + list_ent + list_sim + list_pag
   ```
   to:
   ```
   list_pdf_chunks + list_pdf_images + list_ent + list_sim + list_pag
   ```

## Testing Strategy

### Validation Approach

The testing strategy follows a two-phase approach: first, surface counterexamples that demonstrate the bug on unfixed code, then verify the fix works correctly and preserves existing behavior.

### Exploratory Bug Condition Checking

**Goal**: Surface counterexamples that demonstrate the bug BEFORE implementing the fix. Confirm or refute the root cause analysis. If we refute, we will need to re-hypothesize.

**Test Plan**: Write tests that execute `expandir_vecindad()` against a Neo4j database containing PDFs with both text chunks and images, and verify whether images appear in the results. Run these tests on the UNFIXED code to observe failures and understand the root cause.

**Test Cases**:
1. **Same-PDF Image Retrieval**: Call `expandir_vecindad()` with a node ID from a PDF that has many text chunks and images. Assert images appear in results (will fail on unfixed code).
2. **Image Count Verification**: Call `expandir_vecindad()` and count results with `tipo='imagen'` from Expansion 1. Assert count > 0 when images exist in the same PDF (will fail on unfixed code).
3. **Mixed Content PDF**: Use a PDF with 10+ text chunks and 5+ images. Verify both types appear in results (will fail on unfixed code).
4. **Edge Case - Few Text Chunks**: Use a PDF with only 2 text chunks and 5 images. Verify images still appear (may partially work on unfixed code since limit of 5 has room).

**Expected Counterexamples**:
- Results contain 0 images from Expansion 1 when the PDF has many text chunks
- Possible causes: single `collect()[..5]` fills with text chunks before images

### Fix Checking

**Goal**: Verify that for all inputs where the bug condition holds, the fixed function produces the expected behavior.

**Pseudocode:**
```
FOR ALL input WHERE isBugCondition(input) DO
  result := expandir_vecindad_fixed(input.node_ids)
  images_in_result := filter(result, r -> r.tipo == 'imagen')
  ASSERT images_in_result.length > 0
  ASSERT images from same PDF are present in results
END FOR
```

### Preservation Checking

**Goal**: Verify that for all inputs where the bug condition does NOT hold, the fixed function produces the same result as the original function.

**Pseudocode:**
```
FOR ALL input WHERE NOT isBugCondition(input) DO
  ASSERT expandir_vecindad_original(input) = expandir_vecindad_fixed(input)
END FOR
```

**Testing Approach**: Property-based testing is recommended for preservation checking because:
- It generates many test cases automatically across the input domain
- It catches edge cases that manual unit tests might miss
- It provides strong guarantees that behavior is unchanged for all non-buggy inputs

**Test Plan**: Observe behavior on UNFIXED code first for Expansions 2-4, duplicate filtering, and scoring, then write property-based tests capturing that behavior.

**Test Cases**:
1. **Expansion 2 Preservation**: Verify entity-sharing chunk retrieval (`list_ent`) produces identical results before and after fix
2. **Expansion 3 Preservation**: Verify similar-image retrieval (`list_sim`) produces identical results before and after fix
3. **Expansion 4 Preservation**: Verify same-page image retrieval (`list_pag`) produces identical results before and after fix
4. **Scoring Preservation**: Verify similarity scores (0.95 and 0.3) are assigned identically before and after fix
5. **Limit Preservation**: Verify total results never exceed 15 before and after fix

### Unit Tests

- Test that `list_pdf_chunks` contains only non-`:Imagen` nodes (text chunks)
- Test that `list_pdf_images` contains only `:Imagen` nodes
- Test that `list_pdf_chunks` is limited to 3 results
- Test that `list_pdf_images` is limited to 5 results
- Test edge case: PDF with no images returns empty `list_pdf_images`
- Test edge case: PDF with no text chunks returns empty `list_pdf_chunks`

### Property-Based Tests

- Generate random graph states with varying numbers of text chunks and images per PDF, verify that images always get dedicated slots in results
- Generate random node configurations and verify that Expansions 2-4 produce identical results with both old and new queries
- Generate random inputs and verify the total result count never exceeds 15

### Integration Tests

- Test full query flow: search "neuroglia" → `expandir_vecindad()` → verify both text chunks and images in final `resultados_busqueda`
- Test that image paths are correctly returned for images retrieved via Expansion 1
- Test that the combined `list_pdf_chunks + list_pdf_images` feeds correctly into `vecinos_raw` aggregation
