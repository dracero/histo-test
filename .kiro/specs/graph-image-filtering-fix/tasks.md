# Implementation Plan

## Overview

This task list implements the fix for incorrect filtering of graph-expanded images. The workflow follows the exploratory bugfix methodology: first write tests to understand the bug (exploration), then write tests to preserve existing behavior (preservation), then implement the fix with confidence.

---

## Tasks

- [x] 1. Write bug condition exploration test
  - **Property 1: Bug Condition** - Graph-Expanded Images Incorrectly Filtered
  - **CRITICAL**: This test MUST FAIL on unfixed code - failure confirms the bug exists
  - **DO NOT attempt to fix the test or the code when it fails**
  - **NOTE**: This test encodes the expected behavior - it will validate the fix when it passes after implementation
  - **GOAL**: Surface counterexamples that demonstrate the bug exists
  - **Scoped PBT Approach**: Scope the property to concrete failing case: query "tejido muscular" retrieves images with similitud=0.3 from graph expansion
  - Test that images retrieved by `expandir_vecindad()` with similitud=0.3 are preserved in `resultados_validos` after filtering
  - The test assertions should match the Expected Behavior Properties from design:
    - Images with `origen: "vecindad"` and similitud < 0.70 should be in `resultados_validos`
    - Graph connectivity guarantees contextual relevance regardless of similarity score
  - Run test on UNFIXED code
  - **EXPECTED OUTCOME**: Test FAILS (this is correct - it proves the bug exists)
  - Document counterexamples found to understand root cause (e.g., "Image with similitud=0.3 from expandir_vecindad() is removed from resultados_validos")
  - Mark task complete when test is written, run, and failure is documented
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 2.1, 2.2, 2.3, 2.4, 2.5_

- [x] 2. Write preservation property tests (BEFORE implementing fix)
  - **Property 2: Preservation** - Non-Graph Images Filtered Identically
  - **IMPORTANT**: Follow observation-first methodology
  - Observe behavior on UNFIXED code for non-buggy inputs:
    - Semantic search images with similitud < 0.70 are filtered out
    - Text chunks with similitud < threshold are filtered out
    - Images with invalid paths are filtered out regardless of origin
    - Images with similitud > 0.70 are preserved
  - Write property-based tests capturing observed behavior patterns from Preservation Requirements:
    - Test 1: Semantic search images (no `origen` field) with similitud < 0.70 are filtered out
    - Test 2: Text chunks with similitud < threshold are filtered out
    - Test 3: Images with invalid paths are filtered out (even if they had `origen: "vecindad"`)
    - Test 4: Images with similitud > 0.70 are preserved
  - Property-based testing generates many test cases for stronger guarantees
  - Run tests on UNFIXED code
  - **EXPECTED OUTCOME**: Tests PASS (this confirms baseline behavior to preserve)
  - Mark task complete when tests are written, run, and passing on unfixed code
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_

- [x] 3. Fix for graph image filtering bug

  - [x] 3.1 Add `origen: "vecindad"` marker in `expandir_vecindad()` query
    - Modify the Cypher query RETURN clause in `expandir_vecindad()` method (line ~760 in `ne4j-histo.py`)
    - Add `'vecindad' AS origen` to the RETURN statement
    - This marks all results from graph expansion with a distinctive field
    - Verify the marker flows through to `resultados_busqueda` (no changes needed in `busqueda_hibrida()` - `agregar()` preserves all fields)
    - _Bug_Condition: isBugCondition(input) where input["tipo"] == "imagen" AND input["similitud"] < 0.70 AND input was retrieved by expandir_vecindad() AND NOT input.get("origen") == "vecindad"_
    - _Expected_Behavior: Images from graph expansion are preserved in resultados_validos regardless of similitud score because graph connectivity guarantees contextual relevance_
    - _Preservation: Semantic search images, text chunks, and invalid images are filtered identically before and after fix_
    - _Requirements: 2.1, 2.2, 2.5_

  - [x] 3.2 Modify `_nodo_filtrar_contexto()` to skip similarity filtering for marked images
    - Modify the image filtering logic in `_nodo_filtrar_contexto()` method (line ~2695 in `ne4j-histo.py`)
    - Add check: `es_vecindad = r.get("origen") == "vecindad"`
    - Skip similarity threshold check if `es_vecindad` is True
    - Ensure invalid path check still applies to ALL images (preservation requirement)
    - Text filtering logic remains unchanged
    - _Bug_Condition: isBugCondition(input) where input["tipo"] == "imagen" AND input["similitud"] < 0.70 AND input.get("origen") == "vecindad"_
    - _Expected_Behavior: Graph-expanded images bypass similarity threshold and are preserved in resultados_validos_
    - _Preservation: Non-graph images (semantic search, invalid paths) are filtered using existing threshold logic_
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2, 3.3, 3.4, 3.5_

  - [x] 3.3 Verify bug condition exploration test now passes
    - **Property 1: Expected Behavior** - Graph-Expanded Images Preserved
    - **IMPORTANT**: Re-run the SAME test from task 1 - do NOT write a new test
    - The test from task 1 encodes the expected behavior
    - When this test passes, it confirms the expected behavior is satisfied
    - Run bug condition exploration test from step 1
    - **EXPECTED OUTCOME**: Test PASSES (confirms bug is fixed)
    - Verify that images with `origen: "vecindad"` and similitud=0.3 are now in `resultados_validos`
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

  - [x] 3.4 Verify preservation tests still pass
    - **Property 2: Preservation** - Non-Graph Images Filtered Identically
    - **IMPORTANT**: Re-run the SAME tests from task 2 - do NOT write new tests
    - Run preservation property tests from step 2
    - **EXPECTED OUTCOME**: Tests PASS (confirms no regressions)
    - Confirm all preservation tests still pass after fix:
      - Semantic search images with similitud < 0.70 are still filtered out
      - Text chunks with similitud < threshold are still filtered out
      - Images with invalid paths are still filtered out (even graph-expanded ones)
      - Images with similitud > 0.70 are still preserved
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_

- [x] 4. Checkpoint - Ensure all tests pass
  - Run all tests (bug condition + preservation)
  - Verify no regressions in existing functionality
  - If any tests fail, investigate and fix before proceeding
  - Ask the user if questions arise

---

## Notes

- The `origen: "vecindad"` marker is added in the Cypher query RETURN clause, not in Python code
- The marker automatically flows through `busqueda_hibrida()` via the `agregar()` function
- Invalid path checking applies to ALL images, regardless of `origen` field
- Text chunk filtering logic remains completely unchanged
- This fix only affects images retrieved via `expandir_vecindad()` - semantic search images are unaffected
