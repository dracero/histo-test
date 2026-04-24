# Implementation Plan

- [ ] 1. Write bug condition exploration test
  - **Property 1: Bug Condition** - Images Crowded Out by Text Chunks in Expansion 1
  - **CRITICAL**: This test MUST FAIL on unfixed code - failure confirms the bug exists
  - **DO NOT attempt to fix the test or the code when it fails**
  - **NOTE**: This test encodes the expected behavior - it will validate the fix when it passes after implementation
  - **GOAL**: Surface counterexamples that demonstrate images are missing from Expansion 1 results
  - **Scoped PBT Approach**: Scope the property to a concrete case: call `expandir_vecindad()` with a node ID from a PDF that has both text chunks and images (e.g., `arch2.pdf`), and assert that images appear in the results
  - Since testing Cypher queries requires a live Neo4j connection, test at the Python level by mocking `self.run()` to return simulated query results that reflect the buggy behavior
  - Mock approach: patch `Neo4jManager.run()` to simulate the Cypher query returning only text chunks (0 images) when a PDF has both types — this mirrors the bug condition where `collect(DISTINCT vecino_pdf)[..5]` fills with text chunks
  - Assert that when the mock returns results with `tipo='imagen'` count > 0, the function includes images (will fail on unfixed code because the Cypher query itself excludes them)
  - Alternative: if Neo4j is available, run the actual query against the database and count image results
  - Test that `expandir_vecindad(["arch2_pag5_chunk1"])` returns at least 1 result with `tipo='imagen'` (Bug Condition: `isBugCondition(input)` where `text_chunks.length > 0 AND images.length > 0 AND text_chunks.length + images.length > 5`)
  - Run test on UNFIXED code
  - **EXPECTED OUTCOME**: Test FAILS (this is correct - it proves images are crowded out)
  - Document counterexamples found (e.g., "expandir_vecindad returns 0 images despite PDF having 20 images")
  - Mark task complete when test is written, run, and failure is documented
  - _Requirements: 1.1, 1.2, 1.3, 2.1, 2.2, 2.3_

- [ ] 2. Write preservation property tests (BEFORE implementing fix)
  - **Property 2: Preservation** - Expansions 2-4 and Scoring Unchanged
  - **IMPORTANT**: Follow observation-first methodology
  - Since this bug only affects the Cypher query string inside `expandir_vecindad()`, preservation tests verify the query structure for Expansions 2-4 is unchanged
  - Test approach: extract the Cypher query string from the method and verify structural properties
  - Observe: Expansion 2 (`list_ent`) collects `vecino_entidad:Chunk` via `:MENCIONA` with limit 5
  - Observe: Expansion 3 (`list_sim`) collects `vecino_similar:Imagen` via `:SIMILAR_A` with limit 5
  - Observe: Expansion 4 (`list_pag`) collects `img_pag:Imagen` via `:EN_PAGINA` with limit 5
  - Observe: Final `LIMIT 15` is present
  - Observe: Duplicate filtering `NOT v.id IN ids_originales` is present
  - Observe: Similarity scoring assigns 0.95 for same-page/same-source and 0.3 for others
  - Write property-based tests: for any valid node_ids input, the query structure preserves Expansions 2-4 patterns, LIMIT 15, dedup filter, and scoring logic
  - If Neo4j is available, also run actual queries and verify Expansions 2-4 produce identical results before and after
  - Verify tests pass on UNFIXED code
  - **EXPECTED OUTCOME**: Tests PASS (this confirms baseline behavior to preserve)
  - Mark task complete when tests are written, run, and passing on unfixed code
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6_

- [x] 3. Fix for images crowded out by text chunks in Expansion 1

  - [x] 3.1 Split Expansion 1 into two OPTIONAL MATCH clauses
    - In `ne4j-histo.py`, method `expandir_vecindad()` (~line 706), modify the Cypher query string
    - Replace the single Expansion 1 OPTIONAL MATCH that collects all PDF neighbors into `list_pdf[..5]` with two separate clauses:
      - First: `OPTIONAL MATCH (n)-[:PERTENECE_A]->(pdf:PDF)<-[:PERTENECE_A]-(vecino_pdf)` with `WHERE vecino_pdf.id <> nid AND NOT vecino_pdf:Imagen` → `collect(DISTINCT vecino_pdf)[..3] AS list_pdf_chunks`
      - Second: `OPTIONAL MATCH (n)-[:PERTENECE_A]->(pdf2:PDF)<-[:PERTENECE_A]-(img_pdf:Imagen)` with `WHERE img_pdf.id <> nid` → `collect(DISTINCT img_pdf)[..5] AS list_pdf_images`
    - _Bug_Condition: isBugCondition(input) where text_chunks.length > 0 AND images.length > 0 AND total > 5_
    - _Expected_Behavior: text chunks collected separately (limit 3) and images collected separately (limit 5)_
    - _Preservation: Expansions 2-4 unchanged_
    - _Requirements: 1.1, 1.2, 1.3, 2.1, 2.2, 2.3_

  - [x] 3.2 Update WITH clauses and vecinos_raw aggregation
    - Thread `list_pdf_chunks` and `list_pdf_images` through all subsequent WITH clauses (replacing `list_pdf`)
    - Update the `vecinos_raw` concatenation from `list_pdf + list_ent + list_sim + list_pag` to `list_pdf_chunks + list_pdf_images + list_ent + list_sim + list_pag`
    - Ensure all WITH clauses between Expansion 1 and the final aggregation carry both new variables
    - _Requirements: 2.1, 2.2, 2.3, 2.4_

  - [x] 3.3 Verify bug condition exploration test now passes
    - **Property 1: Expected Behavior** - Images Collected Separately with Dedicated Limit
    - **IMPORTANT**: Re-run the SAME test from task 1 - do NOT write a new test
    - The test from task 1 encodes the expected behavior (images appear in results)
    - When this test passes, it confirms images are no longer crowded out by text chunks
    - Run bug condition exploration test from step 1
    - **EXPECTED OUTCOME**: Test PASSES (confirms bug is fixed)
    - _Requirements: 2.1, 2.2, 2.3, 2.4_

  - [x] 3.4 Verify preservation tests still pass
    - **Property 2: Preservation** - Expansions 2-4 and Scoring Unchanged
    - **IMPORTANT**: Re-run the SAME tests from task 2 - do NOT write new tests
    - Run preservation property tests from step 2
    - **EXPECTED OUTCOME**: Tests PASS (confirms no regressions)
    - Confirm Expansions 2-4, LIMIT 15, dedup, and scoring are all unchanged
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6_

- [ ] 4. Checkpoint - Ensure all tests pass
  - Run all tests (bug condition + preservation) together
  - Verify bug condition test passes (images now appear in Expansion 1 results)
  - Verify preservation tests pass (Expansions 2-4 unchanged)
  - If Neo4j is available, run a live query for "neuroglia" and verify both text chunks and images appear in results
  - Ensure all tests pass, ask the user if questions arise.
