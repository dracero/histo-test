# Bugfix Requirements Document

## Introduction

This bugfix addresses a critical issue in the `expandir_vecindad()` method where images from the same PDF are not being returned in neighborhood expansion results. When users query topics like "neuroglia", the system retrieves text chunks but no images, even though relevant images exist in the same PDF. This occurs because the current implementation retrieves ANY nodes (chunks OR images) with a single limit of 5 results, and text chunks fill this limit before images can be included.

The bug prevents users from seeing relevant images that exist in the database, effectively breaking the image extraction feature since no images reach `resultados_busqueda`. This undermines the graph connectivity that was designed to surface related visual content.

## Bug Analysis

### Current Behavior (Defect)

1.1 WHEN `expandir_vecindad()` executes Expansion 1 to retrieve nodes from the same PDF THEN the system collects both text chunks and images into a single list limited to 5 results

1.2 WHEN the same PDF contains many text chunks (which is common) THEN the system returns only text chunks because they fill the 5-result limit first

1.3 WHEN images exist in the same PDF as the query results THEN the system fails to include any images in the neighborhood expansion results

1.4 WHEN a user queries "neuroglia" THEN the system returns 15 text chunks and 0 images, despite images existing in arch2.pdf

### Expected Behavior (Correct)

2.1 WHEN `expandir_vecindad()` executes Expansion 1 to retrieve nodes from the same PDF THEN the system SHALL collect text chunks and images into separate lists with independent limits

2.2 WHEN the same PDF contains many text chunks THEN the system SHALL still retrieve images because they have a separate limit

2.3 WHEN images exist in the same PDF as the query results THEN the system SHALL include those images in the neighborhood expansion results

2.4 WHEN a user queries "neuroglia" THEN the system SHALL return both text chunks and images from arch2.pdf

### Unchanged Behavior (Regression Prevention)

3.1 WHEN `expandir_vecindad()` executes Expansion 2 (chunks sharing entities) THEN the system SHALL CONTINUE TO retrieve chunks that share entities with the original nodes

3.2 WHEN `expandir_vecindad()` executes Expansion 3 (similar images by embedding) THEN the system SHALL CONTINUE TO retrieve images similar by embedding

3.3 WHEN `expandir_vecindad()` executes Expansion 4 (images from the same page) THEN the system SHALL CONTINUE TO retrieve images from the same page

3.4 WHEN the total result limit of 15 is applied THEN the system SHALL CONTINUE TO limit the final results to 15 items

3.5 WHEN duplicate nodes are filtered THEN the system SHALL CONTINUE TO exclude nodes that match the original query node IDs

3.6 WHEN similarity scores are assigned THEN the system SHALL CONTINUE TO assign 0.95 for same-page/same-source matches and 0.3 for other neighborhood matches
