"""
Preservation Property Tests for Graph Image Filtering Fix

**Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5**

These tests capture the EXISTING filtering behavior that must be preserved after the fix.
They run on UNFIXED code to document baseline behavior.

EXPECTED OUTCOME: All tests PASS (confirms what behavior to preserve)

Property 2: Preservation - Non-Graph Images Filtered Identically
For any image/text result that does NOT originate from expandir_vecindad:
  - Semantic search images with similitud < 0.70 are filtered out
  - Text chunks with similitud < threshold are filtered out
  - Images with invalid paths are filtered out (even if they had origen="vecindad")
  - Images with similitud > 0.70 are preserved

The fixed _nodo_filtrar_contexto function SHALL produce exactly the same filtering
decision as the original function for these non-graph-expanded results.
"""

import asyncio
import os
import sys
import tempfile
from typing import Dict, List
from PIL import Image

# Import hypothesis for property-based testing
try:
    from hypothesis import given, strategies as st, settings, assume
    from hypothesis.strategies import composite
except ImportError:
    print("ERROR: hypothesis library not found. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "hypothesis"])
    from hypothesis import given, strategies as st, settings, assume
    from hypothesis.strategies import composite

# Import the main module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import using importlib to handle the hyphen in filename
import importlib.util
spec = importlib.util.spec_from_file_location("ne4j_histo", "ne4j-histo.py")
ne4j_histo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ne4j_histo)

AsistenteHistologiaNeo4j = ne4j_histo.AsistenteHistologiaNeo4j
AgentState = ne4j_histo.AgentState


# ============================================================================
# Test Helpers
# ============================================================================

def create_temp_image() -> str:
    """Create a temporary valid image file for testing."""
    temp_fd, temp_path = tempfile.mkstemp(suffix='.png')
    os.close(temp_fd)
    img = Image.new('RGB', (100, 100), color='white')
    img.save(temp_path)
    return temp_path


def cleanup_temp_image(path: str):
    """Remove temporary image file."""
    if path and os.path.exists(path):
        try:
            os.unlink(path)
        except:
            pass


# ============================================================================
# Hypothesis Strategies
# ============================================================================

@composite
def semantic_search_image_below_threshold(draw):
    """
    Generate semantic search images with similitud < 0.70.
    These should be filtered out (preservation requirement).
    """
    similitud = draw(st.floats(min_value=0.0, max_value=0.69, allow_nan=False, allow_infinity=False))
    temp_image = create_temp_image()
    
    return {
        "id": f"img_semantic_{draw(st.integers(min_value=1, max_value=10000))}",
        "tipo": "imagen",
        "similitud": similitud,
        # No origen field - semantic search image
        "imagen_path": temp_image,
        "texto": draw(st.text(min_size=5, max_size=100)),
        "fuente": "arch2.pdf",
        "nombre_archivo": "test_image.png",
        "etiqueta": "Test Image"
    }


@composite
def semantic_search_image_above_threshold(draw):
    """
    Generate semantic search images with similitud >= 0.70.
    These should be preserved (preservation requirement).
    """
    similitud = draw(st.floats(min_value=0.70, max_value=1.0, allow_nan=False, allow_infinity=False))
    temp_image = create_temp_image()
    
    return {
        "id": f"img_semantic_high_{draw(st.integers(min_value=1, max_value=10000))}",
        "tipo": "imagen",
        "similitud": similitud,
        # No origen field - semantic search image
        "imagen_path": temp_image,
        "texto": draw(st.text(min_size=5, max_size=100)),
        "fuente": "arch2.pdf",
        "nombre_archivo": "test_image.png",
        "etiqueta": "Test Image"
    }


@composite
def text_chunk_below_threshold(draw):
    """
    Generate text chunks with similitud below threshold.
    Threshold is 0.45 (solo_texto mode) or 0.6 (multimodal mode).
    """
    # Generate similitud below the lower threshold (0.45)
    similitud = draw(st.floats(min_value=0.0, max_value=0.44, allow_nan=False, allow_infinity=False))
    
    return {
        "id": f"text_{draw(st.integers(min_value=1, max_value=10000))}",
        "tipo": "texto",
        "similitud": similitud,
        "texto": draw(st.text(min_size=10, max_size=500)),
        "fuente": "arch2.pdf"
    }


@composite
def text_chunk_above_threshold(draw):
    """
    Generate text chunks with similitud above threshold.
    Should be preserved in both modes.
    """
    # Generate similitud above the higher threshold (0.6)
    similitud = draw(st.floats(min_value=0.61, max_value=1.0, allow_nan=False, allow_infinity=False))
    
    return {
        "id": f"text_high_{draw(st.integers(min_value=1, max_value=10000))}",
        "tipo": "texto",
        "similitud": similitud,
        "texto": draw(st.text(min_size=10, max_size=500)),
        "fuente": "arch2.pdf"
    }


@composite
def image_with_invalid_path(draw):
    """
    Generate images with invalid paths (null or non-existent).
    These should ALWAYS be filtered out, regardless of origen or similitud.
    """
    similitud = draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
    
    # Choose between null path or non-existent path
    path_type = draw(st.sampled_from(['null', 'nonexistent']))
    
    if path_type == 'null':
        imagen_path = None
    else:
        imagen_path = f"/nonexistent/path/image_{draw(st.integers(min_value=1, max_value=10000))}.png"
    
    result = {
        "id": f"img_invalid_{draw(st.integers(min_value=1, max_value=10000))}",
        "tipo": "imagen",
        "similitud": similitud,
        "imagen_path": imagen_path,
        "texto": draw(st.text(min_size=5, max_size=100)),
        "fuente": "arch2.pdf"
    }
    
    # Sometimes add origen="vecindad" to test that invalid paths are filtered
    # even for graph-expanded images
    if draw(st.booleans()):
        result["origen"] = "vecindad"
    
    return result


# ============================================================================
# Property-Based Tests
# ============================================================================

@given(semantic_search_image_below_threshold())
@settings(max_examples=50, deadline=None)
def test_property_semantic_images_below_threshold_filtered(image_data):
    """
    Property 2.1: Semantic search images with similitud < 0.70 are filtered out.
    
    **Validates: Requirements 3.1**
    
    This test verifies that images from semantic search (no origen field)
    with similitud below the threshold (0.70) are correctly filtered out.
    This behavior must be preserved after the fix.
    
    EXPECTED: Test PASSES on unfixed code (confirms baseline behavior)
    """
    async def run_test():
        asistente = AsistenteHistologiaNeo4j()
        
        state: AgentState = {
            "resultados_busqueda": [image_data],
            "tiene_imagen": False,
            "trayectoria": []
        }
        
        try:
            result = await asistente._nodo_filtrar_contexto(state)
            
            # Assert: Image should be filtered out (not in resultados_validos)
            assert len(result["resultados_validos"]) == 0, (
                f"Expected 0 valid results for semantic image with similitud={image_data['similitud']:.3f} < 0.70, "
                f"got {len(result['resultados_validos'])}. "
                "Semantic search images below threshold should be filtered out."
            )
            
        finally:
            # Cleanup temp image
            cleanup_temp_image(image_data.get("imagen_path"))
    
    asyncio.run(run_test())


@given(semantic_search_image_above_threshold())
@settings(max_examples=50, deadline=None)
def test_property_semantic_images_above_threshold_preserved(image_data):
    """
    Property 2.2: Semantic search images with similitud >= 0.70 are preserved.
    
    **Validates: Requirements 3.1**
    
    This test verifies that images from semantic search with similitud
    above the threshold are correctly preserved. This behavior must be
    preserved after the fix.
    
    EXPECTED: Test PASSES on unfixed code (confirms baseline behavior)
    """
    async def run_test():
        asistente = AsistenteHistologiaNeo4j()
        
        state: AgentState = {
            "resultados_busqueda": [image_data],
            "tiene_imagen": False,
            "trayectoria": []
        }
        
        try:
            result = await asistente._nodo_filtrar_contexto(state)
            
            # Assert: Image should be preserved (in resultados_validos)
            assert len(result["resultados_validos"]) == 1, (
                f"Expected 1 valid result for semantic image with similitud={image_data['similitud']:.3f} >= 0.70, "
                f"got {len(result['resultados_validos'])}. "
                "Semantic search images above threshold should be preserved."
            )
            
            assert result["resultados_validos"][0]["id"] == image_data["id"]
            assert result["resultados_validos"][0]["similitud"] == image_data["similitud"]
            
        finally:
            # Cleanup temp image
            cleanup_temp_image(image_data.get("imagen_path"))
    
    asyncio.run(run_test())


@given(text_chunk_below_threshold())
@settings(max_examples=50, deadline=None)
def test_property_text_chunks_below_threshold_filtered(text_data):
    """
    Property 2.3: Text chunks with similitud < threshold are filtered out.
    
    **Validates: Requirements 3.2**
    
    This test verifies that text chunks with similitud below the threshold
    (0.45 for solo_texto mode, 0.6 for multimodal) are correctly filtered out.
    This behavior must be preserved after the fix.
    
    EXPECTED: Test PASSES on unfixed code (confirms baseline behavior)
    """
    async def run_test():
        asistente = AsistenteHistologiaNeo4j()
        
        # Test in solo_texto mode (threshold = 0.45)
        state: AgentState = {
            "resultados_busqueda": [text_data],
            "tiene_imagen": False,  # solo_texto mode
            "trayectoria": []
        }
        
        result = await asistente._nodo_filtrar_contexto(state)
        
        # Assert: Text should be filtered out (similitud < 0.45)
        assert len(result["resultados_validos"]) == 0, (
            f"Expected 0 valid results for text with similitud={text_data['similitud']:.3f} < 0.45, "
            f"got {len(result['resultados_validos'])}. "
            "Text chunks below threshold should be filtered out."
        )
    
    asyncio.run(run_test())


@given(text_chunk_above_threshold())
@settings(max_examples=50, deadline=None)
def test_property_text_chunks_above_threshold_preserved(text_data):
    """
    Property 2.4: Text chunks with similitud > threshold are preserved.
    
    **Validates: Requirements 3.2**
    
    This test verifies that text chunks with similitud above the threshold
    are correctly preserved. This behavior must be preserved after the fix.
    
    EXPECTED: Test PASSES on unfixed code (confirms baseline behavior)
    """
    async def run_test():
        asistente = AsistenteHistologiaNeo4j()
        
        # Test in multimodal mode (threshold = 0.6)
        state: AgentState = {
            "resultados_busqueda": [text_data],
            "tiene_imagen": True,  # multimodal mode
            "trayectoria": []
        }
        
        result = await asistente._nodo_filtrar_contexto(state)
        
        # Assert: Text should be preserved (similitud > 0.6)
        assert len(result["resultados_validos"]) == 1, (
            f"Expected 1 valid result for text with similitud={text_data['similitud']:.3f} > 0.6, "
            f"got {len(result['resultados_validos'])}. "
            "Text chunks above threshold should be preserved."
        )
        
        assert result["resultados_validos"][0]["id"] == text_data["id"]
        assert result["resultados_validos"][0]["similitud"] == text_data["similitud"]
    
    asyncio.run(run_test())


@given(image_with_invalid_path())
@settings(max_examples=50, deadline=None)
def test_property_invalid_path_images_always_filtered(image_data):
    """
    Property 2.5: Images with invalid paths are ALWAYS filtered out.
    
    **Validates: Requirements 3.3**
    
    This test verifies that images with invalid paths (null or non-existent)
    are ALWAYS filtered out, regardless of:
    - similitud score (even if > 0.70)
    - origen field (even if origen="vecindad")
    
    This is a critical preservation requirement: invalid images must never
    reach the user, even if they come from graph expansion.
    
    EXPECTED: Test PASSES on unfixed code (confirms baseline behavior)
    """
    async def run_test():
        asistente = AsistenteHistologiaNeo4j()
        
        state: AgentState = {
            "resultados_busqueda": [image_data],
            "tiene_imagen": False,
            "trayectoria": []
        }
        
        result = await asistente._nodo_filtrar_contexto(state)
        
        # Assert: Image should be filtered out (invalid path)
        assert len(result["resultados_validos"]) == 0, (
            f"Expected 0 valid results for image with invalid path "
            f"(path={image_data.get('imagen_path')}, similitud={image_data['similitud']:.3f}, "
            f"origen={image_data.get('origen', 'NOT SET')}), "
            f"got {len(result['resultados_validos'])}. "
            "Images with invalid paths should ALWAYS be filtered out."
        )
    
    asyncio.run(run_test())


# ============================================================================
# Unit Tests (Concrete Examples)
# ============================================================================

def test_concrete_semantic_image_filtered():
    """
    Concrete example: Semantic search image with similitud=0.5 is filtered out.
    
    **Validates: Requirements 3.1**
    """
    async def run_test():
        asistente = AsistenteHistologiaNeo4j()
        temp_image = create_temp_image()
        
        try:
            state: AgentState = {
                "resultados_busqueda": [
                    {
                        "id": "img_semantic_001",
                        "tipo": "imagen",
                        "similitud": 0.5,
                        # No origen field
                        "imagen_path": temp_image,
                        "texto": "Test image",
                        "fuente": "arch2.pdf"
                    }
                ],
                "tiene_imagen": False,
                "trayectoria": []
            }
            
            result = await asistente._nodo_filtrar_contexto(state)
            
            assert len(result["resultados_validos"]) == 0, (
                "Semantic search image with similitud=0.5 should be filtered out"
            )
            
            print("✅ Concrete test passed: Semantic image (0.5) filtered out")
            
        finally:
            cleanup_temp_image(temp_image)
    
    asyncio.run(run_test())


def test_concrete_high_similarity_image_preserved():
    """
    Concrete example: Image with similitud=0.85 is preserved.
    
    **Validates: Requirements 3.1**
    """
    async def run_test():
        asistente = AsistenteHistologiaNeo4j()
        temp_image = create_temp_image()
        
        try:
            state: AgentState = {
                "resultados_busqueda": [
                    {
                        "id": "img_high_001",
                        "tipo": "imagen",
                        "similitud": 0.85,
                        "imagen_path": temp_image,
                        "texto": "High similarity image",
                        "fuente": "arch2.pdf"
                    }
                ],
                "tiene_imagen": False,
                "trayectoria": []
            }
            
            result = await asistente._nodo_filtrar_contexto(state)
            
            assert len(result["resultados_validos"]) == 1, (
                "Image with similitud=0.85 should be preserved"
            )
            assert result["resultados_validos"][0]["similitud"] == 0.85
            
            print("✅ Concrete test passed: High similarity image (0.85) preserved")
            
        finally:
            cleanup_temp_image(temp_image)
    
    asyncio.run(run_test())


def test_concrete_text_chunk_filtered():
    """
    Concrete example: Text chunk with similitud=0.4 is filtered out.
    
    **Validates: Requirements 3.2**
    """
    async def run_test():
        asistente = AsistenteHistologiaNeo4j()
        
        state: AgentState = {
            "resultados_busqueda": [
                {
                    "id": "text_001",
                    "tipo": "texto",
                    "similitud": 0.4,
                    "texto": "Low similarity text chunk",
                    "fuente": "arch2.pdf"
                }
            ],
            "tiene_imagen": False,  # solo_texto mode, threshold=0.45
            "trayectoria": []
        }
        
        result = await asistente._nodo_filtrar_contexto(state)
        
        assert len(result["resultados_validos"]) == 0, (
            "Text chunk with similitud=0.4 should be filtered out (threshold=0.45)"
        )
        
        print("✅ Concrete test passed: Text chunk (0.4) filtered out")
    
    asyncio.run(run_test())


def test_concrete_invalid_path_with_vecindad_filtered():
    """
    Concrete example: Image with origen="vecindad" but invalid path is filtered out.
    
    **Validates: Requirements 3.3**
    
    This is a critical test: even graph-expanded images must be filtered
    if their paths are invalid.
    """
    async def run_test():
        asistente = AsistenteHistologiaNeo4j()
        
        state: AgentState = {
            "resultados_busqueda": [
                {
                    "id": "img_invalid_vecindad_001",
                    "tipo": "imagen",
                    "similitud": 0.95,  # High similarity
                    "origen": "vecindad",  # From graph expansion
                    "imagen_path": "/nonexistent/path.png",  # Invalid path
                    "texto": "Graph-expanded image with invalid path",
                    "fuente": "arch2.pdf"
                }
            ],
            "tiene_imagen": False,
            "trayectoria": []
        }
        
        result = await asistente._nodo_filtrar_contexto(state)
        
        assert len(result["resultados_validos"]) == 0, (
            "Image with invalid path should be filtered out even if origen='vecindad'"
        )
        
        print("✅ Concrete test passed: Invalid path with vecindad filtered out")
    
    asyncio.run(run_test())


# ============================================================================
# Test Suite Runner
# ============================================================================

def run_all_tests():
    """Run all preservation property tests."""
    print("\n" + "="*70)
    print("PRESERVATION PROPERTY TEST SUITE")
    print("Graph Image Filtering Fix - Task 2")
    print("="*70)
    print("\nThese tests capture EXISTING behavior that must be preserved.")
    print("All tests should PASS on unfixed code.")
    print("="*70)
    
    print("\n📋 Running concrete unit tests...")
    print("-" * 70)
    
    try:
        test_concrete_semantic_image_filtered()
        test_concrete_high_similarity_image_preserved()
        test_concrete_text_chunk_filtered()
        test_concrete_invalid_path_with_vecindad_filtered()
        print("\n✅ All concrete unit tests passed!")
    except AssertionError as e:
        print(f"\n❌ Concrete test failed: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Concrete test crashed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n📋 Running property-based tests (50 examples each)...")
    print("-" * 70)
    print("This may take a minute...\n")
    
    try:
        # Run property tests
        print("Testing Property 2.1: Semantic images < 0.70 filtered...")
        test_property_semantic_images_below_threshold_filtered()
        print("✅ Property 2.1 passed (50 examples)")
        
        print("\nTesting Property 2.2: Semantic images >= 0.70 preserved...")
        test_property_semantic_images_above_threshold_preserved()
        print("✅ Property 2.2 passed (50 examples)")
        
        print("\nTesting Property 2.3: Text chunks < threshold filtered...")
        test_property_text_chunks_below_threshold_filtered()
        print("✅ Property 2.3 passed (50 examples)")
        
        print("\nTesting Property 2.4: Text chunks > threshold preserved...")
        test_property_text_chunks_above_threshold_preserved()
        print("✅ Property 2.4 passed (50 examples)")
        
        print("\nTesting Property 2.5: Invalid path images always filtered...")
        test_property_invalid_path_images_always_filtered()
        print("✅ Property 2.5 passed (50 examples)")
        
        print("\n" + "="*70)
        print("✅ ALL PRESERVATION TESTS PASSED!")
        print("="*70)
        print("\n📝 Summary:")
        print("   - 4 concrete unit tests passed")
        print("   - 5 property-based tests passed (250 total examples)")
        print("\n✅ Baseline behavior documented successfully!")
        print("   These behaviors must be preserved after implementing the fix.")
        print("="*70)
        
        return True
        
    except AssertionError as e:
        print(f"\n❌ Property test failed: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Property test crashed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
