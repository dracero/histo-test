"""
Bug Condition Exploration Test for Graph Image Filtering Fix

**Validates: Requirements 1.1, 1.2, 1.3, 1.4, 2.1, 2.2, 2.3, 2.4, 2.5**

This test explores the bug condition where images retrieved via graph neighborhood
expansion (expandir_vecindad) with similitud=0.3 are incorrectly filtered out by
_nodo_filtrar_contexto() despite being contextually relevant through graph relationships.

CRITICAL: This test is EXPECTED TO FAIL on unfixed code.
- Failure confirms the bug exists (images with similitud=0.3 are removed)
- After implementing the fix (adding origen="vecindad" marker), this test should PASS

Property 1: Bug Condition - Graph-Expanded Images Incorrectly Filtered
For any image result where:
  - tipo == "imagen"
  - similitud < 0.70 (specifically 0.3 from graph expansion)
  - origen == "vecindad" (marker added by fix)
The fixed _nodo_filtrar_contexto function SHALL preserve the image in resultados_validos.
"""

import asyncio
import os
import sys
from typing import Dict, List

# Import the main module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import using importlib to handle the hyphen in filename
import importlib.util
spec = importlib.util.spec_from_file_location("ne4j_histo", "ne4j-histo.py")
ne4j_histo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ne4j_histo)

AsistenteHistologiaNeo4j = ne4j_histo.AsistenteHistologiaNeo4j
AgentState = ne4j_histo.AgentState


async def test_bug_condition_graph_images_filtered():
    """
    Test that images from graph expansion with similitud=0.3 are preserved.
    
    This test simulates the bug condition:
    1. Create a state with an image that has similitud=0.3
    2. Mark it with origen="vecindad" (as the fix will do)
    3. Run _nodo_filtrar_contexto()
    4. Assert the image is preserved in resultados_validos
    
    EXPECTED BEHAVIOR (after fix):
    - Images with origen="vecindad" should bypass similarity threshold
    - Image should be in resultados_validos regardless of similitud=0.3
    
    CURRENT BEHAVIOR (unfixed code):
    - Images with similitud < 0.70 are filtered out
    - Test will FAIL because origen marker is not checked
    """
    print("\n" + "="*70)
    print("TEST: Bug Condition - Graph-Expanded Images Incorrectly Filtered")
    print("="*70)
    
    # Create a minimal AsistenteHistologiaNeo4j instance
    # We only need the _nodo_filtrar_contexto method
    asistente = AsistenteHistologiaNeo4j()
    
    # Create a dummy valid image path for testing
    # Use one of the existing images in the project
    test_image_path = "imagenes_extraidas/arch2_pag1.png"
    
    if not os.path.exists(test_image_path):
        print(f"⚠️  Test image not found: {test_image_path}")
        print("   Creating a dummy file for testing...")
        # If the image doesn't exist, we'll skip the file existence check
        # by using a different approach
        test_image_path = None
    
    # Setup: Create state with graph-expanded image (similitud=0.3)
    # This simulates an image retrieved by expandir_vecindad() with:
    # - similitud=0.3 (standard score for graph neighbors)
    # - origen="vecindad" (marker that the fix will add)
    state: AgentState = {
        "resultados_busqueda": [
            {
                "id": "img_test_001",
                "tipo": "imagen",
                "similitud": 0.3,  # Below threshold (0.70)
                "origen": "vecindad",  # Marker added by fix
                "imagen_path": test_image_path or "/tmp/test_image.png",
                "texto": "Tejido muscular estriado voluntario",
                "fuente": "arch2.pdf",
                "nombre_archivo": "test_image.png",
                "etiqueta": "Imagen 13.1"
            }
        ],
        "tiene_imagen": False,
        "trayectoria": []
    }
    
    print("\n📋 Test Setup:")
    print(f"   - Image ID: {state['resultados_busqueda'][0]['id']}")
    print(f"   - Similitud: {state['resultados_busqueda'][0]['similitud']}")
    print(f"   - Origen: {state['resultados_busqueda'][0].get('origen', 'NOT SET')}")
    print(f"   - Image Path: {state['resultados_busqueda'][0]['imagen_path']}")
    
    # If we don't have a real image, create a temporary one
    temp_file_created = False
    if test_image_path is None:
        import tempfile
        from PIL import Image
        # Create a minimal valid image file
        temp_fd, test_image_path = tempfile.mkstemp(suffix='.png')
        os.close(temp_fd)
        img = Image.new('RGB', (100, 100), color='white')
        img.save(test_image_path)
        state['resultados_busqueda'][0]['imagen_path'] = test_image_path
        temp_file_created = True
        print(f"   ✓ Created temporary test image: {test_image_path}")
    
    try:
        # Execute the filtering function
        print("\n🔍 Executing _nodo_filtrar_contexto()...")
        result = await asistente._nodo_filtrar_contexto(state)
        
        # Check results
        print("\n📊 Results:")
        print(f"   - resultados_validos count: {len(result['resultados_validos'])}")
        print(f"   - contexto_suficiente: {result['contexto_suficiente']}")
        
        if result['resultados_validos']:
            for i, r in enumerate(result['resultados_validos']):
                print(f"   - Valid result {i+1}:")
                print(f"     - ID: {r['id']}")
                print(f"     - Similitud: {r['similitud']}")
                print(f"     - Origen: {r.get('origen', 'NOT SET')}")
        
        # Assertions
        print("\n✅ Assertions:")
        
        # Assert 1: Image should be in resultados_validos
        assert len(result["resultados_validos"]) == 1, (
            f"Expected 1 valid result, got {len(result['resultados_validos'])}. "
            "Graph-expanded images should be preserved regardless of similarity score."
        )
        print("   ✓ Image is in resultados_validos")
        
        # Assert 2: The preserved image should be our test image
        assert result["resultados_validos"][0]["id"] == "img_test_001", (
            f"Expected image ID 'img_test_001', got '{result['resultados_validos'][0]['id']}'"
        )
        print("   ✓ Correct image was preserved")
        
        # Assert 3: Similitud should still be 0.3
        assert result["resultados_validos"][0]["similitud"] == 0.3, (
            f"Expected similitud 0.3, got {result['resultados_validos'][0]['similitud']}"
        )
        print("   ✓ Similitud value preserved correctly")
        
        # Assert 4: Origen marker should be present
        assert result["resultados_validos"][0].get("origen") == "vecindad", (
            f"Expected origen='vecindad', got '{result['resultados_validos'][0].get('origen')}'"
        )
        print("   ✓ Origen marker is present")
        
        print("\n" + "="*70)
        print("✅ TEST PASSED: Graph-expanded images are correctly preserved!")
        print("="*70)
        print("\nThis means the fix is working correctly:")
        print("- Images with origen='vecindad' bypass similarity threshold")
        print("- Graph connectivity guarantees contextual relevance")
        print("- Images with similitud=0.3 from graph expansion are preserved")
        
        return True
        
    except AssertionError as e:
        print("\n" + "="*70)
        print("❌ TEST FAILED (EXPECTED on unfixed code)")
        print("="*70)
        print(f"\nAssertion Error: {e}")
        print("\n📝 Bug Confirmed:")
        print("   The current code filters out images with similitud < 0.70")
        print("   even when they come from graph neighborhood expansion.")
        print("\n🔍 Root Cause:")
        print("   1. expandir_vecindad() does not add 'origen' marker")
        print("   2. _nodo_filtrar_contexto() applies blanket similarity threshold")
        print("   3. Graph-expanded images (similitud=0.3) are incorrectly removed")
        print("\n💡 Expected Fix:")
        print("   1. Add 'origen: \"vecindad\"' in expandir_vecindad() Cypher query")
        print("   2. Modify _nodo_filtrar_contexto() to check origen field")
        print("   3. Skip similarity threshold for images with origen='vecindad'")
        print("\n" + "="*70)
        
        return False
        
    finally:
        # Cleanup temporary file if created
        if temp_file_created and os.path.exists(test_image_path):
            os.unlink(test_image_path)
            print(f"\n🧹 Cleaned up temporary test image")


async def test_same_page_images_preserved():
    """
    Test that images with similitud=0.95 (same page) are preserved.
    
    This test verifies that the existing behavior for same-page images
    continues to work correctly (they already pass the threshold).
    
    EXPECTED: This test should PASS on both unfixed and fixed code.
    """
    print("\n" + "="*70)
    print("TEST: Same Page Images Preserved (Control Test)")
    print("="*70)
    
    asistente = AsistenteHistologiaNeo4j()
    
    # Create a temporary test image
    import tempfile
    from PIL import Image
    temp_fd, test_image_path = tempfile.mkstemp(suffix='.png')
    os.close(temp_fd)
    img = Image.new('RGB', (100, 100), color='white')
    img.save(test_image_path)
    
    try:
        state: AgentState = {
            "resultados_busqueda": [
                {
                    "id": "img_test_002",
                    "tipo": "imagen",
                    "similitud": 0.95,  # Above threshold (same page)
                    # No origen field - semantic search or same-page expansion
                    "imagen_path": test_image_path,
                    "texto": "Imagen de la misma página",
                    "fuente": "arch2.pdf"
                }
            ],
            "tiene_imagen": False,
            "trayectoria": []
        }
        
        print("\n📋 Test Setup:")
        print(f"   - Image ID: {state['resultados_busqueda'][0]['id']}")
        print(f"   - Similitud: {state['resultados_busqueda'][0]['similitud']}")
        print(f"   - Origen: {state['resultados_busqueda'][0].get('origen', 'NOT SET')}")
        
        result = await asistente._nodo_filtrar_contexto(state)
        
        print("\n📊 Results:")
        print(f"   - resultados_validos count: {len(result['resultados_validos'])}")
        
        assert len(result["resultados_validos"]) == 1, (
            f"Expected 1 valid result, got {len(result['resultados_validos'])}"
        )
        assert result["resultados_validos"][0]["similitud"] == 0.95
        
        print("\n✅ TEST PASSED: Same-page images are preserved (existing behavior)")
        return True
        
    finally:
        if os.path.exists(test_image_path):
            os.unlink(test_image_path)


async def main():
    """Run all bug condition exploration tests."""
    print("\n" + "="*70)
    print("BUG CONDITION EXPLORATION TEST SUITE")
    print("Graph Image Filtering Fix")
    print("="*70)
    print("\nThese tests explore the bug on UNFIXED code.")
    print("Test 1 is EXPECTED TO FAIL (confirms bug exists)")
    print("Test 2 should PASS (control test for existing behavior)")
    print("="*70)
    
    results = []
    
    # Test 1: Bug condition (EXPECTED TO FAIL on unfixed code)
    try:
        result1 = await test_bug_condition_graph_images_filtered()
        results.append(("Bug Condition Test", result1))
    except Exception as e:
        print(f"\n❌ Test 1 crashed with exception: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Bug Condition Test", False))
    
    # Test 2: Control test (should PASS)
    try:
        result2 = await test_same_page_images_preserved()
        results.append(("Same Page Test", result2))
    except Exception as e:
        print(f"\n❌ Test 2 crashed with exception: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Same Page Test", False))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUITE SUMMARY")
    print("="*70)
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status}: {test_name}")
    
    print("\n📝 Expected Results on UNFIXED code:")
    print("   - Bug Condition Test: FAIL (confirms bug exists)")
    print("   - Same Page Test: PASS (existing behavior works)")
    
    print("\n📝 Expected Results AFTER implementing fix:")
    print("   - Bug Condition Test: PASS (bug is fixed)")
    print("   - Same Page Test: PASS (no regression)")
    print("="*70)
    
    # Return exit code based on expected behavior
    # On unfixed code, we expect test 1 to fail and test 2 to pass
    bug_test_failed = not results[0][1]  # Bug test should fail
    control_test_passed = results[1][1]  # Control test should pass
    
    if bug_test_failed and control_test_passed:
        print("\n✅ Bug exploration successful: Bug confirmed, control test passed")
        return 0
    else:
        print("\n⚠️  Unexpected test results - review output above")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
