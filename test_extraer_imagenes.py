#!/usr/bin/env python3
"""
Script de prueba para verificar la extracción de imágenes.
"""
import os
import sys

# Mock the Neo4jClient class with just the method we need
class MockNeo4jClient:
    def extraer_imagenes_de_resultados(self, resultados, top_k=5):
        """
        Extrae y valida imágenes desde resultados de búsqueda híbrida.
        """
        # Logging inicial
        print(f"   📋 Total de resultados: {len(resultados)}")
        
        # Paso 1: Filtrar solo imágenes
        imagenes = [r for r in resultados if r.get("tipo") == "imagen"]
        print(f"   🖼️ Resultados de tipo imagen: {len(imagenes)}")
        
        if not imagenes:
            print("   ⚠️ No hay imágenes en los resultados de búsqueda")
            return []
        
        # Paso 2: Validar integridad y transformar formato
        imagenes_validas = []
        
        for img in imagenes:
            # Validar path (puede estar como 'imagen_path' o 'path')
            img_path = img.get("imagen_path") or img.get("path")
            if not img_path:
                print(f"   ⚠️ Imagen {img.get('id')} sin path, omitida")
                continue
            
            # Validar existencia en disco
            if not os.path.exists(img_path):
                print(f"   ⚠️ Archivo no existe: {img_path}")
                continue
            
            # Validar/asignar nombre_archivo
            nombre_archivo = img.get("nombre_archivo", "")
            if not nombre_archivo:
                nombre_archivo = os.path.basename(img_path)
            
            # Transformar formato
            imagen_transformada = {
                "id": img.get("id", ""),
                "path": img_path,
                "caption": img.get("texto", ""),  # Renombrar texto → caption
                "nombre_archivo": nombre_archivo,
                "etiqueta": img.get("etiqueta", ""),
                "fuente": img.get("fuente", ""),
                "similitud_semantica": img.get("similitud", 0.0),  # Renombrar similitud → similitud_semantica
            }
            
            # Validar propiedades requeridas
            propiedades_requeridas = [
                "id", "path", "caption", "nombre_archivo", 
                "etiqueta", "fuente", "similitud_semantica"
            ]
            
            if all(prop in imagen_transformada for prop in propiedades_requeridas):
                imagenes_validas.append(imagen_transformada)
            else:
                faltantes = [p for p in propiedades_requeridas if p not in imagen_transformada]
                print(f"   ⚠️ Imagen {img.get('id')} con propiedades faltantes: {faltantes}")
        
        print(f"   ✅ Imágenes válidas: {len(imagenes_validas)}")
        
        # Paso 3: Limitar a top-K
        imagenes_finales = imagenes_validas[:top_k]
        
        # Logging de resultados finales
        if imagenes_finales:
            print(f"   📷 Top-{len(imagenes_finales)} imágenes:")
            for img in imagenes_finales[:3]:
                print(f"      {img['nombre_archivo']} | sim={img['similitud_semantica']:.3f} | {img['fuente']}")
        
        return imagenes_finales


def test_extraccion_basica():
    """Test básico de extracción de imágenes."""
    print("\n=== Test 1: Extracción básica ===")
    
    client = MockNeo4jClient()
    
    # Simular resultados con imágenes reales del directorio
    resultados = [
        {
            "id": "img1",
            "tipo": "imagen",
            "imagen_path": "imagenes_extraidas/arch2_pag1.png",
            "texto": "Tejido epitelial simple",
            "nombre_archivo": "arch2_pag1.png",
            "etiqueta": "Epitelio",
            "fuente": "arch2.pdf",
            "similitud": 0.85
        },
        {
            "id": "chunk1",
            "tipo": "texto",
            "texto": "El tejido epitelial es...",
            "fuente": "arch2.pdf",
            "similitud": 0.75
        },
        {
            "id": "img2",
            "tipo": "imagen",
            "imagen_path": "imagenes_extraidas/arch2_pag2.png",
            "texto": "Tejido conectivo",
            "nombre_archivo": "arch2_pag2.png",
            "etiqueta": "Conectivo",
            "fuente": "arch2.pdf",
            "similitud": 0.80
        }
    ]
    
    imagenes = client.extraer_imagenes_de_resultados(resultados, top_k=3)
    
    print(f"\n✅ Resultado: {len(imagenes)} imágenes extraídas")
    assert len(imagenes) == 2, f"Esperaba 2 imágenes, obtuve {len(imagenes)}"
    assert imagenes[0]["caption"] == "Tejido epitelial simple"
    assert imagenes[0]["similitud_semantica"] == 0.85
    print("✅ Test 1 pasado!")


def test_sin_imagenes():
    """Test con resultados sin imágenes."""
    print("\n=== Test 2: Sin imágenes ===")
    
    client = MockNeo4jClient()
    
    resultados = [
        {
            "id": "chunk1",
            "tipo": "texto",
            "texto": "El tejido epitelial es...",
            "fuente": "arch2.pdf",
            "similitud": 0.75
        }
    ]
    
    imagenes = client.extraer_imagenes_de_resultados(resultados, top_k=3)
    
    print(f"\n✅ Resultado: {len(imagenes)} imágenes extraídas")
    assert len(imagenes) == 0, f"Esperaba 0 imágenes, obtuve {len(imagenes)}"
    print("✅ Test 2 pasado!")


def test_path_no_existe():
    """Test con path que no existe."""
    print("\n=== Test 3: Path no existe ===")
    
    client = MockNeo4jClient()
    
    resultados = [
        {
            "id": "img1",
            "tipo": "imagen",
            "imagen_path": "imagenes_extraidas/no_existe.png",
            "texto": "Imagen inexistente",
            "nombre_archivo": "no_existe.png",
            "etiqueta": "Test",
            "fuente": "arch2.pdf",
            "similitud": 0.85
        }
    ]
    
    imagenes = client.extraer_imagenes_de_resultados(resultados, top_k=3)
    
    print(f"\n✅ Resultado: {len(imagenes)} imágenes extraídas")
    assert len(imagenes) == 0, f"Esperaba 0 imágenes, obtuve {len(imagenes)}"
    print("✅ Test 3 pasado!")


if __name__ == "__main__":
    print("🧪 Ejecutando tests de extracción de imágenes...")
    
    try:
        test_extraccion_basica()
        test_sin_imagenes()
        test_path_no_existe()
        
        print("\n" + "="*50)
        print("✅ TODOS LOS TESTS PASARON!")
        print("="*50)
    except AssertionError as e:
        print(f"\n❌ Test falló: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
