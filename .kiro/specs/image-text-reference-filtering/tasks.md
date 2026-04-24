# Implementation Plan: Image Extraction from Graph Neighborhood

## Overview

Este plan implementa la extracción simplificada de imágenes desde los resultados de búsqueda híbrida existentes, eliminando la necesidad de búsquedas adicionales. La solución consiste en:

1. Crear una función `extraer_imagenes_de_resultados()` que filtra y valida imágenes de `resultados_busqueda`
2. Modificar `_nodo_buscar_neo4j()` para invocar la nueva función en lugar de `busqueda_imagenes_semantica()`
3. Validar con tests unitarios, property-based tests, e integration tests

**Complejidad**: Baja (~50 líneas de código nuevo)  
**Archivos a modificar**: `ne4j-histo.py`  
**Lenguaje**: Python

## Tasks

- [x] 1. Implementar función de extracción de imágenes
  - [x] 1.1 Crear método `extraer_imagenes_de_resultados()` en clase `Neo4jClient`
    - Implementar filtrado por tipo "imagen"
    - Implementar validación de path (existencia en disco)
    - Implementar renombrado de propiedades (`texto` → `caption`, `similitud` → `similitud_semantica`)
    - Implementar fallback de `nombre_archivo` usando `os.path.basename(path)`
    - Implementar límite top-K con preservación de orden
    - Implementar logging detallado en cada paso
    - _Requirements: 1.1, 1.2, 1.5, 2.1, 2.2, 2.4, 2.5, 3.1, 3.2, 4.1, 4.2, 5.1, 5.2, 5.3, 7.1, 7.2, 7.3_
  
  - [ ]* 1.2 Escribir property test para filtrado correcto por tipo
    - **Property 1: Filtrado Correcto por Tipo**
    - **Validates: Requirements 1.1**
    - Usar generator `lista_resultados_busqueda()` con tipos mixtos
    - Verificar que solo resultados con `tipo == "imagen"` son retornados
  
  - [ ]* 1.3 Escribir property test para validación de path obligatoria
    - **Property 2: Validación de Path Obligatoria**
    - **Validates: Requirements 2.1, 2.2, 2.3**
    - Usar generator con paths nulos y no existentes
    - Verificar que imágenes sin path válido son omitidas
  
  - [ ]* 1.4 Escribir property test para renombrado de propiedades
    - **Property 3: Renombrado de Propiedades Consistente**
    - **Validates: Requirements 3.1, 3.2, 4.1, 4.2**
    - Verificar transformación `texto` → `caption` y `similitud` → `similitud_semantica`
    - Verificar preservación exacta de valores
  
  - [ ]* 1.5 Escribir property test para fallback de nombre de archivo
    - **Property 4: Fallback de Nombre de Archivo**
    - **Validates: Requirements 2.4, 2.5**
    - Usar generator con `nombre_archivo` vacío o nulo
    - Verificar uso de `os.path.basename(path)` como fallback

- [x] 2. Implementar validación y límites
  - [x] 2.1 Implementar validación de propiedades requeridas
    - Verificar que cada imagen tiene: `id`, `path`, `caption`, `nombre_archivo`, `etiqueta`, `fuente`, `similitud_semantica`
    - Omitir imágenes con propiedades faltantes y registrar advertencia
    - _Requirements: 2.6, 10.2, 10.3, 10.4, 10.5_
  
  - [ ]* 2.2 Escribir property test para límite top-K
    - **Property 5: Límite Top-K Respetado**
    - **Validates: Requirements 5.1, 5.2, 5.3, 5.4**
    - Usar generators para lista de imágenes y valores de top_k
    - Verificar que longitud de salida <= top_k
    - Verificar preservación de orden original
  
  - [ ]* 2.3 Escribir property test para formato de salida completo
    - **Property 6: Formato de Salida Completo**
    - **Validates: Requirements 2.6, 10.2, 10.3, 10.4, 10.5**
    - Verificar que todas las imágenes retornadas tienen claves requeridas
    - Verificar que todos los valores son no nulos
  
  - [ ]* 2.4 Escribir property test para preservación de orden
    - **Property 7: Preservación de Orden**
    - **Validates: Requirements 1.5**
    - Verificar que orden relativo de imágenes se mantiene después de filtrado

- [ ] 3. Checkpoint - Validar función de extracción
  - Ejecutar todos los property tests con 100 iteraciones
  - Verificar que todos los tests pasan
  - Revisar logs de ejecución para validar comportamiento
  - Ensure all tests pass, ask the user if questions arise.

- [x] 4. Implementar manejo de casos especiales
  - [x] 4.1 Implementar manejo de lista vacía
    - Retornar lista vacía cuando `resultados` está vacío
    - Registrar mensaje informativo
    - _Requirements: 1.3, 8.1_
  
  - [x] 4.2 Implementar manejo de sin imágenes en resultados
    - Retornar lista vacía cuando no hay resultados tipo "imagen"
    - Registrar advertencia específica
    - _Requirements: 1.4, 8.2, 8.4_
  
  - [x] 4.3 Implementar robustez ante errores individuales
    - Continuar procesamiento cuando una imagen falla validación
    - No lanzar excepciones, siempre retornar lista
    - _Requirements: 8.5_
  
  - [ ]* 4.4 Escribir property test para lista vacía sin imágenes
    - **Property 8: Lista Vacía Cuando No Hay Imágenes**
    - **Validates: Requirements 1.3, 1.4, 8.1, 8.2, 8.4**
    - Usar generator con solo resultados tipo "texto"
    - Verificar que resultado es lista vacía
  
  - [ ]* 4.5 Escribir property test para robustez ante errores
    - **Property 9: Robustez ante Errores Individuales**
    - **Validates: Requirements 8.5**
    - Usar generator con mezcla de imágenes válidas e inválidas
    - Verificar que imágenes válidas son retornadas, inválidas omitidas

- [ ] 5. Escribir unit tests específicos
  - [ ]* 5.1 Escribir test para lista vacía
    - Verificar que `extraer_imagenes_de_resultados([])` retorna `[]`
  
  - [ ]* 5.2 Escribir test para solo resultados de texto
    - Verificar que resultados sin tipo "imagen" retornan lista vacía
  
  - [ ]* 5.3 Escribir test para path no existente
    - Mock `os.path.exists()` para retornar False
    - Verificar que imagen es omitida
  
  - [ ]* 5.4 Escribir test para path nulo
    - Verificar que imagen sin `path` o `imagen_path` es omitida
  
  - [ ]* 5.5 Escribir test para fallback de nombre_archivo
    - Verificar uso de `os.path.basename()` cuando `nombre_archivo` está vacío
  
  - [ ]* 5.6 Escribir test para propiedades faltantes
    - Verificar que imagen sin propiedades requeridas es omitida
  
  - [ ]* 5.7 Escribir test para top_k menor que disponibles
    - Verificar que solo top_k imágenes son retornadas
  
  - [ ]* 5.8 Escribir test para top_k mayor que disponibles
    - Verificar que todas las imágenes disponibles son retornadas
  
  - [ ]* 5.9 Escribir test para renombrado correcto
    - Verificar transformación exacta de propiedades
  
  - [ ]* 5.10 Escribir test para formato de logging
    - Capturar stdout y verificar formato de logs

- [ ] 6. Checkpoint - Validar tests unitarios
  - Ejecutar todos los unit tests
  - Verificar cobertura de código >= 100% para `extraer_imagenes_de_resultados()`
  - Ensure all tests pass, ask the user if questions arise.

- [x] 7. Modificar nodo de búsqueda Neo4j
  - [x] 7.1 Modificar `_nodo_buscar_neo4j()` para invocar extracción
    - Reemplazar invocación de `busqueda_imagenes_semantica()` con `extraer_imagenes_de_resultados()`
    - Pasar `state["resultados_busqueda"]` como parámetro
    - Mantener mismo flujo de actualización de `state["imagenes_para_mostrar"]`
    - Actualizar logging para indicar extracción en lugar de búsqueda
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6_
  
  - [x] 7.2 Actualizar trayectoria con nuevo indicador
    - Agregar campo `imagenes_extraidas_de_vecindad: True` a la trayectoria
    - _Requirements: 7.6_
  
  - [x] 7.3 Comentar código de búsqueda adicional
    - Comentar (no eliminar) invocación de `busqueda_imagenes_semantica()`
    - Agregar comentario explicativo sobre el cambio
    - _Requirements: 9.1, 9.2, 9.3_

- [ ] 8. Escribir integration tests
  - [ ]* 8.1 Escribir test para verificar no invocación de búsqueda adicional
    - **Property 10: No Invocación de Búsqueda Adicional**
    - **Validates: Requirements 6.4, 9.1, 9.3**
    - Mock `busqueda_imagenes_semantica()` para verificar que NO es llamado
    - Ejecutar `_nodo_buscar_neo4j()` con `mostrar_imagenes=True`
  
  - [ ]* 8.2 Escribir test para invocación de extracción
    - Verificar que `extraer_imagenes_de_resultados()` es invocado
    - Verificar que recibe `state["resultados_busqueda"]` como parámetro
  
  - [ ]* 8.3 Escribir test para actualización de state
    - Verificar que `state["imagenes_para_mostrar"]` es actualizado correctamente
    - Verificar que `state["contexto_suficiente"]` es establecido cuando hay imágenes
  
  - [ ]* 8.4 Escribir test para actualización de trayectoria
    - Verificar que trayectoria incluye `imagenes_extraidas_de_vecindad=True`
  
  - [ ]* 8.5 Escribir test para flujo completo con imágenes
    - Simular consulta completa con `mostrar_imagenes=True`
    - Verificar que imágenes son extraídas y mostradas correctamente
  
  - [ ]* 8.6 Escribir test para flujo completo sin imágenes
    - Simular `resultados_busqueda` sin tipo "imagen"
    - Verificar comportamiento cuando no hay imágenes disponibles
  
  - [ ]* 8.7 Escribir test para compatibilidad con frontend
    - Verificar que formato de salida es idéntico al esperado por frontend
    - Verificar estructura de diccionarios retornados

- [ ] 9. Checkpoint final - Validar integración completa
  - Ejecutar todos los tests (unit + property + integration)
  - Verificar cobertura de código >= 95% para código nuevo
  - Revisar logs de ejecución para validar comportamiento end-to-end
  - Ensure all tests pass, ask the user if questions arise.

- [x] 10. Validación manual y documentación
  - [x] 10.1 Ejecutar sistema en modo interactivo
    - Probar con consultas que solicitan imágenes explícitamente
    - Verificar que imágenes mostradas son relevantes al contexto
    - Verificar que no hay errores en logs
  
  - [x] 10.2 Comparar rendimiento con implementación anterior
    - Medir tiempo de ejecución de extracción vs búsqueda adicional
    - Verificar mejora de rendimiento esperada (25-50x)
  
  - [x] 10.3 Actualizar comentarios en código
    - Documentar función `extraer_imagenes_de_resultados()` con docstring completo
    - Documentar cambios en `_nodo_buscar_neo4j()`
    - Agregar comentarios explicativos sobre el flujo simplificado

## Notes

- Las tareas marcadas con `*` son opcionales (tests) y pueden omitirse para un MVP más rápido
- Cada tarea referencia requirements específicos para trazabilidad
- Los checkpoints aseguran validación incremental del progreso
- Property tests validan propiedades universales de corrección
- Unit tests validan casos específicos y edge cases
- Integration tests validan el flujo completo end-to-end
- La función `busqueda_imagenes_semantica()` se mantiene sin cambios para uso futuro o legacy
- El cambio es transparente para el usuario (mismo comportamiento observable)
- Se espera una mejora de rendimiento significativa (25-50x más rápido)
