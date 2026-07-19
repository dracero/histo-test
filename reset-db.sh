#!/bin/bash
# Script para resetear completamente la base de datos Neo4j

echo "⚠️  ADVERTENCIA: Esto borrará TODA la base de datos Neo4j"
echo "   Se eliminarán todos los chunks, imágenes e índices"
read -p "¿Estás seguro? (s/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Ss]$ ]]; then
    echo "❌ Operación cancelada"
    exit 1
fi

echo "🗑️  Reseteando base de datos Neo4j..."

# 1. Detener el contenedor
echo "   Deteniendo Neo4j..."
docker stop neo4j-local-histo 2>/dev/null || true

# 2. Eliminar el contenedor
echo "   Eliminando contenedor..."
docker rm neo4j-local-histo 2>/dev/null || true

# 3. Borrar los datos (requiere sudo)
echo "   Borrando datos (requiere contraseña sudo)..."
sudo rm -rf ./neo4j-local/data/*

# 4. Borrar imágenes extraídas
echo "   Borrando imágenes extraídas..."
rm -rf ./imagenes_extraidas/*

echo "✅ Base de datos reseteada completamente"
echo "   Ejecuta 'npm run dev' para reiniciar con BD limpia"
