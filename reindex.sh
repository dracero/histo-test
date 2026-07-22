#!/bin/bash
# Script para reindexar completamente la base de datos

set -e

echo "🔄 Reindexación completa de la base de datos..."

# 1. Verificar que Neo4j esté corriendo
if ! ./status-neo4j.sh >/dev/null 2>&1; then
    echo "❌ Neo4j no está corriendo. Ejecuta 'npm run dev' primero."
    exit 1
fi

echo "✅ Neo4j está corriendo"

# 2. Ejecutar reindexación forzada
echo "📚 Ejecutando reindexación con --reindex --force..."
uv run python ne4j-histo.py --reindex --force

echo "✅ Reindexación completada"
