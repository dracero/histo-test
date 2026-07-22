#!/bin/bash
# Script para iniciar el sistema completo

set -e  # Salir si hay error

echo "🚀 Iniciando RAG Histología Neo4j..."

# 1. Iniciar Neo4j Local
echo "📦 Iniciando Neo4j local..."
./start-neo4j.sh

# 2. Esperar a que Neo4j esté completamente listo
echo "⏳ Esperando a que Neo4j esté listo..."
MAX_RETRIES=30
RETRY_COUNT=0

until curl -s http://localhost:7474 > /dev/null 2>&1; do
    RETRY_COUNT=$((RETRY_COUNT + 1))
    if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
        echo "❌ Neo4j no respondió después de $MAX_RETRIES intentos"
        exit 1
    fi
    echo "   Intento $RETRY_COUNT/$MAX_RETRIES - esperando 2s..."
    sleep 2
done

# Esperar 15 segundos adicionales para que la base de datos 'neo4j' esté completamente lista
# Esto es especialmente importante después de limpiar la base de datos
echo "   ✅ HTTP disponible. Esperando 15s para que la BD esté completamente lista..."
sleep 15

echo "✅ Neo4j está listo!"

# 3. Iniciar el servidor Python con uvicorn
echo "🐍 Iniciando servidor Python..."
uv run uvicorn server:app --reload --host 0.0.0.0 --port 10005
