#!/bin/bash
# Script para detener el sistema completo

echo "🛑 Deteniendo RAG Histología Neo4j..."

# 1. Detener el servidor Python (buscar y matar procesos)
echo "🐍 Deteniendo servidor Python..."
pkill -f "uvicorn server:app" || echo "   No hay procesos uvicorn corriendo"
pkill -f "python.*server.py" || echo "   No hay procesos server.py corriendo"

# 2. Detener Neo4j Local
echo "📦 Deteniendo Neo4j local..."
./stop-neo4j.sh

echo "✅ Sistema detenido completamente"
