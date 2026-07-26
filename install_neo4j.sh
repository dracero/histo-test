#!/bin/bash
set -e

# Configuración
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_DIR="$PROJECT_DIR/neo4j-local"
JDK_TAR="$LOCAL_DIR/jdk21.tar.gz"
NEO4J_TAR="$LOCAL_DIR/neo4j.tar.gz"

echo "=== Iniciando instalación local de Neo4j (User Space) ==="

# Crear directorio neo4j-local si no existe
mkdir -p "$LOCAL_DIR"

# 1. Descargar JDK 21 si no existe
if [ ! -d "$LOCAL_DIR/jdk" ]; then
    echo "Descargando JDK 21..."
    wget -O "$JDK_TAR" "https://api.adoptium.net/v3/binary/latest/21/ga/linux/x64/jdk/hotspot/normal/eclipse"
    echo "Extrayendo JDK 21..."
    tar -xzf "$JDK_TAR" -C "$LOCAL_DIR/"
    mv "$LOCAL_DIR"/jdk-21* "$LOCAL_DIR/jdk"
    rm "$JDK_TAR"
    echo "JDK 21 instalado en $LOCAL_DIR/jdk"
else
    echo "JDK 21 ya instalado."
fi

# Establecer variables de entorno de Java para el script
export JAVA_HOME="$LOCAL_DIR/jdk"
export PATH="$JAVA_HOME/bin:$PATH"

# 2. Descargar Neo4j Community Server 5.20.0 si no existe
if [ ! -d "$LOCAL_DIR/neo4j" ]; then
    echo "Descargando Neo4j Community 5.20.0..."
    wget -O "$NEO4J_TAR" "https://dist.neo4j.org/neo4j-community-5.20.0-unix.tar.gz"
    echo "Extrayendo Neo4j..."
    tar -xzf "$NEO4J_TAR" -C "$LOCAL_DIR/"
    mv "$LOCAL_DIR/neo4j-community-5.20.0" "$LOCAL_DIR/neo4j"
    rm "$NEO4J_TAR"
    echo "Neo4j instalado en $LOCAL_DIR/neo4j"
else
    echo "Neo4j ya instalado."
fi

# 3. Configurar APOC
echo "Habilitando APOC..."
cp "$LOCAL_DIR/neo4j/labs"/apoc-*-core.jar "$LOCAL_DIR/neo4j/plugins/"

# Agregar configuraciones de APOC en neo4j.conf
CONF_FILE="$LOCAL_DIR/neo4j/conf/neo4j.conf"
if ! grep -q "dbms.security.procedures.unrestricted" "$CONF_FILE"; then
    echo "" >> "$CONF_FILE"
    echo "# Habilitar procedimientos APOC" >> "$CONF_FILE"
    echo "dbms.security.procedures.unrestricted=apoc.*" >> "$CONF_FILE"
    echo "dbms.security.procedures.allowlist=apoc.*" >> "$CONF_FILE"
fi

# Crear apoc.conf para habilitar importaciones/exportaciones de archivos
APOC_CONF="$LOCAL_DIR/neo4j/conf/apoc.conf"
echo "apoc.import.file.enabled=true" > "$APOC_CONF"
echo "apoc.export.file.enabled=true" >> "$APOC_CONF"

# 4. Establecer contraseña inicial (password123)
echo "Configurando contraseña inicial..."
# Si la base de datos no se ha iniciado antes, esto creará las credenciales correctas
"$LOCAL_DIR/neo4j/bin/neo4j-admin" dbms set-initial-password password123 || echo "Nota: La contraseña inicial ya estaba establecida o la BD ya se inició anteriormente."

# 5. Crear scripts de control
echo "Creando scripts de inicio, parada y estado..."

# start-neo4j.sh
cat << 'EOF' > "$PROJECT_DIR/start-neo4j.sh"
#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export JAVA_HOME="$SCRIPT_DIR/neo4j-local/jdk"
export PATH="$JAVA_HOME/bin:$PATH"

if "$SCRIPT_DIR/neo4j-local/neo4j/bin/neo4j" status >/dev/null 2>&1; then
    echo "Neo4j is already running."
else
    "$SCRIPT_DIR/neo4j-local/neo4j/bin/neo4j" start
fi
EOF
chmod +x "$PROJECT_DIR/start-neo4j.sh"

# stop-neo4j.sh
cat << 'EOF' > "$PROJECT_DIR/stop-neo4j.sh"
#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export JAVA_HOME="$SCRIPT_DIR/neo4j-local/jdk"
export PATH="$JAVA_HOME/bin:$PATH"
"$SCRIPT_DIR/neo4j-local/neo4j/bin/neo4j" stop
EOF
chmod +x "$PROJECT_DIR/stop-neo4j.sh"

# status-neo4j.sh
cat << 'EOF' > "$PROJECT_DIR/status-neo4j.sh"
#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export JAVA_HOME="$SCRIPT_DIR/neo4j-local/jdk"
export PATH="$JAVA_HOME/bin:$PATH"
"$SCRIPT_DIR/neo4j-local/neo4j/bin/neo4j" status
EOF
chmod +x "$PROJECT_DIR/status-neo4j.sh"

echo "=== Instalación completada con éxito ==="
echo "Puedes iniciar Neo4j ejecutando: ./start-neo4j.sh"
echo "Puedes detener Neo4j ejecutando: ./stop-neo4j.sh"
echo "Puedes verificar el estado ejecutando: ./status-neo4j.sh"
