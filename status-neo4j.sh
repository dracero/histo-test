#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export JAVA_HOME="$SCRIPT_DIR/neo4j-local/jdk"
export PATH="$JAVA_HOME/bin:$PATH"
"$SCRIPT_DIR/neo4j-local/neo4j/bin/neo4j" status
