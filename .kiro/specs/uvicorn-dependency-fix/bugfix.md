# Bugfix Requirements Document

## Introduction

El comando `npm run dev` falla al intentar iniciar el servidor FastAPI con uvicorn. Aunque uvicorn>=0.30.0 está correctamente declarado en pyproject.toml como dependencia del proyecto, el módulo no se encuentra disponible cuando se ejecuta `uv run uvicorn`. El error específico es "ModuleNotFoundError: No module named 'uvicorn'", y los logs indican que uv reinstala el paquete pero uvicorn no está disponible en la ruta esperada (/home/dracero/.local/bin/uvicorn). Este bug impide el arranque del servidor de desarrollo.

## Bug Analysis

### Current Behavior (Defect)

1.1 WHEN se ejecuta `npm run dev` THEN el sistema falla con "ModuleNotFoundError: No module named 'uvicorn'"

1.2 WHEN uv intenta ejecutar uvicorn mediante `uv run uvicorn server:app --reload --host 0.0.0.0 --port 10005` THEN el módulo uvicorn no se encuentra disponible en el entorno virtual

1.3 WHEN uv reinstala las dependencias THEN uvicorn no queda accesible en la ruta /home/dracero/.local/bin/uvicorn

### Expected Behavior (Correct)

2.1 WHEN se ejecuta `npm run dev` THEN el sistema SHALL iniciar el servidor FastAPI con uvicorn exitosamente

2.2 WHEN uv ejecuta uvicorn mediante `uv run uvicorn server:app --reload --host 0.0.0.0 --port 10005` THEN el módulo uvicorn SHALL estar disponible y ejecutable

2.3 WHEN las dependencias están sincronizadas THEN uvicorn SHALL estar accesible en el entorno virtual gestionado por uv

### Unchanged Behavior (Regression Prevention)

3.1 WHEN docker compose se ejecuta en el script dev THEN el sistema SHALL CONTINUE TO iniciar los contenedores correctamente

3.2 WHEN otras dependencias de pyproject.toml se utilizan THEN el sistema SHALL CONTINUE TO resolverlas correctamente con uv

3.3 WHEN se ejecuta `npm run start` con python server.py THEN el sistema SHALL CONTINUE TO funcionar como antes

3.4 WHEN se utilizan las dependencias de torch desde el índice pytorch-cu128 THEN el sistema SHALL CONTINUE TO resolverlas desde el índice personalizado
