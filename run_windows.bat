@echo off
echo ==============================================
echo Inciando Servidor del Agente de Histologia...
echo ==============================================
echo.

call uv run python server.py

echo.
echo Presione cualquier tecla para salir...
pause >nul
