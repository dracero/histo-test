@echo off
echo ==============================================
echo Inciando Servidor del Agente de Histologia...
echo ==============================================
echo.

echo Abriendo la interfaz en su navegador...
start "" cmd /c "timeout /t 5 >nul & start http://localhost:10005"

call uv run python server.py

echo.
echo Presione cualquier tecla para salir...
pause >nul
