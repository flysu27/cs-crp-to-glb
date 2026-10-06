@echo off
rem crp2glb drag-and-drop converter (ASCII-only to avoid encoding issues)
setlocal enabledelayedexpansion
chcp 65001 >nul
set "SCRIPT=%~dp0crp2glb.py"

if "%~1"=="" (
  echo Drag and drop one or more .crp files onto this .bat to convert them to .glb
  echo.
  pause
  exit /b 1
)

set "PYEXE=python"
where py >nul 2>nul && set "PYEXE=py"

:loop
if "%~1"=="" goto end
echo === Converting: %~nx1
"%PYEXE%" "%SCRIPT%" "%~1"
shift
goto loop

:end
echo.
echo All done. Check the .glb files next to the input files.
pause
