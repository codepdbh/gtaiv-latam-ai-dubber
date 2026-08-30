@echo off
setlocal
if "%~1"=="" (set "PROJECT_ROOT=%~dp0") else (set "PROJECT_ROOT=%~1")
for /f "delims=" %%I in ('wsl.exe wslpath -u "%~dp0scripts\gui.sh"') do set "WSL_SCRIPT=%%I"
wsl.exe bash "%WSL_SCRIPT%" "%PROJECT_ROOT%"
exit /b %ERRORLEVEL%

