@echo off
setlocal
for /f "delims=" %%I in ('wsl.exe wslpath -u "%~dp0scripts\status_mass.sh"') do set "WSL_SCRIPT=%%I"
wsl.exe bash "%WSL_SCRIPT%"
pause
exit /b %ERRORLEVEL%

