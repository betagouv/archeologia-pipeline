@echo off
REM Talisman pre-push hook for Windows shells.
REM Requires 'talisman' to be available in PATH.

talisman --githook pre-push %*
IF %ERRORLEVEL% NEQ 0 (
  echo [talisman] Blocking push (potential secret detected or talisman error)
  exit /b %ERRORLEVEL%
)

exit /b 0
