@echo off
setlocal EnableDelayedExpansion

REM Path to your .env file
set "envfile=.env"

REM Check if .env file exists
if not exist "%envfile%" (
    echo .env file not found!
    exit /b 1
)

REM Read and set each variable from .env file
for /f "usebackq tokens=1,2 delims==" %%a in ("%envfile%") do (
    set "key=%%a"
    set "value=%%b"
    set "!key!=!value!"
)

REM Optional: Display environment variables for confirmation
set

REM Your script can follow here, utilizing the environment variables

endlocal
