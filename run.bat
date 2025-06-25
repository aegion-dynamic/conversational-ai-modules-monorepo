@echo off
REM Windows batch file for project commands
REM Supports both Poetry and uv workflows

if "%1"=="" goto help
if "%1"=="help" goto help
if "%1"=="install-poetry" goto install-poetry
if "%1"=="install-uv" goto install-uv
if "%1"=="install" goto install
if "%1"=="sync" goto sync
if "%1"=="test" goto test
if "%1"=="lint" goto lint
if "%1"=="format" goto format
if "%1"=="clean" goto clean
if "%1"=="setup-uv" goto setup-uv

:help
echo Available commands:
echo   install-poetry  - Install dependencies using Poetry (traditional)
echo   install-uv      - Install dependencies using uv (fast)
echo   install         - Install using uv if available, fallback to Poetry
echo   sync            - Sync uv with Poetry lock file
echo   test            - Run tests
echo   lint            - Run linting with pylint
echo   format          - Format code with black and isort
echo   clean           - Clean build artifacts
echo   setup-uv        - Set up uv alongside Poetry
goto end

:install-poetry
echo Installing dependencies using Poetry...
poetry install
goto end

:install-uv
echo Installing dependencies using uv...
where uv >nul 2>&1
if %errorlevel% neq 0 (
    echo uv not found. Installing...
    pip install uv
)
uv pip install -r requirements.txt
goto end

:install
where uv >nul 2>&1
if %errorlevel% equ 0 (
    if exist requirements.txt (
        echo Using uv for fast installation...
        uv pip install -r requirements.txt
    ) else (
        echo requirements.txt not found. Using Poetry...
        poetry install
    )
) else (
    echo Using Poetry for installation...
    poetry install
)
goto end

:sync
echo Exporting Poetry dependencies for uv...
poetry export -f requirements.txt --output requirements.txt --with dev
echo Dependencies exported to requirements.txt for uv usage
goto end

:test
where uv >nul 2>&1
if %errorlevel% equ 0 (
    uv run pytest
) else (
    poetry run pytest
)
goto end

:lint
where uv >nul 2>&1
if %errorlevel% equ 0 (
    uv run pylint nlqs\ expert_system\ tot\ utils\ tog\ state_machine\
) else (
    poetry run pylint nlqs\ expert_system\ tot\ utils\ tog\ state_machine\
)
goto end

:format
where uv >nul 2>&1
if %errorlevel% equ 0 (
    uv run black .
    uv run isort .
) else (
    poetry run black .
    poetry run isort .
)
goto end

:clean
echo Cleaning build artifacts...
for /d /r %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d"
del /s /q *.pyc >nul 2>&1
for /d /r %%d in (*.egg-info) do @if exist "%%d" rd /s /q "%%d"
if exist build rd /s /q build
if exist dist rd /s /q dist
if exist .coverage del .coverage
if exist htmlcov rd /s /q htmlcov
if exist .pytest_cache rd /s /q .pytest_cache
if exist .mypy_cache rd /s /q .mypy_cache
goto end

:setup-uv
echo Setting up uv alongside Poetry...
pwsh -ExecutionPolicy Bypass -File setup_uv.ps1
goto end

:end
