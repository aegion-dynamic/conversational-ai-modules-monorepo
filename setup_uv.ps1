# PowerShell script for setting up uv alongside Poetry
# Provides fast dependency management while maintaining Poetry compatibility

Write-Host "🚀 Setting up uv alongside Poetry for faster dependency management" -ForegroundColor Green
Write-Host "=" * 60

function Test-CommandExists {
    param($Command)
    try {
        Get-Command $Command -ErrorAction Stop | Out-Null
        return $true
    }
    catch {
        return $false
    }
}

function Invoke-SafeCommand {
    param(
        [string]$Command,
        [string]$Description
    )
    Write-Host "Running: $Description" -ForegroundColor Yellow
    Write-Host "Command: $Command" -ForegroundColor Gray
    
    try {
        Invoke-Expression $Command
        Write-Host "✅ $Description completed successfully" -ForegroundColor Green
        return $true
    }
    catch {
        Write-Host "❌ $Description failed: $_" -ForegroundColor Red
        return $false
    }
}

# Check Poetry installation
if (-not (Test-CommandExists "poetry")) {
    Write-Host "❌ Poetry not found. Please install Poetry first." -ForegroundColor Red
    exit 1
}
Write-Host "✅ Poetry is installed" -ForegroundColor Green

# Install uv if not already installed
if (-not (Test-CommandExists "uv")) {
    Write-Host "Installing uv..." -ForegroundColor Yellow
    if (-not (Invoke-SafeCommand "pip install uv" "uv installation")) {
        exit 1
    }
} else {
    Write-Host "✅ uv is already installed" -ForegroundColor Green
}

# Check if poetry.lock exists
if (-not (Test-Path "poetry.lock")) {
    Write-Host "❌ poetry.lock not found. Run 'poetry install' first." -ForegroundColor Red
    exit 1
}

# Export Poetry dependencies and install with uv
Write-Host "Syncing uv with Poetry lock file..." -ForegroundColor Yellow
if (Invoke-SafeCommand "poetry export -f requirements.txt --output requirements.txt --with dev" "Export Poetry dependencies") {
    if (Invoke-SafeCommand "uv pip install -r requirements.txt" "Install dependencies with uv") {
        Write-Host "🎉 Setup completed successfully!" -ForegroundColor Green
        Write-Host ""
        Write-Host "Usage recommendations:" -ForegroundColor Cyan
        Write-Host "- Use Poetry for adding/removing dependencies: poetry add <package>" -ForegroundColor White
        Write-Host "- Use uv for fast installations: uv pip install <package>" -ForegroundColor White
        Write-Host "- Use uv for virtual environments: uv venv" -ForegroundColor White
        Write-Host "- To sync uv with Poetry changes, run this script again" -ForegroundColor White
    }
}
