# Path to your .env file
$envFile = ".env"

# Check if .env file exists
if (-Not (Test-Path $envFile)) {
    Write-Host ".env file not found!"
    exit 1
}

# Read the .env file and set environment variables
Get-Content $envFile | ForEach-Object {
    if ($_ -match "^\s*([^#].*?)\s*=\s*(.*?)\s*$") {
        $name = $matches[1]
        $value = $matches[2]
        [System.Environment]::SetEnvironmentVariable($name, $value)
    }
}

# Optional: Display environment variables for confirmation
Get-ChildItem Env:
