# List of models you want to test
$models = @("gemma3:1b", "gemma3:2b-instruct-q4", "llama3.1:8b-instruct-q4_0")

foreach ($model in $models) {
    Write-Host "====================================="
    Write-Host "Starting PrivateGPT with model: $model"
    Write-Host "====================================="

    # Set environment variables
    $env:OLLAMA_HOST = "http://127.0.0.1:11434"
    $env:PGPT_PROFILES = "ollama"
    $env:PGPT__LLM__PROVIDER = "ollama"
    $env:PGPT__LLM__OLLAMA__MODEL = $model
    $env:PGPT__EMBEDDING__PROVIDER = "ollama"
    $env:PGPT__EMBEDDING__OLLAMA__MODEL = "nomic-embed-text"

    # Start PrivateGPT
    py -3.11 -m poetry run python -m private_gpt

    # Wait for you to press Enter before moving to next model
    Read-Host "Press Enter to stop this model and continue with the next..."
}
