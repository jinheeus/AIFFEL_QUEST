# PowerShell Script to start the Chatbot System on Windows

Write-Host "🚀 Starting Agentic RAG Chatbot System (Windows/PowerShell)..."

# Variables to store process objects
$backendProcess = $null
$frontendProcess = $null

# Function to kill processes on exit
function Cleanup {
    Write-Host "🛑 Shutting down..."
    if ($backendProcess -and -not $backendProcess.HasExited) {
        Stop-Process -Id $backendProcess.Id -Force -ErrorAction SilentlyContinue
    }
    if ($frontendProcess -and -not $frontendProcess.HasExited) {
        Stop-Process -Id $frontendProcess.Id -Force -ErrorAction SilentlyContinue
    }
}

try {
    # 1. Start Backend
    Write-Host "🔹 [Backend] Starting FastAPI Server on port 8000..."
    $env:PYTHONUNBUFFERED = "1"
    $env:PYTHONIOENCODING = "utf-8"
    
    # Start python backend
    # Assumes python is in PATH
    $backendProcess = Start-Process -FilePath "python" `
        -ArgumentList "web_app/backend/main.py" `
        -RedirectStandardOutput "backend.log" `
        -RedirectStandardError "backend.log" `
        -PassThru `
        -NoNewWindow
    
    Write-Host "   -> Backend running (PID: $($backendProcess.Id)). Logs at backend.log"

    # 2. Start Frontend
    Write-Host "🔹 [Frontend] Starting Next.js App on port 3000..."
    
    # Detect npm command (npm.cmd on Windows)
    $npmPath = "npm"
    if (Get-Command "npm.cmd" -ErrorAction SilentlyContinue) {
        $npmPath = "npm.cmd"
    }

    # Start frontend
    # Using WorkingDirectory to run inside the frontend folder without changing global PWD
    $frontendProcess = Start-Process -FilePath $npmPath `
        -ArgumentList "run dev" `
        -WorkingDirectory "web_app/frontend" `
        -RedirectStandardOutput "../../frontend.log" `
        -RedirectStandardError "../../frontend.log" `
        -PassThru `
        -NoNewWindow
    
    Write-Host "   -> Frontend running (PID: $($frontendProcess.Id)). Logs at frontend.log"

    Write-Host "✅ System is UP!"
    Write-Host "   - Frontend: http://localhost:3000"
    Write-Host "   - Backend:  http://localhost:8000"
    Write-Host "   (Press Ctrl+C to stop all)"

    # Keep script running to monitor processes and catch Ctrl+C
    while ($true) {
        if ($backendProcess.HasExited) {
            Write-Error "⚠️ Backend process exited unexpectedly! Check backend.log."
            break
        }
        if ($frontendProcess.HasExited) {
            Write-Error "⚠️ Frontend process exited unexpectedly! Check frontend.log."
            break
        }
        Start-Sleep -Seconds 1
    }
}
catch {
    # Catch-all for script errors or interruptions
    Write-Host "Interrupted or Error: $_"
}
finally {
    Cleanup
}
