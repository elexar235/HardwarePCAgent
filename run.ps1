param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Query
)

$PythonExe = "$env:APPDATA\uv\python\cpython-3.11-windows-x86_64-none\python.exe"
if (-not (Test-Path $PythonExe)) {
    $PythonExe = "python"
}

if ($Query.Count -gt 0) {
    & $PythonExe "$PSScriptRoot\src\main.py" $Query
} else {
    & $PythonExe "$PSScriptRoot\src\main.py"
}
