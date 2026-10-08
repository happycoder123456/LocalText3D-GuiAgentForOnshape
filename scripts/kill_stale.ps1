# Stop leftover `python -m agent login` processes and agent-profile Chrome.
$ErrorActionPreference = 'SilentlyContinue'
Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object {
    $_.CommandLine -like '*agent*login*'
} | ForEach-Object {
    Stop-Process -Id $_.ProcessId -Force
    Write-Output ("killed python " + $_.ProcessId)
}
Get-CimInstance Win32_Process -Filter "Name='chrome.exe'" | Where-Object {
    $_.CommandLine -like '*agent-profile*'
} | ForEach-Object {
    Stop-Process -Id $_.ProcessId -Force
    Write-Output ("killed chrome " + $_.ProcessId)
}
