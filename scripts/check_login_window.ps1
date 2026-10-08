# Count agent-profile Chrome windows (login window check).
$p = Get-CimInstance Win32_Process -Filter "Name='chrome.exe'" |
    Where-Object { $_.CommandLine -like '*agent-profile*' }
Write-Output ("agent-chrome count: " + @($p).Count)
