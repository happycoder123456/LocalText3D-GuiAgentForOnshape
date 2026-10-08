Add-Type -AssemblyName System.Drawing
Add-Type @"
using System;using System.Runtime.InteropServices;using System.Text;
public class W {
 [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr h);
 [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr h,int c);
 [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr h, out RECT r);
 [DllImport("user32.dll")] public static extern int GetWindowText(IntPtr h, StringBuilder s, int n);
 [StructLayout(LayoutKind.Sequential)] public struct RECT { public int L,T,R,B; }
}
"@
$found = $false
Get-Process | Where-Object { $_.MainWindowHandle -ne 0 } | ForEach-Object {
  $sb = New-Object System.Text.StringBuilder(256)
  [W]::GetWindowText($_.MainWindowHandle, $sb, 256) | Out-Null
  if ($sb.ToString() -like "*Onshape*") {
    Write-Output ("FOUND: " + $sb.ToString())
    [W]::ShowWindow($_.MainWindowHandle, 9) | Out-Null
    [W]::SetForegroundWindow($_.MainWindowHandle) | Out-Null
    Start-Sleep -Milliseconds 800
    $r = New-Object W+RECT
    [W]::GetWindowRect($_.MainWindowHandle, [ref]$r) | Out-Null
    Write-Output ("RECT: " + $r.L + "," + $r.T + "," + $r.R + "," + $r.B)
    $w = $r.R - $r.L; $h = $r.B - $r.T
    $bmp = New-Object System.Drawing.Bitmap($w, $h)
    $g = [System.Drawing.Graphics]::FromImage($bmp)
    $g.CopyFromScreen($r.L, $r.T, 0, 0, (New-Object System.Drawing.Size($w,$h)))
    $bmp.Save((Join-Path $PSScriptRoot "_win.png"))
    $found = $true
  }
}
if (-not $found) { Write-Output "NO_WINDOW" }
