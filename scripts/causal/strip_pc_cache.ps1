param([Parameter(Mandatory=$true)][string]$InputPath,
      [Parameter(Mandatory=$true)][string]$OutputPath)
$ErrorActionPreference = 'Stop'
$taskCompilerOptions = New-Object System.CodeDom.Compiler.CompilerParameters
$taskCompilerOptions.CompilerOptions = '/optimize+'
$taskCompilerOptions.GenerateInMemory = $true
[void]$taskCompilerOptions.ReferencedAssemblies.Add('System.dll')
[void]$taskCompilerOptions.ReferencedAssemblies.Add('System.Core.dll')
Add-Type -Path (Join-Path $PSScriptRoot 'PcCacheStripper.cs') -CompilerParameters $taskCompilerOptions
[PcCacheStripper]::Rewrite($InputPath, $OutputPath) | ConvertTo-Json -Compress
