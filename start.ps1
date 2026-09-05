$ErrorActionPreference = 'Stop'
# FastVideo CUDA requires Linux. Use the user's default WSL distribution.
$linuxPath = & wsl.exe --exec wslpath -a $PSScriptRoot
if ($LASTEXITCODE -ne 0 -or -not $linuxPath) {
    throw 'Zainstaluj Ubuntu: wsl --install -d Ubuntu, uruchom ponownie komputer i skonfiguruj konto Linux.'
}
& wsl.exe --cd $linuxPath.Trim() --exec bash ./start.sh @args
exit $LASTEXITCODE
