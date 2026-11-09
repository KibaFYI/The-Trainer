# build (and optionally push) the image from Windows (Docker Desktop)
#   .\build.ps1                                  -> axolotl-forge:latest
#   .\build.ps1 -Image you/axolotl-forge -Push   -> builds + pushes :0.21.1-cu130 and :latest
#   .\build.ps1 -Image you/axolotl-forge -Fa4 -DeepEP   (Hopper extras)
param(
    [string]$Image = "axolotl-forge",
    [string]$AxolotlVersion = "0.21.1",
    [switch]$Push,
    [switch]$Fa4,
    [switch]$DeepEP,
    [string]$DeepEPArch = "9.0",
    [int]$MaxJobs = 8
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$tag = "$AxolotlVersion-cu130"
$output = if ($Push) { "--push" } else { "--load" }
docker buildx build --platform linux/amd64 `
    --build-arg AXOLOTL_VERSION=$AxolotlVersion `
    --build-arg WITH_FA4=$([int]$Fa4.IsPresent) `
    --build-arg WITH_DEEPEP=$([int]$DeepEP.IsPresent) `
    --build-arg DEEPEP_ARCH=$DeepEPArch `
    --build-arg MAX_JOBS=$MaxJobs `
    -t "${Image}:$tag" -t "${Image}:latest" $output .
if ($LASTEXITCODE -ne 0) { throw "docker build failed" }
Write-Host "built ${Image}:$tag" -ForegroundColor Magenta
