$ErrorActionPreference = "Stop"

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$root = Resolve-Path (Join-Path $here "..\..")  # Chess2/
$sf   = Join-Path $root "stockfish\src"

$outDir = $here
$outDll = Join-Path $outDir "sf_nnue_bridge.dll"

Write-Host "Building Stockfish NNUE bridge DLL..."
Write-Host "  root: $root"
Write-Host "  stockfish src: $sf"
Write-Host "  out: $outDll"

# Stockfish embeds default nets via INCBIN. The build will fail if these files
# don't exist, even if we never use the embedded nets at runtime.
# Create tiny placeholders in the build working directory (repo root).
$defaultBig = Join-Path $root "nn-c288c895ea92.nnue"
$defaultSml = Join-Path $root "nn-37f18f62d772.nnue"
foreach ($p in @($defaultBig, $defaultSml)) {
  if (-not (Test-Path -LiteralPath $p)) {
    Write-Host "Creating placeholder: $p"
    [System.IO.File]::WriteAllBytes($p, [byte[]](0))
  }
}

# Minimal source set for evaluation + NNUE.
$src = @(
  (Join-Path $here "sf_nnue_bridge.cpp"),
  # Stockfish core (everything except main.cpp)
  (Join-Path $sf "benchmark.cpp"),
  (Join-Path $sf "bitboard.cpp"),
  (Join-Path $sf "engine.cpp"),
  (Join-Path $sf "evaluate.cpp"),
  (Join-Path $sf "memory.cpp"),
  (Join-Path $sf "misc.cpp"),
  (Join-Path $sf "movegen.cpp"),
  (Join-Path $sf "movepick.cpp"),
  (Join-Path $sf "position.cpp"),
  (Join-Path $sf "score.cpp"),
  (Join-Path $sf "search.cpp"),
  (Join-Path $sf "thread.cpp"),
  (Join-Path $sf "timeman.cpp"),
  (Join-Path $sf "tt.cpp"),
  (Join-Path $sf "tune.cpp"),
  (Join-Path $sf "uci.cpp"),
  (Join-Path $sf "ucioption.cpp"),

  # Syzygy (required by some Position helpers; harmless if not used)
  (Join-Path $sf "syzygy\tbprobe.cpp"),

  # NNUE
  (Join-Path $sf "nnue\network.cpp"),
  (Join-Path $sf "nnue\nnue_accumulator.cpp"),
  (Join-Path $sf "nnue\nnue_misc.cpp"),
  (Join-Path $sf "nnue\features\full_threats.cpp"),
  (Join-Path $sf "nnue\features\half_ka_v2_hm.cpp")
)

# Compile flags:
# - USE_SSE2 is required for NNUE SIMD width definitions in Stockfish.
# - -static-libgcc/-static-libstdc++ makes the DLL easier to run on machines without MinGW runtimes.
$cxx = "g++"
$args = @(
  "-shared",
  "-O3",
  "-std=c++17",
  "-DNDEBUG",
  "-DUSE_SSE2",
  "-msse2",
  "-I", $sf,
  "-I", (Join-Path $sf "incbin"),
  "-static-libgcc",
  "-static-libstdc++",
  "-o", $outDll
) + $src

# INCBIN in network.cpp resolves nn-*.nnue relative to the compiler cwd; placeholders live in repo root.
Push-Location $root
try {
  & $cxx @args
} finally {
  Pop-Location
}

Write-Host "Done."

