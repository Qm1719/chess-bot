#!/usr/bin/env bash
set -eu
# pipefail is bash-specific and may be disabled in some environments.
set -o pipefail 2>/dev/null || true

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "${here}/../.." && pwd)"   # Chess2/submission v5
sf="${root}/engine/stockfish/src"
out="${here}/sf_nnue_bridge.so"

echo "Building Stockfish NNUE bridge (.so)"
echo "  stockfish src: ${sf}"
echo "  out: ${out}"

# Stockfish embeds default nets via INCBIN. The build will fail if these files
# don't exist, even if we never use the embedded nets at runtime.
defaultBig="${root}/nn-c288c895ea92.nnue"
defaultSml="${root}/nn-37f18f62d772.nnue"
for p in "${defaultBig}" "${defaultSml}"; do
  if [[ ! -f "${p}" ]]; then
    echo "Creating placeholder: ${p}"
    : > "${p}"
  fi
done

src=(
  "${here}/sf_nnue_bridge.cpp"
  "${sf}/benchmark.cpp"
  "${sf}/bitboard.cpp"
  "${sf}/engine.cpp"
  "${sf}/evaluate.cpp"
  "${sf}/memory.cpp"
  "${sf}/misc.cpp"
  "${sf}/movegen.cpp"
  "${sf}/movepick.cpp"
  "${sf}/position.cpp"
  "${sf}/score.cpp"
  "${sf}/search.cpp"
  "${sf}/thread.cpp"
  "${sf}/timeman.cpp"
  "${sf}/tt.cpp"
  "${sf}/tune.cpp"
  "${sf}/uci.cpp"
  "${sf}/ucioption.cpp"
  "${sf}/syzygy/tbprobe.cpp"
  "${sf}/nnue/network.cpp"
  "${sf}/nnue/nnue_accumulator.cpp"
  "${sf}/nnue/nnue_misc.cpp"
  "${sf}/nnue/features/full_threats.cpp"
  "${sf}/nnue/features/half_ka_v2_hm.cpp"
)

# INCBIN in Stockfish resolves nn-*.nnue relative to the compiler cwd; placeholders live in submission v5 root.
pushd "${root}" >/dev/null
g++ -shared -fPIC -O3 -std=c++17 -DNDEBUG -DUSE_SSE2 -msse2 \
  -I "${sf}" -I "${sf}/incbin" \
  -Wno-unused-parameter \
  -o "${out}" \
  "${src[@]}"
popd >/dev/null

echo "Done."

