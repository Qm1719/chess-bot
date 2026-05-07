// sf_nnue_bridge.cpp
//
// Stockfish NNUE evaluation bridge (DLL) for the Python v4 engine.
// Uses full Eval::evaluate() + UCIEngine::to_cp() so scores match Stockfish UCI.
//
// C ABI:
//   - sf_nnue_init(evalfile_path_big)  — small net: same dir + EvalFileDefaultNameSmall
//   - sf_nnue_eval_cp(bb12, side_to_move)  — legacy; FEN omits castling/ep
//   - sf_nnue_eval_cp_fen(fen)  — preferred: full FEN from the Python engine
//   - sf_nnue_close()
//
// Build: see build_mingw.ps1 in this folder.

#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>

#include "../../stockfish/src/bitboard.h"
#include "../../stockfish/src/evaluate.h"
#include "../../stockfish/src/misc.h"
#include "../../stockfish/src/nnue/network.h"  // Networks, NetworkBig/Small
#include "../../stockfish/src/nnue/nnue_accumulator.h"
#include "../../stockfish/src/position.h"
#include "../../stockfish/src/types.h"
#include "../../stockfish/src/uci.h"

using namespace Stockfish;

#if defined(_WIN32) || defined(__CYGWIN__)
    #define SFNNUE_API __declspec(dllexport)
#else
    #define SFNNUE_API __attribute__((visibility("default")))
#endif

namespace {

std::once_flag g_init_once;
std::mutex     g_mutex;
bool           g_ready = false;

Eval::NNUE::Networks*           g_networks = nullptr;
Eval::NNUE::AccumulatorCaches*  g_caches   = nullptr;

void init_stockfish_tables() {
    Bitboards::init();
    Position::init();
}

// Build a minimal FEN from 12 bitboards (legacy path; castling/ep not preserved).
std::string fen_from_bitboards(const std::uint64_t bb12[12], int stm) {
    char board[64];
    std::memset(board, '.', sizeof(board));

    auto put = [&](int piece, char ch) {
        std::uint64_t b = bb12[piece];
        while (b) {
            int sq = pop_lsb(b);
            board[sq] = ch;
        }
    };

    put(0, 'P');
    put(1, 'N');
    put(2, 'B');
    put(3, 'R');
    put(4, 'Q');
    put(5, 'K');
    put(6, 'p');
    put(7, 'n');
    put(8, 'b');
    put(9, 'r');
    put(10, 'q');
    put(11, 'k');

    std::string fen;
    fen.reserve(128);
    for (int r = 7; r >= 0; --r) {
        int empty = 0;
        for (int f = 0; f < 8; ++f) {
            int sq = r * 8 + f;
            char c = board[sq];
            if (c == '.') {
                empty++;
            } else {
                if (empty) {
                    fen.push_back(char('0' + empty));
                    empty = 0;
                }
                fen.push_back(c);
            }
        }
        if (empty)
            fen.push_back(char('0' + empty));
        if (r)
            fen.push_back('/');
    }

    fen.push_back(' ');
    fen.push_back(stm == 0 ? 'w' : 'b');
    fen += " - - 0 1";
    return fen;
}

// White-centric UCI centipawns; v must be from White's perspective (Stockfish Value units).
int eval_white_cp_from_position(Position& pos) {
    if (pos.checkers()) {
        Value se = Value(Eval::simple_eval(pos));
        Value w  = pos.side_to_move() == WHITE ? se : -se;
        return UCIEngine::to_cp(w, pos);
    }

    thread_local Eval::NNUE::AccumulatorStack accum;
    accum.reset();
    Value v = Eval::evaluate(*g_networks, pos, accum, *g_caches, 0);
    Value w = pos.side_to_move() == WHITE ? v : -v;
    return UCIEngine::to_cp(w, pos);
}

}  // namespace

extern "C" {

SFNNUE_API int sf_nnue_init(const char* evalfile_path) {
    std::call_once(g_init_once, init_stockfish_tables);

    std::lock_guard<std::mutex> lock(g_mutex);

    if (g_ready)
        return 0;

    try {
        std::string path = evalfile_path ? std::string(evalfile_path) : std::string();

        g_networks =
          new Eval::NNUE::Networks(Eval::NNUE::EvalFile{EvalFileDefaultNameBig, "None", ""},
                                   Eval::NNUE::EvalFile{EvalFileDefaultNameSmall, "None", ""});

        std::string rootDir;
        std::string fileName = path;
        const auto  p1       = path.find_last_of("/\\");
        if (p1 != std::string::npos) {
            rootDir  = path.substr(0, p1 + 1);
            fileName = path.substr(p1 + 1);
        }

        // Big net from user path; small net same directory, official default filename.
        // Do NOT call Network::verify(): on mismatch it calls exit(EXIT_FAILURE) and kills the host
        // Python process (exit code 1). Use get_content_hash() instead to detect failed loads.
        const std::string smallName = std::string(EvalFileDefaultNameSmall);
        g_networks->big.load(rootDir, fileName);
        g_networks->small.load(rootDir, smallName);

        if (g_networks->big.get_content_hash() == 0 || g_networks->small.get_content_hash() == 0) {
            delete g_networks;
            g_networks = nullptr;
            g_ready = false;
            return 1;
        }

        g_caches = new Eval::NNUE::AccumulatorCaches(*g_networks);

        g_ready = true;
        return 0;
    } catch (...) {
        g_ready = false;
        if (g_networks) {
            delete g_networks;
            g_networks = nullptr;
        }
        return 1;
    }
}

SFNNUE_API int sf_nnue_eval_cp(const std::uint64_t bb12[12], int side_to_move) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_ready || !g_networks || !g_caches)
        return 0;

    StateInfo st;
    Position  pos;
    const std::string fen = fen_from_bitboards(bb12, side_to_move);
    pos.set(fen, false, &st);

    return eval_white_cp_from_position(pos);
}

SFNNUE_API int sf_nnue_eval_cp_fen(const char* fen_cstr) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_ready || !g_networks || !g_caches || !fen_cstr)
        return 0;

    StateInfo st;
    Position  pos;
    pos.set(std::string(fen_cstr), false, &st);

    return eval_white_cp_from_position(pos);
}

SFNNUE_API void sf_nnue_close() {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_caches) {
        delete g_caches;
        g_caches = nullptr;
    }
    if (g_networks) {
        delete g_networks;
        g_networks = nullptr;
    }
    g_ready = false;
}

}  // extern "C"
