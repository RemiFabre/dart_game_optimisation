# Reference Python implementation

This is the original 2022 vacation-code solver, frozen here as the **verification oracle** for the new package in `src/darts/`. Do not modify the source files — their entire purpose is to serve as a reference that the new code's numerics can be checked against.

## Files

- `aiming_spots.py` — Monte-Carlo EV-per-aim-spot solver. The new `darts.board.get_score` was tested for exact pixel-by-pixel agreement with `aiming_spots.get_score` over a dense grid; see `tests/test_board.py`.
- `end_game.py` — Backwards-DP solver for the simple-mode 301 game (no double-out, one throw at a time, miss == bust). Cross-checked against the new solver's simple mode in `tests/test_endgame.py`.
- `scores.py` — Helper for deriving a "player σ" from a list of running scores.
- `scores_ev_and_pos_2601_size10000_sx0.07_sy0.07` — Pickled end-game DP result for the good player. Loaded by the verification test.
- `proba*`, `sorted_spots*` — Pickled intermediate caches from the original runs.
- `img/`, root PNGs, `opti_shots_good_player.md` — Generated outputs from the original work. Kept for provenance.

## Known issues (preserved as-is)

- `traceback.format_exc(e)` is called incorrectly (the first argument is supposed to be an integer `limit`); the original silently relies on the surrounding pickle path being a cache-hit. Don't fix here.
- The infinite-series bust correction in `end_game.approximate_spot_ev` truncates at depth 9; the new solver uses the equivalent closed form `1 / (1 - p_bust)`.
- Score 0 (off-board / miss) is conflated with "bust", which is a valid simplification but ignores genuine misses.

The new `darts.endgame.solve_simple` reproduces this reference's V values to within Monte-Carlo noise (~0.5 throws over the full 301 game).
