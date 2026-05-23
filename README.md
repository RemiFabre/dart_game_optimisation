![](results/heatmaps/ev_heatmap_good_sx0.07_sy0.07.png)

# Dart game optimisation

Solver for the optimal dart aim point given a player's accuracy. Find where to aim to maximise the expected score per dart, and (with the official rules) the optimal aim for every state of the 501 end game.

Both questions reduce to **convolving the dartboard score field with a 2D Gaussian** modelling the player's aim error. The new implementation does this exactly via FFT, replacing the original Monte Carlo brute force.

The math, the design choices, and the rewrite plan are in [`ANALYSIS.md`](ANALYSIS.md).

## What's in here

```
src/darts/         the new solver (Python package, pip-installable)
├─ board.py        dartboard geometry + score function (vectorised)
├─ ev.py           FFT-based expected-value map
├─ outcomes.py     enumeration of the 63 single-dart outcomes
├─ proba.py        per-outcome probability cube (also via FFT)
├─ endgame.py      end-game DP (simple 301 mode + official 501 rules)
├─ covariance.py   2D Gaussian fit for estimating a player's σ
├─ viz.py          heatmap renderer
└─ io.py           .npy + .json output format
scripts/           runnable entry points
data/              solver outputs (per σ): .npy + .json sidecars
results/           plots: heatmaps, V(s) curves, optimum trail
tests/             34 tests including FFT-vs-Monte-Carlo verification
reference_python/  the original 2022 code, frozen as the verification oracle
ANALYSIS.md        review of the original code + rewrite plan
```

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest                                     # 34 tests, ~6 seconds
```

## Reproduce all results

```bash
python scripts/generate_results.py         # data/ — about 2 minutes
python scripts/generate_heatmaps.py        # results/heatmaps/
python scripts/plot_endgame.py             # results/endgame/
```

## Estimate your own σ

Manual interactive tool that loads a dartboard photo, lets you calibrate scale (click centre + outer-bull ring) and click each hit, then fits a 2D Gaussian:

```bash
python scripts/click_hits.py path/to/your/photo.jpg
```

(Eventually this lives on a website — for now the desktop tool is the v1.)

## Results: best aim by player skill (FFT solver)

The σ values are in normalised board units (multiply by 340 mm for millimetres). The tiers and their literature anchors are documented in [`ANALYSIS.md`](ANALYSIS.md) §10.7.

Axis convention (matches Tibshirani 2011 and Haugh & Wang 2022/2024): **x = horizontal** (+x toward the 6 wedge, right), **y = vertical** (+y toward the 20 wedge, up). Both normalised to [-0.5, 0.5].

| Player level | σ_x, σ_y | mm | best aim (x, y) | EV / dart | V(301) simple | V(501) official |
|---|---|---|---|---|---|---|
| `perfect`        | 0, 0          | 0    | T20 (any pixel) | **60.00** | **6.0** | **9.0** |
| `world_champion` | 0.015, 0.015  | 5    | (0.000, +0.303) — T20 | 42.62 | 7.7 | **13.1** |
| `excellent`      | 0.02, 0.02    | 6.8  | (-0.001, +0.303) — T20 | 37.18 | 8.8 | — |
| `good`           | 0.07, 0.07    | 23.8 | (-0.114, -0.296) — T19 region | 16.17 | 19.8 | **40.5** |
| `average`        | 0.09, 0.15    | 30, 51 | (-0.255, -0.042) — near T11 | 13.52 | 25.3 | — |
| `bad`            | 0.20, 0.20    | 68   | (-0.076, -0.028) — near bull | 12.04 | 33.1 | — |

The lower three tiers are anchored on Tibshirani et al.'s empirical measurements (beginner σ≈65 mm, skilled amateur σ≈27 mm). `world_champion` (5 mm) matches Tibshirani's "perfect" pedagogical reference and the σ implied by peak-PDC pros' ~40% T20 hit rate; `perfect` (σ=0) is the mathematical upper bound — its EV map is literally the dartboard score field, and the 501 game closes in the canonical **9-darter**.

The optimum migrates smoothly from triple-20 → triple-19 → bull as σ grows; see [`results/endgame/optimum_trail.png`](results/endgame/optimum_trail.png) and the V(s) overlay in [`results/endgame/v_curves_official.png`](results/endgame/v_curves_official.png).

## State of the art

Cross-checks against the reference code (Monte-Carlo solver from 2022), and consistent with:

- Tibshirani, Price & Taylor, [*A Statistician Plays Darts*](https://www.stat.cmu.edu/~ryantibs/papers/darts.pdf) — same single-dart EV findings.
- Haugh & Wang, [*Play Like the Pros? Solving the Game of Darts as a DP*](https://arxiv.org/pdf/2011.11031) — extends to the adversarial multi-turn game.
- [DataGenetics, *Throwing Darts*](https://www.datagenetics.com/blog/january12012/index.html) and the [CodeProject Monte-Carlo article](https://www.codeproject.com/Articles/461044/Throwing-Darts-in-Monte-Carlo) for visual comparisons.

The original work used Monte Carlo over a 51×51 aim grid; the new solver computes the same quantities exactly on a 1024×1024 grid via Gaussian convolution. Verification: the new `darts.board.get_score` matches the reference's pixel-by-pixel on a 200×200 grid (`tests/test_board.py`); the FFT EV agrees with 1M-sample Monte Carlo for six (aim, σ) configurations (`tests/test_fft_vs_mc.py`); the end-game DP agrees with the reference's pickled 301-game V values within Monte Carlo noise (`tests/test_endgame.py`).
