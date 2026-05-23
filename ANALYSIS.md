# Dart Game Optimization — Repo Review & Rewrite Plan

> Working doc. Read top-to-bottom, then answer the questions at the bottom inline. Once we converge, I'll execute. Nothing has been changed in the repo yet beyond creating the `cleanup` branch.

---

## 1. What the repo does today

Three Python files, ~600 lines total:

- **`aiming_spots.py`** — board geometry + Monte Carlo expected-value per aiming spot.
  - `get_score(x, y)` returns the dartboard score at a point. Coordinates normalized to `[-0.5, 0.5]`, board diameter 340 mm.
  - `ev_area(x, y, sx, sy)` draws `size**2` samples from `N((x,y), diag(sx², sy²))`, averages the score → estimate of `EV(x, y)`.
  - `explore_ev_area(...)` does this on a `points_per_line × points_per_line` grid (default 51×51 = 2601 aim points), pickles the result.
  - `optimal_for_player(...)` runs the full pipeline for a given (σx, σy) and dumps a bunch of PNGs.
- **`end_game.py`** — backwards DP from 301 down to find the optimal aiming spot for each remaining-score state.
  - For each aim spot, Monte Carlo estimates a discrete probability distribution over single-throw outcomes (`proba_area`).
  - For each `current_score`, computes `EV(spot)` = expected number of throws to close. Uses an infinite-series approximation for the bust-loop.
  - Treats one-throw-at-a-time (no 3-dart turn, no double-out rule).
- **`scores.py`** — converts a list of running totals into an average throw value, used to back-out an "equivalent σ" for a real player.

Plus: README with results for 4 player tiers, `opti_shots_good_player.md` (301 images), pickle files, ~26 PNGs.

---

## 2. Is the work correct?

**Yes, with caveats.** Cross-checked against the Tibshirani paper and the DataGenetics post; the qualitative findings line up:

- The optimal aim point for an excellent player (σ ≈ 0.02) sits near triple-20.
- It drifts down/inward as σ grows; for σ ≈ 0.07 it's around (-0.3 to -0.06, ~0.1–0.24), i.e. near triple-19 / triple-16 area.
- For σ ≥ 0.15 it collapses toward the bull region.

That matches the published consensus, so the result is **directionally sound**.

The end-game curves (number of throws-to-finish vs current score) also have the expected shape: small spike at score 1, smooth decrease toward 0 as you approach 301, with the score=2 case requiring extra throws (only double-1 closes if you're enforcing "must finish on a double" — but you don't enforce that, see §3).

### Where the numbers are noisy

This is a **Monte Carlo + coarse grid** estimate. With `size=100` (10 000 darts per aim point) and a score range of 0–60:

- Per-spot standard error of the mean ≈ `σ_score / √10000` ≈ 0.13 score points.
- The difference between the best and ~10th-best aim points is often well under 1 point.

So the **claimed "best aim spot" is partly noise**. Re-running with a different random seed will pick a different but nearby spot. The 51×51 grid (≈ 6.7 mm spacing) also under-resolves the triple ring (8 mm wide). The published optima are correct *to within the noise/grid floor*, but not pinpoint.

---

## 3. Bugs / issues found

### Real bugs

1. **`traceback.format_exc(e)`** — `aiming_spots.py:241` and `end_game.py:93,105`. `traceback.format_exc()` does not take an exception argument; the first positional arg is `limit: int`. Passing an `Exception` object either raises a `TypeError` (newer Python) or silently produces wrong output. Currently masked because the `try` block only fails when the pickle is missing — first run prints garbage / errors out instead of "no cache, computing". Fix: `traceback.print_exc()` or `print(repr(e))`.

2. **`create_empty_proba_dict()` vs `get_empty_proba()`** — both exist, only the second is used. The first is dead but inconsistent (it includes 0 but enumerates differently). Easy to delete one.

3. **`scores_ev_and_pos` initial seed is wrong** — `end_game.py:94` initializes with `{301: [0.0, 'N/A']}`. The DP loop in `__main__` iterates `current_score = 300 - i` from 300 down to 0, so the first solved score is 300, then 299, etc. The recursion needs `EV(301) = 0` (game over), which is in the dict. Good. But the dict is also missing entries for `>301` that should never be queried — that's fine, but if scoring 0 logic ever changes, this is fragile. Not a bug, just brittle.

4. **`approximate_spot_ev` truncation** — `end_game.py:186`. The infinite series `Σ pf^i · (i+1) = 1/(1-pf)²` has a **closed form**. The current code sums to `depth=9`. For `pf < 0.6` truncation error is small (~0.15 throws at the boundary) but it's unnecessary; the closed form is both faster and exact:

       EV = (1 + Σ_v P_v · EV[score+v]) / (1 - pf)

   The current code's overall formula *is* algebraically correct — I verified — but truncation is avoidable.

5. **`pf > 0.6` returns `1000`** — same function. This is a workaround for the truncation blowing up. With the closed form, this hack disappears.

### Modeling simplifications worth flagging on the video

- **One throw at a time, no turn structure.** Real 301/501 is played in 3-dart turns; busting in a turn rewinds the whole turn. The code's "bust = stay" approximation lower-bounds the true throw count. The 2022 paper (Haugh & Wang, arxiv 2011.11031) handles this properly with the adversarial component.
- **No double-out rule.** Standard 301/501 requires the final dart to land in a double (or bull). Massively changes the end-game strategy below score ~50.
- **Diagonal covariance only.** No off-diagonal ρ term; a real player's miss pattern is often correlated (e.g., trembling on one axis).
- **Score 0 = bust collapsed.** "Astuce!" comment in `end_game.py:140`. Algebraically fine *for throw count*, but if you ever wanted to compute "expected darts wasted off-board" that conflation hides it.

### Code cleanliness

- Mixed responsibilities: simulation, plotting, file IO, and CLI all in `__main__` blocks.
- Pickle files in repo root + an `img/` of 320+ PNGs. The repo is ~4 MB and the binary blobs are version-pinned to one specific σ. Should not be in the main tree.
- `2022-12-25-*.png` files in root — same.
- No tests, no requirements file, no entry point. `if __name__ == '__main__'` blocks are commented-out experiment scratch.
- Inconsistent coordinate convention (`x+ is up, y+ is left`) is documented in one place and then matplotlib has its axes flipped in plotting code. Works, but fragile.

None of this is *wrong* — it's vacation code, as you said. But for a teaching video and a public-facing repo, it's worth a real cleanup.

---

## 4. The big algorithmic win (independent of language)

**The single most impactful change is replacing Monte Carlo with FFT-based convolution.** This is also how the cited papers do it.

The expected score when aiming at `(x, y)` with a 2D Gaussian miss model is:

    EV(x, y) = ∫∫ S(x', y') · G_σ(x' - x, y' - y) dx' dy'

i.e. the **convolution of the score field S with the Gaussian kernel G_σ**. Rasterize S once at, say, 1024×1024 resolution; convolve via FFT — and **every aim point's EV pops out at once** in a single 2D operation. ~50 ms in NumPy/SciPy. No sampling noise. Same convolution machinery handles the per-value probability maps for the end game (one convolution per possible outcome value, ~62 of them).

Practical impact:

| Quantity              | Current (MC + grid) | FFT convolution      |
|-----------------------|---------------------|----------------------|
| EV map for one σ      | minutes, noisy      | ~50 ms, exact†       |
| Full end-game DP      | hours               | seconds              |
| Sweep over σ          | impractical         | tens of seconds      |
| Resolution            | 51×51               | 1024×1024 easily     |

† exact up to grid quantization; sub-pixel accuracy is easy with up-sampling or bilinear weighting.

This single change matters more than the language choice. **It's also pedagogically beautiful** for the 3B1B-style video: "the score function is a static image, your aim uncertainty is a Gaussian blur, expected score is just the blurred image."

---

## 5. Language recommendation

You can read C, C++, Python. Here are the realistic options ranked by my preference for this project:

### Option A — **Rust + Python visualization layer** *(my recommendation)*

- Hot core in Rust: `ndarray` + `rustfft` (or `realfft`) for the convolutions, plain loops for the DP. ~10–30× faster than equivalent Python; ~2–5× faster than naive C for this kind of work because the vectorization is easier to get right.
- Trivial Python interop via either (a) writing `.npy` files that Python reads with `numpy.load`, or (b) `pyo3` bindings if you want to call directly. **(a) is simpler and we should start there**.
- Cargo, `cargo test`, `cargo bench`, `cargo fmt`, `clippy` — the tooling is honestly the single best part. No Makefile-and-tears situation.
- Memory safety means no segfault hunting in our brute-force loops.
- Mild downside: you said you read C/C++/Python, not Rust. **But** if you can read C++, Rust will be readable within an afternoon for this domain — it's mostly `Vec`, `for`, structs, and a couple of borrow markers.

### Option B — **C++ with Eigen + FFTW** *(pragmatic, familiar)*

- Performance comparable to Rust.
- You read C++ already. Zero ramp-up.
- Bigger ergonomic overhead: CMake, linking FFTW, header soup. The fight is the build system, not the math.
- If we end up wanting to ship this as a teaching artifact, the build complexity hurts contributors.

### Option C — **Plain C with FFTW + a tiny utility lib**

- Maximum portability, maximum "look at this beautiful tight loop" pedagogy.
- For a video about brute-forcing dartboard math, there's something charming about C.
- More boilerplate (memory management, allocations for FFT plans, etc.).
- Slower to iterate than Rust/C++ in practice.

### Option D — **Stay in Python, use NumPy/SciPy + Numba**

- `scipy.signal.fftconvolve` does the heavy lifting.
- With Numba JIT on the DP loop and vectorized NumPy for the rest, **this is likely already fast enough** for the video work.
- Zero new language, zero new build system, zero interop story.
- **Worth considering if the goal is the video, not the engineering exercise.** If the goal is also "have a fast standalone implementation as a side project", then A/B/C.

### My honest take

If you're enjoying the rewrite for its own sake → **Rust**. If you just want the video done fast → **Python with FFT + Numba**. The middle ground (C++) is fine but I think Rust is strictly better unless you have a reason to prefer C++.

We can also do **both**: keep Python as the reference / verification / plotting layer, and put the production solver in Rust. That's the setup I'd lobby for.

---

## 6. Proposed repo layout (after cleanup)

```
dart_game_optimisation/
├── README.md                    (rewritten, points to results and the video)
├── ANALYSIS.md                  (this file, maybe deleted after we converge)
├── reference_python/            (the old code, kept as oracle for tests)
│   ├── aiming_spots.py
│   ├── end_game.py
│   └── scores.py
├── solver/                      (the new fast implementation)
│   ├── Cargo.toml               (or CMakeLists.txt depending on choice)
│   └── src/...
├── viz/                         (your visualization scripts)
│   ├── heatmap.py
│   ├── histogram.py
│   ├── optimal_drift.py
│   └── ...
├── data/                        (output of solver, input to viz, .npy files)
│   └── .gitkeep
├── results/                     (the published curated outputs that go in the README)
│   └── ...
├── tests/                       (cross-language verification)
│   └── verify_python_vs_solver.py
└── docs/
    ├── algorithm.md             (the math, suitable for the video script)
    └── stateofart.md            (papers, comparisons)
```

Things to evict from the main tree (move to a `legacy/` branch or just remove with history preserved on this branch):

- Root-level PNGs (`2022-12-25-*.png`)
- The 301-file `img/optimal_shot_good_player/` directory — regenerate from the solver instead
- Pickle files at root — replace with `.npy` produced by the solver

I want to be careful not to delete history. Plan: move files in commits with `git mv` where applicable; nothing destructive.

---

## 7. Data exchange between solver and Python viz

You mentioned "maybe just through files". Yes, I think that's right. Concretely:

- Solver writes **`.npy` files** (NumPy native binary format, ~10 lines of code to write in any language; one-liner to read in Python via `np.load`).
- Pair each `.npy` with a small **JSON sidecar** that describes the contents: σx, σy, grid resolution, board dimensions, what's stored (EV map vs probability cube vs DP table), units. That way the Python viz can introspect and label plots correctly.
- Naming convention: `data/{kind}_{sx}_{sy}_{res}.{npy,json}`.

If we later want sub-second interactive sliders, we can add a small `pyo3` (Rust) or `pybind11` (C++) binding layer to call the solver directly. **But files first, bindings only if needed.** Files are debuggable, language-agnostic, and let you save canonical results to disk.

For a future "scrub a slider over σ" interactive demo, the most pragmatic path is: pre-compute a grid of (σx, σy) → EV-map at startup (a few seconds), then the slider just indexes into the cube. No live solver calls needed.

---

## 8. Visualization ideas (your domain, but jotting down hooks)

Things the solver should make easy to emit:

1. **EV heatmap** at a chosen σ over the board (your standard chart).
2. **Per-aim-spot score distribution** — histogram of P(score | aim at p, σ). For the optimal p, and for canonical "centre", "triple-20", and a couple of comparison points. **Your new visualization idea** — and yes it's a nice one. Stacked histogram or stem plot, color-coded by single/double/triple zones.
3. **Optimal-aim trail** as σ varies smoothly from 0.005 to 0.30. Animate as a moving dot over the board; trail colored by σ. Probably the most striking single visual.
4. **3D / contour of EV** — same data as the heatmap but as a surface, showing how flat or peaky the optimum is. Critical for explaining "why the optimal aim is fragile near the triples and stable near the centre".
5. **End-game heat ribbon**: a `[score × aim_position]` matrix colored by optimal action. Or simpler: for each current score, draw the recommended aim spot on a tiny board, then tile 301 of them.
6. **Comparison against canonical strategies**: bar chart of `EV(center) vs EV(triple-20) vs EV(optimal)` as a function of σ. Shows the cross-over points cleanly.
7. **Real-player overlay**: take your `points_g` / `points_r` data, derive an MLE σ instead of just average-throw, and overlay them on the strategy chart. Connects the math back to "your friend Robert".

I'd suggest the solver emits a single "study bundle" `.npy` cube indexed by σ, plus per-σ probability cubes. Then all the above plots come from manipulating those arrays in Python.

---

## 9. Math content / state of the art (for the video)

Bullet list of things worth mentioning, in roughly the order they'd appear in the video:

- **The game** — board geometry, scoring, why doubles/triples are where they are (note: the layout is *anti-clustering* — high and low values alternate around the ring, which makes it punishing to miss).
- **The model** — 2D Gaussian miss; what σ corresponds to what player level (great pedagogy hook: show real scatter data).
- **The convolution insight** — score field ⊗ Gaussian = expected score. Beautiful, visual, immediate.
- **The optimum drifts** — animate the trail. Explain *why*: as σ grows, missing triple-20 costs you the bull-of-1-and-5 neighbours, so safer regions take over.
- **End-game DP** — show the recursion. Highlight where it deviates from "just aim at the optimum every throw" (e.g. score 41 wants triple-19, not bull, because of finishing logic).
- **Where you stop** — Tibshirani et al. (single throw), Haugh & Wang (adversarial), DataGenetics (visual), this work (single throw + end-game). The next step is the full multi-turn adversarial game from the 2022 paper.
- **Caveats** — turn structure, double-out, correlated misses. All within reach as v2.

---

## 10. Decisions (locked in)

1. **Language** — Stay in **full Python**. FFT-based core via `scipy.signal.fftconvolve`. If it turns out not fast enough later, the clean Python interface gives us an easy path to rewrite the hot core in C/C++/Rust.
2. **Game rules** — Use the **real rules: 501, double-out, 3-dart turn structure**. The simple "single-throw expected value" study doesn't need these and stays as-is for pedagogy. The end-game / multi-turn results use the real rules.
3. **Covariance** — Diagonal Σ for v1, using the existing σ values (0.02, 0.07, 0.15/0.09, 0.2) so results are directly comparable to the reference. Full Σ comes later, but only with *realistic* values — see the new roadmap item below.
4. **History cleanup** — Don't purge. Move files within the working tree if useful, but keep git history intact.
5. **Repo layout** — Roughly §6, lightly simplified. Reference code stays in `reference_python/` as the verification oracle.
6. **Data format** — `.npy` + `.json` sidecar. Can change later.
7. **Scope of cleanup** —
   - **Do not modify the reference code.** Bugs and all, it stays as-is until the new code's numerics match it. Otherwise we can't tell whose bug a discrepancy belongs to.
   - Add `pyproject.toml`, make the new solver pip-installable.
   - Short docstrings only.
   - Rewrite the README — but not until the new code works and a few representative visualizations are in hand.
   - **Priority: verify numerics first, then ship heatmaps + optimal-spot visuals on the existing σ values, then move on.**
8. **End-game trust** — Trust the existing reference results as a regression target. State-of-the-art papers (Tibshirani, Haugh & Wang, DataGenetics) are the secondary cross-check.

### New roadmap item — covariance estimator from clicked points

Once the basic solver + visualizations are in hand, build a small Python tool that:

- Loads a photo / scan of a dartboard with hits on it.
- Lets the user **calibrate scale** by clicking known landmarks (centre + a known ring).
- Lets the user **click each hit** to record (x, y) in board coordinates.
- Fits a 2D Gaussian (mean, covariance) to the clicks via MLE.

v1 is a desktop Python script (matplotlib clicker is fine). Eventually wraps into a small website where anyone can upload their cardboard. CV-based hit detection is explicitly **not** required for v1 — manual clicks are the contract.

---

## 10.5  What is a "probability cube"? (added during implementation)

For the single-throw EV problem, **one** map per σ is enough: `EV[i, j]` = expected score when aiming at pixel `(i, j)`. That's a single 2D image.

For the **end game**, knowing only the *average* score isn't enough — the optimal strategy depends on the entire **distribution** of possible single-dart outcomes. From score 41, aiming at triple-19 might give a higher average score than aiming at the bullseye, but a *worse* probability of hitting *exactly the right value to set up a double-out*. We need to know, for every aim point, the probability of each possible outcome.

So we build, for a given player σ, a 3D array:

    P[k, i, j]  =  probability of outcome k  when aiming at pixel (i, j)

where `k` indexes the discrete set of single-dart outcomes the dartboard can produce:

- 1 miss (score 0, off-board)
- 20 singles (S1…S20, scores 1…20)
- 20 doubles (D1…D20, scores 2…40, **flagged as doubles** for the double-out rule)
- 20 triples (T1…T20, scores 3…60)
- outer bull (25, single)
- bullseye (50, counts as a double for finishing)

That's ~63 outcomes total. Stacking the 63 probability maps gives an array shaped like `(63, N, N)` — a **cube** (3D block) of probabilities. Hence the name.

Computing each layer is, again, a Gaussian convolution: for outcome `k`, define a binary mask `M_k[i, j] = 1` iff pixel `(i, j)` lies in the region that produces outcome `k`. Convolve `M_k` with the Gaussian kernel `G_σ` and you get `P[k, ·, ·]`. So the cube is ~63 FFT convolutions — still fast.

Sanity properties the cube must satisfy:

- `Σ_k P[k, i, j] = 1` for every `(i, j)` (the masks tile the plane).
- `Σ_k value[k] · P[k, i, j] = EV[i, j]` (the cube is consistent with the direct EV map).

Both become unit tests.

Why the cube enables the end-game DP: given the cube, computing the expected outcome of *any* score-dependent quantity for a given aim point is a single dot product over the `k` axis. The end-game DP repeatedly asks "for this current score, what's the best aim point?" — and answering that becomes 63 multiply-adds per aim point per state, vectorized across all `N²` aim points in one tensor contraction.

---

## 10.7  Player accuracy: literature anchors and tier choices

> Added after a literature review of how real player accuracy is modeled and measured.

### Is a 2D Gaussian a reasonable model?

Yes — it's the canonical one in published dart-optimization work. The key references:

- **Tibshirani, Price & Taylor (2011)** introduce the model `Z = μ + ε, ε ~ N(0, Σ)` and explicitly compare a Gaussian against a skew-Gaussian extension; the simple Gaussian is empirically adequate. See https://www.stat.cmu.edu/~ryantibs/papers/darts.pdf §3-4.
- **Haugh & Wang (2022)** use the same bivariate normal for the full multi-turn 501 DP problem. For pros they allow Σ to depend on the *target region* (a pro's σ at T20 is materially smaller than at T17 because they practice T20 much more): https://arxiv.org/abs/2011.11031.
- A modern walkthrough (Michael Cole, https://mjc239.github.io/maximising-single-dart/) reproduces the comparison and concludes "the added complexity does not provide much additional value over the Gaussian model."

**Two non-obvious points from the literature:**

1. **Anisotropy is real and biomechanical.** Tibshirani measured σ_x = 17.9 mm vs σ_y = 39.1 mm for one skilled amateur (x = horizontal, y = vertical; standard literature convention) — vertical scatter ~2× horizontal. This is mechanistic: "it is common for most players to have a smaller variance in the horizontal direction than in the vertical one, since the throwing motion is up-and-down." Our `average` tier at (σ_x=0.09, σ_y=0.15) — vertical/horizontal ratio 1.67 — sits in the right axis and right ballpark.

2. **Off-diagonal ρ is mostly unidentifiable from score data.** Haugh & Wang's follow-up (https://arxiv.org/abs/2302.10750 §7.1) flags this — without raw landing positions, you can't recover correlation. A diagonal Σ with σ_x ≠ σ_y is therefore the **recommended model**. Our covariance estimator (`darts.covariance`) does have access to raw positions (clicked from the photo) so it *can* fit a correlated Σ, but the solver consumes a diagonal Σ.

### Published σ values vs our tiers

We use the standard convention from the literature: **x = horizontal** (+x toward 6), **y = vertical** (+y toward 20). All σ values below are in this convention.

| Source | Player | σ_x (mm) | σ_y (mm) |
|---|---|---|---|
| Tibshirani §4 example 1 | beginner statistician (100 darts at bull) | 42.7 | 68.6 |
| Tibshirani §4 example 2 | skilled amateur | 17.9 | 39.1 |
| Tibshirani §3 reference | "perfect" pedagogical example | 5 | 5 |
| Haugh & Wang 2022 | top-16 PDC pros at T20 (inferred from T20 hit rate ~40%) | ~6–10 | ~5–8 |

In our normalized board units (divide mm by 340):

| Our tier | σ_x, σ_y (norm) | σ_x, σ_y (mm) | Closest published anchor |
|---|---|---|---|
| `perfect` | 0, 0 | 0, 0 | Mathematical ideal (no measurement) |
| `world_champion` | 0.015, 0.015 | 5, 5 | Tibshirani §3 reference; PDC pro at T20 |
| `excellent` | 0.02, 0.02 | 6.8, 6.8 | Strong pro / club champion |
| `good` | 0.07, 0.07 | 23.8, 23.8 | Tibshirani's "skilled amateur" (σ=26.9 mm isotropic) |
| `average` | 0.09, 0.15 | 30, 51 | Anisotropic amateur; ratio matches Tibshirani's σ_y/σ_x ≈ 1.7 |
| `bad` | 0.20, 0.20 | 68, 68 | Tibshirani's beginner (64.6 mm isotropic) |

**Caveats to communicate in the video:**

- Our `excellent` is already pro-territory; it's mislabeled if "excellent" suggests "an excellent amateur". For pedagogy it's fine because it sits at the breakpoint where T20 starts being a credible aim.
- Our `average` is closer to **novice** than to a median pub player (probably σ ≈ 25–35 mm). A future `casual` tier between `good` and `average` would fill that gap.
- The σ < 17 mm → aim T20 / σ > 17 mm → aim T19 breakpoint (Tibshirani §5) is a clean teaching anchor — it falls between our `excellent` (6.8 mm) and `good` (23.8 mm) tiers.

### How σ is measured in practice

Two methods in the literature, both rest on the same MLE:

1. **From recorded positions** (what our `covariance.fit_gaussian` does): given clicked (x, y) hits, `Σ = (1/n) Σ Z_i Z_i^T`. Closed form. Requires ~30+ throws aimed at the same target.
2. **From scores only**: Tibshirani's EM algorithm with closed-form M-step and importance-sampling E-step. Less informative because s(Z) is many-to-one, but it works without raw positions. R package referenced at https://www.stat.cmu.edu/~ryantibs/darts/.

For a pro, sigma is *target-dependent* (smaller at T20 because that's the practice target), so a fair "pro σ" is fit on throws at the player's primary target, not at the bullseye.

### Two new solver tiers

- **`perfect`** (σ = 0): the dart lands exactly where you aim. EV map degenerates to the score field. Optimal aim is anywhere in T20. V(501) = 9 (the canonical perfect game: T20×3, T20×3, T20+T19+D12). Useful as the upper bound.
- **`world_champion`** (σ = 0.015 ≈ 5 mm): peak van Gerwen / Phil Taylor territory. EV ~58 per dart, V(501) approaches single-digit-turns.

---

## 11. Order of operations

Each step ends in a verifiable artifact. Short commits along the way.

1. Set up the new Python package skeleton (`src/` layout, `pyproject.toml`).
2. Port `get_score` to a new vectorized module; test for exact agreement against `reference_python/aiming_spots.get_score` over a dense grid.
3. Implement FFT-based EV computation.
4. **Verification gate:** for several (aim, σ) points, confirm FFT EV agrees with reference Monte Carlo within MC standard error.
5. Generate heatmaps + optimal-spot plots for the existing σ values; visually and numerically compare against the reference pickles. **Pause here for your review.**
6. Implement per-value probability cubes via FFT.
7. Implement the 501 end-game with double-out + proper 3-dart turn structure. Run a 301 / one-throw-at-a-time mode internally to validate against the reference 301 pickle, then enable real rules.
8. Standardize `.npy` + `.json` output layer for the viz layer.
9. Move reference code into `reference_python/`. Rewrite the README. Open the door to the covariance-estimator tool.

The reference code is the oracle until step 5 passes.
