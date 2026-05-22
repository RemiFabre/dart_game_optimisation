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

## 10. Questions for you

Answer inline below each one.

1. **Re-implementation language** — A/B/C/D from §5. My vote: **A (Rust + Python viz)**, but D (stay in Python with FFT) is the move if the video is the only deliverable.
   - Your answer:

2. **Should the new solver model proper 3-dart turn structure and the double-out rule?** Both are doable, both improve realism, both increase complexity. The video can still cover just the simplified version. (My instinct: model the proper turn structure; double-out optional. The current one-throw-at-a-time model is a known simplification we should note.)
   - Your answer:

3. **Full covariance (correlated σ_x and σ_y)** — worth supporting from day one, or keep diagonal? FFT handles it natively, costs nothing. I'd say yes, support it.
   - Your answer:

4. **History cleanup** — happy to keep all the old PNGs and pickles in git history but move them out of the working tree? Or do you want to fully purge them via `git filter-repo`? I do **not** recommend purging history unless you have a strong reason.
   - Your answer:

5. **Repo layout** — does §6's proposed layout look right? Particularly: keeping `reference_python/` as the verification oracle.
   - Your answer:

6. **Data format** — `.npy + .json sidecar` OK? Or do you have a preferred convention from other projects?
   - Your answer:

7. **Scope of the cleanup pass** — should I also:
   - [ ] Fix the bugs in §3 in the reference Python code?
   - [ ] Add a `requirements.txt` / `pyproject.toml` to make it pip-installable?
   - [ ] Write proper docstrings + a single `__main__` CLI?
   - [ ] Rewrite the README to be cleaner and split the long results into a sub-page?
   - Your answer:

8. **End-game video angle** — do you want me to dig into whether the *current* end-game results are bug-free (re-derive a few cases by hand) before we rewrite? Or just trust them, rewrite, and verify against the new solver?
   - Your answer:

9. **Anything missing?** Anything in the work I haven't addressed that you want covered before we proceed?
   - Your answer:

---

## 11. What I'd do once you've answered

Order of operations (1 commit per chunk, short messages):

1. Pin reference Python in `reference_python/`, fix its bugs, add a smoke test.
2. Move PNGs / pickles into `legacy/` or out of the tree, update README references.
3. Stand up the new solver scaffold in the chosen language.
4. Port `get_score` and verify identical output to Python over a dense grid.
5. Implement FFT-based EV computation; cross-check against the Python Monte Carlo within Monte-Carlo error.
6. Implement the per-value probability cubes; cross-check the same way.
7. Implement the end-game DP; cross-check against the existing pickled end-game result.
8. Write the `.npy` / `.json` output layer.
9. Hand off to your viz layer; iterate on what plots you actually want.

Each step is independently verifiable against the existing Python, which is why I want to keep the Python as an oracle.
