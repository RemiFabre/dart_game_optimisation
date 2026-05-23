# Blog outline — Solving the game of darts

> Working outline. Each section title is followed by a short description (what to cover, in 2–4 sentences), then a bullet list of **visualizations** to make. Two sections are flagged as **DETAILED**: the player-accuracy section is being written first (in `blog/01_player_accuracy.md`), the probability-cube/end-game section comes next.
>
> Working-document conventions:
> - Citations go inline with a numeric reference and a hyperlinked footnote (so the blog reads cleanly online and PDFs in `papers/` are the canonical sources).
> - Where a number comes from our own solver, we say so explicitly and link to the script that produced it.
> - Where a choice is intuition or convenience rather than literature, we flag it as such ("we choose X for simplicity; a stronger model would do Y").

---

## §1 — Why darts? What we're doing, what's new

Frame the question: *given a player's accuracy, where should they aim?* That's the surface question; under it sit several hard sub-problems (modelling accuracy, integrating over uncertainty, sequential decision making). State the elevator-pitch answer (you shouldn't aim at triple-20 unless you're very good) and the structure of the rest of the post. End with what's contributed here vs prior work: an exact FFT solver, a clean 501-with-double-out DP, and a free covariance-estimator tool.

**Visualizations**
- Hero figure: a 2×3 grid of EV heatmaps, one per skill tier (perfect → beginner). One image that summarises the entire surface result.
- A side-by-side "intuition" panel: a player aiming at T20 with a small σ (lots of T20s) vs the same aim with a large σ (mostly S20, S5, S1, lots of edge-rim misses). Same target, two completely different score distributions.
- Optionally: a small animated GIF of the "optimum point" walking across the board as σ slides from 0 to 0.25 (this is `results/endgame/optimum_trail.png` but animated).

---

## §2 — The dartboard

Board geometry, scoring rules (singles/doubles/triples, two bull regions), and the *anti-clustering* number layout (20 next to 1 and 5). The anti-clustering point is pedagogically important — it's the punishment-for-missing structure that creates the whole optimization problem; on a 1–20 sequential board the answer would always be "aim at 20".

**Visualizations**
- Annotated board with all measurements (bull diam, triple ring width, etc.) — could be done as a clean SVG or a labelled matplotlib figure.
- Anti-clustering visualization: a "rearranged" board where numbers are placed sequentially 1–20, side by side with the real layout, both colored by the neighborhood-average score. The contrast makes the design choice obvious.
- A "what does a dart score" decision tree (or simpler: an annotated cross-section showing how distance from center + angle map to score).
- Historical note: where the layout came from (Brian Gamlin attribution).

---

## §3 — Modelling the player **[DETAILED — written in `01_player_accuracy.md`]**

This is the foundational modelling section. Why Gaussian, what does the covariance matrix mean, what does "diagonal" vs "symmetric" vs "full" mean, what σ values are realistic for what skill level, the X/Y orientation convention, and the limits of the model (chief among them: target-dependence). Anchor every numerical claim to a published source.

**Visualizations** (see the section file for the full list)
- Annotated dartboard with the (x, y) axis convention drawn on it.
- A panel of Gaussian contour plots for the six tiers, all on the same axes, all aiming at the bullseye.
- A "real darts" Gaussian fit: take Tibshirani's tabulated scatter or our own click-tool output and overlay the 1σ / 2σ ellipses on a photo.
- A "target-dependent σ" figure if the data supports it (per-region σ for one of Haugh & Wang's PDC pros).
- A "ladder of skill" infographic: σ in mm, the matching skill descriptor, the corresponding 3-dart average if you aim where the solver says.

---

## §4 — The single-throw expected value problem

Define EV(aim point | σ) formally as an integral against a Gaussian. Set up the integral, observe that it's a convolution of the score field with the Gaussian kernel. State that this is the central mathematical move of the whole post. Brief — most of the work is later.

**Visualizations**
- The score field rendered as a heatmap (continuous, not just "the board outlined"). This is what we feed into the convolution.
- A diagram showing "aim ⊕ Gaussian = sampling distribution"; arrow from aim point + scatter cloud → expected score.
- The 2D Gaussian kernel itself rendered for the six tiers (small inset images).

---

## §5 — Computing EV via FFT convolution

The numerical heart of the paper: convolution = pointwise product in Fourier space, so one 2D FFT of the score field, one 2D FFT of the Gaussian kernel, one multiply, one inverse FFT, and you have **the EV for every aim point at once**. This is the "1000× speedup" Tibshirani's brute-force solver doesn't have. Verify it agrees with Monte Carlo (link to our test). Discuss grid resolution and the leak-to-MISS correction.

**Visualizations**
- A four-panel figure: (a) score field S(x,y); (b) Gaussian kernel G_σ(x,y); (c) Fourier magnitudes |Ŝ|, |Ĝ|; (d) result EV(x,y) = (S ⋆ G_σ)(x,y). Self-evidently a convolution.
- A convergence plot: |EV_FFT − EV_MC| as a function of MC sample count, showing the FFT is the limit.
- A timing comparison bar chart: per-(σ, resolution) seconds, FFT vs MC at matched statistical accuracy.

---

## §6 — Single-throw results: where to aim

The classic result: the optimum migrates from triple-20 (for excellent players) through triple-19 (good players) toward the bull (weak players). Walk the reader through the heatmaps, compare with the previous published findings (Tibshirani, DataGenetics). Comment on the *sharpness* of the optimum — at low σ the EV surface has narrow peaks, at high σ it's a broad plateau (this matters for whether tiny aim errors matter).

**Visualizations**
- The full set of six EV heatmaps stacked.
- The "optimum trail": a single board with a sequence of dots colored by σ, showing how the best aim walks. (`results/endgame/optimum_trail.png` — already generated.)
- An "EV cross-section" — pick the line y = 0 (vertical line through the bull and T20) and plot EV(x, 0) for each tier. One chart, six curves, shows the peak structure clearly.
- EV(σ) at canonical aim points: a line plot with x = σ (log scale), one line each for "aim at bull", "aim at T20", "aim at T19", "aim at optimum"; the crossing points are where strategy should change.

---

## §7 — The end game **[DETAILED — to be written next, in `02_endgame.md`]**

The 501 game with double-out and 3-dart turns. State the rules cleanly. Define the value function V(s) = expected throws to finish from current score s. Introduce the **probability cube** as the central data structure (one 2D probability map per single-dart outcome). Walk through how the cube + Bellman recursion + a fixed-point iteration for self-reference at busts give us the full solution. End with the result: V(501) for each tier, the optimal-aim-per-state visualization.

**Visualizations**
- Probability cube as a 3D stack: maybe a perspective render showing the 63 layers like slides of a microscope sample.
- For one player (good), pick one (s, darts_left) state and show the optimal aim heatmap.
- V(s) overlay for all tiers (already have it: `results/endgame/v_curves_official.png`).
- A "turn replay" animation: simulate a turn from s = 41 (a classic checkout score). Show the aim per dart, the outcome, the score after each. Across many such replays, show the distribution of turn outcomes.
- The "all checkouts" chart: for a given player, the optimal first-dart aim for every starting score in [2, 170] (the official checkout range). Twenty-six small boards in a grid, each marked with the optimal aim.

---

## §8 — Verifying the solver

How do we know any of this is right? Three layers of verification: (a) board geometry pixel-identical to the reference Python; (b) FFT EV ≈ Monte Carlo EV at high sample count; (c) end-game DP ≈ direct Monte Carlo simulation of the chosen policy. Mention the cross-check against Tibshirani's published numbers and against the original reference code's pickle.

**Visualizations**
- A QQ-plot or correlation plot of (FFT EV) vs (MC EV) over a grid of aim points.
- A bar chart showing per-tier the agreement between V(s) (solver) and V_empirical(s) (simulation).
- The reference Monte Carlo result laid alongside ours for the canonical good player.

---

## §9 — Estimating your own σ

Pitch the photo-clicker. Walk through one full example: take a photo of your board, click landmarks for scale, click each hit, get σ_x, σ_y, ρ. Discuss how many throws you need for a good estimate, and how to handle the fact that real photos have parallax / angle distortion.

**Visualizations**
- Annotated screenshot of the tool: click here, click there, here are the ellipses.
- A "how many throws is enough" plot: with synthetic data, estimate σ from N=10, 20, 50, 100 throws; show error bars on σ_x and σ_y; identify the knee.
- A real photo with overlaid 1σ and 2σ ellipses.

---

## §10 — State of the art and where we sit

Brief tour of the literature: Tibshirani et al. (the foundational paper, single-dart problem), Haugh & Wang (full 501 DP with the multi-turn adversarial game), DataGenetics (popular-science version with the same essential finding), the OptimalDarts repository (open data). State explicitly what's new in this post vs prior work, and what we don't do (the adversarial game).

**Visualizations**
- A timeline of dart-optimization research.
- A "venn" or feature-comparison table: which method handles single-dart vs end-game vs adversarial vs target-dependent σ.

---

## §11 — Limitations and future work

Honest list of things we don't do: target-dependent σ, full off-diagonal correlation, the adversarial (two-player race) game, sigma drift within a session (fatigue/concentration effects). Suggest the photo-clicker → solver → personalized strategy as the natural next step for any motivated reader. Open-source pitch: every script in the post is in the repo; people can fork and study their own data.

**Visualizations**
- (Optional) a small mock-up of what a personalised strategy card could look like — "for σ = X, your top 3 checkouts are: [...]".

---

## §12 — Conclusion

One paragraph. What you should remember if you only remember one thing.

---

## Cross-cutting visualization wishlist (brainstorm)

These don't fit a single section but could be woven through:

- **"Dart hit distribution histogram" per aim**: pick one aim and one player, plot the empirical histogram of the 63 outcomes, color-coded by region type (single/double/triple/bull/miss). This is your "new way of visualizing" idea — works particularly well as a comparison at a single aim point for three tiers.
- **Strategy contour over (σ, current_score)**: a 2D contour showing the *recommended outcome value* (e.g. "aim for 60 — T20" vs "aim for 57 — T19" vs "aim for 50 — bullseye") as σ and s vary. Captures the entire strategy in one image.
- **Per-turn "probability tree" for a given starting score**: from s=41, three darts, show all reachable subturn states with probabilities and the cumulative chance of closing the turn out. Becomes a story object for the video.
- **EV gradient field**: at each aim point, an arrow pointing in the direction of fastest EV increase. The optimum is where the arrows converge.
- **Confidence ellipses sample**: take the click-tool's output for several real-world players and lay them out as a "skill spectrum" panel.
- **3-darts-per-turn vs 1-dart simplification**: side-by-side V(s) curves to show how much the proper turn structure matters.
- **Cost of suboptimal aim**: heatmap of (EV_optimum − EV(p)), so you see *how much* you lose by aiming somewhere reasonable but wrong (e.g. always aiming at T20 when σ is too large).
- **σ vs 3-dart average**: a single relation chart anchoring our model in PDC commentary terms.

---

## Citation plan

A `papers/` directory holds the source PDFs. In the prose, citations are written as numeric references like `[Tibshirani 2011, §3]` with a footnote pointing to the local file and the original URL. A `papers/README.md` lists each file with one-sentence "what's in it".
