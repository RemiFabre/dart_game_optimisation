# Reply to GoDartsPro founder

> Draft, not sent. Iterate freely. Tone: friendly, substantive, shows we've read the literature carefully and have something concrete to offer. The three-panel anisotropy figure is the visual punch.

---

## Subject

The covariance matrix is the part nobody has actually measured — and that's the interesting part

## Body

Hi [name],

Thanks for the link — that blog is a great writeup of the classic result, and yes, it's basically the same thing I had built initially. The model there (and in every published paper since Tibshirani–Price–Taylor 2011) is the 2D Gaussian with a single isotropic σ, i.e. a symmetric covariance matrix. It gets you the canonical "below σ ≈ 17 mm aim at T20; above that, switch to T19; for very large σ aim at the bull" picture, and it's the right starting point. But I think the actually interesting and undersolved part lives one level deeper, in the *shape* of each player's covariance matrix — and that's where I'd love your view.

The reason I'm convinced this matters is biomechanical. Throwing a dart is fundamentally an up-and-down motion: a release-angle error translates almost entirely to a *vertical* miss, while the much smaller horizontal component of the throw maps to *horizontal* miss. Both authors in Tibshirani 2011 (one weak amateur, one decent amateur) show σ_vertical / σ_horizontal ratios of 1.6 and 2.2 respectively in their full-Σ fits[^1]. Independently, Churron et al. (2014)[^2] dropped horizontal-error analysis from their motion-capture study of trained throwers because for them it was small enough to be negligible — i.e. trained anisotropy is at least as strong as amateur, not less.

What I genuinely don't know is **what happens to that ratio as you climb the skill ladder**. Two equally plausible stories:

1. Training reduces the vertical-bias more than the horizontal — so the ellipse becomes more circular at the top level. The mechanism would be that release consistency improves faster than grip / shoulder kinematics.
2. Training shrinks both axes proportionally — the *shape* of the error stays elongated, just the overall scale drops. Mechanism: the up-and-down motion is the dominant error generator at every skill level, just smaller in absolute terms.

You can probably tell from your data which one is true.

**Why does this matter for play?** Because the optimal aim point depends on the *shape* of the covariance, not just its overall size. I made a quick three-panel simulation — see attachment `fig_anisotropy_trails.png` — that holds the geometric-mean precision σ_geom = √(σ_x σ_y) constant and varies only the shape:

- **Isotropic**: classical Tibshirani migration, T19 → bull as σ grows.
- **Vertical-dominant (σ_y = 2 σ_x)**: vertical scatter dominates. The optimum drifts to a different family of aim points.
- **Horizontal-dominant (σ_x = 2 σ_y)**: now it's the *horizontal* scatter that limits you. The aim points migrate elsewhere again.

The qualitative point is that for the same overall accuracy, two players with different ellipse shapes should aim at genuinely different parts of the board. In the extreme — say a player whose vertical scatter is much larger than horizontal — it can become attractive to aim at a triple whose "long axis" is vertical (the 6 or the 11 wedge) even though they're worth a lot less than 20. Not because the 6 region scores higher, but because the 6 region's geometry forgives the player's error pattern.

**Concretely, here's what I think one could do with measured per-player Σ data:**

1. **Personalised optimal play for any specific game.** My solver currently handles official 501 (3-dart turns, double-out, bust rules); adapting to 301 or any other variant is a small DP change. Given a player's Σ, the solver outputs the EV-maximising aim for every state of the game.

2. **Coaching insight from the eigenvectors of Σ.** If σ_horizontal is unusually high relative to σ_vertical, that's a grip / shoulder / lateral-stability problem. If σ_vertical is the bigger one (the default case), it's a release-consistency / follow-through problem. The covariance matrix encodes *which axis to work on*, not just "you need to improve". A few hundred throws aimed at one target is all it takes to fit Σ — the `darts.covariance.fit_gaussian` function in our code does this with sample covariance directly, given (x, y) points.

3. **Pro-level strategy validation.** Here the published evidence is striking. Haugh & Wang (2024, IJOC)[^3] show that ignoring per-player skill structure costs ~5% of match win-probability on average across the PDC top-16, and up to 12% for Jonny Clayton specifically — and that's before you account for the per-*target* heterogeneity. They also show via simulation that a player with measurable skill differences across regions but who plays *as if* uniformly skilled loses ~70% of a best-of-35 match against a player with the same skills who exploits them correctly[^3]. So the question "do top pros actually play optimally?" is not a foregone conclusion — and it's empirically answerable with the kind of data your platform sits on.

**The data gap I keep running into.** What surprises me most is that even at the top level, **no one has publicly released measured (x, y) landing positions paired with a fitted per-player Σ**. Every academic paper, including Haugh & Wang's exhaustive 16-player PDC analysis, infers Σ via expectation-maximisation from *which region* each dart hit, not where exactly inside that region. We checked the data accompanying their paper directly[^4] — it's purely segment counts (e.g. Michael Smith aiming at T20: 3761 darts in T20, 4624 in S20, 203 in S15, 293 in S5, 58 in S3, 109 in S1). And Haugh & Wang's 2024 follow-up has a §7 result that I find quite striking: from score-only data, **the off-diagonal of Σ is fundamentally not identifiable**. They construct synthetic datasets with very different true correlations (ρ = −0.5, 0, +0.5) that produce identical score frequencies — the EM fits cannot tell them apart. So we don't even know whether the apparent per-region anisotropy in their fits is real or an artifact of the region's geometry.

The biomechanics literature (Lotze 2019[^5], Churron 2014, Morice 2013) has measured positions, but only in lab studies with small samples and only at group level — never with per-player Σ data, never on PDC professionals.

If GoDartsPro has anything in that direction — even raw position logs for a single named player over a few hundred throws — that's a missing piece of the published record. I'd be glad to (a) fit per-player and per-target Σ from such data; (b) feed it into the solver and produce that player's optimal-strategy heatmaps + 501 V(s) curve; (c) compare against what the player actually does in matches to estimate "value left on the table". Any subset of those that fits your situation works.

**One last thing that complicates everything**, which you've probably already encountered: pros don't have the same accuracy at every target. The published 16-PDC-pool numbers from Haugh & Wang are T20 hit rate 41.2%, T19 hit rate 41.7%, T18 36.9%, T17 33.5% — the T20 vs T17 gap of ~8 percentage points reflects pure practice effects. For an individual case the gap can be larger: van Gerwen hits T20 at 45.3% but T17 at only 30.2%. So a complete player skill model is really a per-region Σ, not a global one. Modelling this in the solver isn't hard — it's a per-target lookup — but it does need data at each of those targets.

Happy to share more from our end (the solver is open source, the blog is in draft form, both attached). And of course curious to hear your perspective — given how much landing-position data must be flowing through your platform, the limits of what's been done in academia look kind of surprising from the outside.

Best,
Rémi

---

[^1]: Tibshirani, Price & Taylor (2011), *A Statistician Plays Darts*, JRSS-A 174(1). https://www.stat.cmu.edu/~ryantibs/papers/darts.pdf — see §3 p. 7 for the biomechanical argument; §4 for the two-author full-Σ fits.
[^2]: Churron et al. (2014), *Two Types of Motor Strategy for Accurate Dart Throwing*, PMC3922883. https://pmc.ncbi.nlm.nih.gov/articles/PMC3922883/ — the methodology explicitly drops horizontal-error analysis as negligible at expert level.
[^3]: Haugh & Wang (2024), *An Empirical Bayes Approach for Estimating Skill Models for Professional Darts Players*, arXiv:2302.10750. https://arxiv.org/abs/2302.10750 — Table 3 p. 23 for the cost-of-misspecified-skill numbers; §5 for the heterogeneity match-loss claim.
[^4]: Direct inspection of `Raw_Data.xlsx` in the Haugh & Wang OptimalDarts GitHub repository (https://github.com/wangchunsem/OptimalDarts). The file has three sheets — a README, a Trebles sheet, and a Doubles sheet — all with integer counts per region; no (x, y) columns anywhere. Confirmed by reading the `Fitting_Data` field inside `ALL_Model_Fits.mat`.
[^5]: Lotze et al. (2019), *Is Imagery Better Than Reality? Performance in Real and Imagined Dart Throws*, PMC6520223. https://pmc.ncbi.nlm.nih.gov/articles/PMC6520223/ — reports group-level bivariate variable error (BVE) of ~3.8 cm for the expert group and ~7.4 cm for novices.

---

## Suggested attachments (in priority order)

1. `blog/figures/fig_anisotropy_trails.png` — **the key new visual.** Three-panel comparison of optimum-aim trajectories under isotropic / vertical-dominant / horizontal-dominant covariance shapes. This is the figure the email body builds toward.
2. `results/endgame/optimum_trail.png` — the classic Tibshirani-style aim migration under the isotropic assumption. Good baseline for someone who's read the blog he linked to.
3. `results/heatmaps/ev_heatmap_good_sx0.07_sy0.07.png` — one EV heatmap so the recipient can see what the solver produces (the cost surface around the optimum).
4. `blog/pdf/01_player_accuracy.pdf` — *only* if he asks for depth. ~5000 words, ten figures, full citations. The email body should make the high-level case; this is the deep dive for if/when he wants it.

## Open questions for you before sending

- Tone: too long? Too technical? My instinct is the recipient can take this density given his role, but you know him better.
- The "do pros play optimally?" framing — is that a productive provocation, or does it risk reading as "I'm questioning your platform's value"? Can be softened to "given how much data you have, you can answer this" if so.
- The pitch in the "concretely, here's what one could do" section names three things. We could drop the third (pro strategy validation) if the email feels too much like a sales/value-extraction pitch.
- Whether to mention that the solver code is on GitHub (it is; not pushed yet but local) — invites collaboration but is more commitment to make it presentable.
