# How to model a player's accuracy

> Section §3 of the blog. Detailed treatment of the player-accuracy model, vocabulary, the data behind every numerical choice, and the limits of the model. Read top-to-bottom this is ~4500 words. The early subsections are written so a non-technical reader can follow; the later ones go progressively deeper, and each numerical claim is grounded in a paper that lives in [`papers/`](../papers/). Where we make a choice that's *not* dictated by data, we say so.

## 1. The question, and the simplest possible answer

A dart leaves the player's hand aimed at some point on the board, and lands at some other point nearby. The *aim* point is a deliberate choice; the *landing* point is partly random. To compute where the player should aim, we need a model for that randomness.

We can ask the model two things:

1. **For a given aim, what's the distribution of landing points?** — single-throw question.
2. **How does that distribution depend on who's throwing?** — player-skill question.

The single sharpest answer in the literature is to model the landing point as a **two-dimensional Gaussian distribution centred on the aim**. The Gaussian is fully described by two numbers: where its centre is (we'll set that equal to the aim point) and how "spread out" it is. The spread is captured by a small matrix called the **covariance matrix**, written Σ (capital Greek sigma). This is the model every paper on darts optimization since Tibshirani, Price & Taylor uses [^tibshirani2011], and it's what we use here.

The rest of this section unpacks (a) what those two words — "two-dimensional Gaussian" and "covariance matrix" — actually mean and look like, (b) what numerical values are realistic for each skill level, (c) when the model fails.

> [📊 **FIG-3.1**: Two-panel teaser. Left: a single dart aimed at the bullseye landing exactly there (the "deterministic" mental model most people start with). Right: same aim, 200 simulated throws drawn from a 2D Gaussian with σ = 25 mm, showing the actual cloud-of-hits picture. Caption: "Real darts don't land where you aim. They land in a cloud around it."]

## 2. Vocabulary you can refer back to

Read this once, refer back as needed.

- **σ** is the lowercase Greek letter *sigma* (pronounced "**SIG**-mah"). In statistics it conventionally denotes a **standard deviation** — informally, the typical size of an error. So "σ = 25 mm" means "the typical miss is about 25 mm". σ is always non-negative.
- **σ²** is the variance — sigma squared. Same information as σ, just on the squared scale; you'll see it in formulas because the math is cleaner there.
- **Σ** is the uppercase Greek letter *Sigma* (same name, same pronunciation, capital). In statistics it conventionally denotes a **covariance matrix**, which is the multi-dimensional version of σ². For our 2D darts problem Σ is a 2×2 matrix.
- **Gaussian** (= normal distribution) is the bell-curve probability distribution. In 2D, it's a bell-shaped *surface* (a hill of probability) hovering over the plane.
- **2D Gaussian = bivariate normal**: same thing, two names. We'll use "2D Gaussian".
- **Aim point** (or *intended target*): where the player meant to throw. We write it μ (mu) in formulas — μ is the **mean** of the Gaussian.
- **Landing point**: where the dart actually ended up. We write it Z.

The model is then literally one line:

    Z = μ + ε,        ε ~ N(0, Σ)

In words: the landing point Z equals the aim point μ plus an error ε, where the error follows a Gaussian distribution centred at the origin with covariance Σ.

> [📊 **FIG-3.2**: A clean diagram showing the formula geometrically — the aim point μ as a star, the landing point Z as a dart, the difference vector ε drawn as an arrow, all over the board. Caption: "The model in one picture."]

## 3. What the covariance matrix Σ looks like and what each entry means

Σ is a 2×2 matrix:

         ⎡ Σ_xx  Σ_xy ⎤
    Σ =  ⎢            ⎥
         ⎣ Σ_yx  Σ_yy ⎦

There are three independent numbers in here (Σ_xy = Σ_yx because the matrix is required to be *symmetric* — see the box below):

- **Σ_xx = σ_x²** — the variance along the x-axis (how scattered horizontally, ignoring vertical).
- **Σ_yy = σ_y²** — the variance along the y-axis (how scattered vertically, ignoring horizontal).
- **Σ_xy = ρ · σ_x · σ_y** — the off-diagonal term. ρ (rho) is the **correlation coefficient** between the x and y errors, a number between −1 and +1. ρ = 0 means horizontal and vertical errors are independent; ρ ≠ 0 means they're linked (e.g. when you miss right, you also tend to miss high).

Geometrically, the iso-probability contours of a 2D Gaussian are **ellipses**, and Σ tells you the shape, size, and orientation of those ellipses:

- A circular cloud → σ_x = σ_y, ρ = 0.
- A vertically-stretched ellipse → σ_y > σ_x, ρ = 0.
- A tilted ellipse → ρ ≠ 0.

> [📊 **FIG-3.3**: A three-panel figure showing three Gaussian contour shapes side by side, with their Σ matrices written underneath each one. Panel A: circular (σ_x = σ_y = 20, ρ = 0). Panel B: vertical ellipse (σ_x = 15, σ_y = 30, ρ = 0). Panel C: tilted ellipse (σ_x = 15, σ_y = 30, ρ = 0.5). Each panel shows 1σ and 2σ contour ellipses plus ~500 sampled points. This is the single most important figure of the section.]

> **Sidebar: "symmetric", "diagonal", "isotropic" — which is which?**
>
> These three adjectives describe Σ at different levels of generality.
>
> - A **symmetric** matrix means Σ_xy = Σ_yx. *Every* covariance matrix is symmetric, automatically — it's a property of the math, not a modelling choice. We won't say "symmetric Σ" again because there is no other kind.
> - A **diagonal** matrix means the off-diagonal entries are zero (ρ = 0): horizontal and vertical errors are independent. The ellipses are axis-aligned. "Diagonal" implies axis-aligned but says nothing about whether σ_x = σ_y.
> - An **isotropic** (a.k.a. **spherical**) covariance means it's *both* diagonal *and* σ_x = σ_y. The ellipses are circles, and the model has only one number, σ.
>
> So in order of increasing freedom: isotropic ⊂ diagonal ⊂ full (= symmetric, the most general). All the published dart papers consider at least the full version; some restrict to one of the simpler forms for clarity.

## 4. The axis convention, once and for all

This is the kind of thing that silently causes everyone's heatmaps to be 90° wrong, so we state it up front. We adopt the same convention as every paper cited in this post (Tibshirani 2011 §3 p. 7; Haugh & Wang 2022 §3 p. 5; Haugh & Wang 2024 §7) and as standard math/computer-vision practice:

> **Our convention** (this project; defined in `src/darts/board.py`):
>
> - **x** is the **horizontal** axis. **+x points to the right** (toward the 6 wedge, at the 3 o'clock position).
> - **y** is the **vertical** axis. **+y points upward** (toward the 20 wedge, at the top of the board).
> - The origin is the centre of the bullseye.
> - Both axes are normalised to [-0.5, 0.5], so the full board fits exactly in the unit square.

The biomechanical claim that "vertical scatter is larger than horizontal" reads as σ_y > σ_x in both the papers and in our code. Every numerical σ value we quote from the literature in this post is **stated directly in this convention**, with no rotation footnote needed.

> [📊 **FIG-3.4**: An annotated dartboard with the (x, y) axes drawn on it as arrows: +x to the right (toward 6) and +y upward (toward 20). Caption: "Our axis convention — matches Tibshirani 2011 and Haugh & Wang."]

## 5. What the data says: empirical fits from the literature

There is one foundational dataset we can lean on for amateurs and one for pros.

### 5.1 Amateurs: Tibshirani's two authors

Tibshirani et al. asked two of the paper's authors to each throw 100 darts aimed at the bullseye and recorded only the *scores* (which region each dart landed in, not the precise position). Using an EM algorithm (§3.1 p. 7 of [Tibshirani 2011, in `papers/tibshirani_2011_darts.pdf`][^tibshirani2011]) they fit both a simple isotropic σ and the general 2×2 Σ. Here are the resulting fits, in the paper's (= our) convention:

| Player          | Simple isotropic σ | Full Σ fit                              | Ratio σ_y / σ_x |
|-----------------|--------------------|------------------------------------------|------------------|
| "Author 1" (weak amateur, Tibshirani himself) | 64.6 mm | σ_x = 42.7, σ_y = 68.6, ρ = −0.16 | **1.61** |
| "Author 2" (decent amateur, Andy Price)        | 26.9 mm | σ_x = 17.9, σ_y = 39.1, ρ = −0.22 | **2.19** |

Two important takeaways from this single table:

1. **Even the "weak" beginner has anisotropic error.** Both amateurs land with significantly more vertical (σ_y) scatter than horizontal (σ_x) — a ratio of roughly 1.6 to 2.2. The σ = 64.6 vs σ = 26.9 numbers that the simple isotropic fit returns are just *summaries* of the true asymmetric distributions; they hide the asymmetry but it's there. We have to be careful about citing "the beginner has σ = 65 mm" without noting that the actual error ellipse is taller-than-wide.
2. **Tibshirani §3 attributes the asymmetry to biomechanics**: "[i]t is common for most players to have a smaller variance in the horizontal direction than in the vertical one, since the throwing motion is up-and-down with no … lateral component" (p. 7). This isn't a per-player accident, it's a structural feature of how people throw.

These are the only two amateur-σ data points in the published literature. They anchor our "beginner" and "good amateur" tiers respectively.

### 5.2 Professionals: the 16 PDC top-16 from Haugh & Wang

The professional skill model comes from the data accompanying [Haugh & Wang 2022][^hw2022]. They have the 2019 PDC season's full per-player, per-target hit counts for the top 16 players, and they fit a *separate* Σ for each of six target regions (T20, T19, T18, T17, the inner bullseye, and a lump of all twenty doubles). The numerical fits are in `papers/OptimalDarts_repo/ALL_Model_Fits.mat`. Aggregated over the 16 pros and pooled across all six target regions, the typical σ is **σ_x ≈ 8.6 mm, σ_y ≈ 8.2 mm — almost isotropic**.

A few individual players, just to make it concrete (T20 region):

| Player                | σ_x (mm) | σ_y (mm) | ρ      |
|-----------------------|----------|----------|--------|
| Michael van Gerwen    | 8.78     | 6.06     | +0.36  |
| Gerwyn Price          | 9.67     | 6.16     | +0.29  |
| Peter Wright          | 9.70     | 6.60     | +0.30  |
| Mensur Suljovic       | 8.95     | 7.07     | +0.46  |

Three things stand out:

1. **All four have σ_x > σ_y at T20** — *horizontal* scatter is larger than vertical at this specific target. This is the **opposite** of the amateur biomechanical pattern (where σ_y > σ_x). We address this puzzle in §6.
2. **σ at T20 is in the 6–10 mm range** for top-of-the-table pros, consistent with the rule-of-thumb "5 mm" Tibshirani uses to illustrate a near-perfect thrower.
3. **The correlation ρ is non-zero and consistent in sign** (~+0.3 to +0.5 for these four). We'd love to attach a physical meaning to this but we can't — see §6 again.

> [📊 **FIG-3.5**: Two-panel comparison figure. Left: the two Tibshirani amateurs' fitted ellipses overlaid on a dartboard (axes labelled, ellipses at 1σ and 2σ contours). Right: van Gerwen's six per-region fitted ellipses overlaid on a dartboard, one at each of T20/T19/T18/T17/bull/doubles. Caption: "Two amateurs (left) vs one world champion at six different targets (right). The amateurs' ellipses are large and visibly elongated; the pro's are tight and have different shapes at different targets — see §7."]

## 6. The anisotropy puzzle: does the ratio depend on skill?

Sharp version of the question we landed on while drafting this: **Tibshirani's amateurs have σ_y / σ_x ≈ 1.6 to 2.2 (vertical scatter much larger than horizontal), but the pros' pooled-across-regions average is essentially isotropic (≈ 0.96). Is anisotropy a beginner thing that disappears with skill, or is something else going on?**

The honest answer is "we can't fully tell from the published data, but here's what we can say."

### What we know
- The amateur asymmetry is large, has the same sign for both Tibshirani authors, and is attributed to the throwing motion. It's almost certainly real (Tibshirani 2011 §3 p. 7).
- The pro pooled ratio of ~1.0 is computed from the six per-region Σ matrices, *averaged* across regions. This is the right number to compare to the amateurs' bullseye-only fit only if you assume Σ is global — which the pros' data say it isn't (§7).
- Within a single region, individual pros are *not* isotropic. At T20, four of the four pros above have σ_x noticeably larger than σ_y; at T18 the relationship flips for many of them. The per-region ellipse orientation seems to track the target region's geometry — wide bed → flatter ellipse, tall bed → taller ellipse.

### What the literature has to say about this
[Haugh & Wang 2024 §7.1, in `papers/haugh_wang_2024_eb_darts.pdf`][^hw2024], dedicates a section to exactly this problem. Their conclusion (p. 18, paraphrased): when you only have *scores* (no actual (x, y) landing positions), the off-diagonal entry of the fitted Σ is **not identifiable**. They show a synthetic example (Figure 5, p. 19) of three landing-point datasets with very different true correlations ρ ∈ {−0.5, 0, +0.5} that produce *identical* observed score frequencies — and therefore identical EM fits up to that nuisance parameter.

Worse, their analysis suggests **the fitted ellipse orientation tends to align with the geometric extent of the target region**, not with the player's true error direction. For T20 (a "tall" region — narrow horizontally, deep radially), score-only data plus EM gives you back an ellipse stretched in roughly the radial direction. This is the model's least bad fit to the available information, not a measurement of the player's body.

The implication is sobering: **the pro per-region anisotropy reported in OptimalDarts and analysed in Haugh & Wang 2022 should not be interpreted as "this pro is more accurate horizontally than vertically when throwing at T20."** It is a statistical fit to score data, with known identifiability issues; you cannot read the eigenvectors of the fitted Σ as biomechanical statements.

For amateurs the same caveat technically applies, but the asymmetry is so large and so consistent across the two known datapoints — and the biomechanical explanation is so plausible — that we are comfortable treating "σ_y > σ_x" (vertical scatter > horizontal) as a real signal for amateur-level players.

### Our modelling choice (and we flag this as a choice, not a deduction)

We apply an anisotropy ratio at the **amateur** tiers (beginner, average, good) anchored on Tibshirani's two empirical fits, and we make the **pro/world-champion** tiers isotropic. The rationale:

- For amateurs: the asymmetry is empirically supported and biomechanically grounded. Ignoring it would be a worse model than including it.
- For pros: the asymmetric per-region fits exist but cannot be interpreted as physical anisotropy of the player. The simplest defensible choice is to use a single isotropic σ that summarises the player's overall T20-region accuracy. A genuinely physical pro model would require landing-position data, which doesn't exist in the published record.

A reader interested in the per-region Haugh & Wang model can use the matrices in `papers/OptimalDarts_repo/ALL_Model_Fits.mat` directly — our infrastructure supports diagonal Σ, and a small wrapper would let it consume the full 96-matrix table for the 16 PDC pros. We discuss this in §9 as a future extension.

> [📊 **FIG-3.6**: A scatterplot with skill level on the x-axis (σ in mm, log scale) and σ_y / σ_x ratio on the y-axis. Points: the two Tibshirani amateurs (ratio ~1.6 and ~2.2), each of the 16 pros' pooled-across-region ratio, and the proposed-tier values from §8. A horizontal line at ratio = 1 marks isotropic. The plot should make visible that amateurs cluster well above 1 and pros cluster near 1.]

## 7. Target dependence: how σ varies across the board

This is the model's biggest known limitation and the user asked us to discuss it explicitly. Both Haugh & Wang papers fit *separate* Σ for each target region, and the data shows this matters — for pros.

### The data, in one paragraph

For the 16 PDC top-16 pros pooled (Haugh & Wang 2024 §3 p. 5):

- T20 hit rate: 41.2%
- T19 hit rate: 41.7%
- T18 hit rate: 36.9%
- T17 hit rate: 33.5%

Why the spread? T20 is the practice target. The 8-percentage-point gap from T20 to T17 corresponds to roughly 20-30% larger σ at T17 than at T20 for typical pros, with significant per-player variation. For the example of Michael van Gerwen: T20 hit-rate is 45.3%, T17 is 30.2% (Haugh & Wang 2024 §5 p. 10). For Lewis A. the geometric-mean σ at T17 is ≈ 11.1 mm versus ≈ 8.4 mm at T20 — about 32% larger.

There's a second flavour of target-dependence: **bias** — the player's *actual* aim point can be offset from the *intended* target. From [Haugh & Wang 2024 Table 5 p. S-5, in `papers/haugh_wang_2024_eb_darts.pdf`][^hw2024], averaged across 16 pros, the distance between fitted μ and the region centre is:

- T20: 1.47 mm
- T19: 1.48 mm
- T18: 3.42 mm
- T17: 3.88 mm

Pros aim almost dead-centre at their practice target T20, and noticeably *off*-centre at the less-rehearsed T17. The bivariate normal absorbs this offset into the (μ, Σ) joint fit, which is why the *apparent* σ inflates at less-practised targets even more than the pure error inflates.

For amateurs there is **no published data** on target dependence. Tibshirani 2011 fits one Σ per player, period. Two opposing intuitions:

- Amateurs almost always throw at T20 anyway (it's the high-score target), so they get less differential practice and might have *less* per-region heterogeneity than pros.
- But amateurs' σ is so much larger (25–70 mm vs 5–10 mm) that small per-region differences are dominated by the overall scatter; even a 30% per-region inflation is hard to detect from 100 throws.

### How serious is the limitation?

For **pros**, ignoring target dependence is costly. Haugh & Wang 2024 Table 3 (p. 23) shows that a pro who plays *as if* their Σ were global, when it actually varies by region, loses ~5.4% of match win-probability over a 35-leg match (12% for Jonny Clayton specifically). They write (§5 p. 10): "[assuming global Σ] would result in a drastic underestimation of a player's skill level and should be avoided" — for pros.

For **amateurs** the cost is presumably much smaller; nobody has measured it, but the model error is in the noise of the much larger global σ.

### Why we don't model it (and how a determined reader could)

Building a target-dependent solver would require either:

(a) A continuous function σ(x, y) that the solver can evaluate at any aim point — but we have no principled way to interpolate between Haugh & Wang's six discrete fitted regions (R^2 → R^4 is overkill from six samples), and
(b) A per-region Σ table queried at solve time — but this multiplies the bookkeeping and only helps the high-skill end of the spectrum.

We chose (deliberate choice, not deduction): treat σ as global. The data presented above is the **upper bound on the magnitude of the error** this choice introduces. For amateurs, treating σ as global is probably fine; for pros, it costs a few percent of match win-probability against the per-region truth.

A motivated individual player who wants to refine the model has a clear path: throw 50–100 darts at each of their main targets (T20, T19, T18, T17, bullseye, D-out targets), fit a separate Σ per region (the `darts.covariance` module already supports this — you just need to bucket your throws by aim), and feed the result into a small per-region wrapper around our solver. We sketch this extension in §11.

> [📊 **FIG-3.7**: A bar chart of the four pro T20/T19/T18/T17 hit rates with the per-player range overlaid as error bars. Plus a second bar chart showing the average |μ − region-centre| at the four trebles. Two bar charts, one figure. Caption: "Pros are not equally good at every triple. T20 is their practice target."]

## 8. The six-tier proposal (anchored, not arbitrary)

Putting all of the above together, here is our proposed six-tier scheme. We give for each tier the **σ_x (horizontal) and σ_y (vertical) in normalised board units** so the values plug straight into our solver, plus the literature anchor that motivates the choice.

| Tier              | σ_x (mm) | σ_y (mm) | σ_x norm | σ_y norm | Ratio σ_y/σ_x | Anchored on                                                                  |
|-------------------|----------|----------|----------|----------|----------------|-------------------------------------------------------------------------------|
| **perfect**       | 0        | 0        | 0        | 0        | n/a            | mathematical limit; Tibshirani 2011 Figure 2 (σ=5 mm) is the closest empirical reference |
| **world_champion**| 7        | 7        | 0.021    | 0.021    | 1.0 (isotropic; data limitation) | Best PDC pros at T20 (e.g. van Gerwen geometric-mean σ ≈ 7.3 mm at T20)        |
| **professional**  | 9        | 9        | 0.026    | 0.026    | 1.0 (isotropic; data limitation) | Pooled mean of the 16 PDC top-16 across all six target regions (≈ 8.4 mm)     |
| **good**          | 18       | 39       | 0.053    | 0.115    | 2.17           | Tibshirani Author 2 (decent amateur) full-Σ fit                              |
| **average**       | 30       | 51       | 0.088    | 0.150    | 1.70           | Interpolation between Author 1 and Author 2; ratio chosen as the midpoint    |
| **beginner**      | 43       | 69       | 0.126    | 0.203    | 1.60           | Tibshirani Author 1 (weak amateur) full-Σ fit                                 |

Notes on the choices:

- **Anisotropy applied at amateur tiers, not at pro tiers.** This is the choice the previous subsections built toward. Empirically justified for amateurs (Tibshirani §3); knowingly conservative for pros (where the per-region anisotropy exists but cannot be cleanly disentangled from region geometry).
- **The `professional` tier is anchored on the 16-pro pool**, not on any single player. The "world_champion" tier is the upper edge of that distribution and is calibrated to the best players' σ at T20 specifically (because that's their actual aim 90% of the time).
- **`average` is interpolated**, not directly measured — the gap between Tibshirani's two authors is the only amateur data we have, so we put one tier on each and one in between. We flag this as a modelling choice rather than data.
- **`perfect`** has σ = 0 exactly. The previous version of the codebase used σ = 0.0001 as a safety; the FFT kernel handles σ = 0 (it collapses to a delta function), so we can use the exact zero.

> [📊 **FIG-3.8**: Headliner figure for §3. A single dartboard rendered six times, one per tier. Each panel shows the player's aim (a cross at the bullseye) and their 1σ and 2σ error ellipses, with the same colour-coding throughout. Tier labels and σ values printed beside each ellipse. The visual ladder of "how good is good".]

> [📊 **FIG-3.9**: A small "skill ladder" infographic. Three columns: σ (mm), tier name and one-line description, expected 3-dart-average score at the solver's recommended aim. The third column anchors the abstract σ values to "what would your match average look like".]

## 9. Limitations, restated as honest claims

The model we use makes five assumptions worth restating in one place:

1. **Errors are Gaussian.** The 2D Gaussian is empirically adequate; Tibshirani 2011 §4.1 tests a skew-normal extension and finds it does not change the heatmaps meaningfully (p. 10). Heavy-tailed extensions (Student-t, etc.) are not in the published literature and would require landing-position data to motivate.
2. **Errors are independent throw-to-throw.** Each dart's landing point is sampled independently from N(μ, Σ). There is some evidence (Ötting et al. 2020, cited by Haugh & Wang 2022 — not in the local papers folder) that the first dart of a turn is less accurate than the second and third, but this within-turn structure is not in our model.
3. **Σ is global across the board.** As discussed in §7, this is materially false for pros and probably fine for amateurs.
4. **The aim μ equals the intended target ϑ.** Haugh & Wang 2024 Appendix C shows pros have small but measurable biases (~1.5 mm at T20 to ~5 mm at T17). For amateurs, both μ and the bias should probably be inflated proportionally to σ. We assume μ = ϑ throughout.
5. **No bounce-outs.** A dart that hits the wire and bounces off the board is assumed to score zero (i.e. it gets binned into the MISS outcome). Real bounce rate is ~0.3% for pros (Haugh & Wang 2022 Assumption 2, p. 12).

Each of these is a place where a future, more careful model could improve on this one. Each of them is also a place where the current model is the one used by the entire prior published literature, so we're at least no worse than the state of the art.

## 10. Practical methodology: how someone would measure their own σ

If a reader wants to fit their own personal Σ — and we think they should — there are two reasonable methods, both well-attested in the literature:

1. **Score-only EM (Tibshirani 2011 §2.2, §3.1).** Throw n ≥ 50 darts aimed at a single target (canonically the bullseye), record only which region each one hits. Fit Σ by EM with importance-sampling E-step. Closed form for the isotropic special case (Tibshirani §A.2 p. 13). This is what the papers do because it's all the data they can get from broadcast PDC matches.
2. **Position fit from clicked landing points (this project, `darts.covariance`).** If you have a photo of the board with your hits still in it, click each one and fit Σ as the sample covariance of the (x, y) values. This is *strictly more informative* than score-only — it sees the full 2D distribution rather than just its multinomial projection onto the 63-cell partition — and it's what we provide. The `scripts/click_hits.py` tool does this interactively.

The trade-off: score-only EM works at distance (e.g. you can score someone else's PDC match), but produces a Σ whose off-diagonal is, as discussed above, not identifiable. Position-fit needs you to actually photograph and click your own board, but produces a Σ whose every entry is meaningful.

> [📊 **FIG-3.10**: A small "how many throws do you need" figure. Synthetic 2D Gaussian samples with known true σ, fit with N = 10, 30, 50, 100, 200; plot the 95% CI on the recovered σ_x and σ_y as a function of N. Identify the visible knee. This tells the reader how many throws to commit to.]

## 11. What we'd change if we had more data

A short list, in priority order, for anyone who wants to extend the model:

1. **Landing-position data for pros.** With (x, y) for each televised dart we could *actually* identify the off-diagonal of Σ and verify or refute the "per-region anisotropy is a fitting artifact" claim from Haugh & Wang 2024 §7. The technology to track this exists (PDC trialled a tracking system in 2023); the data is not yet public.
2. **Per-region Σ via the OptimalDarts table.** A small wrapper around our solver that consumes the 96 (16-player × 6-region) matrices in `ALL_Model_Fits.mat` would let us reproduce the Haugh & Wang 2022 / 2024 pro analyses directly and compare them to the global-Σ approximation we use. This is a near-term, no-new-data extension.
3. **Within-turn dependence.** If dart 1 is genuinely less accurate than dart 3, the end-game DP's per-state aim policy might shift. Modelling this requires per-throw-position data, which is presumably easier to extract from raw broadcasts than landing-positions.
4. **Personal photo-click datasets at non-bullseye targets.** Any individual reader with a phone camera could produce per-region Σ data for themselves in an afternoon, and the model fitting is a 10-line modification of `darts.covariance.fit_gaussian`. We'd love to collect a small open dataset of "amateur σ at six target regions" — there is literally none in the published record.

---

[^tibshirani2011]: Tibshirani, R. J., Price, A. & Taylor, J. (2011). *A Statistician Plays Darts.* Journal of the Royal Statistical Society Series A, 174(1), 213–226. Local copy: `papers/tibshirani_2011_darts.pdf`. URL: https://www.stat.cmu.edu/~ryantibs/papers/darts.pdf
[^hw2022]: Haugh, M. B. & Wang, C. (2022). *Play Like the Pros? Solving the Game of Darts as a Dynamic Zero-Sum Game.* INFORMS Journal on Computing, 34(5), 2540–2551 (arXiv:2011.11031). Local: `papers/haugh_wang_2022_dp_darts.pdf`. URL: https://arxiv.org/abs/2011.11031
[^hw2024]: Haugh, M. B. & Wang, C. (2024). *An Empirical Bayes Approach for Estimating Skill Models for Professional Darts Players.* arXiv:2302.10750v3. Local: `papers/haugh_wang_2024_eb_darts.pdf`. URL: https://arxiv.org/abs/2302.10750
