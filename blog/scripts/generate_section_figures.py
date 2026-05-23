"""Generate the ten figures for the §3 (player-accuracy) blog section.

Outputs go to ``blog/figures/`` as PNGs.

Each function builds one figure independently so we can iterate on a single
one without re-running the rest. Run the script once to make everything.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402
import numpy as np  # noqa: E402

from darts import board  # noqa: E402
from darts.viz import draw_board  # noqa: E402

# Output directory (relative to repo root).
HERE = Path(__file__).resolve().parent.parent
OUT_DIR = HERE / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _draw_blank_board(ax, *, with_numbers: bool = True):
    """Draw the dartboard outline at unit scale on ``ax`` (board ∈ [-0.5, 0.5])."""
    draw_board(ax, with_numbers=with_numbers)
    ax.set_aspect("equal")
    ax.set_xlim(-0.55, 0.55)
    ax.set_ylim(-0.55, 0.55)
    ax.set_xticks([])
    ax.set_yticks([])


def _add_ellipse(ax, mean, sigma_x, sigma_y, *, rho=0.0, n_sigma=1.0, **kw):
    """Add a 2D Gaussian iso-probability ellipse at n_sigma."""
    cov = np.array(
        [[sigma_x ** 2, rho * sigma_x * sigma_y],
         [rho * sigma_x * sigma_y, sigma_y ** 2]]
    )
    vals, vecs = np.linalg.eigh(cov)
    angle = float(np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1])))
    width = 2 * n_sigma * float(np.sqrt(vals[1]))
    height = 2 * n_sigma * float(np.sqrt(vals[0]))
    e = mpatches.Ellipse(
        mean, width=width, height=height, angle=angle, fill=False, **kw
    )
    ax.add_patch(e)


# ---------------------------------------------------------------------------
# FIG-3.1 — Real darts land in a cloud
# ---------------------------------------------------------------------------


def fig_31_teaser():
    rng = np.random.default_rng(0)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 5.5))

    # Left: deterministic single dart at bullseye.
    _draw_blank_board(axL, with_numbers=True)
    axL.plot(0, 0, marker="*", color="gold", markersize=22, markeredgecolor="black", zorder=5)
    axL.set_title("The mental model: aim = land", fontsize=13)

    # Right: 200 darts, σ=25 mm (= 0.0735 normalised), isotropic.
    sigma = 25.0 / board.TOTAL_DIAM  # ≈ 0.0735
    samples = rng.multivariate_normal([0, 0], np.diag([sigma ** 2, sigma ** 2]), 200)
    _draw_blank_board(axR, with_numbers=True)
    axR.scatter(samples[:, 0], samples[:, 1], s=20, color="tab:red", alpha=0.55, zorder=4)
    axR.plot(0, 0, marker="*", color="gold", markersize=18, markeredgecolor="black", zorder=5)
    axR.set_title("Reality: aim is the centre of a cloud", fontsize=13)

    fig.suptitle("Real darts don't land where you aim — they land in a cloud around it.",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_3_1_teaser.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# FIG-3.2 — Z = μ + ε diagram
# ---------------------------------------------------------------------------


def fig_32_model_diagram():
    fig, ax = plt.subplots(figsize=(7, 7))
    _draw_blank_board(ax, with_numbers=True)
    # Aim: μ. We put it slightly off-bullseye so the arrow is visible.
    mu = np.array([0.0, 0.30])  # vertical-up = toward 20
    # Landing point: Z = μ + ε, ε ~ N(0, Σ). Pick a deterministic ε for the diagram.
    eps = np.array([0.04, -0.05])
    z = mu + eps
    # Draw aim.
    ax.plot(*mu, marker="*", color="gold", markersize=22, markeredgecolor="black", zorder=6)
    ax.annotate("μ  (aim)", mu, xytext=(mu[0] + 0.04, mu[1] + 0.04),
                fontsize=13, color="darkgoldenrod")
    # Draw landing.
    ax.plot(*z, marker="o", color="tab:red", markersize=14, markeredgecolor="black", zorder=6)
    ax.annotate("Z  (landing)", z, xytext=(z[0] + 0.04, z[1] - 0.06),
                fontsize=13, color="darkred")
    # Draw ε arrow.
    ax.annotate(
        "", xy=z, xytext=mu,
        arrowprops=dict(arrowstyle="->", color="dimgray", lw=2.4),
    )
    ax.text((mu[0] + z[0]) / 2 + 0.02, (mu[1] + z[1]) / 2, "ε",
            fontsize=15, color="dimgray", fontweight="bold")
    # Draw the 1σ and 2σ ellipses for visual context.
    _add_ellipse(ax, mu, 0.05, 0.10, rho=0.0, n_sigma=1.0, color="dimgray", linestyle="--", linewidth=1.4)
    _add_ellipse(ax, mu, 0.05, 0.10, rho=0.0, n_sigma=2.0, color="dimgray", linestyle=":", linewidth=1.2)
    ax.set_title("Z = μ + ε,  ε ~ N(0, Σ)", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_3_2_model_diagram.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# FIG-3.3 — Three covariance shapes
# ---------------------------------------------------------------------------


def fig_33_three_shapes():
    rng = np.random.default_rng(1)
    sigmas = [
        ("Isotropic\nσ_x = σ_y = 20 mm,  ρ = 0", 20.0, 20.0, 0.0),
        ("Diagonal anisotropic\nσ_x = 15, σ_y = 30 mm,  ρ = 0", 15.0, 30.0, 0.0),
        ("Tilted (correlation)\nσ_x = 15, σ_y = 30 mm,  ρ = +0.5", 15.0, 30.0, 0.5),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    for ax, (title, sx_mm, sy_mm, rho) in zip(axes, sigmas):
        sx = sx_mm / board.TOTAL_DIAM
        sy = sy_mm / board.TOTAL_DIAM
        cov = np.array(
            [[sx ** 2, rho * sx * sy], [rho * sx * sy, sy ** 2]]
        )
        samples = rng.multivariate_normal([0, 0], cov, 400)
        _draw_blank_board(ax, with_numbers=False)
        ax.scatter(samples[:, 0], samples[:, 1], s=10, color="tab:gray", alpha=0.45)
        _add_ellipse(ax, [0, 0], sx, sy, rho=rho, n_sigma=1.0,
                     color="tab:blue", linewidth=2.0)
        _add_ellipse(ax, [0, 0], sx, sy, rho=rho, n_sigma=2.0,
                     color="tab:blue", linewidth=1.4, linestyle="--")
        ax.set_title(title, fontsize=11)
    fig.suptitle("The three shapes a covariance matrix can produce", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_3_3_three_shapes.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# FIG-3.4 — Axis convention
# ---------------------------------------------------------------------------


def fig_34_axes():
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    _draw_blank_board(ax, with_numbers=True)
    arrow_len = 0.42
    # +x → right (toward 6).
    ax.annotate(
        "", xy=(arrow_len, 0), xytext=(0, 0),
        arrowprops=dict(arrowstyle="->", color="tab:red", lw=3.5),
    )
    ax.text(arrow_len + 0.02, -0.02, "+x  (horizontal, toward 6)", fontsize=12,
            color="tab:red", fontweight="bold")
    # +y → up (toward 20).
    ax.annotate(
        "", xy=(0, arrow_len), xytext=(0, 0),
        arrowprops=dict(arrowstyle="->", color="tab:blue", lw=3.5),
    )
    ax.text(0.02, arrow_len + 0.02, "+y  (vertical, toward 20)",
            fontsize=12, color="tab:blue", fontweight="bold")
    ax.plot(0, 0, "ko", markersize=8)
    ax.text(0.02, -0.04, "origin\n(bullseye)", fontsize=10, color="black")
    ax.set_title("Axis convention\n(matches Tibshirani 2011 / Haugh & Wang 2022, 2024)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_3_4_axes.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# FIG-3.5 — Tibshirani amateurs vs van Gerwen per-region
# ---------------------------------------------------------------------------


def _load_van_gerwen_per_region():
    """Load van Gerwen's 6 per-region Σ from the OptimalDarts .mat file.

    Returns a dict ``region_name -> Σ`` (2x2, mm²) in the new convention
    (which matches the file's convention — x horizontal, y vertical).
    """
    try:
        from scipy.io import loadmat
    except Exception:
        return None
    mat_path = HERE.parent / "papers" / "OptimalDarts_repo" / "ALL_Model_Fits.mat"
    if not mat_path.exists():
        return None
    data = loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
    regions = {"T20": "ModelFit_T20", "T19": "ModelFit_T19", "T18": "ModelFit_T18",
               "T17": "ModelFit_T17", "B50": "ModelFit_B50",
               "doubles": "ModelFit_All_Doubles"}
    # Player ordering (from runexample.py): VG is index 7 (1-based) → 6 (0-based).
    # alphabetical: Anderson, Aspinall, Chisnall, Clayton, Cross, Cullen, van Gerwen, ...
    VG_IDX = 6
    out = {}
    for name, key in regions.items():
        try:
            arr = data[key]
            entry = arr[VG_IDX] if hasattr(arr, "__len__") else arr
            sigma = getattr(entry, "FittedSigma", None)
            if sigma is None:
                continue
            out[name] = np.array(sigma, dtype=np.float64)
        except Exception:
            continue
    return out if out else None


def fig_35_tibshirani_vs_pro():
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 6.5))

    # Left: Tibshirani two authors (positions in paper convention: x horizontal,
    # y vertical, origin at bullseye, units mm). Convert to normalised board.
    tib_authors = [
        ("Author 1 (weak amateur)\nσ_x=42.7, σ_y=68.6 mm, ρ=−0.16",
         42.7, 68.6, -0.16, "tab:red"),
        ("Author 2 (decent amateur)\nσ_x=17.9, σ_y=39.1 mm, ρ=−0.22",
         17.9, 39.1, -0.22, "tab:orange"),
    ]
    _draw_blank_board(axL, with_numbers=True)
    for label, sx_mm, sy_mm, rho, color in tib_authors:
        sx = sx_mm / board.TOTAL_DIAM
        sy = sy_mm / board.TOTAL_DIAM
        _add_ellipse(axL, [0, 0], sx, sy, rho=rho, n_sigma=1.0, color=color, linewidth=2.0)
        _add_ellipse(axL, [0, 0], sx, sy, rho=rho, n_sigma=2.0, color=color, linewidth=1.3,
                     linestyle="--")
    axL.set_title("Tibshirani's two amateurs (score-only EM fit)\naiming at the bullseye",
                  fontsize=11)
    legend_handles = [
        mpatches.Patch(color="tab:red", label="Author 1 — beginner"),
        mpatches.Patch(color="tab:orange", label="Author 2 — decent amateur"),
    ]
    axL.legend(handles=legend_handles, loc="lower left", fontsize=9, frameon=True)

    # Right: van Gerwen per-region ellipses centred at each region centre.
    _draw_blank_board(axR, with_numbers=True)
    vg = _load_van_gerwen_per_region()
    if vg is not None:
        # Region centres in normalised board coords (new convention).
        r_t = (board.TRIPLE_EXT_DIAM / 2 - board.BORDER / 2) / board.TOTAL_DIAM  # triple ring mid
        # Wedge centres: theta_b for each number.
        # 20 at +y (theta=π/2), 19 at... numbers list in CCW order from 20.
        # angle_for_num: map number → wedge centre theta (in new convention).
        # The wedge centre for number with index i in NUMBERS_CCW is at angle pi/2 + i * 2pi/20
        # measured CCW from +x. So for number n we look up its index.
        numbers_ccw_list = [20, 5, 12, 9, 14, 11, 8, 16, 7, 19, 3, 17, 2, 15, 10, 6, 13, 4, 18, 1]
        angle_step = 2 * np.pi / 20
        def angle_for(num):
            i = numbers_ccw_list.index(num)
            return np.pi / 2 + i * angle_step
        centres = {
            "T20": (r_t * np.cos(angle_for(20)), r_t * np.sin(angle_for(20))),
            "T19": (r_t * np.cos(angle_for(19)), r_t * np.sin(angle_for(19))),
            "T18": (r_t * np.cos(angle_for(18)), r_t * np.sin(angle_for(18))),
            "T17": (r_t * np.cos(angle_for(17)), r_t * np.sin(angle_for(17))),
            "B50": (0.0, 0.0),
        }
        # Doubles: skip — they're 20 separate regions, not a single centre.
        colors = {"T20": "tab:blue", "T19": "tab:green", "T18": "tab:olive",
                  "T17": "tab:purple", "B50": "tab:cyan"}
        for region, (cx, cy) in centres.items():
            if region not in vg:
                continue
            S = vg[region]  # 2x2 in mm² in paper convention
            sx = float(np.sqrt(S[0, 0])) / board.TOTAL_DIAM
            sy = float(np.sqrt(S[1, 1])) / board.TOTAL_DIAM
            rho = float(S[0, 1]) / float(np.sqrt(S[0, 0] * S[1, 1])) if S[0, 0] * S[1, 1] > 0 else 0.0
            _add_ellipse(axR, (cx, cy), sx, sy, rho=rho, n_sigma=1.0,
                         color=colors[region], linewidth=2.0)
            _add_ellipse(axR, (cx, cy), sx, sy, rho=rho, n_sigma=2.0,
                         color=colors[region], linewidth=1.0, linestyle="--")
            axR.plot(cx, cy, "+", color=colors[region], markersize=12, markeredgewidth=2)
    axR.set_title("Michael van Gerwen, per-target Σ (score-only EM fit)\n"
                  "from Haugh & Wang 2022 ALL_Model_Fits.mat",
                  fontsize=11)
    legend_handles = [
        mpatches.Patch(color="tab:blue", label="T20"),
        mpatches.Patch(color="tab:green", label="T19"),
        mpatches.Patch(color="tab:olive", label="T18"),
        mpatches.Patch(color="tab:purple", label="T17"),
        mpatches.Patch(color="tab:cyan", label="Bullseye"),
    ]
    axR.legend(handles=legend_handles, loc="lower left", fontsize=9, frameon=True)

    fig.suptitle("Two amateurs (left) vs one world champion at six targets (right)",
                 fontsize=13, y=1.0)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_3_5_amateurs_vs_pro.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# FIG-3.6 — Anisotropy ratio vs skill level
# ---------------------------------------------------------------------------


def fig_36_anisotropy_ratio():
    fig, ax = plt.subplots(figsize=(10, 6))

    # Tibshirani amateurs (score-only, full Σ fit).
    pts_amateur = [
        ("Tibshirani A1\n(weak)", 56.0, 68.6 / 42.7),   # σ_geom ≈ 56 mm
        ("Tibshirani A2\n(decent)", 27.0, 39.1 / 17.9),
    ]
    for label, sigma, ratio in pts_amateur:
        ax.scatter(sigma, ratio, s=200, color="tab:orange", edgecolors="black", zorder=4)
        ax.annotate(label, (sigma, ratio), xytext=(7, 5), textcoords="offset points",
                    fontsize=9)

    # Lotze position-measured groups (BVE → σ_radial proxy; report unknown anisotropy
    # ratio explicitly as "not measured").
    lotze = [("Lotze expert\n(σ_radial≈19 mm,\nratio not\nreported)", 19.0),
             ("Lotze novice\n(σ_radial≈37 mm,\nratio not\nreported)", 37.0)]
    for label, sigma in lotze:
        ax.scatter(sigma, 1.0, s=170, color="tab:green", edgecolors="black",
                   marker="s", zorder=4)
        ax.annotate(label, (sigma, 1.0), xytext=(7, -42), textcoords="offset points",
                    fontsize=8, color="darkgreen")

    # PDC pros from OptimalDarts (pooled across regions; ratio σ_y/σ_x).
    vg = _load_van_gerwen_per_region()
    pdc_aggregate_ratio = 8.18 / 8.61  # from §5.2 / our extraction
    pdc_aggregate_sigma = 8.4  # mm
    ax.scatter(pdc_aggregate_sigma, pdc_aggregate_ratio, s=200, color="tab:blue",
               edgecolors="black", marker="D", zorder=4)
    ax.annotate("16 PDC pros\n(pooled, score-only)", (pdc_aggregate_sigma, pdc_aggregate_ratio),
                xytext=(7, 5), textcoords="offset points", fontsize=9)

    # Our proposed tier values (from §8 of the blog).
    tiers = [
        ("perfect", 0.1, 1.0, "tab:purple"),  # plotted as a tiny number
        ("world_champion", 6.0, 1.0, "tab:cyan"),
        ("professional", 9.0, 1.0, "tab:blue"),
        ("good", np.sqrt(13 * 26), 26 / 13, "tab:green"),
        ("average", np.sqrt(25 * 50), 50 / 25, "tab:olive"),
        ("beginner", np.sqrt(43 * 69), 69 / 43, "tab:red"),
    ]
    for label, sigma, ratio, color in tiers:
        ax.scatter(sigma, ratio, s=100, color=color, edgecolors="black",
                   marker="*", zorder=5, alpha=0.85)

    ax.axhline(1.0, color="black", linestyle=":", alpha=0.5)
    ax.text(70, 1.04, "isotropic line", fontsize=9, color="dimgray")
    ax.set_xscale("log")
    ax.set_xlabel("σ_geom (mm) — overall scatter magnitude (log scale)")
    ax.set_ylabel("σ_y / σ_x  (vertical / horizontal ratio)")
    ax.set_title("Anisotropy vs skill: amateurs stretch vertical; pros (score-only fits) cluster near 1.0",
                 fontsize=12)
    ax.grid(alpha=0.3, which="both")
    ax.set_xlim(1, 100)
    # legend
    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:orange",
                   markeredgecolor="black", markersize=12, label="Tibshirani amateurs (score-only)"),
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="tab:green",
                   markeredgecolor="black", markersize=11, label="Lotze groups (position-measured, no ratio)"),
        plt.Line2D([0], [0], marker="D", color="w", markerfacecolor="tab:blue",
                   markeredgecolor="black", markersize=12, label="PDC pros (score-only pooled)"),
        plt.Line2D([0], [0], marker="*", color="w", markerfacecolor="gray",
                   markeredgecolor="black", markersize=15, label="Our tier proposal (§8)"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_3_6_anisotropy_ratio.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# FIG-3.7 — Pro hit-rate + bias bars
# ---------------------------------------------------------------------------


def fig_37_target_dependence():
    targets = ["T20", "T19", "T18", "T17"]
    hit_rates = [41.2, 41.7, 36.9, 33.5]  # %
    biases_mm = [1.47, 1.48, 3.42, 3.88]  # mm
    vg_rates = [45.3, 41.0, 36.0, 30.2]   # van Gerwen approximations

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.2))

    x = np.arange(len(targets))
    width = 0.36
    axL.bar(x - width / 2, hit_rates, width, color="tab:blue", label="16-pro pool (avg)",
            edgecolor="black")
    axL.bar(x + width / 2, vg_rates, width, color="tab:orange", label="van Gerwen",
            edgecolor="black")
    axL.set_xticks(x)
    axL.set_xticklabels(targets)
    axL.set_ylabel("Hit rate (% of darts inside the target region)")
    axL.set_title("Pro hit rates: T20 ≫ T17\n(Haugh & Wang 2024 §3 p. 5)", fontsize=12)
    axL.legend(fontsize=10)
    axL.grid(alpha=0.3, axis="y")
    for i, v in enumerate(hit_rates):
        axL.text(x[i] - width / 2, v + 0.7, f"{v:.1f}%", ha="center", fontsize=9)
    for i, v in enumerate(vg_rates):
        axL.text(x[i] + width / 2, v + 0.7, f"{v:.1f}%", ha="center", fontsize=9, color="darkorange")

    axR.bar(targets, biases_mm, color="tab:red", edgecolor="black")
    axR.set_ylabel("Average |μ − region centre| (mm)")
    axR.set_title("Bias: pros aim dead-centre at T20, drift at T17\n"
                  "(Haugh & Wang 2024 Table 5 p. S-5)", fontsize=12)
    axR.grid(alpha=0.3, axis="y")
    for i, v in enumerate(biases_mm):
        axR.text(i, v + 0.07, f"{v:.2f}", ha="center", fontsize=10)

    fig.suptitle("Target dependence: pros are not equally good everywhere",
                 fontsize=13, y=1.03)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_3_7_target_dependence.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# FIG-3.8 — Six-tier ellipse panel (the headliner)
# ---------------------------------------------------------------------------


def fig_38_six_tiers():
    tiers = [
        ("perfect",        0.0,   0.0,   "tab:purple"),
        ("world_champion", 0.018, 0.018, "tab:cyan"),
        ("professional",   0.026, 0.026, "tab:blue"),
        ("good",           0.038, 0.076, "tab:green"),
        ("average",        0.074, 0.147, "tab:olive"),
        ("beginner",       0.126, 0.203, "tab:red"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for ax, (label, sx, sy, color) in zip(axes.flatten(), tiers):
        _draw_blank_board(ax, with_numbers=True)
        # Aim at the bullseye for the ellipse panel; this is the standard
        # comparison framing.
        if sx > 0:
            _add_ellipse(ax, [0, 0], sx, sy, n_sigma=1.0, color=color, linewidth=2.5)
            _add_ellipse(ax, [0, 0], sx, sy, n_sigma=2.0, color=color, linewidth=1.4,
                         linestyle="--")
        else:
            ax.plot(0, 0, marker="*", color=color, markersize=22, markeredgecolor="black")
        sx_mm = sx * board.TOTAL_DIAM
        sy_mm = sy * board.TOTAL_DIAM
        if sx == sy:
            ax.set_title(f"{label}\nσ = {sx_mm:.1f} mm (isotropic)", fontsize=12,
                         color=color)
        else:
            ax.set_title(f"{label}\nσ_x = {sx_mm:.1f}, σ_y = {sy_mm:.1f} mm  (ratio {sy/sx:.1f})",
                         fontsize=12, color=color)
    fig.suptitle("Six tiers of accuracy — the 1σ and 2σ error ellipses when aiming at the bullseye",
                 fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_3_8_six_tiers.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# FIG-3.9 — Skill ladder infographic
# ---------------------------------------------------------------------------


def fig_39_skill_ladder():
    # Rows: tier, σ_geom mm, expected-EV-per-dart (from the solver outputs),
    # one-line description.
    rows = [
        ("perfect",        0.0,   60.0,  "Mathematical ideal — every dart on T20"),
        ("world_champion", 6.0,   None,  "PDC peak (van Gerwen / Phil Taylor)"),
        ("professional",   8.5,   None,  "PDC top-16 average"),
        ("good",           18.4,  None,  "Lotze 'expert' lab group; competitive amateur"),
        ("average",        35.4,  None,  "Lotze 'novice' lab group; casual player"),
        ("beginner",       54.5,  None,  "Tibshirani Author 1; someone who barely plays"),
    ]
    # Plug in EV per dart from existing data files if available.
    import json
    data_dir = HERE.parent / "data"
    ev_lookup = {}
    for label in ["perfect", "world_champion", "professional", "good", "average", "beginner"]:
        meta_path = data_dir / f"ev_{label}.json"
        if not meta_path.exists():
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        npy_path = data_dir / f"ev_{label}.npy"
        if not npy_path.exists():
            continue
        ev = np.load(npy_path)
        ev_lookup[label] = float(ev.max())

    fig, ax = plt.subplots(figsize=(11.5, 6))
    ax.set_axis_off()
    headers = ["Tier", "σ_geom (mm)", "EV / dart (at optimal aim)", "Description"]
    col_x = [0.02, 0.21, 0.39, 0.58]
    for x, h in zip(col_x, headers):
        ax.text(x, 0.94, h, transform=ax.transAxes, fontsize=12, fontweight="bold")
    ax.plot([0, 1], [0.91, 0.91], color="black", linewidth=0.8, transform=ax.transAxes)

    row_h = 0.13
    for k, (label, sigma_mm, _, desc) in enumerate(rows):
        y = 0.83 - k * row_h
        color = {"perfect": "tab:purple", "world_champion": "tab:cyan",
                 "professional": "tab:blue", "good": "tab:green",
                 "average": "tab:olive", "beginner": "tab:red"}[label]
        ax.text(col_x[0], y, label, transform=ax.transAxes, fontsize=13,
                color=color, fontweight="bold")
        ax.text(col_x[1], y, f"{sigma_mm:.1f}", transform=ax.transAxes, fontsize=12)
        ev_text = (f"{ev_lookup.get(label, float('nan')):.2f}"
                   if label in ev_lookup else "—")
        ax.text(col_x[2], y, ev_text, transform=ax.transAxes, fontsize=12)
        ax.text(col_x[3], y, desc, transform=ax.transAxes, fontsize=11)
    ax.set_title("Skill ladder — what σ means in match terms",
                 fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_3_9_skill_ladder.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# FIG-3.10 — How many throws do you need
# ---------------------------------------------------------------------------


def fig_310_n_throws():
    """For a known true (σ_x, σ_y), simulate N throws and fit; report CI."""
    rng = np.random.default_rng(7)
    true_sx_mm = 18.0
    true_sy_mm = 36.0  # 2:1 amateur ratio
    Ns = [10, 20, 30, 50, 75, 100, 150, 250, 500]
    reps = 400
    sx_means, sx_los, sx_his = [], [], []
    sy_means, sy_los, sy_his = [], [], []
    for N in Ns:
        sx_estimates = []
        sy_estimates = []
        for _ in range(reps):
            xs = rng.normal(0, true_sx_mm, N)
            ys = rng.normal(0, true_sy_mm, N)
            sx_estimates.append(np.std(xs, ddof=1))
            sy_estimates.append(np.std(ys, ddof=1))
        sx_arr = np.array(sx_estimates)
        sy_arr = np.array(sy_estimates)
        sx_means.append(sx_arr.mean())
        sx_los.append(np.percentile(sx_arr, 2.5))
        sx_his.append(np.percentile(sx_arr, 97.5))
        sy_means.append(sy_arr.mean())
        sy_los.append(np.percentile(sy_arr, 2.5))
        sy_his.append(np.percentile(sy_arr, 97.5))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axhline(true_sx_mm, color="tab:red", linestyle=":", alpha=0.6,
               label=f"true σ_x = {true_sx_mm} mm")
    ax.axhline(true_sy_mm, color="tab:blue", linestyle=":", alpha=0.6,
               label=f"true σ_y = {true_sy_mm} mm")
    ax.fill_between(Ns, sx_los, sx_his, color="tab:red", alpha=0.18)
    ax.plot(Ns, sx_means, "o-", color="tab:red", label="estimated σ_x (mean ± 95% CI)")
    ax.fill_between(Ns, sy_los, sy_his, color="tab:blue", alpha=0.18)
    ax.plot(Ns, sy_means, "o-", color="tab:blue", label="estimated σ_y (mean ± 95% CI)")
    ax.set_xscale("log")
    ax.set_xlabel("Number of throws N (log scale)")
    ax.set_ylabel("Fitted σ (mm)")
    ax.set_title("How many throws to estimate σ accurately?\n"
                 "Synthetic data with true σ_x = 18 mm, σ_y = 36 mm; sample std of N throws",
                 fontsize=12)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_3_10_n_throws.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    funcs = [
        ("FIG-3.1", fig_31_teaser),
        ("FIG-3.2", fig_32_model_diagram),
        ("FIG-3.3", fig_33_three_shapes),
        ("FIG-3.4", fig_34_axes),
        ("FIG-3.5", fig_35_tibshirani_vs_pro),
        ("FIG-3.6", fig_36_anisotropy_ratio),
        ("FIG-3.7", fig_37_target_dependence),
        ("FIG-3.8", fig_38_six_tiers),
        ("FIG-3.9", fig_39_skill_ladder),
        ("FIG-3.10", fig_310_n_throws),
    ]
    for tag, func in funcs:
        print(f"  generating {tag} ...")
        func()
    print(f"All figures written to {OUT_DIR}")


if __name__ == "__main__":
    main()
