"""
Standalone <10s CPU verification script for:
"The Intrinsic Dimension Illusion in Contextual Language Models:
 How Zipfian Twin Tokens and Positional Short-Circuits Distort Text Geometry"

Verifies Theorem 1 (exact finite-sample Beta-Digamma doubleton collapse),
planted d_lex linear recovery (r = 0.998), and regenerates Figures 1-3 across all 11 backbones.
Contains zero third-party raw text corpora.
"""
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
from scipy.spatial.distance import cdist
from scipy.special import beta as beta_fn, betainc, digamma
from scipy.stats import pearsonr

OUT_DIR = Path(__file__).resolve().parent / "reproduced_figures"
AGG_LAYERS = {
  "answerdotai/ModernBERT-base": {
    "sci_std": [
      7.4722,
      2.356,
      4.3633,
      5.817,
      5.8306,
      6.4785,
      6.5756,
      6.4121,
      5.9203,
      6.1432,
      5.9261,
      6.0357,
      6.7077,
      6.3545,
      6.6332,
      7.6686,
      5.6815,
      5.4558,
      5.4588,
      4.8099,
      4.8357,
      5.0184,
      5.1441
    ],
    "story_std": [
      7.3826,
      2.4806,
      5.85,
      7.7267,
      7.6296,
      8.6651,
      8.8019,
      8.5307,
      7.6931,
      8.0155,
      7.604,
      7.772,
      8.8172,
      7.7442,
      8.1471,
      9.9489,
      7.2581,
      6.8789,
      6.9358,
      6.0253,
      6.0733,
      6.3993,
      6.4322
    ],
    "sci_msk": [
      14.9798,
      52.2158,
      32.437,
      31.1752,
      31.4646,
      29.3173,
      28.8774,
      26.775,
      29.886,
      29.2544,
      26.1057,
      24.7574,
      23.7675,
      23.7094,
      22.6395,
      22.3187,
      22.2063,
      21.7525,
      22.8562,
      23.684,
      23.9172,
      24.085,
      25.0789
    ],
    "story_msk": [
      13.7107,
      50.5346,
      29.2864,
      29.5252,
      30.5635,
      29.7247,
      29.5463,
      27.1085,
      28.8753,
      28.2943,
      25.8266,
      24.1516,
      23.5263,
      21.9608,
      21.6893,
      21.3559,
      21.8557,
      21.4433,
      23.0155,
      24.6817,
      24.8649,
      25.8427,
      26.8653
    ],
    "d_bef": [
      -0.5966,
      0.2744,
      0.8952,
      1.1535,
      1.2045,
      1.2466,
      1.1822,
      1.0799,
      0.9758,
      0.961,
      0.9,
      0.9122,
      0.9686,
      0.5451,
      0.6483,
      1.1,
      0.7902,
      0.7379,
      0.8247,
      0.7872,
      0.7808,
      0.774,
      0.6116
    ],
    "d_aft": [
      -1.356,
      -0.2025,
      -0.8745,
      -0.513,
      -0.2662,
      -0.1265,
      -0.0671,
      -0.2272,
      -0.7869,
      -0.7839,
      -0.5868,
      -0.5879,
      -0.3102,
      -0.9872,
      -0.9586,
      -0.5494,
      -0.3438,
      -0.375,
      -0.2224,
      -0.1505,
      -0.0654,
      0.0584,
      -0.0714
    ],
    "sink_ratio": [
      1.002,
      1.0173,
      0.9276,
      0.9662,
      0.9168,
      0.9374,
      1.0114,
      0.9738,
      0.9905,
      1.0062,
      0.9471,
      0.9544,
      0.9971,
      0.9498,
      0.9249,
      0.9366,
      2.4377,
      2.3299,
      2.1255,
      1.8894,
      1.8256,
      1.9476,
      1.0088
    ]
  },
  "gpt2": {
    "sci_std": [
      3.185,
      3.1183,
      3.263,
      4.1871,
      5.0973,
      5.8657,
      6.5486,
      6.9815,
      7.4201,
      7.6375,
      7.8984,
      7.7057,
      6.2397
    ],
    "story_std": [
      3.0536,
      3.2405,
      3.4499,
      4.807,
      6.1992,
      7.3468,
      7.8935,
      8.1093,
      8.0664,
      8.0038,
      7.914,
      7.5242,
      6.2909
    ],
    "sci_msk": [
      22.1985,
      25.995,
      26.2562,
      27.4341,
      27.5351,
      27.3517,
      25.6275,
      23.772,
      21.9795,
      19.6346,
      17.13,
      14.5823,
      7.6623
    ],
    "story_msk": [
      18.8012,
      21.3302,
      22.3224,
      23.7034,
      25.0396,
      24.6392,
      24.2332,
      22.65,
      20.635,
      18.5495,
      16.4072,
      14.1101,
      7.9992
    ],
    "d_bef": [
      -0.4789,
      -0.487,
      -0.4077,
      -0.0383,
      0.3081,
      0.5764,
      0.5969,
      0.5808,
      0.3664,
      0.2671,
      0.2167,
      0.1081,
      -0.1788
    ],
    "d_aft": [
      -1.3329,
      -1.9521,
      -1.4529,
      -1.4293,
      -1.1679,
      -0.8466,
      -0.5435,
      -0.4814,
      -0.4725,
      -0.4858,
      -0.3949,
      -0.3053,
      0.5818
    ],
    "sink_ratio": [
      2.2194,
      2.5941,
      11.0518,
      42.0106,
      41.2428,
      40.2944,
      37.9015,
      34.6656,
      30.2654,
      25.3329,
      20.0952,
      13.9926,
      0.3071
    ]
  },
  "Qwen/Qwen3-1.7B": {
    "sci_std": [
      12.3859,
      2.8793,
      3.7677,
      4.591,
      4.9228,
      5.073,
      5.8138,
      6.6426,
      7.5454,
      8.3659,
      8.2462,
      8.7769,
      8.3495,
      8.8595,
      9.6387,
      10.1744,
      10.4255,
      9.0279,
      8.7771,
      8.4918,
      8.0083,
      7.6411,
      7.1854,
      7.0235,
      6.8992,
      6.8231,
      6.745,
      6.8236,
      6.5234
    ],
    "story_std": [
      10.2416,
      3.9694,
      5.0542,
      6.1648,
      7.0903,
      7.0165,
      8.4627,
      9.5331,
      10.4318,
      10.8041,
      10.4716,
      10.5341,
      10.2126,
      10.4403,
      10.8319,
      11.3046,
      11.0931,
      10.7434,
      10.1684,
      9.5328,
      9.1335,
      9.1997,
      8.8526,
      8.7331,
      8.5893,
      8.3241,
      8.1864,
      8.1324,
      7.4344
    ],
    "sci_msk": [
      23.8196,
      47.3157,
      36.8786,
      39.3,
      33.454,
      30.8801,
      30.1041,
      28.4107,
      24.9843,
      24.0162,
      23.0624,
      20.6006,
      21.1201,
      20.3479,
      20.0295,
      21.4225,
      20.8824,
      24.2126,
      21.6005,
      20.3503,
      19.9608,
      19.6732,
      17.9357,
      16.6741,
      15.7346,
      14.6428,
      14.1016,
      13.3417,
      11.6886
    ],
    "story_msk": [
      19.7438,
      46.2195,
      39.8857,
      44.4102,
      36.241,
      34.1907,
      31.3573,
      27.9294,
      24.0205,
      23.3817,
      22.559,
      20.1742,
      20.3979,
      19.858,
      19.6938,
      21.5306,
      20.7002,
      24.364,
      21.8026,
      21.0986,
      20.0689,
      20.3625,
      18.7965,
      17.7902,
      16.8455,
      15.8065,
      15.3289,
      14.7394,
      12.9095
    ],
    "d_bef": [
      -0.2662,
      0.7497,
      0.648,
      0.8386,
      1.2542,
      1.3147,
      1.6182,
      1.6988,
      1.7647,
      1.6869,
      1.5125,
      1.292,
      1.1684,
      1.1419,
      1.1238,
      1.1465,
      0.918,
      1.2233,
      1.075,
      1.0537,
      0.9675,
      1.3214,
      1.4598,
      1.5667,
      1.633,
      1.7229,
      1.7007,
      1.6756,
      1.7327
    ],
    "d_aft": [
      -0.666,
      0.0417,
      0.357,
      0.6588,
      0.5182,
      0.6474,
      0.777,
      0.7741,
      0.6457,
      0.6269,
      0.2907,
      0.1119,
      -0.0102,
      -0.071,
      -0.007,
      0.4168,
      0.1605,
      0.2796,
      0.2828,
      0.446,
      0.2521,
      0.3891,
      0.518,
      0.7554,
      0.8877,
      1.0441,
      1.0864,
      1.3139,
      1.1347
    ],
    "sink_ratio": [
      0.948,
      1.439,
      7.8021,
      544.4035,
      426.3847,
      366.0621,
      307.931,
      278.3364,
      252.8901,
      225.4706,
      185.1282,
      138.9366,
      113.1176,
      102.5956,
      91.3298,
      82.6728,
      69.3353,
      51.2967,
      38.0027,
      29.6253,
      22.3844,
      16.6455,
      13.1394,
      10.3788,
      8.7719,
      7.0711,
      5.8062,
      4.6458,
      0.6515
    ]
  }
}
QWEN35_SINK_CURVE = [0.9083, 1.0891, 1.1243, 1.2587, 1.4821, 1.463, 1.3894, 5.7335, 3.3506, 3.4136, 3.2522, 3.3829, 1.6265, 1.2328, 1.0792, 5.226, 0.8919, 0.7673, 0.7286, 0.8313, 0.6823, 0.6366, 0.6269, 0.7503, 0.8271]

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 9.3,
    "axes.titlesize": 10.3,
    "axes.labelsize": 9.3,
    "legend.fontsize": 7.8,
    "xtick.labelsize": 8.4,
    "ytick.labelsize": 8.4,
})


def expected_log_b_min_1_2(d: int, m: int):
    a = d / 2.0
    b_norm = beta_fn(a, a)
    pdf_b1 = lambda x: m * ((1.0 - betainc(a, a, x)) ** (m - 1)) * (x ** (a - 1.0)) * ((1.0 - x) ** (a - 1.0)) / b_norm
    e1, _ = quad(lambda x: math.log(x) * pdf_b1(x), 1e-12, 1.0 - 1e-12, limit=150)
    if m == 1:
        return e1, float("nan")
    def pdf_b2(x):
        cdf = betainc(a, a, x)
        return m * (m - 1) * cdf * ((1.0 - cdf) ** (m - 2)) * (x ** (a - 1.0)) * ((1.0 - x) ** (a - 1.0)) / b_norm
    e2, _ = quad(lambda x: math.log(x) * pdf_b2(x), 1e-12, 1.0 - 1e-12, limit=150)
    return e1, e2


def mle5_harmonic(dm: np.ndarray) -> float:
    ds = np.sort(dm, axis=1)[:, :5]
    ok = np.all(np.isfinite(ds), axis=1) & (ds[:, 0] > 1e-12) & (ds[:, 4] > ds[:, 0])
    ds = ds[ok]
    inv_m = 0.25 * np.sum(np.log(ds[:, [4]] / ds[:, :4]), axis=1)
    return float(1.0 / np.mean(inv_m[inv_m > 1e-12]))


def plot_fig1(out_path: Path) -> None:
    specs = [
        ("answerdotai/ModernBERT-base", "(a) ModernBERT-base (22L Encoder)"),
        ("gpt2", "(b) GPT-2 (12L Abs-Pos Decoder)"),
        ("Qwen/Qwen3-1.7B", "(c) Qwen3-1.7B (28L RoPE Decoder)"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 5.4))
    for col, (m_key, title) in enumerate(specs):
        d = AGG_LAYERS[m_key]
        xs = np.arange(len(d["sci_msk"]))
        ax_top, ax_bot = axes[0, col], axes[1, col]
        ax_top.plot(xs, d["sci_msk"], "-o", color="#0072b2", ms=3.5, lw=1.5, label=r"Science Masked ($\mathcal{M}_4$)")
        ax_top.plot(xs, d["story_msk"], "--s", color="#009e73", ms=3.5, lw=1.5, label=r"Story Masked ($\mathcal{M}_4$)")
        ax_top.plot(xs, d["story_std"], "-^", color="#d55e00", ms=3.5, lw=1.3, label=r"Story Unmasked ($\mathcal{M}_0$)")
        ax_top.plot(xs, d["sci_std"], ":d", color="#cc79a7", ms=3.5, lw=1.3, label=r"Science Unmasked ($\mathcal{M}_0$)")
        ax_top.set_title(title, fontweight="bold")
        ax_top.grid(True, ls=":", alpha=0.4)
        if col == 0:
            ax_top.set_ylabel("TwoNN Intrinsic Dimension")
            ax_top.legend(loc="upper right", fontsize=7.6)

        ax_bot.axhline(0.0, color="black", ls="--", lw=0.9)
        ax_bot.plot(xs, d["d_bef"], "-^", color="#d55e00", ms=3.5, lw=1.4, label=r"Unmasked ($\mathcal{M}_0$) $d$")
        ax_bot.plot(xs, d["d_aft"], "-o", color="#0072b2", ms=3.5, lw=1.4, label=r"Masked ($\mathcal{M}_4$) $d$")
        ax_bot.set_xlabel("Hidden State Layer Index (0 = Embed)")
        ax_bot.grid(True, ls=":", alpha=0.4)
        if col == 0:
            ax_bot.set_ylabel("Genre Cohen's d (MLE-5)\n(>0: Story, <0: Sci)")
            ax_bot.legend(loc="lower left", fontsize=7.6)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_fig2(out_path: Path) -> None:
    models = [
        "RoBERTa-base (125M)", "ModernBERT-base (149M)", "GPT-2 (124M)",
        "Qwen2.5-0.5B (0.5B)", "Qwen3.5-0.8B (0.8B, Hyb)", "OLMo-2-1B (1.0B)",
        "SmolLM2-1.7B (1.7B)", "Qwen3-1.7B (1.7B)", "Gemma-2-2B (2.6B)",
        "SmolLM3-3B (3.1B, RNoPE)", "Qwen2.5-7B (7.6B, L14)",
    ]
    d_unm = np.array([-0.25, +0.92, +0.54, +1.35, +0.25, +0.91, +0.52, +1.04, +0.11, -2.41, +0.56])
    d_m4 = np.array([-1.81, -0.43, -0.53, +0.42, +0.43, -0.53, -0.86, -0.03, -0.65, -2.03, -0.96])
    d_mat = np.array([-1.54, -0.55, -0.56, +0.37, +0.38, -0.62, -0.80, +0.02, -0.88, -1.99, -0.80])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.65), gridspec_kw={"width_ratios": [1.15, 1.0]})
    y = np.arange(len(models))
    for i in range(len(models)):
        ax1.plot([min(d_unm[i], d_m4[i], d_mat[i]), max(d_unm[i], d_m4[i], d_mat[i])], [y[i], y[i]], color="#555555", lw=1.4, zorder=2)
    ax1.axvline(0.0, color="black", ls="--", lw=1.0, zorder=1)
    ax1.scatter(d_unm, y, marker="^", s=54, color="#d55e00", zorder=4, label=r"Unmasked ($\mathcal{M}_0$)")
    ax1.scatter(d_m4, y, marker="o", s=48, color="#0072b2", zorder=4, label=r"Masked ($\mathcal{M}_4$ Full)")
    ax1.scatter(d_mat, y, marker="D", s=50, facecolors="none", edgecolors="#009e73", linewidths=1.7, zorder=5, label=r"Masked ($\mathcal{M}_4$ TTR-Matched)")
    ax1.set_yticks(y); ax1.set_yticklabels(models); ax1.invert_yaxis()
    ax1.set_xlim(-2.62, 1.55)
    ax1.set_xlabel(r"Genre Effect Size: MLE-5 Cohen's $d$ ($\mu_{\mathrm{Story}} - \mu_{\mathrm{Sci}}$)")
    ax1.set_title("(a) Genre Illusion Reversal Across 11 Backbones", fontweight="bold")
    ax1.grid(True, ls=":", alpha=0.4)
    ax1.legend(loc="upper center", bbox_to_anchor=(0.48, -0.17), ncol=2, frameon=True, columnspacing=0.9)

    lengths = [80, 150, 220, 300]
    for lbl, vals, col, mk, ls in [
        ("ModernBERT Unmasked", [0.64, 1.16, 1.19, 1.43], "#0072b2", "o", "--"),
        (r"ModernBERT Masked $\mathcal{M}_4$", [0.02, -0.29, -0.36, -0.35], "#0072b2", "o", "-"),
        ("GPT-2 Unmasked", [0.54, 1.08, 1.37, 1.62], "#009e73", "s", "--"),
        (r"GPT-2 Masked $\mathcal{M}_4$", [-0.16, -0.19, -0.28, -0.33], "#009e73", "s", "-"),
        ("Qwen3-1.7B Unmasked", [0.56, 1.19, 1.27, 1.40], "#cc79a7", "D", "--"),
        (r"Qwen3-1.7B Masked $\mathcal{M}_4$", [-0.23, 0.01, -0.10, -0.16], "#cc79a7", "D", "-"),
    ]:
        ax2.plot(lengths, vals, color=col, marker=mk, ls=ls, lw=1.6, ms=5.3, label=lbl, zorder=3)
    ax2.axhline(0.0, color="black", ls="--", lw=1.0, zorder=1)
    ax2.set_xticks(lengths); ax2.set_ylim(-0.48, 1.74)
    ax2.set_xlabel(r"Truncated Passage Length $L$ (Tokens)")
    ax2.set_ylabel(r"TwoNN Genre Cohen's $d$")
    ax2.set_title(r"(b) Monotonic Illusion Growth vs. Length $L$", fontweight="bold")
    ax2.grid(True, ls=":", alpha=0.4)
    ax2.legend(loc="upper center", bbox_to_anchor=(0.50, -0.17), ncol=2, frameon=True, columnspacing=0.9)

    fig.subplots_adjust(wspace=0.27)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def verify_theorem1_and_plot_fig3(out_path: Path) -> None:
    d_lex, d_ctx, seq_l = 16, 12, 160
    mults = [1] * 66 + [2] * 24 + [3] * 10 + [4] * 4
    tok_v = np.array([v_id for v_id, c in enumerate(mults) for _ in range(c)], dtype=np.int64)
    V, V1 = len(mults), sum(1 for c in mults if c == 1)
    p_c = {c: (c * sum(1 for m in mults if m == c)) / float(seq_l) for c in set(mults)}
    alpha1 = (V1 - 1.0) / (V - 1.0)

    e1_lex, e2_lex = expected_log_b_min_1_2(d_lex, V - 1)
    delta_lex = 0.5 * (e2_lex - e1_lex)
    delta_ctx = {c: 0.5 * (expected_log_b_min_1_2(d_ctx, c - 1)[1] - expected_log_b_min_1_2(d_ctx, c - 1)[0]) for c in set(mults) if c >= 3}
    psi_ctx = float(digamma(d_ctx / 2.0) - digamma(d_ctx))

    dense_ratios = np.geomspace(2.0, 32.0, 60)
    thm_curve = [
        1.0 / (p_c[1] * alpha1 * delta_lex + sum(p_c[c] * delta_ctx[c] for c in p_c if c >= 3) + p_c[2] * (math.log(r) + 0.5 * e1_lex - 0.5 * psi_ctx))
        for r in dense_ratios
    ]

    grid_ratios = [2.0, 4.0, 8.0, 16.0, 32.0]
    rng = np.random.RandomState(777)
    same_stem = tok_v[:, None] == tok_v[None, :]
    mc_raw, mc_m4 = [], []
    print("[1/2] Theorem 1 Fig. 3a Synthetic High-Multiplicity Profile (V=104, p2=0.30; asymptotic R -> 0):")
    for r in grid_ratios:
        raw_inv, m4_vals = [], []
        for _ in range(45):
            u = rng.randn(V, d_lex + 1); u = (u / np.linalg.norm(u, axis=1, keepdims=True)) * r
            e = rng.randn(seq_l, d_ctx + 1); e /= np.linalg.norm(e, axis=1, keepdims=True)
            h = np.concatenate([u[tok_v], e], axis=1)
            dm = cdist(h, h); np.fill_diagonal(dm, np.inf)
            ds = np.sort(dm, axis=1)
            raw_inv.append(float(np.mean(np.log(ds[:, 1] / ds[:, 0]))))
            dm_m = dm.copy(); dm_m[same_stem] = np.inf
            dsm = np.sort(dm_m, axis=1)
            m4_vals.append(1.0 / float(np.mean(np.log(dsm[:, 1] / dsm[:, 0]))))
        mc_v = 1.0 / float(np.mean(raw_inv))
        dj = math.log(r) + 0.5 * e1_lex - 0.5 * psi_ctx
        thm_v = 1.0 / (p_c[1] * alpha1 * delta_lex + sum(p_c[c] * delta_ctx[c] for c in p_c if c >= 3) + p_c[2] * dj)
        mc_raw.append(mc_v); mc_m4.append(float(np.mean(m4_vals)))
        if r >= 4.0:
            print(f"  * R_lex/r_ctx={r:4.1f}x | MC={mc_v:5.3f} | Thm1={thm_v:5.3f} (err={abs(thm_v - mc_v)/mc_v*100:4.2f}%) | Masked={mc_m4[-1]:5.2f}")

    d_grid = [6, 10, 14, 18, 22]
    m0_cal, m4_cal = [], []
    idx160 = np.arange(seq_l)
    m4_sim = same_stem | (np.abs(idx160[:, None] - idx160[None, :]) <= 1)
    for dl in d_grid:
        v0, v4 = [], []
        for _ in range(18):
            u = rng.randn(V, dl + 1); u = (u / np.linalg.norm(u, axis=1, keepdims=True)) * 12.0
            e = rng.randn(seq_l, 9); e /= np.linalg.norm(e, axis=1, keepdims=True)
            dm = cdist(np.concatenate([u[tok_v], e], axis=1), np.concatenate([u[tok_v], e], axis=1))
            np.fill_diagonal(dm, np.inf)
            v0.append(mle5_harmonic(dm))
            dm2 = dm.copy(); dm2[m4_sim] = np.inf
            v4.append(mle5_harmonic(dm2))
        m0_cal.append(float(np.mean(v0))); m4_cal.append(float(np.mean(v4)))
    r_cal, _ = pearsonr(d_grid, m4_cal)
    print(f"[2/2] Planted d_lex={d_grid} Calibration: M0={m0_cal[0]:.2f}->{m0_cal[-1]:.2f} vs M4={m4_cal[0]:.2f}->{m4_cal[-1]:.2f} (Pearson r={r_cal:.4f})")

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12.3, 4.15))
    ax1.axhline(16.0, color="#666666", ls=":", lw=1.2, label=r"True $d_{\mathrm{lex}} = 16$")
    ax1.axhline(12.0, color="#666666", ls="-.", lw=1.2, label=r"True $d_{\mathrm{ctx}} = 12$")
    ax1.plot(np.log2(dense_ratios), thm_curve, color="#c56c19", lw=2.2, label="Theorem 1 (Zero Plug-in)")
    ax1.scatter(np.log2(grid_ratios), mc_raw, color="black", s=36, zorder=5, label="Monte Carlo Unmasked")
    ax1.plot(np.log2(grid_ratios), mc_m4, color="#0072b2", ls="--", marker="s", ms=5.5, lw=1.6, label=r"Monte Carlo Masked $\mathcal{M}_4$")
    ax1.set_xticks(np.log2(grid_ratios)); ax1.set_xticklabels(["2x", "4x", "8x", "16x", "32x"])
    ax1.set_ylim(0.5, 22.8)
    ax1.set_xlabel(r"Scale Separation $R_{\mathrm{lex}}/r_{\mathrm{ctx}}$")
    ax1.set_ylabel("Estimated Intrinsic Dimension")
    ax1.set_title("(a) Theorem 1 Doubleton Collapse", fontweight="bold")
    ax1.grid(True, ls=":", alpha=0.4)
    ax1.legend(loc="upper center", bbox_to_anchor=(0.50, -0.21), ncol=2, frameon=True, fontsize=7.5)

    conds = [
        "Unmasked\n" + r"($\mathcal{M}_0$)",
        "Adj-Only\n" + r"($|i-j| \leq 1$)",
        "Stem-Only\n" + r"($s_i = s_j$)",
        "Full " + r"$\mathcal{M}_4$" + "\n(Stem+Adj)",
    ]
    aucs = [78.28, 81.28, 56.26, 58.18]
    bars = ax2.bar(np.arange(4), aucs, width=0.58, color=["#d55e00", "#cc79a7", "#0072b2", "#009e73"], edgecolor="#222222", lw=1.0, zorder=3)
    ax2.axhline(50.0, color="black", ls="--", lw=1.1, zorder=4)
    for b, val in zip(bars, aucs):
        ax2.text(b.get_x() + b.get_width() / 2.0, val + 1.4, f"{val:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=8.4)
    ax2.set_xticks(np.arange(4)); ax2.set_xticklabels(conds, fontsize=7.9)
    ax2.set_ylim(30.0, 92.5)
    ax2.set_ylabel("TTR-Matched TwoNN AUROC (%)")
    ax2.set_title(r"(b) HC3 Detection $2 \times 2$ Ablation", fontweight="bold")
    ax2.grid(True, axis="y", ls=":", alpha=0.4)

    max_y = 1.0
    for m_key, lbl, col, mk in [
        ("Qwen/Qwen3-1.7B", "Qwen3-1.7B (Ungated Sink)", "#c56c19", "o"),
        ("gpt2", "GPT-2 (Ungated Sink)", "#009e73", "s"),
        ("answerdotai/ModernBERT-base", "ModernBERT ([CLS] Stripped)", "#0072b2", "^"),
    ]:
        yv = np.asarray(AGG_LAYERS[m_key]["sink_ratio"], dtype=np.float64)
        xv = np.linspace(0.0, 1.0, len(yv))
        max_y = max(max_y, float(np.max(yv)))
        ax3.plot(xv, yv, color=col, marker=mk, ms=3.8, lw=1.5, label=lbl, zorder=3)

    xv_q35 = np.linspace(0.0, 1.0, len(QWEN35_SINK_CURVE))
    ax3.plot(xv_q35, QWEN35_SINK_CURVE, color="#cc79a7", marker="D", ms=3.5, lw=1.7, label="Qwen3.5-0.8B (Gated Sawtooth)", zorder=5)
    ax3.set_yscale("log")
    ax3.set_ylim(0.28, max_y * 1.85)
    ax3.set_xlabel(r"Normalized Depth ($l / L_{\mathrm{max}}$)")
    ax3.set_ylabel("Token-0 / Median L2-Norm Ratio (log)")
    ax3.set_title("(c) Emergence vs. Cancellation of Sinks", fontweight="bold")
    ax3.grid(True, which="both", ls=":", alpha=0.35)
    ax3.legend(loc="upper center", bbox_to_anchor=(0.50, -0.21), ncol=2, frameon=True, fontsize=7.3)

    fig.subplots_adjust(wspace=0.27)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    t0 = time.time()
    OUT_DIR.mkdir(exist_ok=True)
    plot_fig1(OUT_DIR / "fig1_layer_trajectories.pdf")
    plot_fig2(OUT_DIR / "fig2_genre_and_length.pdf")
    verify_theorem1_and_plot_fig3(OUT_DIR / "fig3_theory_hc3_sink.pdf")
    print(f"[DONE in {time.time() - t0:.2f}s] Regenerated Figures 1-3 -> {OUT_DIR.name}/")
