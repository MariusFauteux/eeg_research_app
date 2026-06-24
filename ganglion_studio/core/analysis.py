"""EEG signal-analysis and electrode-characterization metrics + figures.

Provides per-channel metrics, pairwise agreement metrics (for comparing two
electrode types recorded simultaneously), group statistics, and matplotlib
figure builders used by the Processing Lab's analysis report.

Plot choices follow conventions common in the dry/PEDOT vs Ag/AgCl electrode
literature: log-log PSD overlays, noise/line-noise bars, magnitude-squared
coherence, correlation scatter, and Bland-Altman agreement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import coherence as _coherence
from scipy.signal import correlate as _correlate
from scipy.stats import pearsonr, spearmanr, ttest_ind

from matplotlib.figure import Figure

from ganglion_studio import palette
# CHANNEL_TYPES / ELECTRODES live in board_config (single source of truth) and are
# re-exported here so the live Channel Setup dialog and the Processing Lab (which
# imports them from this module) always offer identical lists.
from .board_config import CHANNEL_TYPES, ELECTRODES  # noqa: F401  (re-exported)
from .dsp import EEG_BANDS, compute_psd, dominant_frequency, signal_quality

# Electrode-material colours come from the central palette.
_ELECTRODE_COLORS = palette.ELECTRODE_COLORS


@dataclass
class ChannelMeta:
    index: int
    name: str
    ch_type: str = "EEG"
    electrode: str = "Ag/AgCl (wet)"
    enabled: bool = True

    @property
    def is_eeg(self) -> bool:
        return self.ch_type == "EEG"

    @property
    def is_pedot(self) -> bool:
        return self.electrode.startswith("PEDOT")

    @property
    def is_agagcl(self) -> bool:
        return self.electrode.startswith("Ag/AgCl")

    @property
    def material(self) -> str:
        """Coarse material group used for comparison/grouping."""
        if self.is_pedot:
            return "PEDOT"
        if self.is_agagcl:
            return "Ag/AgCl"
        return "Other"


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
# numpy 2.x renamed trapz -> trapezoid; support both.
_trapz = getattr(np, "trapezoid", None) or np.trapz


def _band_powers(freqs: np.ndarray, psd: np.ndarray) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for name, lo, hi in EEG_BANDS:
        mask = (freqs >= lo) & (freqs < hi)
        out[name] = float(_trapz(psd[mask], freqs[mask])) if np.any(mask) else 0.0
    return out


def channel_metrics(channel: np.ndarray, sampling_rate: int) -> dict:
    """Per-channel signal/noise metrics for characterization."""
    sq = signal_quality(channel, sampling_rate)
    freqs, psd = compute_psd(np.ascontiguousarray(channel), sampling_rate)
    bands = _band_powers(freqs, psd) if freqs.size else {b[0]: 0.0 for b in EEG_BANDS}
    total = float(sum(bands.values())) + 1e-12
    alpha = bands.get("Alpha", 0.0)
    # Alpha-band power relative to the other EEG bands, in dB. This is an
    # alpha-dominance ratio, NOT a true signal-to-noise ratio.
    alpha_ratio_db = 10.0 * np.log10(alpha / (total - alpha + 1e-12) + 1e-12)
    return {
        "rms": sq["rms"],
        "ptp": sq["ptp"],
        "line_ratio": sq["line_ratio"],
        "dominant_hz": dominant_frequency(channel, sampling_rate),
        "alpha_ratio_db": float(alpha_ratio_db),
        "bands": bands,
        "rel_bands": {k: v / total for k, v in bands.items()},
    }


def pair_agreement(x: np.ndarray, y: np.ndarray, sampling_rate: int) -> dict:
    """Agreement metrics between two simultaneously-recorded channels."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = min(x.size, y.size)
    x, y = x[:n], y[:n]
    try:
        r, _p = pearsonr(x, y)
    except Exception:
        r = float("nan")
    try:
        rho, _ps = spearmanr(x, y)
    except Exception:
        rho = float("nan")
    # RMSE = typical sample-to-sample difference; NRMSE normalizes it by the
    # signal range so 0 = identical traces, regardless of amplitude.
    rmse = float(np.sqrt(np.mean((x - y) ** 2)))
    rng = float(np.ptp(np.concatenate([x, y]))) + 1e-12
    nrmse = rmse / rng
    # Magnitude-squared coherence: how linearly related the two signals are at
    # each frequency, from 0 (unrelated) to 1 (perfectly related).
    nperseg = int(min(256, max(32, n // 8)))
    f, cxy = _coherence(x, y, fs=sampling_rate, nperseg=nperseg)
    band_mask = (f >= 1.0) & (f <= 30.0)
    mean_coh = float(np.mean(cxy[band_mask])) if np.any(band_mask) else float("nan")
    band_coh = {}
    for name, lo, hi in EEG_BANDS:
        m = (f >= lo) & (f < hi)
        band_coh[name] = float(np.mean(cxy[m])) if np.any(m) else float("nan")
    # Bland-Altman agreement: bias = average difference between the channels;
    # limits of agreement = bias +/- 1.96 SD (where ~95% of differences fall).
    diff = x - y
    bias = float(np.mean(diff))
    sd = float(np.std(diff))
    return {
        "r": float(r),
        "spearman": float(rho),
        "rmse": rmse,
        "nrmse": float(nrmse),
        "coh_freqs": f,
        "coherence": cxy,
        "mean_coherence_1_30": mean_coh,
        "band_coherence": band_coh,
        "ba_bias": bias,
        "ba_loa_low": bias - 1.96 * sd,
        "ba_loa_high": bias + 1.96 * sd,
        "rms_x": float(np.sqrt(np.mean((x - x.mean()) ** 2))),
        "rms_y": float(np.sqrt(np.mean((y - y.mean()) ** 2))),
    }


def eeg_metas(metas: List[ChannelMeta]) -> List[ChannelMeta]:
    return [m for m in metas if m.is_eeg]


def comparison_available(metas: List[ChannelMeta]) -> bool:
    eeg = eeg_metas(metas)
    return any(m.is_pedot for m in eeg) and any(m.is_agagcl for m in eeg)


def group_band_stats(eeg: np.ndarray, sampling_rate: int,
                     metas: List[ChannelMeta]) -> dict:
    """Per-material mean/std relative band power + t-test p-values (PEDOT vs Ag/AgCl)."""
    groups: Dict[str, List[np.ndarray]] = {"PEDOT": [], "Ag/AgCl": []}
    for m in eeg_metas(metas):
        if m.material not in groups:
            continue
        metrics = channel_metrics(eeg[m.index], sampling_rate)
        groups[m.material].append(np.array([metrics["rel_bands"][b[0]] for b in EEG_BANDS]))

    result = {"bands": [b[0] for b in EEG_BANDS], "groups": {}, "pvals": {}}
    for name, rows in groups.items():
        if rows:
            arr = np.vstack(rows)
            result["groups"][name] = {"mean": arr.mean(0), "std": arr.std(0), "n": len(rows)}
    a = groups.get("PEDOT")
    b = groups.get("Ag/AgCl")
    if a and b and len(a) >= 2 and len(b) >= 2:
        aa, bb = np.vstack(a), np.vstack(b)
        pvals = []
        for i in range(len(EEG_BANDS)):
            try:
                _t, p = ttest_ind(aa[:, i], bb[:, i], equal_var=False)
            except Exception:
                p = float("nan")
            pvals.append(float(p))
        result["pvals"] = pvals
    return result


# --------------------------------------------------------------------------- #
# Figure builders
# --------------------------------------------------------------------------- #
def _new_fig(w=8.0, h=5.0) -> Figure:
    fig = Figure(figsize=(w, h), layout="constrained")
    return fig


def _ch_color(meta: ChannelMeta) -> str:
    return _ELECTRODE_COLORS.get(meta.electrode, "#cccccc")


def fig_psd(eeg: np.ndarray, sr: int, metas: List[ChannelMeta]) -> Figure:
    fig = _new_fig()
    ax = fig.add_subplot(111)
    for m in eeg_metas(metas):
        freqs, psd = compute_psd(np.ascontiguousarray(eeg[m.index]), sr)
        if freqs.size:
            ax.semilogy(freqs, psd, label=f"{m.name} ({m.electrode})", lw=1.2)
    for _name, lo, hi in EEG_BANDS:
        ax.axvspan(lo, hi, alpha=0.05, color="gray")
    for name, lo, hi in EEG_BANDS:
        ax.text((lo + hi) / 2, ax.get_ylim()[1], name, ha="center", va="top", fontsize=8, color="gray")
    ax.set_xlim(0, min(70, sr / 2))
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(r"PSD ($\mu V^2$/Hz)")
    ax.set_title("Power spectral density (Welch) - EEG channels")
    ax.grid(True, which="both", alpha=0.2)
    ax.legend(fontsize=8)
    return fig


def fig_band_powers(eeg: np.ndarray, sr: int, metas: List[ChannelMeta]) -> Figure:
    fig = _new_fig()
    ax = fig.add_subplot(111)
    eegm = eeg_metas(metas)
    band_names = [b[0] for b in EEG_BANDS]
    bottoms = np.zeros(len(eegm))
    x = np.arange(len(eegm))
    for bi, bname in enumerate(band_names):
        vals = np.array([channel_metrics(eeg[m.index], sr)["rel_bands"][bname] for m in eegm])
        ax.bar(x, vals, bottom=bottoms, label=bname)
        bottoms += vals
    ax.set_xticks(x)
    ax.set_xticklabels([f"{m.name}\n{m.electrode}" for m in eegm], fontsize=8)
    ax.set_ylabel("Relative band power")
    ax.set_title("Relative EEG band power per channel")
    ax.legend(fontsize=8, ncol=len(band_names))
    return fig


def fig_quality_table(eeg: np.ndarray, sr: int, metas: List[ChannelMeta]) -> Figure:
    fig = _new_fig(8.0, 0.5 + 0.4 * (len(metas) + 1))
    ax = fig.add_subplot(111)
    ax.axis("off")
    cols = ["Channel", "Type", "Electrode", "RMS (uV)", "P-P (uV)", "Line %", "Alpha ratio (dB)", "Dom (Hz)"]
    rows = []
    for m in metas:
        mt = channel_metrics(eeg[m.index], sr)
        rows.append([
            m.name, m.ch_type, m.electrode,
            f"{mt['rms']:.1f}", f"{mt['ptp']:.0f}",
            f"{mt['line_ratio'] * 100:.0f}", f"{mt['alpha_ratio_db']:.1f}", f"{mt['dominant_hz']:.1f}",
        ])
    table = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.3)
    ax.set_title("Per-channel signal quality / noise summary")
    return fig


def fig_char_noise(eeg: np.ndarray, sr: int, metas: List[ChannelMeta]) -> Figure:
    fig = _new_fig(8.0, 6.0)
    eegm = eeg_metas(metas)
    x = np.arange(len(eegm))
    labels = [f"{m.name}\n{m.electrode}" for m in eegm]
    colors = [_ch_color(m) for m in eegm]
    rms = [channel_metrics(eeg[m.index], sr)["rms"] for m in eegm]
    line = [channel_metrics(eeg[m.index], sr)["line_ratio"] * 100 for m in eegm]

    ax1 = fig.add_subplot(211)
    ax1.bar(x, rms, color=colors)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8)
    ax1.set_ylabel("RMS (uV)")
    ax1.set_title("Baseline noise (RMS) by channel / electrode")

    ax2 = fig.add_subplot(212)
    ax2.bar(x, line, color=colors)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_ylabel("Line noise (%)")
    ax2.set_title("Mains-noise fraction (50/60 Hz) by channel / electrode")
    return fig


def fig_char_psd_by_material(eeg: np.ndarray, sr: int, metas: List[ChannelMeta]) -> Figure:
    fig = _new_fig()
    ax = fig.add_subplot(111)
    seen = set()
    for m in eeg_metas(metas):
        freqs, psd = compute_psd(np.ascontiguousarray(eeg[m.index]), sr)
        if not freqs.size:
            continue
        label = m.electrode if m.electrode not in seen else None
        seen.add(m.electrode)
        ax.semilogy(freqs, psd, color=_ch_color(m), lw=1.0, alpha=0.8, label=label)
    ax.set_xlim(0, min(70, sr / 2))
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(r"PSD ($\mu V^2$/Hz)")
    ax.set_title("PSD coloured by electrode material")
    ax.grid(True, which="both", alpha=0.2)
    ax.legend(fontsize=8)
    return fig


# --- comparison figures ---------------------------------------------------- #
def _group_psd(eeg, sr, metas, material) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    psds = []
    freqs = np.array([])
    for m in eeg_metas(metas):
        if m.material != material:
            continue
        f, p = compute_psd(np.ascontiguousarray(eeg[m.index]), sr)
        if f.size:
            freqs = f
            psds.append(p)
    if not psds:
        return freqs, np.array([]), np.array([])
    arr = np.vstack(psds)
    return freqs, arr.mean(0), arr.std(0)


def fig_cmp_psd(eeg: np.ndarray, sr: int, metas: List[ChannelMeta]) -> Figure:
    fig = _new_fig()
    ax = fig.add_subplot(111)
    for material, color in (("PEDOT", palette.MATERIAL_PEDOT), ("Ag/AgCl", palette.MATERIAL_AGAGCL)):
        f, mean, std = _group_psd(eeg, sr, metas, material)
        if f.size and mean.size:
            ax.semilogy(f, mean, color=color, lw=1.6, label=f"{material} (mean)")
            ax.fill_between(f, np.maximum(mean - std, 1e-12), mean + std, color=color, alpha=0.2)
    ax.set_xlim(0, min(70, sr / 2))
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(r"PSD ($\mu V^2$/Hz)")
    ax.set_title("Group-mean PSD +/- std: PEDOT vs Ag/AgCl")
    ax.grid(True, which="both", alpha=0.2)
    ax.legend(fontsize=9)
    return fig


def fig_cmp_coherence(x: np.ndarray, y: np.ndarray, sr: int, ag: dict,
                      label_x: str, label_y: str) -> Figure:
    fig = _new_fig()
    ax = fig.add_subplot(111)
    ax.plot(ag["coh_freqs"], ag["coherence"], color=palette.GOOD, lw=1.4)
    ax.axhline(1.0, color="gray", lw=0.6, ls=":")
    ax.set_xlim(0, min(70, sr / 2))
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude-squared coherence")
    ax.set_title(f"Coherence: {label_x} vs {label_y}\nmean(1-30 Hz) = {ag['mean_coherence_1_30']:.2f}")
    ax.grid(True, alpha=0.2)
    return fig


def fig_cmp_correlation(x: np.ndarray, y: np.ndarray, ag: dict,
                        label_x: str, label_y: str) -> Figure:
    fig = _new_fig(6.0, 6.0)
    ax = fig.add_subplot(111)
    n = min(x.size, y.size)
    step = max(1, n // 3000)
    xs, ys = x[:n:step], y[:n:step]
    ax.scatter(xs, ys, s=4, alpha=0.3, color=palette.ACCENT)
    lim = float(np.nanmax(np.abs(np.concatenate([xs, ys])))) if xs.size else 1.0
    ax.plot([-lim, lim], [-lim, lim], color="gray", lw=0.8, ls="--", label="y = x")
    ax.set_xlabel(f"{label_x} (uV)")
    ax.set_ylabel(f"{label_y} (uV)")
    ax.set_title(f"Sample-by-sample agreement\nr = {ag['r']:.3f}, RMSE = {ag['rmse']:.1f} uV")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)
    return fig


def fig_cmp_bland_altman(x: np.ndarray, y: np.ndarray, ag: dict,
                         label_x: str, label_y: str) -> Figure:
    fig = _new_fig()
    ax = fig.add_subplot(111)
    n = min(x.size, y.size)
    step = max(1, n // 3000)
    mean = (x[:n:step] + y[:n:step]) / 2.0
    diff = x[:n:step] - y[:n:step]
    ax.scatter(mean, diff, s=4, alpha=0.3, color=palette.VIOLET)
    ax.axhline(ag["ba_bias"], color="#333", lw=1.0, label=f"bias = {ag['ba_bias']:.2f}")
    ax.axhline(ag["ba_loa_high"], color=palette.BAD, lw=0.9, ls="--",
               label=f"+1.96 SD = {ag['ba_loa_high']:.2f}")
    ax.axhline(ag["ba_loa_low"], color=palette.BAD, lw=0.9, ls="--",
               label=f"-1.96 SD = {ag['ba_loa_low']:.2f}")
    ax.set_xlabel(f"Mean of {label_x} & {label_y} (uV)")
    ax.set_ylabel(f"Difference ({label_x} - {label_y}) (uV)")
    ax.set_title("Bland-Altman agreement")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)
    return fig


def fig_pair_timeseries(x: np.ndarray, y: np.ndarray, sr: int,
                        label_x: str, label_y: str, window_s: float = 5.0) -> Figure:
    fig = _new_fig(8.0, 4.0)
    ax = fig.add_subplot(111)
    n = int(min(len(x), len(y), window_s * sr))
    t = np.arange(n) / sr
    ax.plot(t, x[:n], color=palette.BAD, lw=0.9, label=label_x)
    ax.plot(t, y[:n], color=palette.ACCENT, lw=0.9, label=label_y)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (uV)")
    ax.set_title(f"Overlaid time series (first {window_s:.0f} s)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    return fig


def fig_pair_psd(x: np.ndarray, y: np.ndarray, sr: int,
                 label_x: str, label_y: str) -> Figure:
    fig = _new_fig()
    ax = fig.add_subplot(111)
    fx, px = compute_psd(np.ascontiguousarray(x), sr)
    fy, py = compute_psd(np.ascontiguousarray(y), sr)
    if fx.size:
        ax.semilogy(fx, px, color=palette.BAD, lw=1.4, label=label_x)
    if fy.size:
        ax.semilogy(fy, py, color=palette.ACCENT, lw=1.4, label=label_y)
    ax.set_xlim(0, min(70, sr / 2))
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(r"PSD ($\mu V^2$/Hz)")
    ax.set_title(f"PSD overlay: {label_x} vs {label_y}")
    ax.grid(True, which="both", alpha=0.2)
    ax.legend(fontsize=9)
    return fig


def fig_pair_histogram(x: np.ndarray, y: np.ndarray,
                       label_x: str, label_y: str) -> Figure:
    fig = _new_fig(7.0, 4.0)
    ax = fig.add_subplot(111)
    lim = float(np.percentile(np.abs(np.concatenate([x, y])), 99)) or 1.0
    bins = np.linspace(-lim, lim, 80)
    ax.hist(x, bins=bins, color=palette.BAD, alpha=0.5, label=label_x, density=True)
    ax.hist(y, bins=bins, color=palette.ACCENT, alpha=0.5, label=label_y, density=True)
    ax.set_xlabel("Amplitude (uV)")
    ax.set_ylabel("Density")
    ax.set_title("Amplitude distribution")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    return fig


def fig_pair_cross_correlation(x: np.ndarray, y: np.ndarray, sr: int,
                               label_x: str, label_y: str,
                               max_lag_s: float = 0.5) -> Figure:
    fig = _new_fig(7.0, 4.0)
    ax = fig.add_subplot(111)
    n = min(len(x), len(y))
    a = (x[:n] - x[:n].mean())
    b = (y[:n] - y[:n].mean())
    denom = (np.std(a) * np.std(b) * n) + 1e-12
    corr = _correlate(a, b, mode="full") / denom
    lags = np.arange(-n + 1, n) / sr
    m = np.abs(lags) <= max_lag_s
    ax.plot(lags[m] * 1000.0, corr[m], color=palette.GOOD, lw=1.2)
    peak_lag = lags[m][int(np.argmax(corr[m]))] * 1000.0
    ax.axvline(peak_lag, color="gray", ls=":", lw=0.8)
    ax.set_xlabel("Lag (ms)")
    ax.set_ylabel("Normalized cross-correlation")
    ax.set_title(f"Cross-correlation: {label_x} vs {label_y} (peak at {peak_lag:.1f} ms)")
    ax.grid(True, alpha=0.2)
    return fig


def fig_pair_bandpower_ratio(x: np.ndarray, y: np.ndarray, sr: int,
                             label_x: str, label_y: str) -> Figure:
    fig = _new_fig(7.0, 4.0)
    ax = fig.add_subplot(111)
    fx, px = compute_psd(np.ascontiguousarray(x), sr)
    fy, py = compute_psd(np.ascontiguousarray(y), sr)
    names = [b[0] for b in EEG_BANDS]
    bx = _band_powers(fx, px) if fx.size else {n: 0.0 for n in names}
    by = _band_powers(fy, py) if fy.size else {n: 0.0 for n in names}
    idx = np.arange(len(names))
    w = 0.38
    ax.bar(idx - w / 2, [bx[n] for n in names], w, color=palette.BAD, label=label_x)
    ax.bar(idx + w / 2, [by[n] for n in names], w, color=palette.ACCENT, label=label_y)
    ax.set_yscale("log")
    ax.set_xticks(idx)
    ax.set_xticklabels(names)
    ax.set_ylabel(r"Band power ($\mu V^2$)")
    ax.set_title("Absolute band power by channel")
    ax.legend(fontsize=9)
    return fig


def fig_pair_stats_table(x: np.ndarray, y: np.ndarray, sr: int, ag: dict,
                         label_x: str, label_y: str) -> Figure:
    mx = channel_metrics(x, sr)
    my = channel_metrics(y, sr)
    rows = [
        ["Pearson r", f"{ag['r']:.3f}"],
        ["Spearman rho", f"{ag['spearman']:.3f}"],
        ["RMSE (uV)", f"{ag['rmse']:.2f}"],
        ["NRMSE", f"{ag['nrmse']:.3f}"],
        ["Mean coherence (1-30 Hz)", f"{ag['mean_coherence_1_30']:.3f}"],
        ["Bland-Altman bias (uV)", f"{ag['ba_bias']:.2f}"],
        ["Bland-Altman LoA (uV)", f"[{ag['ba_loa_low']:.1f}, {ag['ba_loa_high']:.1f}]"],
        [f"RMS {label_x} / {label_y} (uV)", f"{mx['rms']:.1f} / {my['rms']:.1f}"],
        [f"Line noise {label_x} / {label_y} (%)",
         f"{mx['line_ratio'] * 100:.0f} / {my['line_ratio'] * 100:.0f}"],
        [f"Alpha ratio {label_x} / {label_y} (dB)",
         f"{mx['alpha_ratio_db']:.1f} / {my['alpha_ratio_db']:.1f}"],
    ]
    for name in [b[0] for b in EEG_BANDS]:
        rows.append([f"Coherence {name}", f"{ag['band_coherence'].get(name, float('nan')):.3f}"])

    fig = _new_fig(6.5, 0.45 * (len(rows) + 1))
    ax = fig.add_subplot(111)
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=["Metric", "Value"],
                     loc="center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.3)
    ax.set_title(f"Comparison statistics: {label_x} vs {label_y}")
    return fig


def fig_cmp_bandpower(eeg: np.ndarray, sr: int, metas: List[ChannelMeta]) -> Figure:
    fig = _new_fig()
    ax = fig.add_subplot(111)
    stats = group_band_stats(eeg, sr, metas)
    bands = stats["bands"]
    x = np.arange(len(bands))
    width = 0.38
    order = [("PEDOT", palette.MATERIAL_PEDOT, -width / 2), ("Ag/AgCl", palette.MATERIAL_AGAGCL, width / 2)]
    for name, color, off in order:
        g = stats["groups"].get(name)
        if not g:
            continue
        ax.bar(x + off, g["mean"], width, yerr=g["std"], capsize=3, color=color,
               label=f"{name} (n={g['n']})")
    pvals = stats.get("pvals")
    if pvals:
        ymax = ax.get_ylim()[1]
        for i, p in enumerate(pvals):
            if np.isfinite(p):
                star = "*" if p < 0.05 else ""
                ax.text(i, ymax * 0.95, f"p={p:.2f}{star}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(bands)
    ax.set_ylabel("Relative band power")
    ax.set_title("Band power: PEDOT vs Ag/AgCl (mean +/- std)")
    ax.legend(fontsize=9)
    return fig
