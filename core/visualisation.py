"""
core/visualisation.py
===============================================================================
All matplotlib / HTML rendering helpers for the Streamlit app.

Keeping visual logic here keeps app.py clean.
===============================================================================
"""

from __future__ import annotations

import io
from typing import Optional

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from core.matcher import MatchResult

# ── Label colour palette ──────────────────────────────────────────────────────

LABEL_COLORS = {"Strong": "#2ecc71", "Good": "#f39c12", "Weak": "#e74c3c"}
LABEL_EMOJI  = {"Strong": "🟢",      "Good": "🟡",      "Weak": "🔴"}


# ── Skill pie chart ───────────────────────────────────────────────────────────

def skill_pie(
    matched: list[str],
    missing: list[str],
    candidate_name: str,
) -> Optional[io.BytesIO]:
    """Return a PNG BytesIO of the matched/missing skill pie, or None."""
    total = len(matched) + len(missing)
    if total == 0:
        return None

    sizes  = [1] * len(matched) + [1] * len(missing)
    colors = ["#2ecc71"] * len(matched) + ["#e74c3c"] * len(missing)

    fig, ax = plt.subplots(figsize=(4, 4), facecolor="none")
    _, _, autotexts = ax.pie(
        sizes,
        labels=None,
        colors=colors,
        autopct=lambda p: f"{p:.0f}%" if p > 6 else "",
        startangle=140,
        wedgeprops=dict(linewidth=0.5, edgecolor="white"),
        pctdistance=0.75,
    )
    for at in autotexts:
        at.set_fontsize(7)
        at.set_color("white")
        at.set_fontweight("bold")

    handles = (
        [mpatches.Patch(color="#2ecc71", label=s) for s in matched]
        + [mpatches.Patch(color="#e74c3c", label=s) for s in missing]
    )
    ax.legend(handles=handles, loc="center left",
              bbox_to_anchor=(1.0, 0.5), fontsize=7, frameon=False)

    pct = round(len(matched) / total * 100, 1)
    ax.set_title(
        f"{candidate_name}\n{pct}% skill match",
        fontsize=9, fontweight="bold", pad=8,
    )

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=130, bbox_inches="tight", transparent=True)
    plt.close(fig)
    buf.seek(0)
    return buf


# ── Feature importance bar chart ─────────────────────────────────────────────

def feature_bar(contributions: dict[str, float]) -> Optional[io.BytesIO]:
    """Horizontal bar chart of the top feature contributions."""
    if not contributions:
        return None

    # Show only the most interpretable features
    display_keys = [
        "skill_coverage_ratio", "skill_jaccard", "cosine_sim",
        "keyword_overlap", "skill_overlap_count", "resume_yoe",
        "length_ratio",
    ]
    items = {
        k.replace("_", " ").title(): v
        for k, v in contributions.items()
        if k in display_keys
    }
    if not items:
        items = dict(list(contributions.items())[:8])
        items = {k.replace("_", " ").title(): v for k, v in items.items()}

    labels = list(items.keys())
    values = list(items.values())
    colors = ["#3498db" if v >= 0 else "#e74c3c" for v in values]

    fig, ax = plt.subplots(figsize=(5, 3), facecolor="none")
    bars = ax.barh(labels, values, color=colors, edgecolor="none", height=0.55)
    ax.set_xlim(0, max(values) * 1.25 if values else 1)
    ax.set_xlabel("Feature Value", fontsize=8)
    ax.tick_params(labelsize=8)
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=7, color="#444")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=130, bbox_inches="tight", transparent=True)
    plt.close(fig)
    buf.seek(0)
    return buf


# ── HTML components ───────────────────────────────────────────────────────────

def match_badge(label: str) -> str:
    """Inline HTML badge for a match label."""
    color = LABEL_COLORS.get(label, "#888")
    emoji = LABEL_EMOJI.get(label, "")
    return (
        f"<span style='background:{color};color:#fff;padding:3px 12px;"
        f"border-radius:12px;font-size:13px;font-weight:700;"
        f"display:inline-block;'>{emoji} {label}</span>"
    )


def score_bar_html(score_pct: float, label: str) -> str:
    """Animated CSS progress bar for a match score."""
    color = LABEL_COLORS.get(label, "#888")
    return (
        f"<div style='background:#e8e8e8;border-radius:8px;height:26px;"
        f"width:100%;margin:8px 0;'>"
        f"<div style='background:{color};width:{score_pct}%;height:100%;"
        f"border-radius:8px;display:flex;align-items:center;"
        f"padding-left:10px;color:#fff;font-weight:700;font-size:13px;"
        f"transition:width 0.6s ease;'>"
        f"{score_pct}%</div></div>"
    )


def skill_badges_html(
    skills: list[str],
    bg_color: str,
    text_color: str,
    prefix: str = "",
) -> str:
    """Inline HTML skill badge row."""
    return " ".join(
        f"<span style='background:{bg_color};color:{text_color};"
        f"padding:3px 9px;border-radius:12px;font-size:12px;"
        f"display:inline-block;margin:2px;'>{prefix} {s}</span>"
        for s in sorted(skills)
    )


def result_card_html(name: str, match: MatchResult) -> str:
    """Outer card container HTML for a candidate result."""
    color = LABEL_COLORS.get(match.label, "#888")
    return (
        f"<div style='border:1px solid #ddd;border-radius:12px;"
        f"padding:16px;margin-bottom:8px;'>"
        f"<h4 style='margin:0 0 6px 0;'>📄 {name}</h4>"
        f"{match_badge(match.label)}&nbsp;"
        f"<small style='color:#666;'>conf {match.confidence*100:.0f}%</small>"
        f"{score_bar_html(match.score_pct, match.label)}"
        f"</div>"
    )
