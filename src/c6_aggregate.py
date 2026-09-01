from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def flatten_audits(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        audit = read_json(path)
        root_label = Path(audit["run_dir"]).name or f"seed_{audit['root_seed']}"
        for round_result in audit["rounds"]:
            certification = round_result.get("certification") or {}
            pairwise = certification.get("pairwise") or {}
            ecology = certification.get("ecology") or {}
            rows.append(
                {
                    "root": root_label,
                    "run_dir": audit["run_dir"],
                    "root_seed": audit["root_seed"],
                    "round": round_result["evolutionary_round"],
                    "candidate_a_id": round_result["candidate_a"]["id"],
                    "candidate_b_id": round_result["candidate_b"]["id"],
                    "candidate_a_stored_fitness": round_result["candidate_a"]["stored_native_fitness"],
                    "candidate_b_stored_fitness": round_result["candidate_b"]["stored_native_fitness"],
                    "delta_h": round_result["native_audit"]["margin"],
                    "delta_h0": round_result["matched_audit"]["margin"],
                    "delta_h0_minus_h": round_result["margin_change_h0_minus_h"],
                    "native_stable": round_result["native_winner_stable_on_fresh_seeds_and_orders"],
                    "reversal": round_result["robust_third_party_reversal"],
                    "matching_validation_cost": round_result["matching"]["validation"]["total_cost"],
                    "valid_generation_fraction": round_result["valid_generation_fraction"],
                    "n_distinct_valid_candidates": round_result["n_distinct_valid_candidates"],
                    "n_occupied_map_elites_cells": round_result["n_occupied_map_elites_cells"],
                    "a_generality": pairwise.get("a_generality"),
                    "b_generality": pairwise.get("b_generality"),
                    "generality_regret_b_minus_a": pairwise.get("generality_regret_b_minus_a"),
                    "a_ecology_score": ecology.get("a_mean_score"),
                    "b_ecology_score": ecology.get("b_mean_score"),
                    "ecology_regret_b_minus_a": ecology.get("ecology_regret_b_minus_a"),
                }
            )
    return sorted(rows, key=lambda row: (row["root_seed"], row["root"], row["round"]))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        ("root", "Root"),
        ("round", "Round"),
        ("delta_h", "ΔH"),
        ("delta_h0", "ΔH₀"),
        ("delta_h0_minus_h", "ΔH₀−ΔH"),
        ("reversal", "Reversal"),
        ("matching_validation_cost", "Matching cost"),
        ("generality_regret_b_minus_a", "Generality B−A"),
        ("ecology_regret_b_minus_a", "Ecology B−A"),
    ]

    def fmt(value: Any) -> str:
        if value is None:
            return "—"
        if isinstance(value, bool):
            return "yes" if value else "no"
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)

    lines = [
        "| " + " | ".join(label for _, label in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row[key]) for key, _ in columns) + " |")
    lines.extend(
        [
            "",
            "ΔH and ΔH₀ are A−B score margins in the historical and matched counterfactual contexts.",
            "Rounds from the same root share evolutionary history and are not independent replicates.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_figure(fig, output_dir: Path, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(output_dir / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def labels(rows: list[dict[str, Any]]) -> list[str]:
    return [f"{row['root']} · r{row['round']}" for row in rows]


def plot_margin_shift(rows: list[dict[str, Any]], output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, max(4, 0.55 * len(rows))))
    for row in rows:
        ax.plot([0, 1], [row["delta_h"], row["delta_h0"]], marker="o", label=f"{row['root']} · r{row['round']}")
    ax.axhline(0, linewidth=1)
    ax.set_xticks([0, 1], ["Historical H", "Matched H₀"])
    ax.set_ylabel("A − B score margin")
    ax.set_title("Effect of replacing third-party opponents")
    ax.legend(fontsize=8)
    save_figure(fig, output_dir, "margin_shift")


def plot_matching_cost(rows: list[dict[str, Any]], output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(max(7, 1.3 * len(rows)), 4))
    x = list(range(len(rows)))
    ax.bar(x, [row["matching_validation_cost"] for row in rows])
    ax.set_xticks(x, labels(rows), rotation=30, ha="right")
    ax.set_ylabel("Validation matching cost")
    ax.set_title("Counterfactual matching quality")
    save_figure(fig, output_dir, "matching_cost")


def plot_certification(rows: list[dict[str, Any]], output_dir: Path) -> None:
    certified = [row for row in rows if row["a_generality"] is not None]
    if not certified:
        return

    fig, ax = plt.subplots(figsize=(7, max(4, 0.55 * len(certified))))
    for row in certified:
        ax.plot([0, 1], [row["a_generality"], row["b_generality"]], marker="o", label=f"{row['root']} · r{row['round']}")
    ax.set_xticks([0, 1], ["Selected A", "Counterfactual B"])
    ax.set_ylabel("Held-out pairwise generality")
    ax.set_title("Held-out pairwise certification after reversal")
    ax.legend(fontsize=8)
    save_figure(fig, output_dir, "heldout_generality")

    fig, ax = plt.subplots(figsize=(7, max(4, 0.55 * len(certified))))
    for row in certified:
        ax.plot([0, 1], [row["a_ecology_score"], row["b_ecology_score"]], marker="o", label=f"{row['root']} · r{row['round']}")
    ax.set_xticks([0, 1], ["Selected A", "Counterfactual B"])
    ax.set_ylabel("Held-out multiplayer score")
    ax.set_title("Held-out ecological certification after reversal")
    ax.legend(fontsize=8)
    save_figure(fig, output_dir, "heldout_ecology")


def aggregate_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    roots = sorted({row["run_dir"] for row in rows})
    roots_with_reversal = {
        row["run_dir"] for row in rows if row["reversal"]
    }
    return {
        "n_roots": len(roots),
        "n_round_audits": len(rows),
        "n_reversals": sum(bool(row["reversal"]) for row in rows),
        "n_roots_with_reversal": len(roots_with_reversal),
        "n_negative_margin_shifts": sum(row["delta_h0_minus_h"] < 0 for row in rows),
        "mean_margin_shift_descriptive": (
            sum(row["delta_h0_minus_h"] for row in rows) / len(rows) if rows else None
        ),
        "note": "Round-level rows within the same root are not independent replicates.",
        "rows": rows,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate C6 audit JSON files and generate report-ready outputs")
    parser.add_argument("--audits", nargs="+", required=True, help="One or more c6_audit.json files")
    parser.add_argument("--output-dir", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = flatten_audits([Path(path) for path in args.audits])
    if not rows:
        raise ValueError("No round results found in the supplied audits")

    summary = aggregate_summary(rows)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    write_csv(output_dir / "summary.csv", rows)
    write_markdown(output_dir / "report_table.md", rows)
    plot_margin_shift(rows, output_dir)
    plot_matching_cost(rows, output_dir)
    plot_certification(rows, output_dir)

    print(f"Saved {len(rows)} round audits from {summary['n_roots']} roots to {output_dir}")
    print(f"Reversals: {summary['n_reversals']} across {summary['n_roots_with_reversal']} roots")


if __name__ == "__main__":
    main()
