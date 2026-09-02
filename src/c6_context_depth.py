from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Any

import numpy as np

# Needed for pickles created by `python drq.py`.
from drq import Args, MapElites  # noqa: F401
from c6_followup import (
    choose_mechanism_candidates,
    get_round,
    load_panel,
    read_json,
    recover_candidates,
    row_seeds,
    value_sign,
    write_json,
)
from c6_pilot import Evaluator, balanced_orders
import util


def all_orders(n: int) -> list[tuple[int, ...]]:
    """All possible list orders for n warriors."""
    return list(itertools.permutations(range(n)))


def batch_contexts(
    evaluator: Evaluator,
    candidates: list[tuple[str, Any]],
    contexts: list[list[tuple[str, Any]]],
    seeds,
) -> dict[tuple[str, int], dict[str, Any]]:
    """Evaluate every candidate/context under every possible list order."""
    battles = []
    keys = []

    for candidate_id, candidate in candidates:
        for context_index, context in enumerate(contexts):
            items = [candidate, *[warrior for _, warrior in context]]
            for order in all_orders(len(items)):
                battles.append([items[index] for index in order])
                keys.append((candidate_id, context_index, order))

    outputs = evaluator.batch(battles, seeds)
    grouped: dict[tuple[str, int], list[float]] = {}

    for output, (candidate_id, context_index, order) in zip(outputs, keys):
        grouped.setdefault((candidate_id, context_index), []).append(
            float(output["score"][order.index(0)].mean())
        )

    return {
        key: {
            "mean": float(np.mean(values)),
            "std_across_orders": float(np.std(values)),
            "order_scores": values,
        }
        for key, values in grouped.items()
    }


def pairwise_baseline(candidate_row, opponent_names):
    """Mean pairwise score against exactly the requested opponents."""
    return float(
        np.mean(
            [
                candidate_row["pairwise_scores_by_opponent"][name]
                for name in opponent_names
            ]
        )
    )


def summarize_context(
    a_row,
    b_row,
    opponent_names,
    context_size,
    a_context,
    b_context,
):
    """Compare pairwise preference with one exhaustive multiplayer context."""
    pair_a = pairwise_baseline(a_row, opponent_names)
    pair_b = pairwise_baseline(b_row, opponent_names)
    pair_margin = pair_a - pair_b

    order_margins = [
        a - b
        for a, b in zip(
            a_context["order_scores"],
            b_context["order_scores"],
        )
    ]
    context_margin = float(np.mean(order_margins))

    # Core War multiplies raw score by the number of warriors.
    normalized_pair_margin = pair_margin / 2.0
    normalized_context_margin = context_margin / float(context_size)

    pair_sign = value_sign(normalized_pair_margin)
    context_sign = value_sign(normalized_context_margin)
    order_signs = [value_sign(value) for value in order_margins]

    return {
        "opponents": opponent_names,
        "n_warriors": context_size,
        "pairwise_score_a": pair_a,
        "pairwise_score_b": pair_b,
        "context_score_a": a_context["mean"],
        "context_score_b": b_context["mean"],
        "normalized_pairwise_margin_a_minus_b": normalized_pair_margin,
        "normalized_context_margin_a_minus_b": normalized_context_margin,
        "normalized_margin_shift": (
            normalized_context_margin - normalized_pair_margin
        ),
        "context_margin_std_across_orders": float(
            np.std(order_margins) / context_size
        ),
        "normalized_context_margin_by_order": [
            margin / context_size for margin in order_margins
        ],
        "mean_preference_flip": (
            pair_sign != 0
            and context_sign != 0
            and pair_sign != context_sign
        ),
        "fraction_orders_opposite_pairwise_preference": (
            float(np.mean([sign == -pair_sign for sign in order_signs]))
            if pair_sign
            else None
        ),
        "fraction_orders_same_as_pairwise_preference": (
            float(np.mean([sign == pair_sign for sign in order_signs]))
            if pair_sign
            else None
        ),
    }


def prepare_round(
    rerank,
    evolutionary_round,
    drq_args,
    all_rounds,
    human_dir,
    evaluator,
):
    row = get_round(rerank, evolutionary_round)
    selected = recover_candidates(all_rounds, row)
    selected_by_id = {candidate.id: candidate for candidate in selected}
    panel = load_panel(row, drq_args, all_rounds, human_dir, evaluator)
    a_row, b_row, b_role = choose_mechanism_candidates(row)

    candidates = [
        (a_row["id"], selected_by_id[a_row["id"]].warrior),
        (b_row["id"], selected_by_id[b_row["id"]].warrior),
    ]
    return row, panel, a_row, b_row, b_role, candidates


def candidate_metadata(a_row, b_row, b_role):
    return {
        "candidate_a": {
            "id": a_row["id"],
            "role": "balanced_multiplayer_winner",
            "balanced_multiplayer_rank": a_row["multi_rank"],
            "pairwise_rank": a_row["pairwise_rank"],
        },
        "candidate_b": {
            "id": b_row["id"],
            "role": b_role,
            "balanced_multiplayer_rank": b_row["multi_rank"],
            "pairwise_rank": b_row["pairwise_rank"],
        },
    }


def exhaustive_full_round(
    rerank,
    evolutionary_round,
    drq_args,
    all_rounds,
    human_dir,
    evaluator,
):
    """Re-test the full context using all n! list permutations."""
    row, panel, a_row, b_row, b_role, candidates = prepare_round(
        rerank,
        evolutionary_round,
        drq_args,
        all_rounds,
        human_dir,
        evaluator,
    )
    seeds = row_seeds(row)
    evaluated = batch_contexts(evaluator, candidates, [panel], seeds)

    n_warriors = 1 + len(panel)
    opponent_names = [name for name, _ in panel]
    exhaustive = summarize_context(
        a_row,
        b_row,
        opponent_names,
        n_warriors,
        evaluated[(a_row["id"], 0)],
        evaluated[(b_row["id"], 0)],
    )

    balanced_margin = (
        float(a_row["multi_score"]) - float(b_row["multi_score"])
    ) / n_warriors
    exhaustive_margin = exhaustive["normalized_context_margin_a_minus_b"]
    balanced_sign = value_sign(balanced_margin)
    exhaustive_sign = value_sign(exhaustive_margin)

    orders = all_orders(n_warriors)
    return {
        "evolutionary_round": evolutionary_round,
        "historical_panel": row["historical_panel"],
        "fresh_seed_start": row["fresh_seed_start"],
        "repeats": row["repeats"],
        **candidate_metadata(a_row, b_row, b_role),
        "n_orders": len(orders),
        "n_simulation_tasks": (
            2 * len(orders) * int(row["repeats"])
        ),
        "score_scale_note": (
            "Core War multiplies score by the number of warriors. "
            "Normalized margins divide pairwise scores by 2 and "
            f"{n_warriors}-player scores by {n_warriors}."
        ),
        "original_balanced_multiplayer": {
            "n_orders": len(balanced_orders(n_warriors)),
            "normalized_margin_a_minus_b": balanced_margin,
        },
        "exhaustive_full_context": exhaustive,
        "summary": {
            "pairwise_vs_exhaustive_mean_preference_flip": (
                exhaustive["mean_preference_flip"]
            ),
            "balanced_vs_exhaustive_preference_changed": (
                balanced_sign != 0
                and exhaustive_sign != 0
                and balanced_sign != exhaustive_sign
            ),
            "normalized_exhaustive_minus_balanced_margin": (
                exhaustive_margin - balanced_margin
            ),
        },
    }


def four_player_round(
    rerank,
    evolutionary_round,
    drq_args,
    all_rounds,
    human_dir,
    evaluator,
):
    """Test every 3-opponent subset with all 4! list permutations."""
    row, panel, a_row, b_row, b_role, candidates = prepare_round(
        rerank,
        evolutionary_round,
        drq_args,
        all_rounds,
        human_dir,
        evaluator,
    )
    if len(panel) < 3:
        raise ValueError("Four-player audit needs at least three opponents")

    contexts = [list(context) for context in itertools.combinations(panel, 3)]
    seeds = row_seeds(row)
    evaluated = batch_contexts(evaluator, candidates, contexts, seeds)

    results = []
    for context_index, context in enumerate(contexts):
        opponent_names = [name for name, _ in context]
        results.append(
            summarize_context(
                a_row,
                b_row,
                opponent_names,
                4,
                evaluated[(a_row["id"], context_index)],
                evaluated[(b_row["id"], context_index)],
            )
        )

    flips = [result for result in results if result["mean_preference_flip"]]
    robust = max(
        results,
        key=lambda result: (
            result["fraction_orders_opposite_pairwise_preference"]
            if result["fraction_orders_opposite_pairwise_preference"]
            is not None
            else -1.0
        ),
    )

    orders = all_orders(4)
    return {
        "evolutionary_round": evolutionary_round,
        "historical_panel": row["historical_panel"],
        "fresh_seed_start": row["fresh_seed_start"],
        "repeats": row["repeats"],
        **candidate_metadata(a_row, b_row, b_role),
        "n_contexts": len(contexts),
        "orders_per_context": len(orders),
        "n_simulation_tasks": (
            2
            * len(contexts)
            * len(orders)
            * int(row["repeats"])
        ),
        "score_scale_note": (
            "Core War multiplies score by the number of warriors. "
            "Normalized margins divide pairwise scores by 2 and "
            "four-player scores by 4."
        ),
        "contexts": results,
        "summary": {
            "n_mean_preference_flips": len(flips),
            "mean_preference_flip_opponent_sets": [
                result["opponents"] for result in flips
            ],
            "max_fraction_orders_opposite_pairwise_preference": robust[
                "fraction_orders_opposite_pairwise_preference"
            ],
            "most_order_sensitive_opponent_set": robust["opponents"],
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "C6 context-depth controls: exhaustive full-context permutations "
            "or exhaustive four-player subsets"
        )
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--human-dir", required=True)
    parser.add_argument("--rerank-json")
    parser.add_argument("--output")
    parser.add_argument(
        "--mode",
        choices=("full", "four"),
        required=True,
    )
    parser.add_argument("--round", type=int, default=4)
    parser.add_argument("--n-processes", type=int, default=24)
    parser.add_argument("--timeout", type=int, default=7200)
    args = parser.parse_args()

    if args.n_processes < 1 or args.timeout <= 0:
        raise ValueError(
            "--n-processes must be >= 1 and --timeout must be > 0"
        )

    run_dir = Path(args.run_dir)
    rerank_path = (
        Path(args.rerank_json)
        if args.rerank_json
        else run_dir / "c6_rerank.json"
    )
    rerank = read_json(rerank_path)
    drq_args = util.load_pkl(str(run_dir), "args")
    all_rounds = util.load_pkl(str(run_dir), "all_rounds_map_elites")
    evaluator = Evaluator(
        drq_args.simargs,
        args.n_processes,
        args.timeout,
    )
    human_dir = Path(args.human_dir)

    if args.mode == "full":
        audit = exhaustive_full_round(
            rerank,
            args.round,
            drq_args,
            all_rounds,
            human_dir,
            evaluator,
        )
        experiment = "exhaustive full-context permutation control"
        default_output = run_dir / "c6_full_permutations.json"
    else:
        audit = four_player_round(
            rerank,
            args.round,
            drq_args,
            all_rounds,
            human_dir,
            evaluator,
        )
        experiment = "exhaustive four-player context audit"
        default_output = run_dir / "c6_four_player.json"

    result = {
        "run_dir": str(run_dir),
        "root_seed": int(drq_args.seed),
        "rerank_json": str(rerank_path),
        "experiment": experiment,
        "audit": audit,
    }
    output = Path(args.output) if args.output else default_output
    write_json(output, result)

    print(f"Saved {args.mode} context audit to {output}")
    print(
        f"round={audit['evolutionary_round']} "
        f"tasks={audit['n_simulation_tasks']}"
    )
    if args.mode == "full":
        print(
            "pairwise_vs_exhaustive_flip="
            f"{audit['summary']['pairwise_vs_exhaustive_mean_preference_flip']} "
            "balanced_vs_exhaustive_changed="
            f"{audit['summary']['balanced_vs_exhaustive_preference_changed']}"
        )
    else:
        print(
            "four_player_mean_flips="
            f"{audit['summary']['n_mean_preference_flips']}/"
            f"{audit['n_contexts']}"
        )


if __name__ == "__main__":
    main()
