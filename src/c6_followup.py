from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np

# Needed for pickles created by `python drq.py`.
from drq import Args, MapElites  # noqa: F401
from c6_pilot import Evaluator, balanced_orders, ranked_candidates
from c6_rerank import average_ranks_desc, compare_rankings, historical_panel
import util


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def get_round(rerank: dict[str, Any], evolutionary_round: int) -> dict[str, Any]:
    for row in rerank["rounds"]:
        if row["evolutionary_round"] == evolutionary_round:
            return row
    raise ValueError(
        f"Round {evolutionary_round} is missing from c6_rerank.json"
    )


def recover_candidates(all_rounds, row):
    i_round = row["evolutionary_round"] - 1
    by_id = {
        candidate.id: candidate
        for candidate in ranked_candidates(all_rounds, i_round)
    }
    missing = [
        candidate["id"]
        for candidate in row["candidates"]
        if candidate["id"] not in by_id
    ]
    if missing:
        raise ValueError(
            "Could not recover reranked candidates: "
            + ", ".join(candidate_id[:12] for candidate_id in missing)
        )
    return [by_id[candidate["id"]] for candidate in row["candidates"]]


def comparison(ids, left, right, left_name, right_name):
    result = compare_rankings(ids, left, right)
    result["left"] = left_name
    result["right"] = right_name
    result["left_winner_ids"] = result.pop("multi_winner_ids")
    result["right_winner_ids"] = result.pop("pairwise_winner_ids")
    result["n_tied_left_only"] = result.pop("n_tied_multi_only")
    result["n_tied_right_only"] = result.pop("n_tied_pairwise_only")
    return result


def load_panel(row, drq_args, all_rounds, human_dir, evaluator):
    i_round = row["evolutionary_round"] - 1
    panel = historical_panel(
        i_round,
        drq_args,
        all_rounds,
        human_dir,
        evaluator,
    )
    names = [name for name, _ in panel]
    if names != row["historical_panel"]:
        raise ValueError(
            f"Round {i_round + 1}: historical panel does not match "
            "c6_rerank.json"
        )
    return panel


def row_seeds(row):
    start = int(row["fresh_seed_start"])
    repeats = int(row["repeats"])
    return range(start, start + repeats)


def fixed_order_round(row, drq_args, all_rounds, human_dir, evaluator):
    """Fresh DRQ-style multiplayer evaluation with the candidate first."""
    selected = recover_candidates(all_rounds, row)
    panel = load_panel(row, drq_args, all_rounds, human_dir, evaluator)
    seeds = row_seeds(row)
    opponents = [warrior for _, warrior in panel]

    outputs = evaluator.batch(
        [[candidate.warrior, *opponents] for candidate in selected],
        seeds,
    )
    fixed_scores = [float(output["score"][0].mean()) for output in outputs]
    fixed_stds = [float(output["score"][0].std()) for output in outputs]

    ids = [candidate["id"] for candidate in row["candidates"]]
    native = [
        float(candidate["stored_native_fitness"])
        for candidate in row["candidates"]
    ]
    balanced = [
        float(candidate["multi_score"])
        for candidate in row["candidates"]
    ]
    pairwise = [
        float(candidate["pairwise_score"])
        for candidate in row["candidates"]
    ]
    fixed_ranks = average_ranks_desc(fixed_scores)

    return {
        "evolutionary_round": row["evolutionary_round"],
        "historical_panel": row["historical_panel"],
        "fresh_seed_start": row["fresh_seed_start"],
        "repeats": row["repeats"],
        "n_simulation_tasks": len(selected) * int(row["repeats"]),
        "candidates": [
            {
                "id": candidate_id,
                "fresh_fixed_order_score": score,
                "fresh_fixed_order_std_across_seeds": std,
                "fresh_fixed_order_rank": rank,
            }
            for candidate_id, score, std, rank in zip(
                ids,
                fixed_scores,
                fixed_stds,
                fixed_ranks,
            )
        ],
        "comparisons": {
            "native_vs_fresh_fixed_order": comparison(
                ids,
                native,
                fixed_scores,
                "stored_native_drq",
                "fresh_fixed_order_multiplayer",
            ),
            "fresh_fixed_order_vs_balanced_multiplayer": comparison(
                ids,
                fixed_scores,
                balanced,
                "fresh_fixed_order_multiplayer",
                "fresh_order_balanced_multiplayer",
            ),
            "balanced_multiplayer_vs_pairwise": comparison(
                ids,
                balanced,
                pairwise,
                "fresh_order_balanced_multiplayer",
                "fresh_pairwise",
            ),
        },
    }


def choose_mechanism_candidates(row):
    """Choose the two candidates whose preference is most informative."""
    multi_ids = set(row["comparison"]["multi_winner_ids"])
    pair_ids = set(row["comparison"]["pairwise_winner_ids"])
    by_id = {candidate["id"]: candidate for candidate in row["candidates"]}

    candidate_a_id = sorted(multi_ids)[0]
    distinct_pairwise = sorted(pair_ids - multi_ids)

    if distinct_pairwise:
        candidate_b_id = distinct_pairwise[0]
        role = "pairwise_winner"
    else:
        alternatives = sorted(
            (
                candidate
                for candidate in row["candidates"]
                if candidate["id"] not in multi_ids
            ),
            key=lambda candidate: (
                candidate["pairwise_rank"],
                candidate["id"],
            ),
        )
        if not alternatives:
            raise ValueError("Mechanism audit needs two distinct candidates")
        candidate_b_id = alternatives[0]["id"]
        role = "pairwise_runner_up_control"

    return by_id[candidate_a_id], by_id[candidate_b_id], role


def batch_three_player(evaluator, candidates, opponent_pairs, seeds):
    """Evaluate every candidate/context/order in one multiprocessing batch."""
    orders = balanced_orders(3)
    battles = []
    keys = []

    for candidate_id, candidate in candidates:
        for pair_index, pair in enumerate(opponent_pairs):
            items = [candidate, pair[0][1], pair[1][1]]
            for order in orders:
                battles.append([items[index] for index in order])
                keys.append((candidate_id, pair_index, order))

    outputs = evaluator.batch(battles, seeds)
    grouped = {}

    for output, (candidate_id, pair_index, order) in zip(outputs, keys):
        grouped.setdefault((candidate_id, pair_index), []).append(
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


def value_sign(value, atol=1e-12):
    if abs(value) <= atol:
        return 0
    return 1 if value > 0 else -1


def mechanism_round(row, drq_args, all_rounds, human_dir, evaluator):
    selected = recover_candidates(all_rounds, row)
    selected_by_id = {candidate.id: candidate for candidate in selected}
    panel = load_panel(row, drq_args, all_rounds, human_dir, evaluator)
    opponent_pairs = list(itertools.combinations(panel, 2))

    if not opponent_pairs:
        raise ValueError(
            "Mechanism audit needs at least two historical opponents"
        )

    a_row, b_row, b_role = choose_mechanism_candidates(row)
    seeds = row_seeds(row)
    triple = batch_three_player(
        evaluator,
        [
            (a_row["id"], selected_by_id[a_row["id"]].warrior),
            (b_row["id"], selected_by_id[b_row["id"]].warrior),
        ],
        opponent_pairs,
        seeds,
    )

    contexts = []
    for pair_index, ((name_a, _), (name_b, _)) in enumerate(opponent_pairs):
        pair_a = float(
            np.mean(
                [
                    a_row["pairwise_scores_by_opponent"][name_a],
                    a_row["pairwise_scores_by_opponent"][name_b],
                ]
            )
        )
        pair_b = float(
            np.mean(
                [
                    b_row["pairwise_scores_by_opponent"][name_a],
                    b_row["pairwise_scores_by_opponent"][name_b],
                ]
            )
        )
        triple_a = triple[(a_row["id"], pair_index)]
        triple_b = triple[(b_row["id"], pair_index)]

        pair_margin = pair_a - pair_b
        order_margins = [
            a - b
            for a, b in zip(
                triple_a["order_scores"],
                triple_b["order_scores"],
            )
        ]
        triple_margin = float(np.mean(order_margins))
        pair_sign = value_sign(pair_margin)
        triple_sign = value_sign(triple_margin)
        order_signs = [value_sign(value) for value in order_margins]

        normalized_pair_margin = pair_margin / 2.0
        normalized_triple_margin = triple_margin / 3.0

        contexts.append(
            {
                "opponents": [name_a, name_b],
                "pairwise_margin_a_minus_b": pair_margin,
                "three_player_margin_a_minus_b": triple_margin,
                "normalized_pairwise_margin_a_minus_b": (
                    normalized_pair_margin
                ),
                "normalized_three_player_margin_a_minus_b": (
                    normalized_triple_margin
                ),
                "normalized_margin_shift": (
                    normalized_triple_margin - normalized_pair_margin
                ),
                "three_player_margin_std_across_orders": float(
                    np.std(order_margins)
                ),
                "three_player_margin_by_order": order_margins,
                "candidate_a_three_player": triple_a,
                "candidate_b_three_player": triple_b,
                "mean_preference_flip": (
                    pair_sign != 0
                    and triple_sign != 0
                    and pair_sign != triple_sign
                ),
                "fraction_orders_opposite_pairwise_preference": (
                    float(
                        np.mean(
                            [sign == -pair_sign for sign in order_signs]
                        )
                    )
                    if pair_sign
                    else None
                ),
                "fraction_orders_same_as_pairwise_preference": (
                    float(
                        np.mean(
                            [sign == pair_sign for sign in order_signs]
                        )
                    )
                    if pair_sign
                    else None
                ),
            }
        )

    flips = [context for context in contexts if context["mean_preference_flip"]]
    robust = max(
        contexts,
        key=lambda context: (
            context["fraction_orders_opposite_pairwise_preference"]
            if context["fraction_orders_opposite_pairwise_preference"]
            is not None
            else -1.0
        ),
    )

    n_orders = len(balanced_orders(3))
    return {
        "evolutionary_round": row["evolutionary_round"],
        "historical_panel": row["historical_panel"],
        "fresh_seed_start": row["fresh_seed_start"],
        "repeats": row["repeats"],
        "balanced_three_player_orders": [
            list(order) for order in balanced_orders(3)
        ],
        "n_simulation_tasks": (
            2
            * len(opponent_pairs)
            * n_orders
            * int(row["repeats"])
        ),
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
        "primary_endpoint": (
            "preference sign flip from pairwise to balanced "
            "three-player context"
        ),
        "score_scale_note": (
            "Core War score is multiplied by the number of warriors. "
            "Preference signs are the primary endpoint; normalized margins "
            "divide pairwise scores by 2 and three-player scores by 3."
        ),
        "contexts": contexts,
        "summary": {
            "n_opponent_pairs": len(contexts),
            "n_mean_preference_flips": len(flips),
            "mean_preference_flip_opponent_pairs": [
                context["opponents"] for context in flips
            ],
            "max_fraction_orders_opposite_pairwise_preference": robust[
                "fraction_orders_opposite_pairwise_preference"
            ],
            "most_order_robust_opponent_pair": robust["opponents"],
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "C6 follow-up: fresh fixed-order control and targeted "
            "3-player mechanism audit"
        )
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--human-dir", required=True)
    parser.add_argument("--rerank-json")
    parser.add_argument("--fixed-output")
    parser.add_argument("--mechanism-output")
    parser.add_argument("--n-processes", type=int, default=96)
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument(
        "--mechanism-round",
        type=int,
        default=4,
        help="Evolutionary round for the 3-player audit (default: 4)",
    )

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--fixed-only",
        action="store_true",
        help="Run only the fresh fixed-order control",
    )
    mode.add_argument(
        "--mechanism-only",
        action="store_true",
        help="Run only the targeted 3-player mechanism audit",
    )

    args = parser.parse_args()

    if args.n_processes < 1 or args.timeout <= 0:
        raise ValueError(
            "--n-processes must be >= 1 and --timeout must be > 0"
        )

    run_fixed = not args.mechanism_only
    run_mechanism = not args.fixed_only

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

    if run_fixed:
        fixed = [
            fixed_order_round(
                row,
                drq_args,
                all_rounds,
                human_dir,
                evaluator,
            )
            for row in rerank["rounds"]
        ]
        fixed_result = {
            "run_dir": str(run_dir),
            "root_seed": int(drq_args.seed),
            "rerank_json": str(rerank_path),
            "experiment": (
                "native DRQ vs fresh fixed-order multiplayer vs "
                "fresh order-balanced multiplayer"
            ),
            "fixed_order_control": fixed,
        }
        fixed_output = (
            Path(args.fixed_output)
            if args.fixed_output
            else run_dir / "c6_fixed_order.json"
        )
        write_json(fixed_output, fixed_result)
        print(f"Saved fixed-order control to {fixed_output}")

        for row in fixed:
            comparisons = row["comparisons"]
            print(
                f"fixed round={row['evolutionary_round']} "
                f"native/fixed_tau="
                f"{comparisons['native_vs_fresh_fixed_order']['kendall_tau_b']} "
                f"fixed/balanced_tau="
                f"{comparisons['fresh_fixed_order_vs_balanced_multiplayer']['kendall_tau_b']}"
            )

    if run_mechanism:
        mechanism = mechanism_round(
            get_round(rerank, args.mechanism_round),
            drq_args,
            all_rounds,
            human_dir,
            evaluator,
        )
        mechanism_result = {
            "run_dir": str(run_dir),
            "root_seed": int(drq_args.seed),
            "rerank_json": str(rerank_path),
            "experiment": (
                "targeted balanced three-player mechanism audit"
            ),
            "three_player_mechanism": mechanism,
        }
        mechanism_output = (
            Path(args.mechanism_output)
            if args.mechanism_output
            else run_dir / "c6_mechanism.json"
        )
        write_json(mechanism_output, mechanism_result)
        print(f"Saved mechanism audit to {mechanism_output}")

        summary = mechanism["summary"]
        print(
            f"mechanism round={mechanism['evolutionary_round']} "
            f"mean_preference_flips="
            f"{summary['n_mean_preference_flips']}/"
            f"{summary['n_opponent_pairs']} "
            f"max_order_opposition="
            f"{summary['max_fraction_orders_opposite_pairwise_preference']}"
        )


if __name__ == "__main__":
    main()
