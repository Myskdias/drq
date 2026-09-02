from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

# Needed for pickles created by `python drq.py`.
from drq import Args, MapElites  # noqa: F401
from corewar_util import parse_warrior_from_file
from c6_pilot import Evaluator, balanced_orders, ranked_candidates
import util


RERANK_ROUNDS = (1, 2, 3)  # zero-based: evolutionary rounds 2, 3, 4


def historical_panel(i_round, drq_args, all_rounds, human_dir, evaluator):
    if len(drq_args.initial_opps) != 1:
        raise ValueError("Expected exactly one initial opponent")

    initial_name = Path(drq_args.initial_opps[0]).name
    initial = parse_warrior_from_file(
        evaluator.simargs, str(human_dir / initial_name)
    )[1]

    previous = [all_rounds[r].get_best() for r in range(i_round)]
    if any(champion is None for champion in previous):
        raise ValueError(f"Missing champion before round {i_round + 1}")

    return [(f"initial:{initial_name}", initial)] + [
        (f"champion_round_{r + 1}:{champion.id[:12]}", champion.warrior)
        for r, champion in enumerate(previous)
    ]


def multiplayer_scores(evaluator, candidates, opponents, seeds):
    """DRQ score with all historical opponents present simultaneously."""
    orders = balanced_orders(1 + len(opponents))
    battles = []

    for candidate in candidates:
        items = [candidate, *opponents]
        battles.extend([[items[i] for i in order] for order in orders])

    outputs = evaluator.batch(battles, seeds)

    rows = []
    cursor = 0
    for _ in candidates:
        values = []
        for order in orders:
            output = outputs[cursor]
            cursor += 1
            values.append(float(output["score"][order.index(0)].mean()))

        rows.append(
            {
                "score": float(np.mean(values)),
                "std_across_orders": float(np.std(values)),
            }
        )

    return rows


def pairwise_scores(evaluator, candidates, opponents, seeds):
    """Mean 1v1 score against exactly the same historical opponents."""
    battles = []

    for candidate in candidates:
        for opponent in opponents:
            battles.extend(
                (
                    [candidate, opponent],
                    [opponent, candidate],
                )
            )

    outputs = evaluator.batch(battles, seeds)

    rows = []
    cursor = 0

    for _ in candidates:
        values = []

        for _ in opponents:
            forward = outputs[cursor]
            reverse = outputs[cursor + 1]
            cursor += 2

            values.append(
                float(
                    np.mean(
                        [
                            forward["score"][0].mean(),
                            reverse["score"][1].mean(),
                        ]
                    )
                )
            )

        rows.append(
            {
                "score": float(np.mean(values)),
                "scores_by_opponent": values,
            }
        )

    return rows


def average_ranks_desc(scores):
    """Ranks starting at 1, with average ranks for exact ties."""
    order = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True,
    )

    ranks = [0.0] * len(scores)
    start = 0

    while start < len(order):
        end = start + 1

        while (
            end < len(order)
            and scores[order[end]] == scores[order[start]]
        ):
            end += 1

        average_rank = ((start + 1) + end) / 2.0

        for position in range(start, end):
            ranks[order[position]] = average_rank

        start = end

    return ranks


def compare_rankings(ids, multi_scores, pair_scores):
    concordant = 0
    discordant = 0
    tied_multi = 0
    tied_pair = 0
    tied_both = 0

    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            dm = multi_scores[i] - multi_scores[j]
            dp = pair_scores[i] - pair_scores[j]

            if dm == 0 and dp == 0:
                tied_both += 1
            elif dm == 0:
                tied_multi += 1
            elif dp == 0:
                tied_pair += 1
            elif dm * dp > 0:
                concordant += 1
            else:
                discordant += 1

    denominator = math.sqrt(
        (concordant + discordant + tied_multi)
        * (concordant + discordant + tied_pair)
    )

    tau_b = (
        (concordant - discordant) / denominator
        if denominator
        else None
    )

    n_pairs = len(ids) * (len(ids) - 1) // 2
    comparable_pairs = concordant + discordant

    multi_max = max(multi_scores)
    pair_max = max(pair_scores)

    multi_winners = [
        candidate_id
        for candidate_id, score in zip(ids, multi_scores)
        if score == multi_max
    ]

    pair_winners = [
        candidate_id
        for candidate_id, score in zip(ids, pair_scores)
        if score == pair_max
    ]

    return {
        "kendall_tau_b": tau_b,
        "n_pairs": n_pairs,
        "n_discordant_pairs": discordant,
        "discordant_pair_fraction_all": discordant / n_pairs,
        "discordant_pair_fraction_comparable": (
            discordant / comparable_pairs
            if comparable_pairs
            else None
        ),
        "n_tied_multi_only": tied_multi,
        "n_tied_pairwise_only": tied_pair,
        "n_tied_both": tied_both,
        "multi_winner_ids": multi_winners,
        "pairwise_winner_ids": pair_winners,
        "winner_set_changed": (
            set(multi_winners) != set(pair_winners)
        ),
        "winner_reversal": set(multi_winners).isdisjoint(
            pair_winners
        ),
    }


def audit_round(
    i_round,
    drq_args,
    all_rounds,
    human_dir,
    evaluator,
    repeats,
    min_candidates,
    max_candidates,
):
    available = ranked_candidates(all_rounds, i_round)

    if len(available) < min_candidates:
        raise ValueError(
            f"Round {i_round + 1}: only {len(available)} "
            f"distinct valid new candidates; "
            f"need at least {min_candidates}"
        )

    # Candidate set is fixed before seeing the pairwise
    # counterfactual scores.
    selected = available[:max_candidates]

    panel = historical_panel(
        i_round,
        drq_args,
        all_rounds,
        human_dir,
        evaluator,
    )

    opponent_names = [name for name, _ in panel]
    opponents = [warrior for _, warrior in panel]

    root_seed = int(drq_args.seed)

    # Disjoint fresh simulator seeds for every root/round.
    seed_start = (
        400_000
        + 10_000 * root_seed
        + 1_000 * i_round
    )

    seeds = range(seed_start, seed_start + repeats)

    warriors = [
        candidate.warrior
        for candidate in selected
    ]

    multi = multiplayer_scores(
        evaluator,
        warriors,
        opponents,
        seeds,
    )

    pairwise = pairwise_scores(
        evaluator,
        warriors,
        opponents,
        seeds,
    )

    ids = [candidate.id for candidate in selected]

    multi_values = [
        row["score"]
        for row in multi
    ]

    pair_values = [
        row["score"]
        for row in pairwise
    ]

    multi_ranks = average_ranks_desc(multi_values)
    pair_ranks = average_ranks_desc(pair_values)

    candidates = []

    for candidate, m, p, multi_rank, pair_rank in zip(
        selected,
        multi,
        pairwise,
        multi_ranks,
        pair_ranks,
    ):
        candidates.append(
            {
                "id": candidate.id,
                "stored_native_fitness": float(
                    candidate.fitness
                ),
                "multi_score": m["score"],
                "multi_std_across_orders": (
                    m["std_across_orders"]
                ),
                "pairwise_score": p["score"],
                "pairwise_scores_by_opponent": dict(
                    zip(
                        opponent_names,
                        p["scores_by_opponent"],
                    )
                ),
                "multi_rank": multi_rank,
                "pairwise_rank": pair_rank,
                "rank_shift_pairwise_minus_multi": (
                    pair_rank - multi_rank
                ),
            }
        )

    comparison = compare_rankings(
        ids,
        multi_values,
        pair_values,
    )

    rank_shifts = [
        abs(row["rank_shift_pairwise_minus_multi"])
        for row in candidates
    ]

    comparison["mean_absolute_rank_shift"] = float(
        np.mean(rank_shifts)
    )

    comparison["max_absolute_rank_shift"] = float(
        max(rank_shifts)
    )

    return {
        "evolutionary_round": i_round + 1,
        "historical_panel": opponent_names,
        "n_total_warriors_in_multiplayer": (
            len(opponents) + 1
        ),
        "n_distinct_valid_candidates_available": (
            len(available)
        ),
        "n_candidates_evaluated": len(selected),
        "candidate_pool_truncated": (
            len(available) > len(selected)
        ),
        "selection_rule": (
            f"top {len(selected)} distinct valid new "
            "candidates by stored native DRQ fitness"
        ),
        "fresh_seed_start": seed_start,
        "repeats": repeats,
        "comparison": comparison,
        "candidates": candidates,
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "C6 Exp.1: full ranking under simultaneous "
            "multiplayer versus decomposed pairwise evaluation"
        )
    )

    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--human-dir", required=True)
    parser.add_argument("--output")
    parser.add_argument(
        "--n-processes",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=21600,
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=24,
    )
    parser.add_argument(
        "--min-candidates",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=12,
    )

    args = parser.parse_args()

    if args.min_candidates < 2:
        raise ValueError("--min-candidates must be >= 2")

    if args.max_candidates < args.min_candidates:
        raise ValueError(
            "--max-candidates must be >= --min-candidates"
        )

    if not 1 <= args.repeats <= 1000:
        raise ValueError("--repeats must be between 1 and 1000")

    if args.n_processes < 1:
        raise ValueError("--n-processes must be >= 1")

    if args.timeout <= 0:
        raise ValueError("--timeout must be > 0")

    run_dir = Path(args.run_dir)

    drq_args = util.load_pkl(
        str(run_dir),
        "args",
    )

    all_rounds = util.load_pkl(
        str(run_dir),
        "all_rounds_map_elites",
    )

    evaluator = Evaluator(
        drq_args.simargs,
        args.n_processes,
        args.timeout,
    )

    result = {
        "run_dir": str(run_dir),
        "root_seed": int(drq_args.seed),
        "experiment": (
            "same historical opponents: simultaneous "
            "multiplayer vs decomposed pairwise"
        ),
        "rounds": [
            audit_round(
                i_round,
                drq_args,
                all_rounds,
                Path(args.human_dir),
                evaluator,
                args.repeats,
                args.min_candidates,
                args.max_candidates,
            )
            for i_round in RERANK_ROUNDS
        ],
    }

    output = (
        Path(args.output)
        if args.output
        else run_dir / "c6_rerank.json"
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Saved reranking audit to {output}")

    for row in result["rounds"]:
        comparison = row["comparison"]
        tau = comparison["kendall_tau_b"]
        tau_text = "NA" if tau is None else f"{tau:.3f}"

        print(
            f"round={row['evolutionary_round']} "
            f"n={row['n_candidates_evaluated']} "
            f"tau={tau_text} "
            f"inversions="
            f"{comparison['discordant_pair_fraction_all']:.3f} "
            f"winner_reversal="
            f"{comparison['winner_reversal']}"
        )


if __name__ == "__main__":
    main()
