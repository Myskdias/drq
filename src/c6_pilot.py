from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Sequence

import numpy as np

# Pickles created by `python drq.py` may refer to __main__.Args / MapElites.
# Importing these names here makes those pickles loadable from this script.
from drq import Args, MapElites  # noqa: F401
from corewar_util import SimulationArgs, parse_warrior_from_file, run_battle_batch
import util


N_INIT = 24
N_MATCH = 160
N_CERT = 133
N_HUMANS = N_INIT + N_MATCH + N_CERT
PRIMARY_ROUNDS = (2, 3)  # zero-based: evolutionary rounds 3 and 4


def read_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2)


def parse_human(simargs: SimulationArgs, human_dir: Path, name: str):
    return parse_warrior_from_file(simargs, str(human_dir / name))[1]


def make_split(human_dir: Path, output: Path, seed: int) -> None:
    simargs = SimulationArgs()
    names = []
    invalid = []
    for path in sorted(human_dir.glob("*.red")):
        try:
            parse_warrior_from_file(simargs, str(path))
            names.append(path.name)
        except Exception:
            invalid.append(path.name)

    if invalid:
        raise ValueError(f"Invalid human warriors: {invalid}")
    if len(names) != N_HUMANS:
        raise ValueError(
            f"Expected exactly {N_HUMANS} valid .red files in {human_dir}, found {len(names)}. "
            "Refusing to silently change the preregistered 24/160/133 split."
        )

    random.Random(seed).shuffle(names)
    split = {
        "seed": seed,
        "initialization": names[:N_INIT],
        "matching": names[N_INIT : N_INIT + N_MATCH],
        "certification": names[N_INIT + N_MATCH :],
    }
    write_json(output, split)
    print(f"Saved {N_INIT}/{N_MATCH}/{N_CERT} split to {output}")


def balanced_orders(n: int) -> list[tuple[int, ...]]:
    """Cyclic rotations of forward and reversed order, without duplicates."""
    base = tuple(range(n))
    orders = []
    seen = set()
    for sequence in (base, tuple(reversed(base))):
        for shift in range(n):
            order = sequence[shift:] + sequence[:shift]
            if order not in seen:
                orders.append(order)
                seen.add(order)
    return orders


class Evaluator:
    def __init__(self, simargs: SimulationArgs, n_processes: int, timeout: int):
        self.simargs = simargs
        self.n_processes = n_processes
        self.timeout = timeout

    def batch(self, battles: Sequence[Sequence[Any]], seeds: Sequence[int]) -> list[dict[str, np.ndarray]]:
        return run_battle_batch(
            self.simargs,
            [list(battle) for battle in battles],
            list(seeds),
            n_processes=self.n_processes,
            timeout=self.timeout,
        )

    def pairwise_many(
        self, candidate: Any, opponents: Sequence[Any], seeds: Sequence[int]
    ) -> list[tuple[float, float]]:
        """Candidate/opponent scores averaged over both list orders."""
        battles = []
        for opponent in opponents:
            battles.extend(([candidate, opponent], [opponent, candidate]))
        outputs = self.batch(battles, seeds)

        results = []
        for i in range(len(opponents)):
            forward, reverse = outputs[2 * i], outputs[2 * i + 1]
            candidate_score = float(np.mean([forward["score"][0].mean(), reverse["score"][1].mean()]))
            opponent_score = float(np.mean([forward["score"][1].mean(), reverse["score"][0].mean()]))
            results.append((candidate_score, opponent_score))
        return results

    def pairwise(self, candidate: Any, opponent: Any, seeds: Sequence[int]) -> tuple[float, float]:
        return self.pairwise_many(candidate, [opponent], seeds)[0]

    def context(self, candidate: Any, opponents: Sequence[Any], seeds: Sequence[int]) -> dict[str, Any]:
        """Score one candidate while balancing list position and opponent order."""
        items = [candidate, *opponents]
        orders = balanced_orders(len(items))
        outputs = self.batch([[items[i] for i in order] for order in orders], seeds)
        scores = [
            float(output["score"][order.index(0)].mean())
            for output, order in zip(outputs, orders)
        ]
        return {
            "mean": float(np.mean(scores)),
            "std_across_orders": float(np.std(scores)),
            "order_scores": scores,
        }

    def margin(
        self,
        candidate_a: Any,
        candidate_b: Any,
        opponents: Sequence[Any],
        seeds: Sequence[int],
    ) -> dict[str, Any]:
        a = self.context(candidate_a, opponents, seeds)
        b = self.context(candidate_b, opponents, seeds)
        order_margins = [x - y for x, y in zip(a["order_scores"], b["order_scores"])]
        return {
            "a": a,
            "b": b,
            "margin": a["mean"] - b["mean"],
            "margin_std_across_orders": float(np.std(order_margins)),
            "order_margins": order_margins,
        }

def new_round_records(all_rounds: dict[int, Any], i_round: int) -> list[Any]:
    """Exclude warm-started warriors already seen in earlier round histories."""
    prior_ids = {
        warrior.id
        for r in range(i_round)
        for warrior in all_rounds[r].history
        if warrior.id is not None
    }
    return [
        warrior
        for warrior in all_rounds[i_round].history
        if warrior.id is not None and warrior.id not in prior_ids
    ]


def executable_signature(warrior: Any) -> tuple[Any, ...]:
    """Identify a parsed warrior by the Redcode that can affect a battle."""
    return (
        warrior.start,
        tuple(
            (
                instruction.opcode,
                instruction.modifier,
                instruction.a_mode,
                instruction.a_number,
                instruction.b_mode,
                instruction.b_number,
            )
            for instruction in warrior.instructions
        ),
    )


def ranked_candidates(all_rounds: dict[int, Any], i_round: int) -> list[Any]:
    by_id = {}
    for warrior in new_round_records(all_rounds, i_round):
        if warrior.warrior is None or not math.isfinite(float(warrior.fitness)):
            continue
        previous = by_id.get(warrior.id)
        if previous is None or warrior.fitness > previous.fitness:
            by_id[warrior.id] = warrior

    ranked = sorted(by_id.values(), key=lambda warrior: warrior.fitness, reverse=True)
    distinct = []
    seen_signatures = set()
    for candidate in ranked:
        signature = executable_signature(candidate.warrior)
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        distinct.append(candidate)
    return distinct


def load_human_pool(
    evaluator: Evaluator,
    human_dir: Path,
    names: Sequence[str],
) -> list[tuple[str, Any]]:
    return [(name, parse_human(evaluator.simargs, human_dir, name)) for name in names]


def choose_distinct_fillers(
    real_signatures: list[tuple[float, float]],
    filler_signatures: list[tuple[float, float]],
) -> tuple[list[int], float]:
    costs = np.array(
        [
            [abs(real_a - fill_a) + abs(real_b - fill_b) for fill_a, fill_b in filler_signatures]
            for real_a, real_b in real_signatures
        ]
    )
    if len(real_signatures) == 1:
        index = int(np.argmin(costs[0]))
        return [index], float(costs[0, index])
    if len(real_signatures) == 2:
        best = min(
            (float(costs[0, i] + costs[1, j]), i, j)
            for i in range(len(filler_signatures))
            for j in range(len(filler_signatures))
            if i != j
        )
        return [best[1], best[2]], best[0]
    raise ValueError("The preregistered pilot only replaces one or two opponents")


def match_rest_panel(
    evaluator: Evaluator,
    candidate_a: Any,
    candidate_b: Any,
    real_rest: list[tuple[str, Any]],
    matching_pool: list[tuple[str, Any]],
    seeds: Sequence[int],
) -> tuple[list[tuple[str, Any]], dict[str, Any]]:
    real_signatures = []
    real_rows = []
    for name, warrior in real_rest:
        a_score = evaluator.pairwise(candidate_a, warrior, seeds)[0]
        b_score = evaluator.pairwise(candidate_b, warrior, seeds)[0]
        real_signatures.append((a_score, b_score))
        real_rows.append({"name": name, "a_score": a_score, "b_score": b_score})

    filler_warriors = [warrior for _, warrior in matching_pool]
    a_filler_scores = evaluator.pairwise_many(candidate_a, filler_warriors, seeds)
    b_filler_scores = evaluator.pairwise_many(candidate_b, filler_warriors, seeds)
    filler_signatures = [
        (a_result[0], b_result[0])
        for a_result, b_result in zip(a_filler_scores, b_filler_scores)
    ]

    indices, cost = choose_distinct_fillers(real_signatures, filler_signatures)
    selected = [matching_pool[index] for index in indices]
    filler_rows = [
        {
            "name": matching_pool[index][0],
            "a_score": filler_signatures[index][0],
            "b_score": filler_signatures[index][1],
        }
        for index in indices
    ]
    return selected, {"total_cost": cost, "real": real_rows, "fillers": filler_rows}


def validate_matching(
    evaluator: Evaluator,
    candidate_a: Any,
    candidate_b: Any,
    real_rest: list[tuple[str, Any]],
    fillers: list[tuple[str, Any]],
    seeds: Sequence[int],
) -> dict[str, Any]:
    real_warriors = [warrior for _, warrior in real_rest]
    filler_warriors = [warrior for _, warrior in fillers]
    a_real = evaluator.pairwise_many(candidate_a, real_warriors, seeds)
    b_real = evaluator.pairwise_many(candidate_b, real_warriors, seeds)
    a_fill = evaluator.pairwise_many(candidate_a, filler_warriors, seeds)
    b_fill = evaluator.pairwise_many(candidate_b, filler_warriors, seeds)

    rows = []
    for real, filler, ar, br, af, bf in zip(real_rest, fillers, a_real, b_real, a_fill, b_fill):
        cost = abs(ar[0] - af[0]) + abs(br[0] - bf[0])
        rows.append({
            "real": real[0],
            "filler": filler[0],
            "a_real": ar[0],
            "a_filler": af[0],
            "b_real": br[0],
            "b_filler": bf[0],
            "cost": cost,
        })
    return {"total_cost": float(sum(row["cost"] for row in rows)), "rows": rows}


def certify_pairwise(
    evaluator: Evaluator,
    candidate_a: Any,
    candidate_b: Any,
    pool: list[tuple[str, Any]],
    seeds: Sequence[int],
) -> dict[str, Any]:
    opponents = [opponent for _, opponent in pool]
    a_results = evaluator.pairwise_many(candidate_a, opponents, seeds)
    b_results = evaluator.pairwise_many(candidate_b, opponents, seeds)
    rows = [
        {
            "opponent": name,
            "a_score": a_score,
            "a_opponent_score": a_opp,
            "b_score": b_score,
            "b_opponent_score": b_opp,
        }
        for (name, _), (a_score, a_opp), (b_score, b_opp)
        in zip(pool, a_results, b_results)
    ]
    a_generality = float(np.mean([row["a_score"] >= row["a_opponent_score"] for row in rows]))
    b_generality = float(np.mean([row["b_score"] >= row["b_opponent_score"] for row in rows]))
    return {
        "a_mean_score": float(np.mean([row["a_score"] for row in rows])),
        "b_mean_score": float(np.mean([row["b_score"] for row in rows])),
        "a_generality": a_generality,
        "b_generality": b_generality,
        "generality_regret_b_minus_a": b_generality - a_generality,
        "rows": rows,
    }


def certify_ecology(
    evaluator: Evaluator,
    candidate_a: Any,
    candidate_b: Any,
    pool: list[tuple[str, Any]],
    seeds: Sequence[int],
    panel_size: int,
    n_panels: int,
    offset: int,
) -> dict[str, Any]:
    required = panel_size * n_panels
    if offset + required > len(pool):
        raise ValueError("Not enough sealed certification warriors for ecological panels")

    rows = []
    selected = pool[offset : offset + required]
    for start in range(0, required, panel_size):
        panel = selected[start : start + panel_size]
        opponents = [warrior for _, warrior in panel]
        a_score = evaluator.context(candidate_a, opponents, seeds)["mean"]
        b_score = evaluator.context(candidate_b, opponents, seeds)["mean"]
        rows.append(
            {
                "opponents": [name for name, _ in panel],
                "a_score": a_score,
                "b_score": b_score,
            }
        )
    a_mean = float(np.mean([row["a_score"] for row in rows]))
    b_mean = float(np.mean([row["b_score"] for row in rows]))
    return {
        "a_mean_score": a_mean,
        "b_mean_score": b_mean,
        "ecology_regret_b_minus_a": b_mean - a_mean,
        "rows": rows,
    }


def audit_round(
    i_round: int,
    drq_args: Any,
    all_rounds: dict[int, Any],
    split: dict[str, Any],
    human_dir: Path,
    evaluator: Evaluator,
    matching_repeats: int,
    audit_repeats: int,
    certification_repeats: int,
    n_ecology_panels: int,
) -> dict[str, Any]:
    records = new_round_records(all_rounds, i_round)
    candidates = ranked_candidates(all_rounds, i_round)
    if len(candidates) < 2:
        raise ValueError(
            f"Round {i_round + 1} has fewer than two valid semantically distinct candidates"
        )
    candidate_a, candidate_b = candidates[:2]

    if len(drq_args.initial_opps) != 1:
        raise ValueError("C6 expects exactly one initial human opponent per root")
    initial_name = Path(drq_args.initial_opps[0]).name
    if initial_name not in split["initialization"]:
        raise ValueError(f"{initial_name} is not in the preregistered initialization pool")

    initial = parse_human(evaluator.simargs, human_dir, initial_name)
    previous = [all_rounds[r].get_best() for r in range(i_round)]
    if any(champion is None for champion in previous):
        raise ValueError(f"Missing champion before round {i_round + 1}")

    historical = [(f"initial:{initial_name}", initial)] + [
        (f"champion_round_{r + 1}:{champion.id[:12]}", champion.warrior)
        for r, champion in enumerate(previous)
    ]
    focal_names = {name for name, _ in historical[-2:]}
    real_rest = [(name, warrior) for name, warrior in historical if name not in focal_names]

    root_seed = int(drq_args.seed)
    matching_seeds = range(200_000 + 1_000 * root_seed, 200_000 + 1_000 * root_seed + matching_repeats)
    matching_validation_seeds = range(250_000 + 1_000 * root_seed, 250_000 + 1_000 * root_seed + audit_repeats)
    audit_seeds = range(100_000 + 1_000 * root_seed, 100_000 + 1_000 * root_seed + audit_repeats)

    matching_pool = load_human_pool(evaluator, human_dir, split["matching"])
    fillers, matching_selection = match_rest_panel(
        evaluator,
        candidate_a.warrior,
        candidate_b.warrior,
        real_rest,
        matching_pool,
        matching_seeds,
    )
    matching_validation = validate_matching(
        evaluator,
        candidate_a.warrior,
        candidate_b.warrior,
        real_rest,
        fillers,
        matching_validation_seeds,
    )

    filler_iter = iter(fillers)
    counterfactual = []
    replacement_map = {}
    for name, warrior in historical:
        if name in focal_names:
            counterfactual.append((name, warrior))
        else:
            filler_name, filler = next(filler_iter)
            counterfactual.append((f"filler:{filler_name}", filler))
            replacement_map[name] = filler_name

    native = evaluator.margin(
        candidate_a.warrior,
        candidate_b.warrior,
        [warrior for _, warrior in historical],
        audit_seeds,
    )
    matched = evaluator.margin(
        candidate_a.warrior,
        candidate_b.warrior,
        [warrior for _, warrior in counterfactual],
        audit_seeds,
    )
    native_stable = native["margin"] > 0
    reversal = native_stable and matched["margin"] < 0

    result = {
        "evolutionary_round": i_round + 1,
        "n_warriors": len(historical) + 1,
        "n_new_candidate_records": len(records),
        "valid_generation_fraction": (
            sum(record.warrior is not None for record in records) / len(records) if records else None
        ),
        "n_distinct_valid_candidates": len(candidates),
        "n_occupied_map_elites_cells": len(all_rounds[i_round].archive),
        "candidate_a": {"id": candidate_a.id, "stored_native_fitness": float(candidate_a.fitness)},
        "candidate_b": {"id": candidate_b.id, "stored_native_fitness": float(candidate_b.fitness)},
        "historical_panel": [name for name, _ in historical],
        "focal_opponents": sorted(focal_names),
        "replacement_map": replacement_map,
        "matching": {"selection": matching_selection, "validation": matching_validation},
        "native_audit": native,
        "matched_audit": matched,
        "margin_change_h0_minus_h": matched["margin"] - native["margin"],
        "native_winner_stable_on_fresh_seeds_and_orders": native_stable,
        "robust_third_party_reversal": reversal,
        "certification": None,
    }

    if reversal:
        certification_pool = load_human_pool(evaluator, human_dir, split["certification"])
        cert_base = 300_000 + 1_000 * root_seed
        pairwise = certify_pairwise(
            evaluator,
            candidate_a.warrior,
            candidate_b.warrior,
            certification_pool,
            range(cert_base, cert_base + certification_repeats),
        )

        panel_size = len(historical)  # number of opponents; candidate makes total N
        # N=4 uses the first triplets; N=5 starts after them, so pilot panels do not overlap.
        offset = 0 if panel_size == 3 else 3 * n_ecology_panels
        ecology = certify_ecology(
            evaluator,
            candidate_a.warrior,
            candidate_b.warrior,
            certification_pool,
            range(cert_base + 100_000, cert_base + 100_000 + certification_repeats),
            panel_size,
            n_ecology_panels,
            offset,
        )
        result["certification"] = {"pairwise": pairwise, "ecology": ecology}

    return result


def audit_root(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    human_dir = Path(args.human_dir)
    split = read_json(Path(args.split_json))
    drq_args = util.load_pkl(str(run_dir), "args")
    all_rounds = util.load_pkl(str(run_dir), "all_rounds_map_elites")
    evaluator = Evaluator(drq_args.simargs, args.n_processes, args.timeout)

    rounds = [
        audit_round(
            i_round,
            drq_args,
            all_rounds,
            split,
            human_dir,
            evaluator,
            args.matching_repeats,
            args.audit_repeats,
            args.certification_repeats,
            args.n_ecology_panels,
        )
        for i_round in PRIMARY_ROUNDS
    ]
    output = {
        "run_dir": str(run_dir),
        "root_seed": int(drq_args.seed),
        "split_seed": split["seed"],
        "matching_repeats": args.matching_repeats,
        "audit_repeats": args.audit_repeats,
        "certification_repeats": args.certification_repeats,
        "n_ecology_panels": args.n_ecology_panels,
        "rounds": rounds,
    }
    output_path = Path(args.output) if args.output else run_dir / "c6_audit.json"
    write_json(output_path, output)
    print(f"Saved audit to {output_path}")
    for row in rounds:
        print(
            f"round={row['evolutionary_round']} "
            f"delta_H={row['native_audit']['margin']:.4f} "
            f"delta_H0={row['matched_audit']['margin']:.4f} "
            f"reversal={row['robust_third_party_reversal']}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="C6 counterfactual selection audit for Digital Red Queen")
    subparsers = parser.add_subparsers(dest="command", required=True)

    split_parser = subparsers.add_parser("split")
    split_parser.add_argument("--human-dir", required=True)
    split_parser.add_argument("--output", required=True)
    split_parser.add_argument("--seed", type=int, default=20260831)

    audit_parser = subparsers.add_parser("audit-root")
    audit_parser.add_argument("--run-dir", required=True)
    audit_parser.add_argument("--human-dir", required=True)
    audit_parser.add_argument("--split-json", required=True)
    audit_parser.add_argument("--output")
    audit_parser.add_argument("--n-processes", type=int, default=24)
    audit_parser.add_argument("--timeout", type=int, default=900)
    audit_parser.add_argument("--matching-repeats", type=int, default=8)
    audit_parser.add_argument("--audit-repeats", type=int, default=24)
    audit_parser.add_argument("--certification-repeats", type=int, default=8)
    audit_parser.add_argument("--n-ecology-panels", type=int, default=8)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "split":
        make_split(Path(args.human_dir), Path(args.output), args.seed)
    else:
        audit_root(args)


if __name__ == "__main__":
    main()
