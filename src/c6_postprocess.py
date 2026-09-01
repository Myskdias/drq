from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit multiple C6 roots, then aggregate report-ready outputs")
    parser.add_argument("--run-dirs", nargs="+", required=True)
    parser.add_argument("--human-dir", required=True)
    parser.add_argument("--split-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-processes", type=int, default=16)
    parser.add_argument("--timeout", type=int, default=21600)
    parser.add_argument(
        "--reuse-audits",
        action="store_true",
        help="Reuse run_dir/c6_audit.json when it already exists",
    )
    return parser


def run(command: list[str]) -> None:
    print("+", " ".join(command))
    subprocess.run(command, check=True)


def main() -> None:
    args = build_parser().parse_args()
    script_dir = Path(__file__).resolve().parent
    audit_script = script_dir / "c6_pilot.py"
    aggregate_script = script_dir / "c6_aggregate.py"

    audit_paths = []
    for run_dir_text in args.run_dirs:
        run_dir = Path(run_dir_text)
        audit_path = run_dir / "c6_audit.json"
        if not (args.reuse_audits and audit_path.exists()):
            run(
                [
                    sys.executable,
                    str(audit_script),
                    "audit-root",
                    "--run-dir",
                    str(run_dir),
                    "--human-dir",
                    args.human_dir,
                    "--split-json",
                    args.split_json,
                    "--n-processes",
                    str(args.n_processes),
                    "--timeout",
                    str(args.timeout),
                ]
            )
        audit_paths.append(str(audit_path))

    run(
        [
            sys.executable,
            str(aggregate_script),
            "--audits",
            *audit_paths,
            "--output-dir",
            args.output_dir,
        ]
    )


if __name__ == "__main__":
    main()
