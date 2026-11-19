"""Run a benchmark suite: scenarios × controllers × seeds (paired comparison).

    python benchmark.py --suite smoke
    python benchmark.py --suite nominal --jobs 4
    python benchmark.py --suite latency --jobs 4 --report     # + tables and figures
    python benchmark.py --suite path/to/my_suite.yaml --out runs/mine

Results land in runs/<stamp>_<suite>/ (results.csv, summary.md, episodes/);
look at them with `python visualize.py runs/<stamp>_<suite>`.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any


def main(args: argparse.Namespace) -> Path:
    from diffdrive.evaluation import Suite, aggregate, format_table, run_suite

    suite = Suite.from_yaml(args.suite)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = Path(args.out) if args.out else Path("runs") / f"{stamp}_{suite.name}"
    n = len(suite.expand())
    print(f"suite {suite.name!r}: {n} runs → {out}")

    def progress(i: int, total: int, row: dict[str, Any]) -> None:
        flag = "✓" if row.get("success") else "✗"
        print(
            f"  [{i:>4}/{total}] {flag} {row['scenario']:<40.40s} {row['controller']:<16s} "
            f"seed={row['seed']:<3} {row.get('termination', '')}"
        )

    rows = run_suite(
        suite, out, jobs=args.jobs, save_episodes=not args.no_episodes, progress=progress
    )
    table = format_table(aggregate(rows))
    (out / "summary.md").write_text(f"# {suite.name}\n\n{suite.description}\n\n{table}\n")
    print()
    print(table)
    print(f"\nresults: {out / 'results.csv'}")

    if args.report:
        import visualize

        visualize.report(out)
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--suite", required=True, help="builtin suite name or YAML path")
    p.add_argument("--out", help="output directory (default: runs/<stamp>_<suite>)")
    p.add_argument("--jobs", type=int, default=1, help="parallel worker processes")
    p.add_argument("--no-episodes", action="store_true", help="only write results.csv")
    p.add_argument("--report", action="store_true", help="also draw figures (same as visualize.py <out>)")
    return p


if __name__ == "__main__":
    main(build_parser().parse_args())
