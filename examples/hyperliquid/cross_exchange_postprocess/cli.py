"""CLI for the versioned cross-exchange postprocess pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .contracts import read_json
from .pipeline import (
    PostprocessError,
    PostprocessPipeline,
    inspect_campaign,
    validate_pipeline_output,
)
from .profiles import PROFILES
from .reporting import write_reports


def _add_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--campaign-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--symbol-profile", default="skhynix")
    parser.add_argument("--profile", choices=sorted(PROFILES), default="dataset")
    parser.add_argument("--task-id", default="0805T002")
    parser.add_argument("--golden-campaign-dir")
    parser.add_argument("--golden-r0-dir")
    parser.add_argument("--golden-r1-dir")
    parser.add_argument("--clean-output", action="store_true")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run a fresh pipeline.")
    _add_run_arguments(run_parser)

    resume_parser = subparsers.add_parser("resume", help="Resume and reuse valid stages.")
    _add_run_arguments(resume_parser)

    inspect_parser = subparsers.add_parser(
        "inspect",
        help="Inspect an accepted campaign without creating a dataset.",
    )
    inspect_parser.add_argument("--campaign-dir", required=True)
    inspect_parser.add_argument("--symbol-profile", default="skhynix")

    validate_parser = subparsers.add_parser(
        "validate",
        help="Verify pipeline artifacts and source provenance.",
    )
    validate_parser.add_argument("--output-dir", required=True)

    report_parser = subparsers.add_parser(
        "report",
        help="Regenerate the compact reports from a pipeline manifest.",
    )
    report_parser.add_argument("--output-dir", required=True)

    subparsers.add_parser("profiles", help="Print the profile registry.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "profiles":
            result = PROFILES
        elif args.command == "inspect":
            result = inspect_campaign(
                Path(args.campaign_dir),
                args.symbol_profile,
            )
        elif args.command == "validate":
            result = validate_pipeline_output(Path(args.output_dir))
        elif args.command == "report":
            output_dir = Path(args.output_dir).expanduser().resolve()
            result = read_json(output_dir / "pipeline_manifest.json")
            write_reports(output_dir, result)
        else:
            pipeline = PostprocessPipeline(
                campaign_dir=Path(args.campaign_dir),
                output_dir=Path(args.output_dir),
                symbol_profile=args.symbol_profile,
                profile=args.profile,
                task_id=args.task_id,
                resume=args.command == "resume",
                clean_output=args.clean_output,
                golden_campaign_dir=(
                    Path(args.golden_campaign_dir)
                    if args.golden_campaign_dir
                    else None
                ),
                golden_r0_dir=(
                    Path(args.golden_r0_dir) if args.golden_r0_dir else None
                ),
                golden_r1_dir=(
                    Path(args.golden_r1_dir) if args.golden_r1_dir else None
                ),
            )
            result = pipeline.run()
    except (PostprocessError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(json.dumps({"passes": False, "error": str(exc)}, indent=2))
        return 4
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("passes", True) is True else 5


if __name__ == "__main__":
    raise SystemExit(main())
