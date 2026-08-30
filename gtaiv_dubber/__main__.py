from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from gtaiv_dubber.core.config import load_config, load_glossary, relevant_config_hash
from gtaiv_dubber.core.database import Database
from gtaiv_dubber.core.environment import format_environment, inspect_environment
from gtaiv_dubber.core.reports import write_manifests
from gtaiv_dubber.core.scanner import scan_mode, write_scan_report
from gtaiv_dubber.core.verify import format_verify, verify_project
from gtaiv_dubber.utils.logging import configure_logging
from gtaiv_dubber.utils.paths import ensure_runtime_dirs, normalize_project_path


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(prog="gtaiv-dubber", description="GTA IV LATAM AI Dubber")
    cli.add_argument("--verbose", action="store_true")
    commands = cli.add_subparsers(dest="command", required=True)
    for name in ("scan", "verify", "status", "env", "gui"):
        command = commands.add_parser(name)
        command.add_argument("project_root", nargs="?", default=".")
    process = commands.add_parser("process")
    process.add_argument("project_root", nargs="?", default=".")
    selection = process.add_mutually_exclusive_group()
    selection.add_argument("--speech", action="store_true")
    selection.add_argument("--cutscenes", action="store_true")
    process.add_argument("--resume", action="store_true", default=True)
    process.add_argument("--canary-only", action="store_true")
    process.add_argument("--allow-fallback", action="store_true", help=argparse.SUPPRESS)
    return cli


def prepare(args):
    root = normalize_project_path(args.project_root)
    runtime = ensure_runtime_dirs(root)
    config = load_config(root)
    config["_config_hash"] = relevant_config_hash(config)
    config["_glossary"] = load_glossary(config)
    logger = configure_logging(runtime["logs"], args.verbose)
    return root, runtime, config, logger


def status_text(db: Database) -> str:
    grouped: dict[str, dict[str, int]] = {"speech": {}, "cutscene": {}}
    for row in db.status_counts():
        grouped[row["mode"]][row["status"]] = row["count"]
    return json.dumps(grouped, ensure_ascii=False, indent=2)


def process_summary(root: Path, db: Database, elapsed: float) -> str:
    grouped: dict[str, dict[str, int]] = {"speech": {}, "cutscene": {}}
    for row in db.status_counts():
        grouped[row["mode"]][row["status"]] = row["count"]

    def aggregate(mode: str, prefix: str) -> list[str]:
        values = grouped[mode]
        total = sum(values.values())
        review = sum(count for status, count in values.items() if status.startswith("REVIEW"))
        processed = sum(count for status, count in values.items() if status not in ("DISCOVERED", "ANALYZING"))
        return [
            f"{prefix} originales:       {total}",
            f"{prefix} procesados:       {processed}",
            f"{prefix} DONE:             {values.get('DONE', 0)}",
            f"{prefix} REVIEW:           {review}",
            f"{prefix} FALLBACK:         {values.get('FALLBACK', 0)}",
            f"{prefix} ERROR:            {values.get('ERROR', 0)}",
        ]

    hours, remainder = divmod(int(elapsed), 3600)
    minutes, seconds = divmod(remainder, 60)
    return "\n".join(
        ["========================================", "GTA IV LATAM AI DUBBER", "========================================"]
        + aggregate("speech", "Speech")
        + [""]
        + aggregate("cutscene", "Cutscenes")
        + [
            "", f"Tiempo total: {hours:02d}:{minutes:02d}:{seconds:02d}", "",
            f"Output: {root / 'Speech_es'}", f"        {root / 'Cutscenes_es'}", "",
            f"Reportes: {root / 'work' / 'reports'}", "========================================",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    root, runtime, config, logger = prepare(args)
    if args.command == "gui":
        from gtaiv_dubber.gui.main_window import run_gui

        return run_gui(root)
    if args.command == "env":
        print(format_environment(inspect_environment(config)))
        return 0
    with Database(runtime["work"] / "state.sqlite3") as db:
        if args.command == "scan":
            stats = [scan_mode(root, config, db, mode, logger) for mode in ("speech", "cutscene")]
            write_scan_report(runtime["reports"], stats)
            for item in stats:
                print(json.dumps(item.to_dict(), ensure_ascii=False, indent=2))
            return 0 if not any(item.unreadable for item in stats) else 2
        if args.command == "verify":
            report = verify_project(root, config, db)
            write_manifests(db, runtime["reports"])
            print(format_verify(report))
            return 0 if all(item["error"] == 0 for item in report.values()) else 3
        if args.command == "status":
            print(status_text(db))
            return 0
        if args.command == "process":
            from gtaiv_dubber.core.runner import process_project

            modes = ["speech"] if args.speech else ["cutscene"] if args.cutscenes else ["speech", "cutscene"]
            result = process_project(
                root, config, db, modes, runtime, logger,
                canary_only=args.canary_only, allow_fallback=args.allow_fallback,
            )
            print(process_summary(root, db, result["elapsed"]))
            return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
