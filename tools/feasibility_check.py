from __future__ import annotations

import importlib.util
import json
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SOURCE_FILES = [
    "dialogue_simulator.py",
    "integrated_dqn_train.py",
    "R_data.py",
    "script_generator.py",
    "virtual_child_rl_system.py",
]
REQUIRED_LOCAL_MODULES = [
    "dueling_dqn.py",
    "tabular_q_learning.py",
]
REQUIRED_PACKAGES = [
    "numpy",
    "torch",
    "sentence_transformers",
    "pandas",
]
OPTIONAL_PACKAGES = [
    "openai",
    "docx",
]
SECRET_PATTERNS = [
    re.compile(r"sk-proj-[A-Za-z0-9_-]+"),
]


def compile_sources() -> dict:
    results = {}
    for filename in SOURCE_FILES:
        path = ROOT / filename
        try:
            source = path.read_text(encoding="utf-8")
            compile(source, str(path), "exec")
            results[filename] = {"status": "pass"}
        except Exception as exc:
            results[filename] = {"status": "fail", "error": str(exc)}
    return results


def package_status(names: list[str]) -> dict:
    return {
        name: {
            "installed": importlib.util.find_spec(name) is not None,
        }
        for name in names
    }


def local_module_status() -> dict:
    return {
        name: {
            "present": (ROOT / name).exists(),
        }
        for name in REQUIRED_LOCAL_MODULES
    }


def find_reference_status() -> dict:
    reference_dir = ROOT / "專案說明"
    return {
        "path": str(reference_dir),
        "exists": reference_dir.exists(),
        "file_count": sum(1 for _ in reference_dir.iterdir()) if reference_dir.exists() else 0,
    }


def find_sample_files() -> dict:
    session_dir = ROOT / "grandma_session_20250713_185829"
    script_json_candidates = sorted(session_dir.rglob("*.json"), key=lambda p: p.stat().st_size, reverse=True)
    runtime_demo_dir = ROOT / "artifacts" / "runtime_demo"
    runtime_demo_files = sorted(runtime_demo_dir.glob("*")) if runtime_demo_dir.exists() else []
    return {
        "session_dir_exists": session_dir.exists(),
        "script_json_candidates": [str(path.relative_to(ROOT)) for path in script_json_candidates],
        "dialogue_json_exists": (ROOT / "pure_dialogue_20250721_142929.json").exists(),
        "rl_json_exists": (ROOT / "rl_data_20250721_142929.json").exists(),
        "runtime_demo_files": [str(path.relative_to(ROOT)) for path in runtime_demo_files],
    }


def inspect_sample_outputs() -> dict:
    progress = json.loads((ROOT / "grandma_session_20250713_185829" / "progress.json").read_text(encoding="utf-8"))
    dialogue = json.loads((ROOT / "pure_dialogue_20250721_142929.json").read_text(encoding="utf-8"))
    rl = json.loads((ROOT / "rl_data_20250721_142929.json").read_text(encoding="utf-8"))
    runtime_demo_dir = ROOT / "artifacts" / "runtime_demo"
    runtime_transcript = None
    runtime_summary = None
    if runtime_demo_dir.exists():
        transcript_candidates = sorted(runtime_demo_dir.glob("runtime_session_*.json"), key=lambda p: p.stat().st_mtime)
        summary_candidates = sorted(runtime_demo_dir.glob("caregiver_summary_*.md"), key=lambda p: p.stat().st_mtime)
        if transcript_candidates:
            runtime_transcript = json.loads(transcript_candidates[-1].read_text(encoding="utf-8"))
        if summary_candidates:
            runtime_summary = str(summary_candidates[-1].relative_to(ROOT))

    steps = [script["total_steps"] for script in progress["scripts"]]
    deviation_counter = Counter(item["reward_requirements"]["deviation_level"] for item in rl["state_action_data"])
    transition_turns = sum(1 for item in rl["state_action_data"] if item["script_info"]["is_transition_script"])

    report = {
        "script_total": progress["total_scripts"],
        "source_count": len({item["source"] for item in progress["scripts"]}),
        "target_slot_count": len({item["target_slot"] for item in progress["scripts"]}),
        "avg_steps_per_script": round(sum(steps) / len(steps), 2),
        "min_steps_per_script": min(steps),
        "max_steps_per_script": max(steps),
        "sample_dialogue_turns": dialogue["total_turns"],
        "avg_similarity": round(dialogue["statistics"]["average_similarity"], 4),
        "avg_deviation": round(dialogue["statistics"]["average_deviation"], 4),
        "transition_scripts_used": dialogue["statistics"]["transition_scripts_used"],
        "filled_slot_counts": {key: len(value) for key, value in dialogue["final_filled_slots"].items()},
        "rl_records": len(rl["state_action_data"]),
        "deviation_counts": dict(sorted(deviation_counter.items())),
        "transition_turns": transition_turns,
        "high_deviation_threshold": rl["metadata"]["high_deviation_threshold"],
        "deviation_thresholds": rl["metadata"]["deviation_thresholds"],
    }
    if runtime_transcript:
        report["runtime_demo"] = {
            "turns": runtime_transcript["summary"]["total_turns"],
            "algorithm": runtime_transcript["summary"]["algorithm"],
            "average_similarity": runtime_transcript["summary"]["average_similarity"],
            "transitions_used": runtime_transcript["summary"]["transitions_used"],
            "next_focus_slots": runtime_transcript["summary"]["next_focus_slots"],
            "summary_file": runtime_summary,
        }
    return report


def scan_secrets() -> dict:
    findings = []
    for filename in SOURCE_FILES:
        text = (ROOT / filename).read_text(encoding="utf-8")
        for pattern in SECRET_PATTERNS:
            for match in pattern.finditer(text):
                findings.append(
                    {
                        "file": filename,
                        "pattern": pattern.pattern,
                        "match_prefix": match.group(0)[:12],
                    }
                )
    return {
        "count": len(findings),
        "findings": findings,
    }


def build_summary(report: dict) -> dict:
    required_missing = [name for name, result in report["required_packages"].items() if not result["installed"]]
    optional_missing = [name for name, result in report["optional_packages"].items() if not result["installed"]]
    local_missing = [name for name, result in report["local_modules"].items() if not result["present"]]
    compile_failures = [name for name, result in report["syntax_checks"].items() if result["status"] != "pass"]

    reference_dir_empty = report["reference_status"]["exists"] and report["reference_status"]["file_count"] == 0

    if (
        not compile_failures
        and not required_missing
        and not local_missing
        and report["secret_scan"]["count"] == 0
        and not optional_missing
        and not reference_dir_empty
    ):
        readiness = "ready"
    elif not compile_failures:
        readiness = "partial"
    else:
        readiness = "blocked"

    return {
        "readiness": readiness,
        "compile_failures": compile_failures,
        "required_missing": required_missing,
        "optional_missing": optional_missing,
        "local_missing": local_missing,
        "reference_dir_empty": reference_dir_empty,
        "secrets_found": report["secret_scan"]["count"],
    }


def main() -> None:
    report = {
        "project_root": str(ROOT),
        "syntax_checks": compile_sources(),
        "required_packages": package_status(REQUIRED_PACKAGES),
        "optional_packages": package_status(OPTIONAL_PACKAGES),
        "local_modules": local_module_status(),
        "reference_status": find_reference_status(),
        "sample_files": find_sample_files(),
        "sample_outputs": inspect_sample_outputs(),
        "secret_scan": scan_secrets(),
    }
    report["summary"] = build_summary(report)

    output_dir = ROOT / "artifacts"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "feasibility_report.json"
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"Saved report to: {output_path}")


if __name__ == "__main__":
    main()
