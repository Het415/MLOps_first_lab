import json
import sys
from pathlib import Path

COLS = ["accuracy", "precision", "recall", "f1"]


def load_results(root):
    return sorted(
        (json.loads(p.read_text()) for p in Path(root).rglob("metrics.json")),
        key=lambda r: r["f1"],
        reverse=True,
    )


def to_markdown(results):
    if not results:
        return "No results found.\n"
    lines = [
        "## Model comparison",
        "",
        "| Rank | Model | " + " | ".join(c.capitalize() for c in COLS) + " |",
        "|---|---|" + "---|" * len(COLS),
    ]
    for i, r in enumerate(results, 1):
        lines.append(f"| {i} | {r['model']} | " + " | ".join(f"{r[c]:.4f}" for c in COLS) + " |")
    lines += ["", f"Best model by F1: **{results[0]['model']}**", ""]
    return "\n".join(lines)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    root = argv[0] if argv else "artifacts"
    text = to_markdown(load_results(root))
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
