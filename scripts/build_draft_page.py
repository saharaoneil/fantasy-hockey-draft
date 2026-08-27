"""Bake the draft board into a single self-contained HTML page.

    python3 scripts/build_draft_page.py --board out/draft_board.json --out docs/

The data is embedded rather than fetched. Fetching a sibling JSON is blocked by
CORS when the page is opened from `file://`, and a draft tool that only works
when it is online is a draft tool that fails in a basement on bad wifi. One
file, no server, no build step for the user.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PLACEHOLDER = "/*__DRAFT_BOARD_DATA__*/"


def build(board_path: Path, template_path: Path) -> str:
    board = json.loads(board_path.read_text())
    for key in ("players", "league", "target_season"):
        if key not in board:
            raise SystemExit(f"{board_path} has no {key!r}; rebuild the board first")

    template = template_path.read_text()
    if PLACEHOLDER not in template:
        raise SystemExit(f"{template_path} has no {PLACEHOLDER} to fill")

    # `</script>` inside the JSON would close the tag early and break the page.
    payload = json.dumps(board, separators=(",", ":")).replace("</", "<\\/")
    return template.replace(PLACEHOLDER, payload)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board", type=Path, default=Path("out/draft_board.json"))
    parser.add_argument("--template", type=Path, default=Path("web/template.html"))
    parser.add_argument("--out", type=Path, default=Path("docs"))
    args = parser.parse_args()

    html = build(args.board, args.template)
    args.out.mkdir(parents=True, exist_ok=True)
    target = args.out / "index.html"
    target.write_text(html)

    size_kb = target.stat().st_size / 1024
    players = len(json.loads(args.board.read_text())["players"])
    print(f"wrote {target}  ({size_kb:.0f} KB, {players} players, no dependencies)")
    print("open it directly, or serve docs/ from GitHub Pages")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
