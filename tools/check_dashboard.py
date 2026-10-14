"""Static checks on web/index.html, because a blank dashboard fails silently.

There is no DOM here and no browser in the loop, so two classes of bug are invisible
to every other test in this repo:

* a JavaScript syntax error, which kills the whole inline script and leaves a page
  that renders its CSS but never updates a single pixel
* a getElementById('...') that does not match any id in the markup, which throws on
  the first poll and does the same thing

Both are caught here by reading the file rather than running it. The script body is
extracted and handed to `node --check`, and every id the script reaches for is looked
up in the HTML. Run it after editing the dashboard:

    python tools/check_dashboard.py
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

HTML = ROOT / "web" / "index.html"
REPORT = ROOT / "logs" / "dashboard_static_check.txt"

results = []


def check(name, ok, detail=""):
    results.append({"name": name, "ok": bool(ok), "detail": detail})
    print("  [%s] %s%s" % ("PASS" if ok else "FAIL", name, (" -- " + detail) if detail else ""),
          file=sys.stderr)
    return bool(ok)


def main() -> int:
    text = HTML.read_text(encoding="utf-8")
    lines = []

    check("dashboard file exists and is non-trivial", len(text) > 8000, "%d bytes" % len(text))

    scripts = re.findall(r"<script(?![^>]*\bsrc=)[^>]*>(.*?)</script>", text, re.S)
    check("exactly one inline script block", len(scripts) == 1, "%d found" % len(scripts))
    script = scripts[0] if scripts else ""

    # 1. Syntax, checked by the real parser rather than by eyeball.
    if script:
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False, encoding="utf-8") as handle:
            handle.write(script)
            temp_js = handle.name
        try:
            proc = subprocess.run(["node", "--check", temp_js], capture_output=True, text=True)
            check("inline JavaScript parses under node --check",
                  proc.returncode == 0,
                  (proc.stderr or "").strip().splitlines()[0] if proc.returncode else "clean")
        except FileNotFoundError:
            check("node is available to parse the script", False, "node not on PATH")
        finally:
            Path(temp_js).unlink(missing_ok=True)

    # 2. Every id the script reaches for must exist in the markup.
    declared = set(re.findall(r'\bid="([^"]+)"', text))
    used = set(re.findall(r"getElementById\(['\"]([^'\"]+)['\"]\)", script))
    missing = sorted(used - declared)
    check("every getElementById target exists in the markup",
          not missing,
          "missing: %s" % missing if missing else "%d ids all resolve" % len(used))
    unused = sorted(declared - used)
    lines.append("ids declared but never fetched by script (fine, they may be CSS targets): %s" % unused)

    # 3. The ids the dashboard updates on every poll must not be absent.
    critical = ["screen", "stats", "scene", "chips", "decision", "trace", "hdr", "banner"]
    absent = [name for name in critical if name not in declared]
    check("all elements the poll loop updates are present", not absent, "absent: %s" % absent)

    # 4. Selectors used for buttons must match the lamp markup.
    lamps = set(re.findall(r'data-btn="([^"]+)"', text))
    lamp_looked_up = set(re.findall(r"dataset\.btn", script))
    check("button lamps carry data-btn attributes the script reads",
          bool(lamps) and bool(lamp_looked_up), "lamps=%s" % sorted(lamps))

    # 5. No external resources: the dashboard has to work offline, and a CDN that is
    #    unreachable would leave a half-styled page.
    external = re.findall(r'(?:src|href)="(https?://[^"]+)"', text)
    check("no external resources (works offline, no CDN)", not external, str(external))

    # 6. The endpoints the page polls must be the ones the Flask app serves.
    polled = sorted(set(re.findall(r"fetch\('(/api/[a-z.]+)", script)))
    from src.server import app

    routes = {rule.rule for rule in app.url_map.iter_rules()}
    unknown = [p for p in polled if p not in routes]
    check("every endpoint the page fetches is a real Flask route",
          not unknown, "unknown: %s" % unknown if unknown else ", ".join(polled))

    # 7. The frame size the canvas is initialised with must match the emulator's.
    from src.config import SCREEN_H, SCREEN_W

    canvas = re.search(r'<canvas id="screen" width="(\d+)" height="(\d+)"', text)
    check("canvas is sized to the NES framebuffer",
          canvas is not None and (int(canvas.group(1)), int(canvas.group(2))) == (SCREEN_W, SCREEN_H),
          "canvas=%s expected=%s" % (canvas.groups() if canvas else None, (SCREEN_W, SCREEN_H)))
    check("the raw frame buffer size the page waits for matches",
          "W * H * 3" in script, "page compares against W*H*3")

    # 8. The 24-bit -> 32-bit expansion. This is not hypothetical: memcpy-ing the
    #    184320-byte RGB stream into a 245760-byte RGBA ImageData shifts every pixel
    #    by one byte and shears the screen into diagonal colour fringes, while the
    #    emulator underneath is perfectly fine. Nothing else in this repo can see it.
    naive = re.search(r"image\.data\.set\(\s*new Uint8Array", script)
    check("the page never memcpy-s RGB into RGBA ImageData",
          naive is None,
          "found image.data.set(new Uint8Array(...)) -- that shears the image" if naive
          else "RGB is expanded to RGBA instead")

    check("the expansion writes 4 bytes per pixel with opaque alpha",
          ("rgba32" in script and "0xff000000" in script) or re.search(r"q\s*\+=\s*4", script) is not None,
          "expected either a Uint32 fast path or an explicit 4-byte stride")

    rgba_len = SCREEN_W * SCREEN_H * 4
    rgb_len = SCREEN_W * SCREEN_H * 3
    check("RGB and RGBA buffer sizes genuinely differ (so the bug above is possible)",
          rgb_len != rgba_len,
          "rgb=%d rgba=%d" % (rgb_len, rgba_len))

    passed = sum(1 for r in results if r["ok"])
    lines.insert(0, "%d/%d passed" % (passed, len(results)))
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print("\n%d/%d passed -> %s" % (passed, len(results), REPORT), file=sys.stderr)
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
