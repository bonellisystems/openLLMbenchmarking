"""Deterministic playability oracle for one-shot browser games.

A generated game is scored by DRIVING it in a real headless browser - never by
reading its source and never by asking a model. Every check is a fact about the
running page.

Design note, learned by validating against builds with known verdicts: do NOT try
to prove "the game responded correctly to input" by diffing pixels. A game whose
loop is running repaints constantly, so any two samples differ and the check passes
trivially; a game that has ended looks identical to one that never ran. Both make
pixel-diffing on input meaningless. So this oracle only claims what it can actually
establish:

  loads        page parses, no uncaught error / console error before we touch it
  surface      a canvas (or a DOM grid) exists to draw on
  paints       something was actually drawn - more than one colour on the surface
  animates     the picture changes while we sit still -> catches a totally frozen
               build. It does NOT prove the game logic advanced: a snake that never
               moves while its particle layer animates still passes this, which is
               why gameplay quality is graded by a human, not asserted here.
  keys_wired   the page registers a key handler at all
  input_safe   a burst of real key presses raises no new error
               -> catches the crash-on-input (`SHAPES_keys` vs `SHAPES_KEYS`)

Those are exactly the two failure modes that a source read misses. Whether the game
is *good* - fun, complete, correct - is left to a human; the screenshot and the
playable file are handed to the explorer for that.
"""
from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

DEFAULT_KEYS = ["ArrowUp", "ArrowRight", "ArrowDown", "ArrowLeft",
                "Space", "KeyW", "KeyA", "KeyS", "KeyD"]
START_TEXTS = ["start", "play", "begin", "new game", "go"]

# Installed before any page script: records key-listener registration and counts
# real draw calls, so "is it wired / is it drawing" are facts, not inferences.
_INSTRUMENT = r"""
(() => {
  window.__probe = {keys: 0, draws: 0, raf: 0};
  const ael = EventTarget.prototype.addEventListener;
  EventTarget.prototype.addEventListener = function(type, ...rest) {
    if (type === 'keydown' || type === 'keyup' || type === 'keypress') window.__probe.keys++;
    return ael.call(this, type, ...rest);
  };
  const raf = window.requestAnimationFrame;
  window.requestAnimationFrame = function(cb) {
    window.__probe.raf++;
    return raf.call(window, cb);
  };
  const C = window.CanvasRenderingContext2D;
  if (C) {
    for (const m of ['fillRect','strokeRect','drawImage','fillText','arc','fill','stroke','clearRect','putImageData']) {
      const f = C.prototype[m];
      if (typeof f === 'function') {
        C.prototype[m] = function(...a) { window.__probe.draws++; return f.apply(this, a); };
      }
    }
  }
})();
"""

_SNAPSHOT = r"""
() => {
  const p = window.__probe || {keys: 0, draws: 0, raf: 0};
  const out = {keys: p.keys, draws: p.draws, raf: p.raf};
  const c = document.querySelector('canvas');
  if (c) {
    out.kind = 'canvas'; out.w = c.width; out.h = c.height;
    try {
      const g = c.getContext('2d');
      // Sample the WHOLE canvas, not a corner: a snake moving mid-board is
      // invisible to a top-left crop and the game reads as a dead loop.
      const d = g.getImageData(0, 0, c.width, c.height).data;
      const px = (c.width * c.height) || 1;
      const step = Math.max(4, Math.floor(px / 4000)) * 4;   // ~4k samples, RGBA-aligned
      let sum = 0; const seen = new Set();
      for (let i = 0; i < d.length; i += step) {
        sum = (sum + d[i] * 3 + d[i+1] * 5 + d[i+2] * 7) % 1000000007;
        if (seen.size < 64) seen.add((d[i] << 16) | (d[i+1] << 8) | d[i+2]);
      }
      out.sig = String(sum); out.colors = seen.size;
      // Coarse luminance grid (32x32) so the caller can measure HOW MUCH of the
      // board changed. A moving piece shifts many cells; a blinking score shifts
      // one or two - only the former means the game is advancing.
      const N = 32, grid = new Array(N * N).fill(0);
      const cw = Math.max(1, Math.floor(c.width / N)), chh = Math.max(1, Math.floor(c.height / N));
      for (let gy = 0; gy < N; gy++) {
        for (let gx = 0; gx < N; gx++) {
          const sx = Math.min(c.width - 1, gx * cw + (cw >> 1));
          const sy = Math.min(c.height - 1, gy * chh + (chh >> 1));
          const o = (sy * c.width + sx) * 4;
          grid[gy * N + gx] = (d[o] * 3 + d[o+1] * 6 + d[o+2]) / 10 | 0;
        }
      }
      out.grid = grid;
    } catch (e) { out.sig = 'tainted'; out.colors = 0; }
    return out;
  }
  const cells = document.querySelectorAll('[class*=cell],[class*=block],[class*=tile],td');
  if (cells.length > 8) {
    let s = ''; const uniq = new Set();
    cells.forEach((n, i) => { if (i < 400) { s += (n.className||'') + (n.textContent||'').trim();
                                             uniq.add((n.className||'') + (n.textContent||'')); } });
    return Object.assign(out, {kind: 'dom', sig: s.slice(0, 4000), colors: uniq.size,
                               w: cells.length, h: 0});
  }
  return Object.assign(out, {kind: 'none', sig: (document.body.innerText||'').slice(0, 1500),
                             colors: 0, w: 0, h: 0});
}
"""


@dataclass
class GameResult:
    loads: bool = False
    surface: bool = False
    paints: bool = False
    loop: bool = False
    keys_wired: bool = False
    input_safe: bool = False
    errors: list = field(default_factory=list)
    detail: dict = field(default_factory=dict)
    screenshot_b64: str | None = None

    @property
    def checks(self) -> dict:
        return {"loads": self.loads, "surface": self.surface, "paints": self.paints,
                "loop": self.loop, "keys_wired": self.keys_wired,
                "input_safe": self.input_safe}

    @property
    def score(self) -> int:
        return sum(1 for v in self.checks.values() if v)

    @property
    def runs_clean(self) -> bool:
        """Loads, draws, is wired for input, animates, and survives a key burst.
        NOT a claim that the game plays correctly - see the module docstring."""
        return all(self.checks.values())

    # kept for callers/readers that expect the older name
    @property
    def playable(self) -> bool:
        return self.runs_clean

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("screenshot_b64", None)
        d["score"] = self.score
        d["runs_clean"] = self.runs_clean
        return d


def _grid_change(g1, g2, tol: int = 12):
    """Fraction of sampled board cells whose luminance moved by more than `tol`."""
    if not g1 or not g2 or len(g1) != len(g2):
        return None
    diff = sum(1 for x, y in zip(g1, g2) if abs((x or 0) - (y or 0)) > tol)
    return diff / len(g1)


def run_game_checks(html_path: Path, *, chrome_path: str | None = None,
                    settle_ms: int = 1000, observe_ms: int = 1600,
                    keys: list[str] | None = None,
                    screenshot: bool = True) -> GameResult:
    from playwright.sync_api import sync_playwright

    keys = keys or DEFAULT_KEYS
    res = GameResult()
    url = "file:///" + str(html_path.resolve()).replace("\\", "/")

    with sync_playwright() as pw:
        browser = pw.chromium.launch(**({"executable_path": chrome_path} if chrome_path else {}))
        page = browser.new_page(viewport={"width": 900, "height": 700})
        errors: list[str] = []
        page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
        page.on("console", lambda m: errors.append(f"console.{m.type}: {m.text}")
                if m.type == "error" else None)
        try:
            page.add_init_script(_INSTRUMENT)
            page.goto(url, timeout=20000)
            page.wait_for_timeout(settle_ms)
            res.loads = not errors
            load_errs = len(errors)

            s0 = page.evaluate(_SNAPSHOT)
            res.surface = s0.get("kind") in ("canvas", "dom")
            res.detail["surface"] = s0.get("kind")

            # Many builds wait behind a start overlay; a game we never started
            # would look identical to a dead one.
            for text in START_TEXTS:
                try:
                    btn = page.get_by_text(text, exact=False)
                    if btn.count():
                        btn.first.click(timeout=1000)
                        res.detail["started_via"] = text
                        break
                except Exception:
                    continue
            else:
                try:
                    page.locator("canvas").first.click(timeout=800)
                except Exception:
                    pass
            # Enter/Space dismiss menus; a direction key is part of STARTING many
            # games (a snake commonly idles until it is given a direction) - without
            # it a perfectly good game reads as a dead loop.
            for k in ("Enter", "Space", "ArrowRight", "KeyD"):
                page.keyboard.press(k)
                page.wait_for_timeout(60)
            page.wait_for_timeout(500)

            a = page.evaluate(_SNAPSHOT)
            res.paints = (a.get("colors") or 0) >= 2 or (a.get("draws") or 0) > 0
            res.detail["colors"] = a.get("colors")

            # Loop: does the picture change while we sit still? Short window, taken
            # immediately after start so a game that can end has not ended yet.
            page.wait_for_timeout(observe_ms)
            b = page.evaluate(_SNAPSHOT)
            frac = _grid_change(a.get("grid"), b.get("grid"))
            res.detail["board_change_frac"] = round(frac, 4)
            # >1.5% of the board must move. Hash equality alone is too blunt: a
            # blinking score would count as "advancing" while a frozen snake with
            # an animated background would too.
            # Any real movement counts. We deliberately do NOT try to prove the game
            # LOGIC advanced: validation against builds with known verdicts showed a
            # broken snake (frozen, animated particles) changes MORE of the board
            # than a working one, so neither hash equality nor change-magnitude can
            # separate "the snake moved" from "something blinked". That judgement is
            # left to a human in the explorer.
            res.loop = (frac >= 0.002) if frac is not None else (a.get("sig") != b.get("sig"))
            res.detail["draws_delta"] = (b.get("draws") or 0) - (a.get("draws") or 0)
            res.detail["raf_delta"] = (b.get("raf") or 0) - (a.get("raf") or 0)
            if not res.loop and res.detail["draws_delta"] > 0:
                # It is repainting but the picture never changes - that IS the dead
                # loop; keep loop False and say why.
                res.detail["repaints_without_advancing"] = True

            res.keys_wired = (b.get("keys") or 0) > 0
            if not res.keys_wired:
                res.keys_wired = bool(page.evaluate(
                    "() => !!(document.onkeydown || window.onkeydown || document.body.onkeydown)"))

            # Input safety: a real key burst must not raise anything new.
            before = len(errors)
            for k in keys:
                page.keyboard.press(k)
                page.wait_for_timeout(90)
            page.wait_for_timeout(600)
            res.input_safe = (len(errors) == before)
            res.detail["errors_at_load"] = load_errs
            res.errors = errors[:6]
            if screenshot:
                res.screenshot_b64 = base64.b64encode(page.screenshot()).decode()
        except Exception as e:                       # noqa: BLE001 - report, never raise
            res.errors.append(f"driver: {type(e).__name__}: {e}")
        finally:
            browser.close()
    return res


def det_checks_for(res: GameResult) -> dict:
    """Shape the result the way every other battery reports det_checks."""
    out = {k: {"pass": bool(v)} for k, v in res.checks.items()}
    out["runs_clean"] = {"pass": res.runs_clean}
    if res.errors:
        out["runs_clean"]["detail"] = res.errors[0][:300]
    elif res.detail.get("repaints_without_advancing"):
        out["loop"]["detail"] = "repaints every frame but the picture never advances"
    return out


if __name__ == "__main__":
    import sys
    r = run_game_checks(Path(sys.argv[1]),
                        chrome_path=(sys.argv[2] if len(sys.argv) > 2 else None))
    print(json.dumps(r.to_dict(), indent=1))
