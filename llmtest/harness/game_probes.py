"""Game-specific probes for the FIXTURE profile (TESTPLAN 5.6, amendment 18).

The generic gate in `game_oracle` can only see gross failures: it throws, it never
paints, it never moves. That is the right ceiling for model-authored one-shot builds,
which must stay black-box (a bare one-liner prompt cannot also demand instrumentation
hooks). But it cannot see a SUBTLE bug - a collision that is off by one cell, a score
that never increments, a renderer drawing last tick's state.

Fixtures are different: we authored them, so they may expose `window.__game__`, and a
probe can assert real invariants against it. That is what makes the "subtle" planted-bug
tier scorable at all.

Snake invariants asserted here:
    advances        the head moves while the game runs
    turns           a direction key changes the head's travel direction
    grows_on_eat    eating food increments both score and body length together
    dies_at_wall    driving into the wall ends the game - and NOT one cell early or
                    one cell late (the classic off-by-one)
    render_in_sync  what is drawn matches the state that is published
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

TICK = 150         # fixture tick, ms
SETTLE = 400


@dataclass
class ProbeResult:
    checks: dict = field(default_factory=dict)
    detail: dict = field(default_factory=dict)
    errors: list = field(default_factory=list)

    @property
    def score(self) -> int:
        return sum(1 for v in self.checks.values() if v)

    @property
    def all_pass(self) -> bool:
        return bool(self.checks) and all(self.checks.values())

    def to_dict(self):
        return {"checks": dict(self.checks), "detail": self.detail,
                "errors": self.errors[:4], "score": self.score, "all_pass": self.all_pass}


def _reset(page):
    """Every invariant starts from a fresh game. Without this the snake has usually
    already driven into a wall and died, and each later check fails for the wrong
    reason - which is exactly what preflight caught.

    NOTE bubbles:true - the fixture registers its handler on window, and an event
    dispatched on document reaches it only if it bubbles. Without that flag the
    resets and key presses silently did nothing."""
    page.evaluate("() => document.dispatchEvent(new KeyboardEvent('keydown', {key: 'r', bubbles: true}))")
    page.wait_for_timeout(120)


def _state(page):
    return page.evaluate("() => window.__game__ ? JSON.parse(JSON.stringify(window.__game__)) : null")


def probe_snake(html_path: Path, *, chrome_path: str | None = None,
                seed: int = 12345) -> ProbeResult:
    from playwright.sync_api import sync_playwright

    res = ProbeResult()
    url = "file:///" + str(html_path.resolve()).replace("\\", "/")
    with sync_playwright() as pw:
        b = pw.chromium.launch(**({"executable_path": chrome_path} if chrome_path else {}))
        page = b.new_page(viewport={"width": 700, "height": 640})
        page.on("pageerror", lambda e: res.errors.append(f"pageerror: {e}"))
        page.on("console", lambda m: res.errors.append(f"console.{m.type}: {m.text}")
                if m.type == "error" else None)
        try:
            page.add_init_script(f"window.__SEED__ = {seed};")
            page.goto(url, timeout=20000)
            page.wait_for_timeout(SETTLE)

            _reset(page)
            s0 = _state(page)
            if not s0:
                res.checks = {k: False for k in
                              ("advances", "turns", "grows_on_eat", "dies_at_wall", "render_in_sync")}
                res.detail["hook"] = "window.__game__ missing - not a fixture build"
                return res

            # advances: the head moves on its own
            page.wait_for_timeout(TICK * 5)
            s1 = _state(page)
            res.checks["advances"] = (s1["head"] != s0["head"]) and s1["ticks"] > s0["ticks"]
            res.detail["ticks_seen"] = s1["ticks"] - s0["ticks"]

            # turns: a direction key changes travel direction (fresh game, act fast)
            _reset(page)
            a0 = _state(page)
            page.evaluate("() => document.dispatchEvent(new KeyboardEvent('keydown', {key: 'ArrowUp', bubbles: true}))")
            page.wait_for_timeout(TICK * 3)
            a1 = _state(page)
            res.checks["turns"] = a1["head"]["y"] < a0["head"]["y"]
            res.detail["turn"] = {"from": a0["head"], "to": a1["head"]}

            # grows_on_eat: steer onto the food; score and length must rise together
            grew = page.evaluate("""
              async () => {
                const st = () => window.__game__;
                const press = k => document.dispatchEvent(new KeyboardEvent('keydown', {key: k, bubbles: true}));
                const wait = ms => new Promise(r => setTimeout(r, ms));
                document.dispatchEvent(new KeyboardEvent('keydown', {key: 'r', bubbles: true}));
                await wait(150);
                const before = {score: st().score, len: st().length};
                for (let i = 0; i < 260 && st().alive; i++) {
                  const g = st(), dx = g.food.x - g.head.x, dy = g.food.y - g.head.y;
                  const movingH = g.dir.x !== 0;
                  // A snake may not reverse: pressing 'left' while travelling right is
                  // (correctly) ignored and it ploughs into the wall. Always change
                  // axis first, and bail out of a wall before steering for food.
                  const nx = g.head.x + g.dir.x, ny = g.head.y + g.dir.y;
                  const wallAhead = nx < 0 || ny < 0 || nx >= g.size || ny >= g.size;
                  if (wallAhead) {
                    if (movingH) press(g.head.y < g.size - 2 ? 'ArrowDown' : 'ArrowUp');
                    else press(g.head.x < g.size - 2 ? 'ArrowRight' : 'ArrowLeft');
                  } else if (movingH && dy !== 0) {
                    press(dy > 0 ? 'ArrowDown' : 'ArrowUp');
                  } else if (!movingH && dx !== 0) {
                    press(dx > 0 ? 'ArrowRight' : 'ArrowLeft');
                  }
                  await wait(40);
                  if (st().score > before.score) {
                    return {ok: true, dScore: st().score - before.score,
                            dLen: st().length - before.len};
                  }
                }
                return {ok: false, dScore: st().score - before.score,
                        dLen: st().length - before.len, alive: st().alive};
              }
            """)
            res.detail["eat"] = grew
            res.checks["grows_on_eat"] = bool(grew.get("ok")) and grew.get("dLen", 0) >= 1

            # dies_at_wall: reload, drive straight right, and check WHERE it dies.
            # Off-by-one shows up as dying while still on the board, or running past it.
            page.reload()
            page.wait_for_timeout(SETTLE)
            wall = page.evaluate("""
              async () => {
                const st = () => window.__game__;
                const press = k => document.dispatchEvent(new KeyboardEvent('keydown', {key: k, bubbles: true}));
                const wait = ms => new Promise(r => setTimeout(r, ms));
                document.dispatchEvent(new KeyboardEvent('keydown', {key: 'r', bubbles: true}));
                await wait(150);
                press('ArrowRight');
                let last = st().head.x, maxX = last;
                for (let i = 0; i < 120 && st().alive; i++) {
                  await wait(30);
                  maxX = Math.max(maxX, st().head.x);
                }
                return {alive: st().alive, maxX, size: st().size, ticks: st().ticks};
              }
            """)
            res.detail["wall"] = wall
            # A correct game dies having reached the last legal column (size-1) and
            # never rendering a head beyond it.
            res.checks["dies_at_wall"] = (not wall["alive"]) and wall["maxX"] == wall["size"] - 1

            # render_in_sync: the drawn head cell must match the published head
            page.reload()
            page.wait_for_timeout(SETTLE)
            _reset(page)
            page.wait_for_timeout(TICK * 2)
            sync = page.evaluate("""
              () => {
                const g = window.__game__, c = document.querySelector('canvas');
                if (!g || !c) return {ok: false, why: 'no canvas/state'};
                const ctx = c.getContext('2d');
                const cell = Math.round(c.width / g.size);
                const px = ctx.getImageData(g.head.x * cell + cell / 2, g.head.y * cell + cell / 2, 1, 1).data;
                const lit = (px[0] + px[1] + px[2]) > 90;      // background is near-black
                return {ok: lit, rgb: [px[0], px[1], px[2]], head: g.head, cell};
              }
            """)
            res.detail["render"] = sync
            res.checks["render_in_sync"] = bool(sync.get("ok"))
        except Exception as e:                                   # noqa: BLE001
            res.errors.append(f"probe: {type(e).__name__}: {e}")
            for k in ("advances", "turns", "grows_on_eat", "dies_at_wall", "render_in_sync"):
                res.checks.setdefault(k, False)
        finally:
            b.close()
    return res


PROBES = {"snake": probe_snake}


def probe_for(game: str):
    return PROBES.get(game)


if __name__ == "__main__":
    import sys
    r = probe_snake(Path(sys.argv[1]),
                    chrome_path=(sys.argv[2] if len(sys.argv) > 2 else None))
    print(json.dumps(r.to_dict(), indent=1))
