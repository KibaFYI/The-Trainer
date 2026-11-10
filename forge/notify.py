"""Discord webhook notifications (purple embeds). Failures to notify never stop a run."""

from __future__ import annotations

import datetime as _dt
import re

from . import ui

PURPLE = 0xB388FF
PINK = 0xFF5CCB
_HOOK = re.compile(r"^https://(?:\w+\.)?(?:discord|discordapp)\.com/api/webhooks/\d+/[\w-]+")


def valid_webhook(url: str | None) -> bool:
    return bool(url and _HOOK.match(url.strip()))


class Discord:
    def __init__(self, url: str | None, run_name: str = ""):
        self.url = url.strip() if valid_webhook(url) else None
        self.run_name = run_name

    @property
    def enabled(self) -> bool:
        return self.url is not None

    def send(self, title: str, description: str = "", fields: dict | None = None, ok: bool = True) -> None:
        if not self.url:
            return
        import requests

        embed = {
            "title": title[:256],
            "description": description[:4000],
            "color": PURPLE if ok else PINK,
            "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "footer": {"text": f"axolotl-forge{' · ' + self.run_name if self.run_name else ''}"},
            "fields": [
                {"name": str(k)[:256], "value": (str(v) or "-")[:1024], "inline": len(str(v)) < 40}
                for k, v in (fields or {}).items()
                if v not in (None, "")
            ][:25],
        }
        try:
            resp = requests.post(self.url, json={"username": "axolotl-forge", "embeds": [embed]}, timeout=15)
            if resp.status_code >= 300:
                ui.warn(f"discord webhook answered {resp.status_code}")
        except Exception as exc:  # noqa: BLE001
            ui.warn(f"discord webhook failed: {exc}")

    # ── the events ───────────────────────────────────────────────────────────────
    def training_done(self, summary: dict, attempts: int) -> None:
        self.send(
            "🦎 training done",
            f"`{self.run_name}` finished training" + (f" after {attempts} attempts" if attempts > 1 else ""),
            {
                "final loss": summary.get("final_loss"),
                "eval loss": summary.get("final_eval_loss"),
                "steps": summary.get("steps"),
                "epochs": summary.get("epochs"),
                "hours": summary.get("runtime_h"),
            },
        )

    def uploaded(self, urls: list[str]) -> None:
        self.send("📦 uploaded to the Hub", "\n".join(urls) or "nothing uploaded")

    def finished(self, fields: dict) -> None:
        self.send("✅ all finished", f"`{self.run_name}` is done", fields)

    def failed(self, stage: str, error: str, attempts: int | None = None) -> None:
        head, sep, body = error.strip().partition("\n\n")
        if head.startswith("key error lines:") and sep:  # from pipeline.log_tail: the real error is up top
            tail = (head[:900] + "\n…\n" + body[-500:]).replace("```", "'''")
        else:
            tail = error.strip()[-1500:].replace("```", "'''")
        what = f"{stage} failed" + (f" {attempts} times" if attempts else "")
        self.send(f"❌ {what}", f"```\n{tail}\n```", ok=False)
