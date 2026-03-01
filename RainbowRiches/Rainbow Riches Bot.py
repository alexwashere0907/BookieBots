#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Flow
1) Parse Telegram tip: (track, HH:MM, dog, back odds, value)
2) Call Betmaster "matches" feed → pick matchId by (track, time)
3) Call Betmaster race detail → pick runnerId/dog by name
4) Open race page UI → click runner → price check → stake-to-win → place bet → confirm receipt
"""

import asyncio
import os
import re
import time as monotime
import argparse
from dataclasses import replace
from datetime import datetime, date, time, timezone, timedelta
from pathlib import Path
from typing import Optional, Tuple, Dict, Set

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("betmaster")
logger.setLevel(logging.INFO)

from playwright.async_api import Page
from telegram_handler import setup_telegram
from config import CFG
from session_manager import (
    make_persistent_browser,
    ui_keepalive,
    session_heartbeat,
    is_logged_out,
    ensure_logged_in,
    clear_blocking_overlays,
    overlay_sweeper,
)

from zoneinfo import ZoneInfo  # Python 3.9+



LONDON = ZoneInfo("Europe/London")
UTC = timezone.utc

TIME_TOL_MIN = 10   # was 2; 5–6 is realistic for greyhound feeds

from datetime import datetime, date, time, timedelta, timezone
try:
    from zoneinfo import ZoneInfo  # Py3.9+
    _LONDON_TZ = ZoneInfo("Europe/London")
except Exception:
    _LONDON_TZ = None  # we'll fall back to a coarse offset below
    

ATTEMPTED_KEYS: set[str] = set()

def make_attempt_key(track: str, off_time: str, dog: str) -> str:
    return f"{track.strip().lower()}|{off_time.strip()}|{dog.strip().lower()}"

async def clear_betslip(page: Page):
    for _ in range(3):
        remove_btns = page.locator(
            "aside button[aria-label*='remove'], aside button:has(svg)"
        )
        count = await remove_btns.count()
        if count == 0:
            return
        try:
            await remove_btns.nth(0).click(force=True)
            await page.wait_for_timeout(300)
        except Exception:
            return


# -----------------------------
# Config
# -----------------------------

RR_HOST      = os.environ["RR_HOST"]
RR_SITE_ROOT = f"https://{RR_HOST}"
RR_RACE_URL  = RR_SITE_ROOT + "/sports#racing/event/{event_id}"
BASE_URL     = f"https://{RR_HOST}/sports#racing/greyhounds/nextoff"

RR_RACING_FEED = os.environ["RR_RACING_FEED"]

def calculate_stake(target_profit: float, back_odds: float) -> float:
    if back_odds <= 1.01:
        return round(target_profit, 2)
    return round(target_profit / (back_odds - 1.0), 2)


# -----------------------------
# Utilities / Normalizers
# -----------------------------
_BRAND_WORDS = {"star", "even", "seven", "bet", "sports", "sport", "book", "bookie"}
TRACK_ALIASES = {
    "pelaw": "pelaw grange",
    "star pelaw": "pelaw grange",
    # Common UKs
    "oxford": "oxford",
    "romford": "romford",
    "harlow": "harlow",
    "hove": "hove",
    "newcastle": "newcastle",
    "sunderland": "sunderland",
    "yarmouth": "yarmouth",
    "towcester": "towcester",
    "suffolk dwns": "suffolk",
}

def normalize_name(s: str) -> str:
    return (
        s.lower()
        .replace(".", "")
        .replace("'", "")
        .replace("-", " ")
        .replace("_", " ")
        .strip()
    )

def norm_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def norm_key(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())

def round_to_nearest_50p(value: float) -> float:
    return max(0.5, round(value * 2) / 2)

def normalize_track_name(raw: str) -> str:
    s = (raw or "").strip().lower()
    s = re.sub(r"[\s\-]+", " ", s)
    tokens = [t for t in s.split() if t not in _BRAND_WORDS] or s.split()
    s2 = " ".join(tokens)
    return TRACK_ALIASES.get(s2, s2)

def min_acceptable_odds(want_odds: Optional[float]) -> Optional[float]:
    if not want_odds:
        return None
    want_odds = round(want_odds, 2)
    floor_abs = want_odds - CFG.odds_tolerance_abs
    floor_pct = want_odds * (1.0 - (CFG.odds_tolerance_pct / 100.0)) if CFG.odds_tolerance_pct > 0 else floor_abs
    floor_ = max(1.01, min(floor_abs, floor_pct))
    return round(floor_ + 1e-9, 2)


def _london_now():
    """Return 'now' as a timezone-aware datetime in Europe/London."""
    if _LONDON_TZ:
        return datetime.now(tz=_LONDON_TZ)
    # Fallback: coarse BST/GMT guess by month
    now_utc = datetime.now(timezone.utc)
    is_bst = now_utc.month in (4,5,6,7,8,9,10)
    return now_utc + timedelta(hours=1 if is_bst else 0)


def london_today_window_ms() -> tuple[int, int]:
    """
    Return start/end of 'today' in Europe/London, as UTC epoch ms.
    """
    now_ldn = _london_now()
    # Start/end as London-aware, then convert to UTC and epoch-ms
    start_ldn = datetime.combine(now_ldn.date(), time.min, tzinfo=now_ldn.tzinfo)
    end_ldn   = datetime.combine(now_ldn.date(), time.max, tzinfo=now_ldn.tzinfo)

    start_utc = start_ldn.astimezone(timezone.utc)
    end_utc   = end_ldn.astimezone(timezone.utc)

    return int(start_utc.timestamp() * 1000), int(end_utc.timestamp() * 1000)

def is_price_acceptable(api_odds: float, tip_back: float) -> bool:
    if not api_odds or not tip_back:
        return False

    floor = min_acceptable_odds(tip_back)
    return api_odds >= floor

async def resolve_rr_event(page: Page, tip, time_tol_min: int = 10) -> Optional[Tuple[int, str]]:
    """
    Resolve Rainbow Riches greyhound race by track + time.
    Returns (event_id, race_url)
    """

    data = await fetch_json_request(page.context, RR_RACING_FEED)

    events = data.get("events", [])
    if not events:
        logger.warning("[rr] No events returned from racing feed")
        return None

    tip_track = normalize_track_name(tip["track"])
    tip_minutes = hhmm_to_minutes(tip["time"])
    if tip_minutes is None:
        return None

    candidates = []

    for ev in events:
        try:
            track_name = normalize_track_name(ev["name"])
            if track_name != tip_track:
                continue

            start_iso = ev["start"]
            dt = datetime.fromisoformat(start_iso.replace("Z", "+00:00"))
            dt_ldn = dt.astimezone(_LONDON_TZ) if _LONDON_TZ else dt

            race_minutes = dt_ldn.hour * 60 + dt_ldn.minute
            candidates.append((race_minutes, ev))
        except Exception:
            continue

    if not candidates:
        logger.warning(f"[rr] No races for {tip['track']}")
        return None

    chosen = _pick_best_race(tip_minutes, candidates, tol_min=time_tol_min)
    if not chosen:
        logger.warning("[rr] No race within time tolerance")
        return None

    event_id = chosen["id"]
    race_url = RR_RACE_URL.format(event_id=event_id)

    return event_id, race_url

async def enter_stake_rainbow_riches(page, stake: float):
    stake_str = f"{stake:.2f}"

    stake_input = page.locator(
        "input[id^='mod-KambiBC-betslip-stake-input-outcome']"
    ).first

    await stake_input.wait_for(state="visible", timeout=5000)

    # Must CLICK to give React focus
    await stake_input.click(force=True)
    await page.wait_for_timeout(150)

    # Clear existing value (keyboard only)
    await page.keyboard.press("Control+A")
    await page.keyboard.press("Backspace")
    await page.wait_for_timeout(100)

    # Type stake like a human
    await page.keyboard.type(stake_str, delay=120)
    await page.wait_for_timeout(300)

async def place_bet_rainbow_riches(
    page,
    stake: float,
    expected_odds: float,
):
    """
    Assumes:
    - runner already added to betslip
    - odds already validated via API
    """

    # 1) Wait for stake input (REAL readiness signal)
    stake_input = page.locator(
        "input[id^='mod-KambiBC-betslip-stake-input-outcome']"
    ).first

    await stake_input.wait_for(state="visible", timeout=8000)

    # 2) Enter stake
    await enter_stake_rainbow_riches(page, stake)

    # 3) Optional sanity check (highly recommended)
    entered = await stake_input.input_value()
    if not entered or float(entered) <= 0:
        raise RuntimeError("Stake input failed (empty after typing)")

    # 4) Handle odds-change blocker (skip, do NOT approve)
    approve_btn = page.locator("button").filter(
        has_text=re.compile(r"approve odds change", re.I)
    )
    if await approve_btn.count():
        logger.warning("⚠️ Odds change prompt detected — aborting bet")
        await clear_betslip(page)
        return

    # 4.5) Confirm betslip odds haven't moved since the API check
    try:
        odds_el = page.locator("[class*='KambiBC-betslip'][class*='odds']").first
        if await odds_el.count():
            raw_text = (await odds_el.inner_text()).replace("\xa0", " ").strip()
            # Text is like "(1)  @  7/2" — extract the fractional part after "@"
            frac_match = re.search(r"(\d+)\s*/\s*(\d+)", raw_text)
            if frac_match:
                num, den = int(frac_match.group(1)), int(frac_match.group(2))
                live_odds = round(num / den + 1.0, 4)
            else:
                # Fallback: try parsing as a plain decimal
                dec_match = re.search(r"[\d]+\.[\d]+", raw_text)
                live_odds = float(dec_match.group()) if dec_match else None

            if live_odds is None:
                logger.warning(f"⚠️ Could not parse betslip odds from '{raw_text}' — proceeding without confirmation")
            else:
                floor = min_acceptable_odds(expected_odds)
                if live_odds < floor:
                    logger.warning(
                        f"⚠️ Betslip odds {live_odds} dropped below floor {floor} "
                        f"(expected {expected_odds}) — aborting bet"
                    )
                    await clear_betslip(page)
                    return
                logger.info(f"✅ Betslip odds confirmed: {live_odds} (expected {expected_odds})")
        else:
            logger.warning("⚠️ Could not read betslip odds element — proceeding without confirmation")
    except Exception as e:
        logger.warning(f"⚠️ Betslip odds check failed ({e}) — proceeding")

    # 5) Click Place Bet
    place_btn = page.locator("button").filter(
        has_text=re.compile(r"place bet", re.I)
    ).first

    await place_btn.wait_for(state="visible", timeout=5000)
    await place_btn.click()

    # 6) (Optional) confirmation hook later

async def resolve_rr_runner_and_odds(context, event_id: int, runner_name: str) -> dict | None:
    """
    Resolve Rainbow Riches (Kambi) runner + live odds via API ONLY.
    """

    race_api_url = (
        "https://eu1.offering-api.kambicdn.com/"
        f"offering/v2018/rrichesuk/betoffer/event/{event_id}.json"
        "?lang=en_GB&market=GB&client_id=200&channel_id=1"
    )

    data = await fetch_json_request(context, race_api_url)

    bet_offers = data.get("betOffers", [])
    if not bet_offers:
        return None

    # Pick WIN market
    win_bo = next(
        (
            bo for bo in bet_offers
            if bo.get("betOfferType", {}).get("name") == "Winner"
            and bo.get("criterion", {}).get("englishLabel") == "To win"
            and "MAIN" in bo.get("tags", [])
        ),
        None
    )

    if not win_bo:
        return None

    target = normalize_name(runner_name)

    for outcome in win_bo.get("outcomes", []):
        if outcome.get("status") != "OPEN":
            continue

        if normalize_name(outcome.get("participant", "")) != target:
            continue

        odds_milli = outcome.get("odds")
        decimal_odds = odds_milli / 1000.0 if odds_milli else None

        return {
            "betOfferId": win_bo["id"],
            "outcomeId": outcome["id"],
            "participantId": outcome["participantId"],
            "startNr": outcome.get("startNr"),
            "decimal_odds": decimal_odds,
        }

    return None

def _pick_best_race(tip_minutes: int, candidates: list[tuple[int, dict]], tol_min: int = 30) -> dict | None:
    """
    candidates: list of (race_minutes, match_dict)
    Strategy:
      1) exact HH:MM match if present
      2) else nearest race AT/AFTER the tip time within tol
      3) else absolute nearest within tol
    """
    # Exact
    for race_min, m in candidates:
        if race_min == tip_minutes:
            return m
    # Nearest forward (>=)
    fwd = [(race_min, m) for race_min, m in candidates if race_min >= tip_minutes]
    if fwd:
        fwd.sort(key=lambda x: x[0] - tip_minutes)
        if (fwd[0][0] - tip_minutes) <= tol_min:
            return fwd[0][1]
    # Absolute nearest
    candidates.sort(key=lambda x: abs(x[0] - tip_minutes))
    if candidates and abs(candidates[0][0] - tip_minutes) <= tol_min:
        return candidates[0][1]
    return None

def ms_to_london_hhmm(ms: int) -> tuple[str, int]:
    """
    Convert API start_time (epoch ms) → ('HH:MM', minutes_from_midnight) in Europe/London.
    """
    # Always interpret the epoch in UTC, then convert to London
    dt_utc = datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)
    if _LONDON_TZ:
        dt_ldn = dt_utc.astimezone(_LONDON_TZ)
    else:
        # Coarse fallback if zoneinfo missing
        is_bst = dt_utc.month in (4,5,6,7,8,9,10)
        dt_ldn = dt_utc + timedelta(hours=1 if is_bst else 0)
    hhmm = f"{dt_ldn.hour:02d}:{dt_ldn.minute:02d}"
    minutes = dt_ldn.hour * 60 + dt_ldn.minute
    return hhmm, minutes

def hhmm_to_minutes(hhmm: str) -> Optional[int]:
    m = re.search(r"\b(\d{1,2}):(\d{2})\b", hhmm or "")
    if not m: return None
    try: return int(m.group(1)) * 60 + int(m.group(2))
    except: return None

# -----------------------------
# Betmaster API resolver
# -----------------------------
async def fetch_json_request(context, url):
    resp = await context.request.get(url)
    if not resp.ok:
        raise RuntimeError(f"HTTP {resp.status} for {url}")
    return await resp.json()

async def click_runner_by_startnr(pageB: Page, runner_name: str, start_nr: int):
    """
    Deterministically click runner using start number.
    """

    # Wait for race hydration
    await pageB.wait_for_selector(
        ".KambiBC-racing-betoffer-row",
        timeout=15000
    )

    rows = await pageB.query_selector_all(".KambiBC-racing-betoffer-row")

    idx = start_nr - 1
    if idx < 0 or idx >= len(rows):
        raise RuntimeError(f"Runner startNr {start_nr} out of bounds")

    row = rows[idx]

    # Guardrail: verify name
    name_el = await row.query_selector(".KambiBC-racing-participant")
    text = (await name_el.inner_text()).lower()

    if normalize_name(runner_name) not in normalize_name(text):
        raise RuntimeError(
            f"Runner mismatch — expected {runner_name}, got {text}"
        )

    # Click odds / SP button
    btn = await row.query_selector("button.KambiBC-betty-outcome")
    if not btn:
        raise RuntimeError("No odds button found for runner")

    await btn.click()

    # Assert betslip contains runner
    await pageB.wait_for_function(
        f"document.body.innerText.toLowerCase().includes('{normalize_name(runner_name)}')",
        timeout=5000
    )


# -----------------------------
# End-to-end tip handler
# -----------------------------

async def handle_tip(tip, pageA, pageB):
    """
    Handles a single Telegram tip.
    One-and-done per dog.
    JS owns all DOM interaction.
    """

    try:
        # -------------------------------
        # 0) Session guard
        # -------------------------------
        if await is_logged_out(pageA):
            logger.warning("🔒 Session expired — re-logging in before handling tip")
            await ensure_logged_in(pageA)

        # -------------------------------
        # 1) Parse core fields
        # -------------------------------

        dog = tip["dog"]
        track = tip["track"]
        off_time = tip["time"]
        back_odds = tip.get("back_odds")

        logger.info(
            f"[tip] {track} {off_time} | {dog} | back={back_odds} | value={tip.get('value')}"
        )

        if not dog or not track or not off_time or not back_odds:
            logger.warning("❌ Tip missing required fields")
            return


        value = tip.get("value")
        if value is None or value < CFG.min_value_pct:
            logger.info(f"⏭️ Value {value}% below threshold {CFG.min_value_pct}% — skipping")
            return
        

        logger.info(f"[value] Passed value gate ({value}% >= {CFG.min_value_pct}%)")

        # -------------------------------
        # 2) One-and-done dedupe (STRICT)
        # -------------------------------
        attempt_key = make_attempt_key(track, off_time, dog)

        if attempt_key in ATTEMPTED_KEYS:
            logger.info(f"⏭️ Already attempted {attempt_key} — skipping permanently")
            return

        # Lock immediately — regardless of outcome
        logger.info(f"🔓 Attempting {attempt_key}")

        ATTEMPTED_KEYS.add(attempt_key)


        # -------------------------------
        # 3) Resolve race via API
        # -------------------------------
        res = await resolve_rr_event(pageA, tip)
        if not res:
            logger.warning(
                f"[race] Could not resolve race for "
                f"{tip.track} {tip.time} — skipping"
            )
            return

        event_id, race_url = res

        if not race_url:
            logger.warning("❌ Failed to resolve race via API")
            return
        logger.info(f"📍 Resolved race URL: {race_url}")

        # -------------------------------
        # 4) Navigate race page (Tab B)
        # -------------------------------

        # HARD navigation: mimic manual paste + refresh
        await pageB.goto(race_url, wait_until="load")
        await pageB.reload(wait_until="load")

        # Now wait for race hydration
        await pageB.wait_for_selector(
            ".KambiBC-racing-betoffer-row",
            timeout=20000
        )
        
        await clear_betslip(pageB)



        # -------------------------------
        # 5) Resolve runner + odds via API
        # -------------------------------
        runner_res = await resolve_rr_runner_and_odds(
            pageB.context,
            event_id=event_id,
            runner_name=dog,
        )

        if not runner_res:
            logger.warning("❌ Runner not available or SP-only — aborting")
            return

        logger.info(
            f"[runner] Resolved {dog} | startNr={runner_res['startNr']} | "
            f"odds={runner_res['decimal_odds']}"
        )

        # -------------------------------
        # 5.5) Odds validation (Phase 3)
        # -------------------------------
        api_odds = runner_res["decimal_odds"]

        if not is_price_acceptable(api_odds, back_odds):
            logger.warning(
                f"❌ Odds moved: API={api_odds} < min acceptable "
                f"{min_acceptable_odds(back_odds)} — skipping"
            )
            return

        # -------------------------------
        # 6) Click runner deterministically
        # -------------------------------
        await click_runner_by_startnr(
            pageB,
            runner_name=dog,
            start_nr=runner_res["startNr"],
        )

        logger.info("✅ Runner clicked and present in betslip — PHASE 2 COMPLETE")

        # -------------------------------
        # 7) Place bet (stake + click)
        # -------------------------------

        raw_stake = CFG.target_profit / (api_odds - 1.0)
        stake = round_to_nearest_50p(raw_stake)

        logger.info(f"💷 Entering stake: £{stake}")

        await place_bet_rainbow_riches(
            page=pageB,
            stake=stake,
            expected_odds=api_odds,
        )

        logger.info("🎯 Bet placement attempted")
        

    except Exception as e:
        logger.exception(f"🔥 handle_tip crashed: {e}")
# -----------------------------
# Fake-tip mode (no Telegram)
# -----------------------------
async def run_fake_tip_mode(
    queue: asyncio.Queue,
    processing_lock: asyncio.Lock,
    pageA,
    pageB,
    initial_tip: str | None = None,
):
    """
    Replaces the Telegram listener with a local stdin loop.
    Paste a multi-line tip, then press Enter on a blank line to submit.
    Pass an initial tip via --fake-tip to fire one immediately on startup.
    """
    from telegram_handler import parse_telegram_message

    async def worker():
        while True:
            tip_text = await queue.get()
            try:
                async with processing_lock:
                    tip = parse_telegram_message(tip_text)
                    if not tip:
                        logger.warning("[fake] ❌ Could not parse tip — check format")
                        continue
                    await handle_tip(tip, pageA, pageB)
            except Exception as e:
                logger.exception(f"[fake] handle_tip error: {e}")
            finally:
                queue.task_done()

    worker_task = asyncio.create_task(worker())

    if initial_tip:
        logger.info(f"[fake] Firing initial tip from --fake-tip flag")
        await queue.put(initial_tip.replace("\\n", "\n"))

    loop = asyncio.get_event_loop()
    print("\n[fake] ── Fake tip mode active ──────────────────────────────")
    print("[fake] Paste a tip (same format as Telegram), then press Enter")
    print("[fake] on a blank line to submit. Ctrl+C to exit.\n")

    try:
        while True:
            lines = []
            while True:
                line = await loop.run_in_executor(None, input, "")
                if line == "":
                    break
                lines.append(line)
            if lines:
                tip_text = "\n".join(lines)
                logger.info(f"[fake] Enqueuing tip:\n{tip_text}")
                await queue.put(tip_text)
    except (KeyboardInterrupt, EOFError):
        logger.info("[fake] Exiting fake tip mode")
    finally:
        worker_task.cancel()
        try:
            await worker_task
        except Exception:
            pass


# -----------------------------
# Orchestration
# -----------------------------
async def run_bot(fake_tip: str | None = None):
    # --- use persistent profile instead of ephemeral context ---
    playwright, browser, pageA, pageB = await make_persistent_browser()

    print(f"[nav] Tab A → {BASE_URL}")
    try:
        await pageA.goto(BASE_URL, wait_until="domcontentloaded", timeout=40000)
    except Exception as e:
        print(f"[nav] Failed to load homepage {BASE_URL}: {e}")
    await clear_blocking_overlays(pageA)

    # Ensure logged in BEFORE doing anything else
    await ensure_logged_in(pageA)

    # IMPORTANT: clear betslip on TAB B
    await clear_betslip(pageB)


    await pageB.goto("about:blank")
    await clear_blocking_overlays(pageB)

    # Confirm both tabs hydrated properly
    for p in (pageA, pageB):
        try:
            await p.wait_for_load_state("domcontentloaded", timeout=10000)
        except Exception:
            pass
        await clear_blocking_overlays(p)

    # ==========================================================
    # ✅ START UI KEEPALIVE HERE (ONCE, AFTER LOGIN)
    # ==========================================================
    keepalive_task = asyncio.create_task(
        ui_keepalive(pageA, interval_seconds=240)
    )
    logger.info("🫀 UI keepalive started (Tab A)")


    # ==========================================================
    # ✅ START SESSION HEARTBEAT (API, NOT UI)
    # ==========================================================
    heartbeat_task = asyncio.create_task(
        session_heartbeat(browser, interval_seconds=180)
    )
    logger.info("💓 Session heartbeat started (API)")


    queue: asyncio.Queue[str] = asyncio.Queue()
    processing_lock = asyncio.Lock()
    sweeper_task = asyncio.create_task(overlay_sweeper(pageA, pageB, interval_seconds=10.0))

    try:
        if fake_tip is not None:
            # --- Fake tip mode: no Telegram connection needed ---
            await run_fake_tip_mode(queue, processing_lock, pageA, pageB, initial_tip=fake_tip)
        else:
            # --- Live mode: connect to Telegram ---
            client, worker_task = await setup_telegram(queue, processing_lock, handle_tip, pageA, pageB)
            await client.run_until_disconnected()
            worker_task.cancel()
            try:
                await worker_task
            except Exception:
                pass
    finally:
        for task in (sweeper_task, keepalive_task, heartbeat_task):
            task.cancel()
            try:
                await task
            except Exception:
                pass
        try:
            await browser.close()
        except Exception:
            pass
        try:
            await playwright.stop()
        except Exception:
            pass


# -----------------------------
# CLI
# -----------------------------
async def main():

    parser = argparse.ArgumentParser(description="Betmaster Telegram two-tab bot (API resolve + DOM place)")
    parser.add_argument("--min-value", type=float, help="Min Value%% gate (e.g. 102)")
    parser.add_argument("--target-profit", type=float, help="Stake-to-win target profit (e.g. 60)")
    parser.add_argument("--max-odds", type=float, help="Skip if tip Back exceeds this (e.g. 5)")
    parser.add_argument("--odds-tol-abs", type=float, help="Absolute price tolerance (default 0.02)")
    parser.add_argument("--odds-tol-pct", type=float, help="Relative price tolerance in percent")
    parser.add_argument(
        "--fake-tip", type=str, metavar="TIP_TEXT",
        help="Run without Telegram using a fake tip. Use \\n between lines. "
             "Omit to enter tips interactively once the bot starts.",
    )
    args = parser.parse_args()

    cfg = CFG
    if args.min_value is not None:    cfg = replace(cfg, min_value_pct=float(args.min_value))
    if args.target_profit is not None:cfg = replace(cfg, target_profit=float(args.target_profit))
    if args.max_odds is not None:     cfg = replace(cfg, max_odds=float(args.max_odds))
    if args.odds_tol_abs is not None: cfg = replace(cfg, odds_tolerance_abs=float(args.odds_tol_abs))
    if args.odds_tol_pct is not None: cfg = replace(cfg, odds_tolerance_pct=float(args.odds_tol_pct))
    globals()["CFG"] = cfg

    await run_bot(fake_tip=args.fake_tip)

if __name__ == "__main__":
    asyncio.run(main())


async def test_fake_tip(pageA, pageB):
    fake_tip = {
        "dog": "Lady Danielle",
        "track": "Central Park",
        "time": "11:09",
        "back_odds": 0.0,  # unused for now
    }

    logger.info("[TEST] Using hardcoded fake tip")

    event_id = 1025994508
    race_url = f"https://www.rainbowrichescasino.com/sports#racing/event/{event_id}"

    await pageB.goto(race_url, wait_until="load")

    await pageB.reload(wait_until="load")

    logger.info("[TEST] Race page opened")

    res = await resolve_rr_runner_and_odds(
        pageB,
        event_id=event_id,
        runner_name=fake_tip["dog"],
    )

    logger.info(f"[TEST] API resolve result: {res}")
