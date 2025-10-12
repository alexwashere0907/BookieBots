#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Betmaster two-tab bot — Telegram-connected, API-resolve (race/dog), DOM place (betslip)

Flow
1) Parse Telegram tip: (track, HH:MM, dog, back odds, value)
2) Call Betmaster "matches" feed → pick matchId by (track, time)
3) Call Betmaster race detail → pick runnerId/dog by name
4) Open race page UI → click runner → price check → stake-to-win → place bet → confirm receipt
"""

import asyncio
import re
import time as monotime
import argparse
from dataclasses import dataclass, replace
from datetime import datetime, date, time, timezone, timedelta
from typing import Optional, Tuple, List, Dict, Set

from telethon import TelegramClient, events
from playwright.async_api import async_playwright, Page

from zoneinfo import ZoneInfo  # Python 3.9+

LONDON = ZoneInfo("Europe/London")
UTC = timezone.utc

TIME_TOL_MIN = 30   # was 2; 5–6 is realistic for greyhound feeds

from datetime import datetime, date, time, timedelta, timezone
try:
    from zoneinfo import ZoneInfo  # Py3.9+
    _LONDON_TZ = ZoneInfo("Europe/London")
except Exception:
    _LONDON_TZ = None  # we'll fall back to a coarse offset below
    
# -----------------------------
# Config
# -----------------------------
BETMASTER_HOST = "betmaster.co.uk"
SITE_ROOT = f"https://{BETMASTER_HOST}"
RACE_PAGE_URL = SITE_ROOT + "/en/racing/sports/greyhound/match/{match_id}"
MATCHES_FEED_URL_TEMPLATE = (
    SITE_ROOT
    + "/api/racing/pythia/matches/sport/all"
    + "?market=unitedkingdom&markets_set=main&sport_id=4000002"
    + "&start_time_from={start_ms}&start_time_to={end_ms}"
)
RACE_DETAIL_URLS = [
    # Most strict → least strict
    SITE_ROOT + "/api/racing/pythia/match/{match_id}?market=unitedkingdom&markets_set=main&sport_id=4000002",
    SITE_ROOT + "/api/racing/pythia/match/{match_id}?markets_set=main&sport_id=4000002",
    SITE_ROOT + "/api/racing/pythia/match/{match_id}?sport_id=4000002",
    SITE_ROOT + "/api/racing/pythia/match/{match_id}",
]

# Stake & price rules
@dataclass
class BookieConfig:
    target_profit: float = 10.00        # Stake-to-win target
    min_value_pct: float = 100.0        # Only bet if Telegram "Value" >= this %
    max_odds: Optional[float] = None    # e.g. 5.0 → skip if tip odds > 5
    odds_tolerance_abs: float = 0.02    # Accept tiny dip from tip "Back:"
    odds_tolerance_pct: float = 0.0     # Optional relative tolerance

CFG = BookieConfig()

# Telegram credentials (replace in prod)
API_ID    = XXX
API_HASH  = "XXXX"
BOT_TOKEN = "XXXXXXX"

SESSION_NAME = "betmaster_session"
BASE_URL = SITE_ROOT + "/sport/greyhoundracing"

# In-memory guard so we only bet once per (race,dog) this run
PLACED_KEYS: Set[str] = set()

# Optional: only act on Telegram messages that contain these brand names; [] means accept all
BRAND_FILTERS: List[str] = []

IRISH_MARKETS = ["ireland", "roi"]   # or [] to disable


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
}

def norm_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def norm_key(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())

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

async def ensure_market_tab(page, market_text: str = "Win Each Way"):
    # Click the "Win Each Way" tab if it's present; fall back to "Win" / "Win Only"
    candidates = [
        f":is(button,[role='tab']):has-text('{market_text}')",
        ":is(button,[role='tab']):has-text('Win Only')",
        ":is(button,[role='tab']):has-text('Win')",
        "button:has-text('Win Each Way')",
        "button:has-text('Win Only')",
        "button:has-text('Win')",
    ]
    for sel in candidates:
        tab = page.locator(sel).first
        if await tab.count():
            try:
                await tab.click()
                await asyncio.sleep(0.15)
                return True
            except:
                pass
    return False

async def safe_count(locator) -> int:
    try:
        return await locator.count()
    except Exception:
        return 0
    
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
# Telegram parser
# -----------------------------
def parse_telegram_message(text: str):
    try:
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        if len(lines) < 3:
            return None

        brand = lines[0].lower()
        if BRAND_FILTERS and not any(b in brand for b in BRAND_FILTERS):
            return None

        dog = lines[1]
        line2 = lines[2].replace("–", "-")
        track, race_time = line2.split("-")
        track = track.strip()
        race_time = race_time.strip()

        back_line = next((l for l in lines if l.lower().startswith("back:")), None)
        back_odds = None
        if back_line:
            m = re.search(r"(\d+(?:\.\d+)?)", back_line)
            if m:
                back_odds = float(m.group(1))

        value_line = next((l for l in lines if "value" in l.lower()), None)
        value_pct = None
        if value_line:
            m = re.search(r"(\d+(?:\.\d+)?)\s*%", value_line)
            if m:
                value_pct = float(m.group(1))

        return {
            "dog": dog,
            "track": track,
            "time": race_time,
            "back_odds": back_odds,
            "value": value_pct,
        }
    except Exception as e:
        print(f"❌ Parse error: {e}")
        return None

async def fetch_race_detail(page: Page, match_id: int) -> dict:
    """
    Betmaster can require various query params on /api/racing/pythia/match/{id}.
    Try strict→loose UK variants, then optional ROI variants. Log failures.
    """
    tried = []

    def _u(qs: str | None) -> str:
        base = f"{SITE_ROOT}/api/racing/pythia/match/{match_id}"
        return f"{base}?{qs}" if qs else base

    # 1) UK-flavoured attempts (strict → loose)
    uk_variants = [
        "market=unitedkingdom&markets_set=main&sport_id=4000002&provider=py1",
        "market=unitedkingdom&markets_set=main&sport_id=4000002",
        "market=unitedkingdom&markets_set=all&sport_id=4000002&provider=py1",
        "market=unitedkingdom&markets_set=all&sport_id=4000002",
        "markets_set=main&sport_id=4000002&provider=py1",
        "markets_set=main&sport_id=4000002",
        "sport_id=4000002&provider=py1",
        "sport_id=4000002",
        None,  # bare
    ]
    for qs in uk_variants:
        url = _u(qs)
        try:
            return await fetch_json(page, url)
        except Exception as e:
            tried.append(f"{url} -> {e}")

    # 2) Optional: ROI-flavoured attempts (if enabled)
    for market in IRISH_MARKETS:
        roi_variants = [
            f"market={market}&markets_set=main&sport_id=4000002&provider=py1",
            f"market={market}&markets_set=main&sport_id=4000002",
            f"market={market}&markets_set=all&sport_id=4000002&provider=py1",
            f"market={market}&markets_set=all&sport_id=4000002",
            f"market={market}&sport_id=4000002&provider=py1",
            f"market={market}&sport_id=4000002",
        ]
        for qs in roi_variants:
            url = _u(qs)
            try:
                return await fetch_json(page, url)
            except Exception as e:
                tried.append(f"{url} -> {e}")

    # 3) All failed — emit concise log to help us see the shape that’s needed
    print("[api] All race-detail variants failed. Tried:")
    for t in tried[:12]:
        print("   ", t)
    raise RuntimeError(f"Could not fetch race detail for match_id={match_id}")

# -----------------------------
# Overlay handling / page readiness
# -----------------------------
async def clear_blocking_overlays(page: Page, timeout_ms: int = 8000) -> bool:
    """Dismiss common overlays (cookie, reality check, wallet/GBP modal, generic dialogs)."""
    try:
        if page.is_closed():
            return False
    except Exception:
        return False

    dismissed = False
    slice_timeout = max(1500, timeout_ms // 4)

    # A) Usual suspects
    patterns = [
        # Reality check / generic popup overlays
        ("div[data-component='RealityCheckPopup'], .RealityCheckPopupOverlay, .css-fjv5kt-PopupOverlay-RealityCheckPopupOverlay",
         ["Continue", "OK", "I Understand", "Close", "Proceed"]),
        ("[data-component*='Popup'], [role='dialog'], [class*='PopupOverlay'], .modal, .overlay",
         ["Continue", "OK", "Close", "Got it", "Accept", "I agree"]),
        # Cookie banners
        ("[id*='cookie'], [class*='cookie'], [aria-label*='cookie']",
         ["Accept", "I agree", "Got it", "OK"]),
    ]

    for overlay_sel, btn_texts in patterns:
        try:
            overlay = page.locator(overlay_sel).first
            if await overlay.count():
                try:
                    await overlay.wait_for(state="visible", timeout=slice_timeout)
                except Exception:
                    pass
                if await overlay.is_visible():
                    for t in btn_texts:
                        btn = overlay.locator(f":is(button,[role='button']):has-text('{t}')").first
                        if await btn.count():
                            try:
                                await btn.click()
                                dismissed = True
                                break
                            except Exception:
                                try:
                                    await btn.evaluate("el => el.click()")
                                    dismissed = True
                                    break
                                except Exception:
                                    pass
                    if not dismissed:
                        try:
                            await page.keyboard.press("Escape")
                            await page.wait_for_timeout(150)
                            if not await overlay.is_visible():
                                dismissed = True
                        except Exception:
                            pass
        except Exception:
            pass

    # B) Wallet/Deposit modal specific to Betmaster (strings seen in your log)
    try:
        # Look for any dialog/modal containing these cues
        wallet_dialog = page.locator(
            ":is([role='dialog'], [class*='modal'], [data-component*='Popup'], body)"
        ).filter(has_text=re.compile(r"(choose\s+wallet|deposit|gbp)", re.I)).first

        if await wallet_dialog.count():
            # Prefer clicking a wallet/currency or a primary continue/close action
            candidates = [
                ":is(button,[role='button']):has-text('GBP')",
                ":is(button,[role='button']):has-text('Continue')",
                ":is(button,[role='button']):has-text('OK')",
                ":is(button,[role='button']):has-text('Close')",
                "[aria-label='Close']",
            ]
            for sel in candidates:
                btn = wallet_dialog.locator(sel).first
                if await btn.count():
                    try:
                        await btn.click()
                        dismissed = True
                        break
                    except Exception:
                        try:
                            await btn.evaluate("el => el.click()")
                            dismissed = True
                            break
                        except Exception:
                            pass
            # Last resort: press Escape
            if not dismissed:
                try:
                    await page.keyboard.press("Escape")
                    await page.wait_for_timeout(150)
                    dismissed = True
                except Exception:
                    pass
    except Exception:
        pass

    return dismissed

# Betmaster-flavoured readiness (drop-in replacement)
async def wait_for_race_page_ready(page, timeout_ms: int = 45000, runner_hint: str | None = None) -> bool:
    """
    Betmaster-ready version.
    Consider the page 'ready' when we can see either:
      • any WIN market button (data-web-test^='match_' & *WIN*)
      • or visible odds text (fractional/decimal) under <main>
      • or (if runner_hint given) the runner's <p> is visible
    Includes overlay clearing, tab nudge, lazy-load scroll/resize.
    """
    t0 = monotime.monotonic()
    hb_next = 0.0

    def ms_left() -> int:
        return max(0, int(timeout_ms - (monotime.monotonic() - t0) * 1000))

    try:
        await page.wait_for_load_state("domcontentloaded", timeout=min(10000, ms_left()))
    except Exception:
        pass

    await clear_blocking_overlays(page)

    # Nudge the market tab once up-front
    for sel in (
        ":is(button,[role='tab']):has-text('Win Each Way')",
        ":is(button,[role='tab']):has-text('Win Only')",
        ":is(button,[role='tab']):has-text('Win')",
        "button:has-text('Win Each Way')",
        "button:has-text('Win Only')",
        "button:has-text('Win')",
    ):
        try:
            tab = page.locator(sel).first
            if await safe_count(tab):
                await tab.click()
                await asyncio.sleep(0.2)
                break
        except Exception:
            pass

    # Anchors for Betmaster skin
    win_btns   = page.locator("[data-web-test^='match_'][data-web-test*='WIN']")
    frac_odds  = page.locator("main").get_by_text(re.compile(r"\b\d+\s*/\s*\d+\b"))
    dec_odds   = page.locator("main").get_by_text(re.compile(r"\b\d+(?:\.\d+)\b"))
    runner_p   = page.locator("main p").filter(has_text=re.compile(re.escape(runner_hint), re.I)) if runner_hint else None

    # Fast path
    if await safe_count(win_btns) or await safe_count(frac_odds) or await safe_count(dec_odds) or (runner_p and await safe_count(runner_p)):
        print("[race] ✅ Ready immediately (found WIN buttons/odds/runner).")
        return True

    # Poll with heartbeats + gentle DOM nudges
    while ms_left() > 0:
        await clear_blocking_overlays(page)

        n_win  = await safe_count(win_btns)
        n_frac = await safe_count(frac_odds)
        n_dec  = await safe_count(dec_odds)
        n_run  = await safe_count(runner_p) if runner_p else 0

        now = monotime.monotonic()
        if now >= hb_next:
            print(f"[race] …waiting (win_btns={n_win}, frac_hits={n_frac}, dec_hits={n_dec}, runner={n_run}, {ms_left()}ms left)")
            hb_next = now + 2.0

        if n_win or n_frac or n_dec or n_run:
            print(f"[race] ✅ Ready (win_btns={n_win}, frac={n_frac}, dec={n_dec}, runner={n_run}).")
            return True

        # Lazy-load nudges: resize + tiny scrolls + re-click tab if we still see nothing
        try:
            await page.evaluate("window.dispatchEvent(new Event('resize'))")
        except Exception:
            pass
        try:
            await page.mouse.wheel(0, 300)
        except Exception:
            pass
        for sel in ("button:has-text('Win Each Way')", "button:has-text('Win Only')", "button:has-text('Win')"):
            btn = page.locator(sel).first
            if await safe_count(btn):
                try:
                    await btn.click()
                    await asyncio.sleep(0.15)
                    break
                except Exception:
                    pass

        await asyncio.sleep(0.25)

    # One reload rescue
    print("[race] 🔄 Reloading page to rescue readiness…")
    try:
        await page.reload(wait_until="domcontentloaded")
        await clear_blocking_overlays(page)
        if await safe_count(win_btns) or await safe_count(frac_odds) or await safe_count(dec_odds) or (runner_p and await safe_count(runner_p)):
            print("[race] ✅ Ready after reload.")
            return True
    except Exception:
        pass

    # Final snippet for debugging
    try:
        snippet = (await page.locator("main").inner_text())[:800]
    except Exception:
        try:
            snippet = (await page.locator("body").inner_text())[:800]
        except Exception:
            snippet = ""
    print("[race] ❌ Ready check failed. Snippet:\n" + snippet)
    return False


# -----------------------------
# Betmaster API resolver
# -----------------------------
async def fetch_json(page: Page, url: str):
    """Fetch JSON in-page so cookies/session apply. Error includes URL+status+short body."""
    return await page.evaluate(
        """async (url) => {
            const res = await fetch(url, { credentials: 'include' });
            const text = await res.text().catch(() => '');
            if (!res.ok) {
                throw new Error(`HTTP ${res.status} for ${url} :: ${text.slice(0,200)}`);
            }
            try { return JSON.parse(text); } catch {
                throw new Error(`Bad JSON for ${url} :: ${text.slice(0,200)}`);
            }
        }""",
        url,
    )

async def resolve_match(page: Page, tip, time_tol_min: int = TIME_TOL_MIN) -> Optional[Tuple[int, str]]:
    """
    Resolve just the match (race) from the matches feed.
    Compares TIP time (Europe/London) to RACE time converted to Europe/London.
    Returns (match_id, race_url) or None if not found.
    """
    # Build 'today' window in Europe/London, then query the feed
    start_ms, end_ms = london_today_window_ms()
    matches_url = MATCHES_FEED_URL_TEMPLATE.format(start_ms=start_ms, end_ms=end_ms)

    data = await fetch_json(page, matches_url)
    matches = data.get("matches", [])
    if not matches:
        print("[api] No matches returned.")
        return None

    tip_track_norm = normalize_track_name(tip["track"])
    tip_minutes = hhmm_to_minutes(tip["time"])
    if tip_minutes is None:
        print("[tip] Bad time format.")
        return None

    # Collect all races for that track with their *London* minutes
    candidates: List[Tuple[int, dict]] = []
    for m in matches:
        try:
            tname = m["info_static"]["tournament"]["name"]["en"]
            if normalize_track_name(tname) != tip_track_norm:
                continue
            start_ms_race = m["info_static"]["start_time"]
            _hhmm, race_minutes_ldn = ms_to_london_hhmm(start_ms_race)
            candidates.append((race_minutes_ldn, m))
        except Exception:
            continue

    if not candidates:
        print(f"[api] No meeting/time candidate for '{tip['track']}' {tip['time']}")
        return None

    # Pick exact match, else the nearest *forward* start, else absolute nearest — all within tolerance
    chosen = _pick_best_race(tip_minutes, candidates, tol_min=time_tol_min)
    if not chosen:
        # For debugging, show a few nearby options in London time
        preview = []
        for race_min, m in sorted(candidates, key=lambda x: abs(x[0] - tip_minutes))[:8]:
            hh = race_min // 60
            mm = race_min % 60
            preview.append(f"{hh:02d}:{mm:02d} (Δ={abs(race_min - tip_minutes)}) id={m['id']}")
        print("[api] No race within tolerance. Nearby:", "; ".join(preview))
        return None

    match_id = chosen["id"]
    race_url = RACE_PAGE_URL.format(match_id=match_id)
    return match_id, race_url



# -----------------------------
# Race-page DOM: add runner / betslip
# -----------------------------
async def select_win_only_if_present(page: Page):
    for txt in ("Win Only", "Win"):
        tab = page.get_by_role("tab", name=txt)
        if await tab.count():
            try:
                await tab.click(); await asyncio.sleep(0.15); return
            except Exception:
                pass
    loc = page.locator("button:has-text('Win Only'), [data-test='market-tab']:has-text('Win')").first
    if await loc.count():
        try: await loc.click()
        except: pass

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().lower()

def _norm_txt(s: str) -> str:
    import re
    return re.sub(r"\s+", " ", (s or "")).strip().lower()

async def list_visible_runners(page: Page, limit: int = 24) -> List[str]:
    # Broad sweep: any <p> inside main/race-card that looks like a runner line (not “Age:”/owner/etc.)
    scope = page.locator("main, [data-test='race-card'], [data-component='RaceCard']").first
    ps = scope.locator("p")
    n = await ps.count()
    names = []
    for i in range(min(n, limit*4)):  # allow extra; we filter texty lines
        try:
            t = (await ps.nth(i).inner_text()).strip()
        except Exception:
            continue
        tl = t.lower()
        if not t or len(t) < 2:
            continue
        if any(k in tl for k in ("age:", "owner:", "trainer:", "o:", "t:", "class", "each way", "win each way", "result")):
            continue
        # Skip odds-like fragments
        if any(ch in t for ch in ("/", "SP")) and len(t) <= 6:
            continue
        names.append(t)
        if len(names) >= limit:
            break
    return names

def _lc(s): return (s or "").strip().lower()

async def add_runner_from_race_page(page: Page, match_id: int, runner_name: str):
    """
    Betmaster skin: buttons carry data-web-test like 'match_<id>_WIN...'.
    Find a button for this match_id, then climb to its container and pick the nearest <p> name.
    Click the button whose nearby name matches runner_name (fuzzy).
    """
    scope = page.locator("main, [data-test='race-card'], [data-component='RaceCard']").first
    # All WIN buttons for this match
    btns = scope.locator(f":is(button,a)[data-web-test^='match_{match_id}_WIN']").filter(has_text="").all()  # we’ll iterate via count
    count = await scope.locator(f":is(button,a)[data-web-test^='match_{match_id}_WIN']").count()
    if count == 0:
        # fallback: any odds-looking button
        count = await scope.locator(":is(button,a):has-text('/'), :is(button,a):has-text('SP'), :is(button,a):has-text('.')").count()

    want = _norm_txt(runner_name)

    # Walk each price button, read a local name, pick best match
    best = None  # (score, button_locator)
    buttons = scope.locator(f":is(button,a)[data-web-test^='match_{match_id}_WIN'], :is(button,a):has-text('/'), :is(button,a):has-text('SP'), :is(button,a):has-text('.')")

    for i in range(min(await buttons.count(), 24)):
        b = buttons.nth(i)
        try:
            # Find a nearby name: query a few ancestor levels then the first <p> inside
            name_txt = await b.evaluate("""
                (el) => {
                  const up = (n, k) => { let t=n; while(t && k--) t=t.parentElement; return t||n; };
                  const host = up(el, 3); // small card chunk
                  const ps = host.querySelectorAll('p');
                  let best = '';
                  for (const p of ps) {
                    const t = (p.textContent||'').trim();
                    if (!t) continue;
                    if (/age:|owner:|trainer:|class|win each way|result/i.test(t)) continue;
                    // Prefer a shortish human name line
                    if (t.length <= 60) { best = t; break; }
                    if (!best) best = t;
                  }
                  return best;
                }
            """)
        except Exception:
            name_txt = ""

        name_norm = _norm_txt(name_txt)
        if not name_norm:
            continue

        # simple fuzzy: exact, then substring either way
        score = 0
        if name_norm == want:
            score = 100
        elif want in name_norm or name_norm in want:
            score = 80
        # keep the best
        if score > 0 and (best is None or score > best[0]):
            best = (score, b)

    if best:
        btn = best[1]
        try:
            await btn.scroll_into_view_if_needed()
        except Exception:
            pass
        try:
            await btn.click()
            return (True, "[race] Selection clicked via match-linked button.")
        except Exception:
            try:
                await btn.evaluate("el => el.click()")
                return (True, "[race] Selection DOM-clicked via match-linked button.")
            except Exception:
                pass

    # If we get here, surface what we could see to help debug
    seen = await list_visible_runners(page)
    return (False, f"[race] Runner not found for '{runner_name}'. Seen: {seen[:8]}")

# -----------------------------
# Betslip utils
# -----------------------------
CURRENCY_RE = re.compile(r"[£$€]")

async def open_betslip(page: Page):
    if await page.locator('aside[data-component="AccountSidebar"]').count():
        return
    for s in ("div[data-test='Tab']", "button:has-text('Betslip')", "button[aria-label*='Betslip']"):
        loc = page.locator(s).first
        if await loc.count():
            try:
                await loc.click(); await asyncio.sleep(0.15)
                break
            except Exception:
                pass

def _parse_odds_from_text(t: str) -> Optional[float]:
    if not t: return None
    s = t.strip()
    if s.upper() == "SP": return None
    low = s.lower()
    if CURRENCY_RE.search(s) or ("stake" in low) or ("return" in low) or ("returns" in low):
        return None
    mf = re.search(r"\b(\d+)\s*/\s*(\d+)\b", s)
    if mf:
        try: n, d = float(mf.group(1)), float(mf.group(2)); return 1.0 + n/d
        except: return None
    md = re.search(r"\b\d+(?:\.\d+)?\b", s)
    if md:
        try: return float(md.group(0))
        except: return None
    return None

async def read_betslip_price_for_runner(page: Page, runner_name: str) -> Tuple[Optional[float], str]:
    await open_betslip(page)
    aside = page.locator('aside[data-component="AccountSidebar"]')
    if await aside.count() == 0:
        return (None, "[betslip] AccountSidebar not found.")

    item = aside.locator(
        "[data-test='betslip-item'], [data-component*='BetItem'], .betItem, [class*='bet']"
    ).filter(has_text=runner_name).first
    if await item.count() == 0:
        item = aside

    candidates = [
        "[data-test='selection-odds']",
        "[data-component*='Odds']",
        "[class*='odds']",
        "[class*='price']",
        "button:has-text('/')",
        "button:has-text('.')",
        "span:has-text('/')",
        "span:has-text('.')",
        "input[aria-label*='odds']",
    ]
    for sel in candidates:
        el = item.locator(sel).first
        if not await el.count():
            continue
        txt = ""
        try:
            txt = (await el.inner_text()).strip()
        except:
            try: txt = (await el.input_value()).strip()
            except: txt = ""
        if txt.strip().upper() == "SP":
            return (None, "[betslip] SP shown — no numeric odds.")
        dec = _parse_odds_from_text(txt)
        if dec is not None:
            return (dec, f"[betslip] Matched odds via '{sel}' → {dec}")
    return (None, "[betslip] No numeric odds element found (SP or missing).")

async def click_place_bet_button(page: Page) -> Tuple[bool, str]:
    aside = page.locator('aside[data-component="AccountSidebar"]')
    if not await aside.count():
        return (False, "[betslip] AccountSidebar not found.")

    try:
        await aside.evaluate("(n)=>{n.scrollTop = n.scrollHeight}")
        await asyncio.sleep(0.1)
    except:
        pass

    structure_selectors = [
        "aside[data-component='AccountSidebar'] [data-component='BetSlipButtons'] [data-component='PlaceButtons'] button",
        "aside[data-component='AccountSidebar'] [data-component='BetSlipButtons'] [data-component='PlaceButtons'] [role='button']",
    ]
    text_fallbacks = [
        "aside[data-component='AccountSidebar'] button:has-text('Place Bet')",
        "aside[data-component='AccountSidebar'] [role='button']:has-text('Place Bet')",
        "aside[data-component='AccountSidebar'] :has-text('Place Bet £')",
    ]

    for sel in structure_selectors + text_fallbacks:
        btn = page.locator(sel).first
        if await btn.count():
            try:
                await btn.scroll_into_view_if_needed()
                if await btn.is_visible():
                    try:
                        await btn.click()
                        return (True, f"[betslip] Clicked Place Bet via '{sel}'.")
                    except:
                        box = await btn.bounding_box()
                        if box:
                            await page.mouse.click(box["x"] + box["width"]/2, box["y"] + box["height"]/2)
                            return (True, f"[betslip] Mouse-clicked Place Bet via '{sel}'.")
                        await btn.evaluate("el => el.click()")
                        return (True, f"[betslip] DOM-clicked Place Bet via '{sel}'.")
            except Exception as e:
                print(f"[betslip] Click attempt failed on {sel}: {e}")

    try:
        snippet = (await aside.inner_text())[:600]
        return (False, "[betslip] Place Bet button not found. Sidebar snippet:\n" + snippet)
    except:
        return (False, "[betslip] Place Bet button not found and sidebar unreadable.")

async def remove_all_from_betslip(page: Page) -> bool:
    aside = page.locator('aside[data-component="AccountSidebar"]')
    if not await aside.count():
        return False

    for sel in (
        "aside[data-component='AccountSidebar'] button:has-text('Remove all')",
        "aside[data-component='AccountSidebar'] [role='button']:has-text('Remove all')",
        "aside[data-component='AccountSidebar'] a:has-text('Remove all')",
    ):
        btn = aside.locator(sel).first
        if await btn.count():
            try:
                await btn.scroll_into_view_if_needed()
                await btn.click()
                await asyncio.sleep(0.2)
                return True
            except Exception:
                pass

    # Fallback: click per-item removes
    for _ in range(12):
        rm = aside.locator(
            ".bet-item__remove, [aria-label='Remove'], button:has-text('Remove'), .remove, .trash"
        ).first
        if not await rm.count():
            break
        try:
            await rm.click()
            await asyncio.sleep(0.1)
        except Exception:
            break

    return not await aside.locator(
        "[data-test='betslip-item'], [data-component*='BetItem'], .betItem"
    ).count()

_MONEY_RE = re.compile(r"[£€$]\s*([0-9]+(?:\.[0-9]{1,2})?)")

async def wait_for_receipt(page: Page, expected_stake: float, expected_odds: float,
                           timeout_ms: int = 15000) -> Tuple[bool, str]:
    aside = page.locator('aside[data-component="AccountSidebar"]')
    if not await aside.count():
        return (False, "[betslip] Sidebar disappeared while waiting for receipt.")

    # Busy-wait for receipt or error cues
    try:
        await page.wait_for_function(
            """
            () => {
              const aside = document.querySelector('aside[data-component="AccountSidebar"]');
              if (!aside) return false;
              const txt = (aside.textContent||'').toLowerCase();
              if (/bet receipt/.test(txt) && /success/.test(txt)) return true;
              if (/error|insufficient|minim|suspend|reject|declin|login|price|change/.test(txt)) return true;
              return false;
            }
            """,
            timeout=timeout_ms
        )
    except Exception:
        pass

    try:
        txt = (await aside.inner_text()).replace("\n", " ")
    except Exception:
        txt = ""

    if re.search(r"bet receipt", txt, re.I) and re.search(r"success", txt, re.I):
        stake_m = re.search(r"Total\s+Stake[^£€$]*" + _MONEY_RE.pattern, txt)
        rets_m  = re.search(r"Total\s+Potential\s+Returns[^£€$]*" + _MONEY_RE.pattern, txt)
        stake_ui = float(stake_m.group(1)) if stake_m else None
        rets_ui  = float(rets_m.group(1)) if rets_m else None
        expected_returns = round(expected_stake * expected_odds, 2)
        ok_amounts = (
            (stake_ui is None or abs(stake_ui - expected_stake) <= 0.01) and
            (rets_ui  is None or abs(rets_ui  - expected_returns) <= 0.02)
        )
        if ok_amounts:
            return (True, "[decision] Bet confirmed (receipt visible).")
        else:
            return (True, f"[decision] Bet confirmed but amounts differ (stake={stake_ui}, returns={rets_ui}).")

    if re.search(r"error|insufficient|minim|suspend|reject|declin|login|price|change", txt, re.I):
        return (False, f"[betslip] Bookmaker rejected the bet: {txt[:240]}")

    return (False, "[betslip] Could not confirm bet (no receipt cue).")

async def place_bet_if_price_ok(page: Page, runner_name: str, want_odds: float) -> Tuple[bool, str]:
    price, why = await read_betslip_price_for_runner(page, runner_name)
    print(why)

    if price is None:
        removed = await remove_all_from_betslip(page)
        print("[betslip] Removed SP/missing-odds selection." if removed else "[betslip] Could not remove selection.")
        return (False, "[decision] No price available in betslip (SP or missing odds).")

    min_ok = min_acceptable_odds(want_odds)
    if min_ok and price + 1e-9 < min_ok:
        _ = await remove_all_from_betslip(page)
        print(f"[decision] Odds too low: current={price}, target={want_odds}, min_ok={min_ok}.")
        return (False, f"[decision] Odds too low: current={price}, target={want_odds}, min_ok={min_ok}.")

    # Stake-to-win at the CURRENT price
    stake = CFG.target_profit / max(price - 1.0, 0.01)
    stake = round(stake, 2)

    aside = page.locator('aside[data-component="AccountSidebar"]')
    stake_inputs = aside.locator("input[type='number'], input[role='spinbutton'], input")
    filled = False
    n = await stake_inputs.count()
    for i in range(min(n, 5)):
        inp = stake_inputs.nth(i)
        try:
            await inp.fill(str(stake))
            filled = True
            print(f"[betslip] Entered stake £{stake:.2f} (target win £{CFG.target_profit:.2f} @ {price}).")
            break
        except:
            continue
    if not filled:
        print("[betslip] Could not find a stake input; trying to place anyway.")

    # Accept price change if prompted (pre-click)
    for apc in (
        "aside[data-component='AccountSidebar'] button:has-text('Accept Price Changes')",
        "aside[data-component='AccountSidebar'] button:has-text('Accept')",
        "aside[data-component='AccountSidebar'] [role='button']:has-text('Accept')",
    ):
        btn = aside.locator(apc).first
        if await btn.count() and await btn.is_visible():
            try:
                await btn.click()
                print("[betslip] Accepted price change prompt.")
                break
            except:
                pass

    await clear_blocking_overlays(page)
    clicked, msg = await click_place_bet_button(page)
    print(msg)
    if not clicked:
        return (False, "[betslip] Could not find/place the bet button.")

    # Post-click Accept prompts; try a second click if needed
    for _ in range(2):
        accept_btn = aside.locator(":is(button,[role='button']):has-text('Accept')").first
        try:
            if await accept_btn.count() and await accept_btn.is_visible():
                await accept_btn.click()
                print("[betslip] Accepted price change after click.")
                clicked2, msg2 = await click_place_bet_button(page)
                print(msg2)
                if not clicked2:
                    break
            else:
                break
        except Exception:
            break

    ok, final_msg = await wait_for_receipt(page, expected_stake=stake, expected_odds=price, timeout_ms=15000)
    return (ok, final_msg)

# -----------------------------
# End-to-end tip handler
# -----------------------------
def make_bet_key(track: str, race_time: str, dog: str, match_id: Optional[int]) -> str:
    dog_key = norm_key(dog)
    if match_id is not None:
        return f"match:{match_id}|{dog_key}"
    return f"ptt:{normalize_track_name(track)}|{race_time.strip()}|{dog_key}"

async def handle_tip(pageA: Page, pageB: Page, tip_text: str):
    # Normalize incoming text
    if "\\n" in tip_text and ("\n" not in tip_text):
        tip_text = tip_text.replace("\\n", "\n")

    tip = parse_telegram_message(tip_text)
    if not tip:
        return

    # Quick gates
    if tip.get("value") is None or tip["value"] < CFG.min_value_pct:
        print(f"ℹ️ Tip skipped: Value {tip.get('value')}% below threshold {CFG.min_value_pct}%.")
        return
    if CFG.max_odds is not None and tip.get("back_odds") and tip["back_odds"] > CFG.max_odds:
        print(f"ℹ️ Tip skipped: tip Back {tip['back_odds']} exceeds max_odds {CFG.max_odds}.")
        return

    provisional_key = make_bet_key(tip["track"], tip["time"], tip["dog"], match_id=None)
    if provisional_key in PLACED_KEYS:
        print(f"ℹ️ Skipped duplicate (window): {provisional_key}")
        return

    await remove_all_from_betslip(pageB)

    print(f"[tip] Parsed: runner='{tip['dog']}' track='{tip['track']}' time='{tip['time']}' "
          f"want>={tip['back_odds']} value={tip.get('value')}%")

    # --- API resolution for match only ---
    resolved = await resolve_match(pageA, tip)
    if not resolved:
        print("❌ Could not resolve via API.")
        return

    match_id, race_url = resolved
    print(f"[api] Resolved match_id={match_id} → {race_url}")

    final_key = make_bet_key(tip["track"], tip["time"], tip["dog"], match_id=match_id)
    if final_key in PLACED_KEYS:
        print(f"ℹ️ Skipped duplicate (match): {final_key}")
        return

    # --- Navigate and place ---
    print(f"[nav] Tab B → {race_url}")
    await pageB.goto(race_url, wait_until="domcontentloaded")
    print("[nav] …DOM content loaded on Tab B, starting readiness check")
    ok_ready = await wait_for_race_page_ready(pageB, timeout_ms=45000, runner_hint=tip["dog"])
    print(f"[nav] …readiness result = {ok_ready}")
    if not ok_ready:
        print("❌ Race UI did not become ready in time.")
        return

    ok, msg = await add_runner_from_race_page(pageB, match_id, tip["dog"])
    if not ok:
        print(f"❌ {msg}")
        return

    placed, reason = await place_bet_if_price_ok(pageB, tip["dog"], tip["back_odds"] or 0.0)
    if placed:
        PLACED_KEYS.add(provisional_key)
        PLACED_KEYS.add(final_key)
        print(f"✅ {reason}  (dedupe keys saved)")
    else:
        print(f"ℹ️ {reason}")


# -----------------------------
# Overlay sweeper (periodic)
# -----------------------------
async def overlay_sweeper(pageA: Page, pageB: Page, interval_seconds: float = 10.0):
    while True:
        try:
            await clear_blocking_overlays(pageA)
            await clear_blocking_overlays(pageB)
        except Exception:
            pass
        await asyncio.sleep(interval_seconds)

# -----------------------------
# Orchestration
# -----------------------------
async def run_bot():
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=False)
        context = await browser.new_context()

        # Two tabs
        pageA = await context.new_page()
        print(f"[nav] Tab A → {BASE_URL}")
        try:
            await pageA.goto(BASE_URL, wait_until="domcontentloaded")
        except Exception as e:
            print(f"[nav] Failed to load homepage {BASE_URL}: {e}")
        await clear_blocking_overlays(pageA)

        pageB = await context.new_page()
        await pageB.goto("about:blank")
        await clear_blocking_overlays(pageB)

        print("Please log in on Tab A (first tab). Handle any cookie/geo/age banners.")
        input("Press Enter here once logged in...")

        for p in (pageA, pageB):
            try: await p.wait_for_load_state("domcontentloaded", timeout=10000)
            except: pass
            await clear_blocking_overlays(p)

        # Telegram client
        client = TelegramClient(SESSION_NAME, API_ID, API_HASH)
        await client.start(bot_token=BOT_TOKEN)
        print("🤖 Listening for Telegram tips... (Ctrl+C to stop)")

        queue: asyncio.Queue[str] = asyncio.Queue()
        processing_lock = asyncio.Lock()

        @client.on(events.NewMessage)
        async def _on_msg(event):
            try:
                text = event.message.raw_text or ""
                if BRAND_FILTERS and not any(b in text.lower() for b in BRAND_FILTERS):
                    return
                await queue.put(text)
            except Exception as e:
                print(f"⚠️ Telegram handler error: {e}")

        async def worker():
            while True:
                tip_text = await queue.get()
                try:
                    async with processing_lock:
                        await handle_tip(pageA, pageB, tip_text)
                except Exception as e:
                    print(f"⚠️ handle_tip error: {e}")
                finally:
                    queue.task_done()

        worker_task = asyncio.create_task(worker())
        sweeper_task = asyncio.create_task(overlay_sweeper(pageA, pageB, interval_seconds=10.0))

        try:
            await client.run_until_disconnected()
        finally:
            for task in (worker_task, sweeper_task):
                task.cancel()
                try: await task
                except Exception: pass
            try: await context.close()
            except Exception: pass
            try: await browser.close()
            except Exception: pass

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
    parser.add_argument("--dry-run", type=str, help="Test a single tip text (\\n between lines)")
    args = parser.parse_args()

    cfg = CFG
    if args.min_value is not None:    cfg = replace(cfg, min_value_pct=float(args.min_value))
    if args.target_profit is not None:cfg = replace(cfg, target_profit=float(args.target_profit))
    if args.max_odds is not None:     cfg = replace(cfg, max_odds=float(args.max_odds))
    if args.odds_tol_abs is not None: cfg = replace(cfg, odds_tolerance_abs=float(args.odds_tol_abs))
    if args.odds_tol_pct is not None: cfg = replace(cfg, odds_tolerance_pct=float(args.odds_tol_pct))
    globals()["CFG"] = cfg

    if args.dry_run:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=False)
            context = await browser.new_context()
            pageA = await context.new_page(); await pageA.goto(BASE_URL, wait_until="domcontentloaded")
            await clear_blocking_overlays(pageA)
            pageB = await context.new_page(); await pageB.goto("about:blank")
            await clear_blocking_overlays(pageB)

            print("Please log in on Tab A (first tab).")
            input("Press Enter here once logged in...")

            sweeper_task = asyncio.create_task(overlay_sweeper(pageA, pageB, interval_seconds=10.0))
            await handle_tip(pageA, pageB, args.dry_run)

            print("\n-- Browser left open. Press Ctrl+C to exit --")
            try:
                while True:
                    await asyncio.sleep(3600)
            except KeyboardInterrupt:
                pass
            finally:
                sweeper_task.cancel()

            await context.close(); await browser.close()
    else:
        await run_bot()

if __name__ == "__main__":
    asyncio.run(main())
