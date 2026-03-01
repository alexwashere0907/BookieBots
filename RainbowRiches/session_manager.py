#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Session management for the Rainbow Riches bot.
Responsible for:
  - Launching the persistent Playwright browser
  - Login detection and automated login flow
  - Session keepalive (UI heartbeat + Kambi API heartbeat)
  - Overlay/popup dismissal
"""

import asyncio
import os
import re
import logging
from pathlib import Path

from dotenv import load_dotenv
from playwright.async_api import async_playwright, Page

load_dotenv(Path(__file__).parent / ".env")

logger = logging.getLogger("betmaster")

# -----------------------------
# Config from environment
# -----------------------------
PROFILE_PATH = os.environ["PLAYWRIGHT_PROFILE_PATH"]
RR_USERNAME  = os.environ["RR_USERNAME"]
RR_PASSWORD  = os.environ["RR_PASSWORD"]

RR_OPEN_HEARTBEAT_URL = (
    "https://eu1.offering-api.kambicdn.com/"
    "offering/v2018/rrichesuk/event/live/open.json"
    "?lang=en_GB&market=GB&client_id=200&channel_id=1"
)


# -----------------------------
# Browser setup
# -----------------------------
async def make_persistent_browser():
    """
    Launch a Playwright Chromium window with a persistent profile.
    Returns (playwright, context, pageA, pageB).
    """
    pw = await async_playwright().start()

    context = await pw.chromium.launch_persistent_context(
        user_data_dir=PROFILE_PATH,
        headless=False,
        args=[
            "--disable-blink-features=AutomationControlled",
            "--disable-features=IsolateOrigins,site-per-process",
            "--no-default-browser-check",
            "--no-first-run",
        ],
    )

    pages = context.pages
    pageA = pages[0] if pages else await context.new_page()
    pageB = pages[1] if len(pages) > 1 else await context.new_page()

    print("[browser] ✅ Launched Playwright persistent browser (DOM fully hydrated).")
    return pw, context, pageA, pageB


# -----------------------------
# Session keepalive
# -----------------------------
async def ui_keepalive(page: Page, interval_seconds: int = 240):
    """Periodically performs harmless UI actions to keep the session alive."""
    while True:
        try:
            if page.is_closed():
                return
            await page.mouse.wheel(0, 50)
            await page.wait_for_timeout(200)
            await page.mouse.wheel(0, -50)
            await page.evaluate("() => document.body.focus()")
            logger.debug("[keepalive] UI heartbeat sent")
        except Exception:
            pass
        await asyncio.sleep(interval_seconds)


async def session_heartbeat(context, interval_seconds: int = 180):
    """Periodically calls Kambi open.json to keep the auth session alive (no UI touch)."""
    logger.info("💓 Session heartbeat started (open.json)")
    while True:
        try:
            resp = await context.request.get(RR_OPEN_HEARTBEAT_URL, timeout=10_000)
            if resp.ok:
                logger.debug("💓 Heartbeat OK")
            else:
                logger.warning(f"💔 Heartbeat failed HTTP {resp.status}")
        except Exception as e:
            logger.warning(f"💔 Heartbeat error: {e}")
        await asyncio.sleep(interval_seconds)


# -----------------------------
# Login state helpers
# -----------------------------
async def safe_count(locator) -> int:
    try:
        return await locator.count()
    except Exception:
        return 0


async def is_logged_in(page: Page) -> bool:
    """Logged-in = Deposit button visible."""
    try:
        deposit_btn = page.locator("button").filter(has_text=re.compile(r"deposit", re.I))
        return await deposit_btn.count() > 0
    except Exception:
        return False


async def is_logged_out(page: Page) -> bool:
    """Logged-out = Login / Join Now buttons visible."""
    try:
        login_btn = page.locator("button, a").filter(has_text=re.compile(r"(login|join)", re.I))
        return await login_btn.count() > 0
    except Exception:
        return False




# -----------------------------
# Overlay dismissal
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

    patterns = [
        ("div[data-component='RealityCheckPopup'], .RealityCheckPopupOverlay, .css-fjv5kt-PopupOverlay-RealityCheckPopupOverlay",
         ["Continue", "OK", "I Understand", "Close", "Proceed"]),
        ("[data-component*='Popup'], [role='dialog'], [class*='PopupOverlay'], .modal, .overlay",
         ["Continue", "OK", "Close", "Got it", "Accept", "I agree"]),
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

    # Wallet/Deposit modal
    try:
        wallet_dialog = page.locator(
            ":is([role='dialog'], [class*='modal'], [data-component*='Popup'], body)"
        ).filter(has_text=re.compile(r"(choose\s+wallet|deposit|gbp)", re.I)).first

        if await wallet_dialog.count():
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


# -----------------------------
# Login flow
# -----------------------------
async def ensure_logged_in(page: Page):
    try:
        await page.wait_for_selector(
            "button:has-text('Deposit'), button:has-text('Login'), a:has-text('Login')",
            timeout=8000
        )
    except Exception as e:
        logger.exception(f"⚠️ ensure_logged_in failed safely: {e}")
        return

    login_btn   = page.locator("button, a").filter(has_text=re.compile(r"(login|sign\\s*in)", re.I))
    deposit_btn = page.locator("button").filter(has_text=re.compile(r"deposit", re.I))

    login_visible   = await safe_count(login_btn) > 0 and await login_btn.first.is_visible()
    deposit_visible = await safe_count(deposit_btn) > 0

    if deposit_visible:
        logger.info("🔐 Logged in (deposit visible)")
        return

    clicked_login = False
    if login_visible and not await page.locator("form[data-qa='login_form']").count():
        logger.info("🔑 Clicking Login button")
        await login_btn.first.click()
        clicked_login = True

    if not clicked_login:
        logger.info("⏭️ Did not initiate login — skipping")
        return

    try:
        await page.wait_for_selector("form[data-qa='login__form']", timeout=10000)
    except Exception:
        logger.warning("❌ Login form did not appear after click — aborting login attempt")
        return

    await page.click("#username")
    await page.keyboard.press("Control+A")
    await page.keyboard.press("Backspace")
    await page.keyboard.type(RR_USERNAME, delay=60)

    await page.click("#password")
    await page.keyboard.press("Control+A")
    await page.keyboard.press("Backspace")
    await page.keyboard.type(RR_PASSWORD, delay=60)

    await page.get_by_role("button", name=re.compile(r"log in", re.I)).click()
    await page.wait_for_selector("button:has-text('Deposit')", timeout=15000)
    logger.info("✅ Login successful")
    await clear_blocking_overlays(page)


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
