#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Telegram message handling for the Rainbow Riches bot.
Responsible for:
  - Telegram client setup and authentication
  - Incoming message filtering and queuing
  - Tip parsing
  - Worker loop that dispatches parsed tips to the betting handler
"""

import asyncio
import os
import re
import traceback
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from telethon import TelegramClient, events

load_dotenv(Path(__file__).parent / ".env")

# -----------------------------
# Telegram credentials
# -----------------------------
API_ID    = int(os.environ["TELEGRAM_API_ID"])
API_HASH  = os.environ["TELEGRAM_API_HASH"]
BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]

SESSION_NAME = os.getenv("TELEGRAM_SESSION_NAME", "Rainbow_session")

# Optional: only act on messages containing these brand names; [] means accept all
BRAND_FILTERS: List[str] = []


# -----------------------------
# Tip parser
# -----------------------------
def parse_telegram_message(text: str) -> Optional[dict]:
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


# -----------------------------
# Telegram client + worker setup
# -----------------------------
async def setup_telegram(queue: asyncio.Queue, processing_lock: asyncio.Lock, handle_tip_fn, pageA, pageB):
    """
    Creates and starts the Telegram client, registers the message handler,
    and launches the worker task that dispatches tips to handle_tip_fn.

    Returns:
        (client, worker_task) — caller is responsible for cleanup.
    """
    client = TelegramClient(SESSION_NAME, API_ID, API_HASH)
    await client.start(bot_token=BOT_TOKEN)
    print("🤖 Listening for Telegram tips... (Ctrl+C to stop)")

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
                    tip = parse_telegram_message(tip_text)
                    if not tip:
                        continue
                    await handle_tip_fn(tip, pageA, pageB)
            except Exception as e:
                print(f"⚠️ handle_tip error: {e}")
                traceback.print_exc()
            finally:
                queue.task_done()

    worker_task = asyncio.create_task(worker())
    return client, worker_task

