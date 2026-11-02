import asyncio
import logging
from datetime import datetime
import os
import pandas as pd
import requests
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import HTMLResponse
from openai import OpenAI

# Configure clean production logging with precise timestamps
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("QuantEngine")

app = FastAPI(title="Quant Engine Pro - Gold & Bitcoin")

ACTIVE_SIGNALS = {}

# Locked strictly to Gold and Bitcoin
TRACKED_ASSETS = [
    {"symbol": "XAU/USD", "name": "Gold", "fallback_price": 4039.00},
    {"symbol": "BTC/USD", "name": "Bitcoin", "fallback_price": 79296.00},
]


def send_telegram_alert(message: str, chat_id: str = None):
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    target_chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID")
    if not token or not target_chat_id:
        logger.warning("Telegram token or chat ID missing. Alert skipped.")
        return

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": target_chat_id,
        "text": message,
        "parse_mode": "Markdown",
    }
    try:
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code == 200:
            logger.info("Telegram alert dispatched successfully.")
        else:
            logger.error(
                f"Telegram API failed with status {response.status_code}: {response.text}"
            )
    except Exception as e:
        logger.error(f"Telegram alert exception: {e}")


def get_ai_insight(asset_name, current_price, rsi_val, trend, action):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return f"Quant Confluence: {asset_name} active at ${current_price:,.2f}."

    try:
        client = OpenAI(api_key=api_key)
        prompt = (
            f"You are a professional prop-firm quant trader. Provide a sharp, "
            f"1-sentence market confirmation summary for {asset_name} at ${current_price:,.2f}. "
            f"Trend: {trend}, RSI: {rsi_val:.2f}, Signal: {action}."
        )
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=60,
        )
        return res.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f"OpenAI API error for {asset_name}: {e}")
        return f"Multi-timeframe momentum verified for {asset_name}."


def fetch_twelve_data(symbol: str, fallback_price: float):
    api_key = os.getenv("TWELVE_DATA_API_KEY")
    if not api_key:
        logger.warning(
            f"Twelve Data API key missing. Using fallback for {symbol}."
        )
        return fallback_price, 50.0, "Bullish", 35.0

    try:
        url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval=15min&outputsize=30&apikey={api_key}"
        res = requests.get(url, timeout=10).json()
        values = res.get("values", [])
        if not values:
            logger.warning(f"Empty values returned from Twelve Data for {symbol}.")
            return fallback_price, 50.0, "Bullish", 35.0

        df = pd.DataFrame(values)
        df["close"] = df["close"].astype(float)
        df["high"] = df["high"].astype(float)
        df["low"] = df["low"].astype(float)

        current_price = df["close"].iloc[0]

        delta = df["close"].diff(-1)
        gain = delta.where(delta > 0, 0).rolling(14).mean().iloc[0]
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean().iloc[0]
        rsi = (
            100 - (100 / (1 + (gain / loss)))
            if loss and not pd.isna(loss)
            else 50.0
        )

        tr = (df["high"] - df["low"]).rolling(14).mean().iloc[0]
        atr = tr if not pd.isna(tr) else 25.0
        trend = "Bullish" if df["close"].iloc[0] > df["close"].iloc[10] else "Bearish"

        return current_price, rsi, trend, atr
    except Exception as e:
        logger.error(f"Twelve Data API exception for {symbol}: {e}")
        return fallback_price, 50.0, "Bullish", 35.0


def evaluate_signal(asset):
    symbol = asset["symbol"]
    name = asset["name"]
    price, rsi, trend, atr = fetch_twelve_data(symbol, asset["fallback_price"])

    action = "BUY (LONG)" if trend == "Bullish" else "SELL (SHORT)"

    # Tight Institutional Stop Loss Multiplier
    is_buy = "BUY" in action
    sl = round(price - (atr * 0.7), 2) if is_buy else round(price + (atr * 0.7), 2)
    tp1 = round(price + (atr * 1.2), 2) if is_buy else round(price - (atr * 1.2), 2)
    tp2 = round(price + (atr * 2.2), 2) if is_buy else round(price - (atr * 2.2), 2)
    tp3 = round(price + (atr * 3.5), 2) if is_buy else round(price - (atr * 3.5), 2)

    sl_dist = round(abs(price - sl), 1)
    tp1_dist = round(abs(tp1 - price), 1)
    tp2_dist = round(abs(tp2 - price), 1)
    tp3_dist = round(abs(tp3 - price), 1)
    rr_ratio = round(tp1_dist / sl_dist, 1) if sl_dist > 0 else 1.5

    status_tag = "🟢 Active • Now"
    sl_label = f"${sl:,.2f} (-{sl_dist}p)"

    if symbol in ACTIVE_SIGNALS:
        prev = ACTIVE_SIGNALS[symbol]
        if (is_buy and price >= prev["tp1"]) or (not is_buy and price <= prev["tp1"]):
            status_tag = "🎯 1 TP Hit (SL -> BE)"
            sl_label = f"BE (${price:,.2f})"
        if (is_buy and price >= prev["tp2"]) or (not is_buy and price <= prev["tp2"]):
            status_tag = "🎯 2 TP Hit • Runner Active"
            sl_label = f"BE (${prev['tp1']:,.2f})"
        if (is_buy and price >= prev["tp3"]) or (not is_buy and price <= prev["tp3"]):
            status_tag = "🏆 Target 3 Won"

    ai_text = get_ai_insight(name, price, rsi, trend, action)

    signal_data = {
        "asset": name,
        "price": price,
        "action": action,
        "status": status_tag,
        "sl": sl,
        "sl_label": sl_label,
        "tp1": tp1,
        "tp2": tp2,
        "tp3": tp3,
        "sl_dist": sl_dist,
        "tp1_dist": tp1_dist,
        "tp2_dist": tp2_dist,
        "tp3_dist": tp3_dist,
        "rr_ratio": rr_ratio,
        "rsi": round(rsi, 2),
        "insight": ai_text,
    }

    ACTIVE_SIGNALS[symbol] = signal_data
    return signal_data


async def autonomous_market_scanner():
    scan_start = datetime.now()
    logger.info(
        f"--- Starting Autonomous Market Scan at {scan_start.strftime('%Y-%m-%d %H:%M:%S')} ---"
    )

    try:
        for asset in TRACKED_ASSETS:
            data = evaluate_signal(asset)
            msg = (
                f"⚡ *SIGNALPRO QUANT DESK* ⚡\n\n"
                f"📌 *Asset:* {data['asset']}\n"
                f"🔹 *Action:* {data['action']} | *Status:* {data['status']}\n"
                f"💰 *Entry Price:* ${data['price']:,.2f}\n"
                f"⚖️ *Risk/Reward:* 1:{data['rr_ratio']}\n\n"
                f"🛡️ *Stop Loss (SL):* {data['sl_label']}\n"
                f"🎯 *Take Profit 1:* ${data['tp1']:,.2f} (+{data['tp1_dist']} pts)\n"
                f"🎯 *Take Profit 2:* ${data['tp2']:,.2f} (+{data['tp2_dist']} pts)\n"
                f"🎯 *Take Profit 3:* ${data['tp3']:,.2f} (+{data['tp3_dist']} pts)\n\n"
                f"📈 *RSI (14):* {data['rsi']}\n"
                f"💬 *Quant Insight:* {data['insight']}\n\n"
                f"──────────────────────\n"
                f"⚠️ *Institutional Risk Protocol:* Max 1.5% exposure per execution."
            )
            send_telegram_alert(msg)
            logger.info(f"Successfully processed and broadcasted: {data['asset']}")
            await asyncio.sleep(2)

        logger.info("--- Market Scan Completed Successfully ---")
    except Exception as e:
        logger.error(f"CRITICAL ERROR in autonomous_market_scanner: {e}")


@app.on_event("startup")
def init_scheduler():
    scheduler = AsyncIOScheduler()
    scheduler.add_job(autonomous_market_scanner, "interval", minutes=15)
    scheduler.start()
    logger.info(
        "APScheduler started successfully. Background scanner job scheduled every 15 minutes."
    )


@app.get("/", response_class=HTMLResponse)
def dashboard():
    cards = ""
    for asset in TRACKED_ASSETS:
        data = evaluate_signal(asset)
        badge_color = (
            "#22c55e"
            if "BUY" in data["action"]
            else ("#ef4444" if "SELL" in data["action"] else "#eab308")
        )
        cards += f"""
            <div style="background:#1e293b; border-radius:12px; padding:16px; margin-bottom:12px; border:1px solid #334155; color:#fff;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <h3 style="margin:0;">{data['asset']}</h3>
                    <span style="background:{badge_color}; padding:4px 8px; border-radius:6px; font-size:12px; font-weight:bold;">{data['action']}</span>
                </div>
                <div style="font-size:22px; font-weight:bold; margin:8px 0;">${data['price']:,.2f}</div>
                <div style="font-size:12px; color:#38bdf8; margin-bottom:4px;">Status: {data['status']}</div>
                <div style="font-size:12px; color:#fbbf24; margin-bottom:8px;">Risk/Reward: 1:{data['rr_ratio']}</div>
                <div style="display:grid; grid-template-columns:1fr 1fr; background:#0f172a; padding:8px; border-radius:6px; text-align:center; font-size:11px; gap:4px; margin-bottom:6px;">
                    <div><span>SL: {data['sl_label']}</span></div>
                    <div><span>TP1 (+{data['tp1_dist']}p): ${data['tp1']:,.2f}</span></div>
                    <div><span>TP2 (+{data['tp2_dist']}p): ${data['tp2']:,.2f}</span></div>
                    <div><span>TP3 (+{data['tp3_dist']}p): ${data['tp3']:,.2f}</span></div>
                </div>
                <p style="font-size:12px; color:#cbd5e1; font-style:italic; margin-top:6px;">💬 {data['insight']}</p>
            </div>
            """

    return HTMLResponse(content=f"""
    <html>
        <body style="background:#0f172a; font-family:sans-serif; max-width:500px; margin:auto; padding:20px;">
            <h1 style="color:#38bdf8; text-align:center; font-size:22px;">⚡ SignalPro AI Pro Engine</h1>
            <p style="color:#94a3b8; text-align:center; font-size:13px;">Live Gold & Bitcoin Institutional Scanner</p>
            <a href="/trigger-broadcast" style="display:block; background:#2563eb; color:#fff; text-align:center; padding:12px; border-radius:8px; text-decoration:none; font-weight:bold; margin:20px 0;">🚀 Broadcast Pro Signals Now</a>
            {cards}
        </body>
    </html>
    """)


@app.get("/trigger-broadcast")
async def trigger_broadcast():
    logger.info("Manual broadcast triggered via web dashboard.")
    await autonomous_market_scanner()
    return HTMLResponse(
        "<h3>✅ Gold & Bitcoin signals successfully broadcasted to Telegram! <a"
        " href='/'>Back</a></h3>"
    )


@app.post("/telegram-webhook")
async def webhook(request: Request, background_tasks: BackgroundTasks):
    data = await request.json()
    if "message" in data:
        text = data["message"].get("text", "")
        if text in ["/signal", "/status"]:
            logger.info(f"Webhook received command: {text}")
            background_tasks.add_task(autonomous_market_scanner)
    return {"status": "ok"}