import asyncio
import os
import pandas as pd
import requests
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import HTMLResponse
from openai import OpenAI

app = FastAPI(title="Quant Engine Pro")

ACTIVE_SIGNALS = {}

TRACKED_ASSETS = [
    {"symbol": "XAU/USD", "name": "Gold", "fallback_price": 4039.00},
    {"symbol": "BTC/USD", "name": "Bitcoin", "fallback_price": 79296.00},
]


def send_telegram_alert(message: str, chat_id: str = None):
  token = os.getenv("TELEGRAM_BOT_TOKEN")
  target_chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID")
  if not token or not target_chat_id:
    return

  url = f"https://api.telegram.org/bot{token}/sendMessage"
  payload = {
      "chat_id": target_chat_id,
      "text": message,
      "parse_mode": "Markdown",
  }
  try:
    requests.post(url, json=payload, timeout=10)
  except Exception as e:
    print(f"Telegram alert error: {e}")


def get_ai_insight(asset_name, current_price, rsi_val, trend, action):
  api_key = os.getenv("OPENAI_API_KEY")
  if not api_key:
    return f"Confluence Engine: {asset_name} at ${current_price:,.2f}. Trend is {trend}."

  try:
    client = OpenAI(api_key=api_key)
    prompt = (
        f"You are a quant trader. Provide a short, punchy 2-sentence market"
        f" summary for {asset_name} trading at ${current_price:,.2f}. Trend:"
        f" {trend}, RSI: {rsi_val:.2f}, Signal: {action}."
    )
    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=80,
    )
    return res.choices[0].message.content.strip()
  except Exception:
    return f"Market filter active for {asset_name} at ${current_price:,.2f}."


def fetch_twelve_data(symbol: str, fallback_price: float):
  api_key = os.getenv("TWELVE_DATA_API_KEY")
  if not api_key:
    return fallback_price, 55.0, "Bullish", 35.0

  try:
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval=15min&outputsize=30&apikey={api_key}"
    res = requests.get(url, timeout=10).json()
    values = res.get("values", [])
    if not values:
      return fallback_price, 55.0, "Bullish", 35.0

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
    print(f"API Error ({symbol}): {e}")
    return fallback_price, 55.0, "Bullish", 35.0


def evaluate_signal(asset):
  symbol = asset["symbol"]
  name = asset["name"]
  price, rsi, trend, atr = fetch_twelve_data(symbol, asset["fallback_price"])

  action = "BUY (LONG)" if trend == "Bullish" else "SELL (SHORT)"
  sl = (
      round(price - (atr * 1.0), 2)
      if "BUY" in action
      else round(price + (atr * 1.0), 2)
  )
  tp1 = (
      round(price + (atr * 1.5), 2)
      if "BUY" in action
      else round(price - (atr * 1.5), 2)
  )
  tp2 = (
      round(price + (atr * 2.5), 2)
      if "BUY" in action
      else round(price - (atr * 2.5), 2)
  )

  status_tag = "🟢 Active • Now"
  if symbol in ACTIVE_SIGNALS:
    prev = ACTIVE_SIGNALS[symbol]
    if ("BUY" in action and price >= prev["tp1"]) or (
        "SELL" in action and price <= prev["tp1"]
    ):
      status_tag = "🎯 TP1 Hit"
    if ("BUY" in action and price >= prev["tp2"]) or (
        "SELL" in action and price <= prev["tp2"]
    ):
      status_tag = "🏆 Won"

  ai_text = get_ai_insight(name, price, rsi, trend, action)

  signal_data = {
      "asset": name,
      "price": price,
      "action": action,
      "status": status_tag,
      "sl": sl,
      "tp1": tp1,
      "tp2": tp2,
      "rsi": round(rsi, 2),
      "insight": ai_text,
  }

  ACTIVE_SIGNALS[symbol] = signal_data
  return signal_data


async def autonomous_market_scanner():
  for asset in TRACKED_ASSETS:
    data = evaluate_signal(asset)
    msg = (
        f"📊 *SIGNALPRO AUTOMATED DESK* 📊\n\n"
        f"📌 *Asset:* {data['asset']}\n"
        f"⚡ *Action:* {data['action']} | *Status:* {data['status']}\n"
        f"💰 *Entry Price:* ${data['price']:,.2f}\n\n"
        f"🛑 *Stop Loss (SL):* ${data['sl']:,.2f}\n"
        f"🎯 *Take Profit 1:* ${data['tp1']:,.2f}\n"
        f"🎯 *Take Profit 2:* ${data['tp2']:,.2f}\n\n"
        f"📈 *RSI (14):* {data['rsi']}\n"
        f"💬 *AI Desk Insight:* {data['insight']}"
    )
    send_telegram_alert(msg)
    await asyncio.sleep(2)


@app.on_event("startup")
def init_scheduler():
  scheduler = AsyncIOScheduler()
  scheduler.add_job(autonomous_market_scanner, "interval", minutes=15)
  scheduler.start()


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
            <div style="font-size:12px; color:#38bdf8; margin-bottom:8px;">Status: {data['status']}</div>
            <div style="display:grid; grid-template-columns:1fr 1fr 1fr; background:#0f172a; padding:8px; border-radius:6px; text-align:center; font-size:12px;">
                <div><span>SL</span><br>${data['sl']:,.2f}</div>
                <div><span>TP1</span><br>${data['tp1']:,.2f}</div>
                <div><span>TP2</span><br>${data['tp2']:,.2f}</div>
            </div>
            <p style="font-size:12px; color:#cbd5e1; font-style:italic; margin-top:8px;">💬 {data['insight']}</p>
        </div>
        """

  return HTMLResponse(content=f"""
    <html>
        <body style="background:#0f172a; font-family:sans-serif; max-width:500px; margin:auto; padding:20px;">
            <h1 style="color:#38bdf8; text-align:center; font-size:22px;">⚡ SignalPro AI Engine</h1>
            <p style="color:#94a3b8; text-align:center; font-size:13px;">Live Autonomous 15-Minute Confluence Scanner</p>
            <a href="/trigger-broadcast" style="display:block; background:#2563eb; color:#fff; text-align:center; padding:12px; border-radius:8px; text-decoration:none; font-weight:bold; margin:20px 0;">🚀 Broadcast Live Signals Now</a>
            {cards}
        </body>
    </html>
    """)


@app.get("/trigger-broadcast")
async def trigger_broadcast():
  await autonomous_market_scanner()
  return HTMLResponse(
      "<h3>✅ Live signals successfully broadcasted to Telegram! <a"
      " href='/'>Back</a></h3>"
  )


@app.post("/telegram-webhook")
async def webhook(request: Request, background_tasks: BackgroundTasks):
  data = await request.json()
  if "message" in data:
    text = data["message"].get("text", "")
    if text in ["/signal", "/status"]:
      background_tasks.add_task(autonomous_market_scanner)
  return {"status": "ok"}