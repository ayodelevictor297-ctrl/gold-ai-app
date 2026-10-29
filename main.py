import asyncio
import os
import pandas as pd
import requests
import yfinance as yf
from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import HTMLResponse
from openai import OpenAI

app = FastAPI(title="Quant Trading Engine")

TRACKED_ASSETS = [
    {"symbol": "GC=F", "name": "Gold"},
    {"symbol": "BTC-USD", "name": "Bitcoin"},
]


def send_telegram_alert(message: str, chat_id: str = None):
  token = os.getenv("TELEGRAM_BOT_TOKEN")
  target_chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID")

  if not token or not target_chat_id:
    print("Error: Missing Telegram token or chat ID.")
    return

  url = f"https://api.telegram.org/bot{token}/sendMessage"
  payload = {
      "chat_id": target_chat_id,
      "text": message,
      "parse_mode": "Markdown",
  }
  try:
    res = requests.post(url, json=payload, timeout=10)
    print("Telegram Response:", res.json())
  except Exception as e:
    print("Failed to send Telegram message:", e)


def get_ai_insight(
    asset_name, current_price, rsi_val, daily_trend, m15_trend, action
):
  api_key = os.getenv("OPENAI_API_KEY")
  if not api_key:
    return f"Institutional Confluence Engine: {asset_name} is trading at ${current_price:,.2f} with Daily trend as {daily_trend} and 15M trend as {m15_trend}."

  try:
    client = OpenAI(api_key=api_key)
    prompt = (
        f"You are an elite quantitative trading AI assistant. Write a short,"
        f" punchy, professional 2-sentence market insight for {asset_name}"
        f" trading at ${current_price:,.2f}. The Daily macro trend is"
        f" {daily_trend}, the 15M tactical trend is {m15_trend}, RSI is"
        f" {rsi_val:.2f}, and the validated signal action is {action}."
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100,
    )
    return response.choices[0].message.content.strip()
  except Exception as e:
    return f"Engine Filter Active: Daily is {daily_trend}, 15M is {m15_trend}. {asset_name} at ${current_price:,.2f}."


def generate_quant_signal(ticker: str, asset_name: str):
  try:
    df_15m = yf.download(ticker, period="5d", interval="15m", progress=False)
    if df_15m is None or df_15m.empty:
      return None

    if isinstance(df_15m.columns, pd.MultiIndex):
      df_15m.columns = df_15m.columns.get_level_values(0)

    df_1d = yf.download(ticker, period="6mo", interval="1d", progress=False)
    if df_1d is None or df_1d.empty:
      return None

    if isinstance(df_1d.columns, pd.MultiIndex):
      df_1d.columns = df_1d.columns.get_level_values(0)

    close_15m = df_15m["Close"].astype(float)
    high_15m = df_15m["High"].astype(float)
    low_15m = df_15m["Low"].astype(float)

    delta = close_15m.diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df_15m["RSI"] = 100 - (100 / (1 + rs))

    tr1 = high_15m - low_15m
    tr2 = (high_15m - close_15m.shift()).abs()
    tr3 = (low_15m - close_15m.shift()).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df_15m["ATR"] = true_range.rolling(window=14).mean()

    df_15m["EMA_20"] = close_15m.ewm(span=20, adjust=False).mean()
    df_15m["EMA_50"] = close_15m.ewm(span=50, adjust=False).mean()

    close_1d = df_1d["Close"].astype(float)
    df_1d["EMA_20"] = close_1d.ewm(span=20, adjust=False).mean()
    df_1d["EMA_50"] = close_1d.ewm(span=50, adjust=False).mean()

    latest_15m = df_15m.iloc[-1]
    latest_1d = df_1d.iloc[-1]

    current_price = float(latest_15m["Close"])
    rsi_val = (
        float(latest_15m["RSI"]) if not pd.isna(latest_15m["RSI"]) else 50.0
    )
    atr_val = (
        float(latest_15m["ATR"]) if not pd.isna(latest_15m["ATR"]) else 15.0
    )

    daily_trend = (
        "Bullish"
        if float(latest_1d["EMA_20"]) > float(latest_1d["EMA_50"])
        else "Bearish"
    )
    m15_trend = (
        "Bullish"
        if float(latest_15m["EMA_20"]) > float(latest_15m["EMA_50"])
        else "Bearish"
    )

    if daily_trend == "Bullish" and m15_trend == "Bullish":
      trend = "Bullish (Multi-TF Confirmed)"
      signal = "BUY (LONG)"
      stop_loss = round(current_price - (atr_val * 1.0), 2)
      tp_1 = round(current_price + (atr_val * 1.5), 2)
      tp_2 = round(current_price + (atr_val * 2.5), 2)
    elif daily_trend == "Bearish" and m15_trend == "Bearish":
      trend = "Bearish (Multi-TF Confirmed)"
      signal = "SELL (SHORT)"
      stop_loss = round(current_price + (atr_val * 1.0), 2)
      tp_1 = round(current_price - (atr_val * 1.5), 2)
      tp_2 = round(current_price - (atr_val * 2.5), 2)
    else:
      trend = f"Divergence (Daily: {daily_trend} / 15M: {m15_trend})"
      signal = "NO TRADE (Market Divergence)"
      stop_loss, tp_1, tp_2 = 0, 0, 0

    rsi_condition = (
        "Overbought"
        if rsi_val > 70
        else ("Oversold" if rsi_val < 30 else "Neutral")
    )

    ai_commentary = get_ai_insight(
        asset_name, current_price, rsi_val, daily_trend, m15_trend, signal
    )

    return {
        "asset": asset_name,
        "current_price": round(current_price, 2),
        "ai_market_insight": ai_commentary,
        "signal_setup": {
            "action": signal,
            "market_bias": trend,
            "stop_loss_sl": stop_loss,
            "take_profit_1": tp_1,
            "take_profit_2": tp_2,
        },
        "technical_indicators": {
            "rsi_14": round(rsi_val, 2),
            "rsi_condition": rsi_condition,
            "atr_volatility": round(atr_val, 2),
        },
    }
  except Exception as e:
    print(f"Error generating signal for {ticker}: {e}")
    return None


@app.get("/", response_class=HTMLResponse)
def web_dashboard():
  try:
    signals_html = ""
    for asset in TRACKED_ASSETS:
      data = generate_quant_signal(asset["symbol"], asset["name"])
      if not data:
        signals_html += f"""
                <div class="card">
                    <div class="card-header">
                        <h2>{asset['name']}</h2>
                        <span class="badge" style="background: #eab308;">SYNCING</span>
                    </div>
                    <p>🔄 Fetching latest market data...</p>
                </div>
                """
        continue
      setup = data["signal_setup"]
      badge_color = (
          "#22c55e"
          if "BUY" in setup["action"]
          else ("#ef4444" if "SELL" in setup["action"] else "#eab308")
      )

      signals_html += f"""
            <div class="card">
                <div class="card-header">
                    <h2>{data['asset']}</h2>
                    <span class="badge" style="background: {badge_color};">{setup['action']}</span>
                </div>
                <div class="price">${data['current_price']:,.2f}</div>
                <p><strong>Bias:</strong> {setup['market_bias']}</p>
                <div class="levels">
                    <div><span>SL:</span> ${setup['signal_setup']['stop_loss_sl']:,.2f}</div>
                    <div><span>TP1:</span> ${setup['signal_setup']['take_profit_1']:,.2f}</div>
                    <div><span>TP2:</span> ${setup['signal_setup']['take_profit_2']:,.2f}</div>
                </div>
                <p class="insight">💬 {data['ai_market_insight']}</p>
            </div>
            """
  except Exception as e:
    signals_html = f"""
        <div class="card">
            <h2>⚠️ Notice</h2>
            <p>Market engine initializing: {str(e)}</p>
        </div>
        """

  html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Institutional Quant Engine</title>
        <style>
            body {{ background-color: #0f172a; color: #f8fafc; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 0; padding: 20px; }}
            .container {{ max-width: 600px; margin: 0 auto; }}
            h1 {{ text-align: center; color: #38bdf8; font-size: 24px; margin-bottom: 5px; }}
            p.subtitle {{ text-align: center; color: #94a3b8; margin-top: 0; font-size: 14px; }}
            .btn {{ display: block; width: 100%; background: #2563eb; color: white; border: none; padding: 12px; font-size: 16px; font-weight: bold; border-radius: 8px; cursor: pointer; text-align: center; text-decoration: none; margin: 20px 0; transition: background 0.2s; }}
            .btn:hover {{ background: #1d4ed8; }}
            .card {{ background: #1e293b; border-radius: 12px; padding: 16px; margin-bottom: 16px; border: 1px solid #334155; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); }}
            .card-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }}
            .card-header h2 {{ margin: 0; font-size: 18px; }}
            .badge {{ padding: 4px 10px; border-radius: 6px; font-size: 12px; font-weight: bold; color: white; }}
            .price {{ font-size: 22px; font-weight: bold; color: #f1f5f9; margin-bottom: 8px; }}
            .levels {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; background: #0f172a; padding: 10px; border-radius: 8px; margin: 10px 0; font-size: 13px; text-align: center; }}
            .levels span {{ display: block; color: #94a3b8; font-size: 11px; }}
            .insight {{ font-size: 13px; color: #cbd5e1; font-style: italic; margin-top: 10px; border-top: 1px solid #334155; padding-top: 8px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>⚡ Institutional Quant AI Desk</h1>
            <p class="subtitle">Live Multi-Timeframe Confluence Engine for Gold & Bitcoin</p>
            <a href="/trigger-broadcast" class="btn">🚀 Broadcast Signals to Telegram Now</a>
            {signals_html}
        </div>
    </body>
    </html>
    """
  return HTMLResponse(content=html_content)


@app.get("/trigger-broadcast")
def trigger_broadcast():
  try:
    for asset in TRACKED_ASSETS:
      signal_data = generate_quant_signal(asset["symbol"], asset["name"])
      if not signal_data:
        continue
      tech = signal_data["technical_indicators"]
      setup = signal_data["signal_setup"]

      signal_message = (
          f"📊 *LIVE BROWSER BROADCAST: {asset['name'].upper()}* 📊\n\n"
          f"📌 *Action:* {setup['action']}\n"
          f"📊 *Market Bias:* {setup['market_bias']}\n"
          f"💰 *Entry Price:* ${signal_data['current_price']:,.2f}\n\n"
          f"🛑 *Stop Loss (SL):* ${setup['signal_setup']['stop_loss_sl']:,.2f}\n"
          f"🎯 *Take Profit 1:* ${setup['signal_setup']['take_profit_1']:,.2f}\n"
          f"🎯 *Take Profit 2:* ${setup['signal_setup']['take_profit_2']:,.2f}\n\n"
          f"📈 *RSI (14):* {tech['rsi_14']} ({tech['rsi_condition']})\n"
          f"⚡ *ATR Volatility:* {tech['atr_volatility']:,.2f}\n\n"
          f"💬 *AI Desk Insight:* {signal_data['ai_market_insight']}"
      )
      send_telegram_alert(signal_message)
    return HTMLResponse(
        content="<h3>✅ Signals successfully broadcasted to your Telegram! <a"
        " href='/'>Back to Dashboard</a></h3>"
    )
  except Exception as e:
    return HTMLResponse(content=f"<h3>❌ Error: {str(e)}</h3>")


async def process_signal_background(chat_id: str):
  try:
    for asset in TRACKED_ASSETS:
      signal_data = generate_quant_signal(asset["symbol"], asset["name"])
      if not signal_data:
        continue

      tech = signal_data["technical_indicators"]
      setup = signal_data["signal_setup"]

      signal_message = (
          f"📊 *ON-DEMAND {asset['name'].upper()} SIGNAL* 📊\n\n"
          f"📌 *Action:* {setup['action']}\n"
          f"📊 *Market Bias:* {setup['market_bias']}\n"
          f"💰 *Entry Price:* ${signal_data['current_price']:,.2f}\n\n"
          f"🛑 *Stop Loss (SL):* ${setup['signal_setup']['stop_loss_sl']:,.2f}\n"
          f"🎯 *Take Profit 1:* ${setup['signal_setup']['take_profit_1']:,.2f}\n"
          f"🎯 *Take Profit 2:* ${setup['signal_setup']['take_profit_2']:,.2f}\n\n"
          f"📈 *RSI (14):* {tech['rsi_14']} ({tech['rsi_condition']})\n"
          f"⚡ *ATR Volatility:* {tech['atr_volatility']:,.2f}\n\n"
          f"💬 *AI Desk Insight:* {signal_data['ai_market_insight']}"
      )
      send_telegram_alert(signal_message, chat_id=chat_id)
      await asyncio.sleep(1)
  except Exception as e:
    send_telegram_alert(
        f"⚠️ Background Processing Error: {str(e)}", chat_id=chat_id
    )


@app.post("/telegram-webhook")
async def telegram_webhook(request: Request, background_tasks: BackgroundTasks):
  try:
    data = await request.json()
    if "message" in data:
      message = data["message"]
      text = message.get("text", "").strip()
      incoming_chat_id = str(message["chat"]["id"])

      if text == "/signal":
        send_telegram_alert(
            "⏳ *Generating live quant signals for Gold & Bitcoin...*",
            chat_id=incoming_chat_id,
        )
        background_tasks.add_task(
            process_signal_background, chat_id=incoming_chat_id
        )

      elif text == "/status":
        status_message = (
            "🟢 *QUANT ENGINE STATUS: ONLINE*\n\n"
            "🚀 *Platform:* Render Cloud Infrastructure\n"
            "⏱ *Polling Interval:* Webhooks & Dashboard Active\n"
            "📈 *Tracked Assets:* Gold (XAU/USD) & Bitcoin (BTC/USD)\n"
            "🤖 *Webhooks & AI Desk:* Active & Responsive"
        )
        send_telegram_alert(status_message, chat_id=incoming_chat_id)
  except Exception as e:
    print(f"WEBHOOK ERROR: {e}")

  return {"status": "ok"}