import asyncio
import os
import pandas as pd
import requests
import yfinance
from fastapi import FastAPI, HTTPException, Request
from openai import OpenAI

app = FastAPI()

TRACKED_ASSETS = [
    {"symbol": "GC=F", "name": "Gold"},
    {"symbol": "BTC-USD", "name": "Bitcoin"},
]


def send_telegram_alert(message: str):
  TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
  TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
  url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
  payload = {
      "chat_id": TELEGRAM_CHAT_ID,
      "text": message,
      "parse_mode": "Markdown",
  }
  response = requests.post(url, json=payload)
  return response.json()


def get_ai_insight(
    asset_name, current_price, rsi_val, daily_trend, m15_trend, action
):
  api_key = os.getenv("OPENAI_API_KEY")
  if not api_key:
    return f"Institutional Confluence Engine: {asset_name} is trading at ${current_price:,.2f} with Daily trend as {daily_trend} and 15M tactical trend as {m15_trend}. Action locked: {action}."

  try:
    client = OpenAI(api_key=api_key)
    prompt = (
        f"You are an elite quantitative trading AI assistant. Write a short,"
        f" punchy, professional 2-sentence market insight for {asset_name}"
        f" trading at ${current_price:,.2f}. The Daily macro trend is"
        f" {daily_trend}, the 15M tactical trend is {m15_trend}, RSI is"
        f" {rsi_val:.2f}, and the validated signal action is {action}."
        f" Speak like a seasoned head quant desk analyst."
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
    # 1. Fetch 15-Minute Tactical Data
    df_15m = yfinance.download(ticker, period="5d", interval="15m", progress=False)
    if df_15m.empty:
      raise HTTPException(
          status_code=404, detail=f"No 15M market data returned for {ticker}."
      )

    if isinstance(df_15m.columns, pd.MultiIndex):
      df_15m.columns = df_15m.columns.get_level_values(0)

    # 2. Fetch Daily Macro Data
    df_1d = yfinance.download(ticker, period="6mo", interval="1d", progress=False)
    if isinstance(df_1d.columns, pd.MultiIndex):
      df_1d.columns = df_1d.columns.get_level_values(0)

    close_15m = df_15m["Close"].astype(float)
    high_15m = df_15m["High"].astype(float)
    low_15m = df_15m["Low"].astype(float)

    # Technical Indicators (RSI & ATR)
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

    # STRICT CONFLUENCE FILTER: Only trigger if Daily and 15M match perfectly
    if daily_trend == "Bullish" and m15_trend == "Bullish":
      trend = "Bullish (Multi-TF Confirmed)"
      signal = "BUY (LONG)"
      stop_loss = round(current_price - (atr_val * 1.0), 2)
      tp_1 = round(current_price + (atr_val * 1.5), 2)
      tp_2 = round(current_price + (atr_val * 2.5), 2)
      valid_signal = True
    elif daily_trend == "Bearish" and m15_trend == "Bearish":
      trend = "Bearish (Multi-TF Confirmed)"
      signal = "SELL (SHORT)"
      stop_loss = round(current_price + (atr_val * 1.0), 2)
      tp_1 = round(current_price - (atr_val * 1.5), 2)
      tp_2 = round(current_price - (atr_val * 2.5), 2)
      valid_signal = True
    else:
      trend = f"Divergence (Daily: {daily_trend} / 15M: {m15_trend})"
      signal = "NO TRADE (Market Divergence)"
      valid_signal = False
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
        "valid_signal": valid_signal,
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
    raise HTTPException(status_code=500, detail=str(e))


async def autonomous_signal_scheduler():
  await asyncio.sleep(10)
  while True:
    try:
      for asset in TRACKED_ASSETS:
        signal_data = generate_quant_signal(asset["symbol"], asset["name"])

        # ONLY send alert if multi-timeframe trend is strictly confirmed
        if signal_data["valid_signal"]:
          tech = signal_data["technical_indicators"]
          setup = signal_data["signal_setup"]

          message = (
              f"🤖 *AUTONOMOUS {asset['name'].upper()} ALERT* 🤖\n\n"
              f"📌 *Action:* {setup['action']}\n"
              f"📊 *Market Bias:* {setup['market_bias']}\n"
              f"💰 *Entry Price:* ${signal_data['current_price']:,.2f}\n\n"
              f"🛑 *Stop Loss (SL):* ${setup['stop_loss_sl']:,.2f}\n"
              f"🎯 *Take Profit 1:* ${setup['take_profit_1']:,.2f}\n"
              f"🎯 *Take Profit 2:* ${setup['take_profit_2']:,.2f}\n\n"
              f"📈 *RSI (14):* {tech['rsi_14']} ({tech['rsi_condition']})\n"
              f"⚡ *ATR Volatility:* {tech['atr_volatility']:,.2f}\n\n"
              f"💬 *AI Desk Insight:* {signal_data['ai_market_insight']}"
          )
          send_telegram_alert(message)
          await asyncio.sleep(3)

    except Exception as e:
      print(f"AUTONOMOUS ENGINE ERROR: {e}")

    # Sleep for 3 hours (10800 seconds) between sweeps
    await asyncio.sleep(10800)


@app.on_event("startup")
async def startup_event():
  asyncio.create_task(autonomous_signal_scheduler())


@app.post("/telegram-webhook")
async def telegram_webhook(request: Request):
  try:
    data = await request.json()
    if "message" in data:
      message = data["message"]
      text = message.get("text", "").strip()

      if text == "/signal":
        for asset in TRACKED_ASSETS:
          try:
            signal_data = generate_quant_signal(asset["symbol"], asset["name"])
            tech = signal_data["technical_indicators"]
            setup = signal_data["signal_setup"]

            signal_message = (
                f"📊 *ON-DEMAND {asset['name'].upper()} SIGNAL* 📊\n\n"
                f"📌 *Action:* {setup['action']}\n"
                f"📊 *Market Bias:* {setup['market_bias']}\n"
                f"💰 *Entry Price:* ${signal_data['current_price']:,.2f}\n\n"
                f"🛑 *Stop Loss (SL):* ${setup['stop_loss_sl']:,.2f}\n"
                f"🎯 *Take Profit 1:* ${setup['take_profit_1']:,.2f}\n"
                f"🎯 *Take Profit 2:* ${setup['take_profit_2']:,.2f}\n\n"
                f"📈 *RSI (14):* {tech['rsi_14']} ({tech['rsi_condition']})\n"
                f"⚡ *ATR Volatility:* {tech['atr_volatility']:,.2f}\n\n"
                f"💬 *AI Desk Insight:* {signal_data['ai_market_insight']}"
            )
            send_telegram_alert(signal_message)
          except Exception as inner_e:
            send_telegram_alert(
                f"⚠️ Error processing {asset['name']}: {str(inner_e)}"
            )
          await asyncio.sleep(1)

      elif text == "/status":
        status_message = (
            "🟢 *QUANT ENGINE STATUS: ONLINE*\n\n"
            "🚀 *Platform:* Render Cloud Infrastructure\n"
            "⏱ *Polling Interval:* Every 3 Hours\n"
            "📈 *Tracked Assets:* Gold (XAU/USD) & Bitcoin (BTC/USD)\n"
            "🤖 *Webhooks & AI Desk:* Active & Responsive"
        )
        send_telegram_alert(status_message)
  except Exception as e:
    print(f"WEBHOOK ERROR: {e}")

  return {"status": "ok"}