from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from contextlib import asynccontextmanager
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import asyncio
import os
from openai import OpenAI

# Background automation loop
async def autonomous_signal_scheduler():
    """Background worker that runs automatically while the server is live"""
    # Wait 10 seconds after startup before the first check
    await asyncio.sleep(10)
    
    while True:
        try:
            print("[AUTONOMOUS ENGINE] Running scheduled XAU/USD market check...")
            # Automatically trigger the signal and send to Telegram
            signal_data = generate_ai_gold_signal()
            setup = signal_data["signal_setup"]
            tech = signal_data["technical_indicators"]
            
            message = (
                f"🤖 *AUTONOMOUS XAU/USD ALERT* 🤖\n\n"
                f"📌 *Action:* `{setup['action']}`\n"
                f"📊 *Market Bias:* {setup['market_bias']}\n"
                f"💰 *Entry Price:* `${setup['suggested_entry']:,.2f}`\n\n"
                f"🛑 *Stop Loss (SL):* `${setup['stop_loss_sl']:,.2f}`\n"
                f"🎯 *Take Profit 1:* `${setup['take_profit_1']:,.2f}`\n"
                f"🎯 *Take Profit 2:* `${setup['take_profit_2']:,.2f}`\n\n"
                f"📈 *RSI (14):* {tech['rsi_14']} ({tech['rsi_condition']})\n"
                f"⚡ *ATR Volatility:* {tech['atr_volatility']}\n\n"
                f"💬 *Insight:* {signal_data['ai_market_insight']}"
            )
            
            send_telegram_alert(message)
            print("[AUTONOMOUS ENGINE] Automated signal successfully sent to Telegram!")
            
        except Exception as e:
            print(f"[AUTONOMOUS ENGINE ERROR]: {e}")
        
        # Sleep for 1 hour (3600 seconds) before checking the market again
        # (You can change this to 1800 for 30 mins, or 60 for 1 minute during testing)
        await asyncio.sleep(3600)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Launch the background scheduler task
    task = asyncio.create_task(autonomous_signal_scheduler())
    yield
    # Shutdown: Clean up the task if server stops
    task.cancel()

app = FastAPI(
    title="XAU/USD Gold AI & Quant Signal Engine",
    description="Autonomous institutional-grade gold trading dashboard and Telegram bot.",
    version="4.0.0",
    lifespan=lifespan
)

GOLD_TICKER = "GC=F" 

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "your-fallback-key-here"))

# Telegram Configuration
TELEGRAM_BOT_TOKEN = "8802295558:AAFNqNY5utWXzqFXuFnQthXiZwZacqz0rfY"
TELEGRAM_CHAT_ID = "6951483459"

def send_telegram_alert(message: str):
    if TELEGRAM_BOT_TOKEN == "YOUR_BOT_TOKEN_HERE":
        return {"status": "error", "message": "Telegram credentials not configured yet."}
    
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"

    }
    response = requests.post(url, json=payload)
    return response.json()

@app.get("/")
def home():
    return {"status": "online", "message": "Autonomous Gold AI Backend is running. Visit /dashboard."}

@app.get("/ai-signal")
def generate_ai_gold_signal():
  try:
    df = yf.download(GOLD_TICKER, period="1mo", interval="1h", progress=False)

    if df.empty:
      raise HTTPException(
          status_code=404, detail="No market data returned for Gold."
      )

    if isinstance(df.columns, pd.MultiIndex):
      df.columns = df.columns.get_level_values(0)

    close_prices = df["Close"].astype(float)
    high_prices = df["High"].astype(float)
    low_prices = df["Low"].astype(float)

    # Calculate RSI (14)
    delta = close_prices.diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # Calculate ATR (14)
    tr1 = high_prices - low_prices
    tr2 = (high_prices - close_prices.shift()).abs()
    tr3 = (low_prices - close_prices.shift()).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["ATR"] = true_range.rolling(window=14).mean()

    # Calculate EMA 20 & EMA 50 Trend Confluence
    df["EMA_20"] = close_prices.ewm(span=20, adjust=False).mean()
    df["EMA_50"] = close_prices.ewm(span=50, adjust=False).mean()

    latest = df.iloc[-1]
    current_price = float(latest["Close"])
    rsi_val = float(latest["RSI"]) if not pd.isna(latest["RSI"]) else 50.0
    atr_val = float(latest["ATR"]) if not pd.isna(latest["ATR"]) else 15.0
    ema_20 = (
        float(latest["EMA_20"]) if not pd.isna(latest["EMA_20"]) else current_price
    )
    ema_50 = (
        float(latest["EMA_50"]) if not pd.isna(latest["EMA_50"]) else current_price
    )

    # Confluence Signal Logic
    if ema_20 > ema_50:
      trend = "Bullish"
      signal = "BUY (LONG)"
      stop_loss = round(current_price - (atr_val * 1.5), 2)
      tp_1 = round(current_price + (atr_val * 2.0), 2)
      tp_2 = round(current_price + (atr_val * 3.5), 2)
    else:
      trend = "Bearish"
      signal = "SELL (SHORT)"
      stop_loss = round(current_price + (atr_val * 1.5), 2)
      tp_1 = round(current_price - (atr_val * 2.0), 2)
      tp_2 = round(current_price - (atr_val * 3.5), 2)

    rsi_condition = (
        "Overbought"
        if rsi_val > 70
        else ("Oversold" if rsi_val < 30 else "Neutral")
    )

    ai_commentary = f"Quantitative Engine Active (EMA Confluence): Gold is showing a {trend.lower()} bias at ${current_price:.2f} with RSI at {rsi_val:.2f} ({rsi_condition}). EMA 20 is at ${ema_20:.2f} and EMA 50 is at ${ema_50:.2f}. Risk management locked in with a {signal} setup."

    return {
        "asset": "Gold (XAU/USD)",
        "timestamp": str(df.index[-1]),
        "current_price": round(current_price, 2),
        "ai_market_insight": ai_commentary,
        "signal_setup": {
            "action": signal,
            "market_bias": trend,
            "suggested_entry": round(current_price, 2),
            "stop_loss_sl": stop_loss,
            "take_profit_1": tp_1,
            "take_profit_2": tp_2,
        },
        "technical_indicators": {
            "rsi_14": round(rsi_val, 2),
            "rsi_condition": rsi_condition,
            "atr_volatility": round(atr_val, 2),
            "ema_20": round(ema_20, 2),
            "ema_50": round(ema_50, 2),
        },
    }

  except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))

@app.get("/send-telegram-signal")
def trigger_telegram_signal():
    try:
        signal_data = generate_ai_gold_signal()
        setup = signal_data["signal_setup"]
        tech = signal_data["technical_indicators"]
        
        message = (
            f"🚨 *XAU/USD MANUAL SIGNAL* 🚨\n\n"
            f"📌 *Action:* `{setup['action']}`\n"
            f"📊 *Market Bias:* {setup['market_bias']}\n"
            f"💰 *Entry Price:* `${setup['suggested_entry']:,.2f}`\n\n"
            f"🛑 *Stop Loss (SL):* `${setup['stop_loss_sl']:,.2f}`\n"
            f"🎯 *Take Profit 1:* `${setup['take_profit_1']:,.2f}`\n"
            f"🎯 *Take Profit 2:* `${setup['take_profit_2']:,.2f}`\n\n"
            f"📈 *RSI (14):* {tech['rsi_14']} ({tech['rsi_condition']})\n"
            f"⚡ *ATR Volatility:* {tech['atr_volatility']}\n\n"
            f"💬 *Insight:* {signal_data['ai_market_insight']}"
        )
        
        result = send_telegram_alert(message)
        return {"status": "success", "telegram_response": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>XAU/USD Autonomous Terminal</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background-color: #0d1117;
                color: #c9d1d9;
                margin: 0;
                padding: 20px;
                display: flex;
                flex-direction: column;
                align-items: center;
            }
            .header { text-align: center; margin-bottom: 30px; }
            .header h1 { color: #f0abfc; margin: 0; font-size: 28px; }
            .header p { color: #8b949e; font-size: 14px; }
            .card {
                background-color: #161b22;
                border: 1px solid #30363d;
                border-radius: 12px;
                width: 100%;
                max-width: 600px;
                padding: 25px;
                box-shadow: 0 8px 24px rgba(0,0,0,0.5);
            }
            .signal-badge {
                display: inline-block;
                padding: 8px 16px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 16px;
                margin-bottom: 20px;
            }
            .buy { background-color: #238636; color: #ffffff; }
            .sell { background-color: #da3633; color: #ffffff; }
            .metric-grid {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 15px;
                margin-bottom: 20px;
            }
            .metric-box {
                background-color: #21262d;
                padding: 12px 15px;
                border-radius: 8px;
                border: 1px solid #30363d;
            }
            .metric-box span { display: block; font-size: 12px; color: #8b949e; }
            .metric-box strong { font-size: 18px; color: #f0f6fc; }
            .insight-box {
                background-color: #1f6feb22;
                border: 1px solid #1f6feb;
                padding: 15px;
                border-radius: 8px;
                margin-bottom: 20px;
                font-size: 14px;
                line-height: 1.5;
                color: #58a6ff;
            }
            .btn-group { display: flex; gap: 10px; }
            .action-btn {
                background-color: #238636;
                color: white;
                border: none;
                padding: 12px 20px;
                border-radius: 6px;
                font-size: 15px;
                cursor: pointer;
                flex: 1;
                font-weight: bold;
                transition: background 0.2s;
            }
            .action-btn:hover { background-color: #2ea043; }
            .telegram-btn { background-color: #1f6feb; }
            .telegram-btn:hover { background-color: #388bfd; }
            .loading { text-align: center; color: #8b949e; font-style: italic; }
        </style>
    </head>
    <body>

        <div class="header">
            <h1>🟡 XAU/USD Autonomous Terminal</h1>
            <p>100% Automated Background Quant & Telegram Engine</p>
        </div>

        <div class="card" id="dashboard-card">
            <div class="loading">Loading live market data...</div>
        </div>

        <script>
            async function fetchSignal() {
                const card = document.getElementById('dashboard-card');
                card.innerHTML = '<div class="loading">Fetching live market data...</div>';
                
                try {
                    const response = await fetch('/ai-signal');
                    const data = await response.json();
                    
                    const actionClass = data.signal_setup.action.includes('BUY') ? 'buy' : 'sell';
                    
                    card.innerHTML = `
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div class="signal-badge ${actionClass}">${data.signal_setup.action}</div>
                            <span style="font-size: 13px; color: #8b949e;">${data.timestamp}</span>
                        </div>

                        <div class="insight-box">
                            <strong>Market Insight:</strong><br>
                            ${data.ai_market_insight}
                        </div>

                        <div class="metric-grid">
                            <div class="metric-box">
                                <span>Current Gold Price</span>
                                <strong>$${data.current_price.toLocaleString()}</strong>
                            </div>
                            <div class="metric-box">
                                <span>Market Bias</span>
                                <strong style="color: ${data.signal_setup.market_bias === 'Bullish' ? '#3fb950' : '#f85149'};">${data.signal_setup.market_bias}</strong>
                            </div>
                            <div class="metric-box">
                                <span>Stop Loss (SL)</span>
                                <strong style="color: #f85149;">$${data.signal_setup.stop_loss_sl.toLocaleString()}</strong>
                            </div>
                            <div class="metric-box">
                                <span>Take Profit 1 (TP1)</span>
                                <strong style="color: #3fb950;">$${data.signal_setup.take_profit_1.toLocaleString()}</strong>
                            </div>
                            <div class="metric-box">
                                <span>Take Profit 2 (TP2)</span>
                                <strong style="color: #3fb950;">$${data.signal_setup.take_profit_2.toLocaleString()}</strong>
                            </div>
                            <div class="metric-box">
                                <span>RSI (14) / ATR</span>
                                <strong>${data.technical_indicators.rsi_14} (${data.technical_indicators.atr_volatility})</strong>
                            </div>
                        </div>

                        <div class="btn-group">
                            <button class="action-btn" onclick="fetchSignal()">Refresh Signal</button>
                            <button class="action-btn telegram-btn" onclick="sendTelegram()">Send Manual 📱</button>
                        </div>
                    `;
                } catch (err) {
                    card.innerHTML = `<div style="color: #f85149; text-align: center;">Error loading signal data.</div>`;
                }
            }

            async function sendTelegram() {
                try {
                    const res = await fetch('/send-telegram-signal');
                    const result = await res.json();
                    if(result.status === 'success') {
                        alert('Manual signal sent to Telegram!');
                    } else {
                        alert('Check Telegram credentials in main.py');
                    }
                } catch(e) {
                    alert('Failed to send Telegram alert.');
                }
            }

            fetchSignal();
        </script>

    </body>
    </html>
    """