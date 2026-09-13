# 📈 AI-Powered Stock Market Advisor

An AI-powered web application that combines live market data, technical analysis, dual time-series forecasting models, and an LLM-based agentic chatbot into a single interactive Streamlit dashboard.

Built to explore how large language models can be grounded in real, live financial data — rather than relying on a model's own (often outdated) knowledge — while giving users a genuine conversational interface for stock research.

---

## ✨ Features

- **Live price & charting** — candlestick charts with SMA-20/SMA-50 overlays, powered by [yfinance](https://pypi.org/project/yfinance/)
- **Technical indicators** — RSI(14) and MACD, computed directly from OHLCV data
- **Dual price forecasting** — a classical **ARIMA** model and a 2-layer **LSTM** neural network, compared side by side over a 30-day horizon
- **AI-generated analysis report** — an LLM (Llama 3.3 70B, served via Groq) synthesizes price data, indicators, forecasts, and news into a structured report, strictly grounded in the retrieved data to minimize hallucination
- **Live news retrieval** — recent, relevant news pulled via the [Tavily](https://tavily.com/) search API
- **Agentic AI chatbot** — not a static Q&A bot: the assistant has real callable tools (price lookup, indicators, forecasting, news, and fund-name search) and decides for itself which to call, for **any** stock symbol — not just the one currently loaded
- **Mutual fund / SIP lookup** — search for a fund by name (e.g. "SBI Bluechip Fund") to resolve its ticker and pull NAV history, useful for funds where the exact symbol isn't known
- **Session-aware caching** — avoids redundant API calls and retrains models only once per session

## 🏗️ Architecture

```
User (Browser)
      │
      ▼
Streamlit Web App (app.py)
      │
      ├── Data Acquisition ─────────── yfinance (Yahoo Finance)
      ├── Technical Indicators ─────── SMA, RSI, MACD
      ├── Forecasting ──────────────── ARIMA (statsmodels) + LSTM (TensorFlow/Keras)
      ├── News Retrieval ───────────── Tavily Search API
      └── LLM Reasoning Layer ──────── Groq (Llama 3.3 70B) via LangChain
              ├── AI Report Generator
              └── Agentic Chatbot (tool-calling)
```

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| Web Framework | Streamlit |
| Market Data | yfinance |
| News Search | Tavily API |
| LLM / Agent Framework | LangChain + Groq (Llama 3.3 70B) |
| Forecasting | statsmodels (ARIMA), TensorFlow/Keras (LSTM), scikit-learn |
| Visualization | Plotly |

## 🚀 Getting Started

### Prerequisites
- Python 3.10 or higher
- Free API keys from:
  - [Groq Console](https://console.groq.com) — for LLM access
  - [Tavily](https://app.tavily.com) — for news search

### Installation

```bash
git clone https://github.com/hardikarora1605-web/ai-stock-market-advisor.git
cd ai-stock-market-advisor
pip install -r requirements.txt
```

> **Note:** if you don't need the LSTM forecast, you can skip installing `tensorflow` — just leave the "Include LSTM Forecast" checkbox unchecked in the app.

### Run locally

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`. Paste your Groq and Tavily API keys into the sidebar, enter a stock symbol, and click **Analyze Stock**.

### Deploy (optional)

The easiest option is [Streamlit Community Cloud](https://share.streamlit.io):
1. Push this repo to GitHub (already done if you're reading this there)
2. Go to share.streamlit.io → sign in with GitHub → select this repo
3. Set `app.py` as the entry point and deploy

## 📂 Project Structure

```
ai-stock-market-advisor/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md           # You are here
```

## ⚠️ Known Limitations

- Trained LSTM models are cached only for the active session and are not persisted to disk — they retrain if the server restarts
- No user authentication or persistent multi-session storage (watchlists, chat history) yet
- Mutual fund coverage via Yahoo Finance is inconsistent, especially for Indian AMC schemes; NAVs may lag by a day
- Forecasts are based on price history alone and do not incorporate fundamental or macroeconomic data
- This tool is for **educational purposes only** and does not constitute financial advice

## 🗺️ Roadmap

- [ ] Persist trained models and user data via a lightweight database
- [ ] Add authentication and per-user watchlists
- [ ] Integrate a dedicated sentiment-scoring model alongside LLM-based news analysis
- [ ] Add fundamental data (P/E, EPS, dividend yield) and peer/sector comparison
- [ ] Backtest forecast accuracy (MAPE) against realized prices
- [ ] Optional RAG module for uploaded 10-K/annual report analysis

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 👤 Author

**Hardik Arora**
- GitHub: [@hardikarora1605-web](https://github.com/hardikarora1605-web)
- LinkedIn: [Hardik Arora](https://www.linkedin.com/in/hardik-arora-a19129298/)

---

*Disclaimer: This application is built for educational and demonstration purposes. It does not provide financial advice, and any AI-generated recommendations should not be used as the sole basis for investment decisions.*
