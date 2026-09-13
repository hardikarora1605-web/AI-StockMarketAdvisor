"""
AI Stock Market Advisor — Streamlit Web App
Run with:  streamlit run app.py
"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go

# ----------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="AI Stock Market Advisor",
    page_icon="📈",
    layout="wide"
)

# ----------------------------------------------------------------------
# SESSION STATE (for chatbot memory)
# ----------------------------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []   # list of {"role": ..., "content": ...}
if "last_symbol" not in st.session_state:
    st.session_state.last_symbol = None
if "stock_context" not in st.session_state:
    st.session_state.stock_context = ""  # summary of current stock, given to the chatbot

# ----------------------------------------------------------------------
# SIDEBAR — API KEYS & INPUTS
# ----------------------------------------------------------------------
st.sidebar.title("⚙️ Settings")

groq_key = st.sidebar.text_input(
    "Groq API Key",
    type="password",
    value=os.environ.get("GROQ_API_KEY", "")
)
tavily_key = st.sidebar.text_input(
    "Tavily API Key",
    type="password",
    value=os.environ.get("TAVILY_API_KEY", "")
)

symbol = st.sidebar.text_input("Stock Symbol", value="AAPL").upper().strip()

st.sidebar.markdown("---")
show_forecast_arima = st.sidebar.checkbox("Include ARIMA Forecast", value=True)
show_forecast_lstm = st.sidebar.checkbox(
    "Include LSTM Forecast (slower — trains a model live)",
    value=False
)

run_button = st.sidebar.button("🔍 Analyze Stock", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Get a free Groq key at console.groq.com and a free Tavily key at "
    "app.tavily.com. Keys are only used for this session, never stored."
)

# ----------------------------------------------------------------------
# TITLE
# ----------------------------------------------------------------------
st.title("📈 AI Stock Market Advisor")
st.caption(
    "Live prices, technical indicators, news sentiment, and AI-generated "
    "analysis — powered by yfinance, Tavily, and Groq."
)

# ----------------------------------------------------------------------
# HELPER FUNCTIONS (cached so repeat calls are fast)
# ----------------------------------------------------------------------

@st.cache_data(ttl=300, show_spinner=False)
def fetch_history(sym, period="1y"):
    stock = yf.Ticker(sym)
    df = stock.history(period=period)
    return df


@st.cache_data(ttl=300, show_spinner=False)
def fetch_info(sym):
    stock = yf.Ticker(sym)
    try:
        return stock.info
    except Exception:
        return {}


def add_indicators(df):
    df = df.copy()
    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()

    delta = df["Close"].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain, index=df.index).rolling(14).mean()
    avg_loss = pd.Series(loss, index=df.index).rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    df["EMA12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA26"] = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = df["EMA12"] - df["EMA26"]
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    return df


def plot_price_chart(df, sym):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"], name="Price"
    ))
    fig.add_trace(go.Scatter(x=df.index, y=df["SMA20"], name="SMA 20", line=dict(width=1.5)))
    fig.add_trace(go.Scatter(x=df.index, y=df["SMA50"], name="SMA 50", line=dict(width=1.5)))
    fig.update_layout(
        title=f"{sym} Price with Moving Averages",
        xaxis_title="Date", yaxis_title="Price",
        template="plotly_white", height=500,
        xaxis_rangeslider_visible=False
    )
    return fig


def plot_rsi_chart(df, sym):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI", line=dict(color="orange")))
    fig.add_hline(y=70, line_dash="dash", annotation_text="Overbought", line_color="red")
    fig.add_hline(y=30, line_dash="dash", annotation_text="Oversold", line_color="green")
    fig.update_layout(title=f"{sym} RSI", template="plotly_white", height=350)
    return fig


def plot_macd_chart(df, sym):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD"))
    fig.add_trace(go.Scatter(x=df.index, y=df["Signal"], name="Signal"))
    fig.update_layout(title=f"{sym} MACD", template="plotly_white", height=350)
    return fig


def get_news(company, api_key, max_results=5):
    from tavily import TavilyClient
    client = TavilyClient(api_key=api_key)
    response = client.search(
        query=f"{company} latest stock news",
        topic="news",
        max_results=max_results
    )
    return response.get("results", [])


def ai_report(sym, info, df, news, api_key, arima_text, lstm_text):
    from langchain_groq import ChatGroq

    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, api_key=api_key)

    latest = df.iloc[-1]
    news_text = "\n\n".join(
        f"Title: {n.get('title')}\nContent: {n.get('content')}" for n in news
    ) or "No news available."

    trend = "Bullish" if latest["SMA20"] > latest["SMA50"] else "Bearish"

    prompt = f"""
You are a professional financial analyst.

Analyze the stock ONLY using the data provided below. Do not use outside
knowledge, do not invent numbers, and do not invent risks. If information
is missing, write "Not available from the collected data."

Stock: {sym}
Company: {info.get('longName', 'N/A')}
Sector: {info.get('sector', 'N/A')}
Industry: {info.get('industry', 'N/A')}
Current Price: {latest['Close']:.2f}
Market Cap: {info.get('marketCap', 'N/A')}
SMA20: {latest['SMA20']:.2f}
SMA50: {latest['SMA50']:.2f}
RSI: {latest['RSI']:.2f}
MACD: {latest['MACD']:.2f}
Signal Line: {latest['Signal']:.2f}
Trend (SMA-based): {trend}

ARIMA Forecast:
{arima_text}

LSTM Forecast:
{lstm_text}

Recent News:
{news_text}

Generate a report in this exact format:

# 📈 Stock Analysis Report

## Executive Summary
## Current Market Status
## Technical Indicators
## Latest News
(For each article: Headline, Impact Positive/Negative/Neutral, one-sentence reason)
## Forecast Comparison
## Risk Assessment
## Final Recommendation
Recommendation: BUY / HOLD / SELL
Confidence: Low / Medium / High
Reason: (3 bullet points using ONLY the data above)

This is educational analysis only, not financial advice.
"""
    response = llm.invoke(prompt)
    return response.content


def build_stock_context(sym, info, df, news):
    """One-paragraph summary of the loaded stock, given to the chatbot as grounding."""
    latest = df.iloc[-1]
    news_lines = "\n".join(f"- {n.get('title')}" for n in news[:5]) or "No recent news available."
    return f"""
Stock: {sym}
Company: {info.get('longName', 'N/A')}
Sector: {info.get('sector', 'N/A')} | Industry: {info.get('industry', 'N/A')}
Current Price: {latest['Close']:.2f}
SMA20: {latest['SMA20']:.2f} | SMA50: {latest['SMA50']:.2f}
RSI(14): {latest['RSI']:.2f}
MACD: {latest['MACD']:.2f} | Signal: {latest['Signal']:.2f}
52-Week High: {info.get('fiftyTwoWeekHigh', 'N/A')} | 52-Week Low: {info.get('fiftyTwoWeekLow', 'N/A')}
Market Cap: {info.get('marketCap', 'N/A')}

Recent News Headlines:
{news_lines}
""".strip()


def chatbot_response(user_question, context, chat_history, api_key):
    """Answers a follow-up question using Groq, grounded only in the loaded stock's data."""
    from langchain_groq import ChatGroq
    from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3, api_key=api_key)

    system_prompt = f"""You are a helpful stock market assistant chatting with a user
about ONE specific stock. Answer ONLY using the data below and the conversation
history. If asked something the data doesn't cover, say so honestly instead of
guessing. Keep answers concise (a few sentences, or a short list). Never present
your answers as certified financial advice — you can discuss the data's implications,
but remind the user this is educational, not investment advice, if they ask for a
recommendation.

STOCK DATA:
{context}
"""

    messages = [SystemMessage(content=system_prompt)]
    for turn in chat_history:
        if turn["role"] == "user":
            messages.append(HumanMessage(content=turn["content"]))
        else:
            messages.append(AIMessage(content=turn["content"]))
    messages.append(HumanMessage(content=user_question))

    response = llm.invoke(messages)
    return response.content


def arima_forecast(df, sym):
    from statsmodels.tsa.arima.model import ARIMA
    data = df["Close"]
    model = ARIMA(data, order=(5, 1, 0))
    fit = model.fit()
    forecast = fit.forecast(steps=30)
    future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=30, freq="B")
    change = ((forecast.iloc[-1] - data.iloc[-1]) / data.iloc[-1]) * 100
    text = (
        f"Current Price: ${data.iloc[-1]:.2f}\n"
        f"30-Day Forecast: ${forecast.iloc[-1]:.2f}\n"
        f"Expected Change: {change:.2f}%"
    )
    return text, future_dates, forecast


@st.cache_resource(show_spinner=False)
def train_lstm(sym):
    """Trains a fresh LSTM for this symbol and caches it for the session."""
    from sklearn.preprocessing import MinMaxScaler
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense

    stock = yf.Ticker(sym)
    hist = stock.history(period="2y")
    data = hist[["Close"]]

    train_size = int(len(data) * 0.8)
    train_raw = data.iloc[:train_size]

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_raw)  # fit ONLY on training data to avoid leakage
    scaled_data = scaler.transform(data)

    sequence_length = 60
    X, y = [], []
    for i in range(sequence_length, train_size):
        X.append(scaled_data[i - sequence_length:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(60, 1)),
        LSTM(50),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(X, y, epochs=8, batch_size=32, verbose=0)

    return model, scaler, scaled_data, data


def lstm_forecast(sym):
    model, scaler, scaled_data, data = train_lstm(sym)

    last_60 = scaled_data[-60:]
    current_batch = last_60.reshape(1, 60, 1)
    future_predictions = []
    for _ in range(30):
        pred = model.predict(current_batch, verbose=0)[0]
        future_predictions.append(pred)
        current_batch = np.append(current_batch[:, 1:, :], [[pred]], axis=1)

    future_predictions = scaler.inverse_transform(future_predictions)
    current_price = data.iloc[-1].values[0]
    predicted_price = future_predictions[-1][0]
    change = ((predicted_price - current_price) / current_price) * 100
    future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=30, freq="B")

    text = (
        f"Current Price: ${current_price:.2f}\n"
        f"30-Day Forecast: ${predicted_price:.2f}\n"
        f"Expected Change: {change:.2f}%"
    )
    return text, future_dates, future_predictions.flatten()


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
if "analyzed" not in st.session_state:
    st.session_state.analyzed = False
if "active_symbol" not in st.session_state:
    st.session_state.active_symbol = None

if run_button:
    if not symbol:
        st.error("Please enter a stock symbol.")
        st.stop()
    st.session_state.analyzed = True
    st.session_state.active_symbol = symbol

# Stay "analyzed" across chat reruns (st.chat_input triggers a full rerun,
# and a plain st.button always resets to False on rerun otherwise).
if st.session_state.analyzed:
    symbol = st.session_state.active_symbol

    with st.spinner(f"Fetching data for {symbol}..."):
        hist = fetch_history(symbol)
        info = fetch_info(symbol)

    if hist.empty:
        st.error(f"No data found for '{symbol}'. Check the symbol and try again.")
        st.stop()

    hist = add_indicators(hist)
    latest = hist.iloc[-1]

    # Reset chatbot memory when the user switches to a different symbol
    if st.session_state.last_symbol != symbol:
        st.session_state.chat_history = []
        st.session_state.last_symbol = symbol

    # Fetch news once here so both the AI Report tab and the Chatbot tab can use it
    news = []
    if tavily_key:
        try:
            news = get_news(info.get("longName", symbol), tavily_key)
        except Exception as e:
            st.warning(f"News fetch failed: {e}")

    st.session_state.stock_context = build_stock_context(symbol, info, hist, news)

    # --- Top metrics ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Current Price", f"${latest['Close']:.2f}")
    col2.metric("RSI (14)", f"{latest['RSI']:.1f}")
    trend_label = "Bullish 🟢" if latest["SMA20"] > latest["SMA50"] else "Bearish 🔴"
    col3.metric("Trend (SMA)", trend_label)
    mcap = info.get("marketCap")
    col4.metric("Market Cap", f"${mcap/1e9:.1f}B" if mcap else "N/A")

    tabs = st.tabs(["📊 Charts", "📉 Indicators", "📰 News & AI Report", "🔮 Forecast", "💬 Chatbot"])

    # --- Charts tab ---
    with tabs[0]:
        st.plotly_chart(plot_price_chart(hist, symbol), use_container_width=True)
        with st.expander("Company Info"):
            st.write(f"**Company:** {info.get('longName', 'N/A')}")
            st.write(f"**Sector:** {info.get('sector', 'N/A')}")
            st.write(f"**Industry:** {info.get('industry', 'N/A')}")
            st.write(f"**52-Week High:** {info.get('fiftyTwoWeekHigh', 'N/A')}")
            st.write(f"**52-Week Low:** {info.get('fiftyTwoWeekLow', 'N/A')}")

    # --- Indicators tab ---
    with tabs[1]:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(plot_rsi_chart(hist, symbol), use_container_width=True)
        with c2:
            st.plotly_chart(plot_macd_chart(hist, symbol), use_container_width=True)

    # --- News & AI report tab ---
    with tabs[2]:
        arima_text, lstm_text = "Not generated.", "Not generated."
        arima_dates = arima_vals = lstm_dates = lstm_vals = None

        if show_forecast_arima:
            with st.spinner("Running ARIMA forecast..."):
                try:
                    arima_text, arima_dates, arima_vals = arima_forecast(hist, symbol)
                except Exception as e:
                    arima_text = f"ARIMA failed: {e}"

        if show_forecast_lstm:
            with st.spinner("Training LSTM model (this can take a minute)..."):
                try:
                    lstm_text, lstm_dates, lstm_vals = lstm_forecast(symbol)
                except Exception as e:
                    lstm_text = f"LSTM failed: {e}"

        if not groq_key or not tavily_key:
            st.warning("Enter your Groq and Tavily API keys in the sidebar to generate the AI report.")
        else:
            with st.spinner("Generating AI report..."):
                try:
                    report = ai_report(symbol, info, hist, news, groq_key, arima_text, lstm_text)
                    st.markdown(report)
                except Exception as e:
                    st.error(f"AI report generation failed: {e}")

            if news:
                with st.expander("Raw News Sources"):
                    for n in news:
                        st.markdown(f"**{n.get('title')}**")
                        st.write(n.get("content"))
                        st.markdown(f"[Source]({n.get('url')})")
                        st.markdown("---")

    # --- Forecast tab ---
    with tabs[3]:
        if show_forecast_arima and arima_vals is not None:
            st.subheader("ARIMA Forecast")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=hist.index[-100:], y=hist["Close"][-100:], name="Historical"))
            fig.add_trace(go.Scatter(x=arima_dates, y=arima_vals, name="ARIMA Forecast", line=dict(color="red")))
            fig.update_layout(template="plotly_white", height=450)
            st.plotly_chart(fig, use_container_width=True)
            st.text(arima_text)

        if show_forecast_lstm and lstm_vals is not None:
            st.subheader("LSTM Forecast")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=hist.index[-100:], y=hist["Close"][-100:], name="Historical"))
            fig.add_trace(go.Scatter(x=lstm_dates, y=lstm_vals, name="LSTM Forecast", line=dict(color="purple")))
            fig.update_layout(template="plotly_white", height=450)
            st.plotly_chart(fig, use_container_width=True)
            st.text(lstm_text)

        if not show_forecast_arima and not show_forecast_lstm:
            st.info("Enable ARIMA or LSTM forecasting in the sidebar to see predictions here.")

    # --- Chatbot tab ---
    with tabs[4]:
        st.caption(f"Ask me anything about **{symbol}** — I'll answer using the data loaded above.")

        if not groq_key:
            st.warning("Enter your Groq API key in the sidebar to use the chatbot.")
        else:
            # Show existing conversation
            for turn in st.session_state.chat_history:
                with st.chat_message(turn["role"]):
                    st.markdown(turn["content"])

            user_question = st.chat_input(f"Ask something about {symbol}...")

            if user_question:
                st.session_state.chat_history.append({"role": "user", "content": user_question})
                with st.chat_message("user"):
                    st.markdown(user_question)

                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        try:
                            answer = chatbot_response(
                                user_question,
                                st.session_state.stock_context,
                                st.session_state.chat_history[:-1],  # history before this question
                                groq_key
                            )
                        except Exception as e:
                            answer = f"Sorry, something went wrong: {e}"
                    st.markdown(answer)

                st.session_state.chat_history.append({"role": "assistant", "content": answer})

            if st.session_state.chat_history:
                if st.button("🗑️ Clear conversation"):
                    st.session_state.chat_history = []
                    st.rerun()

else:
    st.info("👈 Enter your API keys and a stock symbol in the sidebar, then click **Analyze Stock**.")