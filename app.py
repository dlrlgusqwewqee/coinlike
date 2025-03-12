import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import ccxt
import requests
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import logging
from datetime import datetime
from streamlit_autorefresh import st_autorefresh

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 1. 자동 새로고침 (15초 간격)
st_autorefresh(interval=15000, limit=10000, key="realtime_refresh")

# 2. Coindesk 뉴스 스크래핑 및 감성 분석
def fetch_coindesk_headlines():
    url = "https://www.coindesk.com/price/bitcoin"
    headlines = []
    try:
        resp = requests.get(url)
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, "html.parser")
            for h in soup.find_all("h3"):
                text = h.get_text(strip=True)
                if text:
                    headlines.append(text)
        else:
            logging.error(f"Coindesk 응답 코드: {resp.status_code}")
    except Exception as e:
        logging.error(f"Coindesk 뉴스 스크래핑 오류: {e}")
    return headlines

def analyze_news_sentiment(headlines):
    if not headlines:
        return 0
    analyzer = SentimentIntensityAnalyzer()
    scores = [analyzer.polarity_scores(headline)["compound"] for headline in headlines]
    return np.mean(scores)

# 3. Bitget OHLCV 데이터 가져오기 (ccxt 사용)
def fetch_ohlcv(symbol="BTC/USDT", timeframe="1m", limit=300):
    exchange = ccxt.bitget({"enableRateLimit": True})
    try:
        data = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.sort_values("timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df
    except Exception as e:
        st.error(f"데이터 불러오기 오류: {e}")
        return None

# 4. 지표 계산: MA50, MA200, MACD
def add_indicators(df):
    df["MA50"] = df["close"].rolling(50).mean()
    df["MA200"] = df["close"].rolling(200).mean()
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_signal"]
    return df

# 5. 단일 시간대 신호 판단
def get_signal(df):
    if pd.isna(df["MA50"].iloc[-1]) or pd.isna(df["MA200"].iloc[-1]):
        return 0
    ma50 = df["MA50"].iloc[-1]
    ma200 = df["MA200"].iloc[-1]
    macd_hist = df["MACD_hist"].iloc[-1]
    if ma50 > ma200 and macd_hist > 0:
        return 1
    elif ma50 < ma200 and macd_hist < 0:
        return -1
    else:
        return 0

# 6. 차트 생성 (Plotly)
def create_candle_chart(df, title):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df["timestamp"],
        open=df["open"],
        high=df["high"],
        low=df["low"],
        close=df["close"],
        name="캔들"
    ))
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["MA50"],
                             mode="lines", name="MA50", line=dict(color="orange", width=1.5)))
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["MA200"],
                             mode="lines", name="MA200", line=dict(color="blue", width=1.5)))
    fig.update_layout(title=title,
                      xaxis_title="시간",
                      yaxis_title="가격 (USDT)",
                      xaxis_rangeslider_visible=False,
                      height=600)
    return fig

# 7. 여러 시간대 신호 취합 및 종합 추천
def aggregate_signals(timeframes):
    signals = {}
    for tf in timeframes:
        df = fetch_ohlcv("BTC/USDT", timeframe=tf, limit=300)
        if df is not None:
            df = add_indicators(df)
            sig = get_signal(df)
            signals[tf] = sig
        else:
            signals[tf] = 0
    return signals

# ----- Streamlit UI 구성 -----
st.title("실시간 비트코인 차트 & 뉴스 기반 롱/숏 추천 (Bitget)")

# --- (1) 뉴스 감성 분석 ---
st.subheader("1. 최신 뉴스 감성 (Coindesk)")
news_headlines = fetch_coindesk_headlines()
news_sentiment = analyze_news_sentiment(news_headlines)
st.write(f"평균 감성 점수: {news_sentiment:.2f}")

# --- (2) 실시간 Bitget 차트 ---
st.subheader("2. 실시간 Bitget 차트 (여러 시간대)")
timeframes = ["1m", "15m", "30m", "1h", "4h", "1d"]
tabs = st.tabs(["1분", "15분", "30분", "1시간", "4시간", "1일"])
technical_signals = {}
for i, tf in enumerate(timeframes):
    with tabs[i]:
        df_tf = fetch_ohlcv("BTC/USDT", timeframe=tf, limit=300)
        if df_tf is not None:
            df_tf = add_indicators(df_tf)
            candle_fig = create_candle_chart(df_tf, title=f"BTC/USDT {tf} 차트")
            st.plotly_chart(candle_fig, use_container_width=True)
            sig = get_signal(df_tf)
            technical_signals[tf] = sig
            st.write(f"[{tf}] 신호: {'롱' if sig == 1 else '숏' if sig == -1 else '중립'}")

# --- (3) 종합 신호 ---
st.subheader("3. 종합 롱/숏 추천")
agg_signals = aggregate_signals(timeframes)
st.write("각 시간대 신호:", agg_signals)
