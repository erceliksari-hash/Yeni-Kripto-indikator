import streamlit as st
import pandas as pd
import sqlite3
import time
import yfinance as yf

# Sayfa Yapılandırması
st.set_page_config(
    page_title="Akıllı Trading Botu & Piyasa Tarayıcısı",
    page_icon="📈",
    layout="wide"
)

# Veritabanı Bağlantısı
DB_NAME = "trading_bot_virtual.db"

def get_connection():
    return sqlite3.connect(DB_NAME, check_same_thread=False)

def init_db():
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.executescript("""
            CREATE TABLE IF NOT EXISTS assets (
                id INTEGER PRIMARY KEY AUTOINCREMENT, 
                category TEXT, 
                symbol TEXT UNIQUE,
                is_manual INTEGER DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS portfolio (
                id INTEGER PRIMARY KEY CHECK (id = 1), 
                balance REAL
            );
            INSERT OR IGNORE INTO portfolio (id, balance) VALUES (1, 10000.0);
        """)
        # Varsayılan Varlıklar
        default_assets = [
            ("Kripto", "BTC/USDT", 0), ("Kripto", "ETH/USDT", 0),
            ("Kripto", "SOL/USDT", 0), ("Kripto", "AVAX/USDT", 0),
            ("BIST", "THYAO.IS", 0), ("BIST", "EREGL.IS", 0),
            ("NASDAQ", "AAPL", 0), ("NASDAQ", "TSLA", 0)
        ]
        cursor.executemany("INSERT OR IGNORE INTO assets (category, symbol, is_manual) VALUES (?, ?, ?)", default_assets)
        conn.commit()

init_db()

# --- ANALİZ MOTORU ---
def analyze_market_asset(symbol):
    try:
        yf_symbol = symbol
        if "/" in symbol:
            parts = symbol.split("/")
            yf_symbol = f"{parts[0]}-{parts[1].replace('USDT', 'USD')}"
        
        df = yf.download(yf_symbol, period="60d", interval="1d", progress=False)
        if df.empty or len(df) < 30:
            return None
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        # Göstergeler (RSI & MACD)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        df['RSI'] = 100 - (100 / (1 + (gain / loss)))

        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

        df['Highs_5'] = df['High'].rolling(window=5).max()
        df['Lows_5'] = df['Low'].rolling(window=5).min()

        last = df.iloc[-1]
        prev = df.iloc[-2]

        close = float(last['Close'])
        rsi = float(last['RSI'])
        macd = float(last['MACD'])
        macd_sig = float(last['MACD_Signal'])
        prev_high = float(prev['Highs_5'])
        prev_low = float(prev['Lows_5'])

        signal = "NEUTRAL"
        if close > prev_high and rsi < 60 and macd > macd_sig:
            signal = "LONG"
        elif close < prev_low and rsi > 40 and macd < macd_sig:
            signal = "SHORT"

        return {
            "symbol": symbol,
            "price": close,
            "rsi": rsi,
            "signal": signal
        }
    except Exception:
        return None

# --- STREAMLIT ARAYÜZÜ ---
st.title("🤖 Akıllı Trading Botu & Piyasa Tarayıcı Panosu")
st.markdown("Pine Script strateji kurallarına (`RSI`, `MACD`, `BOS/CHoCH`) göre tüm piyasaları tarayın ve fırsatları yakalayın.")

# Kenar Çubuğu (Sidebar) - Portföy ve Varlık Yönetimi
st.sidebar.header("💼 Sanal Kasa & Ayarlar")
with get_connection() as conn:
    df_portfolio = pd.read_sql("SELECT balance FROM portfolio WHERE id = 1", conn)
    balance = df_portfolio['balance'].iloc[0]

st.sidebar.metric(label="Toplam Sanal Bakiye", value=f"${balance:,.2f}")

st.sidebar.divider()
st.sidebar.subheader("➕ Manuel Varlık Ekle")
new_cat = st.sidebar.selectbox("Kategori", ["Kripto", "BIST", "NASDAQ"])
new_sym = st.sidebar.text_input("Sembol (Örn: SOL/USDT veya KRDMD.IS)")

if st.sidebar.button("Listeye Ekle"):
    if new_sym:
        with get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute("INSERT INTO assets (category, symbol, is_manual) VALUES (?, ?, 1)", (new_cat, new_sym.upper()))
                conn.commit()
                st.sidebar.success(f"{new_sym.upper()} başarıyla eklendi!")
            except sqlite3.IntegrityError:
                st.sidebar.warning("Bu varlık zaten listede mevcut.")

# Ana Ekran - Varlık Listesi ve Tarama
st.subheader("📋 Takip Edilen Varlık Havuzu")
with get_connection() as conn:
    df_assets = pd.read_sql("SELECT category, symbol, is_manual FROM assets", conn)

st.dataframe(df_assets, use_container_width=True)

st.divider()

if st.button("🚀 Piyasaları Şimdi Tara (Scan Market)", type="primary"):
    with st.spinner("Tüm varlıklar analiz ediliyor, göstergeler hesaplanıyor... Lütfen bekleyin."):
        assets = df_assets['symbol'].tolist()
        opportunities = []
        
        progress_bar = st.progress(0)
        for i, sym in enumerate(assets):
            res = analyze_market_asset(sym)
            if res and res["signal"] != "NEUTRAL":
                opportunities.append(res)
            progress_bar.progress((i + 1) / len(assets))
            time.sleep(0.2)
            
        st.success("Tarama tamamlandı!")
        
        if opportunities:
            st.markdown("### 🚨 Tespit Edilen Aktif Fırsatlar")
            for opp in opportunities:
                color = "green" if opp["signal"] == "LONG" else "red"
                st.markdown(f"""
                <div style="padding: 15px; border-radius: 5px; border: 1px solid #ddd; margin-bottom: 10px;">
                    <h4>📌 {opp['symbol']} — <span style="color: {color};">{opp['signal']}</span></h4>
                    <p><b>Fiyat:</b> {opp['price']:.2f} | <b>RSI:</b> {opp['rsi']:.1f}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Şu an strateji kriterlerine uyan aktif bir sinyal bulunamadı. Piyasalar yatay seyrediyor.")
