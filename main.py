import os
import sqlite3
import time
import logging
from datetime import datetime
import yfinance as yf
import pandas as pd
import numpy as np
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, CallbackQueryHandler, ContextTypes

# --- LOGLAMA AYARLARI ---
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# --- VERİTABANI YÖNETİCİSİ ---
class DatabaseManager:
    def __init__(self, db_name="trading_bot_virtual.db"):
        self.db_name = db_name
        self._init_db()

    def _get_connection(self):
        return sqlite3.connect(self.db_name)

    def _init_db(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.executescript("""
                CREATE TABLE IF NOT EXISTS assets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, 
                    category TEXT, 
                    symbol TEXT UNIQUE,
                    is_manual INTEGER DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, 
                    symbol TEXT, 
                    action TEXT, 
                    entry_price REAL, 
                    exit_price REAL,
                    pnl REAL, 
                    status TEXT, 
                    error_reason TEXT, 
                    pyramid_level INTEGER, 
                    timestamp DATETIME, 
                    rsi_at_entry REAL
                );
                CREATE TABLE IF NOT EXISTS portfolio (
                    id INTEGER PRIMARY KEY CHECK (id = 1), 
                    balance REAL
                );
                INSERT OR IGNORE INTO portfolio (id, balance) VALUES (1, 10000.0);
            """)
            
            # --- GENİŞ OTOMATİK PİYASA HAVUZU + MANUEL LİSTE ---
            default_assets = [
                # Kriptolar
                ("Kripto", "BTC/USDT", 0),
                ("Kripto", "ETH/USDT", 0),
                ("Kripto", "SOL/USDT", 0),
                ("Kripto", "AVAX/USDT", 0),
                ("Kripto", "BNB/USDT", 0),
                ("Kripto", "XRP/USDT", 0),
                # BIST Hisseleri
                ("BIST", "THYAO.IS", 0),
                ("BIST", "EREGL.IS", 0),
                ("BIST", "GARAN.IS", 0),
                ("BIST", "ASELS.IS", 0),
                ("BIST", "KRDMD.IS", 0),
                # NASDAQ Hisseleri
                ("NASDAQ", "AAPL", 0),
                ("NASDAQ", "TSLA", 0),
                ("NASDAQ", "NVDA", 0),
                ("NASDAQ", "MSFT", 0)
            ]
            cursor.executemany(
                "INSERT OR IGNORE INTO assets (category, symbol, is_manual) VALUES (?, ?, ?)", 
                default_assets
            )
            conn.commit()

    def get_all_assets(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT category, symbol, is_manual FROM assets")
            return cursor.fetchall()

    def add_asset(self, category, symbol):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR IGNORE INTO assets (category, symbol, is_manual) VALUES (?, ?, 1)", 
                (category, symbol)
            )
            conn.commit()

# --- TEKNİK ANALİZ VE PİNE SCRIPT STRATEJİ MOTORU ---
class StrategyAnalyzer:
    def __init__(self, rsi_length=14, macd_fast=12, macd_slow=26, macd_signal=9):
        self.rsi_length = rsi_length
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal

    def fetch_data(self, symbol, interval="1d", period="60d"):
        try:
            # Yahoo Finance sembol formatı dönüşümü
            yf_symbol = symbol
            if "/" in symbol: # Örn: BTC/USDT -> BTC-USD
                parts = symbol.split("/")
                yf_symbol = f"{parts[0]}-{parts[1].replace('USDT', 'USD')}"
            
            df = yf.download(yf_symbol, period=period, interval=interval, progress=False)
            if df.empty or len(df) < 30:
                return None
            
            # Çoklu sütun düzeltmesi (yfinance güncellemeleri için önlem)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
                
            return df
        except Exception as e:
            logger.error(f"{symbol} veri çekme hatası: {e}")
            return None

    def calculate_indicators(self, df):
        # RSI Hesaplama
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_length).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_length).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # MACD Hesaplama
        exp1 = df['Close'].ewm(span=self.macd_fast, adjust=False).mean()
        exp2 = df['Close'].ewm(span=self.macd_slow, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=self.macd_signal, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

        # Yapı Kırılımı (BOS / CHoCH Simülasyonu - Pivot bazlı)
        df['Highs_5'] = df['High'].rolling(window=5).max()
        df['Lows_5'] = df['Low'].rolling(window=5).min()
        
        return df

    def analyze_asset(self, symbol):
        df = self.fetch_data(symbol)
        if df is None:
            return None, "Veri Alınamadı"

        df = self.calculate_indicators(df)
        last_row = df.iloc[-1]
        prev_row = df.iloc[-2]

        close = float(last_row['Close'])
        rsi = float(last_row['RSI'])
        macd = float(last_row['MACD'])
        macd_sig = float(last_row['MACD_Signal'])
        prev_high = float(prev_row['Highs_5'])
        prev_low = float(prev_row['Lows_5'])

        # Pine Script Strateji Kuralları:
        # Long Koşulu: Fiyat son tepeyi yukarı kesti (BOS) ve RSI < 60 ile MACD sinyal üstünde
        long_condition = (close > prev_high) and (rsi < 60 and macd > macd_sig)
        
        # Short Koşulu: Fiyat son dip seviyeyi aşağı kesti ve RSI > 40 ile MACD sinyal altında
        short_condition = (close < prev_low) and (rsi > 40 and macd < macd_sig)

        signal = "NEUTRAL"
        comment = f"Fiyat: {close:.2f} | RSI: {rsi:.1f} | MACD dengeli, nötr seyir."

        if long_condition:
            signal = "LONG"
            comment = f"🚀 YÜKSELİŞ (LONG) FIRSATI!\n• Fiyat son direnci kırdı (BOS).\n• RSI ({rsi:.1f}) uygun bölgede.\n• MACD pozitif kesişimde."
        elif short_condition:
            signal = "SHORT"
            comment = f"📉 DÜŞÜŞ (SHORT) FIRSATI!\n• Fiyat destek seviyesini kırdı.\n• RSI ({rsi:.1f}) dönüş sinyalinde.\n• MACD negatif kesişimde."

        return {
            "symbol": symbol,
            "signal": signal,
            "price": close,
            "rsi": rsi,
            "comment": comment
        }, "Başarılı"

# --- TELEGRAM BOT KONTROLCÜSÜ ---
class TradingBotApp:
    def __init__(self, token):
        self.token = token
        self.db = DatabaseManager()
        self.analyzer = StrategyAnalyzer()

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        keyboard = [
            [InlineKeyboardButton("🔍 Tüm Piyasayı Tara (Scan)", callback_data="scan_market")],
            [InlineKeyboardButton("➕ Varlık Ekle", callback_data="help_add")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_text(
            "🤖 **Pine Script Entegreli Akıllı Trading Botuna Hoş Geldiniz!**\n\n"
            "Bot, otomatik ve manuel listelerinizdeki tüm varlıkları analiz ederek düşüşe/yükselişe geçenleri tarar.",
            reply_markup=reply_markup,
            parse_mode="Markdown"
        )

    async def scan_market(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        await query.answer()
        await query.edit_message_text("🔍 **Piyasa taranıyor, göstergeler hesaplanıyor...** Lütfen bekleyin.")

        assets = self.db.get_all_assets()
        opportunities = []

        for category, symbol, is_manual in assets:
            result, status = self.analyzer.analyze_asset(symbol)
            if result and result["signal"] != "NEUTRAL":
                opportunities.append(result)
            time.sleep(0.5) # API limit koruması

        if not opportunities:
            await query.message.reply_text("⚠️ Şu an için strateji kriterlerine uyan aktif bir alım/satım fırsatı bulunamadı. Piyasalar yatay seyrediyor.")
            return

        report = "🚨 **PİYASA TARAMA SONUÇLARI - FIRSATLAR** 🚨\n\n"
        for opp in opportunities:
            report += f"📌 **{opp['symbol']}**\n{opp['comment']}\n-----------------------------------\n"

        await query.message.reply_text(report, parse_mode="Markdown")

    async def button_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        if query.data == "scan_market":
            await self.scan_market(update, context)
        elif query.data == "help_add":
            await query.message.reply_text(
                "💡 Varlık eklemek için kod içerisindeki `default_assets` listesine ekleme yapabilir ya da GitHub üzerinden `main.py` dosyanızı güncelleyebilirsiniz."
            )

    def run(self):
        app = ApplicationBuilder().token(self.token).build()
        app.add_handler(CommandHandler("start", self.start))
        app.add_handler(CallbackQueryHandler(self.button_handler))
        
        logger.info("Bot çalıştırılıyor...")
        app.run_polling()

if __name__ == "__main__":
    # Telegram Bot Token'ınızı buraya yazın veya çevre değişkeni (Environment Variable) olarak tanımlayın
    TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "BURAYA_TELEGRAM_BOT_TOKEN_YAZIN")
    
    if TOKEN == "BURAYA_TELEGRAM_BOT_TOKEN_YAZIN":
        print("Lütfen geçerli bir Telegram Bot Token girin!")
    else:
        bot = TradingBotApp(TOKEN)
        bot.run()
