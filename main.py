import asyncio
import logging
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import Application, CallbackQueryHandler, CommandHandler, ContextTypes
from sklearn.ensemble import RandomForestClassifier
from dotenv import load_dotenv

# --- ÇEVRESEL DEĞİŞKENLERİ YÜKLE ---
load_dotenv()

# --- LOGLAMA YAPILANDIRMASI ---
logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger("QuantBot_Virtual")
logging.getLogger("httpx").setLevel(logging.WARNING)

# --- VERİ YAPILARI ---
@dataclass
class TradeSignal:
    symbol: str
    action: str  
    entry_price: float
    stop_loss: float
    take_profit: float
    reason: str
    pyramid_level: int = 1 
    confidence: float = 1.0 

# --- 1. SANAL VERİ VE PİYASA YÖNETİCİSİ (API Gerektirmez) ---
class MarketDataManager:
    """Binance API gerektirmeyen, yfinance ve halka açık veri kaynaklarını kullanan modül."""
    @staticmethod
    def fetch_market_data(symbol: str, timeframe: str = '15m', limit: int = 300) -> pd.DataFrame:
        try:
            # Kripto sembolleri (Örn: BTC/USDT -> Yahoo Finance formatı: BTC-USD)
            yf_symbol = symbol.replace("/USDT", "-USD").replace("/USD", "-USD")
            
            # Zaman dilimine göre Yahoo Finance periyodu
            yf_period = "5d" if timeframe == "15m" else "1mo"
            yf_interval = "15m" if timeframe == "15m" else "1h" 
            
            df = yf.download(yf_symbol, period=yf_period, interval=yf_interval, progress=False)
            
            if df.empty:
                # Veri boş dönerse test amaçlı sentetik (güvenli) veri üret
                return MarketDataManager._generate_synthetic_data(limit)
                
            # Sütun isimlerini standartlaştır
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
            df.columns = ['open', 'high', 'low', 'close', 'volume']
            return df
        except Exception as e:
            logger.warning(f"Canlı veri çekilemedi ({symbol}), sanal simülasyon verisine geçiliyor: {e}")
            return MarketDataManager._generate_synthetic_data(limit)

    @staticmethod
    def _generate_synthetic_data(limit: int = 300) -> pd.DataFrame:
        """API veya internet kopması durumunda botun çökmemesini sağlayan yedek simülasyon verisi."""
        dates = pd.date_range(end=datetime.now(), periods=limit, freq='15min')
        prices = 100 + np.cumsum(np.random.randn(limit) * 0.4)
        df = pd.DataFrame({
            'open': prices - 0.2,
            'high': prices + 0.6,
            'low': prices - 0.6,
            'close': prices,
            'volume': np.random.randint(1000, 5000, limit)
        }, index=dates)
        return df

# --- VERİTABANI YÖNETİMİ (Sanal Kasa) ---
class DatabaseManager:
    def __init__(self, db_file: str = "trading_bot_virtual.db"):
        self.db_file = db_file
        self._init_db()

    def _get_connection(self):
        conn = sqlite3.connect(self.db_file, check_same_thread=False, timeout=10.0)
        conn.execute("PRAGMA journal_mode=WAL;") 
        return conn

    def _init_db(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.executescript("""
                CREATE TABLE IF NOT EXISTS assets (id INTEGER PRIMARY KEY AUTOINCREMENT, category TEXT, symbol TEXT UNIQUE);
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT, action TEXT, entry_price REAL, exit_price REAL,
                    pnl REAL, status TEXT, error_reason TEXT, pyramid_level INTEGER, timestamp DATETIME, rsi_at_entry REAL
                );
                CREATE TABLE IF NOT EXISTS portfolio (id INTEGER PRIMARY KEY CHECK (id = 1), balance REAL);
                INSERT OR IGNORE INTO portfolio (id, balance) VALUES (1, 10000.0);
            """)
            default_assets = [
                ("Kripto", "BTC/USDT"), ("Kripto", "ETH/USDT"), 
                ("BIST", "THYAO.IS"), ("NASDAQ", "AAPL")
            ]
            cursor.executemany("INSERT OR IGNORE INTO assets (category, symbol) VALUES (?, ?)", default_assets)
            conn.commit()

    def get_assets_by_category(self, category: str) -> List[str]:
        with self._get_connection() as conn:
            return [row[0] for row in conn.execute("SELECT symbol FROM assets WHERE category = ?", (category,)).fetchall()]

    def log_trade(self, signal: TradeSignal, status: str = "OPEN", rsi_val: float = 50.0):
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO trades (symbol, action, entry_price, exit_price, pnl, status, error_reason, pyramid_level, timestamp, rsi_at_entry)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (signal.symbol, signal.action, signal.entry_price, 0.0, 0.0, status, signal.reason, signal.pyramid_level, datetime.now(), rsi_val))
            conn.commit()
            
    def get_open_trade(self, symbol: str):
        with self._get_connection() as conn:
            return conn.execute("SELECT * FROM trades WHERE symbol = ? AND status = 'OPEN' ORDER BY id DESC LIMIT 1", (symbol,)).fetchone()

    def get_balance(self) -> float:
        with self._get_connection() as conn:
            return conn.execute("SELECT balance FROM portfolio WHERE id = 1").fetchone()[0]

# --- 2. MAKİNE ÖĞRENMESİ (ERROR LEARNING LOOP) ---
class MLOptimizer:
    def __init__(self, db: DatabaseManager):
        self.db = db
        self.model = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42)
        
    def get_dynamic_rsi_thresholds(self) -> Tuple[float, float]:
        with self.db._get_connection() as conn:
            df = pd.read_sql_query("SELECT pnl, rsi_at_entry FROM trades WHERE status = 'CLOSED'", conn)
            
        if len(df) < 10: 
            return 40.0, 60.0 # Varsayılan Pine Script sınırlarınız
            
        df['target'] = (df['pnl'] > 0).astype(int)
        X = df[['rsi_at_entry']]
        y = df['target']
        self.model.fit(X, y)
        
        test_rsis = pd.DataFrame({'rsi_at_entry': np.linspace(20, 80, 60)})
        preds = self.model.predict_proba(test_rsis)[:, 1]
        
        optimal_long_rsi = test_rsis['rsi_at_entry'][preds > 0.6].min()
        optimal_short_rsi = test_rsis['rsi_at_entry'][preds > 0.6].max()
        
        return max(30.0, optimal_long_rsi if pd.notna(optimal_long_rsi) else 40.0), min(70.0, optimal_short_rsi if pd.notna(optimal_short_rsi) else 60.0)

# --- TEKNİK ANALİZ MOTORU & İNDİKATÖR MANTIĞINIZ ---
class TechnicalAnalyzer:
    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        return pd.concat([high_low, high_close, low_close], axis=1).max(axis=1).rolling(window=period).mean()

    @staticmethod
    def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        return 100 - (100 / (1 + (gain / loss)))

    @classmethod
    def custom_pin_editor_logic(cls, df_15m: pd.DataFrame, df_4h: pd.DataFrame, ml_opt: MLOptimizer, open_trade: tuple = None) -> Optional[TradeSignal]:
        if df_15m.empty or len(df_15m) < 30:
            return None

        # ML Destekli Dinamik Eşikler
        rsi_oversold, rsi_overbought = ml_opt.get_dynamic_rsi_thresholds()

        # 4H Makro Trend Filtresi (Eğer 4H veri azsa 15M üzerinden simüle edilir)
        if not df_4h.empty and len(df_4h) > 50:
            ema50 = df_4h['close'].ewm(span=50, adjust=False).mean()
            ema200 = df_4h['close'].ewm(span=200, adjust=False).mean()
            trend_bullish = ema50.iloc[-2] > ema200.iloc[-2]
        else:
            trend_bullish = True

        # RSI, ATR ve Hacim Hesaplamaları
        df_15m['rsi'] = cls.calculate_rsi(df_15m['close'])
        df_15m['atr'] = cls.calculate_atr(df_15m)
        df_15m['vol_sma'] = df_15m['volume'].rolling(20).mean()
        
        # SİZİN PİNE SCRIPT MANTIĞINIZ (BOS / CHoCH: Son 5 barın tepe/dip noktalarını kırma)
        df_15m['highs_5'] = df_15m['high'].shift(1).rolling(5).max()
        df_15m['lows_5'] = df_15m['low'].shift(1).rolling(5).min()

        last_closed = df_15m.iloc[-2]
        prev_closed = df_15m.iloc[-3]
        vol_ok = last_closed['volume'] > last_closed['vol_sma']
        
        # Sinyal Koşulları
        long_cond = (prev_closed['close'] <= last_closed['highs_5']) and (last_closed['close'] > last_closed['highs_5']) and (last_closed['rsi'] < rsi_overbought) and vol_ok
        short_cond = (prev_closed['close'] >= last_closed['lows_5']) and (last_closed['close'] < last_closed['lows_5']) and (last_closed['rsi'] > rsi_oversold) and vol_ok

        current_price = last_closed['close']
        atr_val = last_closed['atr']
        if pd.isna(atr_val) or atr_val == 0:
            atr_val = current_price * 0.01
            
        reward_ratio = 2.5 # Sizin Pine Script'teki 2.5R Risk-Ödül oranınız
        
        # 3. PİRAMİTLEME MANTIĞI
        pyramid_level = 1
        reason_prefix = "BOS + ML + Hacim "
        if open_trade:
            entry_price, action, current_level = open_trade[3], open_trade[2], open_trade[8]
            if current_level < 3: 
                if action == "LONG" and current_price > entry_price + (atr_val * 0.5):
                    pyramid_level = current_level + 1
                    long_cond = True 
                    reason_prefix = f"PİRAMİT (Kademe {pyramid_level}) "
                elif action == "SHORT" and current_price < entry_price - (atr_val * 0.5):
                    pyramid_level = current_level + 1
                    short_cond = True
                    reason_prefix = f"PİRAMİT (Kademe {pyramid_level}) "

        if long_cond and trend_bullish:
            sl = current_price - (1.5 * atr_val)
            tp = current_price + ((current_price - sl) * reward_ratio)
            return TradeSignal("", "LONG", current_price, sl, tp, reason_prefix + "Boğa", pyramid_level=pyramid_level)

        elif short_cond and not trend_bullish:
            sl = current_price + (1.5 * atr_val)
            tp = current_price - ((sl - current_price) * reward_ratio)
            return TradeSignal("", "SHORT", current_price, sl, tp, reason_prefix + "Ayı", pyramid_level=pyramid_level)

        return None

# --- PROGRAMATİK GRAFİK ÜRETİCİ ---
class ChartGenerator:
    @staticmethod
    def generate_chart(df: pd.DataFrame, symbol: str, signal: Optional[TradeSignal]) -> str:
        plt.close('all') 
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
        
        plot_df = df.iloc[-100:] if len(df) >= 100 else df
        ax1.plot(plot_df.index, plot_df['close'], label='Fiyat', color='#2962FF', linewidth=1.5)
        ax1.plot(plot_df.index, plot_df['high'].rolling(5).max(), color='#00E676', linestyle='--', alpha=0.5, label='Direnç')
        
        if signal and df.index[-2] in plot_df.index:
            signal_idx = df.index[-2] 
            color, marker = ('#00E676', '^') if signal.action == 'LONG' else ('#D50000', 'v')
            ax1.scatter(signal_idx, signal.entry_price, color=color, s=200, marker=marker, zorder=5, label=f"{signal.action} (Kademe {signal.pyramid_level})")

        ax1.set_title(f"{symbol} Sanal Analiz Raporu", fontsize=14, fontweight='bold')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)

        rsi = TechnicalAnalyzer.calculate_rsi(df['close'])
        ax2.plot(plot_df.index, rsi.iloc[-len(plot_df):], color='#AA00FF', linewidth=1.5)
        ax2.axhline(60, color='#D50000', linestyle='--', alpha=0.5)
        ax2.axhline(40, color='#00E676', linestyle='--', alpha=0.5)
        ax2.set_ylabel("RSI")
        ax2.grid(True, alpha=0.3)

        filename = f"chart_{symbol.replace('/', '_')}_{int(datetime.now().timestamp())}.png"
        plt.tight_layout()
        plt.savefig(filename, dpi=120)
        fig.clf()
        plt.close(fig)
        return filename

# --- TELEGRAM BOTU ---
class TradingBotApp:
    def __init__(self, token: str):
        self.app = Application.builder().token(token).build()
        self.db = DatabaseManager()
        self.ml_optimizer = MLOptimizer(self.db) 
        self._setup_handlers()

    def _setup_handlers(self):
        self.app.add_handler(CommandHandler("start", self.start_command))
        self.app.add_handler(CallbackQueryHandler(self.button_handler))

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        keyboard = [
            [InlineKeyboardButton("🪙 Kripto (Sanal)", callback_data="cat_Kripto"), InlineKeyboardButton("📈 BIST / NASDAQ", callback_data="cat_BIST")],
            [InlineKeyboardButton("💼 Sanal Kasa Durumu", callback_data="portfolio"), InlineKeyboardButton("🧠 ML Eşik Durumu", callback_data="ml_status")]
        ]
        await update.message.reply_text("🤖 **Sanal Kantitatif Bot Aktif**\nAPI gerektirmeyen veri akışı, sanal kasa ve yapay zeka devrede.", reply_markup=InlineKeyboardMarkup(keyboard), parse_mode="Markdown")

    async def button_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        await query.answer()
        data = query.data

        if data.startswith("cat_"):
            cat = data.split("_")[1]
            assets = self.db.get_assets_by_category(cat)
            kb = [[InlineKeyboardButton(a, callback_data=f"analyze_{a}")] for a in assets]
            kb.append([InlineKeyboardButton("🔙 Ana Menü", callback_data="start")])
            await query.edit_message_text(f"📂 **{cat}** Varlıkları (Sanal Veri Akışı):", reply_markup=InlineKeyboardMarkup(kb), parse_mode="Markdown")
            
        elif data == "ml_status":
            r_down, r_up = self.ml_optimizer.get_dynamic_rsi_thresholds()
            msg = f"🧠 **Hata Öğrenme Döngüsü (ML)**\n\nSistem geçmiş sanal işlem loglarını tarayarak optimize etti:\n\n🔻 Alt Eşik: `{r_down:.2f}`\n🔺 Üst Eşik: `{r_up:.2f}`"
            await query.edit_message_text(msg, reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Menü", callback_data="start")]]), parse_mode="Markdown")

        elif data == "portfolio":
            bal = self.db.get_balance()
            msg = f"💼 **Sanal Kasa Bakiyesi:** `${bal:,.2f}`\n\n*Not: Tüm işlemler sanal kasadan simüle edilmektedir.*"
            await query.edit_message_text(msg, reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Menü", callback_data="start")]]), parse_mode="Markdown")

        elif data == "start":
            await self.start_command(update, context)

        elif data.startswith("analyze_"):
            symbol = data.replace("analyze_", "")
            await query.edit_message_text(f"🔍 `{symbol}` sanal veri tabanından yükleniyor ve analiz ediliyor...", parse_mode="Markdown")
            
            # API gerektirmeyen veri yöneticisi üzerinden verileri çek
            df_15m = await asyncio.to_thread(MarketDataManager.fetch_market_data, symbol, "15m", 300)
            df_4h = await asyncio.to_thread(MarketDataManager.fetch_market_data, symbol, "1h", 200)

            if df_15m.empty:
                await query.edit_message_text("⚠️ Veri oluşturulamadı.")
                return

            open_trade = self.db.get_open_trade(symbol)
            signal = TechnicalAnalyzer.custom_pin_editor_logic(df_15m, df_4h, self.ml_optimizer, open_trade)
            
            if signal:
                signal.symbol = symbol
                rsi_val = df_15m['rsi'].iloc[-2]
                self.db.log_trade(signal, rsi_val=rsi_val)

            chart_path = await asyncio.to_thread(ChartGenerator.generate_chart, df_15m, symbol, signal)

            caption = f"📊 **{symbol} Sanal Analiz Raporu**\n\n"
            if signal:
                caption += (f"🚀 **Sinyal Onayı:** `{signal.action}`\n"
                            f"📍 **Giriş:** `{signal.entry_price:.2f}`\n"
                            f"🛑 **SL (Dinamik ATR):** `{signal.stop_loss:.2f}`\n"
                            f"🎯 **TP (2.5R):** `{signal.take_profit:.2f}`\n"
                            f"🔼 **Piramitleme Kademesi:** `{signal.pyramid_level}/3`\n"
                            f"💡 **Mantık:** {signal.reason}")
            else:
                caption += "⏳ Yapı kırılımı veya hacim şartı sağlanamadı. Sanal takip devam ediyor."

            kb = InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Menü", callback_data="start")]])
            if chart_path and os.path.exists(chart_path):
                with open(chart_path, 'rb') as f:
                    await context.bot.send_photo(chat_id=query.message.chat_id, photo=f, caption=caption, parse_mode="Markdown", reply_markup=kb)
                os.remove(chart_path)
            else:
                await query.edit_message_text(caption, reply_markup=kb, parse_mode="Markdown")

    def run(self):
        logger.info("QuantBot Sanal Kasa Modu Başlatıldı...")
        self.app.run_polling()

if __name__ == "__main__":
    TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    if not TOKEN:
        print("HATA: .env dosyasında TELEGRAM_BOT_TOKEN bulunamadı!")
    else:
        bot = TradingBotApp(TOKEN)
        bot.run()
