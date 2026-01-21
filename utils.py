# -*- coding: utf-8 -*-
"""
utils.py - Funzioni Helper per SPX Dashboard
Kriterion Quant - Sistema V4.0 (FIXED & ALIGNED to Notebook)
"""

import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
from datetime import datetime, timedelta
import streamlit as st
from config import *

# ==============================================================================
# FUNZIONI DOWNLOAD DATI
# ==============================================================================

@st.cache_data(ttl=3600)  # Cache per 1 ora
def load_ticker_data(ticker, start_date=START_DATE):
    """
    Carica dati storici con cache Streamlit
    """
    try:
        # Notebook usa auto_adjust=False di default ma cerca Adj Close manualment
        df = yf.download(ticker, start=start_date, end=datetime.now(), progress=False)
        
        if df.empty:
            st.warning(f"⚠️ Nessun dato disponibile per {ticker}")
            return pd.DataFrame()
        
        # Assicura DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        return df
        
    except Exception as e:
        st.error(f"❌ Errore download {ticker}: {e}")
        return pd.DataFrame()

def load_all_indices(indices_dict):
    """
    Carica tutti gli indici definiti in INDICES
    FIX: Priorità a Adj Close e gestione dimensionalità (1D array)
    """
    raw_data = {}
    
    with st.spinner('📥 Download dati indici globali...'):
        progress_bar = st.progress(0)
        total = len(indices_dict)
        
        for idx, (name, ticker) in enumerate(indices_dict.items()):
            df = load_ticker_data(ticker)
            if not df.empty:
                raw_data[name] = df
            progress_bar.progress((idx + 1) / total)
    
    # Allinea date (Inner Join come nel Notebook)
    common_dates = None
    for df in raw_data.values():
        if common_dates is None:
            common_dates = df.index
        else:
            common_dates = common_dates.intersection(df.index)
    
    if len(common_dates) == 0:
        raise ValueError("Nessuna data comune tra gli indici! Verifica i dati.")
    
    # --- HELPER PER ESTRARRE SERIE 1D ---
    def _get_series(source_df, col_priority, dates):
        """
        Estrae una colonna appiattita (1D).
        Cerca in ordine di priorità (es: Adj Close > Close)
        """
        selected_col = None
        # Gestione MultiLevel Columns di yfinance
        cols = source_df.columns
        if isinstance(cols, pd.MultiIndex):
             # Appiattisce se necessario o cerca nel livello corretto
             pass 

        # Logica priorità (come nel Notebook Cell 2)
        for col in col_priority:
            if col in source_df.columns:
                data = source_df.loc[dates, col]
                # Fix dimensionalità: se è DataFrame (N,1), prendi la Series
                if isinstance(data, pd.DataFrame):
                    return data.iloc[:, 0]
                return data
        
        # Fallback se nessuna colonna trovata
        raise ValueError(f"Colonne {col_priority} non trovate nel DataFrame")

    # Crea DataFrame allineati
    # NOTEBOOK ALIGNMENT: Priorità 'Adj Close' > 'Close'
    df_close = pd.DataFrame(
        {name: _get_series(df, ['Adj Close', 'Close'], common_dates) for name, df in raw_data.items()},
        index=common_dates
    )
    
    df_high = pd.DataFrame(
        {name: _get_series(df, ['High'], common_dates) for name, df in raw_data.items()},
        index=common_dates
    )
    
    df_low = pd.DataFrame(
        {name: _get_series(df, ['Low'], common_dates) for name, df in raw_data.items()},
        index=common_dates
    )
    
    # NOTEBOOK ALIGNMENT: Fillna method (anche se dropna sopra dovrebbe aver pulito)
    df_close = df_close.ffill().bfill()
    df_high = df_high.ffill().bfill()
    df_low = df_low.ffill().bfill()

    return raw_data, df_close, df_high, df_low

# ==============================================================================
# FUNZIONI CALCOLO MARKET BREADTH
# ==============================================================================

def calculate_market_breadth(df_close, df_high, df_low):
    """
    Calcola i 3 indicatori di Market Breadth
    """
    
    # 1. % Indici sopra MA125
    # Pandas rolling mean gestisce nativamente le Series
    ma_125 = df_close.rolling(window=BreadthSettings.MA_PERIOD).mean()
    above_ma = (df_close > ma_125).sum(axis=1) / df_close.shape[1]
    
    # 2. RSI Medio
    rsi_values = pd.DataFrame()
    for col in df_close.columns:
        # Assicura che l'input per pandas_ta sia una Series 1D
        series = df_close[col]
        rsi = ta.rsi(series, length=BreadthSettings.RSI_PERIOD)
        rsi_values[col] = rsi
    avg_rsi = rsi_values.mean(axis=1)
    
    # 3. Correlazione Rolling 30D
    returns = df_close.pct_change()
    
    def avg_correlation_robust(window_returns):
        # Replica esatta logica Notebook Cell 4
        valid_assets = window_returns.loc[:, window_returns.var() > 0]
        if valid_assets.shape[1] < 2:
            return np.nan
        try:
            corr_matrix = valid_assets.corr()
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
            upper_tri_values = corr_matrix.where(mask).stack().values
            if len(upper_tri_values) == 0:
                return np.nan
            return np.nanmean(upper_tri_values)
        except:
            return np.nan
    
    rolling_corr = []
    for i in range(len(returns)):
        # Replica loop Notebook: finestra i-29 a i+1 (esclusivo) = 30 giorni
        if i < BreadthSettings.CORR_WINDOW - 1:
            rolling_corr.append(np.nan)
        else:
            window = returns.iloc[i-BreadthSettings.CORR_WINDOW+1:i+1]
            avg_corr = avg_correlation_robust(window)
            rolling_corr.append(avg_corr)
    
    # Crea DataFrame
    breadth_df = pd.DataFrame({
        'Breadth_Pct_Above_MA125': above_ma,
        'Breadth_Avg_RSI_20': avg_rsi,
        'Breadth_Avg_Corr_30D': pd.Series(rolling_corr, index=df_close.index)
    })
    
    # Fix NaN correlazione come da Notebook Cell 4
    breadth_df['Breadth_Avg_Corr_30D'] = breadth_df['Breadth_Avg_Corr_30D'].ffill().fillna(0.5)
    
    # Rimuovi NaN iniziali (warm-up period)
    breadth_df = breadth_df.dropna()
    
    return breadth_df

# ==============================================================================
# FUNZIONI EVENTI TARGET
# ==============================================================================

def calculate_target_events(spx_prices):
    """
    Identifica eventi BOTTOM e TOP storici su S&P 500
    FIX: Corretto errore off-by-one rispetto al Notebook
    """
    target_bottom = []
    target_top = []
    
    # Forza spx_prices a Series 1D se non lo è
    if isinstance(spx_prices, pd.DataFrame):
        spx_prices = spx_prices.iloc[:, 0]

    for i in range(len(spx_prices)):
        current_price = spx_prices.iloc[i]
        
        # NOTEBOOK ALIGNMENT FIX:
        # Notebook Cell 5: future_window = spx_prices.iloc[i+1:end_idx]
        # Repository originale includeva 'i', causando look-ahead bias
        
        end_idx = min(i + TargetEvents.FORWARD_WINDOW + 1, len(spx_prices))
        
        # Finestra futura ESCLUDE oggi (i+1)
        if i + 1 >= len(spx_prices):
            target_bottom.append(0)
            target_top.append(0)
            continue
            
        forward_prices = spx_prices.iloc[i+1:end_idx]
        
        if len(forward_prices) < TargetEvents.FORWARD_WINDOW:
            # Non abbastanza dati futuri
            target_bottom.append(0)
            target_top.append(0)
            continue
        
        # BOTTOM: Rally >= 20%
        # Calcolo rispetto al minimo futuro come nel notebook
        min_forward = forward_prices.min()
        max_forward = forward_prices.max()
        
        # Notebook Logic: max_rally = ((future_max - future_min) / future_min)
        if min_forward > 0:
             rally = (max_forward - min_forward) / min_forward
        else:
             rally = 0
             
        is_bottom = 1 if rally >= TargetEvents.RALLY_THRESHOLD else 0
        target_bottom.append(is_bottom)
        
        # TOP: Drawdown >= 15%
        # Notebook Logic: max_drawdown = ((future_window.min() - current_price) / current_price)
        drawdown = (min_forward - current_price) / current_price
        is_top = 1 if drawdown <= -TargetEvents.DRAWDOWN_THRESHOLD else 0
        target_top.append(is_top)
    
    result = pd.DataFrame({
        'Target_Bottom': target_bottom,
        'Target_Top': target_top
    }, index=spx_prices.index)
    
    return result

# ==============================================================================
# FUNZIONI SEGNALI E ESPOSIZIONE
# ==============================================================================

def calculate_signals(breadth_df):
    """
    Calcola segnali di BOTTOM e RISK MANAGEMENT
    Identico a Notebook Cell 7
    """
    df = breadth_df.copy()
    
    # BOTTOM FORTE
    signal_bottom_strong = (
        (df['Breadth_Pct_Above_MA125'] <= BottomThresholds.STRONG_BREADTH) &
        (df['Breadth_Avg_RSI_20'] <= BottomThresholds.STRONG_RSI) &
        (df['Breadth_Avg_Corr_30D'] > BottomThresholds.STRONG_CORR)
    )
    
    # BOTTOM MODERATO
    signal_bottom_moderate = (
        (df['Breadth_Pct_Above_MA125'] <= BottomThresholds.MODERATE_BREADTH) &
        (df['Breadth_Avg_RSI_20'] <= BottomThresholds.MODERATE_RSI) &
        (df['Breadth_Avg_Corr_30D'] > BottomThresholds.MODERATE_CORR) &
        ~signal_bottom_strong
    )
    
    # RISK: Euforia
    signal_risk_euphoria = (
        (df['Breadth_Pct_Above_MA125'] > RiskThresholds.EUPHORIA_BREADTH) &
        (df['Breadth_Avg_Corr_30D'] < RiskThresholds.EUPHORIA_CORR)
    )
    
    # RISK: Deterioramento
    signal_risk_deterioration = (
        (df['Breadth_Pct_Above_MA125'] < RiskThresholds.DETERIORATION_BREADTH) &
        (df['Breadth_Avg_RSI_20'] > RiskThresholds.DETERIORATION_RSI)
    )
    
    # RISK: Crollo
    df['Breadth_Change_5d'] = df['Breadth_Pct_Above_MA125'].diff(5)
    signal_risk_crash = df['Breadth_Change_5d'] < RiskThresholds.CRASH_BREADTH_CHANGE
    
    # Calcola esposizione
    df['Exposure'] = float(DefaultSettings.NEUTRAL_EXPOSURE)
    df['Signal'] = 'NEUTRAL'
    
    # PRIORITÀ 1: BOTTOM
    df.loc[signal_bottom_strong, 'Exposure'] = float(BottomThresholds.STRONG_EXPOSURE)
    df.loc[signal_bottom_strong, 'Signal'] = 'BUY_STRONG'
    
    df.loc[signal_bottom_moderate, 'Exposure'] = float(BottomThresholds.MODERATE_EXPOSURE)
    df.loc[signal_bottom_moderate, 'Signal'] = 'BUY_MODERATE'
    
    # PRIORITÀ 2: RISK (solo se neutrale)
    neutral_mask = (df['Signal'] == 'NEUTRAL')
    
    df.loc[neutral_mask & signal_risk_euphoria, 'Exposure'] = float(RiskThresholds.EUPHORIA_EXPOSURE)
    df.loc[neutral_mask & signal_risk_euphoria, 'Signal'] = 'RISK_EUPHORIA'
    
    df.loc[neutral_mask & signal_risk_deterioration, 'Exposure'] = float(RiskThresholds.DETERIORATION_EXPOSURE)
    df.loc[neutral_mask & signal_risk_deterioration, 'Signal'] = 'RISK_DETERIORATION'
    
    df.loc[neutral_mask & signal_risk_crash, 'Exposure'] = float(RiskThresholds.CRASH_EXPOSURE)
    df.loc[neutral_mask & signal_risk_crash, 'Signal'] = 'RISK_CRASH'
    
    return df

# ==============================================================================
# FUNZIONI BACKTEST
# ==============================================================================

def run_backtest(df_master, initial_capital=INITIAL_CAPITAL):
    """
    Esegue backtest completo
    FIX: Allineato Sharpe Ratio al Notebook (Rf=0.02)
    """
    df = df_master.copy()
    
    # Returns giornalieri SPX
    # Assicura 1D array per evitare problemi di broadcast
    if isinstance(df['SPX_Price'], pd.DataFrame):
         spx_arr = df['SPX_Price'].iloc[:, 0].values
    else:
         spx_arr = df['SPX_Price'].values
         
    df['SPX_Returns'] = pd.Series(spx_arr, index=df.index).pct_change()
    
    # Buy & Hold
    df['BuyHold_Equity'] = initial_capital * (1 + df['SPX_Returns']).cumprod()
    
    # Strategia Dinamica
    df['Strategy_Returns'] = df['SPX_Returns'] * df['Exposure']
    df['Strategy_Equity'] = initial_capital * (1 + df['Strategy_Returns']).cumprod()
    
    # Calcola metriche
    valid_data = df.dropna(subset=['BuyHold_Equity', 'Strategy_Equity'])
    
    if valid_data.empty:
        # Ritorna dummy metrics se vuoto
        return df, {k: 0 for k in ['years', 'cagr_bh', 'sharpe_bh', 'max_dd_bh', 'cagr_strategy', 'sharpe_strategy', 'max_dd_strategy', 'vol_bh', 'vol_strategy', 'final_bh', 'final_strategy']}

    # CAGR
    years = (valid_data.index[-1] - valid_data.index[0]).days / 365.25
    cagr_bh = (valid_data['BuyHold_Equity'].iloc[-1] / initial_capital) ** (1/years) - 1
    cagr_strategy = (valid_data['Strategy_Equity'].iloc[-1] / initial_capital) ** (1/years) - 1
    
    # Volatilità
    vol_bh = valid_data['SPX_Returns'].std() * np.sqrt(252)
    vol_strategy = valid_data['Strategy_Returns'].std() * np.sqrt(252)
    
    # Sharpe (risk-free = 0.02 come Notebook Cell 8)
    rf_rate = 0.02
    sharpe_bh = (cagr_bh - rf_rate) / vol_bh if vol_bh != 0 else 0
    sharpe_strategy = (cagr_strategy - rf_rate) / vol_strategy if vol_strategy != 0 else 0
    
    # Max Drawdown
    def calc_max_dd(equity):
        peak = equity.expanding(min_periods=1).max()
        dd = (equity - peak) / peak
        return dd.min()
    
    max_dd_bh = calc_max_dd(valid_data['BuyHold_Equity'])
    max_dd_strategy = calc_max_dd(valid_data['Strategy_Equity'])
    
    metrics = {
        'years': years,
        'cagr_bh': cagr_bh,
        'cagr_strategy': cagr_strategy,
        'vol_bh': vol_bh,
        'vol_strategy': vol_strategy,
        'sharpe_bh': sharpe_bh,
        'sharpe_strategy': sharpe_strategy,
        'max_dd_bh': max_dd_bh,
        'max_dd_strategy': max_dd_strategy,
        'final_bh': valid_data['BuyHold_Equity'].iloc[-1],
        'final_strategy': valid_data['Strategy_Equity'].iloc[-1]
    }
    
    return df, metrics

# ==============================================================================
# FUNZIONE MASTER: ANALISI COMPLETA
# ==============================================================================

def run_complete_analysis():
    """
    Esegue l'analisi completa dall'inizio alla fine
    FIX: Gestione dimensionale robusta (1D) per evitare errori numpy/pandas
    """
    
    # 1. Download dati
    raw_data, df_close, df_high, df_low = load_all_indices(INDICES)
    
    # 2. Calcola Market Breadth
    breadth_df = calculate_market_breadth(df_close, df_high, df_low)
    
    # 3. Allinea prezzi SPX
    df_close_aligned = df_close.loc[breadth_df.index]
    
    # FIX: Gestione robusta estrazione colonna
    if isinstance(df_close_aligned['S&P 500'], pd.DataFrame):
        spx_prices = df_close_aligned['S&P 500'].iloc[:, 0]
    else:
        spx_prices = df_close_aligned['S&P 500']
    
    # 4. Calcola eventi target
    target_events = calculate_target_events(spx_prices)
    
    # 5. Costruisci df_master
    df_master = breadth_df.copy()
    df_master = df_master.join(target_events, how='inner')
    
    # FIX: Assegnazione sicura array 1D
    df_master['SPX_Price'] = spx_prices.values
    
    # 6. Calcola segnali (usando la funzione helper per evitare duplicazione codice)
    signals_df = calculate_signals(df_master)
    
    # Merge dei risultati segnali in df_master
    cols_to_update = ['Exposure', 'Signal', 'Breadth_Change_5d']
    df_master[cols_to_update] = signals_df[cols_to_update]
    
    if df_master.empty:
        raise ValueError("DataFrame master vuoto dopo merge!")
    
    # 7. Backtest
    df_master, metrics = run_backtest(df_master)
    
    # 8. Stato attuale
    current_state = df_master.iloc[-1].to_dict()
    current_state['date'] = df_master.index[-1]
    
    return df_master, metrics, current_state

# ==============================================================================
# FUNZIONE EXPORT JSON
# ==============================================================================

def export_to_json(df_master, metrics, current_state):
    """Esporta tutti i dati in formato JSON"""
    
    json_data = {
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'system_version': 'V4.0',
            'period_start': df_master.index[0].strftime('%Y-%m-%d'),
            'period_end': df_master.index[-1].strftime('%Y-%m-%d'),
            'total_days': len(df_master),
            'indices_count': len(INDICES)
        },
        'current_state': {
            'date': current_state['date'].strftime('%Y-%m-%d'),
            'signal': current_state['Signal'],
            'exposure': float(current_state['Exposure']),
            'spx_price': float(current_state['SPX_Price']),
            'breadth_pct_above_ma125': float(current_state['Breadth_Pct_Above_MA125']),
            'breadth_avg_rsi_20': float(current_state['Breadth_Avg_RSI_20']),
            'breadth_avg_corr_30d': float(current_state['Breadth_Avg_Corr_30D'])
        },
        'performance_metrics': {
            'backtest_years': float(metrics['years']),
            'buy_hold': {
                'cagr': float(metrics['cagr_bh']),
                'volatility': float(metrics['vol_bh']),
                'sharpe_ratio': float(metrics['sharpe_bh']),
                'max_drawdown': float(metrics['max_dd_bh']),
                'final_equity': float(metrics['final_bh'])
            },
            'strategy': {
                'cagr': float(metrics['cagr_strategy']),
                'volatility': float(metrics['vol_strategy']),
                'sharpe_ratio': float(metrics['sharpe_strategy']),
                'max_drawdown': float(metrics['max_dd_strategy']),
                'final_equity': float(metrics['final_strategy'])
            }
        },
        'configuration': {
            'bottom_thresholds': {
                'strong': {
                    'breadth': BottomThresholds.STRONG_BREADTH,
                    'rsi': BottomThresholds.STRONG_RSI,
                    'correlation': BottomThresholds.STRONG_CORR,
                    'exposure': BottomThresholds.STRONG_EXPOSURE
                },
                'moderate': {
                    'breadth': BottomThresholds.MODERATE_BREADTH,
                    'rsi': BottomThresholds.MODERATE_RSI,
                    'correlation': BottomThresholds.MODERATE_CORR,
                    'exposure': BottomThresholds.MODERATE_EXPOSURE
                }
            },
            'risk_thresholds': {
                'euphoria': {
                    'breadth': RiskThresholds.EUPHORIA_BREADTH,
                    'correlation': RiskThresholds.EUPHORIA_CORR,
                    'exposure': RiskThresholds.EUPHORIA_EXPOSURE
                },
                'deterioration': {
                    'breadth': RiskThresholds.DETERIORATION_BREADTH,
                    'rsi': RiskThresholds.DETERIORATION_RSI,
                    'exposure': RiskThresholds.DETERIORATION_EXPOSURE
                },
                'crash': {
                    'breadth_change': RiskThresholds.CRASH_BREADTH_CHANGE,
                    'exposure': RiskThresholds.CRASH_EXPOSURE
                }
            },
            'default_exposure': DefaultSettings.NEUTRAL_EXPOSURE
        },
        'historical_data': {
            'dates': df_master.index.strftime('%Y-%m-%d').tolist(),
            'spx_price': df_master['SPX_Price'].tolist(),
            'breadth_pct_above_ma125': df_master['Breadth_Pct_Above_MA125'].tolist(),
            'breadth_avg_rsi_20': df_master['Breadth_Avg_RSI_20'].tolist(),
            'breadth_avg_corr_30d': df_master['Breadth_Avg_Corr_30D'].tolist(),
            'exposure': df_master['Exposure'].tolist(),
            'signal': df_master['Signal'].tolist(),
            'buy_hold_equity': df_master['BuyHold_Equity'].tolist(),
            'strategy_equity': df_master['Strategy_Equity'].tolist()
        },
        'signal_distribution': df_master['Signal'].value_counts().to_dict()
    }
    
    return json_data
