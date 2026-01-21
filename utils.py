# -*- coding: utf-8 -*-
"""
utils.py - Funzioni Helper per SPX Dashboard
Kriterion Quant - Sistema V4.0
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
    
    Args:
        ticker: simbolo ticker (es. '^GSPC')
        start_date: data inizio (default: 2000-01-01)
    
    Returns:
        DataFrame con dati OHLCV
    """
    try:
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
    
    Returns:
        Tuple (raw_data dict, aligned_close, aligned_high, aligned_low)
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
    
    # Allinea date
    common_dates = None
    for df in raw_data.values():
        if common_dates is None:
            common_dates = df.index
        else:
            common_dates = common_dates.intersection(df.index)
    
    # Verifica che ci siano date comuni
    if len(common_dates) == 0:
        raise ValueError("Nessuna data comune tra gli indici! Verifica i dati.")
    
    # Crea DataFrame allineati
    df_close = pd.DataFrame({name: df.loc[common_dates, 'Close'] for name, df in raw_data.items()})
    df_high = pd.DataFrame({name: df.loc[common_dates, 'High'] for name, df in raw_data.items()})
    df_low = pd.DataFrame({name: df.loc[common_dates, 'Low'] for name, df in raw_data.items()})
    
    # Verifica che i DataFrame non siano vuoti
    if df_close.empty or df_high.empty or df_low.empty:
        raise ValueError("DataFrame allineati vuoti! Problema nel download dati.")
    
    return raw_data, df_close, df_high, df_low

# ==============================================================================
# FUNZIONI CALCOLO MARKET BREADTH
# ==============================================================================

def calculate_market_breadth(df_close, df_high, df_low):
    """
    Calcola i 3 indicatori di Market Breadth
    
    Returns:
        DataFrame con colonne: Breadth_Pct_Above_MA125, Breadth_Avg_RSI_20, Breadth_Avg_Corr_30D
    """
    
    # 1. % Indici sopra MA125
    ma_125 = df_close.rolling(window=BreadthSettings.MA_PERIOD).mean()
    above_ma = (df_close > ma_125).sum(axis=1) / df_close.shape[1]
    
    # 2. RSI Medio
    rsi_values = pd.DataFrame()
    for col in df_close.columns:
        rsi = ta.rsi(df_close[col], length=BreadthSettings.RSI_PERIOD)
        rsi_values[col] = rsi
    avg_rsi = rsi_values.mean(axis=1)
    
    # 3. Correlazione Rolling 30D
    returns = df_close.pct_change()
    
    def avg_correlation_robust(window_returns):
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
        if i < BreadthSettings.CORR_WINDOW - 1:
            rolling_corr.append(np.nan)
        else:
            window = returns.iloc[i-BreadthSettings.CORR_WINDOW+1:i+1]
            avg_corr = avg_correlation_robust(window)
            rolling_corr.append(avg_corr)
    
    # Crea DataFrame - IMPORTANTE: converti rolling_corr in Series prima
    breadth_df = pd.DataFrame({
        'Breadth_Pct_Above_MA125': above_ma,
        'Breadth_Avg_RSI_20': avg_rsi,
        'Breadth_Avg_Corr_30D': pd.Series(rolling_corr, index=df_close.index)
    })
    
    # Fix NaN correlazione
    breadth_df['Breadth_Avg_Corr_30D'] = breadth_df['Breadth_Avg_Corr_30D'].ffill().fillna(0.5)
    
    # Rimuovi NaN
    breadth_df = breadth_df.dropna()
    
    return breadth_df

# ==============================================================================
# FUNZIONI EVENTI TARGET
# ==============================================================================

def calculate_target_events(spx_prices):
    """
    Identifica eventi BOTTOM e TOP storici su S&P 500
    
    Returns:
        DataFrame con colonne: Target_Bottom, Target_Top
    """
    # Validazione input
    if not isinstance(spx_prices, pd.Series):
        raise ValueError(f"spx_prices deve essere una Series, ricevuto: {type(spx_prices)}")
    
    if not isinstance(spx_prices.index, pd.DatetimeIndex):
        raise ValueError(f"spx_prices.index deve essere DatetimeIndex, ricevuto: {type(spx_prices.index)}")
    
    target_bottom = []
    target_top = []
    
    for i in range(len(spx_prices)):
        current_price = spx_prices.iloc[i]
        
        # Forward window
        if i + TargetEvents.FORWARD_WINDOW >= len(spx_prices):
            forward_prices = spx_prices.iloc[i:]
        else:
            forward_prices = spx_prices.iloc[i:i+TargetEvents.FORWARD_WINDOW+1]
        
        if len(forward_prices) < 2:
            target_bottom.append(0)
            target_top.append(0)
            continue
        
        # BOTTOM: Rally >= 20%
        max_forward = forward_prices.max()
        rally = (max_forward / current_price) - 1
        is_bottom = 1 if rally >= TargetEvents.RALLY_THRESHOLD else 0
        target_bottom.append(is_bottom)
        
        # TOP: Drawdown >= 15%
        min_forward = forward_prices.min()
        drawdown = (min_forward / current_price) - 1
        is_top = 1 if drawdown <= -TargetEvents.DRAWDOWN_THRESHOLD else 0
        target_top.append(is_top)
    
    # Crea DataFrame con index esplicito
    result = pd.DataFrame({
        'Target_Bottom': target_bottom,
        'Target_Top': target_top
    }, index=spx_prices.index)
    
    print(f"DEBUG calculate_target_events: result shape = {result.shape}, index type = {type(result.index)}")
    
    return result

# ==============================================================================
# FUNZIONI SEGNALI E ESPOSIZIONE
# ==============================================================================

def calculate_signals(breadth_df):
    """
    Calcola segnali di BOTTOM e RISK MANAGEMENT
    
    Returns:
        DataFrame con colonne: Exposure, Signal
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
    df['Exposure'] = DefaultSettings.NEUTRAL_EXPOSURE
    df['Signal'] = 'NEUTRAL'
    
    # PRIORITÀ 1: BOTTOM
    df.loc[signal_bottom_strong, 'Exposure'] = BottomThresholds.STRONG_EXPOSURE
    df.loc[signal_bottom_strong, 'Signal'] = 'BUY_STRONG'
    
    df.loc[signal_bottom_moderate, 'Exposure'] = BottomThresholds.MODERATE_EXPOSURE
    df.loc[signal_bottom_moderate, 'Signal'] = 'BUY_MODERATE'
    
    # PRIORITÀ 2: RISK (solo se neutrale)
    neutral_mask = (df['Signal'] == 'NEUTRAL')
    
    df.loc[neutral_mask & signal_risk_euphoria, 'Exposure'] = RiskThresholds.EUPHORIA_EXPOSURE
    df.loc[neutral_mask & signal_risk_euphoria, 'Signal'] = 'RISK_EUPHORIA'
    
    df.loc[neutral_mask & signal_risk_deterioration, 'Exposure'] = RiskThresholds.DETERIORATION_EXPOSURE
    df.loc[neutral_mask & signal_risk_deterioration, 'Signal'] = 'RISK_DETERIORATION'
    
    df.loc[neutral_mask & signal_risk_crash, 'Exposure'] = RiskThresholds.CRASH_EXPOSURE
    df.loc[neutral_mask & signal_risk_crash, 'Signal'] = 'RISK_CRASH'
    
    return df

# ==============================================================================
# FUNZIONI BACKTEST
# ==============================================================================

def run_backtest(df_master, initial_capital=INITIAL_CAPITAL):
    """
    Esegue backtest completo con Buy&Hold e Strategia Dinamica
    
    Returns:
        DataFrame con equity curves e metriche
    """
    df = df_master.copy()
    
    # Returns giornalieri SPX
    df['SPX_Returns'] = df['SPX_Price'].pct_change()
    
    # Buy & Hold
    df['BuyHold_Equity'] = initial_capital * (1 + df['SPX_Returns']).cumprod()
    
    # Strategia Dinamica
    df['Strategy_Returns'] = df['SPX_Returns'] * df['Exposure']
    df['Strategy_Equity'] = initial_capital * (1 + df['Strategy_Returns']).cumprod()
    
    # Calcola metriche
    valid_data = df.dropna(subset=['BuyHold_Equity', 'Strategy_Equity'])
    
    # CAGR
    years = (valid_data.index[-1] - valid_data.index[0]).days / 365.25
    cagr_bh = (valid_data['BuyHold_Equity'].iloc[-1] / initial_capital) ** (1/years) - 1
    cagr_strategy = (valid_data['Strategy_Equity'].iloc[-1] / initial_capital) ** (1/years) - 1
    
    # Volatilità
    vol_bh = valid_data['SPX_Returns'].std() * np.sqrt(252)
    vol_strategy = valid_data['Strategy_Returns'].std() * np.sqrt(252)
    
    # Sharpe (risk-free = 0)
    sharpe_bh = cagr_bh / vol_bh if vol_bh != 0 else 0
    sharpe_strategy = cagr_strategy / vol_strategy if vol_strategy != 0 else 0
    
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
    
    Returns:
        Tuple (df_master, metrics, current_state)
    """
    
    try:
        # 1. Download dati
        print("DEBUG: Step 1 - Download dati...")
        raw_data, df_close, df_high, df_low = load_all_indices(INDICES)
        print(f"DEBUG: Downloaded {len(raw_data)} indices")
        
        # 2. Calcola Market Breadth
        print("DEBUG: Step 2 - Calcola Market Breadth...")
        breadth_df = calculate_market_breadth(df_close, df_high, df_low)
        print(f"DEBUG: Breadth_df shape: {breadth_df.shape}, index type: {type(breadth_df.index)}")
        
        # 3. Allinea prezzi SPX
        print("DEBUG: Step 3 - Allinea prezzi SPX...")
        df_close_aligned = df_close.loc[breadth_df.index]
        spx_prices = df_close_aligned['S&P 500']
        print(f"DEBUG: SPX prices shape: {spx_prices.shape}, type: {type(spx_prices)}")
        
        # 4. Calcola eventi target
        print("DEBUG: Step 4 - Calcola eventi target...")
        target_events = calculate_target_events(spx_prices)
        print(f"DEBUG: Target events shape: {target_events.shape}")
        
        # 5. Calcola segnali
        print("DEBUG: Step 5 - Calcola segnali...")
        df_signals = calculate_signals(breadth_df)
        print(f"DEBUG: Signals shape: {df_signals.shape}")
        
        # 6. Merge tutto
        print("DEBUG: Step 6 - Merge tutto...")
        print(f"DEBUG: breadth_df index: {len(breadth_df.index)}, type: {type(breadth_df.index)}")
        print(f"DEBUG: target_events index: {len(target_events.index)}, type: {type(target_events.index)}")
        print(f"DEBUG: df_signals index: {len(df_signals.index)}, type: {type(df_signals.index)}")
        print(f"DEBUG: spx_prices index: {len(spx_prices.index)}, type: {type(spx_prices.index)}")
        
        # Converti spx_prices a DataFrame prima
        spx_df = pd.DataFrame({'SPX_Price': spx_prices}, index=spx_prices.index)
        print(f"DEBUG: spx_df shape: {spx_df.shape}, index type: {type(spx_df.index)}")
        
        df_master = pd.concat([
            breadth_df,
            target_events,
            df_signals[['Exposure', 'Signal']],
            spx_df
        ], axis=1, join='inner')
        
        print(f"DEBUG: df_master shape after concat: {df_master.shape}")
        
        # Verifica che il merge sia riuscito
        if df_master.empty:
            raise ValueError("DataFrame master vuoto dopo merge!")
        
        # 7. Backtest
        print("DEBUG: Step 7 - Backtest...")
        df_master, metrics = run_backtest(df_master)
        print("DEBUG: Backtest completato")
        
        # 8. Stato attuale
        print("DEBUG: Step 8 - Stato attuale...")
        current_state = df_master.iloc[-1].to_dict()
        current_state['date'] = df_master.index[-1]
        print("DEBUG: Analisi completata con successo!")
        
        return df_master, metrics, current_state
        
    except Exception as e:
        import traceback
        print(f"DEBUG: ERRORE in step: {traceback.format_exc()}")
        raise

# ==============================================================================
# FUNZIONE EXPORT JSON
# ==============================================================================

def export_to_json(df_master, metrics, current_state):
    """
    Esporta tutti i dati in formato JSON per analisi LLM
    
    Returns:
        Dict completo con tutti i dati
    """
    
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
