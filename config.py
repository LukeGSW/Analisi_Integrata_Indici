# -*- coding: utf-8 -*-
"""
config.py - Configurazione Centralizzata Sistema SPX Dashboard
Kriterion Quant - Sistema V4.0
"""

# ==============================================================================
# SEZIONE 1: SOGLIE BOTTOM (Opportunità Ingresso SPX)
# ==============================================================================

class BottomThresholds:
    """Soglie per identificare opportunità di ingresso su SPX"""
    
    # BOTTOM FORTE (Alta Confidenza - Precision 90%)
    STRONG_BREADTH = 0.125     # 12.5% - % indici sopra MA125
    STRONG_RSI = 29            # RSI medio globale
    STRONG_CORR = 0.71         # Correlazione 30D
    STRONG_EXPOSURE = 1.00     # Esposizione SPX (0.0 - 2.0)
    
    # BOTTOM MODERATO (Media Confidenza - Precision 44%)
    MODERATE_BREADTH = 0.290   # 29% - % indici sopra MA125
    MODERATE_RSI = 42          # RSI medio globale
    MODERATE_CORR = 0.58       # Correlazione 30D
    MODERATE_EXPOSURE = 0.50   # Esposizione SPX (0.0 - 2.0)

# ==============================================================================
# SEZIONE 2: SOGLIE RISK MANAGEMENT (Riduzione Esposizione SPX)
# ==============================================================================

class RiskThresholds:
    """Soglie per risk management - sostituiscono TOP prediction"""
    
    # EUFORIA DECORRELATA
    EUPHORIA_BREADTH = 0.80      # 80% - breadth minimo
    EUPHORIA_CORR = 0.40         # Correlazione massima (sotto = decorrelazione)
    EUPHORIA_EXPOSURE = 0.40     # Esposizione SPX
    
    # DETERIORAMENTO
    DETERIORATION_BREADTH = 0.40 # 40% - breadth massimo
    DETERIORATION_RSI = 55       # RSI minimo
    DETERIORATION_EXPOSURE = 0.30 # Esposizione SPX
    
    # CROLLO RAPIDO
    CRASH_BREADTH_CHANGE = -0.20 # -20% - variazione breadth in 5 giorni
    CRASH_EXPOSURE = 0.20        # Esposizione SPX

# ==============================================================================
# SEZIONE 3: ESPOSIZIONE DEFAULT (Neutrale)
# ==============================================================================

class DefaultSettings:
    """Impostazioni default quando nessun segnale è attivo"""
    
    NEUTRAL_EXPOSURE = 0.80      # Esposizione SPX di default

# ==============================================================================
# SEZIONE 4: PARAMETRI EVENTI TARGET (per backtest)
# ==============================================================================

class TargetEvents:
    """Parametri per definizione eventi TOP e BOTTOM storici"""
    
    FORWARD_WINDOW = 60          # Giorni di osservazione futuri
    DRAWDOWN_THRESHOLD = 0.15    # 15% - definizione TOP
    RALLY_THRESHOLD = 0.20       # 20% - definizione BOTTOM

# ==============================================================================
# SEZIONE 5: PARAMETRI MARKET BREADTH
# ==============================================================================

class BreadthSettings:
    """Parametri per calcolo indicatori Market Breadth"""
    
    MA_PERIOD = 125              # Periodo media mobile per breadth
    RSI_PERIOD = 20              # Periodo RSI
    CORR_WINDOW = 30             # Finestra correlazione rolling

# ==============================================================================
# SEZIONE 6: INDICI GLOBALI
# ==============================================================================

INDICES = {
    # ===== TARGET PRINCIPALE =====
    'S&P 500': '^GSPC',  # 🎯 QUESTO È IL NOSTRO ASSET TARGET
    
    # ===== INDICI USA (Confluenze Domestiche) =====
    'NASDAQ 100': '^NDX',
    'Dow Jones': '^DJI',
    
    # ===== INDICI EUROPA (Confluenze Globali) =====
    'DAX (Germania)': '^GDAXI',
    'CAC 40 (Francia)': '^FCHI',
    'FTSE 100 (UK)': '^FTSE',
    'FTSE MIB (Italia)': 'FTSEMIB.MI',
    'IBEX 35 (Spagna)': '^IBEX',
    'AEX (Olanda)': '^AEX',
    'SMI (Svizzera)': '^SSMI',
    
    # ===== INDICI ASIA (Confluenze Globali) =====
    'Nikkei 225 (Giappone)': '^N225',
    'Hang Seng (Hong Kong)': '^HSI',
    'KOSPI (Corea del Sud)': '^KS11'
}

# ==============================================================================
# CONFIGURAZIONE GENERALE
# ==============================================================================

START_DATE = '2000-01-01'
INITIAL_CAPITAL = 100000  # Capitale iniziale per backtest

# ==============================================================================
# VALIDAZIONE CONFIGURAZIONE
# ==============================================================================

def validate_config():
    """Valida che tutte le configurazioni siano corrette"""
    
    # Verifica esposizioni
    assert 0 <= BottomThresholds.STRONG_EXPOSURE <= 2.0, "Esposizione STRONG deve essere tra 0 e 2.0"
    assert 0 <= BottomThresholds.MODERATE_EXPOSURE <= 2.0, "Esposizione MODERATE deve essere tra 0 e 2.0"
    assert 0 <= DefaultSettings.NEUTRAL_EXPOSURE <= 1.5, "Esposizione NEUTRAL deve essere tra 0 e 1.5"
    assert 0 <= RiskThresholds.EUPHORIA_EXPOSURE <= 1.0, "Esposizione RISK deve essere tra 0 e 1.0"
    
    # Verifica logica soglie
    assert BottomThresholds.STRONG_BREADTH < BottomThresholds.MODERATE_BREADTH, \
        "BOTTOM Forte deve avere breadth più basso di Moderato"
    assert BottomThresholds.STRONG_RSI < BottomThresholds.MODERATE_RSI, \
        "BOTTOM Forte deve avere RSI più basso di Moderato"
    
    return True

# Valida configurazione all'import
validate_config()
