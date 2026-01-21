# 📈 SPX Dashboard - Sistema di Timing S&P 500

## 🎯 Overview

Dashboard interattiva Streamlit per il **timing tattico su S&P 500** utilizzando confluenze di **Market Breadth globale** su 13 indici azionari internazionali.

Sistema sviluppato da **Kriterion Quant** - Versione 4.0

---

## 🚀 Features

### Core Features
- **📊 Analisi Market Breadth Globale**: Monitora 13 indici (USA, Europa, Asia)
- **🎯 Segnali di Timing**: BOTTOM (opportunità) e RISK MANAGEMENT (protezione)
- **📈 Backtest Completo**: Performance dal 2000 con metriche dettagliate
- **💰 Esposizione Dinamica**: Gestione automatica del peso SPX (20% - 100%)
- **📉 6 Grafici Interattivi**: Equity, Esposizione, Drawdown, Breadth, RSI, Correlazione
- **💾 Export JSON**: Dati completi per analisi LLM esterne

### Indicatori Utilizzati
1. **Market Breadth**: % indici sopra MA125
2. **RSI Medio**: Momentum medio globale (20 periodi)
3. **Correlazione Rolling**: Sincronizzazione mercati (30 giorni)

---

## 📦 Installazione

### Prerequisiti
- Python 3.9+
- pip

### Setup Locale

```bash
# 1. Clone repository
git clone https://github.com/tuousername/spx-dashboard.git
cd spx-dashboard

# 2. Crea virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# oppure
venv\Scripts\activate  # Windows

# 3. Installa dipendenze
pip install -r requirements.txt

# 4. Avvia app
streamlit run app.py
```

L'app sarà disponibile su `http://localhost:8501`

---

## 🌐 Deploy su Streamlit Cloud

### Procedura Rapida

1. **Push su GitHub**
   ```bash
   git add .
   git commit -m "Initial commit"
   git push origin main
   ```

2. **Deploy su Streamlit Cloud**
   - Vai su [share.streamlit.io](https://share.streamlit.io)
   - Clicca "New app"
   - Seleziona il repository GitHub
   - Main file path: `app.py`
   - Clicca "Deploy!"

3. **URL Pubblico**
   - L'app sarà disponibile su: `https://tuousername-spx-dashboard-app-xxxxx.streamlit.app`

### Configurazione Avanzata

Se necessario, puoi personalizzare `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#667eea"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"

[server]
headless = true
port = 8501
```

---

## 📊 Struttura del Progetto

```
spx-dashboard/
├── app.py                  # Main Streamlit application
├── utils.py                # Helper functions (download, calcoli, backtest)
├── config.py               # Configurazione centralizzata (soglie)
├── requirements.txt        # Dipendenze Python
├── README.md              # Questo file
└── .streamlit/
    └── config.toml        # Configurazione Streamlit
```

---

## ⚙️ Configurazione Sistema

### Soglie BOTTOM (Opportunità Ingresso)

```python
# config.py - BottomThresholds

STRONG (Precision 90%):
- Breadth ≤ 12.5%
- RSI ≤ 29
- Correlazione > 0.71
- Esposizione: 100%

MODERATE (Precision 44%):
- Breadth ≤ 29%
- RSI ≤ 42
- Correlazione > 0.58
- Esposizione: 50%
```

### Soglie RISK MANAGEMENT (Protezione)

```python
# config.py - RiskThresholds

EUFORIA:
- Breadth > 80%
- Correlazione < 0.40
- Esposizione: 40%

DETERIORAMENTO:
- Breadth < 40%
- RSI > 55
- Esposizione: 30%

CROLLO:
- Breadth perde > 20% in 5 giorni
- Esposizione: 20%
```

### Indici Monitorati

**USA**: S&P 500, NASDAQ 100, Dow Jones
**Europa**: DAX, CAC 40, FTSE 100, FTSE MIB, IBEX 35, AEX, SMI
**Asia**: Nikkei 225, Hang Seng, KOSPI

---

## 🎯 Come Usare il Sistema

### 1. Dashboard Principale
- Visualizza il **segnale attuale** (BOTTOM, RISK, NEUTRAL)
- Consulta le **metriche di performance** (CAGR, Sharpe, Drawdown)
- Analizza l'**equity curve** della strategia vs Buy&Hold

### 2. Analisi Dettagliata
- Esplora i **grafici degli indicatori** (Breadth, RSI, Correlazione)
- Verifica la **distribuzione storica** dei segnali
- Identifica pattern e comportamenti del mercato

### 3. Metodologia
- Comprendi la **logica del sistema**
- Studia i **criteri dei segnali**
- Leggi le **istruzioni operative**

### 4. Export Dati
- Scarica **JSON completo** con tutti i dati
- Usa per **analisi LLM** esterne (Claude, GPT-4)
- Integra con **tool personalizzati**

---

## 📈 Performance Storiche (2000-2025)

### Metriche Sistema (esempio)

| Metrica | Buy & Hold | Strategia | Delta |
|---------|-----------|-----------|-------|
| **CAGR** | 9.5% | 12.3% | +2.8% |
| **Volatilità** | 18.2% | 14.7% | -3.5% |
| **Sharpe Ratio** | 0.52 | 0.84 | +0.32 |
| **Max Drawdown** | -55.2% | -38.4% | +16.8% |
| **Capitale Finale** | $987k | $1,456k | +$469k |

*Nota: Le performance passate non garantiscono risultati futuri*

---

## 🔧 Personalizzazione

### Modificare le Soglie

Edita `config.py` per personalizzare i parametri:

```python
# Esempio: aumentare esposizione BOTTOM Forte a 120% (con leva)
class BottomThresholds:
    STRONG_EXPOSURE = 1.20  # 120% (leva 1.2x)
```

### Aggiungere Indici

Modifica il dizionario `INDICES` in `config.py`:

```python
INDICES = {
    'S&P 500': '^GSPC',
    'Nuovo Indice': 'TICKER',
    # ...
}
```

### Modificare Periodi di Analisi

```python
# config.py
START_DATE = '2010-01-01'  # Cambia data inizio

# config.py - BreadthSettings
MA_PERIOD = 200  # Cambia periodo MA da 125 a 200
```

---

## 💾 Export JSON - Struttura Dati

Il file JSON esportato contiene:

```json
{
  "metadata": {
    "generated_at": "2026-01-21T10:30:00",
    "system_version": "V4.0",
    "period_start": "2000-01-01",
    "period_end": "2026-01-21",
    "total_days": 6570,
    "indices_count": 13
  },
  "current_state": {
    "date": "2026-01-21",
    "signal": "NEUTRAL",
    "exposure": 0.80,
    "spx_price": 6852.34,
    "breadth_pct_above_ma125": 0.68,
    "breadth_avg_rsi_20": 58.3,
    "breadth_avg_corr_30d": 0.62
  },
  "performance_metrics": { ... },
  "configuration": { ... },
  "historical_data": { ... },
  "signal_distribution": { ... }
}
```

---

## 🛠️ Troubleshooting

### Errore Download Dati

```bash
# Se yfinance fallisce, prova:
pip install --upgrade yfinance
```

### Problemi Performance

```python
# Riduci periodo analisi in config.py
START_DATE = '2010-01-01'  # Invece di 2000
```

### Cache Streamlit

```python
# Pulisci cache nell'app:
# Sidebar → 🔄 Aggiorna Dati
```

---

## 📚 Risorse

- **Documentazione Streamlit**: [docs.streamlit.io](https://docs.streamlit.io)
- **yfinance Documentation**: [pypi.org/project/yfinance](https://pypi.org/project/yfinance/)
- **Plotly Charts**: [plotly.com/python](https://plotly.com/python/)
- **pandas-ta**: [github.com/twopirllc/pandas-ta](https://github.com/twopirllc/pandas-ta)

---

## ⚠️ Disclaimer

**IMPORTANTE**: Questo sistema è fornito esclusivamente a scopo **educativo e di ricerca**. 

- ❌ **NON costituisce consulenza finanziaria, fiscale o legale**
- ❌ Le performance passate **non garantiscono risultati futuri**
- ❌ Ogni investimento comporta **rischi**, inclusa la perdita totale del capitale
- ✅ Consulta sempre un **consulente finanziario professionista** prima di investire
- ✅ Usa il sistema **solo per scopi educativi e di backtesting**

---

## 📧 Contatti

**Kriterion Quant**
- Website: [KriterionQuant.com](https://kriterionquant.com)
- Email: info@kriterionquant.com
- GitHub: [@kriterionquant](https://github.com/kriterionquant)

---

## 📄 Licenza

Copyright © 2026 Kriterion Quant. Tutti i diritti riservati.

Questo software è fornito "così com'è", senza garanzie di alcun tipo, esplicite o implicite.

---

## 🙏 Credits

Sviluppato con:
- **Streamlit** - Framework dashboard interattive
- **yfinance** - Download dati finanziari
- **Plotly** - Grafici interattivi
- **pandas-ta** - Indicatori tecnici
- **Claude (Anthropic)** - Assistenza sviluppo

---

**⭐ Se trovi utile questo progetto, lascia una stella su GitHub!**

**🔔 Watch il repository per ricevere aggiornamenti sulle nuove versioni**

---

*Last update: 21 Gennaio 2026 - Version 4.0*
