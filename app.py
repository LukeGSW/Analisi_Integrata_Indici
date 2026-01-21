# -*- coding: utf-8 -*-
"""
app.py - Dashboard Streamlit SPX System
Kriterion Quant - Sistema V4.0
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import datetime

from utils import run_complete_analysis, export_to_json
from config import *

# ==============================================================================
# CONFIGURAZIONE PAGINA
# ==============================================================================

st.set_page_config(
    page_title="SPX Dashboard - Kriterion Quant",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# CUSTOM CSS
# ==============================================================================

st.markdown("""
<style>
    /* Header principale */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #4c5fb0 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    
    .main-header h1 {
        font-size: 2.5em;
        margin-bottom: 0.5rem;
    }
    
    .main-header h2 {
        font-size: 1.3em;
        font-weight: 300;
        opacity: 0.9;
    }
    
    /* Box metriche */
    .metric-box {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        text-align: center;
        height: 100%;
    }
    
    /* Stile segnali */
    .signal-buy { color: #28a745; font-weight: bold; }
    .signal-risk { color: #dc3545; font-weight: bold; }
    .signal-neutral { color: #6c757d; font-weight: bold; }
    
    /* Info boxes */
    .info-box {
        background: #eef2f7;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
    }
    
    .info-box h3 {
        color: #1e3c72;
        border-bottom: 2px solid #667eea;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }
    
    /* Tabella performance */
    .perf-table {
        width: 100%;
        margin: 1rem 0;
    }
    
    /* Footer */
    .footer {
        background: #1a2c47;
        color: #b0c4de;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# FUNZIONI HELPER UI
# ==============================================================================

def get_signal_info(signal):
    """Restituisce emoji, fase e azione per il segnale corrente"""
    
    signal_map = {
        'BUY_STRONG': {
            'emoji': '🟢',
            'phase': 'BOTTOM FORTE',
            'action': f'MASSIMA ESPOSIZIONE SPX ({BottomThresholds.STRONG_EXPOSURE:.0%})',
            'class': 'signal-buy'
        },
        'BUY_MODERATE': {
            'emoji': '🟢',
            'phase': 'BOTTOM MODERATO',
            'action': f'ALTA ESPOSIZIONE SPX ({BottomThresholds.MODERATE_EXPOSURE:.0%})',
            'class': 'signal-buy'
        },
        'NEUTRAL': {
            'emoji': '⚪',
            'phase': 'NEUTRALE',
            'action': f'ESPOSIZIONE NEUTRALE SPX ({DefaultSettings.NEUTRAL_EXPOSURE:.0%})',
            'class': 'signal-neutral'
        },
        'RISK_EUPHORIA': {
            'emoji': '🔴',
            'phase': 'RISK: EUFORIA',
            'action': f'RIDUZIONE ESPOSIZIONE SPX ({RiskThresholds.EUPHORIA_EXPOSURE:.0%})',
            'class': 'signal-risk'
        },
        'RISK_DETERIORATION': {
            'emoji': '🟠',
            'phase': 'RISK: DETERIORAMENTO',
            'action': f'RIDUZIONE ESPOSIZIONE SPX ({RiskThresholds.DETERIORATION_EXPOSURE:.0%})',
            'class': 'signal-risk'
        },
        'RISK_CRASH': {
            'emoji': '🚨',
            'phase': 'RISK: CROLLO',
            'action': f'RIDUZIONE ESPOSIZIONE SPX ({RiskThresholds.CRASH_EXPOSURE:.0%})',
            'class': 'signal-risk'
        }
    }
    
    return signal_map.get(signal, signal_map['NEUTRAL'])

def create_equity_chart(df):
    """Crea grafico equity curves"""
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['BuyHold_Equity'],
        name='Buy & Hold SPX',
        line=dict(color='#6c757d', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Strategy_Equity'],
        name='Strategia Dinamica',
        line=dict(color='#28a745', width=3)
    ))
    
    fig.update_layout(
        title='Performance: Buy & Hold vs Strategia Dinamica',
        xaxis_title='Data',
        yaxis_title='Equity ($)',
        hovermode='x unified',
        template='plotly_white',
        height=500
    )
    
    return fig

def create_exposure_chart(df):
    """Crea grafico esposizione dinamica"""
    
    fig = go.Figure()
    
    # Colora per segnale
    colors = {
        'BUY_STRONG': '#28a745',
        'BUY_MODERATE': '#90EE90',
        'NEUTRAL': '#6c757d',
        'RISK_EUPHORIA': '#dc3545',
        'RISK_DETERIORATION': '#FFA500',
        'RISK_CRASH': '#8B0000'
    }
    
    for signal, color in colors.items():
        mask = df['Signal'] == signal
        if mask.any():
            fig.add_trace(go.Scatter(
                x=df[mask].index,
                y=df[mask]['Exposure'] * 100,
                mode='markers',
                name=signal,
                marker=dict(color=color, size=4),
                showlegend=True
            ))
    
    fig.update_layout(
        title='Esposizione Dinamica SPX (%)',
        xaxis_title='Data',
        yaxis_title='Esposizione (%)',
        hovermode='x unified',
        template='plotly_white',
        height=500
    )
    
    return fig

def create_drawdown_chart(df):
    """Crea grafico confronto drawdown"""
    
    # Calcola drawdown
    def calc_dd(equity):
        peak = equity.expanding(min_periods=1).max()
        dd = (equity - peak) / peak * 100
        return dd
    
    dd_bh = calc_dd(df['BuyHold_Equity'])
    dd_strategy = calc_dd(df['Strategy_Equity'])
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=dd_bh,
        name='Buy & Hold',
        fill='tozeroy',
        line=dict(color='#6c757d', width=1)
    ))
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=dd_strategy,
        name='Strategia',
        fill='tozeroy',
        line=dict(color='#28a745', width=1)
    ))
    
    fig.update_layout(
        title='Confronto Drawdown',
        xaxis_title='Data',
        yaxis_title='Drawdown (%)',
        hovermode='x unified',
        template='plotly_white',
        height=500
    )
    
    return fig

def create_breadth_chart(df):
    """Crea grafico Market Breadth"""
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Breadth_Pct_Above_MA125'] * 100,
        name='% Indici > MA125',
        line=dict(color='#667eea', width=2)
    ))
    
    # Soglie
    fig.add_hline(y=BottomThresholds.STRONG_BREADTH * 100, 
                  line_dash="dash", line_color="green",
                  annotation_text="Bottom Forte")
    fig.add_hline(y=BottomThresholds.MODERATE_BREADTH * 100,
                  line_dash="dash", line_color="lightgreen",
                  annotation_text="Bottom Moderato")
    fig.add_hline(y=RiskThresholds.EUPHORIA_BREADTH * 100,
                  line_dash="dash", line_color="red",
                  annotation_text="Euforia")
    
    fig.update_layout(
        title='Market Breadth: % Indici sopra MA125',
        xaxis_title='Data',
        yaxis_title='% Indici',
        hovermode='x unified',
        template='plotly_white',
        height=500
    )
    
    return fig

def create_rsi_chart(df):
    """Crea grafico RSI medio"""
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Breadth_Avg_RSI_20'],
        name='RSI Medio Globale',
        line=dict(color='#f39c12', width=2)
    ))
    
    # Soglie
    fig.add_hline(y=BottomThresholds.STRONG_RSI,
                  line_dash="dash", line_color="green",
                  annotation_text="Bottom Forte")
    fig.add_hline(y=BottomThresholds.MODERATE_RSI,
                  line_dash="dash", line_color="lightgreen",
                  annotation_text="Bottom Moderato")
    fig.add_hline(y=RiskThresholds.DETERIORATION_RSI,
                  line_dash="dash", line_color="red",
                  annotation_text="Deterioramento")
    
    fig.update_layout(
        title='RSI Medio Globale (20 periodi)',
        xaxis_title='Data',
        yaxis_title='RSI',
        hovermode='x unified',
        template='plotly_white',
        height=500
    )
    
    return fig

def create_correlation_chart(df):
    """Crea grafico correlazione"""
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Breadth_Avg_Corr_30D'],
        name='Correlazione Rolling 30D',
        line=dict(color='#e74c3c', width=2)
    ))
    
    # Soglie
    fig.add_hline(y=BottomThresholds.STRONG_CORR,
                  line_dash="dash", line_color="green",
                  annotation_text="Bottom Forte")
    fig.add_hline(y=BottomThresholds.MODERATE_CORR,
                  line_dash="dash", line_color="lightgreen",
                  annotation_text="Bottom Moderato")
    fig.add_hline(y=RiskThresholds.EUPHORIA_CORR,
                  line_dash="dash", line_color="red",
                  annotation_text="Euforia")
    
    fig.update_layout(
        title='Correlazione Rolling 30 Giorni',
        xaxis_title='Data',
        yaxis_title='Correlazione',
        hovermode='x unified',
        template='plotly_white',
        height=500
    )
    
    return fig

# ==============================================================================
# MAIN APP
# ==============================================================================

def main():
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📈 Dashboard Analisi Integrata SPX - Kriterion Quant</h1>
        <h2>Sistema V4.0 - Timing SPX con Confluenze Breadth Globale</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/200x80/667eea/ffffff?text=Kriterion+Quant", 
                 width="stretch")
        
        st.markdown("---")
        st.markdown("### ⚙️ Controlli")
        
        if st.button("🔄 Aggiorna Dati", width="stretch"):
            st.cache_data.clear()
            st.rerun()
        
        st.markdown("---")
        st.markdown("### 📊 Info Sistema")
        st.markdown(f"""
        - **Indici monitorati:** {len(INDICES)}
        - **Periodo dati:** dal {START_DATE}
        - **Capitale iniziale:** ${INITIAL_CAPITAL:,}
        - **Versione:** V4.0
        """)
        
        st.markdown("---")
        st.markdown("### 🎯 Soglie Configurate")
        
        with st.expander("🟢 BOTTOM Forte"):
            st.markdown(f"""
            - Breadth ≤ {BottomThresholds.STRONG_BREADTH:.1%}
            - RSI ≤ {BottomThresholds.STRONG_RSI}
            - Corr > {BottomThresholds.STRONG_CORR:.2f}
            - **Esposizione:** {BottomThresholds.STRONG_EXPOSURE:.0%}
            """)
        
        with st.expander("🟢 BOTTOM Moderato"):
            st.markdown(f"""
            - Breadth ≤ {BottomThresholds.MODERATE_BREADTH:.1%}
            - RSI ≤ {BottomThresholds.MODERATE_RSI}
            - Corr > {BottomThresholds.MODERATE_CORR:.2f}
            - **Esposizione:** {BottomThresholds.MODERATE_EXPOSURE:.0%}
            """)
        
        with st.expander("🔴 RISK Management"):
            st.markdown(f"""
            **Euforia:** {RiskThresholds.EUPHORIA_EXPOSURE:.0%}
            - Breadth > {RiskThresholds.EUPHORIA_BREADTH:.0%}
            - Corr < {RiskThresholds.EUPHORIA_CORR:.2f}
            
            **Deterioramento:** {RiskThresholds.DETERIORATION_EXPOSURE:.0%}
            - Breadth < {RiskThresholds.DETERIORATION_BREADTH:.0%}
            - RSI > {RiskThresholds.DETERIORATION_RSI}
            
            **Crollo:** {RiskThresholds.CRASH_EXPOSURE:.0%}
            - Breadth perde {-RiskThresholds.CRASH_BREADTH_CHANGE:.0%} in 5gg
            """)
    
    # Esegui analisi
    with st.spinner('🔄 Caricamento dati e analisi in corso...'):
        try:
            df_master, metrics, current_state = run_complete_analysis()
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            st.error(f"❌ Errore durante l'analisi: {e}")
            st.code(error_details, language="python")
            st.stop()
    
    # Info segnale
    signal_info = get_signal_info(current_state['Signal'])
    
    # Summary boxes
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-box">
            <h3>Segnale Attuale</h3>
            <div class="{signal_info['class']}" style="font-size: 2em; margin-top: 1rem;">
                {signal_info['emoji']} {signal_info['phase']}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-box">
            <h3>Azione Suggerita</h3>
            <div style="font-size: 1.5em; margin-top: 1rem; color: #333;">
                {signal_info['action']}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-box">
            <h3>Dati Aggiornati Al</h3>
            <div style="font-size: 1.8em; margin-top: 1rem; color: #333;">
                {current_state['date'].strftime('%d/%m/%Y')}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Tabs principali
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Dashboard Principale",
        "📈 Analisi Dettagliata",
        "💡 Metodologia",
        "💾 Export Dati"
    ])
    
    with tab1:
        # Stato corrente
        st.markdown("### 🎯 Stato Attuale del Mercato")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("S&P 500", f"${current_state['SPX_Price']:,.2f}")
        col2.metric("Breadth", f"{current_state['Breadth_Pct_Above_MA125']:.1%}")
        col3.metric("RSI Medio", f"{current_state['Breadth_Avg_RSI_20']:.1f}")
        col4.metric("Correlazione", f"{current_state['Breadth_Avg_Corr_30D']:.3f}")
        
        st.markdown("---")
        
        # Performance
        st.markdown("### 📈 Performance Sistema")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Buy & Hold SPX")
            st.metric("CAGR", f"{metrics['cagr_bh']*100:.2f}%")
            st.metric("Volatilità", f"{metrics['vol_bh']*100:.2f}%")
            st.metric("Sharpe Ratio", f"{metrics['sharpe_bh']:.2f}")
            st.metric("Max Drawdown", f"{metrics['max_dd_bh']*100:.2f}%")
            st.metric("Capitale Finale", f"${metrics['final_bh']:,.0f}")
        
        with col2:
            st.markdown("#### Strategia Dinamica")
            st.metric("CAGR", f"{metrics['cagr_strategy']*100:.2f}%",
                     delta=f"{(metrics['cagr_strategy']-metrics['cagr_bh'])*100:+.2f}%")
            st.metric("Volatilità", f"{metrics['vol_strategy']*100:.2f}%",
                     delta=f"{(metrics['vol_strategy']-metrics['vol_bh'])*100:+.2f}%")
            st.metric("Sharpe Ratio", f"{metrics['sharpe_strategy']:.2f}",
                     delta=f"{metrics['sharpe_strategy']-metrics['sharpe_bh']:+.2f}")
            st.metric("Max Drawdown", f"{metrics['max_dd_strategy']*100:.2f}%",
                     delta=f"{(metrics['max_dd_strategy']-metrics['max_dd_bh'])*100:+.2f}%")
            st.metric("Capitale Finale", f"${metrics['final_strategy']:,.0f}",
                     delta=f"${metrics['final_strategy']-metrics['final_bh']:+,.0f}")
        
        st.markdown("---")
        
        # Grafici principali
        st.markdown("### 💰 Equity Curves")
        st.plotly_chart(create_equity_chart(df_master), width="stretch")
        
        st.markdown("### 📊 Esposizione Dinamica")
        st.plotly_chart(create_exposure_chart(df_master), width="stretch")
        
        st.markdown("### 📉 Drawdown Comparison")
        st.plotly_chart(create_drawdown_chart(df_master), width="stretch")
    
    with tab2:
        st.markdown("### 📊 Indicatori Market Breadth")
        
        st.plotly_chart(create_breadth_chart(df_master), width="stretch")
        st.plotly_chart(create_rsi_chart(df_master), width="stretch")
        st.plotly_chart(create_correlation_chart(df_master), width="stretch")
        
        st.markdown("---")
        
        # Distribuzione segnali
        st.markdown("### 📈 Distribuzione Segnali Storici")
        
        signal_counts = df_master['Signal'].value_counts()
        signal_pct = (signal_counts / len(df_master) * 100).round(1)
        
        df_signals = pd.DataFrame({
            'Segnale': signal_counts.index,
            'Giorni': signal_counts.values,
            'Percentuale': [f"{p:.1f}%" for p in signal_pct.values]
        })
        
        st.dataframe(df_signals, width="stretch", hide_index=True)
    
    with tab3:
        st.markdown("### 💡 Logica del Sistema")
        
        st.markdown("""
        <div class="info-box">
        <h3>⚙️ Come Funziona</h3>
        <p>
        Il sistema analizza <strong>13 indici azionari globali</strong> per identificare 
        opportunità di timing su <strong>S&P 500 (SPX)</strong>. Utilizza tre indicatori chiave:
        </p>
        <ul>
            <li><strong>Market Breadth:</strong> % di indici sopra la loro media mobile 125 giorni</li>
            <li><strong>RSI Medio:</strong> Momentum medio di tutti gli indici (20 periodi)</li>
            <li><strong>Correlazione:</strong> Sincronizzazione tra mercati (rolling 30 giorni)</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        <h3>🟢 Segnali BOTTOM (Opportunità)</h3>
        <p>
        Quando i mercati globali mostrano <strong>debolezza coordinata</strong> 
        (basso breadth, RSI oversold, alta correlazione), il sistema identifica 
        opportunità di ingresso con elevata probabilità di rimbalzo:
        </p>
        <ul>
            <li><strong>BOTTOM Forte:</strong> Precision 90% - Massima esposizione (100%)</li>
            <li><strong>BOTTOM Moderato:</strong> Precision 44% - Alta esposizione (50%)</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        <h3>🔴 Risk Management (Protezione)</h3>
        <p>
        Il sistema riduce l'esposizione quando rileva <strong>condizioni di mercato sfavorevoli</strong>:
        </p>
        <ul>
            <li><strong>Euforia Decorrelata:</strong> Mercati fortissimi ma non sincronizzati (rischio divergenza)</li>
            <li><strong>Deterioramento:</strong> Mercati deboli nonostante momentum positivo (rischio correzione)</li>
            <li><strong>Crollo Rapido:</strong> Perdita improvvisa di breadth (alta volatilità)</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        <h3>📚 Come Usare il Sistema</h3>
        <ol>
            <li><strong>Consulta il segnale attuale</strong> nella dashboard principale</li>
            <li><strong>Verifica lo stato dei 3 indicatori</strong> per comprendere il contesto</li>
            <li><strong>Applica l'esposizione suggerita</strong> tramite ETF SPY, futures ES, o opzioni</li>
            <li><strong>Rivaluta periodicamente</strong> (settimanale o giornaliero a tua scelta)</li>
            <li><strong>Rispetta i segnali di risk management</strong> per proteggere il capitale</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)
        
        st.warning("""
        ⚠️ **DISCLAIMER**: Questo sistema è uno strumento educativo e di ricerca. 
        Non costituisce consulenza finanziaria. Le performance passate non garantiscono 
        risultati futuri. Ogni investimento comporta rischi, inclusa la perdita del capitale.
        """)
    
    with tab4:
        st.markdown("### 💾 Export Dati per Analisi LLM")
        
        st.info("""
        📥 Scarica tutti i dati del sistema in formato JSON per analisi approfondite 
        con Large Language Models (Claude, GPT-4, ecc.) o per integrazioni personalizzate.
        """)
        
        # Genera JSON
        json_data = export_to_json(df_master, metrics, current_state)
        json_str = json.dumps(json_data, indent=2)
        
        # Statistiche JSON
        col1, col2, col3 = st.columns(3)
        col1.metric("Dimensione", f"{len(json_str) / 1024:.1f} KB")
        col2.metric("Giorni Storici", f"{len(df_master):,}")
        col3.metric("Data Export", datetime.now().strftime('%d/%m/%Y %H:%M'))
        
        st.markdown("---")
        
        # Preview JSON
        with st.expander("👁️ Preview JSON (primi 50 righe)"):
            st.code(json_str[:2000] + "\n...", language="json")
        
        # Download button
        st.download_button(
            label="📥 Download JSON Completo",
            data=json_str,
            file_name=f"spx_system_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            width="stretch"
        )
        
        st.markdown("---")
        
        # Contenuto JSON
        st.markdown("#### 📋 Contenuto Export JSON:")
        st.markdown("""
        - **metadata**: Info generali (periodo, versione, numero indici)
        - **current_state**: Stato attuale del mercato e segnale attivo
        - **performance_metrics**: Metriche backtest complete (CAGR, Sharpe, DD, ecc.)
        - **configuration**: Tutte le soglie e parametri configurati
        - **historical_data**: Serie storiche complete (prezzi, indicatori, esposizione)
        - **signal_distribution**: Distribuzione storica dei segnali
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div class="footer">
        <p><strong>Kriterion Quant</strong> - Sistema V4.0</p>
        <p>Dashboard Analisi Integrata SPX con Confluenze Breadth Globale</p>
        <p style="font-size: 0.9em; margin-top: 1rem;">
            ⚠️ DISCLAIMER: Questo strumento è fornito esclusivamente a scopo educativo e di ricerca. 
            Non costituisce consulenza finanziaria, fiscale o legale. Le performance passate non 
            garantiscono risultati futuri. Ogni investimento comporta rischi, inclusa la perdita 
            del capitale investito. Consulta sempre un consulente finanziario professionista prima 
            di prendere decisioni di investimento.
        </p>
        <p style="font-size: 0.8em; margin-top: 1rem; opacity: 0.7;">
            © 2026 Kriterion Quant - Tutti i diritti riservati
        </p>
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# RUN APP
# ==============================================================================

if __name__ == "__main__":
    main()
