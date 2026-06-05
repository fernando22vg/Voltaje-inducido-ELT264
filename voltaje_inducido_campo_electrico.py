"""
voltaje_inducido_campo_electrico.py
ELT-264 — Líneas de Transmisión de Energía Eléctrica — UMSA
Deducción teórica del voltaje inducido por campo eléctrico (acoplamiento capacitivo)
en un conductor desenergizado paralelo a conductores energizados.

Referencia: Ing. Miranda, Capítulo 4
Ejecutar con: streamlit run voltaje_inducido_campo_electrico.py
"""

# ─── Importaciones ─────────────────────────────────────────────────────────────
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

# ─── Constantes de color ───────────────────────────────────────────────────────
COLOR_CE     = '#1565C0'   # azul oscuro  → conductores energizados
COLOR_NE     = '#C62828'   # rojo oscuro  → conductor desenergizado
COLOR_IMAGEN = '#757575'   # gris         → imágenes eléctricas bajo suelo
COLOR_MUTUA  = '#6A1B9A'   # violeta      → capacitancias mutuas
COLOR_TIERRA = '#4E342E'   # marrón       → tierra

# Estilo global matplotlib
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor']   = '#f8f9fa'
plt.rcParams['font.family']      = 'serif'

PLOTLY_LAYOUT = dict(
    plot_bgcolor='white',
    paper_bgcolor='white',
    font=dict(family='serif', size=13),
    margin=dict(l=40, r=20, t=50, b=40),
)

SECCIONES = [
    "1 · Introducción",
    "2 · ¿Qué es jω?",
    "3 · De Q=CV a I=jωCV",
    "4 · Capacitancia vs Admitancia",
    "5 · Circuito equivalente capacitivo",
    "6 · Ecuación nodal [I]=[Y][V]",
    "7 · Partición de la matriz",
    "8 · Condición I_ne = 0",
    "9 · Cancelación de jω",
    "10 · Ecuación final [V_ne]",
    "11 · ¿Campo E o campo B?",
]


# ═══════════════════════════════════════════════════════════════════════════════
# FUNCIONES DE DIBUJO
# ═══════════════════════════════════════════════════════════════════════════════

def plot_geometry():
    """
    Diagrama esquemático en Plotly — limpio y pedagógico.
    Conductores, campo E, imágenes y cotas geométricas.
    Menos realista, más fácil de entender.
    """
    x_ce1, h_ce1 = 2.5, 3.4
    x_ce2, h_ce2 = 5.5, 4.2
    x_ne,  h_ne  = 9.0, 3.0
    XL, XR = 0.5, 11.5

    fig = go.Figure()

    # ── Fondos zonales ─────────────────────────────────────────────────────────
    # Cielo (azul muy suave)
    fig.add_shape(type='rect', x0=XL, y0=0, x1=XR, y1=6.0,
                  fillcolor='#EEF4FF', line_width=0, layer='below')
    # Tierra (marrón muy suave)
    fig.add_shape(type='rect', x0=XL, y0=-2.2, x1=XR, y1=0,
                  fillcolor='#F5EFE6', line_width=0, layer='below')

    # ── Línea del suelo ────────────────────────────────────────────────────────
    fig.add_shape(type='line', x0=XL, y0=0, x1=XR, y1=0,
                  line=dict(color=COLOR_TIERRA, width=3))
    for xi in np.arange(0.8, XR, 0.55):
        fig.add_shape(type='line', x0=xi, y0=0, x1=xi - 0.22, y1=-0.30,
                      line=dict(color=COLOR_TIERRA, width=1.1))
    fig.add_annotation(x=XL + 0.3, y=-0.35,
                       text='<b>Suelo  —  V = 0  (referencia)</b>',
                       showarrow=False, font=dict(color=COLOR_TIERRA, size=12),
                       xanchor='left')

    # ── Postes de soporte (líneas punteadas verticales) ────────────────────────
    for x, h in [(x_ce1, h_ce1), (x_ce2, h_ce2), (x_ne, h_ne)]:
        fig.add_shape(type='line', x0=x, y0=0.02, x1=x, y1=h - 0.22,
                      line=dict(color='#BDBDBD', width=1.8, dash='dot'))

    # ── Cotas de altura (flechas dobles) ──────────────────────────────────────
    cota_data = [
        (x_ce1 - 0.72, h_ce1, COLOR_CE,  'h<sub>ce₁</sub>'),
        (x_ce2 - 0.72, h_ce2, COLOR_CE,  'h<sub>ce₂</sub>'),
        (x_ne  - 0.72, h_ne,  COLOR_NE,  'h<sub>ne</sub>'),
    ]
    for xc, h, col, lbl in cota_data:
        fig.add_annotation(x=xc, y=h, ax=xc, ay=0.05,
                           axref='x', ayref='y',
                           arrowhead=2, arrowsize=1.1, arrowwidth=1.8,
                           arrowcolor=col, arrowside='start+end')
        fig.add_annotation(x=xc - 0.22, y=h / 2,
                           text=f'<i>{lbl}</i>',
                           showarrow=False,
                           font=dict(color=col, size=12))

    # ── Cotas horizontales (distancias entre conductores) ─────────────────────
    for x1, x2, yy, lbl in [
        (x_ce1, x_ce2, -0.50, 'd<sub>12</sub>'),
        (x_ce2, x_ne,  -0.75, 'd<sub>2,ne</sub>'),
        (x_ce1, x_ne,  -1.00, 'd<sub>1,ne</sub>'),
    ]:
        fig.add_annotation(x=x2 - 0.1, y=yy, ax=x1 + 0.1, ay=yy,
                           axref='x', ayref='y',
                           arrowhead=2, arrowsize=1.0, arrowwidth=1.5,
                           arrowcolor='#777', arrowside='start+end')
        fig.add_annotation(x=(x1 + x2) / 2, y=yy - 0.22,
                           text=f'<i><b>{lbl}</b></i>',
                           showarrow=False, font=dict(color='#555', size=11))

    # ── Líneas de campo E  (ce1 → ne, 2 flechas naranjas) ─────────────────────
    t_vals = [0.35, 0.65]
    for t in t_vals:
        xm = x_ce1 + t * (x_ne - x_ce1)
        ym = h_ce1 + t * (h_ne - h_ce1) + 0.7 * np.sin(np.pi * t)
        fig.add_annotation(x=xm + 0.40, y=ym, ax=xm - 0.40, ay=ym,
                           axref='x', ayref='y',
                           arrowhead=3, arrowsize=1.3, arrowwidth=2.8,
                           arrowcolor='#E65100')
    # ce2 → ne
    for t in [0.38, 0.68]:
        xm = x_ce2 + t * (x_ne - x_ce2)
        ym = h_ce2 + t * (h_ne - h_ce2) - 0.3 * np.sin(np.pi * t)
        fig.add_annotation(x=xm + 0.35, y=ym, ax=xm - 0.35, ay=ym,
                           axref='x', ayref='y',
                           arrowhead=3, arrowsize=1.3, arrowwidth=2.8,
                           arrowcolor='#FF8F00')
    # Etiqueta campo E
    fig.add_annotation(x=6.4, y=4.35,
                       text='<b>→  campo  Ē</b>',
                       showarrow=False,
                       font=dict(color='#E65100', size=14),
                       bgcolor='rgba(255,255,255,0.88)',
                       bordercolor='#E65100', borderwidth=2, borderpad=5)

    # ── Conductores (círculos grandes, interactivos) ───────────────────────────
    cond_data = [
        (x_ce1, h_ce1, COLOR_CE, 'ce₁', 'V<sub>ce₁</sub>', f'h = {h_ce1} u.a.'),
        (x_ce2, h_ce2, COLOR_CE, 'ce₂', 'V<sub>ce₂</sub>', f'h = {h_ce2} u.a.'),
        (x_ne,  h_ne,  COLOR_NE, 'ne',  'V<sub>ne</sub> = ?', f'h = {h_ne} u.a.  |  I = 0'),
    ]
    for x, h, col, lbl, vlbl, info in cond_data:
        fig.add_trace(go.Scatter(
            x=[x], y=[h],
            mode='markers+text',
            marker=dict(color=col, size=38,
                        line=dict(color='white', width=3)),
            text=[f'<b>{lbl}</b>'],
            textfont=dict(color='white', size=14, family='Arial Black'),
            textposition='middle center',
            showlegend=True,
            name='Conductor energizado (ce)' if col == COLOR_CE else 'Conductor desenergizado (ne)',
            hovertemplate=(
                f'<b>{lbl}</b><br>'
                f'Voltaje: {vlbl}<br>'
                f'{info}<extra></extra>'
            )
        ))
        # Etiqueta de voltaje encima
        fig.add_annotation(x=x, y=h + 0.62,
                           text=f'<b>{vlbl}</b>',
                           showarrow=False,
                           font=dict(color=col, size=12),
                           bgcolor='white',
                           bordercolor=col, borderwidth=2, borderpad=5)

    # ── Imágenes eléctricas (bajo tierra, mayor profundidad para separación) ─────
    scale_img = 0.72
    for x, h, col, lbl, vlbl, info in cond_data:
        y_img = -h * scale_img
        fig.add_trace(go.Scatter(
            x=[x], y=[y_img],
            mode='markers',
            marker=dict(color='white', size=24,
                        line=dict(color=COLOR_IMAGEN, width=2.5),
                        symbol='circle'),
            showlegend=False,
            hovertemplate=(
                f'Imagen eléctrica de {lbl}<br>'
                f'Profundidad = {h} u.a.<br>'
                f'Carga efectiva: −{lbl}<extra></extra>'
            )
        ))
        fig.add_shape(type='line', x0=x, y0=-0.08, x1=x, y1=y_img + 0.22,
                      line=dict(color=COLOR_IMAGEN, width=1.3, dash='dot'))
        fig.add_annotation(x=x, y=y_img - 0.35,
                           text=f'<i>−{lbl}′</i><br><small>prof = {h} u.a.</small>',
                           showarrow=False,
                           font=dict(color=COLOR_IMAGEN, size=11),
                           bgcolor='rgba(255,255,255,0.75)',
                           borderpad=2)

    # Etiqueta zona imágenes
    fig.add_annotation(x=XL + 0.3, y=-2.8,
                       text='<i>Imágenes eléctricas  (método de Maxwell)</i>',
                       showarrow=False,
                       font=dict(color=COLOR_IMAGEN, size=11),
                       xanchor='left')

    # ── Separador zona real / imágenes ─────────────────────────────────────────
    # Ya está la línea del suelo; agregar texto de separación
    fig.add_annotation(x=XR - 0.3, y=2.8,
                       text=(
                           '<b>Región aérea</b><br>'
                           '<i>Campo E induce V<sub>ne</sub></i>'
                       ),
                       showarrow=False,
                       font=dict(color='#1565C0', size=11),
                       bgcolor='rgba(255,255,255,0.8)',
                       bordercolor='#1565C0', borderwidth=1, borderpad=6,
                       xanchor='right')

    # ── Leyenda manual de imágenes ─────────────────────────────────────────────
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
                              marker=dict(color='white', size=14,
                                          line=dict(color=COLOR_IMAGEN, width=2.5),
                                          symbol='circle'),
                              name='Imágenes eléctricas (−h)'))

    # ── Layout ─────────────────────────────────────────────────────────────────
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=(
            '<b>Geometría de conductores</b>  —  '
            'Acoplamiento capacitivo por campo eléctrico<br>'
            '<sup>Conductor ne aislado:  I<sub>ne</sub> = 0  →  '
            'incógnita: V<sub>ne</sub> = ?</sup>'
        ),
        xaxis=dict(
            range=[XL, XR], showgrid=False, zeroline=False,
            tickvals=[x_ce1, x_ce2, x_ne],
            ticktext=['x<sub>ce₁</sub>', 'x<sub>ce₂</sub>', 'x<sub>ne</sub>'],
            tickfont=dict(size=13),
        ),
        yaxis=dict(
            range=[-3.6, 6.0],
            title_text='Altura (u.a.)',
            showgrid=True, gridcolor='#eee', zeroline=False,
            tickvals=[0, 1, 2, 3, 4, 5],
        ),
        height=660,
        showlegend=True,
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.9)',
                    bordercolor='#ddd', borderwidth=1),
    )
    return fig



def plot_complex_plane():
    """Plano complejo mejorado: círculo unitario, 4 potencias de j y cuadrantes."""
    fig = go.Figure()

    # ── Grid sutil ─────────────────────────────────────────────────────────────
    for v in [-1.0, -0.5, 0.5, 1.0]:
        for args in [dict(x0=v, y0=-1.55, x1=v, y1=1.55),
                     dict(x0=-1.55, y0=v, x1=1.55, y1=v)]:
            fig.add_shape(type='line', **args, line=dict(color='#e8e8e8', width=1))

    # ── Círculo unitario ────────────────────────────────────────────────────────
    theta_c = np.linspace(0, 2 * np.pi, 300)
    fig.add_trace(go.Scatter(
        x=np.cos(theta_c), y=np.sin(theta_c),
        mode='lines', line=dict(color='#bdbdbd', width=1.8, dash='dot'),
        name='|z| = 1  (círculo unitario)', showlegend=True
    ))

    # ── Ejes ────────────────────────────────────────────────────────────────────
    for x0, y0, x1, y1 in [(-1.55, 0, 1.55, 0), (0, -1.55, 0, 1.55)]:
        fig.add_shape(type='line', x0=x0, y0=y0, x1=x1, y1=y1,
                      line=dict(color='black', width=1.8))

    # ── 4 potencias de j como vectores ─────────────────────────────────────────
    potencias = [
        ( 1.0,  0.0, COLOR_CE,   'j⁰ = 1',  ( 1.10, -0.22)),
        ( 0.0,  1.0, COLOR_NE,   'j¹ = j',  ( 0.12,  1.13)),
        (-1.0,  0.0, '#616161',  'j² = −1', (-1.45, -0.22)),
        ( 0.0, -1.0, '#8E24AA',  'j³ = −j', ( 0.12, -1.20)),
    ]
    for px, py, col, lbl, (tx, ty) in potencias:
        fig.add_annotation(x=px, y=py, ax=0, ay=0, axref='x', ayref='y',
                           arrowhead=3, arrowsize=1.5, arrowwidth=3.5,
                           arrowcolor=col)
        fig.add_annotation(x=tx, y=ty, text=f'<b>{lbl}</b>',
                           showarrow=False, font=dict(color=col, size=15))

    # ── Arcos de rotación ×j (3 primeros cuartos) ──────────────────────────────
    arcos = [
        (0,         np.pi/2,     'green',   '×j → +90°', ( 0.47,  0.52)),
        (np.pi/2,   np.pi,       '#E65100', '×j → +90°', (-0.52,  0.47)),
        (np.pi,     3*np.pi/2,   '#555',    '×j → +90°', (-0.47, -0.47)),
    ]
    for t0, t1, col, lbl, (tx, ty) in arcos:
        th = np.linspace(t0, t1, 50)
        fig.add_trace(go.Scatter(
            x=0.58 * np.cos(th), y=0.58 * np.sin(th),
            mode='lines', line=dict(color=col, width=2.2, dash='dot'),
            showlegend=False
        ))
        fig.add_annotation(x=tx, y=ty, text=lbl,
                           showarrow=False, font=dict(color=col, size=11))

    # ── Etiquetas de cuadrantes ─────────────────────────────────────────────────
    for qx, qy, qt in [(0.78, 0.78, 'I'), (-0.78, 0.78, 'II'),
                        (-0.78, -0.78, 'III'), (0.78, -0.78, 'IV')]:
        fig.add_annotation(x=qx, y=qy, text=f'<span style="color:#ccc">{qt}</span>',
                           showarrow=False, font=dict(size=13))

    # ── Marcas en los ejes (±1, ±j) ────────────────────────────────────────────
    for v, lbl, xoff, yoff in [(1, '1', 0.04, -0.14), (-1, '−1', -0.18, -0.14)]:
        fig.add_annotation(x=v, y=0, text=f'<b>{lbl}</b>',
                           showarrow=False, font=dict(size=11, color='black'),
                           xshift=xoff*40, yshift=yoff*40)
    for v, lbl in [(1, 'j'), (-1, '−j')]:
        fig.add_annotation(x=0.12, y=v, text=f'<b>{lbl}</b>',
                           showarrow=False, font=dict(size=11, color='black'))

    # ── Etiquetas de ejes ────────────────────────────────────────────────────────
    fig.add_annotation(x=1.50, y=0,    text='<b>Re</b>',
                       showarrow=False, font=dict(size=15, color='black'))
    fig.add_annotation(x=0.07, y=1.50, text='<b>Im</b>',
                       showarrow=False, font=dict(size=15, color='black'))

    # ── Anotación jω ────────────────────────────────────────────────────────────
    fig.add_annotation(
        x=-1.4, y=-1.4,
        text='<b>j·ω</b> = multiplicar por j<br>+ escalar por ω',
        showarrow=False,
        font=dict(size=11, color='green'),
        bgcolor='#f1f8e9', bordercolor='green', borderwidth=1
    )

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title='Plano complejo — j rota 90° antihorario; cada j² invierte el signo',
        xaxis=dict(range=[-1.65, 1.65], showgrid=False, zeroline=False,
                   showticklabels=False),
        yaxis=dict(range=[-1.65, 1.65], showgrid=False, zeroline=False,
                   showticklabels=False, scaleanchor='x'),
        showlegend=True,
        height=460,
        legend=dict(x=0.02, y=0.02),
    )
    return fig


def plot_sinusoidal(freq=50):
    """Senoidal mejorada: amplitud, RMS, semiciclos coloreados, divisiones de T."""
    omega = 2 * np.pi * freq
    T     = 1.0 / freq
    t     = np.linspace(0, 2.5 * T, 900)
    V     = np.sin(omega * t)
    t_ms  = t * 1000
    Vrms  = 1.0 / np.sqrt(2)

    fig = go.Figure()

    # ── Rellenos de semiciclo ───────────────────────────────────────────────────
    V_pos = np.where(V >= 0, V, 0.0)
    V_neg = np.where(V <= 0, V, 0.0)
    fig.add_trace(go.Scatter(
        x=t_ms, y=V_pos, fill='tozeroy',
        fillcolor='rgba(21,101,192,0.10)',
        line=dict(color='rgba(0,0,0,0)'), mode='lines',
        showlegend=False, hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=t_ms, y=V_neg, fill='tozeroy',
        fillcolor='rgba(198,40,40,0.09)',
        line=dict(color='rgba(0,0,0,0)'), mode='lines',
        showlegend=False, hoverinfo='skip'
    ))

    # ── Curva principal ─────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=t_ms, y=V, mode='lines',
        line=dict(color=COLOR_CE, width=2.8),
        name='V(t) = Vm · sen(ωt)'
    ))

    # ── Línea horizontal V = 0 ──────────────────────────────────────────────────
    fig.add_hline(y=0, line_dash='dot', line_color='#ccc', line_width=1.2)

    # ── Líneas de amplitud pico ±Vm ─────────────────────────────────────────────
    for yv, lbl, pos in [(1.0, 'Vm (pico +)', 'right'),
                          (-1.0, '−Vm (pico −)', 'right')]:
        fig.add_hline(y=yv, line_dash='dash', line_color='green', line_width=1.3,
                      annotation_text=lbl, annotation_position=pos,
                      annotation_font_color='green', annotation_font_size=11)

    # ── Línea RMS ───────────────────────────────────────────────────────────────
    fig.add_hline(y=Vrms, line_dash='dot', line_color='#FF6F00', line_width=1.5,
                  annotation_text=f'Vrms = Vm/√2 ≈ {Vrms:.4f}',
                  annotation_position='right',
                  annotation_font_color='#FF6F00', annotation_font_size=11)

    # ── Marcadores en los picos ─────────────────────────────────────────────────
    t_pico1 = T / 4
    t_pico2 = 3 * T / 4
    fig.add_trace(go.Scatter(
        x=[t_pico1 * 1000, t_pico2 * 1000],
        y=[1.0, -1.0],
        mode='markers+text',
        marker=dict(color=['green', 'red'], size=11, symbol='circle'),
        text=['  +Vm', '  −Vm'],
        textposition='middle right',
        textfont=dict(size=11, color='green'),
        showlegend=False
    ))

    # ── Divisiones de período (T/4, T/2, 3T/4, T) ──────────────────────────────
    fracs = [(0.25, 'T/4'), (0.5, 'T/2'), (0.75, '3T/4'), (1.0, 'T')]
    for frac, lbl in fracs:
        xv = frac * T * 1000
        fig.add_shape(type='line', x0=xv, y0=-1.05, x1=xv, y1=1.05,
                      line=dict(color='#bbb', width=0.9, dash='dot'))
        fig.add_annotation(x=xv, y=-1.22, text=f'<i>{lbl}</i>',
                           showarrow=False, font=dict(size=10, color='#888'))

    # ── Cota del período T ──────────────────────────────────────────────────────
    fig.add_shape(type='line', x0=0, y0=1.18, x1=T * 1000, y1=1.18,
                  line=dict(color='gray', width=1.4, dash='dash'))
    fig.add_annotation(x=T * 500, y=1.27,
                       text=f'<b>T = {T*1000:.3f} ms</b>',
                       showarrow=False, font=dict(size=12, color='gray'))

    # ── Anotación ω ─────────────────────────────────────────────────────────────
    fig.add_annotation(
        x=2.3 * T * 1000, y=0.65,
        text=f'<b>ω = 2πf<br>= {omega:.2f} rad/s</b>',
        showarrow=False, font=dict(size=11, color=COLOR_CE),
        bgcolor='#e3f2fd', bordercolor=COLOR_CE, borderwidth=1
    )

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=(f'V(t) = Vm · sen(ωt)  |  f = {freq} Hz  |  '
               f'ω = 2πf = {omega:.2f} rad/s  |  T = {T*1000:.3f} ms'),
        xaxis_title='Tiempo (ms)',
        yaxis_title='V(t) / Vm  [p.u.]',
        height=390,
        yaxis=dict(range=[-1.42, 1.42]),
    )
    return fig


def plot_vi_capacitor():
    """Desfase 90° V-I: gráfica temporal + diagrama fasorial como subplots."""
    t = np.linspace(0, 2 * np.pi, 700)
    V = np.cos(t)
    I = -np.sin(t)   # I adelanta 90° a V

    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.63, 0.37],
        subplot_titles=('Dominio del tiempo  —  I(t) adelanta 90° a V(t)',
                        'Diagrama fasorial'),
        specs=[[{'type': 'xy'}, {'type': 'xy'}]]
    )

    # ── Relleno semiciclos V ────────────────────────────────────────────────────
    V_pos = np.where(V >= 0, V, 0.0)
    fig.add_trace(go.Scatter(x=t, y=V_pos, fill='tozeroy',
                              fillcolor='rgba(21,101,192,0.08)',
                              line=dict(color='rgba(0,0,0,0)'), mode='lines',
                              showlegend=False, hoverinfo='skip'), row=1, col=1)

    # ── Curvas temporales ───────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(x=t, y=V, mode='lines',
                              line=dict(color=COLOR_CE, width=2.8),
                              name='V(t) = Vm · cos(ωt)'), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=I, mode='lines',
                              line=dict(color=COLOR_NE, width=2.8, dash='dot'),
                              name='I(t) = ωCVm · cos(ωt + 90°)'), row=1, col=1)

    # ── Líneas verticales de pico ────────────────────────────────────────────────
    fig.add_shape(type='line', x0=0,       y0=-0.05, x1=0,       y1=1.02,
                  line=dict(color=COLOR_CE, width=1.3, dash='dot'), row=1, col=1)
    fig.add_shape(type='line', x0=np.pi/2, y0=0.05,  x1=np.pi/2, y1=-1.02,
                  line=dict(color=COLOR_NE, width=1.3, dash='dot'), row=1, col=1)

    # ── Arco de 90° entre picos ──────────────────────────────────────────────────
    th_arc = np.linspace(0, np.pi / 2, 40)
    fig.add_trace(go.Scatter(
        x=0.42 * np.cos(th_arc), y=0.42 * np.sin(th_arc),
        mode='lines', line=dict(color='green', width=2.2),
        showlegend=False, hoverinfo='skip'
    ), row=1, col=1)
    fig.add_annotation(x=0.28, y=0.52, text='<b>90°</b>',
                       showarrow=False, font=dict(color='green', size=14), row=1, col=1)

    # ── Marcadores de pico ────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=[0, np.pi / 2], y=[1, -1], mode='markers+text',
        marker=dict(color=[COLOR_CE, COLOR_NE], size=11),
        text=['  Vm', '  Im'], textposition='middle right',
        textfont=dict(size=11), showlegend=False
    ), row=1, col=1)

    # ── Divisiones de eje x ────────────────────────────────────────────────────
    fig.update_xaxes(
        tickvals=[0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi],
        ticktext=['0', 'π/2', 'π', '3π/2', '2π'],
        title_text='ωt (rad)', row=1, col=1
    )
    fig.update_yaxes(range=[-1.42, 1.42], title_text='Amplitud normalizada [p.u.]',
                     row=1, col=1)

    # ══ DIAGRAMA FASORIAL ════════════════════════════════════════════════════════
    # Círculo unitario de referencia
    th_circ = np.linspace(0, 2 * np.pi, 200)
    fig.add_trace(go.Scatter(
        x=0.92 * np.cos(th_circ), y=0.92 * np.sin(th_circ),
        mode='lines', line=dict(color='#ddd', width=1.5, dash='dot'),
        showlegend=False, hoverinfo='skip'
    ), row=1, col=2)

    # Ejes
    for x0, y0, x1, y1 in [(-1.15, 0, 1.15, 0), (0, -1.15, 0, 1.15)]:
        fig.add_shape(type='line', x0=x0, y0=y0, x1=x1, y1=y1,
                      line=dict(color='#aaa', width=1.2), row=1, col=2)

    # Fasor V (∠0°, eje real)
    fig.add_annotation(x=0.92, y=0, ax=0, ay=0, axref='x2', ayref='y2',
                       arrowhead=4, arrowsize=1.4, arrowwidth=3.5,
                       arrowcolor=COLOR_CE, row=1, col=2)
    fig.add_annotation(x=1.0, y=-0.16, text='<b>V</b><br>∠0°',
                       showarrow=False, font=dict(color=COLOR_CE, size=13),
                       row=1, col=2)

    # Fasor I (∠+90°, eje imaginario positivo — adelantado)
    fig.add_annotation(x=0, y=0.92, ax=0, ay=0, axref='x2', ayref='y2',
                       arrowhead=4, arrowsize=1.4, arrowwidth=3.5,
                       arrowcolor=COLOR_NE, row=1, col=2)
    fig.add_annotation(x=0.12, y=1.02, text='<b>I</b><br>∠+90°',
                       showarrow=False, font=dict(color=COLOR_NE, size=13),
                       row=1, col=2)

    # Arco 90° en el fasorial
    th_f = np.linspace(0, np.pi / 2, 40)
    fig.add_trace(go.Scatter(
        x=0.28 * np.cos(th_f), y=0.28 * np.sin(th_f),
        mode='lines', line=dict(color='green', width=2.0),
        showlegend=False, hoverinfo='skip'
    ), row=1, col=2)
    fig.add_annotation(x=0.22, y=0.30, text='<b>90°</b>',
                       showarrow=False, font=dict(color='green', size=12),
                       row=1, col=2)

    # Etiquetas de ejes del fasorial
    fig.add_annotation(x=1.10, y=0,    text='<b>Re</b>',
                       showarrow=False, font=dict(size=13), row=1, col=2)
    fig.add_annotation(x=0.07, y=1.10, text='<b>Im</b>',
                       showarrow=False, font=dict(size=13), row=1, col=2)

    # Ecuación Y = jωC en el fasorial
    fig.add_annotation(
        x=-1.0, y=-0.85,
        text='<b>I = jωC·V</b><br>Y = jωC',
        showarrow=False, font=dict(size=11, color='#555'),
        bgcolor='#f9f9f9', bordercolor='#aaa', borderwidth=1,
        row=1, col=2
    )

    fig.update_xaxes(range=[-1.22, 1.22], showgrid=False, zeroline=False,
                     showticklabels=False, row=1, col=2)
    fig.update_yaxes(range=[-1.22, 1.22], showgrid=False, zeroline=False,
                     showticklabels=False, scaleanchor='x2', row=1, col=2)

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title='V e I en un capacitor — la corriente se adelanta 90° al voltaje  (I = jωC·V)',
        height=420,
        legend=dict(x=0.02, y=0.98),
    )
    return fig


def plot_capacitive_circuit():
    """Circuito equivalente capacitivo — versión Plotly limpia y pedagógica."""
    fig = go.Figure()

    x_ce1, x_ce2, x_ne = 2.8, 6.5, 10.8
    y_node = 7.8
    y_gnd  = 1.3
    div_x  = 9.0

    # ── Fondos ─────────────────────────────────────────────────────────────────
    fig.add_shape(type='rect', x0=-1, y0=0, x1=div_x, y1=10.8,
                  fillcolor='#E3F2FD', opacity=0.35, line_width=0, layer='below')
    fig.add_shape(type='rect', x0=div_x, y0=0, x1=13.5, y1=10.8,
                  fillcolor='#FFF3E0', opacity=0.40, line_width=0, layer='below')
    fig.add_shape(type='line', x0=div_x, y0=0, x1=div_x, y1=10.8,
                  line=dict(color='#B0BEC5', width=2, dash='dash'))

    # ── Tierra ─────────────────────────────────────────────────────────────────
    fig.add_shape(type='line', x0=-0.8, y0=y_gnd, x1=13.2, y1=y_gnd,
                  line=dict(color=COLOR_TIERRA, width=3.5))
    for xi in np.arange(-0.4, 13.2, 0.55):
        fig.add_shape(type='line', x0=xi, y0=y_gnd, x1=xi - 0.20, y1=y_gnd - 0.28,
                      line=dict(color=COLOR_TIERRA, width=1.2))

    # ── Helpers: dibujo de capacitores ─────────────────────────────────────────
    def cap_vert(x, y_top, y_bot, lbl_C, lbl_Y, color, half=0.48, gap=0.28):
        mid = (y_top + y_bot) / 2
        for y0_, y1_ in [(y_top, mid + gap), (mid - gap, y_bot)]:
            fig.add_shape(type='line', x0=x, y0=y0_, x1=x, y1=y1_,
                          line=dict(color='#444', width=2.2))
        for yp in [mid + gap, mid - gap]:
            fig.add_shape(type='line', x0=x - half, y0=yp, x1=x + half, y1=yp,
                          line=dict(color=color, width=7))
        fig.add_annotation(x=x + 0.62, y=mid + 0.20, text=f'<b>{lbl_C}</b>',
                           showarrow=False, font=dict(color=color, size=12), xanchor='left')
        fig.add_annotation(x=x + 0.62, y=mid - 0.28, text=f'<i>{lbl_Y}</i>',
                           showarrow=False, font=dict(color='#777', size=9), xanchor='left')

    def cap_horiz(x_l, x_r, y, lbl_C, lbl_Y, color, half=0.45, gap=0.26, y_off=0.55):
        mid = (x_l + x_r) / 2
        for x0_, x1_ in [(x_l, mid - gap), (mid + gap, x_r)]:
            fig.add_shape(type='line', x0=x0_, y0=y, x1=x1_, y1=y,
                          line=dict(color='#444', width=2.2))
        for xp in [mid - gap, mid + gap]:
            fig.add_shape(type='line', x0=xp, y0=y - half, x1=xp, y1=y + half,
                          line=dict(color=color, width=7))
        fig.add_annotation(x=mid + 0.55, y=y, ax=mid - 0.55, ay=y,
                           axref='x', ayref='y',
                           arrowhead=3, arrowsize=1.0, arrowwidth=2.0, arrowcolor=color)
        fig.add_annotation(x=mid, y=y + y_off, text=f'<b>{lbl_C}</b>',
                           showarrow=False, font=dict(color=color, size=11), xanchor='center')
        fig.add_annotation(x=mid, y=y + y_off + 0.40, text=f'<i>{lbl_Y}</i>',
                           showarrow=False, font=dict(color='#777', size=9), xanchor='center')

    # ── Capacitores propios (vertical → tierra) ────────────────────────────────
    cap_vert(x_ce1, y_node - 0.28, y_gnd + 0.14,
             'C<sub>11</sub>', 'Y<sub>11</sub>=jωC<sub>11</sub>', COLOR_CE)
    cap_vert(x_ce2, y_node - 0.28, y_gnd + 0.14,
             'C<sub>22</sub>', 'Y<sub>22</sub>=jωC<sub>22</sub>', COLOR_CE)
    cap_vert(x_ne,  y_node - 0.28, y_gnd + 0.14,
             'C<sub>ne,ne</sub>', 'Y<sub>ne</sub>=jωC<sub>ne,ne</sub>', COLOR_NE)

    # ── Flechas de corriente descendente ──────────────────────────────────────
    for x, col, lbl in [(x_ce1, COLOR_CE,  'I<sub>11</sub>'),
                         (x_ce2, COLOR_CE,  'I<sub>22</sub>'),
                         (x_ne,  COLOR_NE,  'I<sub>ne,ne</sub>')]:
        fig.add_annotation(x=x, y=3.8, ax=x, ay=5.0,
                           axref='x', ayref='y',
                           arrowhead=3, arrowsize=1.2, arrowwidth=2.5, arrowcolor=col)
        fig.add_annotation(x=x + 0.28, y=4.4, text=f'<i>{lbl}</i>',
                           showarrow=False, font=dict(color=col, size=10), xanchor='left')

    # ── Capacitores mutuos (horizontal) ───────────────────────────────────────
    cap_horiz(x_ce1 + 0.28, x_ce2 - 0.28, y_node,
              'C<sub>12</sub>', 'Y<sub>12</sub>=jωC<sub>12</sub>', COLOR_MUTUA)
    cap_horiz(x_ce2 + 0.28, x_ne  - 0.28, y_node,
              'C<sub>2,ne</sub>', 'Y<sub>2,ne</sub>=jωC<sub>2,ne</sub>', COLOR_MUTUA)

    # C1,ne: arco por arriba
    y_arc = 9.35
    for xv in [x_ce1, x_ne]:
        fig.add_shape(type='line', x0=xv, y0=y_node + 0.28, x1=xv, y1=y_arc,
                      line=dict(color='#444', width=2.0))
    cap_horiz(x_ce1, x_ne, y_arc,
              'C<sub>1,ne</sub>', 'Y<sub>1,ne</sub>=jωC<sub>1,ne</sub>',
              COLOR_MUTUA, y_off=0.40)

    # ── Nodos (puntos de conexión) ─────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=[x_ce1, x_ce2], y=[y_node, y_node],
        mode='markers+text',
        marker=dict(size=22, color=COLOR_CE, line=dict(color='white', width=2.5)),
        text=['ce₁', 'ce₂'], textposition='top center',
        textfont=dict(size=14, color=COLOR_CE, family='Arial Black'),
        name='Conductores energizados (ce)',
        hovertemplate='<b>%{text}</b><br>Voltaje conocido<extra></extra>',
    ))
    fig.add_trace(go.Scatter(
        x=[x_ne], y=[y_node],
        mode='markers+text',
        marker=dict(size=22, color=COLOR_NE, line=dict(color='white', width=2.5)),
        text=['ne'], textposition='top center',
        textfont=dict(size=14, color=COLOR_NE, family='Arial Black'),
        name='Conductor desenergizado (ne)',
        hovertemplate='<b>ne</b><br>V<sub>ne</sub> = ? (incógnita)<extra></extra>',
    ))

    # Etiquetas de voltaje sobre cada nodo
    for x, lbl, col in [(x_ce1, 'V<sub>ce₁</sub>', COLOR_CE),
                         (x_ce2, 'V<sub>ce₂</sub>', COLOR_CE),
                         (x_ne,  'V<sub>ne</sub> = ?', COLOR_NE)]:
        fig.add_annotation(x=x, y=y_node + 1.05, text=f'<b>{lbl}</b>',
                           showarrow=False, font=dict(color=col, size=12),
                           bgcolor='white', bordercolor=col, borderwidth=2, borderpad=4)

    # ── Condición I_ne = 0 ─────────────────────────────────────────────────────
    fig.add_annotation(
        x=x_ne, y=2.90,
        text='<b>I<sub>ne</sub> = 0</b><br>Conductor aislado<br>→ sin camino a tierra',
        showarrow=True, ax=x_ne, ay=7.0, axref='x', ayref='y',
        arrowhead=2, arrowwidth=1.5, arrowcolor='green', arrowdash='dash',
        font=dict(color='green', size=12),
        bgcolor='#e8f5e9', bordercolor='green', borderwidth=2, borderpad=6,
    )

    # ── Etiquetas de región ────────────────────────────────────────────────────
    fig.add_annotation(x=4.5, y=10.55,
                       text='<b>Región  ce  (conductores energizados — voltajes conocidos)</b>',
                       showarrow=False, font=dict(color=COLOR_CE, size=11),
                       bgcolor='#E3F2FD', bordercolor=COLOR_CE, borderwidth=1.5, borderpad=4)
    fig.add_annotation(x=11.2, y=10.55,
                       text='<b>Región  ne</b><br>(incógnita: V<sub>ne</sub>)',
                       showarrow=False, font=dict(color=COLOR_NE, size=10),
                       bgcolor='#FFF3E0', bordercolor=COLOR_NE, borderwidth=1.5, borderpad=4)
    fig.add_annotation(x=-0.8, y=y_gnd - 0.55,
                       text='<i>Tierra — plano de referencia  V = 0</i>',
                       showarrow=False, font=dict(color=COLOR_TIERRA, size=9), xanchor='left')

    # ── Panel de propiedades ───────────────────────────────────────────────────
    props_html = (
        '<b>Propiedades de C<sub>ij</sub></b><br>'
        f'<span style="color:{COLOR_CE}">▮</span> C<sub>ii</sub> diagonal: &gt; 0 F/m<br>'
        f'<span style="color:{COLOR_MUTUA}">▮</span> C<sub>ij</sub> (i≠j): &lt; 0 F/m<br>'
        '▮ C = [P]⁻¹ — solo geometría<br>'
        '▮ Y<sub>ij</sub> = jωC<sub>ij</sub> admitancia<br><br>'
        '<span style="color:green"><b>Ecuación nodal (4.2):</b><br>'
        '[I] = [Y][V] = jω[C][V]</span>'
    )
    fig.add_annotation(x=-0.8, y=6.5, text=props_html,
                       showarrow=False, font=dict(size=9.5),
                       bgcolor='#F5F5F5', bordercolor='#BDBDBD', borderwidth=1.5,
                       borderpad=8, xanchor='left', align='left')

    # ── Layout ─────────────────────────────────────────────────────────────────
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(
            text=('<b>Circuito equivalente capacitivo — 3 conductores (ce₁, ce₂, ne)</b><br>'
                  '<sup>Y<sub>ij</sub> = jωC<sub>ij</sub>  |  '
                  '[I] = jω[C][V]  |  '
                  'Condición: I<sub>ne</sub> = 0 (conductor aislado)</sup>'),
            x=0.5, font=dict(size=13),
        ),
        xaxis=dict(range=[-1.8, 14.0], visible=False),
        yaxis=dict(range=[-0.5, 11.2], visible=False),
        height=640,
        showlegend=True,
        legend=dict(x=0.01, y=0.02, bgcolor='rgba(255,255,255,0.9)',
                    bordercolor='#ddd', borderwidth=1),
    )
    return fig



def plot_matrix_partition():
    """Partición de [C] mejorada: elementos simbólicos, dimensiones y ecuación."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5),
                              gridspec_kw={'width_ratios': [1.1, 0.9]})
    ax, ax2 = axes
    for a in axes:
        a.set_facecolor('white')
        a.axis('off')

    # ════ PANEL IZQUIERDO: Matriz particionada ═══════════════════════════════
    ax.set_xlim(-0.7, 4.2)
    ax.set_ylim(-1.0, 4.0)
    ax.set_aspect('equal')

    # ── Bloques coloreados ──────────────────────────────────────────────────────
    bloques = [
        ((0, 1), 2, 2, '#BBDEFB'),   # C_I
        ((2, 1), 1, 2, '#E1BEE7'),   # C_II
        ((0, 0), 2, 1, '#E1BEE7'),   # C_III
        ((2, 0), 1, 1, '#FFCDD2'),   # C_IV
    ]
    for (bx, by), bw, bh, bc in bloques:
        ax.add_patch(mpatches.FancyBboxPatch((bx, by), bw, bh,
                     boxstyle='square,pad=0', color=bc, zorder=1))

    # ── Líneas de separación ────────────────────────────────────────────────────
    ax.plot([2, 2], [0, 3], 'k--', lw=2.5, zorder=3)
    ax.plot([0, 3], [1, 1], 'k--', lw=2.5, zorder=3)

    # ── Bordes de la matriz ─────────────────────────────────────────────────────
    for x in [0, 3]:
        ax.plot([x, x], [0, 3], 'k-', lw=3.2, zorder=4)
    for y in [0, 3]:
        ax.plot([0, 3], [y, y], 'k-', lw=3.2, zorder=4)

    # ── Etiquetas columnas y filas ──────────────────────────────────────────────
    col_lbl = [r'$ce_1$', r'$ce_2$', r'$ne$']
    row_lbl = [r'$ne$',   r'$ce_2$', r'$ce_1$']
    for i, lbl in enumerate(col_lbl):
        c = COLOR_CE if i < 2 else COLOR_NE
        ax.text(0.5 + i, 3.32, lbl, ha='center', va='bottom',
                fontsize=14, fontweight='bold', color=c)
    for i, lbl in enumerate(row_lbl):
        c = COLOR_NE if i == 0 else COLOR_CE
        ax.text(-0.28, 0.5 + i, lbl, ha='right', va='center',
                fontsize=14, fontweight='bold', color=c)

    # ── Nombres y contenido simbólico de cada bloque ───────────────────────────
    # [C_I] — 2×2: mostrar elementos en dos filas de texto
    ax.text(1.0, 2.68, r'$[\mathbf{C_I}]$', ha='center', va='center',
            fontsize=14, fontweight='bold', color='#0D47A1', zorder=5)
    ax.text(1.0, 2.30, r'$C_{11}\ \ C_{12}$', ha='center', va='center',
            fontsize=9.5, color='#0D47A1', zorder=5)
    ax.text(1.0, 1.95, r'$C_{21}\ \ C_{22}$', ha='center', va='center',
            fontsize=9.5, color='#0D47A1', zorder=5)
    # Corchetes laterales manuales (líneas)
    for yy_c in [(1.88, 2.44)]:
        ax.plot([0.38, 0.38], yy_c, color='#0D47A1', lw=1.8, zorder=5)
        ax.plot([0.38, 0.46], [yy_c[0], yy_c[0]], color='#0D47A1', lw=1.8, zorder=5)
        ax.plot([0.38, 0.46], [yy_c[1], yy_c[1]], color='#0D47A1', lw=1.8, zorder=5)
        ax.plot([1.62, 1.62], yy_c, color='#0D47A1', lw=1.8, zorder=5)
        ax.plot([1.54, 1.62], [yy_c[0], yy_c[0]], color='#0D47A1', lw=1.8, zorder=5)
        ax.plot([1.54, 1.62], [yy_c[1], yy_c[1]], color='#0D47A1', lw=1.8, zorder=5)
    ax.text(1.0, 1.10, 'ce × ce  (nce×nce)', ha='center', va='bottom',
            fontsize=8.5, color='#0D47A1', style='italic', zorder=5)

    # [C_II] — 2×1: columna
    ax.text(2.5, 2.58, r'$[\mathbf{C_{II}}]$', ha='center', va='center',
            fontsize=12, fontweight='bold', color=COLOR_MUTUA, zorder=5)
    ax.text(2.5, 2.22, r'$C_{1,ne}$', ha='center', va='center',
            fontsize=9.5, color=COLOR_MUTUA, zorder=5)
    ax.text(2.5, 1.88, r'$C_{2,ne}$', ha='center', va='center',
            fontsize=9.5, color=COLOR_MUTUA, zorder=5)
    for yy_c in [(1.78, 2.35)]:
        ax.plot([2.15, 2.15], yy_c, color=COLOR_MUTUA, lw=1.8, zorder=5)
        ax.plot([2.15, 2.23], [yy_c[0], yy_c[0]], color=COLOR_MUTUA, lw=1.8, zorder=5)
        ax.plot([2.15, 2.23], [yy_c[1], yy_c[1]], color=COLOR_MUTUA, lw=1.8, zorder=5)
        ax.plot([2.85, 2.85], yy_c, color=COLOR_MUTUA, lw=1.8, zorder=5)
        ax.plot([2.77, 2.85], [yy_c[0], yy_c[0]], color=COLOR_MUTUA, lw=1.8, zorder=5)
        ax.plot([2.77, 2.85], [yy_c[1], yy_c[1]], color=COLOR_MUTUA, lw=1.8, zorder=5)
    ax.text(2.5, 1.10, 'ce×ne', ha='center', va='bottom',
            fontsize=8.5, color=COLOR_MUTUA, style='italic', zorder=5)

    # [C_III] — 1×2: fila
    ax.text(1.0, 0.65, r'$[\mathbf{C_{III}}]$', ha='center', va='center',
            fontsize=12, fontweight='bold', color=COLOR_MUTUA, zorder=5)
    ax.text(1.0, 0.28, r'$C_{ne,1}\ \ C_{ne,2}$', ha='center', va='center',
            fontsize=9.5, color=COLOR_MUTUA, zorder=5)
    for yy_c in [(0.18, 0.42)]:
        ax.plot([0.30, 0.30], yy_c, color=COLOR_MUTUA, lw=1.8, zorder=5)
        ax.plot([0.30, 0.38], [yy_c[0], yy_c[0]], color=COLOR_MUTUA, lw=1.8, zorder=5)
        ax.plot([0.30, 0.38], [yy_c[1], yy_c[1]], color=COLOR_MUTUA, lw=1.8, zorder=5)
        ax.plot([1.70, 1.70], yy_c, color=COLOR_MUTUA, lw=1.8, zorder=5)
        ax.plot([1.62, 1.70], [yy_c[0], yy_c[0]], color=COLOR_MUTUA, lw=1.8, zorder=5)
        ax.plot([1.62, 1.70], [yy_c[1], yy_c[1]], color=COLOR_MUTUA, lw=1.8, zorder=5)

    # [C_IV] — escalar
    ax.text(2.5, 0.65, r'$[\mathbf{C_{IV}}]$', ha='center', va='center',
            fontsize=12, fontweight='bold', color='#B71C1C', zorder=5)
    ax.text(2.5, 0.28, r'$C_{ne,ne}$', ha='center', va='center',
            fontsize=11, color='#B71C1C', zorder=5)

    # ── Llaves de fila / columna con dimensiones ────────────────────────────────
    ax.text(3.48, 2.0, 'Filas ce\n(nce filas)', ha='left', va='center',
            fontsize=9.5, color=COLOR_CE, style='italic')
    ax.text(3.48, 0.5, 'Fila ne\n(1 fila)', ha='left', va='center',
            fontsize=9.5, color=COLOR_NE, style='italic')
    ax.plot([3.42, 3.42], [0.05, 2.95], 'k-', lw=1.5)
    for yv in [2.0, 0.5]:
        ax.plot([3.42, 3.46], [yv, yv], 'k-', lw=1.5)

    # Dimensión columnas
    ax.text(1.0, 3.72, 'nce columnas', ha='center', va='bottom',
            fontsize=9, color=COLOR_CE, style='italic')
    ax.text(2.5, 3.72, '1 col.', ha='center', va='bottom',
            fontsize=9, color=COLOR_NE, style='italic')
    ax.plot([0.05, 2.95], [3.65, 3.65], 'k-', lw=1.2)
    ax.plot([2.05, 2.95], [3.65, 3.65], color=COLOR_NE, lw=1.5)

    ax.set_title('Partición de la matriz de capacitancias [C]',
                 fontsize=12, pad=18)

    p1 = mpatches.Patch(color='#BBDEFB', label='[C_I]  — acopl. ce×ce  (voltajes conocidos)')
    p2 = mpatches.Patch(color='#E1BEE7', label='[C_II],[C_III] — cruzado ce↔ne')
    p3 = mpatches.Patch(color='#FFCDD2', label='[C_IV] — capacitancia propia ne  (incógnita)')
    ax.legend(handles=[p1, p2, p3], loc='lower center',
              fontsize=9, ncol=1, bbox_to_anchor=(0.4, -0.17), framealpha=0.9)

    # ════ PANEL DERECHO: Ecuaciones resultantes ═══════════════════════════════
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)

    # Título
    ax2.text(5, 9.5, 'Ecuaciones de la partición', ha='center', va='top',
             fontsize=12, fontweight='bold', color='#333')

    # Ecuación fila ce
    ax2.add_patch(mpatches.FancyBboxPatch((0.3, 6.8), 9.4, 1.8,
                  boxstyle='round,pad=0.15', color='#E3F2FD', zorder=1))
    ax2.text(5, 8.35, 'Fila ce  (Ec. 4.5):', ha='center', va='center',
             fontsize=10, color='#0D47A1', fontweight='bold')
    ax2.text(5, 7.55,
             r'$[I_{ce}] = [C_I]\cdot[V_{ce}] + [C_{II}]\cdot[V_{ne}]$',
             ha='center', va='center', fontsize=11, color='#0D47A1')

    # Ecuación fila ne
    ax2.add_patch(mpatches.FancyBboxPatch((0.3, 4.7), 9.4, 1.8,
                  boxstyle='round,pad=0.15', color='#FCE4EC', zorder=1))
    ax2.text(5, 6.25, 'Fila ne  (Ec. 4.6):', ha='center', va='center',
             fontsize=10, color='#B71C1C', fontweight='bold')
    ax2.text(5, 5.45,
             r'$[\mathbf{0}] = [C_{III}]\cdot[V_{ce}] + [C_{IV}]\cdot[V_{ne}]$',
             ha='center', va='center', fontsize=11, color='#B71C1C')

    # Condición clave
    ax2.add_patch(mpatches.FancyBboxPatch((0.3, 2.8), 9.4, 1.6,
                  boxstyle='round,pad=0.15', color='#E8F5E9', zorder=1))
    ax2.text(5, 4.15, r'Condición física: $I_{ne} = 0$',
             ha='center', va='center', fontsize=10, color='#2E7D32', fontweight='bold')
    ax2.text(5, 3.45,
             r'Conductor aislado $\Rightarrow$ no hay camino a tierra',
             ha='center', va='center', fontsize=10, color='#2E7D32', style='italic')

    # Resultado final
    ax2.add_patch(mpatches.FancyBboxPatch((0.3, 0.5), 9.4, 2.0,
                  boxstyle='round,pad=0.15', color='#FFF9C4', ec='#F9A825',
                  lw=2, zorder=1))
    ax2.text(5, 2.25, 'Despejando  [V_ne]  →  Ec. (4.7):',
             ha='center', va='center', fontsize=10, color='#E65100', fontweight='bold')
    ax2.text(5, 1.35,
             r'$[V_{ne}] = -[C_{IV}]^{-1}\cdot[C_{III}]\cdot[V_{ce}]$',
             ha='center', va='center', fontsize=12, color='#E65100', fontweight='bold')

    ax2.set_title('Flujo algebraico de la partición', fontsize=11, pad=10)

    fig.tight_layout(pad=2.0)
    return fig


def plot_conductores_aislado():
    """Comparación mejorada: circuitos con corriente, campos y análisis técnico."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax in [ax1, ax2]:
        ax.set_facecolor('#f8f9fa')
        ax.set_xlim(-0.2, 6.2)
        ax.set_ylim(-0.2, 5.5)
        ax.axis('off')

    # ═══ CASO 1: Conductor a tierra — inducción magnética ══════════════════════
    ax1.set_title('Conductor ne  CONECTADO a tierra en ambos extremos\n'
                  '→ Inducción de campo magnético  (Sección 4.3)',
                  fontsize=10, color='navy', pad=8, fontweight='bold')

    # Conductor horizontal
    ax1.plot([0.8, 5.2], [3.6, 3.6], color='navy', lw=5.5, solid_capstyle='round')
    ax1.text(3.0, 3.85, 'Conductor ne  (conectado a tierra)',
             ha='center', fontsize=9, color='navy', fontweight='bold')

    # Conductor energizado (fuente del campo B) arriba
    ax1.plot([0.8, 5.2], [4.8, 4.8], color=COLOR_CE, lw=4, solid_capstyle='round')
    ax1.text(3.0, 5.05, r'Conductor $ce$  (energizado,  $I_{ce} \neq 0$)',
             ha='center', fontsize=8.5, color=COLOR_CE)

    # Líneas de campo magnético (círculos concéntricos alrededor del conductor ce)
    for r, a in [(0.28, 0.35), (0.5, 0.22), (0.72, 0.12)]:
        ax1.add_patch(plt.Circle((3.0, 4.8), r, color='#7B1FA2',
                                  fill=False, lw=1.5, alpha=a, ls='--'))
    ax1.annotate('', xy=(3.5, 4.3), xytext=(3.35, 4.3),
                arrowprops=dict(arrowstyle='->', color='#7B1FA2', lw=1.5))
    ax1.text(3.55, 4.28, r'$\vec{B}$', fontsize=13, color='#7B1FA2',
             fontweight='bold', style='italic')

    # Conexiones a tierra con resistencias
    for xr in [0.8, 5.2]:
        ax1.plot([xr, xr], [3.6, 2.6], color='gray', lw=2.2)
        # Resistencia
        rect = mpatches.Rectangle((xr - 0.22, 2.1), 0.44, 0.85,
                                   fill=True, fc='#fff', ec='#444', lw=2)
        ax1.add_patch(rect)
        ax1.text(xr, 2.52, 'R', ha='center', va='center', fontsize=10, color='#444')
        ax1.plot([xr, xr], [2.1, 1.6], color='gray', lw=2.2)
        # Tierra
        ax1.plot([xr - 0.4, xr + 0.4], [1.6, 1.6], color=COLOR_TIERRA, lw=3)
        for xxi in np.arange(xr - 0.38, xr + 0.4, 0.18):
            ax1.plot([xxi, xxi - 0.14], [1.6, 1.3], color=COLOR_TIERRA, lw=1.3)
        ax1.text(xr, 1.08, 'Tierra', ha='center', fontsize=8, color=COLOR_TIERRA,
                 style='italic')

    # Flecha de corriente inducida
    ax1.annotate('', xy=(4.2, 3.6), xytext=(2.5, 3.6),
                 arrowprops=dict(arrowstyle='->', color='red', lw=3.0,
                                 mutation_scale=18))
    ax1.text(3.35, 3.1, r'$I_{ne,inducida} \neq 0$', ha='center',
             fontsize=12, color='red', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', fc='#ffebee', ec='red', alpha=0.9))

    # Incógnita: I_ne
    ax1.text(3.0, 0.55,
             'Incógnita: $I_{ne}$\n(circula corriente peligrosa)',
             ha='center', fontsize=9.5, color='red', style='italic',
             bbox=dict(boxstyle='round,pad=0.4', fc='#ffcdd2', ec='red', alpha=0.85))

    # ═══ CASO 2: Conductor aislado — inducción eléctrica ══════════════════════
    ax2.set_title('Conductor ne  AISLADO en ambos extremos\n'
                  '→ Inducción de campo eléctrico  (Sección 4.2)  ← este caso',
                  fontsize=10, color='darkred', pad=8, fontweight='bold')

    # Conductor horizontal
    ax2.plot([1.0, 5.0], [3.6, 3.6], color=COLOR_NE, lw=5.5, solid_capstyle='round')
    ax2.text(3.0, 3.85, 'Conductor ne  (aislado)',
             ha='center', fontsize=9, color=COLOR_NE, fontweight='bold')

    # Conductor energizado arriba
    ax2.plot([1.0, 5.0], [4.8, 4.8], color=COLOR_CE, lw=4, solid_capstyle='round')
    ax2.text(3.0, 5.05, r'Conductor $ce$  (energizado,  $V_{ce} \neq 0$)',
             ha='center', fontsize=8.5, color=COLOR_CE)

    # Líneas de campo eléctrico (flechas verticales ce → ne)
    for xf in [1.6, 2.4, 3.2, 4.0, 4.6]:
        ax2.annotate('', xy=(xf, 3.78), xytext=(xf, 4.62),
                    arrowprops=dict(arrowstyle='->', color='darkorange',
                                    lw=1.6, mutation_scale=12))
    ax2.text(5.1, 4.22, r'$\vec{E}$', fontsize=14, color='darkorange',
             fontweight='bold', style='italic')

    # Interruptores abiertos en ambos extremos
    for xs, lado in [(1.0, -1), (5.0, 1)]:
        ax2.plot([xs, xs], [3.6, 2.7], color='gray', lw=2.2)
        # Interruptor abierto (línea diagonal)
        ax2.plot([xs, xs + lado * 0.42], [2.7, 2.2],
                 color='gray', lw=2.2)
        ax2.plot([xs, xs], [2.7, 2.7], 'o',
                 color='gray', markersize=7, markerfacecolor='white', mew=2)
        # Texto abierto
        ax2.text(xs + lado * 0.25, 2.45, 'Abierto\n(aislado)',
                 ha='center', fontsize=7.5, color='gray', style='italic')
        # Tierra simbólica (sin conexión)
        ax2.plot([xs, xs], [1.8, 1.4], color=COLOR_TIERRA, lw=2, ls='--', alpha=0.4)
        ax2.plot([xs - 0.4, xs + 0.4], [1.4, 1.4], color=COLOR_TIERRA, lw=2.5,
                 alpha=0.35)
        for xxi in np.arange(xs - 0.38, xs + 0.4, 0.18):
            ax2.plot([xxi, xxi - 0.14], [1.4, 1.1], color=COLOR_TIERRA,
                     lw=1.2, alpha=0.35)
        ax2.text(xs, 0.85, '(sin tierra)', ha='center', fontsize=8,
                 color=COLOR_TIERRA, style='italic', alpha=0.55)

    # Condición clave: I_ne = 0
    ax2.text(3.0, 2.05, r'$I_{ne} = 0$', ha='center', fontsize=15,
             color='green', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.35', fc='#e8f5e9', ec='green', alpha=0.9))

    # Voltaje inducido (incógnita)
    ax2.text(3.0, 3.08, r'$V_{ne} = \;?$  (inducido)', ha='center',
             fontsize=11, color=COLOR_NE, style='italic', fontweight='bold')

    # Incógnita
    ax2.text(3.0, 0.32,
             'Incógnita: $V_{ne}$\nVoltaje peligroso aunque no circule corriente',
             ha='center', fontsize=9.5, color=COLOR_NE, style='italic',
             bbox=dict(boxstyle='round,pad=0.4', fc='#ffcdd2', ec=COLOR_NE, alpha=0.85))

    fig.suptitle(
        r'Dos modos de inducción: campo magnético ($I_{ne}=?$) vs campo eléctrico ($V_{ne}=?$)',
        fontsize=11, fontweight='bold', y=1.01
    )
    fig.tight_layout(pad=1.5)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIONES DE CONTENIDO
# ═══════════════════════════════════════════════════════════════════════════════

def seccion_1():
    st.title("Sección 1 — Introducción")
    st.markdown("---")

    st.markdown("""
    ### ¿Qué es la inducción de campo eléctrico?

    Cuando un conductor está **desenergizado** (sin tensión propia) y ubicado en
    forma paralela a conductores **energizados**, el campo eléctrico generado por
    estos últimos actúa sobre el conductor abierto mediante **acoplamiento capacitivo**
    (inducción electrostática).

    Como resultado, aparece un **potencial eléctrico** en el conductor $ne$ aunque
    no circule corriente alguna por él. Este fenómeno se estudia en la
    **Sección 4.2** del libro de Ing. Miranda y es de gran importancia para la
    seguridad en trabajos de mantenimiento en líneas de transmisión.
    """)

    fig = plot_geometry()
    st.plotly_chart(fig, use_container_width=True)

    st.info(r"""
    **Condiciones clave del conductor $ne$ (desenergizado):**

    - $I_{ne} = 0$ — Está aislado en ambos extremos; **no existe camino para que circule corriente**.
    - $V_{ne} = \;?$ — Esta es la **incógnita** a determinar mediante la deducción.
    """)

    st.markdown("""
    ### ¿Por qué es importante calcular V_ne?

    Un operario que toque el conductor $ne$ proporciona el camino a tierra que
    convierte ese voltaje inducido en una **corriente eléctrica peligrosa** a través
    de su cuerpo. Por eso, antes de cualquier trabajo en una línea desenergizada,
    es fundamental estimar el valor de $V_{ne}$.
    """)


def seccion_2():
    st.title("Sección 2 — ¿Qué es jω?")
    st.markdown("---")
    st.markdown("""
    El factor $j\\omega$ aparece en toda la deducción. Es esencial entender qué
    representa físicamente antes de continuar.
    """)

    tabs = st.tabs(["¿Qué es ω?", "¿Qué es j?", "jω juntos", "En el capacitor"])

    # ── Tab 1 ──────────────────────────────────────────────────────────────────
    with tabs[0]:
        st.markdown("### La frecuencia angular ω")
        st.markdown("""
        La frecuencia angular $\\omega$ expresa la **velocidad de oscilación** de una
        magnitud sinusoidal, medida en radianes por segundo.
        """)
        st.latex(r"\omega = 2\pi f \quad [\text{rad/s}]")

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("""
            | Sistema | f (Hz) | ω (rad/s) |
            |---------|--------|-----------|
            | Bolivia / Europa | 50 | 314.16 |
            | Norteamérica | 60 | 376.99 |
            """)
        with col_b:
            st.markdown(r"""
            **Período:**
            $$T = \frac{1}{f} = \frac{2\pi}{\omega} \quad [\text{s}]$$
            """)

        freq = st.slider("Frecuencia f (Hz)", 1, 100, 50, 1, key='freq_slider')
        fig_s = plot_sinusoidal(freq)
        st.plotly_chart(fig_s, use_container_width=True)
        st.markdown(f"""
        > Al aumentar $f$, el período $T = {1/freq*1000:.2f}$ ms se acorta y la senoidal
        oscila más rápido. El ω cuantifica esa velocidad:
        $\\omega = 2\\pi \\times {freq} = {2*np.pi*freq:.1f}$ rad/s.
        """)

    # ── Tab 2 ──────────────────────────────────────────────────────────────────
    with tabs[1]:
        st.markdown("### El operador imaginario j")
        st.markdown(r"""
        El número $j$ no es solo una abstracción matemática: es un
        **operador de rotación de 90°** en el plano complejo.
        """)
        st.latex(r"j^2 = -1 \quad \Rightarrow \quad j = \sqrt{-1}")
        st.markdown(r"""
        Multiplicar un vector por $j$ lo **rota 90° en sentido antihorario**.
        Multiplicar dos veces rota 180°, lo que equivale a negar el vector
        (de ahí que $j^2 = -1$).
        """)
        fig_cp = plot_complex_plane()
        st.plotly_chart(fig_cp, use_container_width=True)

    # ── Tab 3 ──────────────────────────────────────────────────────────────────
    with tabs[2]:
        st.markdown("### El operador jω — la derivada en dominio fasorial")
        st.markdown(r"""
        En **régimen senoidal permanente** (AC de frecuencia fija), la derivada
        temporal de un fasor equivale a multiplicar por $j\omega$:
        """)
        st.latex(r"\frac{d}{dt} \;\longrightarrow\; \times\, j\omega \qquad"
                 r"\text{(válido solo en AC permanente, } \omega = \text{cte.)}")
        st.markdown("**Tabla comparativa:**")
        df = pd.DataFrame({
            "Dominio tiempo": ["d/dt", "I = C · dV/dt", "V = L · dI/dt", "V = R · I"],
            "Dominio fasorial": [r"× jω", r"I = jωC · V", r"V = jωL · I", "V = R · I"],
            "Elemento": ["Operador derivada", "Capacitor", "Inductor", "Resistor"],
        })
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.info(r"""
        Esta sustitución convierte las **ecuaciones diferenciales** del dominio
        tiempo en **ecuaciones algebraicas** del dominio fasorial. Es la base
        del análisis de circuitos en CA.
        """)

    # ── Tab 4 ──────────────────────────────────────────────────────────────────
    with tabs[3]:
        st.markdown("### Corriente en un capacitor — desfase de 90°")
        fig_vi = plot_vi_capacitor()
        st.plotly_chart(fig_vi, use_container_width=True)
        st.latex(r"I = j\omega C \cdot V")
        st.success(r"""
        **Interpretación física:**

        - La **j** indica que la corriente está **desfasada 90°** respecto al voltaje
          (la corriente se adelanta al voltaje en un capacitor).
        - El **ω** indica que a mayor frecuencia, más corriente circula por el capacitor
          (el capacitor "deja pasar" más corriente a frecuencias altas).
        - El **C** indica que mayor capacitancia → mayor corriente para el mismo voltaje.
        """)


def seccion_3():
    st.title("Sección 3 — De Q = CV a I = jωCV")
    st.markdown("---")
    st.markdown(r"""
    La relación $I = j\omega C \cdot V$ no es una definición arbitraria.
    Es el resultado directo de **derivar la relación carga-voltaje** del capacitor
    y aplicar la transformación fasorial. Veamos el proceso paso a paso:
    """)

    pasos = [
        (
            "Definición de capacitancia",
            r"Q = C \cdot V",
            r"La carga $Q$ almacenada es directamente proporcional al voltaje $V$ aplicado."
            r" $C$ es la constante de proporcionalidad (capacitancia, en Faradios)."
        ),
        (
            "Derivar respecto al tiempo",
            r"\frac{dQ}{dt} = C \cdot \frac{dV}{dt}",
            r"Derivamos ambos lados respecto al tiempo. $C$ es constante"
            r" (no depende del tiempo para un capacitor lineal e invariante)."
        ),
        (
            "Identificar la corriente eléctrica",
            r"I = \frac{dQ}{dt} \quad \Rightarrow \quad I = C \cdot \frac{dV}{dt}",
            r"Por definición, la corriente eléctrica es la tasa de cambio de carga:"
            r" $I = dQ/dt$. Sustituyendo obtenemos la **ecuación diferencial del capacitor**."
        ),
        (
            "Régimen senoidal — reemplazar d/dt por jω",
            r"I = C \cdot j\omega \cdot V = j\omega C \cdot V",
            r"En AC permanente, derivar equivale a multiplicar por $j\omega$."
            r" La ecuación diferencial se convierte en una ecuación **algebraica fasorial**."
        ),
        (
            "Definición de admitancia capacitiva",
            r"Y_{cap} = \frac{I}{V} = j\omega C",
            r"La **admitancia** $Y$ es la inversa de la impedancia: $Y = I/V$."
            r" Para un capacitor, $Y_{cap} = j\omega C$."
            r" No es una definición nueva — es la misma física con otro nombre."
        ),
    ]

    for i, (titulo, eq, texto) in enumerate(pasos, 1):
        with st.expander(f"Paso {i} — {titulo}", expanded=True):
            c1, c2 = st.columns([1, 2])
            with c1:
                st.latex(eq)
            with c2:
                st.markdown(texto)

    st.success(r"""
    **Resumen:**

    $$[Y] = j\omega \cdot [C]$$

    Es solo nomenclatura. La física es la misma. La admitancia $Y$ es simplemente
    la capacitancia $C$ multiplicada por el operador fasorial $j\omega$.
    """)


def seccion_4():
    st.title("Sección 4 — Capacitancia vs Admitancia")
    st.markdown("---")
    st.markdown(r"""
    En el análisis de líneas de transmisión aparecen dos matrices relacionadas:
    la matriz de capacitancias $[C]$ y la matriz de admitancias $[Y]$.
    Contienen la **misma información física** expresada de distinta forma.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### :blue[Matriz de Capacitancias [C]]")
        st.markdown(r"""
        - Obtenida como $[C] = [P]^{-1}$, donde $[P]$ es la **Matriz de Potenciales de Maxwell**
        - **Unidades:** F/m (Faradios por metro de longitud)
        - **Depende solo de la geometría:** alturas, radios, separaciones entre conductores
        - Es un número **real** (sin parte imaginaria)
        - **No cambia** con la frecuencia
        """)
        st.latex(r"""
        [C] = \begin{bmatrix}
          C_{11}    & C_{12}    & C_{1,ne} \\
          C_{21}    & C_{22}    & C_{2,ne} \\
          C_{ne,1}  & C_{ne,2}  & C_{ne,ne}
        \end{bmatrix}
        \left[\frac{\text{F}}{\text{m}}\right]
        """)

    with col2:
        st.markdown("#### :violet[Matriz de Admitancias [Y]]")
        st.markdown(r"""
        - Obtenida como $[Y] = j\omega[C]$
        - **Unidades:** S/m (Siemens por metro de longitud)
        - **Depende de geometría Y de la frecuencia** $\omega$
        - Es **compleja** (tiene parte imaginaria, por el factor $j$)
        - Permite escribir la ecuación nodal en la forma estándar $[I]=[Y][V]$
        """)
        st.latex(r"""
        [Y] = j\omega \cdot [C] = j\omega
        \begin{bmatrix}
          C_{11}    & C_{12}    & C_{1,ne} \\
          C_{21}    & C_{22}    & C_{2,ne} \\
          C_{ne,1}  & C_{ne,2}  & C_{ne,ne}
        \end{bmatrix}
        """)

    st.warning(r"""
    **¿Por qué el libro usa [Y] y no [C] directamente?**

    El libro introduce $[Y]$ para escribir la ecuación nodal en la forma estándar:

    $$[I] = [Y][V]$$

    igual que en cualquier circuito eléctrico. Esto permite aplicar la
    **partición matricial** de forma sistemática (ver Sección 7).

    Pero al final del álgebra, el factor $j\omega$ se **cancela** (ver Sección 9)
    y el resultado final contiene solo $[C]$. El voltaje inducido depende
    únicamente de la **geometría**, no de la frecuencia.
    """)


def seccion_5():
    st.title("Sección 5 — Circuito equivalente capacitivo")
    st.markdown("---")
    st.markdown(r"""
    La matriz $[C]$ tiene una interpretación física directa como un
    **circuito de capacitores** entre conductores y hacia tierra.
    Cada elemento $C_{ij}$ corresponde a un capacitor en el circuito equivalente.
    """)

    fig = plot_capacitive_circuit()
    st.plotly_chart(fig, use_container_width=True)

    st.info(r"""
    **Interpretación de los elementos de [C]:**

    - **Elementos diagonales** $C_{ii}$: capacitancia propia del nodo $i$ hacia tierra.
      Es la suma de todas las capacitancias conectadas al nodo $i$.
      **Siempre son positivos.**

    - **Elementos fuera de diagonal** $C_{ij}$ ($i \neq j$): capacitancia mutua entre
      los nodos $i$ y $j$. Representan el acoplamiento entre conductores.
      **Son negativos** en la convención matricial nodal estándar
      (la corriente sale del nodo hacia los vecinos).
    """)

    st.markdown("### Relación con la Matriz de Potenciales de Maxwell")
    st.latex(r"[C] = [P]^{-1}")
    st.markdown(r"""
    Donde $[P]$ se calcula a partir de las posiciones geométricas de los conductores
    y sus imágenes eléctricas, usando la **fórmula de Maxwell para conductores
    cilíndricos sobre un plano conductor** (método de las imágenes eléctricas):
    """)
    st.latex(r"""
    P_{ii} = \frac{1}{2\pi\varepsilon_0} \ln\frac{2h_i}{r_i}
    \qquad
    P_{ij} = \frac{1}{2\pi\varepsilon_0} \ln\frac{D_{ij}'}{D_{ij}}
    """)
    st.markdown(r"""
    donde $h_i$ es la altura del conductor $i$, $r_i$ su radio,
    $D_{ij}$ la distancia entre los conductores $i$ y $j$, y
    $D_{ij}'$ la distancia entre el conductor $i$ y la imagen del conductor $j$.
    """)


def seccion_6():
    st.title("Sección 6 — Ecuación nodal [I] = [Y][V]")
    st.markdown("---")
    st.markdown(r"""
    La **ecuación nodal** relaciona las corrientes inyectadas en cada conductor
    con los voltajes en todos los conductores, a través de la matriz de admitancias.

    Para $n = 3$ conductores (ce₁, ce₂, ne) — **Ecuación (4.2)** del libro:
    """)

    # Ecuación (4.2) — Ing. Miranda, Cap. 4
    st.latex(r"""
    \begin{bmatrix} I_1 \\ I_2 \\ I_{ne} \end{bmatrix}
    = j\omega
    \begin{bmatrix}
      C_{11}     & C_{12}     & C_{1,ne}   \\
      C_{21}     & C_{22}     & C_{2,ne}   \\
      C_{ne,1}   & C_{ne,2}   & C_{ne,ne}
    \end{bmatrix}
    \begin{bmatrix} V_1 \\ V_2 \\ V_{ne} \end{bmatrix}
    """)
    st.caption("Ecuación (4.2) — Ing. Miranda, Cap. 4")

    st.markdown("### Significado de cada fila")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(r"""
        **Fila 1 — $I_1$:**
        Corriente que ingresa al conductor ce₁.
        Depende de su propio voltaje $V_1$ y de los voltajes de todos los demás.
        """)
    with col2:
        st.markdown(r"""
        **Fila 2 — $I_2$:**
        Corriente que ingresa al conductor ce₂.
        Ídem para ce₂.
        """)
    with col3:
        st.markdown(r"""
        **Fila 3 — $I_{ne}$:**
        Corriente que ingresa al conductor ne.
        **En nuestro problema: $I_{ne} = 0$**
        (conductor aislado).
        """)

    with st.expander(r"Interpretación detallada de cada elemento $C_{ij}$"):
        st.markdown("**Elemento diagonal $C_{ii}$ — efecto propio:**")
        st.latex(r"C_{ii} = \frac{\text{Corriente saliente del nodo } i}{"
                 r"V_i = 1 \text{ V}} \bigg|_{V_{j \neq i} = 0}")
        st.markdown("**Elemento fuera de diagonal $C_{ij}$ — efecto mutuo ($i \\neq j$):**")
        st.latex(r"C_{ij} = \frac{\text{Corriente saliente del nodo } i}{"
                 r"V_j = 1 \text{ V}} \bigg|_{V_{k \neq j} = 0}")
        st.markdown(r"""
        Los $C_{ij}$ con $i \neq j$ son **negativos** porque la corriente fluye
        del nodo de mayor potencial hacia el de menor potencial.
        Cuando $V_j = 1$ y $V_i = 0$, la corriente entra al nodo $i$ desde $j$,
        lo que equivale a que la corriente saliente de $i$ sea negativa.
        """)


def seccion_7():
    st.title("Sección 7 — Partición de la matriz")
    st.markdown("---")
    st.markdown(r"""
    Para resolver el sistema, separamos los conductores en dos grupos:

    - **ce** (conductores energizados): voltajes **conocidos** (tensión nominal de la red)
    - **ne** (conductor desenergizado): voltaje **incógnita**

    Reordenamos las filas y columnas para tener los conductores ce primero
    y ne al final, y particionamos en **cuatro bloques**:
    """)

    col_fig, col_eq = st.columns([1, 1])

    with col_fig:
        fig = plot_matrix_partition()
        st.pyplot(fig)
        plt.close(fig)

    with col_eq:
        st.markdown("#### Ecuaciones resultantes de la partición")
        st.markdown("**Fila ce** (conductores energizados) — Ecuación (4.5):")
        # Ecuación (4.5) — Ing. Miranda, Cap. 4
        st.latex(r"[I_{ce}] = [C_I]\cdot[V_{ce}] + [C_{II}]\cdot[V_{ne}]")
        st.caption("Ecuación (4.5) — Ing. Miranda, Cap. 4")

        st.markdown("**Fila ne** (conductor desenergizado) — Ecuación (4.6):")
        # Ecuación (4.6) — Ing. Miranda, Cap. 4
        st.latex(r"[0] = [C_{III}]\cdot[V_{ce}] + [C_{IV}]\cdot[V_{ne}]")
        st.caption("Ecuación (4.6) — Ing. Miranda, Cap. 4")

        st.markdown("---")
        st.markdown("#### Dimensiones de las submatrices")
        df = pd.DataFrame({
            "Submatriz": [r"[C_I]", r"[C_II]", r"[C_III] = [C_II]ᵀ", r"[C_IV]"],
            "Dimensión": [r"nce × nce", r"nce × nne", r"nne × nce", r"nne × nne"],
            "Contenido": [
                "Acopl. entre cond. energizados",
                "Acopl. cruzado ce → ne",
                "Acopl. cruzado ne → ce",
                "Capac. propia del cond. ne",
            ],
        })
        st.dataframe(df, use_container_width=True, hide_index=True)

    st.info(r"""
    Cuando hay un **solo conductor ne**, $[C_{IV}]$ es un **escalar** (número, no
    matriz) y $[C_{III}]$ es un **vector fila**. El despeje se simplifica
    a una sola división escalar (ver Sección 10).
    """)


def seccion_8():
    st.title("Sección 8 — Condición I_ne = 0")
    st.markdown("---")
    st.markdown(r"""
    La condición $I_{ne} = 0$ es la **hipótesis física central** de la Sección 4.2.
    Proviene directamente del estado del conductor ne: aislado en ambos extremos.
    Comparemos las dos situaciones posibles:
    """)

    fig = plot_conductores_aislado()
    st.pyplot(fig)
    plt.close(fig)

    col1, col2 = st.columns(2)

    with col1:
        st.error(r"""
        **Conductor conectado a tierra en ambos extremos**

        - Existe circuito cerrado → puede circular corriente
        - $I_{ne} \neq 0$
        - Se analiza con **inducción de campo magnético** (Sección 4.3)
        - La corriente $I_{ne}$ es la incógnita
        """)

    with col2:
        st.success(r"""
        **Conductor aislado en ambos extremos (este caso)**

        - No existe circuito cerrado → no puede circular corriente
        - Por la Ley de Corriente de Kirchhoff: $I_{ne} = 0$
        - Se analiza con **inducción de campo eléctrico** (Sección 4.2)
        - El **voltaje** $V_{ne}$ es la incógnita
        """)

    st.markdown("### Ecuación de la fila ne con la condición $I_{ne} = 0$ aplicada")
    # Ecuación (4.6) con I=0 — Ing. Miranda, Cap. 4
    st.latex(r"[\mathbf{0}] = [C_{III}]\cdot[V_{ce}] + [C_{IV}]\cdot[V_{ne}]")

    st.info(r"""
    **Esta es la ecuación clave del problema.**

    - $[V_{ce}]$ es **conocido**: son los voltajes nominales de la red eléctrica.
    - $I_{ne} = 0$ es **condición del sistema**: el conductor está aislado.
    - $[V_{ne}]$ es la **única incógnita**: el voltaje inducido que buscamos despejar.
    """)


def seccion_9():
    st.title("Sección 9 — Cancelación de jω")
    st.markdown("---")
    st.markdown(r"""
    La ecuación de la fila $ne$ fue escrita originalmente con el factor $j\omega$
    proveniente de la matriz de admitancias $[Y] = j\omega[C]$.
    Veamos cómo este factor **se cancela algebraicamente**:
    """)

    col_pasos, col_nota = st.columns([3, 1])

    with col_pasos:
        st.markdown("#### Paso 1 — Ecuación de la fila ne con jω explícito")
        st.latex(r"[\mathbf{0}] = j\omega\,[C_{III}]\,[V_{ce}]"
                 r" + j\omega\,[C_{IV}]\,[V_{ne}]")

        st.markdown("#### Paso 2 — Factorizar el factor común jω")
        st.latex(r"[\mathbf{0}] = j\omega"
                 r"\left([C_{III}]\,[V_{ce}] + [C_{IV}]\,[V_{ne}]\right)")

        st.markdown(r"#### Paso 3 — Dividir ambos lados entre $j\omega$ (válido pues $\omega \neq 0$ en AC)")
        st.latex(r"\frac{[\mathbf{0}]}{j\omega} = [C_{III}]\,[V_{ce}]"
                 r" + [C_{IV}]\,[V_{ne}]")
        st.latex(r"[\mathbf{0}] = [C_{III}]\,[V_{ce}] + [C_{IV}]\,[V_{ne}]")

    with col_nota:
        st.markdown("#### En AC de 50 Hz:")
        st.latex(r"\omega = 2\pi \cdot 50")
        st.latex(r"\omega \approx 314 \;\frac{\text{rad}}{\text{s}}")
        st.markdown(r"""
        Como $\omega \neq 0$,
        el factor $j\omega \neq 0$
        y la división es **matemáticamente válida**.
        """)

    st.success(r"""
    **¡El factor $j\omega$ desaparece completamente!**

    El voltaje inducido $V_{ne}$:

    - **NO depende de la frecuencia** de la red eléctrica
    - **NO depende de la longitud** de la línea
      (los coeficientes de $[C]$ son por unidad de longitud y se cancelan en el cociente)
    - Depende **únicamente de la geometría** de la estructura
      (alturas, radios, separaciones entre conductores)
    """)

    with st.expander(r"¿Por qué este análisis no funciona en corriente continua (DC)?"):
        st.markdown(r"""
        En corriente continua, $f = 0$ Hz, por lo tanto $\omega = 0$:
        """)
        st.latex(r"\text{DC:} \quad \omega = 0 \quad \Rightarrow \quad j\omega = 0")
        st.markdown(r"""
        Dividir entre $j\omega = 0$ **no está matemáticamente permitido**.

        El análisis fasorial ($I = j\omega C V$) es válido **solo en régimen senoidal
        permanente** con $\omega \neq 0$. En DC, el capacitor ideal es circuito abierto
        en estado estacionario: no conduce corriente y no hay acoplamiento capacitivo
        dinámico. El fenómeno de inducción por campo eléctrico es esencialmente **AC**.
        """)


def seccion_10():
    st.title("Sección 10 — Ecuación final [V_ne]")
    st.markdown("---")
    st.markdown(r"""
    Tenemos la ecuación (tras partición y cancelación de $j\omega$):
    $$[\mathbf{0}] = [C_{III}]\cdot[V_{ce}] + [C_{IV}]\cdot[V_{ne}]$$
    Despejamos $[V_{ne}]$ paso a paso:
    """)

    pasos_alg = [
        ("Paso 1 — Pasar el término con $[V_{ce}]$ al lado izquierdo",
         r"[C_{IV}]\cdot[V_{ne}] = -[C_{III}]\cdot[V_{ce}]"),
        (r"Paso 2 — Premultiplicar ambos lados por $[C_{IV}]^{-1}$",
         r"[C_{IV}]^{-1}\cdot[C_{IV}]\cdot[V_{ne}]"
         r" = -[C_{IV}]^{-1}\cdot[C_{III}]\cdot[V_{ce}]"),
        (r"Paso 3 — Simplificar: $[C_{IV}]^{-1}[C_{IV}] = [I]$ (identidad)",
         r"[I]\cdot[V_{ne}] = -[C_{IV}]^{-1}\cdot[C_{III}]\cdot[V_{ce}]"),
        ("Resultado",
         r"[V_{ne}] = -[C_{IV}]^{-1}\cdot[C_{III}]\cdot[V_{ce}]"),
    ]

    for titulo, eq in pasos_alg:
        st.markdown(f"**{titulo}:**")
        st.latex(eq)

    st.markdown("---")
    st.markdown("### Ecuación (4.7) — Resultado final")

    st.markdown("""
    <div style="border:3px solid #1565C0; border-radius:12px;
                padding:22px 28px; background:#E3F2FD; text-align:center;
                margin:10px 0;">
    """, unsafe_allow_html=True)
    # Ecuación (4.7) — Ing. Miranda, Cap. 4
    st.latex(r"\boxed{\;[V_{ne}] = -[C_{IV}]^{-1} \cdot [C_{III}] \cdot [V_{ce}]\;}")
    st.markdown("</div>", unsafe_allow_html=True)
    st.caption("Ecuación (4.7) — Ing. Miranda, Cap. 4")

    st.info(r"""
    **Significado de cada término:**

    | Término | Descripción |
    |---------|-------------|
    | $[V_{ne}]$ | Vector de voltajes inducidos en los conductores ne — **incógnita** |
    | $[C_{IV}]^{-1}$ | Inversa de la submatriz ne×ne de capacitancias propias del conductor ne |
    | $[C_{III}]$ | Submatriz ne×ce de capacitancias mutuas ne ↔ ce |
    | $[V_{ce}]$ | Vector de voltajes conocidos de los conductores energizados — **dato** |
    """)

    st.markdown("### Caso especial: un solo conductor ne (resultado escalar)")
    st.markdown(r"""
    Cuando hay un solo conductor ne, $[C_{IV}]$ es un escalar y $[C_{III}]$
    es un vector fila. La ecuación matricial se reduce a:
    """)
    st.latex(r"""
    V_{ne} = -\frac{C_{ne,ce_1} \cdot V_{ce_1}
                  + C_{ne,ce_2} \cdot V_{ce_2}
                  + \cdots
                  + C_{ne,ce_n} \cdot V_{ce_n}}{C_{ne,ne}}
    """)
    st.markdown(r"""
    Esta es la forma de un **divisor de voltaje capacitivo**: el voltaje inducido en
    $ne$ es la suma ponderada de las contribuciones de cada conductor energizado,
    dividida por la capacitancia propia de $ne$ a tierra.
    """)

    # ── Sección interactiva mejorada ───────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Exploración interactiva — Sensibilidad geométrica de V_ne")
    st.markdown(r"""
    Mueve los sliders para explorar cómo la **geometría** determina $V_{ne}$.
    Los modelos son heurísticos normalizados — muestran la **tendencia cualitativa**
    basada en la física de las capacitancias de Maxwell.
    **(No corresponden a datos de ningún ejercicio específico.)**
    """)

    col_sl1, col_sl2, col_sl3 = st.columns(3)
    with col_sl1:
        D_val = st.slider("Distancia ne–ce  (m)", 1.0, 30.0, 8.0, 0.5, key='slD')
    with col_sl2:
        h_val = st.slider("Altura conductor ne  (m)", 5.0, 50.0, 15.0, 1.0, key='slH')
    with col_sl3:
        r_val = st.slider("Radio conductor ne  (mm)", 5, 50, 15, 1, key='slR')

    # ── Modelos heurísticos ─────────────────────────────────────────────────────
    D  = np.linspace(1.0,  30.0, 300)
    h  = np.linspace(5.0,  50.0, 300)
    rr = np.linspace(5.0,  50.0, 300)

    # C_mutua ∝ ln(D'/D) → decrece al aumentar D
    C_mut_D  = np.log1p(20.0 / D)
    # C_propia ∝ ln(2h/r) → crece con h (mayor C_propia → menor V_ne)
    C_prop_h = np.log(2 * h / (r_val / 1000.0 + 0.01))
    # C_propia crece al reducir r
    C_prop_r = np.log(2 * h_val / (rr / 1000.0 + 0.01))

    V_D  = C_mut_D  / C_mut_D.max()
    V_H  = 1.0 / (C_prop_h / C_prop_h.min())
    V_R  = 1.0 / (C_prop_r / C_prop_r.min())

    idx_D = int(np.argmin(np.abs(D  - D_val)))
    idx_H = int(np.argmin(np.abs(h  - h_val)))
    idx_R = int(np.argmin(np.abs(rr - r_val)))

    # ── Gráfico 1: curvas 1D (3 paneles) ───────────────────────────────────────
    fig_curvas = make_subplots(
        rows=1, cols=3,
        subplot_titles=(
            'Efecto de la DISTANCIA  D  ne–ce',
            'Efecto de la ALTURA  h  del ne',
            'Efecto del RADIO  r  del conductor ne',
        ),
        shared_yaxes=True,
    )

    curvas = [
        (D,  V_D,  D_val,  idx_D, COLOR_CE,    'D (m)',    'circle'),
        (h,  V_H,  h_val,  idx_H, COLOR_NE,    'h (m)',    'diamond'),
        (rr, V_R,  r_val,  idx_R, COLOR_MUTUA, 'r (mm)',   'square'),
    ]
    for col_idx, (xx, yy, val, idx, col, xtitle, sym) in enumerate(curvas, 1):
        # Relleno bajo la curva
        fig_curvas.add_trace(go.Scatter(
            x=xx, y=yy, fill='tozeroy',
            fillcolor=f'rgba({int(col[1:3],16)},{int(col[3:5],16)},{int(col[5:7],16)},0.10)',
            line=dict(color='rgba(0,0,0,0)'), mode='lines',
            showlegend=False, hoverinfo='skip'
        ), row=1, col=col_idx)
        # Curva
        fig_curvas.add_trace(go.Scatter(
            x=xx, y=yy, mode='lines',
            line=dict(color=col, width=2.5),
            showlegend=False
        ), row=1, col=col_idx)
        # Punto actual
        fig_curvas.add_trace(go.Scatter(
            x=[val], y=[yy[idx]],
            mode='markers',
            marker=dict(color=col, size=14, symbol=sym,
                        line=dict(color='white', width=2)),
            showlegend=False,
            hovertemplate=f'{xtitle[:-4]} = {val:.1f}  →  V_ne = {yy[idx]:.3f} p.u.<extra></extra>'
        ), row=1, col=col_idx)
        # Líneas cruzadas en el punto
        fig_curvas.add_shape(
            type='line', x0=val, y0=0, x1=val, y1=yy[idx],
            line=dict(color=col, width=1.2, dash='dot'), row=1, col=col_idx
        )
        fig_curvas.add_shape(
            type='line', x0=xx[0], y0=yy[idx], x1=val, y1=yy[idx],
            line=dict(color=col, width=1.2, dash='dot'), row=1, col=col_idx
        )
        fig_curvas.update_xaxes(title_text=xtitle, row=1, col=col_idx)

    fig_curvas.update_yaxes(title_text='V_ne  (p.u.)', range=[-0.05, 1.15], row=1, col=1)
    fig_curvas.update_layout(
        **PLOTLY_LAYOUT,
        title='Sensibilidad de V_ne a los parámetros geométricos  (modelos normalizados)',
        height=340,
    )
    st.plotly_chart(fig_curvas, use_container_width=True)

    # ── Gráfico 2: mapa de calor 2D  V_ne(D, h) ────────────────────────────────
    st.markdown("#### Mapa de calor 2D — V_ne en función de D y h simultáneamente")
    st.markdown(r"""
    El mapa muestra cómo $V_{ne}$ varía al cambiar la **distancia** y la **altura** al
    mismo tiempo.  La estrella ★ marca el punto seleccionado por los sliders.
    """)

    D2 = np.linspace(1.0, 30.0, 120)
    H2 = np.linspace(5.0, 50.0, 120)
    DD, HH = np.meshgrid(D2, H2)

    # V_ne ∝ C_mut / C_prop  (modelo heurístico)
    C_mut2  = np.log1p(20.0 / DD)
    C_prop2 = np.log(2 * HH / (r_val / 1000.0 + 0.01))
    Vne_2D  = C_mut2 / C_prop2
    Vne_2D  = Vne_2D / Vne_2D.max()    # normalizar 0…1

    fig_heat = go.Figure()

    # Mapa de calor (contorno suavizado)
    fig_heat.add_trace(go.Contour(
        x=D2, y=H2, z=Vne_2D,
        colorscale='RdYlGn_r',
        contours=dict(
            coloring='heatmap',
            showlabels=True,
            labelfont=dict(size=10, color='white'),
            start=0.0, end=1.0, size=0.1,
        ),
        colorbar=dict(
            title=dict(text='V_ne (p.u.)', side='right'),
            tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
        ),
        hovertemplate='D=%{x:.1f} m  h=%{y:.1f} m<br>V_ne=%{z:.3f} p.u.<extra></extra>',
        name='V_ne(D,h)'
    ))

    # Líneas de corte: D=D_val (vertical) y h=h_val (horizontal)
    fig_heat.add_shape(type='line', x0=D_val, y0=5, x1=D_val, y1=50,
                       line=dict(color='white', width=1.8, dash='dash'))
    fig_heat.add_shape(type='line', x0=1, y0=h_val, x1=30, y1=h_val,
                       line=dict(color='white', width=1.8, dash='dash'))

    # Punto actual (estrella)
    idx2_D = int(np.argmin(np.abs(D2 - D_val)))
    idx2_H = int(np.argmin(np.abs(H2 - h_val)))
    v_actual = Vne_2D[idx2_H, idx2_D]

    fig_heat.add_trace(go.Scatter(
        x=[D_val], y=[h_val],
        mode='markers+text',
        marker=dict(symbol='star', size=18, color='white',
                    line=dict(color='black', width=1.5)),
        text=[f'  V={v_actual:.3f}'],
        textfont=dict(size=12, color='white'),
        textposition='middle right',
        showlegend=False,
        hovertemplate=f'D={D_val:.1f} m,  h={h_val:.1f} m<br>V_ne={v_actual:.3f} p.u.<extra></extra>'
    ))

    fig_heat.update_layout(
        **PLOTLY_LAYOUT,
        title=(f'Mapa V_ne(D, h)  |  r = {r_val} mm  |  '
               f'★ D={D_val:.1f} m, h={h_val:.1f} m → V_ne={v_actual:.3f} p.u.'),
        xaxis_title='Distancia  D  ne–ce  (m)',
        yaxis_title='Altura  h  conductor ne  (m)',
        height=430,
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    # ── Métricas y tabla de sensibilidad ───────────────────────────────────────
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("V_ne  (efecto distancia D)",
                  f"{V_D[idx_D]:.3f} p.u.",
                  delta=f"D={D_val:.1f} m → {'↑ más alto (cercano)' if D_val < 10 else '↓ más bajo (lejano)'}")
    with m2:
        st.metric("V_ne  (efecto altura h)",
                  f"{V_H[idx_H]:.3f} p.u.",
                  delta=f"h={h_val:.1f} m → {'↑ alto si h pequeño' if h_val < 20 else '↓ bajo si h grande'}")
    with m3:
        st.metric("V_ne  (efecto radio r)",
                  f"{V_R[idx_R]:.3f} p.u.",
                  delta=f"r={r_val} mm → {'↑ alto si r pequeño' if r_val < 20 else '↓ bajo si r grande'}")

    st.info(r"""
    **Conclusiones del análisis de sensibilidad:**

    | Parámetro ↑ | Efecto en $C_{mutua}$ | Efecto en $C_{propia}$ | Efecto en $V_{ne}$ |
    |-------------|----------------------|------------------------|---------------------|
    | Distancia D ↑ | Disminuye            | Sin efecto directo     | **Disminuye** ↓     |
    | Altura h ↑    | Sin efecto directo   | Aumenta                | **Disminuye** ↓     |
    | Radio r ↑     | Sin efecto directo   | Disminuye              | **Aumenta** ↑       |

    Recuerda: $V_{ne} = -C_{IV}^{-1} C_{III} V_{ce}$ — solo depende de la **geometría**.
    """)


def seccion_11():
    st.title("Sección 11 — ¿Campo E o campo B?")
    st.markdown("---")
    st.markdown(r"""
    En sistemas de líneas de transmisión paralelas coexisten **dos mecanismos de
    inducción** distintos. Conviene distinguirlos claramente:
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"### ✅ Inducción por campo **Eléctrico**")
        st.markdown("**Acoplamiento capacitivo — Sección 4.2 (este análisis)**")
        st.markdown(r"""
        **Condiciones necesarias:**
        - Voltaje en los conductores ce (ce energizados)
        - Conductor ne **aislado** en ambos extremos → $I_{ne} = 0$

        **Resultado:**
        - $V_{ne} \neq 0$ aunque no circule corriente
        - El voltaje existe por acoplamiento capacitivo entre conductores

        **Dependencias:**
        - Solo depende de la **geometría** (alturas, radios, separaciones)
        - **NO** depende de la longitud de la línea
        - **NO** depende de la frecuencia (el $j\omega$ se cancela)

        **Ecuación:**
        """)
        st.latex(r"[V_{ne}] = -[C_{IV}]^{-1}[C_{III}][V_{ce}]")
        st.caption("Ecuación (4.7) — Ing. Miranda, Cap. 4")

    with col2:
        st.markdown("### ❌ Inducción por campo **Magnético**")
        st.markdown("**Acoplamiento inductivo — Sección 4.3 (otro caso)**")
        st.markdown(r"""
        **Condiciones necesarias:**
        - Corriente en los conductores ce ($I_{ce} \neq 0$)
        - Conductor ne con **circuito cerrado** (extremos conectados a tierra)

        **Resultado:**
        - Si ne está **abierto**: existe $V_{inducido}$ pero $I_{ne} = 0$
          → no hay corriente de riesgo
        - Si ne está **cerrado**: $I_{ne,inducida} \neq 0$ (peligrosa)

        **Dependencias:**
        - **SÍ** depende de la longitud (impedancia mutua $Z_{ab} \propto L$)
        - **SÍ** depende de la frecuencia ($Z_{ab} \propto \omega$)

        **Ecuación (forma simplificada):**
        """)
        st.latex(r"I_{ne} \approx -\frac{Z_{ab}}{Z_{bb}} \cdot I_{ce}")
        st.caption("Inducción por campo magnético — requiere circuito cerrado")

    st.warning(r"""
    **Conclusión para este problema:**

    El conductor $ne$ tiene los interruptores **abiertos** en ambos extremos.
    Eso elimina el circuito cerrado necesario para que la inducción magnética
    produzca corriente peligrosa. El **riesgo real** proviene del voltaje inducido
    por **campo eléctrico**, el cual existe **aunque no circule corriente**.

    Un operario que toque el conductor $ne$ proporciona el camino a tierra que
    convierte ese voltaje en una corriente eléctrica a través de su cuerpo.
    Por eso es fundamental estimar $V_{ne}$ antes de cualquier trabajo en la
    línea desenergizada.
    """)

    st.markdown("---")
    st.markdown("### Tabla resumen comparativa")
    df = pd.DataFrame({
        "Característica": [
            "Mecanismo físico",
            "Condición en conductor ne",
            "Incógnita principal",
            "Depende de la longitud",
            "Depende de la frecuencia",
            "Depende de la geometría",
            "Ecuación clave",
            "Sección en Ing. Miranda",
        ],
        "Campo Eléctrico (Capacitivo)": [
            "Acoplamiento capacitivo [C]",
            "Aislado — I_ne = 0",
            "Voltaje V_ne",
            "No (se cancela en cociente)",
            "No (jω se cancela)",
            "Sí — alturas, radios, separaciones",
            "[V_ne] = −[C_IV]⁻¹[C_III][V_ce]",
            "Sección 4.2",
        ],
        "Campo Magnético (Inductivo)": [
            "Acoplamiento inductivo [Z]",
            "Conectado a tierra — I_ne ≠ 0",
            "Corriente I_ne",
            "Sí — Z_ab ∝ longitud",
            "Sí — Z_ab ∝ ω",
            "Sí — posiciones relativas",
            "I_ne = −(Z_ab / Z_bb) · I_ce",
            "Sección 4.3",
        ],
    })
    st.dataframe(df, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# FUNCIÓN PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Función principal — configura la página y despacha secciones."""
    st.set_page_config(
        page_title="ELT-264 | Voltaje Inducido por Campo Eléctrico",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── Sidebar ────────────────────────────────────────────────────────────────
    st.sidebar.title("ELT-264 · UMSA")
    st.sidebar.markdown("**Voltaje Inducido**")
    st.sidebar.markdown("Inducción de Campo Eléctrico")
    st.sidebar.markdown("*Acoplamiento capacitivo*")
    st.sidebar.markdown("---")

    seccion = st.sidebar.radio(
        "Navegación por secciones",
        SECCIONES,
        index=0,
    )

    idx_actual = SECCIONES.index(seccion) + 1
    st.sidebar.markdown("---")
    st.sidebar.progress(idx_actual / len(SECCIONES))
    st.sidebar.caption(f"Sección {idx_actual} de {len(SECCIONES)}")

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Referencia:** Ing. Miranda, Cap. 4")
    st.sidebar.markdown("**Materia:** Líneas de Transmisión de Energía")
    st.sidebar.markdown("*Contenido puramente deductivo.*")
    st.sidebar.markdown("*Sin datos de ejercicios específicos.*")

    # ── Despacho de secciones ──────────────────────────────────────────────────
    dispatch = {
        SECCIONES[0]:  seccion_1,
        SECCIONES[1]:  seccion_2,
        SECCIONES[2]:  seccion_3,
        SECCIONES[3]:  seccion_4,
        SECCIONES[4]:  seccion_5,
        SECCIONES[5]:  seccion_6,
        SECCIONES[6]:  seccion_7,
        SECCIONES[7]:  seccion_8,
        SECCIONES[8]:  seccion_9,
        SECCIONES[9]:  seccion_10,
        SECCIONES[10]: seccion_11,
    }
    dispatch[seccion]()


if __name__ == "__main__":
    main()
