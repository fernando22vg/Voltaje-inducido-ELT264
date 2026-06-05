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
    """Diagrama de conductores (ce1, ce2, ne) con campo eléctrico e imágenes."""
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.set_facecolor('#f0f4ff')

    x_ce1, h_ce1 = 1.8, 3.2
    x_ce2, h_ce2 = 3.8, 3.8
    x_ne,  h_ne  = 6.0, 2.9
    suelo_y = 0.0

    # ── Suelo ──────────────────────────────────────────────────────────────────
    ax.axhline(suelo_y, color=COLOR_TIERRA, lw=2.5)
    ax.fill_between([0.5, 7.5], suelo_y - 0.7, suelo_y,
                    color='#D7CCC8', alpha=0.55)
    for xi in np.arange(0.7, 7.5, 0.45):
        ax.plot([xi, xi - 0.18], [suelo_y, suelo_y - 0.22],
                color=COLOR_TIERRA, lw=1.1, alpha=0.7)
    ax.text(0.6, -0.55, 'Suelo (plano conductor de referencia)',
            fontsize=8, color=COLOR_TIERRA, style='italic')

    conductores = [
        (x_ce1, h_ce1, COLOR_CE,  r'$ce_1$', r'$V_{ce_1}$'),
        (x_ce2, h_ce2, COLOR_CE,  r'$ce_2$', r'$V_{ce_2}$'),
        (x_ne,  h_ne,  COLOR_NE,  r'$ne$',   r'$V_{ne}=\;?$'),
    ]

    # ── Columnas soporte y nodos ────────────────────────────────────────────────
    for x, h, col, lbl, lbl_v in conductores:
        ax.plot([x, x], [suelo_y, h], color='#BDBDBD', lw=1.5, ls='--', alpha=0.6)
        circ = plt.Circle((x, h), 0.2, color=col, zorder=6)
        ax.add_patch(circ)
        ax.text(x, h + 0.38, lbl, ha='center', va='bottom',
                fontsize=12, fontweight='bold', color=col, zorder=7)
        ax.text(x, h + 0.78, lbl_v, ha='center', va='bottom',
                fontsize=10, color=col,
                bbox=dict(boxstyle='round,pad=0.25', fc='white', ec=col,
                          alpha=0.9, lw=1.5),
                zorder=7)

    # ── Imágenes eléctricas (método de las imágenes) ───────────────────────────
    for x, h, col, *_ in conductores:
        circ_img = plt.Circle((x, -h), 0.2, color=COLOR_IMAGEN,
                               linestyle='--', fill=False, lw=1.8, zorder=5)
        ax.add_patch(circ_img)
        ax.plot([x, x], [suelo_y, -h], color=COLOR_IMAGEN,
                lw=1.0, ls=':', alpha=0.5)

    # ── Líneas de campo eléctrico (flechas curvas entre conductores) ───────────
    def campo_entre(x1, y1, x2, y2, color='darkorange', n=3, curv=0.35):
        for i in range(n):
            t = (i + 1) / (n + 1)
            ax.annotate(
                '', xy=(x1 + t * (x2 - x1), y1 + t * (y2 - y1)),
                xytext=(x1 + (t - 0.15) * (x2 - x1),
                         y1 + (t - 0.15) * (y2 - y1)),
                arrowprops=dict(arrowstyle='->', color=color,
                                lw=1.6, mutation_scale=14),
                zorder=4
            )

    # ce1 → ne
    for t in [0.25, 0.5, 0.75]:
        xm = x_ce1 + t * (x_ne - x_ce1)
        ym = h_ce1 + t * (h_ne - h_ce1) + 0.5 * np.sin(np.pi * t)
        dx = (x_ne - x_ce1) * 0.08
        ax.annotate('', xy=(xm + dx, ym),
                    xytext=(xm - dx * 0.5, ym),
                    arrowprops=dict(arrowstyle='->', color='darkorange',
                                    lw=1.6, mutation_scale=13), zorder=4)

    # ce2 → ne
    for t in [0.3, 0.6]:
        xm = x_ce2 + t * (x_ne - x_ce2)
        ym = h_ce2 + t * (h_ne - h_ce2) - 0.25 * np.sin(np.pi * t)
        dx = (x_ne - x_ce2) * 0.1
        ax.annotate('', xy=(xm + dx, ym),
                    xytext=(xm - dx * 0.5, ym),
                    arrowprops=dict(arrowstyle='->', color='orangered',
                                    lw=1.6, mutation_scale=13), zorder=4)

    ax.text(3.9, 3.2, r'$\vec{E}$', fontsize=16, color='darkorange',
            fontweight='bold', style='italic')

    ax.set_xlim(0.5, 7.5)
    ax.set_ylim(-1.1, 5.2)
    ax.set_xlabel('Separación transversal', fontsize=10)
    ax.set_ylabel('Altura sobre el suelo (u.a.)', fontsize=10)
    ax.set_title('Geometría de conductores — Inducción por campo eléctrico',
                 fontsize=11, pad=10)
    ax.set_xticks([])

    patch_ce  = mpatches.Patch(color=COLOR_CE,     label='Conductores energizados (ce)')
    patch_ne  = mpatches.Patch(color=COLOR_NE,     label='Conductor desenergizado (ne)')
    patch_img = mpatches.Patch(color=COLOR_IMAGEN, label='Imágenes eléctricas (bajo suelo)')
    ax.legend(handles=[patch_ce, patch_ne, patch_img],
              loc='upper left', fontsize=9, framealpha=0.9)

    fig.tight_layout()
    return fig


def plot_complex_plane():
    """Plano complejo mostrando el operador j como rotación de 90°."""
    fig = go.Figure()

    # Ejes
    for x0, y0, x1, y1 in [(-1.5, 0, 1.5, 0), (0, -1.5, 0, 1.5)]:
        fig.add_shape(type='line', x0=x0, y0=y0, x1=x1, y1=y1,
                      line=dict(color='black', width=1.5))

    # Vector 1 (real)
    fig.add_annotation(x=1.0, y=0, ax=0, ay=0, axref='x', ayref='y',
                       arrowhead=3, arrowsize=1.5, arrowwidth=3,
                       arrowcolor=COLOR_CE)
    fig.add_annotation(x=1.08, y=-0.18, text='<b>1</b> (real)',
                       showarrow=False, font=dict(color=COLOR_CE, size=15))

    # Vector j (imaginario)
    fig.add_annotation(x=0, y=1.0, ax=0, ay=0, axref='x', ayref='y',
                       arrowhead=3, arrowsize=1.5, arrowwidth=3,
                       arrowcolor=COLOR_NE)
    fig.add_annotation(x=0.14, y=1.1, text='<b>j</b> (imaginario)',
                       showarrow=False, font=dict(color=COLOR_NE, size=15))

    # Vector -1 (j²)
    fig.add_annotation(x=-1.0, y=0, ax=0, ay=0, axref='x', ayref='y',
                       arrowhead=3, arrowsize=1.2, arrowwidth=2,
                       arrowcolor='gray')
    fig.add_annotation(x=-1.1, y=-0.18, text='<b>j²=-1</b>',
                       showarrow=False, font=dict(color='gray', size=12))

    # Arco de rotación 90°
    theta = np.linspace(0, np.pi / 2, 60)
    fig.add_trace(go.Scatter(
        x=0.52 * np.cos(theta), y=0.52 * np.sin(theta),
        mode='lines', line=dict(color='green', width=2.5, dash='dot'),
        name='Rotación 90° (×j)'
    ))
    fig.add_annotation(x=0.42, y=0.42, text='×j → 90°',
                       showarrow=False, font=dict(color='green', size=13))

    # Etiquetas de ejes
    fig.add_annotation(x=1.42, y=0,   text='Re', showarrow=False,
                       font=dict(size=14, color='black'))
    fig.add_annotation(x=0.06, y=1.42, text='Im', showarrow=False,
                       font=dict(size=14, color='black'))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title='Plano complejo — El operador j rota 90° en sentido antihorario',
        xaxis=dict(range=[-1.6, 1.6], showgrid=False, zeroline=False,
                   showticklabels=False),
        yaxis=dict(range=[-1.6, 1.6], showgrid=False, zeroline=False,
                   showticklabels=False, scaleanchor='x'),
        showlegend=True,
        height=400,
    )
    return fig


def plot_sinusoidal(freq=50):
    """Senoidal interactiva V(t) = Vm·sen(ωt) con período marcado."""
    omega = 2 * np.pi * freq
    T     = 1.0 / freq
    t     = np.linspace(0, 2.5 * T, 600)
    V     = np.sin(omega * t)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t * 1000, y=V,
        mode='lines',
        line=dict(color=COLOR_CE, width=2.5),
        name='V(t) = Vm·sen(ωt)'
    ))
    # Marcar período T
    fig.add_shape(type='line',
                  x0=0, y0=1.08, x1=T * 1000, y1=1.08,
                  line=dict(color='gray', width=1.5, dash='dash'))
    fig.add_annotation(x=T * 500, y=1.18,
                       text=f'T = {T*1000:.2f} ms',
                       showarrow=False, font=dict(size=12, color='gray'))
    # Línea de referencia cero
    fig.add_hline(y=0, line_dash='dot', line_color='lightgray')

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=(f'V(t) = Vm · sen(ωt)    '
               f'f = {freq} Hz    '
               f'ω = {omega:.1f} rad/s'),
        xaxis_title='Tiempo (ms)',
        yaxis_title='V(t) / Vm',
        height=350,
        yaxis=dict(range=[-1.35, 1.35]),
    )
    return fig


def plot_vi_capacitor():
    """Desfase de 90° entre V e I en un capacitor ideal."""
    t = np.linspace(0, 2 * np.pi, 600)
    V = np.cos(t)
    I = -np.sin(t)   # I adelanta 90° a V

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=V, mode='lines',
                              line=dict(color=COLOR_CE, width=2.5),
                              name='V(t) = Vm · cos(ωt)'))
    fig.add_trace(go.Scatter(x=t, y=I, mode='lines',
                              line=dict(color=COLOR_NE, width=2.5, dash='dot'),
                              name='I(t) = −ωCVm · sen(ωt)'))

    # Líneas de referencia para el desfase
    fig.add_shape(type='line', x0=0,        y0=0, x1=0,        y1=1.0,
                  line=dict(color=COLOR_CE, width=1.2, dash='dot'))
    fig.add_shape(type='line', x0=np.pi/2,  y0=0, x1=np.pi/2,  y1=-1.0,
                  line=dict(color=COLOR_NE, width=1.2, dash='dot'))
    fig.add_annotation(x=np.pi / 4, y=0.65,
                       text='← 90° →', showarrow=False,
                       font=dict(size=13, color='green'))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title='V e I en un capacitor — la corriente se adelanta 90° al voltaje',
        xaxis_title='ωt (rad)',
        yaxis_title='Amplitud normalizada',
        height=360,
        xaxis=dict(tickvals=[0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi],
                   ticktext=['0', 'π/2', 'π', '3π/2', '2π']),
        yaxis=dict(range=[-1.35, 1.35]),
    )
    return fig


def plot_capacitive_circuit():
    """Circuito equivalente capacitivo para 3 conductores (ce1, ce2, ne)."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.set_facecolor('#f8f9fa')
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(0, 7)
    ax.axis('off')

    nodes = {'ce1': (2.0, 5.8), 'ce2': (5.0, 5.8), 'ne': (8.0, 5.8)}
    colors_n = {'ce1': COLOR_CE, 'ce2': COLOR_CE, 'ne': COLOR_NE}
    labels_n = {'ce1': r'$ce_1$', 'ce2': r'$ce_2$', 'ne': r'$ne$'}

    tierra_y = 0.9
    # Tierra
    ax.plot([-0.2, 10.2], [tierra_y, tierra_y], color=COLOR_TIERRA, lw=3)
    for xi in np.arange(0.2, 10.0, 0.55):
        ax.plot([xi, xi - 0.2], [tierra_y, tierra_y - 0.28],
                color=COLOR_TIERRA, lw=1.2, alpha=0.8)
    ax.text(0.0, tierra_y - 0.55, 'Tierra (referencia potencial cero)',
            fontsize=8, color=COLOR_TIERRA, style='italic')

    def cap_vertical(ax, x, y_top, y_bot, label, color):
        """Símbolo de capacitor vertical (dos placas paralelas horizontales)."""
        mid  = (y_top + y_bot) / 2
        gap  = 0.22
        half = 0.32
        ax.plot([x, x], [y_top, mid + gap], color=color, lw=2.2)
        ax.plot([x - half, x + half], [mid + gap, mid + gap], color=color, lw=3.5)
        ax.plot([x - half, x + half], [mid - gap, mid - gap], color=color, lw=3.5)
        ax.plot([x, x], [mid - gap, y_bot], color=color, lw=2.2)
        ax.text(x + 0.42, mid, label, ha='left', va='center',
                fontsize=10, color=color)

    def cap_horizontal(ax, x_l, x_r, y, label, color, y_offset=0.45):
        """Símbolo de capacitor horizontal (dos placas paralelas verticales)."""
        mid  = (x_l + x_r) / 2
        gap  = 0.18
        half = 0.28
        ax.plot([x_l, mid - gap], [y, y], color=color, lw=2.2)
        ax.plot([mid - gap, mid - gap], [y - half, y + half], color=color, lw=3.5)
        ax.plot([mid + gap, mid + gap], [y - half, y + half], color=color, lw=3.5)
        ax.plot([mid + gap, x_r], [y, y], color=color, lw=2.2)
        ax.text(mid, y + y_offset, label, ha='center', va='bottom',
                fontsize=10, color=color)

    # Nodos
    for key, (x, y) in nodes.items():
        circ = plt.Circle((x, y), 0.25, color=colors_n[key], zorder=5)
        ax.add_patch(circ)
        ax.text(x, y + 0.45, labels_n[key], ha='center', va='bottom',
                fontsize=13, fontweight='bold', color=colors_n[key])

    # Capacitores diagonales (a tierra)
    cap_vertical(ax, 2.0, 5.55, tierra_y + 0.1, r'$C_{11}$',     COLOR_CE)
    cap_vertical(ax, 5.0, 5.55, tierra_y + 0.1, r'$C_{22}$',     COLOR_CE)
    cap_vertical(ax, 8.0, 5.55, tierra_y + 0.1, r'$C_{ne,ne}$',  COLOR_NE)

    # Capacitores mutuos (entre conductores)
    cap_horizontal(ax, 2.25, 4.75, 5.8,  r'$C_{12}$',    COLOR_MUTUA)
    cap_horizontal(ax, 5.25, 7.75, 5.8,  r'$C_{2,ne}$',  COLOR_MUTUA)

    # ce1 — ne: arco por encima
    ax.plot([2.0, 2.0], [5.8, 6.65], color=COLOR_MUTUA, lw=1.8, ls='--')
    ax.plot([8.0, 8.0], [5.8, 6.65], color=COLOR_MUTUA, lw=1.8, ls='--')
    cap_horizontal(ax, 2.0, 8.0, 6.65, r'$C_{1,ne}$', COLOR_MUTUA, y_offset=0.4)

    ax.set_title('Circuito equivalente capacitivo — 3 conductores (ce₁, ce₂, ne)',
                 fontsize=12, pad=12)

    p1 = mpatches.Patch(color=COLOR_CE,    label='Capacitancia propia  Cᵢᵢ (diagonal → a tierra)')
    p2 = mpatches.Patch(color=COLOR_MUTUA, label='Capacitancia mutua   Cᵢⱼ (fuera de diagonal)')
    ax.legend(handles=[p1, p2], loc='lower center',
              fontsize=9, ncol=2, framealpha=0.9)

    fig.tight_layout()
    return fig


def plot_matrix_partition():
    """Visualización de la partición de [C] en cuatro bloques coloreados."""
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.set_facecolor('white')
    ax.set_xlim(-0.5, 3.6)
    ax.set_ylim(-0.6, 3.6)
    ax.axis('off')
    ax.set_aspect('equal')

    # Bloques
    ax.add_patch(mpatches.FancyBboxPatch((0, 1), 2, 2,
                 boxstyle='square,pad=0', color='#BBDEFB', zorder=1))   # C_I
    ax.add_patch(mpatches.FancyBboxPatch((2, 1), 1, 2,
                 boxstyle='square,pad=0', color='#E1BEE7', zorder=1))   # C_II
    ax.add_patch(mpatches.FancyBboxPatch((0, 0), 2, 1,
                 boxstyle='square,pad=0', color='#E1BEE7', zorder=1))   # C_III
    ax.add_patch(mpatches.FancyBboxPatch((2, 0), 1, 1,
                 boxstyle='square,pad=0', color='#FFCDD2', zorder=1))   # C_IV

    # Líneas de separación (partición)
    ax.plot([2, 2], [0, 3], 'k--', lw=2.2, zorder=3)
    ax.plot([0, 3], [1, 1], 'k--', lw=2.2, zorder=3)

    # Borde de la matriz
    for x in [0, 3]:
        ax.plot([x, x], [0, 3], 'k-', lw=3, zorder=4)
    for y in [0, 3]:
        ax.plot([0, 3], [y, y], 'k-', lw=3, zorder=4)

    # Etiquetas de columnas y filas
    col_lbl = [r'$ce_1$', r'$ce_2$', r'$ne$']
    row_lbl = [r'$ne$',   r'$ce_2$', r'$ce_1$']
    for i, lbl in enumerate(col_lbl):
        c = COLOR_CE if i < 2 else COLOR_NE
        ax.text(0.5 + i, 3.28, lbl, ha='center', va='bottom',
                fontsize=14, fontweight='bold', color=c)
    for i, lbl in enumerate(row_lbl):
        c = COLOR_NE if i == 0 else COLOR_CE
        ax.text(-0.22, 0.5 + i, lbl, ha='right', va='center',
                fontsize=14, fontweight='bold', color=c)

    # Nombres de submatrices
    ax.text(1.0, 2.0, r'$[\mathbf{C_I}]$' + '\nce × ce',
            ha='center', va='center', fontsize=13, fontweight='bold',
            color='#0D47A1', zorder=5)
    ax.text(2.5, 2.0, r'$[\mathbf{C_{II}}]$' + '\nce × ne',
            ha='center', va='center', fontsize=11, fontweight='bold',
            color=COLOR_MUTUA, zorder=5)
    ax.text(1.0, 0.5, r'$[\mathbf{C_{III}}]$' + '\nne × ce',
            ha='center', va='center', fontsize=11, fontweight='bold',
            color=COLOR_MUTUA, zorder=5)
    ax.text(2.5, 0.5, r'$[\mathbf{C_{IV}}]$' + '\nne × ne',
            ha='center', va='center', fontsize=12, fontweight='bold',
            color='#B71C1C', zorder=5)

    # Llave de filas
    ax.text(3.35, 2.0, 'Filas ce', ha='left', va='center',
            fontsize=10, color=COLOR_CE, style='italic')
    ax.text(3.35, 0.5, 'Fila ne',  ha='left', va='center',
            fontsize=10, color=COLOR_NE, style='italic')
    ax.plot([3.28, 3.28], [0.05, 2.95], 'k-', lw=1.5)
    ax.plot([3.28, 3.32], [2.0,  2.0],  'k-', lw=1.5)
    ax.plot([3.28, 3.32], [0.5,  0.5],  'k-', lw=1.5)

    ax.set_title('Partición de la matriz de capacitancias [C]',
                 fontsize=12, pad=18)

    p1 = mpatches.Patch(color='#BBDEFB', label='[C_I]  — acopl. ce×ce')
    p2 = mpatches.Patch(color='#E1BEE7', label='[C_II],[C_III] — cruzado ce↔ne')
    p3 = mpatches.Patch(color='#FFCDD2', label='[C_IV] — capacitancia propia ne')
    ax.legend(handles=[p1, p2, p3], loc='lower center',
              fontsize=9, ncol=1, bbox_to_anchor=(0.45, -0.16),
              framealpha=0.9)

    fig.tight_layout()
    return fig


def plot_conductores_aislado():
    """Comparación: conductor conectado a tierra vs. aislado."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax in [ax1, ax2]:
        ax.set_facecolor('#f8f9fa')
        ax.set_xlim(0, 6)
        ax.set_ylim(0, 4.5)
        ax.axis('off')

    # ── Caso 1: conductor conectado a tierra (inducción magnética) ─────────────
    ax1.set_title('Conductor conectado a tierra\n→ Inducción de campo magnético',
                  fontsize=10, color='navy', pad=6)
    ax1.plot([0.6, 5.4], [2.8, 2.8], color='navy', lw=4)
    ax1.text(3.0, 3.1, 'Conductor ne', ha='center', fontsize=9, color='navy')
    for xr in [0.6, 5.4]:
        ax1.plot([xr, xr], [2.8, 1.5], color='gray', lw=2)
        rect = mpatches.Rectangle((xr - 0.18, 1.6), 0.36, 0.75,
                                   fill=False, ec='#555', lw=2)
        ax1.add_patch(rect)
        # símbolo R
        ax1.text(xr, 1.97, 'R', ha='center', va='center',
                 fontsize=8, color='#555')
        ax1.plot([xr, xr], [1.6, 1.3], color='gray', lw=2)
        ax1.plot([xr - 0.3, xr + 0.3], [1.3, 1.3], color=COLOR_TIERRA, lw=2.5)
        for xx in np.arange(xr - 0.28, xr + 0.3, 0.16):
            ax1.plot([xx, xx - 0.14], [1.3, 1.05], color=COLOR_TIERRA, lw=1.1)
    ax1.annotate('', xy=(4.5, 2.8), xytext=(3.0, 2.8),
                 arrowprops=dict(arrowstyle='->', color='red', lw=2.5))
    ax1.text(3.75, 3.0, r'$I_{ne} \neq 0$', ha='center',
             fontsize=11, color='red', fontweight='bold')

    # ── Caso 2: conductor aislado (inducción eléctrica) ───────────────────────
    ax2.set_title('Conductor aislado en ambos extremos\n→ Inducción de campo eléctrico',
                  fontsize=10, color='darkred', pad=6)
    ax2.plot([1.0, 5.0], [2.8, 2.8], color=COLOR_NE, lw=4)
    ax2.text(3.0, 3.1, 'Conductor ne', ha='center', fontsize=9, color=COLOR_NE)
    for xs, lado in [(1.0, -1), (5.0, 1)]:
        ax2.plot([xs, xs], [2.8, 2.0], color='gray', lw=2)
        # Interruptor abierto
        ax2.plot([xs, xs + lado * 0.35], [2.0, 1.55],
                 color='gray', lw=2.2)
        ax2.plot([xs, xs], [1.3, 1.0], color=COLOR_TIERRA, lw=2)
        ax2.plot([xs - 0.3, xs + 0.3], [1.0, 1.0], color=COLOR_TIERRA, lw=2.5)
        for xx in np.arange(xs - 0.28, xs + 0.3, 0.16):
            ax2.plot([xx, xx - 0.14], [1.0, 0.75], color=COLOR_TIERRA, lw=1.1)
        ax2.text(xs, 0.55, 'abierto', ha='center', fontsize=7.5,
                 color='gray', style='italic')
    ax2.text(3.0, 1.5, r'$I_{ne} = 0$', ha='center', fontsize=13,
             color='green', fontweight='bold')
    ax2.text(3.0, 3.55, r'$V_{ne} = ?$ (inducido)',
             ha='center', fontsize=10, color=COLOR_NE, style='italic')

    fig.tight_layout()
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
    st.pyplot(fig)
    plt.close(fig)

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
    st.pyplot(fig)
    plt.close(fig)

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

    # ── Sección interactiva normalizada ────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Exploración interactiva — Efecto de la geometría sobre V_ne")
    st.markdown(r"""
    Los parámetros geométricos afectan las capacitancias y, por lo tanto, el voltaje
    inducido. Mueve los sliders para visualizar la tendencia cualitativa.
    **(Valores normalizados — no corresponden a ningún ejercicio específico.)**
    """)

    cs1, cs2 = st.columns(2)
    with cs1:
        D_val = st.slider("Distancia ne–ce (m)", 1.0, 30.0, 8.0, 0.5, key='slD')
    with cs2:
        h_val = st.slider("Altura conductor ne sobre suelo (m)", 5.0, 50.0, 15.0, 1.0, key='slH')

    D = np.linspace(1.0, 30.0, 400)
    h = np.linspace(5.0, 50.0, 400)

    # Modelos heurísticos normalizados (sin datos de ningún ejercicio)
    # C_mutua decrece con la distancia → V_ne disminuye al aumentar D
    C_mut_D = 1.0 / (1.0 + 0.15 * D)
    # C_propia crece (aprox.) con la altura → V_ne disminuye al bajar el conductor
    C_prop_h = 1.0 + 0.06 * h

    V_D = C_mut_D / C_mut_D.max()        # normalizado 0…1
    V_H = (1.0 / C_prop_h) / (1.0 / C_prop_h).max()

    idx_D = np.argmin(np.abs(D - D_val))
    idx_H = np.argmin(np.abs(h - h_val))

    fig_int = go.Figure()
    fig_int.add_trace(go.Scatter(x=D, y=V_D, mode='lines', name='V_ne vs Distancia',
                                  line=dict(color=COLOR_CE, width=2.5)))
    fig_int.add_trace(go.Scatter(x=[D_val], y=[V_D[idx_D]], mode='markers',
                                  marker=dict(color=COLOR_CE, size=13, symbol='circle'),
                                  name=f'D = {D_val:.1f} m → {V_D[idx_D]:.3f} p.u.',
                                  showlegend=True))
    fig_int.add_trace(go.Scatter(x=h, y=V_H, mode='lines', name='V_ne vs Altura',
                                  line=dict(color=COLOR_NE, width=2.5)))
    fig_int.add_trace(go.Scatter(x=[h_val], y=[V_H[idx_H]], mode='markers',
                                  marker=dict(color=COLOR_NE, size=13, symbol='diamond'),
                                  name=f'h = {h_val:.1f} m → {V_H[idx_H]:.3f} p.u.',
                                  showlegend=True))

    fig_int.update_layout(
        **PLOTLY_LAYOUT,
        title='Variación cualitativa de V_ne en función de parámetros geométricos',
        xaxis_title='Parámetro geométrico (m)',
        yaxis_title='V_ne normalizado (p.u.)',
        height=390,
        legend=dict(x=0.4, y=0.95),
    )
    st.plotly_chart(fig_int, use_container_width=True)

    m1, m2 = st.columns(2)
    with m1:
        st.metric("V_ne relativo (efecto distancia)",
                  f"{V_D[idx_D]:.3f} p.u.",
                  delta=f"D = {D_val:.1f} m  →  {'↑ cercano' if D_val < 10 else '↓ lejano'}")
    with m2:
        st.metric("V_ne relativo (efecto altura)",
                  f"{V_H[idx_H]:.3f} p.u.",
                  delta=f"h = {h_val:.1f} m  →  {'↑ bajo' if h_val < 20 else '↓ alto'}")


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
