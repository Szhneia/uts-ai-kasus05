import base64
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from fitur import KOLOM_INPUT_MENTAH, rekayasa_fitur

PATH_MODEL = Path("models") / "model_kasus05.joblib"
PATH_ILUSTRASI = Path("assets") / "weather_icon.png"

# Palet warna 
BG_CREAM = "#FFF8E8"
NAVY = "#172755"
SLATE_BLUE = "#6682A4"
SUN_YELLOW = "#F7DC6F"
SOFT_YELLOW = "#FCE694"
WHITE = "#FFFFFF"

LABEL_CUACA_ID = {
    "sun": "Cerah (Sun)",
    "fog": "Berkabut (Fog)",
    "rain": "Hujan (Rain)",
    "drizzle": "Gerimis (Drizzle)",
    "snow": "Salju (Snow)",
}

PRESET_CUACA = {
    "Cerah Musim Panas": {
        "date": date(2015, 7, 15),
        "precipitation": 0.0,
        "temp_max": 26.5,
        "temp_min": 14.0,
        "wind": 2.8,
    },
    "Hujan Musim Gugur": {
        "date": date(2015, 11, 12),
        "precipitation": 16.4,
        "temp_max": 11.2,
        "temp_min": 6.0,
        "wind": 4.8,
    },
    "Salju Musim Dingin": {
        "date": date(2015, 1, 10),
        "precipitation": 10.5,
        "temp_max": 3.0,
        "temp_min": -1.5,
        "wind": 4.2,
    },
}

st.set_page_config(
    page_title="Klasifikasi Kondisi Cuaca Harian di Seattle",
    layout="wide",
)


def buat_ilustrasi_banner() -> str:
    """
    Mengembalikan HTML ilustrasi tanpa baris baru/indentasi agar tidak
    dianggap code block oleh Markdown Streamlit.
    """
    if PATH_ILUSTRASI.exists():
        b64 = base64.b64encode(PATH_ILUSTRASI.read_bytes()).decode("utf-8")
        return f'<img src="data:image/png;base64,{b64}" class="hero-art-img" alt="Ilustrasi Cuaca" />'

    svg = (
        '<svg class="hero-art-svg" viewBox="0 0 260 155" fill="none" xmlns="http://www.w3.org/2000/svg">'
        "<defs>"
        '<radialGradient id="sunGrad" cx="42%" cy="38%" r="60%">'
        '<stop offset="0%" stop-color="#FFF5A8"/>'
        '<stop offset="65%" stop-color="#FFD966"/>'
        '<stop offset="100%" stop-color="#F2B340"/>'
        "</radialGradient>"
        '<linearGradient id="cloudFront" x1="0" y1="0" x2="0" y2="1">'
        '<stop offset="0%" stop-color="#E7EEF6"/>'
        '<stop offset="65%" stop-color="#BAC9DC"/>'
        '<stop offset="100%" stop-color="#8FA6C4"/>'
        "</linearGradient>"
        '<linearGradient id="cloudBack" x1="0" y1="0" x2="0" y2="1">'
        '<stop offset="0%" stop-color="#D5E1EF"/>'
        '<stop offset="100%" stop-color="#96ACC8"/>'
        "</linearGradient>"
        "</defs>"
        '<ellipse cx="127" cy="18" rx="7" ry="13" transform="rotate(-6 127 18)" fill="#F3B849"/>'
        '<ellipse cx="163" cy="24" rx="7" ry="12" transform="rotate(24 163 24)" fill="#F3B849"/>'
        '<ellipse cx="94" cy="34" rx="7" ry="12" transform="rotate(-38 94 34)" fill="#F3B849"/>'
        '<ellipse cx="192" cy="49" rx="6.5" ry="11" transform="rotate(55 192 49)" fill="#F3B849"/>'
        '<ellipse cx="77" cy="65" rx="6.5" ry="10.5" transform="rotate(-70 77 65)" fill="#F3B849"/>'
        '<circle cx="136" cy="70" r="36" fill="url(#sunGrad)"/>'
        '<path d="M138 118H225C239 118 249 108 249 95C249 83 240 73 228 72C225 57 212 46 196 46C182 46 170 55 166 67C163 66 159 65 156 65C142 65 131 76 131 90C131 91 131 92 131 93C124 96 120 102 120 109C120 114 128 118 138 118Z" fill="url(#cloudBack)"/>'
        '<path d="M44 138H156C171 138 182 127 182 113C182 100 172 89 159 88C155 72 141 61 124 61C110 61 98 69 92 81C88 79 84 78 79 78C64 78 52 90 52 105C52 106 52 107 52 108C40 110 31 118 31 128C31 134 37 138 44 138Z" fill="url(#cloudFront)"/>'
        "</svg>"
    )
    return svg


st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fredoka:wght@600;700&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Plus Jakarta Sans', sans-serif;
    }}

    .stApp {{
        background-color: {BG_CREAM};
        color: {NAVY};
    }}

    header[data-testid="stHeader"] {{
        display: none !important;
    }}
    div[data-testid="stDecoration"] {{
        display: none !important;
    }}
    #MainMenu, footer {{
        visibility: hidden !important;
        height: 0 !important;
    }}

    /* Lebar proporsional terpusat */
    .block-container {{
        max-width: 1180px !important;
        margin: 0 auto !important;
        padding-top: 32px !important;
        padding-bottom: 32px !important;
        padding-left: 36px !important;
        padding-right: 36px !important;
    }}

    /* Banner Navy */
    .hero-banner {{
        position: relative;
        background-color: {NAVY};
        border-radius: 28px;
        padding: 1.55rem 2.4rem;
        margin-bottom: 18px;
        height: 152px;
        display: flex;
        align-items: center;
        justify-content: flex-start;
        overflow: visible;
        box-sizing: border-box;
    }}
    .hero-text {{
        max-width: 68%;
        z-index: 2;
    }}
    .hero-text h1 {{
        font-family: 'Fredoka', 'Plus Jakarta Sans', sans-serif;
        color: {WHITE};
        font-size: 1.85rem;
        font-weight: 700;
        letter-spacing: 0.4px;
        padding: 0 !important;
        margin: 0 0 0.3rem 0 !important;
        line-height: 1.15;
    }}
    .hero-text p {{
        color: {SOFT_YELLOW};
        font-size: 0.94rem;
        line-height: 1.42;
        padding: 0 !important;
        margin: 0 !important;
        font-weight: 500;
    }}
    .hero-art-svg {{
        position: absolute;
        right: 22px;
        top: -8px;
        width: 255px;
        height: 170px;
        pointer-events: none;
        z-index: 3;
    }}
    .hero-art-img {{
        position: absolute;
        right: 22px;
        top: -8px;
        height: 172px;
        width: auto;
        pointer-events: none;
        z-index: 3;
    }}

    /* Tabs rata tengah, garis kuning penuh, tanpa kotak fokus samping */
    div[data-testid="stTabs"] [data-baseweb="tab-list"],
    div[data-testid="stTabs"] [role="tablist"] {{
        display: flex !important;
        justify-content: center !important;
        width: 100% !important;
        gap: 2.4rem !important;
        border-bottom: 2.5px solid {SUN_YELLOW} !important;
        margin-bottom: 22px !important;
        padding-bottom: 0 !important;
    }}
    div[data-testid="stTabs"] button[data-baseweb="tab"] {{
        background: transparent !important;
        border-top: none !important;
        border-left: none !important;
        border-right: none !important;
        border-bottom: 3px solid transparent !important;
        margin-bottom: -2.5px !important;
        padding: 0.38rem 0.65rem 0.55rem 0.65rem !important;
        color: {NAVY} !important;
        font-weight: 600 !important;
        font-size: 0.96rem !important;
        outline: none !important;
        box-shadow: none !important;
    }}
    div[data-testid="stTabs"] button[data-baseweb="tab"]:focus,
    div[data-testid="stTabs"] button[data-baseweb="tab"]:focus-visible,
    div[data-testid="stTabs"] button[data-baseweb="tab"]:active {{
        outline: none !important;
        box-shadow: none !important;
        border-top: none !important;
        border-left: none !important;
        border-right: none !important;
    }}
    div[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] {{
        border-bottom: 3px solid {NAVY} !important;
    }}
    div[data-testid="stTabs"] button[data-baseweb="tab"] p {{
        color: {NAVY} !important;
        font-weight: 600 !important;
        font-size: 0.96rem !important;
    }}
    div[data-testid="stTabs"] [data-baseweb="tab-highlight"],
    div[data-testid="stTabs"] [data-baseweb="tab-border"] {{
        display: none !important;
    }}

    /* Baris dua kartu sejajar tinggi seimbang */
    div[data-testid="stHorizontalBlock"]:has(.panel-left) {{
        align-items: stretch !important;
        gap: 22px !important;
    }}

    /* Khusus tab Evaluasi: tinggi kartu mengikuti isi masing-masing */
    div[data-testid="stHorizontalBlock"]:has(.cm-wrap) {{
        align-items: flex-start !important;
    }}
    div[data-testid="stHorizontalBlock"]:has(.cm-wrap) > div[data-testid="stColumn"],
    div[data-testid="stHorizontalBlock"]:has(.cm-wrap) > div[data-testid="column"] {{
        height: fit-content !important;
        align-self: flex-start !important;
    }}

    div[data-testid="stColumn"]:has(.panel-left),
    div[data-testid="column"]:has(.panel-left) {{
        background-color: {SLATE_BLUE} !important;
        border-radius: 26px !important;
        padding: 1.75rem 2rem 2.1rem 2rem !important;
        box-sizing: border-box !important;
    }}

    div[data-testid="stColumn"]:has(.panel-right),
    div[data-testid="column"]:has(.panel-right) {{
        background-color: {NAVY} !important;
        border-radius: 26px !important;
        padding: 1.75rem 2rem 2.1rem 2rem !important;
        box-sizing: border-box !important;
    }}

    .card-title-left {{
        font-family: 'Fredoka', 'Plus Jakarta Sans', sans-serif;
        color: {WHITE};
        font-size: 1.36rem;
        font-weight: 700;
        letter-spacing: 0.35px;
        margin: 0 0 1.05rem 0;
        line-height: 1.15;
        text-transform: uppercase;
    }}
    .card-title-right {{
        font-family: 'Fredoka', 'Plus Jakarta Sans', sans-serif;
        color: {SUN_YELLOW};
        font-size: 1.36rem;
        font-weight: 700;
        letter-spacing: 0.35px;
        margin: 0 0 0.75rem 0;
        line-height: 1.15;
        text-transform: uppercase;
    }}

    /* 3 Tombol Preset Pill di kartu kanan */
    div[data-testid="stColumn"]:has(.panel-right) div.stButton > button,
    div[data-testid="column"]:has(.panel-right) div.stButton > button {{
        border-radius: 999px !important;
        background-color: rgba(247, 220, 111, 0.14) !important;
        border: 1.2px solid rgba(247, 220, 111, 0.55) !important;
        color: {BG_CREAM} !important;
        padding: 0.32rem 0.55rem !important;
        min-height: 34px !important;
        transition: all 0.18s ease !important;
    }}
    div[data-testid="stColumn"]:has(.panel-right) div.stButton > button p,
    div[data-testid="column"]:has(.panel-right) div.stButton > button p {{
        color: {BG_CREAM} !important;
        font-size: 0.77rem !important;
        font-weight: 600 !important;
        white-space: nowrap !important;
    }}
    div[data-testid="stColumn"]:has(.panel-right) div.stButton > button:hover,
    div[data-testid="column"]:has(.panel-right) div.stButton > button:hover {{
        background-color: rgba(247, 220, 111, 0.28) !important;
        border-color: {SUN_YELLOW} !important;
    }}
    div[data-testid="stColumn"]:has(.panel-right) div.stButton > button:hover p,
    div[data-testid="column"]:has(.panel-right) div.stButton > button:hover p {{
        color: {SUN_YELLOW} !important;
    }}

    /* Form & Input Pill di kartu kanan (Navy) */
    div[data-testid="stColumn"]:has(.panel-right) label,
    div[data-testid="stColumn"]:has(.panel-right) label p,
    div[data-testid="column"]:has(.panel-right) label,
    div[data-testid="column"]:has(.panel-right) label p {{
        color: {BG_CREAM} !important;
        font-size: 0.84rem !important;
        font-weight: 600 !important;
        margin-bottom: 0.22rem !important;
    }}

    div[data-testid="stColumn"]:has(.panel-right) div[data-testid="stForm"],
    div[data-testid="column"]:has(.panel-right) div[data-testid="stForm"] {{
        border: none !important;
        padding: 0 !important;
        margin: 0 !important;
        background: transparent !important;
    }}

    div[data-testid="stColumn"]:has(.panel-right) div[data-testid="stForm"] div[data-testid="stVerticalBlock"] {{
        gap: 0.72rem !important;
    }}

    div[data-testid="stColumn"]:has(.panel-right) div[data-baseweb="input"] > div,
    div[data-testid="stColumn"]:has(.panel-right) div[data-baseweb="base-input"] > div,
    div[data-testid="column"]:has(.panel-right) div[data-baseweb="input"] > div {{
        background-color: {BG_CREAM} !important;
        border: 1.5px solid transparent !important;
        border-radius: 999px !important;
        min-height: 40px !important;
        overflow: hidden !important;
        padding-left: 0.35rem !important;
    }}

    div[data-testid="stColumn"]:has(.panel-right) div[data-baseweb="input"] input,
    div[data-testid="column"]:has(.panel-right) div[data-baseweb="input"] input {{
        color: {NAVY} !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
    }}

    div[data-testid="stColumn"]:has(.panel-right) div[data-baseweb="input"] > div:focus-within {{
        border-color: {SUN_YELLOW} !important;
    }}

    div[data-testid="stColumn"]:has(.panel-right) button[data-testid="stNumberInputStepDown"],
    div[data-testid="stColumn"]:has(.panel-right) button[data-testid="stNumberInputStepUp"] {{
        background-color: #EFE5CE !important;
        color: {NAVY} !important;
        border-radius: 0 !important;
    }}

    /* Tombol Submit Pill (compact rata kiri seperti referensi) */
    div.stFormSubmitButton {{
        width: auto !important;
        margin-bottom: 0 !important;
    }}
    div.stFormSubmitButton > button {{
        background-color: {SUN_YELLOW} !important;
        color: {NAVY} !important;
        border: none !important;
        border-radius: 999px !important;
        font-weight: 700 !important;
        font-size: 0.92rem !important;
        padding: 0.56rem 1.85rem !important;
        width: auto !important;
        min-width: 215px !important;
        margin-top: 0.45rem !important;
        margin-bottom: 0 !important;
        box-shadow: 0 5px 14px rgba(0, 0, 0, 0.22) !important;
        transition: all 0.18s ease !important;
    }}
    div.stFormSubmitButton > button:hover {{
        background-color: {SOFT_YELLOW} !important;
        transform: translateY(-1px);
    }}
    div.stFormSubmitButton > button p {{
        color: {NAVY} !important;
        font-weight: 700 !important;
    }}

    /* Kartu Kiri: Hasil Prediksi & Probabilitas */
    .pred-summary {{
        background: rgba(23, 39, 85, 0.28);
        border: 1px solid rgba(255, 255, 255, 0.22);
        border-radius: 16px;
        padding: 0.95rem 1.2rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 1.15rem;
    }}
    .pred-main-label {{
        font-family: 'Fredoka', 'Plus Jakarta Sans', sans-serif;
        font-size: 1.35rem;
        font-weight: 700;
        color: {SUN_YELLOW};
        line-height: 1.15;
    }}
    .pred-sub-label {{
        font-size: 0.82rem;
        color: rgba(255, 255, 255, 0.88);
        margin-top: 0.18rem;
    }}
    .pred-conf-pill {{
        background: {SUN_YELLOW};
        color: {NAVY};
        font-family: 'Fredoka', 'Plus Jakarta Sans', sans-serif;
        font-size: 1.1rem;
        font-weight: 700;
        padding: 0.36rem 1rem;
        border-radius: 999px;
        white-space: nowrap;
    }}

    .proba-list {{
        display: flex;
        flex-direction: column;
        gap: 0.78rem;
        margin-bottom: 1.15rem;
    }}
    .proba-row {{
        display: grid;
        grid-template-columns: 132px 1fr 52px;
        align-items: center;
        gap: 0.7rem;
        font-size: 0.88rem;
        color: {WHITE};
        font-weight: 600;
    }}
    .proba-track {{
        height: 13px;
        background: rgba(23, 39, 85, 0.30);
        border-radius: 999px;
        overflow: hidden;
    }}
    .proba-fill-top {{
        height: 100%;
        background: {SUN_YELLOW};
        border-radius: 999px;
    }}
    .proba-fill {{
        height: 100%;
        background: rgba(255, 248, 232, 0.82);
        border-radius: 999px;
    }}
    .proba-pct {{
        text-align: right;
        font-variant-numeric: tabular-nums;
        font-size: 0.86rem;
        color: {BG_CREAM};
    }}

    .fe-strip {{
        background: rgba(255, 248, 232, 0.16);
        border-radius: 14px;
        padding: 0.75rem 1.05rem;
        font-size: 0.82rem;
        color: {BG_CREAM};
        display: flex;
        flex-wrap: wrap;
        justify-content: space-between;
        gap: 0.5rem;
        margin-bottom: 0;
    }}
    .fe-strip b {{
        color: {SUN_YELLOW};
    }}

    /* Tab Evaluasi */
    .kpi-grid {{
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 0.55rem;
        margin-bottom: 0.85rem;
    }}
    .kpi-box {{
        background: rgba(23, 39, 85, 0.28);
        border: 1px solid rgba(255, 255, 255, 0.18);
        border-radius: 12px;
        padding: 0.55rem 0.5rem;
        text-align: center;
    }}
    .kpi-label {{
        font-size: 0.68rem;
        color: rgba(255, 248, 232, 0.88);
        text-transform: uppercase;
        letter-spacing: 0.3px;
    }}
    .kpi-val {{
        font-family: 'Fredoka', 'Plus Jakarta Sans', sans-serif;
        font-size: 1.08rem;
        font-weight: 700;
        color: {SUN_YELLOW};
        margin-top: 0.1rem;
    }}

    .sub-heading {{
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.4px;
        color: {BG_CREAM};
        margin: 0.6rem 0 0.32rem 0;
    }}

    .compact-table {{
        width: 100%;
        border-collapse: collapse;
        font-size: 0.78rem;
        color: {WHITE};
        margin-bottom: 0.6rem;
    }}
    .compact-table:last-child {{
        margin-bottom: 0;
    }}
    .compact-table th {{
        text-align: left;
        padding: 0.34rem 0.48rem;
        background: rgba(23, 39, 85, 0.38);
        color: {SUN_YELLOW};
        font-weight: 700;
    }}
    .compact-table th:not(:first-child),
    .compact-table td:not(:first-child) {{
        text-align: right;
        font-variant-numeric: tabular-nums;
    }}
    .compact-table td {{
        padding: 0.3rem 0.48rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.14);
    }}
    .compact-table tr.row-best td {{
        background: rgba(247, 220, 111, 0.18);
        font-weight: 700;
    }}

    /* Confusion Matrix Grid: rata tengah & proporsional */
    .cm-wrap {{
        width: 100%;
        margin: 0 auto;
    }}
    .cm-caption {{
        font-size: 0.8rem;
        color: rgba(255, 248, 232, 0.86);
        margin-bottom: 0.65rem;
    }}
    .cm-table {{
        width: 100%;
        table-layout: fixed;
        border-collapse: separate;
        border-spacing: 6px;
        font-size: 0.82rem;
        text-align: center;
        margin: 0 auto;
    }}
    .cm-table th {{
        color: {SOFT_YELLOW};
        font-weight: 700;
        padding: 0.25rem;
        font-size: 0.78rem;
        text-align: center !important;
    }}
    .cm-table th:first-child,
    .cm-table td.cm-row-label {{
        width: 58px;
    }}
    .cm-table td.cm-row-label {{
        color: {BG_CREAM};
        font-weight: 700;
        text-align: left;
        padding-right: 0.4rem;
        font-size: 0.79rem;
    }}
    .cm-cell {{
        border-radius: 9px;
        padding: 0.56rem 0.25rem;
        font-weight: 700;
        font-variant-numeric: tabular-nums;
        text-align: center !important;
    }}
    .split-note {{
        margin-top: 1.05rem;
        margin-bottom: 0.25rem;
        background: rgba(255, 248, 232, 0.09);
        border: 1px solid rgba(247, 220, 111, 0.28);
        border-radius: 14px;
        padding: 0.78rem 0.95rem;
        font-size: 0.8rem;
        color: {BG_CREAM};
        line-height: 1.5;
    }}
    .split-note b {{
        color: {SUN_YELLOW};
    }}
    .split-note code {{
        background: rgba(247, 220, 111, 0.16) !important;
        color: {SUN_YELLOW} !important;
        padding: 0.1rem 0.42rem !important;
        border-radius: 6px !important;
        font-size: 0.77rem !important;
    }}

    div[data-testid="stVerticalBlock"] {{
        gap: 0.55rem !important;
    }}
    div[data-testid="stColumn"]:has(.panel-left) div[data-testid="stMarkdownContainer"] p,
    div[data-testid="stColumn"]:has(.panel-right) div[data-testid="stMarkdownContainer"] p {{
        margin-bottom: 0 !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def muat_artefak(path_model: Path):
    if not path_model.exists():
        return None
    return joblib.load(path_model)


def inisialisasi_state():
    bawaan = PRESET_CUACA["Cerah Musim Panas"]
    if "inp_date" not in st.session_state:
        st.session_state["inp_date"] = bawaan["date"]
        st.session_state["inp_precip"] = bawaan["precipitation"]
        st.session_state["inp_tmax"] = bawaan["temp_max"]
        st.session_state["inp_tmin"] = bawaan["temp_min"]
        st.session_state["inp_wind"] = bawaan["wind"]


def render_tabel_cv(hasil_cv: list[dict], nama_terbaik: str) -> str:
    rows_html = []
    for r in hasil_cv:
        cls = ' class="row-best"' if r["model"] == nama_terbaik else ""
        rows_html.append(
            f"<tr{cls}>"
            f"<td>{r['model']}</td>"
            f"<td>{r['cv_f1_macro_mean']:.4f}</td>"
            f"<td>{r['cv_balanced_acc_mean']:.4f}</td>"
            f"<td>{r['cv_accuracy_mean']:.4f}</td>"
            f"</tr>"
        )
    return (
        '<table class="compact-table">'
        "<thead><tr><th>Algoritma (5-Fold CV)</th><th>F1-Macro</th><th>Bal. Acc</th><th>Accuracy</th></tr></thead>"
        f"<tbody>{''.join(rows_html)}</tbody></table>"
    )


def render_tabel_report(rep: dict, labels: list[str]) -> str:
    rows_html = []
    for lbl in labels:
        d = rep[lbl]
        rows_html.append(
            "<tr>"
            f"<td>{lbl}</td>"
            f"<td>{d['precision']:.2f}</td>"
            f"<td>{d['recall']:.2f}</td>"
            f"<td>{d['f1-score']:.2f}</td>"
            f"<td>{int(d['support'])}</td>"
            "</tr>"
        )
    return (
        '<table class="compact-table">'
        "<thead><tr><th>Kelas (Test)</th><th>Prec</th><th>Recall</th><th>F1</th><th>Supp</th></tr></thead>"
        f"<tbody>{''.join(rows_html)}</tbody></table>"
    )


def render_confusion_matrix_html(cm: np.ndarray, labels: list[str]) -> str:
    vmax = float(cm.max()) if cm.max() > 0 else 1.0
    header_cols = "".join(f"<th>{lbl}</th>" for lbl in labels)
    body_rows = []
    for i, lbl in enumerate(labels):
        cells = [f'<td class="cm-row-label">{lbl}</td>']
        for j in range(len(labels)):
            val = int(cm[i, j])
            ratio = val / vmax
            if ratio >= 0.55:
                bg = SUN_YELLOW
                fg = NAVY
            elif ratio >= 0.18:
                bg = SLATE_BLUE
                fg = WHITE
            elif val > 0:
                bg = "rgba(102, 130, 164, 0.42)"
                fg = BG_CREAM
            else:
                bg = "rgba(255, 248, 232, 0.08)"
                fg = "rgba(255, 248, 232, 0.45)"
            cells.append(
                f'<td class="cm-cell" style="background:{bg};color:{fg};">{val}</td>'
            )
        body_rows.append(f"<tr>{''.join(cells)}</tr>")

    return (
        '<div class="cm-wrap">'
        '<div class="cm-caption">Baris = Label Aktual &bull; Kolom = Prediksi Model (Data Test)</div>'
        '<table class="cm-table">'
        f"<thead><tr><th></th>{header_cols}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        "</table></div>"
    )


def main():
    inisialisasi_state()

    ilustrasi_html = buat_ilustrasi_banner()
    banner_html = (
        '<div class="hero-banner">'
        '<div class="hero-text">'
        "<h1>KLASIFIKASI KONDISI CUACA HARIAN DI SEATTLE</h1>"
        "<p>Prediksi cuaca multikelas berbasis pipeline scikit-learn dengan ekstraksi fitur musiman dan penanganan ketidakseimbangan kelas.</p>"
        "</div>"
        f"{ilustrasi_html}"
        "</div>"
    )
    st.markdown(banner_html, unsafe_allow_html=True)

    artefak = muat_artefak(PATH_MODEL)
    if artefak is None:
        st.error(
            f"File model tidak ditemukan di `{PATH_MODEL}`. "
            "Jalankan `python latih.py` terlebih dahulu."
        )
        st.stop()

    pipeline = artefak["pipeline"]
    nama_model = artefak["nama_model_terbaik"]
    kelas_target = artefak["kelas_target"]
    hasil_cv = artefak["hasil_cv"]
    metrik_test = artefak["metrik_test"]
    info_split = artefak["info_split"]

    tab_prediksi, tab_evaluasi = st.tabs(["Prediksi Cuaca", "Evaluasi & Metadata Model"])

    #  Tab 1 Prediksi Cuaca
    with tab_prediksi:
        kol_kiri, kol_kanan = st.columns(2, gap="medium")

        with kol_kanan:
            st.markdown(
                '<div class="card-title-right panel-right">INPUT DATA CUACA MENTAH</div>',
                unsafe_allow_html=True,
            )

            kol_p = st.columns(3, gap="small")
            for idx, (nama_preset, nilai) in enumerate(PRESET_CUACA.items()):
                if kol_p[idx].button(nama_preset, use_container_width=True):
                    st.session_state["inp_date"] = nilai["date"]
                    st.session_state["inp_precip"] = nilai["precipitation"]
                    st.session_state["inp_tmax"] = nilai["temp_max"]
                    st.session_state["inp_tmin"] = nilai["temp_min"]
                    st.session_state["inp_wind"] = nilai["wind"]

            with st.form("form_prediksi_cuaca", border=False):
                r1c1, r1c2 = st.columns(2)
                with r1c1:
                    tanggal_input = st.date_input(
                        "Tanggal (date)",
                        key="inp_date",
                    )
                with r1c2:
                    precipitation = st.number_input(
                        "Curah hujan (mm)",
                        min_value=0.0,
                        max_value=200.0,
                        step=0.1,
                        key="inp_precip",
                    )

                r2c1, r2c2 = st.columns(2)
                with r2c1:
                    temp_max = st.number_input(
                        "Suhu maksimum (°C)",
                        min_value=-30.0,
                        max_value=55.0,
                        step=0.1,
                        key="inp_tmax",
                    )
                with r2c2:
                    temp_min = st.number_input(
                        "Suhu minimum (°C)",
                        min_value=-35.0,
                        max_value=50.0,
                        step=0.1,
                        key="inp_tmin",
                    )

                wind = st.number_input(
                    "Kecepatan angin (km/jam)",
                    min_value=0.0,
                    max_value=100.0,
                    step=0.1,
                    key="inp_wind",
                )

                st.form_submit_button("Prediksi Kondisi Cuaca")

        with kol_kiri:
            st.markdown(
                '<div class="card-title-left panel-left">HASIL PREDIKSI &amp; PROBABILITAS</div>',
                unsafe_allow_html=True,
            )

            if temp_max < temp_min:
                st.warning(
                    "Suhu maksimum lebih kecil dari suhu minimum. Perbaiki input di sebelah kanan."
                )
            else:
                df_input = pd.DataFrame(
                    [
                        {
                            "date": tanggal_input.strftime("%Y-%m-%d"),
                            "precipitation": float(precipitation),
                            "temp_max": float(temp_max),
                            "temp_min": float(temp_min),
                            "wind": float(wind),
                        }
                    ],
                    columns=KOLOM_INPUT_MENTAH,
                )

                prediksi = pipeline.predict(df_input)[0]
                probabilitas = pipeline.predict_proba(df_input)[0]

                df_proba = (
                    pd.DataFrame(
                        {
                            "Kelas": kelas_target,
                            "Kategori": [LABEL_CUACA_ID.get(k, k) for k in kelas_target],
                            "Probabilitas": probabilitas,
                        }
                    )
                    .sort_values("Probabilitas", ascending=False)
                    .reset_index(drop=True)
                )

                keyakinan = float(df_proba.iloc[0]["Probabilitas"]) * 100.0
                df_fe = rekayasa_fitur(df_input).iloc[0]

                bar_rows = []
                for idx, row in df_proba.iterrows():
                    pct = float(row["Probabilitas"]) * 100.0
                    fill_cls = "proba-fill-top" if idx == 0 else "proba-fill"
                    bar_rows.append(
                        f'<div class="proba-row">'
                        f'<span>{row["Kategori"]}</span>'
                        f'<div class="proba-track"><div class="{fill_cls}" style="width:{max(pct, 1.5):.1f}%;"></div></div>'
                        f'<span class="proba-pct">{pct:.1f}%</span>'
                        f"</div>"
                    )

                hasil_html = (
                    '<div class="pred-summary">'
                    "<div>"
                    f'<div class="pred-main-label">{LABEL_CUACA_ID.get(prediksi, prediksi).upper()}</div>'
                    f'<div class="pred-sub-label">Model aktif: {nama_model}</div>'
                    "</div>"
                    f'<div class="pred-conf-pill">{keyakinan:.1f}%</div>'
                    "</div>"
                    f'<div class="proba-list">{"".join(bar_rows)}</div>'
                    '<div class="fe-strip">'
                    f'<span>Bulan: <b>{int(df_fe["month"])}</b></span>'
                    f'<span>Hari ke-<b>{int(df_fe["day_of_year"])}</b></span>'
                    f'<span>Musim: <b>{str(df_fe["season"]).capitalize()}</b></span>'
                    f'<span>Rentang Suhu: <b>{df_fe["temp_range"]:.1f}°C</b></span>'
                    "</div>"
                )
                st.markdown(hasil_html, unsafe_allow_html=True)

    # Tab 2 Evaluasi & Metadata Model
    with tab_evaluasi:
        ekiri, ekanan = st.columns(2, gap="medium")

        with ekiri:
            st.markdown(
                '<div class="card-title-left panel-left">METRIK &amp; PERBANDINGAN MODEL</div>',
                unsafe_allow_html=True,
            )
            eval_kiri_html = (
                '<div class="kpi-grid">'
                f'<div class="kpi-box"><div class="kpi-label">Macro F1</div><div class="kpi-val">{metrik_test["f1_macro"]:.4f}</div></div>'
                f'<div class="kpi-box"><div class="kpi-label">Bal. Acc</div><div class="kpi-val">{metrik_test["balanced_accuracy"]:.4f}</div></div>'
                f'<div class="kpi-box"><div class="kpi-label">Weight F1</div><div class="kpi-val">{metrik_test["f1_weighted"]:.4f}</div></div>'
                f'<div class="kpi-box"><div class="kpi-label">Accuracy</div><div class="kpi-val">{metrik_test["accuracy"]:.4f}</div></div>'
                "</div>"
                '<div class="sub-heading">Seleksi Model (5-Fold Stratified CV)</div>'
                f"{render_tabel_cv(hasil_cv, nama_model)}"
                '<div class="sub-heading">Evaluasi per Kelas (Data Test)</div>'
                f"{render_tabel_report(metrik_test['classification_report_dict'], metrik_test['labels'])}"
            )
            st.markdown(eval_kiri_html, unsafe_allow_html=True)

        with ekanan:
            st.markdown(
                '<div class="card-title-right panel-right">CONFUSION MATRIX &amp; METADATA</div>',
                unsafe_allow_html=True,
            )
            cm = np.array(metrik_test["confusion_matrix"])
            eval_kanan_html = (
                f"{render_confusion_matrix_html(cm, metrik_test['labels'])}"
                '<div class="split-note">'
                f"<b>Model terpilih:</b> {nama_model}<br/>"
                f"<b>Pembagian data:</b> {info_split['jumlah_train']} train &amp; {info_split['jumlah_test']} test (Stratified Split, evaluasi test set 1 kali)<br/>"
                "<b>Penanganan imbalance:</b> Stratified Split/CV + <code>class_weight='balanced'</code> dengan metrik utama Macro F1-Score."
                "</div>"
            )
            st.markdown(eval_kanan_html, unsafe_allow_html=True)


if __name__ == "__main__":
    main()