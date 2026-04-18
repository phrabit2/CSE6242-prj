import streamlit as st
from config import (DARK, PANEL, BORDER, GOLD, GOLD_LT, TEAL, TEAL_LT,
                    RED, RED_LT, GREY, TEXT, TEXT_MUTED)


def inject_styles():
    """Inject all global CSS styles into the Streamlit app."""
    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
html, body, [class*="css"] {{
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: {DARK}; color: {TEXT};
}}
h1,h2,h3 {{ font-family:'Bebas Neue',sans-serif; letter-spacing:0.06em; color:{TEXT}; }}
.block-container {{ padding-top:1.2rem; background-color:{DARK}; max-width:1400px; }}
[data-testid="stSidebar"] {{ background-color:{PANEL}; border-right:1px solid {BORDER}; }}
[data-testid="stSidebar"] * {{ color:{TEXT} !important; }}
.stRadio label,.stSelectbox label,.stMultiSelect label,.stSlider label {{
    color:{TEXT_MUTED} !important; font-size:0.72rem;
    text-transform:uppercase; letter-spacing:0.1em; font-family:'IBM Plex Mono',monospace;
}}
.card-row {{ display:flex; gap:10px; margin-bottom:1rem; flex-wrap:wrap; }}
.card {{ background:{DARK}; border:1px solid {BORDER}; border-top:3px solid {GOLD};
    border-radius:4px; padding:0.9rem 1.1rem; flex:1; min-width:130px; }}
.card.teal {{ border-top-color:{TEAL_LT}; }}
.card.grey {{ border-top-color:{GREY}; }}
.card-label {{ font-size:0.62rem; text-transform:uppercase; letter-spacing:0.12em;
    color:{TEXT_MUTED}; font-family:'IBM Plex Mono',monospace; margin-bottom:4px; }}
.card-val {{ font-size:1.6rem; font-weight:600; color:{TEXT};
    font-family:'Bebas Neue',sans-serif; letter-spacing:0.04em; line-height:1.1; }}
.card-sub {{ font-size:0.72rem; color:{TEXT_MUTED}; font-family:'IBM Plex Mono',monospace; margin-top:2px; }}
.sec {{ font-size:0.65rem; text-transform:uppercase; letter-spacing:0.15em;
    color:{TEAL_LT}; font-family:'IBM Plex Mono',monospace;
    border-bottom:1px solid {BORDER}; padding-bottom:4px; margin:1.1rem 0 0.7rem 0; }}
.cpd-minor {{ display:inline-block; background:#f0f2f6; color:{TEXT_MUTED}; border-radius:3px; padding:2px 8px; font-size:0.7rem; font-family:'IBM Plex Mono',monospace; margin:2px; }}
.cpd-mod   {{ display:inline-block; background:#fff5b1; color:{GOLD_LT}; border:1px solid {GOLD}; border-radius:3px; padding:2px 8px; font-size:0.7rem; font-family:'IBM Plex Mono',monospace; margin:2px; }}
.cpd-sig   {{ display:inline-block; background:#ffeef0; color:{RED_LT}; border:1px solid {RED}; border-radius:3px; padding:2px 8px; font-size:0.7rem; font-family:'IBM Plex Mono',monospace; margin:2px; }}
.preamble {{ background:{PANEL}; border:1px solid {BORDER}; border-left:4px solid {GOLD};
    border-radius:4px; padding:1.2rem 1.4rem; margin-bottom:1.4rem;
    font-size:0.92rem; line-height:1.7; color:{TEXT_MUTED}; }}
.preamble b {{ color:{TEXT}; }}
.narrative {{ background:{PANEL}; border:1px solid {BORDER}; border-radius:4px;
    padding:1rem 1.2rem; font-family:'IBM Plex Mono',monospace;
    font-size:0.82rem; line-height:1.6; color:{TEXT}; margin-bottom:1rem; }}
.rel-high {{ color:{TEAL_LT}; font-family:'IBM Plex Mono',monospace; font-size:0.75rem; }}
.rel-med  {{ color:{GOLD_LT}; font-family:'IBM Plex Mono',monospace; font-size:0.75rem; }}
.rel-low  {{ color:{RED_LT};  font-family:'IBM Plex Mono',monospace; font-size:0.75rem; }}

/* Navigation */
[data-testid="stSidebar"] {{ display: none; }}
.nav-logo {{
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.8rem;
    color: #111418;
    letter-spacing: 0.05em;
}}
div[data-testid="stHorizontalBlock"] > div:has(button) {{
    display: flex;
    justify-content: center;
}}
button[kind="secondary"] {{
    background: transparent !important;
    border: none !important;
    color: #586069 !important;
    font-weight: 500 !important;
    padding: 0.5rem 1rem !important;
    border-bottom: 2px solid transparent !important;
    border-radius: 0 !important;
    transition: all 0.2s ease !important;
}}
button[kind="secondary"]:hover {{
    color: #0969da !important;
    background: #f6f8fa !important;
}}
button[kind="primary"] {{
    background: transparent !important;
    border: none !important;
    color: #0969da !important;
    font-weight: 700 !important;
    padding: 0.5rem 1rem !important;
    border-bottom: 3px solid #58A6FF !important;
    border-radius: 0 !important;
}}
</style>
""", unsafe_allow_html=True)
