#!/usr/bin/env python
# coding: utf-8
"""
MLB Batting Pulse — Performance Inflection Dashboard
=====================================================
Entry point. Run with:

    streamlit run dashboard/final_dashboard.py

All logic is split into focused modules:
  config.py         — constants, colours, indicator mappings
  styles.py         — global CSS injection
  data_loader.py    — data fetching & caching
  cpd_engine.py     — change-point detection algorithms
  ui_components.py  — shared UI helpers & deep-dive renderer
  pages/            — one module per dashboard page
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st

# st.set_page_config must be the very first Streamlit call
st.set_page_config(
    page_title="Team 26: Performance Inflection Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

from styles import inject_styles
from data_loader import load_data, build_perf_index
import pages.welcome          as page_welcome
import pages.snapshot         as page_snapshot
import pages.peer_comparison  as page_peer
import pages.univariate       as page_univariate
import pages.multivariate     as page_multivariate

# ── Global styles ─────────────────────────────────────────────────────────────
inject_styles()

# ── Data ──────────────────────────────────────────────────────────────────────
df       = load_data()
players  = sorted(df["player_name"].dropna().unique())
min_year = int(df["year"].min())
max_year = int(df["year"].max())
df_idx   = build_perf_index(df)

# ── Navigation ────────────────────────────────────────────────────────────────
NAV_ITEMS = [
    "Welcome",
    "Player Snapshot",
    "👥 Peer Comparison",
    "Univariate Change Analyzer",
    "Multivariate Change Analyzer",
]

if "nav_page" not in st.session_state:
    st.session_state.nav_page = "Welcome"

t_col1, t_col2 = st.columns([1, 2])
with t_col1:
    st.markdown("<div class='nav-logo'>Team 26: Performance Inflection Dashboard</div>",
                unsafe_allow_html=True)
with t_col2:
    nav_cols = st.columns(len(NAV_ITEMS))
    for i, item in enumerate(NAV_ITEMS):
        is_active = st.session_state.nav_page == item
        with nav_cols[i]:
            if st.button(item, key=f"top_nav_{item}",
                         type="primary" if is_active else "secondary"):
                st.session_state.nav_page = item
                st.rerun()

st.markdown("---")

# ── Page routing ──────────────────────────────────────────────────────────────
page = st.session_state.nav_page

if "Welcome" in page:
    page_welcome.render(df, min_year, max_year)

elif "Snapshot" in page:
    page_snapshot.render(df, players, max_year)

elif "Peer" in page:
    page_peer.render(df, players, max_year)

elif "Univariate" in page:
    page_univariate.render(df, df_idx, players, max_year)

elif "Multivariate" in page:
    page_multivariate.render(df, players, max_year)
