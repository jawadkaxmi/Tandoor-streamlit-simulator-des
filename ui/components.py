# ui/components.py
import streamlit as st
from core.models import ALL_9_MODELS

def section(title: str, subtitle: str = ""):
    st.markdown(f"<div class='card'><h2>{title}</h2><p>{subtitle}</p>", unsafe_allow_html=True)
    return st.container()

def end_section():
    st.markdown("</div>", unsafe_allow_html=True)

def model_gallery(key: str, default_label: str):
    """
    Visually distinct model picker (not a dropdown). Returns selected label.
    """
    st.markdown("**Choose a queueing model (Kendall notation)**")
    labels = [m.label for m in ALL_9_MODELS]

    # Create a 3x3 gallery
    cols = st.columns(3)
    selected = st.session_state.get(key, default_label)

    for i, lab in enumerate(labels):
        with cols[i % 3]:
            active = (lab == selected)
            btn_label = f"✅ {lab}" if active else lab
            if st.button(btn_label, key=f"{key}_{lab}", use_container_width=True):
                selected = lab
                st.session_state[key] = selected

    return selected
